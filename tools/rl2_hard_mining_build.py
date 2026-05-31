"""RL-2 Stage B-1 (사용자 directive 2026-05-31): Hard-Example Mining.

사용자 directive 박제:
> "B-0 진단으로 위험 경계면이 localization 됨: synthetic_severe × FootLock/VelocitySmoothing
>  × u ∈ [0.05, 0.35]. 목적은 단순 데이터 증가가 아니라 Q/P_safe 가 가장 많이 틀린 경계면을
>  학습시키는 것. 우선순위: high-utility but gate-fail / safe-unsafe boundary adjacent /
>  STOP-vs-weak ambiguous."

Stage B-0 (reports/2026-05-30.md §3-5) 의 root cause: synthetic high-utility-unsafe 영역에서
argmax + leaky P_safe 가 unsafe 후보를 +18.8% lift 로 끌어올림. Stage B-1 = 본 경계면 직접 mining.

설계:
  - state pool: synthetic_severe oracle 60 ids + NEW 100 sampled (seed-deterministic, disjoint).
  - tools: FootLock, VelocitySmoothing (BoneProjection 제외 — audit 0.1% issue only).
  - u-grid: dense boundary {0.05, 0.075, ..., 0.35} (13 점, u=0.2 audit peak 포함).
  - 각 transition 의 hard-case 카테고리:
      * high_util_gate_fail: safe_utility > 0 AND gate_result == hard_violation.
      * boundary_adjacent: same (state,tool) 의 인접 u 와 gate flip.
      * stop_vs_weak_ambiguous: 0 < safe_utility ≤ 0.02 (STOP=0 과 애매).
  - active learning hint: Stage A coarse {0,0.5,1.0} 학습 Q+P_safe 로 본 mined transitions
    예측 → Q-disagreement marking (Q 가 틀린 경계면 직접 확인).

출력 schema: state-deduped (states dict + transitions list, 컴팩트).

CLI:
    python -m tools.rl2_hard_mining_build \
        --u-grid 0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.225,0.25,0.275,0.30,0.325,0.35 \
        --n-new-states 100 \
        --output evals/snapshots/rl2_hard_mining_stageB1_v1.json

근거 (AGENTS.md §3-22): boundary / hard-negative mining (Shrivastava et al. CVPR 2016
"Training Region-based Object Detectors with Online Hard Example Mining"), active learning
(Settles 2010); offline RL OOD action coverage (CQL Kumar NeurIPS 2020).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from tools.synthetic_injection import inject_foot_floating, inject_jitter
from tools.safe_sequence_oracle_run import _gate_scores, _gate_violation
from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, REAL_DISTRIBUTION, common_snapshot_metadata
from tools.rl2_build_training_data import ARTIFACT_EVALUATORS, PHYSICAL_EVALUATORS, TOOL_TARGET
from tools.rl2_transition_build import _sweep_state, PROTOCOL_A_DISTS
from tools.rl2_continuous_argmax_eval import _state_feats, _cand_feats, _build_heads

TARGET_TOOLS = ("FootLockTool", "VelocitySmoothingTool")
DEFAULT_CALIBRATION = REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json"
STOP_VS_WEAK_UPPER = 0.02  # |util| <= 0.02 이면 STOP 과 애매.


def _pick_new_synthetic_ids(data_dir, exclude_ids, n, seed):
    rng = np.random.default_rng(seed)
    files = sorted(data_dir.glob("*.npy"))
    stems = [f.stem for f in files if f.stem not in exclude_ids]
    rng.shuffle(stems)
    return stems[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_synthetic_severe_3level_v1.json")
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--stage-a-dataset", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_dataset_stageA_v1.json",
                        help="Stage A coarse-u 학습 Q 로 Q-disagreement marking. 없으면 skip.")
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--select-seed", type=int, default=20260531)
    parser.add_argument("--n-new-states", type=int, default=100)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--u-grid", type=str,
                        default="0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.225,0.25,0.275,0.30,0.325,0.35")
    parser.add_argument("--stop-vs-weak-upper", type=float, default=STOP_VS_WEAK_UPPER)
    parser.add_argument("--coarse-u", type=str, default="0.0,0.5,1.0",
                        help="Stage A 의 coarse-u 학습 Q 의 re-train 용 (B-0 audit 과 일치).")
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_hard_mining_stageB1_v1.json")
    args = parser.parse_args()

    u_grid = sorted([round(float(x), 4) for x in args.u_grid.split(",")])
    coarse_u = set(round(float(x), 3) for x in args.coarse_u.split(","))
    print(f"[INFO] u boundary grid ({len(u_grid)}): {u_grid}")
    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_thresholds = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                       if calib["summary"][n].get("n", 0) > 0}
    alpha = float(CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1["alpha"])
    evaluators = list(DEFAULT_EVALUATORS)
    gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)

    # Existing synthetic_severe oracle ids + new ids (disjoint).
    syn_data = json.load(open(args.synthetic_oracle, encoding="utf-8"))
    existing_ids = [ps["trial_id"] for ps in syn_data["per_sample"]]
    new_ids = _pick_new_synthetic_ids(args.data_dir, set(existing_ids), args.n_new_states, args.select_seed)
    all_ids = list(dict.fromkeys(existing_ids + new_ids))  # preserve order, dedupe.
    print(f"[INFO] synthetic state pool: {len(existing_ids)} existing + {len(new_ids)} new = {len(all_ids)}")

    # === Sweep transitions at boundary u-grid (TARGET tools only via post-filter) ===
    states_out = {}  # state_id → {distribution, state_feats[8]}
    transitions = []
    for i, tid in enumerate(all_ids, 1):
        npy = args.data_dir / f"{tid}.npy"
        if not npy.exists():
            continue
        clean = np.load(str(npy)).astype(np.float64)
        if clean.ndim != 3 or clean.shape[1] != 22:
            continue
        m1 = inject_foot_floating(clean, lift_height=0.08, seed=args.synthetic_seed)
        corrupted = inject_jitter(m1, noise_std=0.05, seed=args.synthetic_seed + 1000)
        rows = _sweep_state(corrupted, "synthetic_severe", clean, evaluators, gate_evaluators,
                            gate_thresholds, u_grid, alpha, tid)
        rows = [r for r in rows if r["tool"] in TARGET_TOOLS]
        if not rows:
            continue
        # State feats (8) — same per sample.
        sfeat = _state_feats({"artifact_scores": rows[0]["state"]["artifact_scores"],
                              "physical_scores": rows[0]["state"]["physical_scores"]})
        states_out[tid] = {"distribution": "synthetic_severe", "state_feats": [round(x, 6) for x in sfeat]}
        # Mark boundary_adjacent within (state, tool) over u.
        rows_by_tool = defaultdict(list)
        for r in rows:
            rows_by_tool[r["tool"]].append(r)
        for tool, trows in rows_by_tool.items():
            trows.sort(key=lambda x: x["u"])
            for j, r in enumerate(trows):
                prev_diff = (j > 0 and trows[j - 1]["gate_result"] != r["gate_result"])
                next_diff = (j < len(trows) - 1 and trows[j + 1]["gate_result"] != r["gate_result"])
                boundary_adjacent = bool(prev_diff or next_diff)
                util = r["safe_utility"]
                high_util_gate_fail = bool(r["gate_result"] == "hard_violation" and util > 0.0)
                stop_vs_weak = bool(0.0 < util <= args.stop_vs_weak_upper)
                transitions.append({
                    "state_id": tid, "tool": tool, "u": r["u"],
                    "gate_result": r["gate_result"], "safe_utility": util,
                    "artifact_reduction": r["artifact_reduction"],
                    "fidelity_loss": r["fidelity_loss"],
                    "correction_magnitude": r["correction_magnitude"],
                    "is_high_util_gate_fail": high_util_gate_fail,
                    "is_boundary_adjacent": boundary_adjacent,
                    "is_stop_vs_weak": stop_vs_weak,
                    "is_hard": (high_util_gate_fail or boundary_adjacent or stop_vs_weak),
                })
        if i % 20 == 0:
            print(f"   {i}/{len(all_ids)} ({len(transitions)} transitions so far)")

    # === Active learning hint: Stage A coarse-u Q 의 disagreement marking ===
    q_disagree_summary = None
    if args.stage_a_dataset.exists():
        sa = json.load(open(args.stage_a_dataset, encoding="utf-8"))
        Xtr, ytr_u, ytr_safe = [], [], []
        for r in sa["rows"]:
            if r["u"] not in coarse_u:
                continue
            Xtr.append(_cand_feats(_state_feats(r["state"]), r["tool"], r["u"]))
            ytr_u.append(r["safe_utility"]); ytr_safe.append(1 if r["gate_result"] == "pass" else 0)
        if Xtr:
            util_head, safe_head = _build_heads()
            util_head.fit(np.array(Xtr), np.array(ytr_u))
            safe_head.fit(np.array(Xtr), np.array(ytr_safe))
            # Predict on hard-mined transitions.
            Xev = np.array([_cand_feats(states_out[t["state_id"]]["state_feats"], t["tool"], t["u"])
                            for t in transitions])
            p_safe = safe_head.predict_proba(Xev)[:, 1]
            u_pred = util_head.predict(Xev)
            for j, t in enumerate(transitions):
                actual_safe = (t["gate_result"] == "pass")
                t["q_pred_safe"] = float(p_safe[j])
                t["q_pred_util"] = float(u_pred[j])
                t["q_pred_safe_class"] = bool(p_safe[j] >= 0.5)
                t["q_disagreement_unsafe_as_safe"] = bool(p_safe[j] >= 0.5 and not actual_safe)
                t["q_disagreement_safe_as_unsafe"] = bool(p_safe[j] < 0.5 and actual_safe)
                t["q_disagreement"] = bool(t["q_disagreement_unsafe_as_safe"] or t["q_disagreement_safe_as_unsafe"])
            q_disagree_summary = {
                "n_total": len(transitions),
                "n_unsafe_as_safe": sum(1 for t in transitions if t["q_disagreement_unsafe_as_safe"]),
                "n_safe_as_unsafe": sum(1 for t in transitions if t["q_disagreement_safe_as_unsafe"]),
                "n_disagreement": sum(1 for t in transitions if t["q_disagreement"]),
                "rate_unsafe_as_safe": float(np.mean([t["q_disagreement_unsafe_as_safe"] for t in transitions])),
                "rate_safe_as_unsafe": float(np.mean([t["q_disagreement_safe_as_unsafe"] for t in transitions])),
                "rate_disagreement": float(np.mean([t["q_disagreement"] for t in transitions])),
            }
            print(f"[INFO] Q-disagreement (active learning hint) on hard set: "
                  f"unsafe-as-safe {q_disagree_summary['rate_unsafe_as_safe']*100:.1f}%, "
                  f"safe-as-unsafe {q_disagree_summary['rate_safe_as_unsafe']*100:.1f}%, "
                  f"total {q_disagree_summary['rate_disagreement']*100:.1f}%")

    # Per-category + per-tool + per-u counts.
    def _count(field):
        return sum(1 for t in transitions if t.get(field))
    cat_counts = {
        "high_util_gate_fail": _count("is_high_util_gate_fail"),
        "boundary_adjacent": _count("is_boundary_adjacent"),
        "stop_vs_weak_ambiguous": _count("is_stop_vs_weak"),
        "any_hard": _count("is_hard"),
        "total": len(transitions),
    }
    per_tool_hard = defaultdict(lambda: {"n": 0, "n_hard": 0,
                                          "n_huf": 0, "n_ba": 0, "n_sw": 0,
                                          "n_unsafe_as_safe_q": 0})
    for t in transitions:
        c = per_tool_hard[t["tool"]]
        c["n"] += 1
        c["n_hard"] += int(t.get("is_hard", False))
        c["n_huf"] += int(t.get("is_high_util_gate_fail", False))
        c["n_ba"] += int(t.get("is_boundary_adjacent", False))
        c["n_sw"] += int(t.get("is_stop_vs_weak", False))
        c["n_unsafe_as_safe_q"] += int(t.get("q_disagreement_unsafe_as_safe", False))
    per_u_hard = defaultdict(lambda: {"n": 0, "n_hard": 0, "n_unsafe_as_safe_q": 0})
    for t in transitions:
        c = per_u_hard[t["u"]]
        c["n"] += 1
        c["n_hard"] += int(t.get("is_hard", False))
        c["n_unsafe_as_safe_q"] += int(t.get("q_disagreement_unsafe_as_safe", False))

    out = {
        "schema_version": "1.0.0", "record_type": "rl2_hard_mining_transitions",
        "task_id": "rl2_hard_mining_stageB1_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "rl2_hard_mining_stageB1_v1",
            oracle_type="action_effect_transition",
            action_grid="continuous-dense-boundary-u005-035",
            stage="RL-2-stageB1-hard-mining",
            evidence_tier=[CONTROLLED_DIAGNOSTIC],  # synthetic_severe only
            evaluators=evaluators, gate_evaluators=gate_evaluators,
        ),
        "directive": "audit-driven boundary mining: synthetic_severe × {FootLock, VelocitySmoothing} × u∈[0.05, 0.35].",
        "u_grid": u_grid,
        "target_tools": list(TARGET_TOOLS),
        "alpha": alpha,
        "stop_vs_weak_upper": args.stop_vs_weak_upper,
        "state_pool": {"existing_synthetic_oracle": len(existing_ids), "new_synthetic": len(new_ids),
                       "n_states": len(states_out)},
        "category_counts": cat_counts,
        "per_tool_hard_counts": dict(per_tool_hard),
        "per_u_hard_counts": {str(u): v for u, v in sorted(per_u_hard.items())},
        "q_disagreement_summary": q_disagree_summary,
        "safe_utility_metric": "Category C internal routing reward (metric_provenance §4-1-1)",
        "states": states_out,
        "transitions": transitions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== Stage B-1 Hard-Example Mining ===")
    print(f"  state pool: {len(states_out)} (existing {len(existing_ids)} + new {len(new_ids)})")
    print(f"  transitions: {len(transitions)} (u-grid {len(u_grid)} × tools {len(TARGET_TOOLS)})")
    print(f"\n  hard-case category counts:")
    for k, v in cat_counts.items():
        if k == "total": continue
        print(f"    {k:<24} {v:5d}  ({v / cat_counts['total'] * 100:5.1f}%)")
    print(f"\n  per-tool:")
    for t, c in per_tool_hard.items():
        print(f"    {t:<24} hard={c['n_hard']}/{c['n']} ({c['n_hard']/c['n']*100:4.1f}%)  "
              f"huf={c['n_huf']}  ba={c['n_ba']}  sw={c['n_sw']}  Q-uas={c['n_unsafe_as_safe_q']}")
    print(f"\n  per-u (hard density):")
    for u in sorted(per_u_hard):
        c = per_u_hard[u]
        print(f"    u={u:<6} hard={c['n_hard']}/{c['n']} ({c['n_hard']/c['n']*100:4.1f}%)  "
              f"Q-uas={c['n_unsafe_as_safe_q']}")
    if q_disagree_summary:
        s = q_disagree_summary
        print(f"\n  Q-disagreement (Stage A coarse-u Q on hard set):")
        print(f"    unsafe-as-safe: {s['n_unsafe_as_safe']}/{s['n_total']} ({s['rate_unsafe_as_safe']*100:.1f}%)")
        print(f"    safe-as-unsafe: {s['n_safe_as_unsafe']}/{s['n_total']} ({s['rate_safe_as_unsafe']*100:.1f}%)")
        print(f"    total disagreement: {s['rate_disagreement']*100:.1f}%")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
