"""RL-2 Stage A (사용자 directive 2026-05-29): action-effect transition dataset builder.

사용자 directive 박제:
> "지금 부족한 건 motion sample 이 아니라 action-effect transition sample 이다. 우리가 학습
>  하려는 것은 motion distribution 이 아니라 Q_safe(s, tool, u). 필요한 데이터는
>  (state, tool, u, after_state, gate_result, utility). MotionGPT 는 state 를 만들고, tool
>  application 이 transition 을 만든다. Stage A pilot: G2 natural 300 + synthetic severe 60
>  + clean/no-harm 100 + near-boundary 50, × 3 tools × 11 u-grid, horizon 1."

핵심: 하나의 state 에서 **여러 (tool, u) 를 실제로 적용** 해 action-effect transition 을 만든다.
state coverage = decision boundary 를 덮는 4 distribution (random motion 많이 아님).

State distribution (Stage A):
  g2_natural        — G2 generated (weak correction vs STOP 경계)
  synthetic_severe  — foot_floating(0.08)+jitter(0.05) (active correction 학습)
  clean             — clean HumanML3D, corruption 없음 (STOP / no-harm 경계)
  near_boundary     — clean + moderate jitter(0.025) (gate boundary 근처, Stage A 조작적 정의)

u_grid = {0.0, 0.1, ..., 1.0} (11). u → tool intensity (action_space_provenance §5-2-2):
  FootLock / BoneProjection: continuous_factor = u  (∈ [0,1])
  VelocitySmoothing: continuous_sigma = 2.0·u       (∈ [0, 2.0])
u=0 = no-op identity (STOP anchor).

Transition schema (per (state, tool, u)):
  sample_id, state_id, distribution, step_index=0,
  state {artifact_scores 3, physical_scores 5}, tool, u, tool_params,
  after_state {...}, artifact_delta {...}, physical_delta {...},
  gate_result (pass | hard_violation), safe_utility, correction_magnitude, fidelity_loss

utility/fidelity protocol: g2_natural = Protocol B (vs state motion); clean-derived
(synthetic_severe / clean / near_boundary) = Protocol A (vs clean GT).

CLI:
    python -m tools.rl2_transition_build \
        --n-clean 100 --n-near-boundary 50 \
        --u-grid 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
        --output evals/snapshots/rl2_transition_dataset_stageA_v1.json

근거 (AGENTS.md §3-22): offline RL state-action coverage (Levine et al. 2020 survey);
continuous action OOD value overestimation → conservative/reranking (Kumar CQL NeurIPS 2020).
safe_utility = Category C internal routing reward (metric_provenance §4-1-1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS, EvaluatorReport
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from tools.synthetic_injection import inject_foot_floating, inject_jitter
from tools.safe_sequence_oracle_run import _gate_scores, _gate_violation
from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, REAL_DISTRIBUTION, common_snapshot_metadata
from tools.rl2_build_training_data import ARTIFACT_EVALUATORS, PHYSICAL_EVALUATORS, TOOL_TARGET

TOOL_BY_NAME: dict[str, CorrectionTool] = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}
TOOLS_ORDER = ("FootLockTool", "BoneProjectionTool", "VelocitySmoothingTool")
TARGET_EVALUATORS_A = ("FootFloatingEvaluator", "VelocityJitterEvaluator")
PROTOCOL_A_DISTS = {"synthetic_severe", "clean", "near_boundary"}
DEFAULT_CALIBRATION = REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json"


def _u_to_tool_params(tool: str, u: float) -> dict:
    """Normalized intensity u∈[0,1] → tool-specific metadata override."""
    if tool == "VelocitySmoothingTool":
        return {"continuous_sigma": 2.0 * u}
    return {"continuous_factor": u}  # FootLock / BoneProjection: factor = u


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _scores(motion, evaluators, names) -> dict:
    by = {ev.name: ev.evaluate(motion) for ev in evaluators}
    return {n: round(_max_score(by.get(n, [])), 6) for n in names}


def _mpjpe(a, b) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _target(scores: dict, dist: str) -> float:
    names = TARGET_EVALUATORS_A if dist in PROTOCOL_A_DISTS else ARTIFACT_EVALUATORS
    return float(np.mean([scores[n] for n in names]))


def _sweep_state(motion_state, dist, clean, evaluators, gate_evaluators, gate_thresholds,
                 u_grid, alpha, sample_id):
    """state 에서 모든 (tool, u) 의 single-step transition 측정."""
    T = motion_state.shape[0]
    art_before = _scores(motion_state, evaluators, ARTIFACT_EVALUATORS)
    phy_before = _scores(motion_state, gate_evaluators, PHYSICAL_EVALUATORS)
    gate_parent = {n: phy_before[n] for n in PHYSICAL_EVALUATORS}
    target_before = _target(art_before, dist)
    if dist in PROTOCOL_A_DISTS:
        mpjpe_before_clean = _mpjpe(motion_state, clean)
    rows = []
    for tool in TOOLS_ORDER:
        tp = TOOL_TARGET[tool]
        for u in u_grid:
            params = _u_to_tool_params(tool, u)
            try:
                after, report = TOOL_BY_NAME[tool].apply(
                    motion_state, target_part=tp, target_joints=[],
                    frame_range=(0, T - 1), strength="medium", metadata=params)
            except ValueError:
                continue
            art_after = _scores(after, evaluators, ARTIFACT_EVALUATORS)
            phy_after = _scores(after, gate_evaluators, PHYSICAL_EVALUATORS)
            target_after = _target(art_after, dist)
            artifact_reduction = target_before - target_after
            if dist in PROTOCOL_A_DISTS:
                fidelity_loss = _mpjpe(after, clean) - mpjpe_before_clean
            else:
                fidelity_loss = _mpjpe(after, motion_state)
            decisions = _gate_violation(phy_after, gate_parent, gate_thresholds)
            violation = any(d == "hard_violation" for d in decisions.values())
            safe_utility = artifact_reduction - alpha * fidelity_loss
            rows.append({
                "sample_id": sample_id, "state_id": f"{sample_id}#0", "distribution": dist,
                "step_index": 0, "tool": tool, "u": round(float(u), 3), "tool_params": params,
                "state": {"artifact_scores": art_before, "physical_scores": phy_before},
                "after_state": {"artifact_scores": art_after, "physical_scores": phy_after},
                "artifact_delta": {n: round(art_after[n] - art_before[n], 6) for n in ARTIFACT_EVALUATORS},
                "physical_delta": {n: round(phy_after[n] - phy_before[n], 6) for n in PHYSICAL_EVALUATORS},
                "gate_result": "hard_violation" if violation else "pass",
                "safe_utility": round(float(safe_utility), 6),
                "correction_magnitude": round(float(report.correction_magnitude), 6),
                "fidelity_loss": round(float(fidelity_loss), 6),
                "artifact_reduction": round(float(artifact_reduction), 6),
            })
    return rows


def _pick_clean_ids(data_dir, exclude_ids, n, seed):
    rng = np.random.default_rng(seed)
    files = sorted(data_dir.glob("*.npy"))
    stems = [f.stem for f in files if f.stem not in exclude_ids]
    rng.shuffle(stems)
    return stems[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--synthetic-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_synthetic_severe_3level_v1.json")
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--n-clean", type=int, default=100)
    parser.add_argument("--n-near-boundary", type=int, default=50)
    parser.add_argument("--near-boundary-jitter", type=float, default=0.025)
    parser.add_argument("--select-seed", type=int, default=20260529)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--u-grid", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_dataset_stageA_v1.json")
    args = parser.parse_args()

    u_grid = [float(x) for x in args.u_grid.split(",")]
    print(f"[INFO] u_grid ({len(u_grid)}): {u_grid}")
    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_thresholds = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                       if calib["summary"][n].get("n", 0) > 0}
    alpha = float(CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1["alpha"])
    evaluators = list(DEFAULT_EVALUATORS)
    gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)

    all_rows = []
    n_states = {"g2_natural": 0, "synthetic_severe": 0, "clean": 0, "near_boundary": 0}

    # === g2_natural ===
    g2_files = sorted(args.g2_batch_dir.glob("motion_*.npy"))
    print(f"[INFO] g2_natural: {len(g2_files)} states")
    for i, p in enumerate(g2_files, 1):
        m = np.load(str(p)).astype(np.float64)
        all_rows += _sweep_state(m, "g2_natural", m, evaluators, gate_evaluators,
                                 gate_thresholds, u_grid, alpha, p.stem)
        n_states["g2_natural"] += 1
        if i % 100 == 0:
            print(f"   g2 {i}/{len(g2_files)}")

    # === synthetic_severe (60 oracle ids) ===
    syn = json.load(open(args.synthetic_oracle, encoding="utf-8"))
    syn_ids = [ps["trial_id"] for ps in syn["per_sample"]]
    print(f"[INFO] synthetic_severe: {len(syn_ids)} states")
    for i, tid in enumerate(syn_ids, 1):
        npy = args.data_dir / f"{tid}.npy"
        if not npy.exists():
            continue
        clean = np.load(str(npy)).astype(np.float64)
        m1 = inject_foot_floating(clean, lift_height=0.08, seed=args.synthetic_seed)
        corrupted = inject_jitter(m1, noise_std=0.05, seed=args.synthetic_seed + 1000)
        all_rows += _sweep_state(corrupted, "synthetic_severe", clean, evaluators, gate_evaluators,
                                 gate_thresholds, u_grid, alpha, tid)
        n_states["synthetic_severe"] += 1

    # === clean / near_boundary (disjoint from synthetic ids) ===
    exclude = set(syn_ids)
    clean_ids = _pick_clean_ids(args.data_dir, exclude, args.n_clean, args.select_seed)
    exclude |= set(clean_ids)
    nb_ids = _pick_clean_ids(args.data_dir, exclude, args.n_near_boundary, args.select_seed + 1)
    print(f"[INFO] clean: {len(clean_ids)} states, near_boundary: {len(nb_ids)} states")
    for tid in clean_ids:
        npy = args.data_dir / f"{tid}.npy"
        clean = np.load(str(npy)).astype(np.float64)
        if clean.ndim != 3 or clean.shape[1] != 22:
            continue
        all_rows += _sweep_state(clean, "clean", clean, evaluators, gate_evaluators,
                                 gate_thresholds, u_grid, alpha, tid)
        n_states["clean"] += 1
    for tid in nb_ids:
        npy = args.data_dir / f"{tid}.npy"
        clean = np.load(str(npy)).astype(np.float64)
        if clean.ndim != 3 or clean.shape[1] != 22:
            continue
        nb = inject_jitter(clean, noise_std=args.near_boundary_jitter, seed=args.synthetic_seed + 2000)
        all_rows += _sweep_state(nb, "near_boundary", clean, evaluators, gate_evaluators,
                                 gate_thresholds, u_grid, alpha, tid)
        n_states["near_boundary"] += 1

    # Surface summary per (distribution, tool, u): mean utility / violation / artifact / fidelity.
    def _summary(dist):
        agg = {}
        for r in all_rows:
            if r["distribution"] != dist:
                continue
            k = f"{r['tool']}|u={r['u']}"
            a = agg.setdefault(k, {"util": [], "viol": [], "art": [], "fid": []})
            a["util"].append(r["safe_utility"]); a["viol"].append(1.0 if r["gate_result"] == "hard_violation" else 0.0)
            a["art"].append(r["artifact_reduction"]); a["fid"].append(r["fidelity_loss"])
        return {k: {"mean_utility": round(float(np.mean(v["util"])), 6),
                    "violation_rate": round(float(np.mean(v["viol"])), 4),
                    "mean_artifact_reduction": round(float(np.mean(v["art"])), 6),
                    "mean_fidelity_loss": round(float(np.mean(v["fid"])), 6),
                    "n": len(v["util"])} for k, v in sorted(agg.items())}

    out = {
        "schema_version": "1.0.0", "record_type": "rl2_transition_dataset",
        "task_id": "rl2_transition_dataset_stageA_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "rl2_transition_dataset_stageA_v1",
            oracle_type="action_effect_transition", action_grid="continuous-u-11grid",
            stage="RL-2-Q-surface-stageA",
            evidence_tier=[REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
            evaluators=evaluators, gate_evaluators=gate_evaluators,
        ),
        "formulation": "action-effect transition (state, tool, u, after_state, gate_result, utility) for Q_safe",
        "u_grid": u_grid,
        "u_to_tool_params": {"FootLockTool": "continuous_factor=u", "BoneProjectionTool": "continuous_factor=u",
                             "VelocitySmoothingTool": "continuous_sigma=2.0*u"},
        "alpha": alpha, "horizon": 1,
        "distributions": {
            "g2_natural": "G2 generated (Protocol B, real-distribution)",
            "synthetic_severe": "foot_floating(0.08)+jitter(0.05) (Protocol A, controlled diagnostic)",
            "clean": "clean HumanML3D no corruption (Protocol A, STOP/no-harm boundary)",
            "near_boundary": f"clean + jitter({args.near_boundary_jitter}) (Protocol A, gate boundary — Stage A operational def)",
        },
        "safe_utility_metric": "Category C internal routing reward (metric_provenance §4-1-1)",
        "n_states": n_states, "n_states_total": sum(n_states.values()),
        "n_transitions": len(all_rows),
        "surface_summary_g2_natural": _summary("g2_natural"),
        "surface_summary_synthetic_severe": _summary("synthetic_severe"),
        "surface_summary_clean": _summary("clean"),
        "surface_summary_near_boundary": _summary("near_boundary"),
        "rows": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== RL-2 Action-Effect Transition Dataset (Stage A) ===")
    print(f"  states: {sum(n_states.values())} {n_states}")
    print(f"  transitions: {len(all_rows)} (= states × 3 tools × {len(u_grid)} u)")
    for dist in ("clean", "near_boundary", "g2_natural", "synthetic_severe"):
        vs = out[f"surface_summary_{dist}"]
        # VelocitySmoothing curve (utility over u) — 핵심 진단.
        vel = {k: v for k, v in vs.items() if k.startswith("VelocitySmoothing")}
        peak = max(vel.items(), key=lambda kv: kv[1]["mean_utility"]) if vel else (None, {"mean_utility": 0})
        print(f"  [{dist}] VelSmooth peak util at {peak[0]} = {peak[1]['mean_utility']:+.4f}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
