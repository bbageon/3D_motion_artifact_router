"""Step F-5 (사용자 directive 2026-05-28): violation decomposition (thesis 핵심 figure).

사용자 directive:
> "B2의 39% violation이 thesis의 핵심 figure가 될 가능성이 큽니다. 이 39%가 무엇인지
>  분해해야 합니다. 어떤 evaluator (BoneCV/Jerk/Penetrate/Skate/Float)? 어떤 tool/strength?
>  필수 표: Method | Violation Rate | BoneCV | Jerk | Penetrate | Skate | Float."

본 도구는 각 method 의 final motion 의 physical violation 을 **evaluator별 분해** +
B2 의 strength별 + tool별 violation 원인 분석.

Methods: no-op / B2-small/medium/large / B2-best / RL-2 RF / RL-2 HGB / safe oracle /
unsafe oracle. 각각의 per-evaluator hard_violation rate.

CLI:
    python -m tools.rl2_violation_decomposition --seeds 0,1,2 \
        --output evals/snapshots/rl2_violation_decomposition_v1.json
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
from tools.synthetic_injection import inject_foot_floating, inject_jitter
from tools.safe_sequence_oracle_run import _gate_scores, _gate_violation
from tools.rl2_build_training_data import PHYSICAL_EVALUATORS
from tools.rl2_train_imitation import _flatten_state, _sample_level_split, _build_models
from tools.rl2_closed_loop_eval import _run_policy_closed_loop, _idx_to_action
from tools.rl2_build_training_data import _artifact_scores, _physical_scores

TOOL_BY_NAME: dict[str, CorrectionTool] = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}
GATE_EVALUATORS = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)


def _violation_breakdown(final_motion, ref_motion, gate_thresholds) -> dict:
    """Per-evaluator hard_violation (final vs ref)."""
    before = _gate_scores(ref_motion, GATE_EVALUATORS)
    after = _gate_scores(final_motion, GATE_EVALUATORS)
    dec = _gate_violation(after, before, gate_thresholds)
    return {ev: (dec.get(ev) == "hard_violation") for ev in PHYSICAL_EVALUATORS}


def _apply_seq(motion, seq):
    T = motion.shape[0]
    out = motion.copy()
    for step in seq:
        if step[0] in ("STOP", "APPLY_FAIL", "SCORE_VIOLATION_ROLLBACK"):
            break
        tool = TOOL_BY_NAME[step[0]]
        out, _ = tool.apply(out, target_part=step[1], target_joints=[],
                            frame_range=(0, T - 1), strength=step[2])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_imitation_dataset_v1.json")
    parser.add_argument("--g2-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_g2_hml3d_test300_v1.json")
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--synthetic-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_synthetic_severe_v1.json")
    parser.add_argument("--synthetic-data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--calibration", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_violation_decomposition_v1.json")
    args = parser.parse_args()

    data = json.load(open(args.dataset, encoding="utf-8"))
    rows = data["rows"]
    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_thresholds = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                       if calib["summary"][n].get("n", 0) > 0}
    seeds = [int(s) for s in args.seeds.split(",")]
    evaluators = list(DEFAULT_EVALUATORS)

    g2_oracle = {p["trial_id"]: p for p in json.load(open(args.g2_oracle, encoding="utf-8"))["per_sample"]}
    syn_oracle = {p["trial_id"]: p for p in json.load(open(args.synthetic_oracle, encoding="utf-8"))["per_sample"]}

    models = _build_models()
    X_all = np.array([_flatten_state(r["state"], False) for r in rows])
    y_all = np.array([r["action_idx"] for r in rows])

    # Accumulate per-method, per-evaluator violation (over seeds × samples).
    # method -> list of breakdown dicts (+ overall any-violation).
    method_breakdowns = defaultdict(list)
    # B2 strength-specific tracking (synthetic, which strength violates).
    b2_strength_viol = defaultdict(lambda: {"count": 0, "viol": 0})

    for seed in seeds:
        train_idx, eval_idx = _sample_level_split(rows, seed)
        eval_syn = sorted({rows[i]["sample_id"] for i in eval_idx if rows[i]["distribution"] == "synthetic"})
        eval_g2 = sorted({rows[i]["sample_id"] for i in eval_idx if rows[i]["distribution"] == "g2"})
        trained = {mn: models[mn]().fit(X_all[train_idx], y_all[train_idx]) for mn in ("RF", "HGB")}

        for dist, eval_ids in (("synthetic", eval_syn), ("g2", eval_g2)):
            for tid in eval_ids:
                if dist == "synthetic":
                    npy = args.synthetic_data_dir / f"{tid}.npy"
                    if not npy.exists():
                        continue
                    clean = np.load(str(npy)).astype(np.float64)
                    m1 = inject_foot_floating(clean, lift_height=0.08, seed=args.synthetic_seed)
                    ref = inject_jitter(m1, noise_std=0.05, seed=args.synthetic_seed + 1000)
                    oracle = syn_oracle
                else:
                    npy = args.g2_batch_dir / f"{tid}.npy"
                    if not npy.exists():
                        continue
                    ref = np.load(str(npy)).astype(np.float64)
                    oracle = g2_oracle

                dtag = 1 if dist == "synthetic" else 0

                # B2-best (track per-strength) + B2-best overall.
                best_ng, best_motion, best_strength = None, ref, "noop"
                for st in ("small", "medium", "large"):
                    out, _ = TOOL_BY_NAME["VelocitySmoothingTool"].apply(
                        ref, target_part="full_body", target_joints=[], frame_range=(0, ref.shape[0]-1), strength=st)
                    bd = _violation_breakdown(out, ref, gate_thresholds)
                    any_v = any(bd.values())
                    b2_strength_viol[f"{dist}|{st}"]["count"] += 1
                    b2_strength_viol[f"{dist}|{st}"]["viol"] += int(any_v)
                    # NetGain proxy for "best" — use target reduction (artifact).
                    # Simplest: track motion that reduces artifact most (use neg total).
                    art = sum(_artifact_scores(out, evaluators))
                    ng = -art  # higher = less artifact.
                    if best_ng is None or ng > best_ng:
                        best_ng, best_motion, best_strength = ng, out, st
                method_breakdowns[f"{dist}|B2-best"].append(_violation_breakdown(best_motion, ref, gate_thresholds))

                # RL-2 RF / HGB.
                for mn in ("RF", "HGB"):
                    final, _ = _run_policy_closed_loop(ref, trained[mn], evaluators, GATE_EVALUATORS,
                                                       gate_thresholds, dist_tag=dtag, max_depth=args.max_depth)
                    method_breakdowns[f"{dist}|RL-2 {mn}"].append(_violation_breakdown(final, ref, gate_thresholds))

                # Safe oracle / unsafe oracle (from snapshot sequences).
                sb = oracle[tid].get("safe_best")
                ub = oracle[tid].get("unsafe_best")
                if sb is not None:
                    safe_motion = _apply_seq(ref, sb["sequence"]) if sb["length"] > 0 else ref
                    method_breakdowns[f"{dist}|safe oracle"].append(_violation_breakdown(safe_motion, ref, gate_thresholds))
                if ub is not None:
                    unsafe_motion = _apply_seq(ref, ub["sequence"]) if ub["length"] > 0 else ref
                    method_breakdowns[f"{dist}|unsafe oracle"].append(_violation_breakdown(unsafe_motion, ref, gate_thresholds))
        print(f"[seed {seed}] decomposition done")

    # Aggregate.
    def _agg(breakdowns):
        n = len(breakdowns)
        if n == 0:
            return {}
        per_ev = {ev: float(np.mean([b[ev] for b in breakdowns])) for ev in PHYSICAL_EVALUATORS}
        any_v = float(np.mean([any(b.values()) for b in breakdowns]))
        return {"n": n, "any_violation_rate": any_v, "per_evaluator": per_ev}

    decomposition = {m: _agg(bds) for m, bds in method_breakdowns.items()}
    b2_strength_summary = {k: {"count": v["count"], "violation_rate": v["viol"] / max(v["count"], 1)}
                           for k, v in b2_strength_viol.items()}

    out = {
        "schema_version": "1.0.0", "record_type": "rl2_violation_decomposition",
        "task_id": "rl2_violation_decomposition_v1", "seeds": seeds,
        "physical_evaluators": list(PHYSICAL_EVALUATORS),
        "decomposition": decomposition,
        "b2_strength_violation": b2_strength_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Step F-5: Violation Decomposition (per-evaluator) ===")
    for dist in ("synthetic", "g2"):
        print(f"\n[{dist}]")
        print(f"  {'method':<18} {'AnyViol':<10} {'BoneCV':<9} {'Jerk':<9} {'Penetr':<9} {'Skate':<9} {'Float'}")
        for m in ["B2-best", "RL-2 RF", "RL-2 HGB", "safe oracle", "unsafe oracle"]:
            key = f"{dist}|{m}"
            d = decomposition.get(key, {})
            if not d:
                continue
            pe = d["per_evaluator"]
            print(f"  {m:<18} {d['any_violation_rate']*100:<9.0f}% "
                  f"{pe['BoneLengthCVEvaluator']*100:<8.0f}% {pe['JerkSpikeEvaluator']*100:<8.0f}% "
                  f"{pe['PenetrateEvaluator']*100:<8.0f}% {pe['SkateEvaluator']*100:<8.0f}% "
                  f"{pe['FloatEvaluator']*100:.0f}%")
    print("\n=== B2 strength-specific violation (synthetic) ===")
    for k, v in sorted(b2_strength_summary.items()):
        if k.startswith("synthetic"):
            print(f"  {k}: {v['violation_rate']*100:.0f}% ({v['count']} samples)")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
