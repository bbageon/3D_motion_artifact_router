"""Step D (사용자 directive 2026-05-26): 기존 correction 결과 의 PhysicalGate 재평가.

4 G2 top correction sample (motion_006/007/008/028) 의 5 method:
  - Original (no correction, reference)
  - B2-small / B2-medium / B2-large (fixed smoothing baseline family)
  - 5-level oracle (best correction sequence per oracle)

각 method 에 대해 3-column 측정:
  1. NetGain change vs Original (internal routing reward)
  2. PhysicalGate before vs after — 5 evaluator scores (Penetrate/Float/Skate/JerkSpike/BoneLengthCV)
  3. Gate decision (pass / soft_violation / hard_violation) — clean p99 calibration 기준

본 도구는 NetGain-only 결과 의 hidden physical violation 의 정량 evidence — Safe
Orchestration framing 의 직접 motivation.

CLI:
    python -m tools.physical_gate_existing_correction_reeval

Calibration: evals/snapshots/physical_gate_clean_calibration_v1.json 의 p95 / p99.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import (
    BoneLengthCVEvaluator, FloatEvaluator, JerkSpikeEvaluator,
    PenetrateEvaluator, SkateEvaluator,
)

TOOL_BY_NAME: dict[str, CorrectionTool] = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}
TOP4_SAMPLES = ["motion_006", "motion_007", "motion_008", "motion_028"]

EVALUATORS = [
    PenetrateEvaluator(),
    FloatEvaluator(),
    SkateEvaluator(),
    JerkSpikeEvaluator(),
    BoneLengthCVEvaluator(),
]
EVAL_NAMES = [e.name for e in EVALUATORS]


def _apply_sequence(motion: np.ndarray, sequence: list) -> np.ndarray:
    T = motion.shape[0]
    out = motion.copy()
    for step in sequence:
        if step[0] in ("STOP", "APPLY_FAIL", "SCORE_VIOLATION_ROLLBACK"):
            break
        tool = TOOL_BY_NAME[step[0]]
        out, _ = tool.apply(out, target_part=step[1], target_joints=[],
                             frame_range=(0, T - 1), strength=step[2])
    return out


def _apply_b2(motion: np.ndarray, strength: str) -> np.ndarray:
    T = motion.shape[0]
    out, _ = TOOL_BY_NAME["VelocitySmoothingTool"].apply(
        motion, target_part="full_body", target_joints=[],
        frame_range=(0, T - 1), strength=strength,
    )
    return out


def _compute_gate_scores(motion: np.ndarray) -> dict[str, float]:
    """Compute 5 evaluator's max score for the motion."""
    result = {}
    for ev in EVALUATORS:
        try:
            reports = ev.evaluate(motion)
        except Exception:
            result[ev.name] = float("nan")
            continue
        if not reports:
            result[ev.name] = 0.0
        else:
            result[ev.name] = float(max(r.score for r in reports))
    return result


def _gate_decision(
    before: dict[str, float], after: dict[str, float], thresholds: dict[str, float],
    eps: float = 1e-5, jerk_relax: float = 1.05,
) -> dict[str, str]:
    """Per-evaluator decision: 'pass' / 'soft_violation' / 'hard_violation'.

    Rule per user directive 2026-05-26:
      if Penetrate_after > max(Penetrate_before + eps, CleanP99):   hard
      if BoneLengthCV_after > max(BoneLengthCV_before + eps, CleanP99): hard
      if Jerk_after > max(Jerk_before * jerk_relax, CleanP99):  hard
      etc.

    Returns {evaluator_name: decision}.
    """
    decisions = {}
    for name, after_score in after.items():
        before_score = before.get(name, 0.0)
        clean_p99 = thresholds.get(name, 0.0)
        # Use multiplicative relaxation for JerkSpike (per user directive).
        relax_thresh = (before_score * jerk_relax if name == "JerkSpikeEvaluator"
                        else before_score + eps)
        unsafe_thresh = max(relax_thresh, clean_p99)
        if after_score > unsafe_thresh:
            # Severity: hard if also exceeds clean p99, soft if just regression vs before.
            decisions[name] = "hard_violation" if after_score > clean_p99 + eps else "soft_violation"
        else:
            decisions[name] = "pass"
    return decisions


def main() -> None:
    snap_path = REPO_ROOT / "evals" / "snapshots" / "oracle_sequence_g2_natural_5level_v1.json"
    calib_path = REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json"
    data_dir = REPO_ROOT / "external_assets" / "g2_generated_v1"
    output_path = REPO_ROOT / "evals" / "snapshots" / "physical_gate_existing_correction_reeval_v1.json"

    with open(snap_path, encoding="utf-8") as f:
        snap = json.load(f)
    with open(calib_path, encoding="utf-8") as f:
        calib = json.load(f)
    snap_by = {s["trial_id"]: s for s in snap["per_sample"]}

    # Extract p99 thresholds (calibration 기준).
    thresholds_p99 = {n: calib["summary"][n]["p99"] for n in EVAL_NAMES if n in calib["summary"]}
    thresholds_p95 = {n: calib["summary"][n]["p95"] for n in EVAL_NAMES if n in calib["summary"]}
    print(f"[INFO] thresholds_p99: {thresholds_p99}")
    print(f"[INFO] thresholds_p95: {thresholds_p95}")
    print()

    results = []
    for tid in TOP4_SAMPLES:
        npy = data_dir / f"{tid}.npy"
        motion = np.load(str(npy)).astype(np.float64)
        s = snap_by[tid]
        oracle_seq = s["best"]["sequence"]
        oracle_ng = s["best"]["netgain"]
        oracle_full_seq = " → ".join(f"{a[0].replace('Tool','')}/{a[2]}" for a in oracle_seq)

        # Build method-applied motions.
        methods = {
            "Original": motion,
            "B2-small": _apply_b2(motion, "small"),
            "B2-medium": _apply_b2(motion, "medium"),
            "B2-large": _apply_b2(motion, "large"),
            "5-level_oracle": _apply_sequence(motion, oracle_seq),
        }

        sample_result = {
            "trial_id": tid,
            "prompt": s.get("g2_prompt", "")[:120],
            "oracle_netgain": oracle_ng,
            "oracle_sequence": oracle_full_seq,
            "T": motion.shape[0],
            "methods": {},
        }

        # Compute gate scores for each method.
        original_scores = _compute_gate_scores(methods["Original"])
        print(f"[{tid}] T={motion.shape[0]}, oracle_ng={oracle_ng:+.4f}, sequence={oracle_full_seq}")
        print(f"  Original gate scores: {{")
        for n in EVAL_NAMES:
            print(f"    {n:30s}: {original_scores[n]:.5f}")
        print("  }")

        for method, m in methods.items():
            after_scores = _compute_gate_scores(m)
            decisions = _gate_decision(original_scores, after_scores, thresholds_p99)
            # Hard violations = methods with at least one hard_violation in decisions.
            n_hard = sum(1 for d in decisions.values() if d == "hard_violation")
            n_soft = sum(1 for d in decisions.values() if d == "soft_violation")
            n_pass = sum(1 for d in decisions.values() if d == "pass")

            sample_result["methods"][method] = {
                "gate_scores": after_scores,
                "gate_decisions_p99": decisions,
                "n_hard_violations": n_hard,
                "n_soft_violations": n_soft,
                "n_pass": n_pass,
                "any_hard_violation": n_hard > 0,
                "any_violation": n_hard + n_soft > 0,
            }
            print(f"  {method}: pass={n_pass}, soft={n_soft}, hard={n_hard}")

        results.append(sample_result)
        print()

    # Aggregate summary.
    method_names = ["Original", "B2-small", "B2-medium", "B2-large", "5-level_oracle"]
    summary = {
        "n_samples": len(results),
        "threshold_basis": "p99 of HumanML3D clean (n=493 calibration)",
        "thresholds_p99": thresholds_p99,
        "thresholds_p95": thresholds_p95,
        "method_aggregate": {},
    }
    for method in method_names:
        n_with_any_hard = sum(1 for r in results if r["methods"][method]["any_hard_violation"])
        n_with_any_viol = sum(1 for r in results if r["methods"][method]["any_violation"])
        n_total_hard = sum(r["methods"][method]["n_hard_violations"] for r in results)
        n_total_soft = sum(r["methods"][method]["n_soft_violations"] for r in results)
        summary["method_aggregate"][method] = {
            "n_samples_with_any_hard_violation": n_with_any_hard,
            "n_samples_with_any_violation": n_with_any_viol,
            "total_hard_violations": n_total_hard,
            "total_soft_violations": n_total_soft,
        }
        print(f"[summary] {method}: {n_with_any_hard}/{len(results)} samples w/ hard, "
              f"{n_total_hard} total hard, {n_total_soft} total soft")

    out = {
        "schema_version": "1.0.0",
        "record_type": "physical_gate_existing_correction_reeval",
        "task_id": "physical_gate_existing_correction_reeval_v1",
        "calibration_source": str(calib_path),
        "per_sample": results,
        "summary": summary,
    }
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {output_path}")


if __name__ == "__main__":
    main()
