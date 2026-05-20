"""Multi-artifact synthetic case measurement — H-2026-205 Stage 0 의 follow-up.

같은 sample 에 두 artifact (foot_floating + global_jitter) 동시 inject → multi-
artifact motion. B2 / B5 / B6 single-step decision 의 multi-artifact handling 비교.

NetGain definition (multi-artifact):
  TotalArtifactScore = mean(FootFloating max score, VelocityJitter max score).
  NetGain = -ΔTotalArtifactScore - α·FidelityLoss - β·CorrectionMagnitude - γ·ToolCallCost.

사용자 framing 의 "큰 이득은 multi-artifact case 또는 G2 natural artifact 에서"
의 multi-artifact 측면 직접 검증.

CLI:
    python -m tools.baseline_multi_artifact_run \\
        --baseline-type b5_rule_based \\
        --n-samples 30 --seed 42 \\
        --task-id baseline_b5_multi_v1 \\
        --output evals/snapshots/baseline_b5_multi_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from correction_tools import BoneProjectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, EvaluatorReport
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from orchestrator.rule_based import RuleBasedOrchestrator
from orchestrator.supervised_selector import (
    ARTIFACT_TO_TARGET_PART,
    ARTIFACT_TO_TARGET_TOOL,
    SupervisedSelector,
    split_train_eval_by_trial,
    tuples_to_arrays,
)
from tools.synthetic_injection import inject_foot_floating, inject_jitter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "baseline_multi_artifact_sample"

TOOL_BY_NAME = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}

# Multi-artifact: foot + jitter chained.
TARGET_EVALUATORS = ("FootFloatingEvaluator", "VelocityJitterEvaluator")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _get_severity_version(evaluator: Any) -> str:
    mod = sys.modules.get(type(evaluator).__module__)
    if mod is None:
        return "unversioned"
    return getattr(mod, "SEVERITY_VERSION", "unversioned")


def _multi_inject(clean: np.ndarray, seed: int) -> np.ndarray:
    """Multi-artifact: foot_floating + global_jitter chained."""
    m1 = inject_foot_floating(clean, lift_height=0.08, seed=seed)
    m2 = inject_jitter(m1, noise_std=0.05, seed=seed + 1000)
    return m2


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _max_score(reports: list[EvaluatorReport]) -> float:
    if not reports:
        return 0.0
    return float(max(r.score for r in reports))


def _total_artifact_score(reports_dict: dict[str, list[EvaluatorReport]]) -> float:
    """Multi-target 의 total = mean(target evaluator max scores)."""
    scores = [_max_score(reports_dict.get(name, [])) for name in TARGET_EVALUATORS]
    return float(np.mean(scores))


def _apply_baseline(
    baseline_type: str,
    motion: np.ndarray,
    reports_before_dict: dict[str, list[EvaluatorReport]],
    supervised_selector: Optional[SupervisedSelector] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """baseline 별로 single-step apply."""
    T = motion.shape[0]
    frame_range = (0, T - 1)
    all_reports = []
    for rs in reports_before_dict.values():
        all_reports.extend(rs)

    trace: dict[str, Any] = {"baseline_type": baseline_type}
    if baseline_type == "b2_fixed_smoothing":
        tool = TOOL_BY_NAME["VelocitySmoothingTool"]
        try:
            corrected, report = tool.apply(motion, target_part="full_body", target_joints=[],
                                           frame_range=frame_range, strength="medium")
            trace.update({"tool": "VelocitySmoothingTool", "strength": "medium",
                          "target_part": "full_body",
                          "correction_magnitude": float(report.correction_magnitude)})
            return corrected, trace
        except ValueError as e:
            trace.update({"applied": False, "skip_reason": str(e)})
            return motion.copy(), trace

    if baseline_type == "b5_rule_based":
        orch = RuleBasedOrchestrator(tool_registry=list(TOOL_BY_NAME.values()))
        # Multi-artifact: hint 가 ambiguous (foot + jitter). hint 없이 default
        # severity 기반 primary 선택.
        decision = orch.decide(all_reports, tool_history=[], artifact_kind_hint=None)
        trace.update({
            "decision": decision.decision,
            "primary_error": decision.primary_error,
            "tool": decision.selected_tool,
            "strength": decision.strength,
            "target_part": decision.target_part,
        })
        if decision.decision == "revise" and decision.selected_tool in TOOL_BY_NAME:
            tool = TOOL_BY_NAME[decision.selected_tool]
            try:
                corrected, report = tool.apply(motion, target_part=decision.target_part or "full_body",
                                               target_joints=[], frame_range=frame_range,
                                               strength=decision.strength or "medium")
                trace["correction_magnitude"] = float(report.correction_magnitude)
                return corrected, trace
            except ValueError as e:
                trace["skip_reason"] = str(e)
                return motion.copy(), trace
        return motion.copy(), trace

    if baseline_type == "b6_supervised":
        # Multi-artifact: state vector 는 모든 evaluator scores. artifact_kind 는
        # ambiguous — primary artifact 를 score 기반으로 결정.
        eval_scores = [
            _max_score(reports_before_dict.get("FootFloatingEvaluator", [])),
            _max_score(reports_before_dict.get("BoneLengthEvaluator", [])),
            _max_score(reports_before_dict.get("VelocityJitterEvaluator", [])),
        ]
        # primary artifact = highest score among target evaluators.
        target_scores = {
            "foot_floating": eval_scores[0],
            "global_jitter": eval_scores[2],
        }
        primary_artifact = max(target_scores, key=target_scores.get)
        artifact_kinds_list = ["foot_floating", "bone_stretch_right_arm", "global_jitter"]
        artifact_onehot = [1 if k == primary_artifact else 0 for k in artifact_kinds_list]
        state_vector = np.array(artifact_onehot + eval_scores, dtype=np.float64)
        pred = supervised_selector.predict(state_vector, primary_artifact)
        trace.update({
            "primary_artifact": primary_artifact,
            "tool": pred.tool_name,
            "strength": pred.strength,
            "target_part": pred.target_part,
        })
        tool = TOOL_BY_NAME.get(pred.tool_name)
        if tool is None:
            return motion.copy(), trace
        try:
            corrected, report = tool.apply(motion, target_part=pred.target_part,
                                           target_joints=[], frame_range=frame_range,
                                           strength=pred.strength)
            trace["correction_magnitude"] = float(report.correction_magnitude)
            return corrected, trace
        except ValueError as e:
            trace["skip_reason"] = str(e)
            return motion.copy(), trace

    if baseline_type == "b1_no_refinement":
        trace.update({"applied": False})
        return motion.copy(), trace

    raise ValueError(f"unknown baseline_type {baseline_type!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-artifact synthetic measurement (B1/B2/B5/B6)")
    parser.add_argument("--baseline-type", type=str, required=True,
                        choices=["b1_no_refinement", "b2_fixed_smoothing", "b5_rule_based", "b6_supervised"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--split-id", type=str, default="multi_artifact_v1")
    parser.add_argument("--training-data", type=Path, default=None,
                        help="b6_supervised 시 필요한 oracle_training_data 경로.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--model-random-state", type=int, default=42)
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
    alpha = netgain_weights["alpha"]
    beta = netgain_weights["beta"]
    gamma = netgain_weights["gamma"]

    # Setup supervised selector if needed.
    supervised_selector: Optional[SupervisedSelector] = None
    if args.baseline_type == "b6_supervised":
        if args.training_data is None:
            raise ValueError("--training-data required for b6_supervised")
        with open(args.training_data, encoding="utf-8") as f:
            td = json.load(f)
        tuples = td["tuples"]
        train_tuples, _ = split_train_eval_by_trial(tuples, train_ratio=args.train_ratio, seed=args.seed)
        X_train, tool_train, strength_train = tuples_to_arrays(train_tuples)
        supervised_selector = SupervisedSelector(model_type="random_forest",
                                                 random_state=args.model_random_state)
        supervised_selector.train(X_train, tool_train, strength_train,
                                  feature_names=td.get("feature_names"))
        print(f"[INFO] supervised trained on {len(train_tuples)} tuples")

    rng = np.random.default_rng(args.seed)
    npy_files = sorted(args.data_dir.glob("*.npy"))
    n = min(args.n_samples, len(npy_files))
    chosen_idx = rng.choice(len(npy_files), size=n, replace=False)
    chosen = [npy_files[i] for i in chosen_idx]

    evaluators = list(DEFAULT_EVALUATORS)
    evaluator_config_hashes = {ev.name: ev.evaluator_class_hash() for ev in evaluators}
    evaluator_severity_versions = {ev.name: _get_severity_version(ev) for ev in evaluators}
    tool_class_hashes = {name: tool.tool_class_hash() for name, tool in TOOL_BY_NAME.items()}

    raw_dir = Path(args.raw_output_dir).resolve() if args.raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for path in chosen:
        trial_id = path.stem
        try:
            clean = np.load(str(path)).astype(np.float64)
            if clean.ndim != 3 or clean.shape[1] != 22 or clean.shape[2] != 3:
                continue
        except Exception as e:
            print(f"[WARN] skip {trial_id}: {e}")
            continue

        corrupted = _multi_inject(clean, args.seed)
        mpjpe_corrupted = _mpjpe(corrupted, clean)
        reports_before_dict = {ev.name: ev.evaluate(corrupted) for ev in evaluators}
        target_score_before = _total_artifact_score(reports_before_dict)

        corrected, trace = _apply_baseline(args.baseline_type, corrupted, reports_before_dict,
                                           supervised_selector=supervised_selector)
        reports_after_dict = {ev.name: ev.evaluate(corrected) for ev in evaluators}
        target_score_after = _total_artifact_score(reports_after_dict)
        target_delta = target_score_after - target_score_before
        mpjpe_corrected = _mpjpe(corrected, clean)
        fidelity_loss = mpjpe_corrected - mpjpe_corrupted
        artifact_reduction = -target_delta
        correction_mag = float(trace.get("correction_magnitude", 0.0))
        n_calls = 1 if trace.get("tool") else 0
        netgain = artifact_reduction - alpha * fidelity_loss - beta * correction_mag - gamma * float(n_calls)

        result = {
            "trial_id": trial_id,
            "target_score_before": target_score_before,
            "target_score_after": target_score_after,
            "target_delta": target_delta,
            "fidelity_loss_protocol_a": fidelity_loss,
            "correction_magnitude": correction_mag,
            "netgain_provisional": netgain,
            "trace": trace,
            "per_evaluator_before_max": {n: _max_score(reports_before_dict.get(n, [])) for n in evaluators[0:3].__class__.__dict__.get('_ignore', [None]) or [e.name for e in evaluators]},
        }
        # Simplified per_evaluator_before_max
        result["per_evaluator_before_max"] = {ev.name: _max_score(reports_before_dict.get(ev.name, [])) for ev in evaluators}
        result["per_evaluator_after_max"] = {ev.name: _max_score(reports_after_dict.get(ev.name, [])) for ev in evaluators}
        results.append(result)
        if raw_dir is not None:
            timestamp = _now_iso()
            record = {
                "schema_version": SCHEMA_VERSION,
                "record_type": RECORD_TYPE,
                "timestamp": timestamp,
                "task_id": args.task_id,
                "split_id": args.split_id,
                "trial_id": trial_id,
                "baseline_type": args.baseline_type,
                "sample_path": str(path),
                "generator_id": "humanml3d_gt",
                "multi_artifact_recipe": ["foot_floating(lift=0.08, seed=base)",
                                          "global_jitter(noise=0.05, seed=base+1000)"],
                "target_evaluators": list(TARGET_EVALUATORS),
                "total_artifact_score_definition": "mean(FootFloating_max, VelocityJitter_max)",
                "netgain_weight_status": "calibrated_protocol_a_v1",
                "netgain_weights": dict(netgain_weights),
                "evaluator_config_hashes": evaluator_config_hashes,
                "evaluator_severity_versions": evaluator_severity_versions,
                "tool_class_hashes": tool_class_hashes,
                "selections": {"foot_floating_AND_global_jitter": result},
                "negative_result": False,
            }
            out = raw_dir / f"{timestamp}_{args.task_id}_{trial_id}.json"
            out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    ng = np.array([r["netgain_provisional"] for r in results], dtype=np.float64)
    td_ = np.array([r["target_delta"] for r in results], dtype=np.float64)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "baseline_multi_artifact_summary",
        "task_id": args.task_id,
        "split_id": args.split_id,
        "baseline_type": args.baseline_type,
        "n_samples": len(results),
        "multi_artifact_recipe": ["foot_floating", "global_jitter"],
        "target_evaluators": list(TARGET_EVALUATORS),
        "netgain_provisional": {
            "mean": float(ng.mean()), "median": float(np.median(ng)),
            "min": float(ng.min()), "max": float(ng.max()),
            "p25": float(np.percentile(ng, 25)), "p75": float(np.percentile(ng, 75)),
        },
        "target_delta": {"mean": float(td_.mean()), "median": float(np.median(td_))},
        "trial_ids": [r["trial_id"] for r in results],
        "results": results,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OK] wrote summary to {args.output}")
        print(f"  baseline_type: {args.baseline_type}")
        print(f"  n={len(results)}")
        print(f"  NetGain median={summary['netgain_provisional']['median']:+.5f}, "
              f"mean={summary['netgain_provisional']['mean']:+.5f}")
        print(f"  target_delta median={summary['target_delta']['median']:+.5f}")


if __name__ == "__main__":
    main()
