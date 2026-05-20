"""B6 closed-loop multi-artifact — Step 3.5.

사용자 directive: "B6 를 반복 루프 안에 넣기. evaluate → B6 action → correct →
re-evaluate → STOP/rollback. 비교 대상: B2, B5 single-step, B6 single-step,
B6 closed-loop, 가능하면 sequence oracle."

Multi-artifact (foot + jitter) closed-loop:
  1. evaluate corrupted (multi-artifact).
  2. B6 (SupervisedSelectorOrchestrator) decide → primary artifact + tool + strength.
  3. apply tool.
  4. re-evaluate.
  5. score 비감소 검증 (RefinementLoop default tolerance).
  6. STOP / rollback / continue.

H-2026-204 RQ2 (closed-loop > single-step) + H-2026-205 (learned > rule) 결합 검증.

CLI:
    python -m tools.baseline_b6_closed_loop_multi_run \\
        --training-data evals/snapshots/oracle_training_data_v2.json \\
        --n-samples 30 --seed 42 \\
        --max-iterations 4 \\
        --task-id baseline_b6_closed_loop_multi_v1 \\
        --output evals/snapshots/baseline_b6_closed_loop_multi_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from correction_tools import BoneProjectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, EvaluatorReport
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from orchestrator.supervised_selector import (
    SupervisedSelector,
    SupervisedSelectorOrchestrator,
    split_train_eval_by_trial,
    tuples_to_arrays,
)
from refinement_loop.loop import RefinementLoop
from tools.synthetic_injection import inject_foot_floating, inject_jitter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "baseline_b6_closed_loop_multi_sample"

TARGET_EVALUATORS = ("FootFloatingEvaluator", "VelocityJitterEvaluator")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _get_severity_version(evaluator: Any) -> str:
    mod = sys.modules.get(type(evaluator).__module__)
    if mod is None:
        return "unversioned"
    return getattr(mod, "SEVERITY_VERSION", "unversioned")


def _multi_inject(clean: np.ndarray, seed: int) -> np.ndarray:
    m1 = inject_foot_floating(clean, lift_height=0.08, seed=seed)
    return inject_jitter(m1, noise_std=0.05, seed=seed + 1000)


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _max_score(reports: list[EvaluatorReport]) -> float:
    if not reports:
        return 0.0
    return float(max(r.score for r in reports))


def _total_artifact_score_target(reports_by_ev: dict[str, list[EvaluatorReport]]) -> float:
    """Multi-target score: mean(target evaluator max scores)."""
    scores = [_max_score(reports_by_ev.get(name, [])) for name in TARGET_EVALUATORS]
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="B6 closed-loop multi-artifact measurement")
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--score-increase-tolerance", type=float, default=0.01)
    parser.add_argument("--stop-score-threshold", type=float, default=0.02)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--model-random-state", type=int, default=1)
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--split-id", type=str, default="multi_artifact_v1")
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
    alpha = netgain_weights["alpha"]
    beta = netgain_weights["beta"]
    gamma = netgain_weights["gamma"]

    # Train supervised selector.
    with open(args.training_data, encoding="utf-8") as f:
        td = json.load(f)
    tuples = td["tuples"]
    train_tuples, _ = split_train_eval_by_trial(tuples, train_ratio=args.train_ratio, seed=args.seed)
    X_train, tool_train, strength_train = tuples_to_arrays(train_tuples)
    selector = SupervisedSelector(model_type="random_forest", random_state=args.model_random_state)
    selector.train(X_train, tool_train, strength_train, feature_names=td.get("feature_names"))
    print(f"[INFO] supervised trained on {len(train_tuples)} tuples (artifact_kind onehot + scores).")

    # Build orchestrator wrapper + RefinementLoop.
    # target_evaluators 를 multi-artifact recipe (foot + jitter) 에 명시. BoneLength 는
    # NOT a target — 본 multi-artifact 의 recipe 는 foot_floating + global_jitter only.
    # 이 명시 없이 default ([foot, bone, jitter]) 면 jitter 가 bone length 에 spillover
    # 한 sample 에서 primary 가 BoneLength 로 잘못 선택됨 (B6 closed-loop v1 bug).
    orchestrator = SupervisedSelectorOrchestrator(
        selector=selector,
        stop_score_threshold=args.stop_score_threshold,
        target_evaluators=list(TARGET_EVALUATORS),
    )
    tool_registry = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "BoneProjectionTool": BoneProjectionTool(),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
    }
    evaluators = list(DEFAULT_EVALUATORS)
    loop = RefinementLoop(
        evaluators=evaluators,
        correction_tools=tool_registry,
        orchestrator=orchestrator,
        max_iterations=args.max_iterations,
        score_increase_tolerance=args.score_increase_tolerance,
    )

    # Choose 30 samples (same as multi_artifact_v1 split).
    rng = np.random.default_rng(args.seed)
    npy_files = sorted(args.data_dir.glob("*.npy"))
    n = min(args.n_samples, len(npy_files))
    chosen_idx = rng.choice(len(npy_files), size=n, replace=False)
    chosen = [npy_files[i] for i in chosen_idx]

    evaluator_config_hashes = {ev.name: ev.evaluator_class_hash() for ev in evaluators}
    evaluator_severity_versions = {ev.name: _get_severity_version(ev) for ev in evaluators}
    tool_class_hashes = {name: tool.tool_class_hash() for name, tool in tool_registry.items()}

    raw_dir = Path(args.raw_output_dir).resolve() if args.raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for path in chosen:
        trial_id = path.stem
        try:
            clean = np.load(str(path)).astype(np.float64)
        except Exception as e:
            print(f"[WARN] {trial_id}: {e}")
            continue
        if clean.ndim != 3 or clean.shape[1] != 22 or clean.shape[2] != 3:
            continue

        corrupted = _multi_inject(clean, args.seed)
        mpjpe_corrupted = _mpjpe(corrupted, clean)
        reports_before_dict = {ev.name: ev.evaluate(corrupted) for ev in evaluators}
        target_score_before = _total_artifact_score_target(reports_before_dict)

        # Run closed-loop.
        result = loop.run(corrupted, fps=20)
        refined = result.refined_motion

        # Compute multi-artifact NetGain.
        reports_after_dict = {ev.name: ev.evaluate(refined) for ev in evaluators}
        target_score_after = _total_artifact_score_target(reports_after_dict)
        target_delta = target_score_after - target_score_before
        mpjpe_refined = _mpjpe(refined, clean)
        fidelity_loss = mpjpe_refined - mpjpe_corrupted
        artifact_reduction = -target_delta

        # Cumulative correction magnitude.
        cumulative_mag = sum(r.correction_magnitude for r in result.tool_history)
        n_calls = len(result.tool_history)
        netgain = artifact_reduction - alpha * fidelity_loss - beta * cumulative_mag - gamma * float(n_calls)

        # decision trace summary.
        decision_summary = []
        for d in result.decision_history:
            decision_summary.append({
                "decision": d.decision,
                "primary_error": d.primary_error,
                "selected_tool": d.selected_tool,
                "strength": d.strength,
                "target_part": d.target_part,
                "metadata_primary_artifact_kind": d.metadata.get("primary_artifact_kind"),
                "metadata_primary_evaluator": d.metadata.get("primary_evaluator"),
            })

        results.append({
            "trial_id": trial_id,
            "target_score_before": target_score_before,
            "target_score_after": target_score_after,
            "target_delta": target_delta,
            "fidelity_loss_protocol_a": fidelity_loss,
            "correction_magnitude": cumulative_mag,
            "tool_call_count": n_calls,
            "netgain_provisional": netgain,
            "converged": result.converged,
            "max_iterations_reached": result.max_iterations_reached,
            "rolled_back": result.rolled_back,
            "stop_reason": result.metadata.get("stop_reason"),
            "decision_trace": decision_summary,
            "score_trace": list(result.score_trace),
            "per_evaluator_before_max": {ev.name: _max_score(reports_before_dict.get(ev.name, [])) for ev in evaluators},
            "per_evaluator_after_max": {ev.name: _max_score(reports_after_dict.get(ev.name, [])) for ev in evaluators},
        })

        if raw_dir is not None:
            timestamp = _now_iso()
            record = {
                "schema_version": SCHEMA_VERSION,
                "record_type": RECORD_TYPE,
                "timestamp": timestamp,
                "task_id": args.task_id,
                "split_id": args.split_id,
                "trial_id": trial_id,
                "baseline_type": "b6_closed_loop",
                "sample_path": str(path),
                "generator_id": "humanml3d_gt",
                "max_iterations": args.max_iterations,
                "stop_score_threshold": args.stop_score_threshold,
                "score_increase_tolerance": args.score_increase_tolerance,
                "multi_artifact_recipe": ["foot_floating(lift=0.08, seed=base)",
                                          "global_jitter(noise=0.05, seed=base+1000)"],
                "target_evaluators": list(TARGET_EVALUATORS),
                "total_artifact_score_definition": "mean(FootFloating_max, VelocityJitter_max)",
                "netgain_weight_status": "calibrated_protocol_a_v1",
                "netgain_weights": dict(netgain_weights),
                "evaluator_config_hashes": evaluator_config_hashes,
                "evaluator_severity_versions": evaluator_severity_versions,
                "tool_class_hashes": tool_class_hashes,
                "model_random_state": args.model_random_state,
                "train_ratio": args.train_ratio,
                "selections": {"foot_floating_AND_global_jitter": results[-1]},
                "negative_result": False,
            }
            out = raw_dir / f"{timestamp}_{args.task_id}_{trial_id}.json"
            out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    ng = np.array([r["netgain_provisional"] for r in results], dtype=np.float64)
    td_ = np.array([r["target_delta"] for r in results], dtype=np.float64)
    n_calls_arr = np.array([r["tool_call_count"] for r in results], dtype=np.int64)
    rollback_count = sum(1 for r in results if r["rolled_back"])
    max_iter_count = sum(1 for r in results if r["max_iterations_reached"])
    stop_threshold_count = sum(1 for r in results if r["stop_reason"] == "all_targets_below_threshold")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "baseline_b6_closed_loop_multi_summary",
        "task_id": args.task_id,
        "split_id": args.split_id,
        "baseline_type": "b6_closed_loop",
        "max_iterations": args.max_iterations,
        "stop_score_threshold": args.stop_score_threshold,
        "score_increase_tolerance": args.score_increase_tolerance,
        "n_samples": len(results),
        "multi_artifact_recipe": ["foot_floating", "global_jitter"],
        "target_evaluators": list(TARGET_EVALUATORS),
        "netgain_provisional": {
            "mean": float(ng.mean()), "median": float(np.median(ng)),
            "min": float(ng.min()), "max": float(ng.max()),
            "p25": float(np.percentile(ng, 25)), "p75": float(np.percentile(ng, 75)),
        },
        "target_delta": {"mean": float(td_.mean()), "median": float(np.median(td_))},
        "tool_call_count": {
            "mean": float(n_calls_arr.mean()), "median": float(np.median(n_calls_arr)),
            "max": int(n_calls_arr.max()), "min": int(n_calls_arr.min()),
        },
        "termination": {
            "rolled_back": rollback_count,
            "max_iterations_reached": max_iter_count,
            "stop_below_threshold": stop_threshold_count,
        },
        "trial_ids": [r["trial_id"] for r in results],
        "results": results,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OK] wrote summary to {args.output}")
        print(f"  n={len(results)}")
        print(f"  NetGain median={summary['netgain_provisional']['median']:+.5f}, "
              f"mean={summary['netgain_provisional']['mean']:+.5f}")
        print(f"  tool_call_count median={summary['tool_call_count']['median']:.1f}, "
              f"max={summary['tool_call_count']['max']}")
        print(f"  termination: {summary['termination']}")


if __name__ == "__main__":
    main()
