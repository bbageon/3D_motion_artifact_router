"""B7 contextual bandit (closed-loop, multi-artifact) — H-2026-205 Step 4.

학습:
  - bandit training data (build_bandit_training_data.py 산출).
  - train/eval split (sample 단위 disjoint).
  - Q-regression (state, action) → reward.

Eval:
  - eval trial 의 multi-artifact corrupted motion.
  - RefinementLoop(orchestrator=ContextualBanditOrchestrator).
  - max_iterations=4, score_increase_tolerance=0.01.

비교: B6 closed-loop, B6 single, B2, B5.

CLI:
    python -m tools.baseline_b7_bandit_run \\
        --training-data evals/snapshots/bandit_training_data_v1.json \\
        --n-samples 60 --seed 42 \\
        --task-id baseline_b7_bandit_multi_v1 \\
        --output evals/snapshots/baseline_b7_bandit_multi_v1.json
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
from orchestrator.contextual_bandit import (
    ContextualBandit,
    ContextualBanditOrchestrator,
    split_train_eval_by_trial,
    tuples_to_arrays,
)
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from refinement_loop.loop import RefinementLoop
from tools.synthetic_injection import inject_foot_floating, inject_jitter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "baseline_b7_bandit_multi_sample"
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


def _target_score(reports_dict: dict[str, list[EvaluatorReport]]) -> float:
    return float(np.mean([_max_score(reports_dict.get(name, [])) for name in TARGET_EVALUATORS]))


def main() -> None:
    parser = argparse.ArgumentParser(description="B7 contextual bandit closed-loop measurement")
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--score-increase-tolerance", type=float, default=0.01)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--model-type", type=str, default="random_forest",
                        choices=["random_forest", "linear"])
    parser.add_argument("--model-random-state", type=int, default=1)
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--split-id", type=str, default="bandit_eval_v1")
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
    alpha = netgain_weights["alpha"]
    beta = netgain_weights["beta"]
    gamma = netgain_weights["gamma"]

    # Load training data.
    with open(args.training_data, encoding="utf-8") as f:
        td = json.load(f)
    tuples = td["tuples"]

    # Split.
    train_tuples, eval_tuples = split_train_eval_by_trial(
        tuples, train_ratio=args.train_ratio, seed=args.seed,
    )
    print(f"[INFO] train: {len(train_tuples)} tuples ({len({t['trial_id'] for t in train_tuples})} trials), "
          f"eval: {len(eval_tuples)} tuples ({len({t['trial_id'] for t in eval_tuples})} trials)")

    states_tr, actions_tr, rewards_tr = tuples_to_arrays(train_tuples)
    bandit = ContextualBandit(model_type=args.model_type, random_state=args.model_random_state)
    train_metrics = bandit.train(states_tr, actions_tr, rewards_tr)
    print(f"[INFO] train metrics: {train_metrics}")

    # Eval: closed-loop on each eval trial.
    eval_trial_ids = sorted({t["trial_id"] for t in eval_tuples})

    orchestrator = ContextualBanditOrchestrator(bandit=bandit)
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

    raw_dir = Path(args.raw_output_dir).resolve() if args.raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    evaluator_config_hashes = {ev.name: ev.evaluator_class_hash() for ev in evaluators}
    evaluator_severity_versions = {ev.name: _get_severity_version(ev) for ev in evaluators}
    tool_class_hashes = {name: tool.tool_class_hash() for name, tool in tool_registry.items()}

    results: list[dict[str, Any]] = []
    for trial_id in eval_trial_ids:
        sample_path = args.data_dir / f"{trial_id}.npy"
        if not sample_path.exists():
            continue
        clean = np.load(str(sample_path)).astype(np.float64)
        if clean.ndim != 3 or clean.shape[1] != 22 or clean.shape[2] != 3:
            continue
        corrupted = _multi_inject(clean, args.seed)
        mpjpe_corrupted = _mpjpe(corrupted, clean)
        reports_before = {ev.name: ev.evaluate(corrupted) for ev in evaluators}
        target_before = _target_score(reports_before)

        result = loop.run(corrupted, fps=20)
        refined = result.refined_motion
        reports_after = {ev.name: ev.evaluate(refined) for ev in evaluators}
        target_after = _target_score(reports_after)
        target_delta = target_after - target_before
        mpjpe_refined = _mpjpe(refined, clean)
        fidelity_loss = mpjpe_refined - mpjpe_corrupted
        artifact_reduction = -target_delta
        cumulative_mag = sum(r.correction_magnitude for r in result.tool_history)
        n_calls = len(result.tool_history)
        netgain = artifact_reduction - alpha * fidelity_loss - beta * cumulative_mag - gamma * float(n_calls)

        decision_summary = []
        for d in result.decision_history:
            decision_summary.append({
                "decision": d.decision,
                "selected_tool": d.selected_tool,
                "strength": d.strength,
                "action_id": d.metadata.get("action_id"),
                "q_values": d.metadata.get("q_values"),
            })

        results.append({
            "trial_id": trial_id,
            "target_score_before": target_before,
            "target_score_after": target_after,
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
            "per_evaluator_before_max": {ev.name: _max_score(reports_before.get(ev.name, [])) for ev in evaluators},
            "per_evaluator_after_max": {ev.name: _max_score(reports_after.get(ev.name, [])) for ev in evaluators},
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
                "baseline_type": "b7_bandit_closed_loop",
                "sample_path": str(sample_path),
                "generator_id": "humanml3d_gt",
                "max_iterations": args.max_iterations,
                "score_increase_tolerance": args.score_increase_tolerance,
                "model_type": args.model_type,
                "model_random_state": args.model_random_state,
                "train_ratio": args.train_ratio,
                "multi_artifact_recipe": ["foot_floating", "global_jitter"],
                "target_evaluators": list(TARGET_EVALUATORS),
                "total_artifact_score_definition": "mean(FootFloating_max, VelocityJitter_max)",
                "state_features": list(("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")),
                "netgain_weight_status": "calibrated_protocol_a_v1",
                "netgain_weights": dict(netgain_weights),
                "evaluator_config_hashes": evaluator_config_hashes,
                "evaluator_severity_versions": evaluator_severity_versions,
                "tool_class_hashes": tool_class_hashes,
                "selections": {"foot_floating_AND_global_jitter": results[-1]},
                "negative_result": False,
            }
            out = raw_dir / f"{timestamp}_{args.task_id}_{trial_id}.json"
            out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    ng = np.array([r["netgain_provisional"] for r in results], dtype=np.float64)
    n_calls_arr = np.array([r["tool_call_count"] for r in results], dtype=np.int64)
    rollback_count = sum(1 for r in results if r["rolled_back"])
    max_iter_count = sum(1 for r in results if r["max_iterations_reached"])
    bandit_stop_count = sum(1 for r in results if r["stop_reason"] == "bandit_chose_stop")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "baseline_b7_bandit_multi_summary",
        "task_id": args.task_id,
        "split_id": args.split_id,
        "baseline_type": "b7_bandit_closed_loop",
        "max_iterations": args.max_iterations,
        "model_type": args.model_type,
        "model_random_state": args.model_random_state,
        "train_ratio": args.train_ratio,
        "n_train_tuples": len(train_tuples),
        "n_eval_trials": len(results),
        "train_metrics": train_metrics,
        "multi_artifact_recipe": ["foot_floating", "global_jitter"],
        "target_evaluators": list(TARGET_EVALUATORS),
        "netgain_provisional": {
            "mean": float(ng.mean()), "median": float(np.median(ng)),
            "min": float(ng.min()), "max": float(ng.max()),
            "p25": float(np.percentile(ng, 25)), "p75": float(np.percentile(ng, 75)),
        },
        "tool_call_count": {
            "mean": float(n_calls_arr.mean()) if len(n_calls_arr) else 0.0,
            "median": float(np.median(n_calls_arr)) if len(n_calls_arr) else 0.0,
            "max": int(n_calls_arr.max()) if len(n_calls_arr) else 0,
        },
        "termination": {
            "rolled_back": rollback_count,
            "max_iterations_reached": max_iter_count,
            "bandit_chose_stop": bandit_stop_count,
        },
        "trial_ids": [r["trial_id"] for r in results],
        "results": results,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OK] wrote summary to {args.output}")
        print(f"  n_eval={len(results)}")
        print(f"  NetGain median={summary['netgain_provisional']['median']:+.5f}, "
              f"mean={summary['netgain_provisional']['mean']:+.5f}")
        print(f"  tool_call_count median={summary['tool_call_count']['median']:.1f}, "
              f"max={summary['tool_call_count']['max']}")
        print(f"  termination: {summary['termination']}")


if __name__ == "__main__":
    main()
