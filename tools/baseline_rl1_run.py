"""RL-1 sequence-oracle imitation policy — closed-loop evaluation.

사용자 directive (2026-05-25):
> "1. synthetic에서는 oracle의 multi-step sequence를 얼마나 따라가는가
>  2. G2에서는 STOP을 얼마나 잘 배우는가
>  3. B2/B5/B6/B7보다 NetGain이 좋아지는가
>  4. oracle gap을 얼마나 닫는가"

본 도구는 (state, action) imitation 학습 + closed-loop evaluation:
  1. Training data 로드 (rl1_imitation_training_data_v1.json).
  2. Sample-level disjoint split (synthetic + G2 각각 train_ratio=0.7).
  3. SequenceImitationSelector train.
  4. Eval: closed-loop policy roll-out (max_iterations=k_max).
     - 매 step state 재계산 → selector.predict() → action 적용 (or STOP).
  5. Compare to: B2, oracle ceiling.

Synthetic eval: Step 3 multi-artifact recipe (inject foot+jitter) on HumanML3D.
G2 eval: G2 generated motions (original = corrupted, no clean GT).

NetGain (per distribution):
  - Synthetic: target=mean(foot+jitter), reference=clean_gt (Protocol A simplified).
  - G2: target=mean(all 3), reference=original_g2 (Protocol B simplified).

CLI:
    python -m tools.baseline_rl1_run \\
        --training-data evals/snapshots/rl1_imitation_training_data_v1.json \\
        --k-max 5 \\
        --train-ratio 0.7 --seed 42 --model-random-state 1 \\
        --task-id baseline_rl1_imitation_v1 \\
        --raw-output-dir evals/raw \\
        --output evals/snapshots/baseline_rl1_imitation_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, EvaluatorReport
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from orchestrator.sequence_imitation import (
    ACTIONS,
    SequenceImitationSelector,
    TOOL_TO_TARGET_PART,
    make_state,
    pairs_to_arrays,
    split_train_eval_by_trial,
)
from tools.synthetic_injection import inject_foot_floating, inject_jitter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMANML3D_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
DEFAULT_G2_DIR = REPO_ROOT / "external_assets" / "g2_generated_v1"

ALL_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
TARGET_EVALUATORS_SYN_A = ("FootFloatingEvaluator", "VelocityJitterEvaluator")  # synthetic multi recipe.

SCHEMA_VERSION = "1.0.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _get_severity_version(evaluator: Any) -> str:
    mod = sys.modules.get(type(evaluator).__module__)
    if mod is None:
        return "unversioned"
    return getattr(mod, "SEVERITY_VERSION", "unversioned")


def _max_score(reports: list[EvaluatorReport]) -> float:
    if not reports:
        return 0.0
    return float(max(r.score for r in reports))


def _eval_scores(motion: np.ndarray, evaluators: list[Any]) -> dict[str, float]:
    return {n: _max_score(ev.evaluate(motion)) for n, ev in zip(ALL_EVALUATORS, evaluators)}


def _target_score_syn_A(scores: dict[str, float]) -> float:
    return float(np.mean([scores[n] for n in TARGET_EVALUATORS_SYN_A]))


def _target_score_full(scores: dict[str, float]) -> float:
    return float(np.mean([scores[n] for n in ALL_EVALUATORS]))


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _multi_inject(clean: np.ndarray, seed: int) -> np.ndarray:
    m1 = inject_foot_floating(clean, lift_height=0.08, seed=seed)
    return inject_jitter(m1, noise_std=0.05, seed=seed + 1000)


def _rollout_synthetic(
    *,
    trial_id: str,
    clean_motion: np.ndarray,
    selector: SequenceImitationSelector,
    tools_by_name: dict[str, CorrectionTool],
    evaluators: list[Any],
    netgain_weights: dict[str, float],
    k_max: int,
    seed: int,
) -> dict[str, Any]:
    """Synthetic multi-artifact closed-loop roll-out. NetGain target = mean(foot+jitter)."""
    corrupted = _multi_inject(clean_motion, seed=seed)
    motion = corrupted.copy()
    T = motion.shape[0]
    frame_range = (0, T - 1)
    scores_init = _eval_scores(motion, evaluators)
    target_init_A = _target_score_syn_A(scores_init)
    target_init_full = _target_score_full(scores_init)
    mpjpe_corrupted_clean = _mpjpe(corrupted, clean_motion)

    prev_scores: Optional[dict[str, float]] = None
    prev_tool = "NONE"
    prev_strength = "NONE"
    sequence: list[list[str]] = []
    cum_corr_mag = 0.0
    stop_reason = "max_depth_no_stop"

    for step_idx in range(k_max):
        current_scores = _eval_scores(motion, evaluators)
        state = make_state(
            current_scores=current_scores, prev_scores=prev_scores,
            step_index=step_idx, prev_tool=prev_tool, prev_strength=prev_strength,
            k_max=k_max,
        )
        pred = selector.predict(state)
        if pred.is_stop:
            sequence.append(["STOP", "n/a", "NONE"])
            stop_reason = "selector_stop"
            break
        # Apply action.
        tool = tools_by_name[pred.tool_name]
        try:
            new_motion, report = tool.apply(motion, target_part=pred.target_part, target_joints=[],
                                            frame_range=frame_range, strength=pred.strength)
        except ValueError as e:
            sequence.append(["APPLY_FAIL", pred.tool_name, pred.strength])
            stop_reason = f"apply_fail: {e}"
            break
        # Score 비감소 check (AGENTS.md §3-4).
        new_scores = _eval_scores(new_motion, evaluators)
        total_before = sum(current_scores.values())
        total_after = sum(new_scores.values())
        if total_after > total_before + 0.01:
            sequence.append(["SCORE_VIOLATION_ROLLBACK", pred.tool_name, pred.strength])
            stop_reason = "score_violation_rollback"
            break
        motion = new_motion
        sequence.append([pred.tool_name, pred.target_part, pred.strength])
        cum_corr_mag += float(report.correction_magnitude)
        prev_scores = current_scores
        prev_tool = pred.tool_name
        prev_strength = pred.strength
    # NetGain.
    scores_final = _eval_scores(motion, evaluators)
    target_final_A = _target_score_syn_A(scores_final)
    target_delta_A = target_final_A - target_init_A
    fidelity_loss = _mpjpe(motion, clean_motion) - mpjpe_corrupted_clean
    alpha = float(netgain_weights["alpha"])
    netgain_A = -target_delta_A - alpha * fidelity_loss
    return {
        "trial_id": trial_id, "distribution": "synthetic_multi",
        "sequence": sequence, "length": len([s for s in sequence if s[0] not in ("STOP", "APPLY_FAIL", "SCORE_VIOLATION_ROLLBACK")]),
        "stop_reason": stop_reason,
        "target_score_init_A": target_init_A, "target_score_final_A": target_final_A,
        "target_delta_A": float(target_delta_A),
        "fidelity_loss_protocol_a": float(fidelity_loss),
        "cumulative_correction_magnitude": float(cum_corr_mag),
        "netgain_A": float(netgain_A),
        "scores_init": scores_init, "scores_final": scores_final,
    }


def _rollout_g2(
    *,
    trial_id: str,
    motion: np.ndarray,  # = original_g2.
    selector: SequenceImitationSelector,
    tools_by_name: dict[str, CorrectionTool],
    evaluators: list[Any],
    netgain_weights: dict[str, float],
    k_max: int,
) -> dict[str, Any]:
    original_g2 = motion.copy()
    T = motion.shape[0]
    frame_range = (0, T - 1)
    scores_init = _eval_scores(motion, evaluators)
    target_init_full = _target_score_full(scores_init)

    prev_scores: Optional[dict[str, float]] = None
    prev_tool = "NONE"
    prev_strength = "NONE"
    sequence: list[list[str]] = []
    cum_corr_mag = 0.0
    stop_reason = "max_depth_no_stop"

    for step_idx in range(k_max):
        current_scores = _eval_scores(motion, evaluators)
        state = make_state(
            current_scores=current_scores, prev_scores=prev_scores,
            step_index=step_idx, prev_tool=prev_tool, prev_strength=prev_strength,
            k_max=k_max,
        )
        pred = selector.predict(state)
        if pred.is_stop:
            sequence.append(["STOP", "n/a", "NONE"])
            stop_reason = "selector_stop"
            break
        tool = tools_by_name[pred.tool_name]
        try:
            new_motion, report = tool.apply(motion, target_part=pred.target_part, target_joints=[],
                                            frame_range=frame_range, strength=pred.strength)
        except ValueError as e:
            sequence.append(["APPLY_FAIL", pred.tool_name, pred.strength])
            stop_reason = f"apply_fail: {e}"
            break
        new_scores = _eval_scores(new_motion, evaluators)
        total_before = sum(current_scores.values())
        total_after = sum(new_scores.values())
        if total_after > total_before + 0.01:
            sequence.append(["SCORE_VIOLATION_ROLLBACK", pred.tool_name, pred.strength])
            stop_reason = "score_violation_rollback"
            break
        motion = new_motion
        sequence.append([pred.tool_name, pred.target_part, pred.strength])
        cum_corr_mag += float(report.correction_magnitude)
        prev_scores = current_scores
        prev_tool = pred.tool_name
        prev_strength = pred.strength

    scores_final = _eval_scores(motion, evaluators)
    target_final_full = _target_score_full(scores_final)
    target_delta_full = target_final_full - target_init_full
    fidelity_loss = _mpjpe(motion, original_g2)
    alpha = float(netgain_weights["alpha"])
    netgain = -target_delta_full - alpha * fidelity_loss
    return {
        "trial_id": trial_id, "distribution": "g2_natural",
        "sequence": sequence, "length": len([s for s in sequence if s[0] not in ("STOP", "APPLY_FAIL", "SCORE_VIOLATION_ROLLBACK")]),
        "stop_reason": stop_reason,
        "target_score_init_full": target_init_full, "target_score_final_full": target_final_full,
        "target_delta_full": float(target_delta_full),
        "fidelity_loss_protocol_b": float(fidelity_loss),
        "cumulative_correction_magnitude": float(cum_corr_mag),
        "netgain": float(netgain),
        "scores_init": scores_init, "scores_final": scores_final,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-1 sequence imitation policy — closed-loop")
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--humanml3d-dir", type=Path, default=DEFAULT_HUMANML3D_DIR)
    parser.add_argument("--g2-dir", type=Path, default=DEFAULT_G2_DIR)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-random-state", type=int, default=1)
    parser.add_argument("--model-type", type=str, default="random_forest")
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--split-id", type=str, default="rl1_v1")
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1

    # Load training data.
    with open(args.training_data, encoding="utf-8") as f:
        td = json.load(f)
    pairs = td["pairs"]
    print(f"[INFO] training data loaded: {len(pairs)} pairs")

    # Sample-level disjoint split, per-distribution.
    train_pairs, eval_pairs = split_train_eval_by_trial(
        pairs, train_ratio=args.train_ratio, seed=args.seed,
    )
    print(f"[INFO] split: train {len(train_pairs)} pairs, eval {len(eval_pairs)} pairs")
    train_trials_syn = sorted({p["trial_id"] for p in train_pairs if p["distribution"] == "synthetic_multi"})
    eval_trials_syn = sorted({p["trial_id"] for p in eval_pairs if p["distribution"] == "synthetic_multi"})
    train_trials_g2 = sorted({p["trial_id"] for p in train_pairs if p["distribution"] == "g2_natural"})
    eval_trials_g2 = sorted({p["trial_id"] for p in eval_pairs if p["distribution"] == "g2_natural"})
    print(f"  synthetic: train {len(train_trials_syn)} | eval {len(eval_trials_syn)}")
    print(f"  G2: train {len(train_trials_g2)} | eval {len(eval_trials_g2)}")

    # Train selector.
    X_train, y_train = pairs_to_arrays(train_pairs)
    selector = SequenceImitationSelector(model_type=args.model_type, random_state=args.model_random_state)
    train_metrics = selector.train(X_train, y_train)
    print(f"[INFO] selector trained: {train_metrics}")

    # Eval-pair-level accuracy (held-out (state, action) match).
    X_eval, y_eval = pairs_to_arrays(eval_pairs)
    eval_acc = float(selector.model.score(X_eval, y_eval))
    print(f"[INFO] eval action-prediction accuracy: {eval_acc:.4f} (on {len(eval_pairs)} held-out pairs)")

    # Closed-loop rollout on eval trials.
    tools_by_name: dict[str, CorrectionTool] = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "BoneProjectionTool": BoneProjectionTool(),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
    }
    evaluators = list(DEFAULT_EVALUATORS)
    evaluator_config_hashes = {ev.name: ev.evaluator_class_hash() for ev in evaluators}
    evaluator_severity_versions = {ev.name: _get_severity_version(ev) for ev in evaluators}
    tool_class_hashes = {name: t.tool_class_hash() for name, t in tools_by_name.items()}

    raw_dir = Path(args.raw_output_dir).resolve() if args.raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    syn_results: list[dict[str, Any]] = []
    for tid in eval_trials_syn:
        path = args.humanml3d_dir / f"{tid}.npy"
        if not path.exists():
            print(f"[WARN] missing humanml3d {tid}")
            continue
        clean = np.load(str(path)).astype(np.float64)
        res = _rollout_synthetic(
            trial_id=tid, clean_motion=clean, selector=selector,
            tools_by_name=tools_by_name, evaluators=evaluators,
            netgain_weights=netgain_weights, k_max=args.k_max, seed=args.seed,
        )
        syn_results.append(res)
        if raw_dir is not None:
            ts = _now_iso()
            record = {
                "schema_version": SCHEMA_VERSION, "record_type": "baseline_rl1_imitation_sample",
                "timestamp": ts, "task_id": args.task_id, "split_id": args.split_id,
                "trial_id": tid, "distribution": "synthetic_multi", "baseline_type": "rl1_imitation",
                "sample_path": str(path), "generator_id": "humanml3d_gt",
                "evaluator_config_hashes": evaluator_config_hashes,
                "evaluator_severity_versions": evaluator_severity_versions,
                "tool_class_hashes": tool_class_hashes,
                "k_max": args.k_max, "train_ratio": args.train_ratio,
                "model_random_state": args.model_random_state, "model_type": args.model_type,
                "netgain_weight_status": "calibrated_protocol_a_v1",
                "netgain_weights": dict(netgain_weights),
                "policy_decision": res, "negative_result": False,
            }
            (raw_dir / f"{ts}_{args.task_id}_{tid}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    g2_results: list[dict[str, Any]] = []
    for tid in eval_trials_g2:
        path = args.g2_dir / f"{tid}.npy"
        if not path.exists():
            print(f"[WARN] missing g2 {tid}")
            continue
        motion = np.load(str(path)).astype(np.float64)
        res = _rollout_g2(
            trial_id=tid, motion=motion, selector=selector,
            tools_by_name=tools_by_name, evaluators=evaluators,
            netgain_weights=netgain_weights, k_max=args.k_max,
        )
        g2_results.append(res)
        if raw_dir is not None:
            ts = _now_iso()
            record = {
                "schema_version": SCHEMA_VERSION, "record_type": "baseline_rl1_imitation_sample",
                "timestamp": ts, "task_id": args.task_id, "split_id": args.split_id,
                "trial_id": tid, "distribution": "g2_natural", "baseline_type": "rl1_imitation",
                "sample_path": str(path), "generator_id": "G2_motiongpt",
                "evaluator_config_hashes": evaluator_config_hashes,
                "evaluator_severity_versions": evaluator_severity_versions,
                "tool_class_hashes": tool_class_hashes,
                "k_max": args.k_max, "train_ratio": args.train_ratio,
                "model_random_state": args.model_random_state, "model_type": args.model_type,
                "netgain_weight_status": "calibrated_protocol_a_v1",
                "netgain_weights": dict(netgain_weights),
                "policy_decision": res, "negative_result": False,
            }
            (raw_dir / f"{ts}_{args.task_id}_{tid}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    def _agg(results: list[dict[str, Any]], ng_field: str) -> dict[str, Any]:
        if not results:
            return {"n": 0}
        ng = np.array([r[ng_field] for r in results])
        lens = Counter(r["length"] for r in results)
        stop_reasons = Counter(r["stop_reason"] for r in results)
        first_actions = Counter(
            (r["sequence"][0][0] if len(r["sequence"]) > 0 else "n/a") for r in results
        )
        return {
            "n": len(results),
            "netgain": {"mean": float(ng.mean()), "median": float(np.median(ng)),
                        "min": float(ng.min()), "max": float(ng.max()),
                        "p25": float(np.percentile(ng, 25)), "p75": float(np.percentile(ng, 75))},
            "length_freq": dict(lens),
            "stop_reason_freq": dict(stop_reasons),
            "first_action_freq": dict(first_actions),
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "baseline_rl1_imitation_summary",
        "task_id": args.task_id, "split_id": args.split_id,
        "k_max": args.k_max, "train_ratio": args.train_ratio,
        "model_type": args.model_type, "model_random_state": args.model_random_state,
        "seed": args.seed,
        "n_train_pairs": len(train_pairs), "n_eval_pairs": len(eval_pairs),
        "train_trials_syn": train_trials_syn, "eval_trials_syn": eval_trials_syn,
        "train_trials_g2": train_trials_g2, "eval_trials_g2": eval_trials_g2,
        "train_metrics": train_metrics,
        "eval_action_prediction_accuracy": eval_acc,
        "netgain_weight_status": "calibrated_protocol_a_v1",
        "netgain_weights": dict(netgain_weights),
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,
        "tool_class_hashes": tool_class_hashes,
        "aggregate_synthetic_A": _agg(syn_results, "netgain_A"),
        "aggregate_g2_full": _agg(g2_results, "netgain"),
        "per_sample_synthetic": syn_results,
        "per_sample_g2": g2_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {args.output}")
    print(f"\n=== Synthetic eval (n={summary['aggregate_synthetic_A']['n']}, NetGain_A) ===")
    if summary["aggregate_synthetic_A"]["n"] > 0:
        print(f"  NetGain median={summary['aggregate_synthetic_A']['netgain']['median']:+.5f}, mean={summary['aggregate_synthetic_A']['netgain']['mean']:+.5f}")
        print(f"  length_freq: {summary['aggregate_synthetic_A']['length_freq']}")
        print(f"  first_action_freq: {summary['aggregate_synthetic_A']['first_action_freq']}")
        print(f"  stop_reason_freq: {summary['aggregate_synthetic_A']['stop_reason_freq']}")
    print(f"\n=== G2 eval (n={summary['aggregate_g2_full']['n']}, NetGain Protocol B) ===")
    if summary["aggregate_g2_full"]["n"] > 0:
        print(f"  NetGain median={summary['aggregate_g2_full']['netgain']['median']:+.5f}, mean={summary['aggregate_g2_full']['netgain']['mean']:+.5f}")
        print(f"  length_freq: {summary['aggregate_g2_full']['length_freq']}")
        print(f"  first_action_freq: {summary['aggregate_g2_full']['first_action_freq']}")
        print(f"  stop_reason_freq: {summary['aggregate_g2_full']['stop_reason_freq']}")


if __name__ == "__main__":
    main()
