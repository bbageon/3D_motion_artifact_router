"""RL-1 alternative model comparison — 사용자 directive (2026-05-25 Order 3.6-B).

추천 4 model (사용자 directive):
  1. RandomForest (RF, baseline)
  2. LogisticRegression (LR, linear)
  3. HistGradientBoosting (HGB, tabular 강함)
  4. MLP (nonlinear, regularized)

5 평가 기준:
  - eval_acc (action prediction accuracy on held-out pairs)
  - closed-loop NetGain (synthetic + G2)
  - rollback ratio (synthetic 의 score_violation_rollback)
  - B2 paired test (synthetic + G2)
  - oracle gap closure ratio

본 도구는 4 model 각각 학습 + closed-loop rollout + paired test 통합. 단일 JSON
output 으로 비교 표.

CLI:
    python -m tools.rl1_alt_models_compare \\
        --training-data evals/snapshots/rl1_imitation_training_data_v2.json \\
        --b2-multi evals/snapshots/baseline_b2_fixed_smoothing_multi_v2_n60.json \\
        --b2-g2 evals/snapshots/baseline_b2_fixed_smoothing_g2_v2.json \\
        --oracle-multi evals/snapshots/oracle_sequence_multi_v2_n60.json \\
        --oracle-g2 evals/snapshots/oracle_sequence_g2_v1.json \\
        --output evals/snapshots/rl1_alt_models_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import stats

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, EvaluatorReport
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from orchestrator.sequence_imitation import (
    ACTIONS,
    SequenceImitationSelector,
    make_state,
    pairs_to_arrays,
    split_train_eval_by_trial,
)
from tools.synthetic_injection import inject_foot_floating, inject_jitter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMANML3D_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
DEFAULT_G2_DIR = REPO_ROOT / "external_assets" / "g2_generated_v1"

ALL_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
TARGET_EVALUATORS_SYN_A = ("FootFloatingEvaluator", "VelocityJitterEvaluator")
TOOL_TO_TARGET_PART = {
    "FootLockTool": "both_feet", "BoneProjectionTool": "right_arm",
    "VelocitySmoothingTool": "full_body",
}

MODELS_TO_COMPARE = ["random_forest", "logistic_regression", "hist_gradient_boosting", "mlp"]


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _eval_scores(motion: np.ndarray, evaluators: list[Any]) -> dict[str, float]:
    return {n: _max_score(ev.evaluate(motion)) for n, ev in zip(ALL_EVALUATORS, evaluators)}


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _multi_inject(clean: np.ndarray, seed: int) -> np.ndarray:
    m1 = inject_foot_floating(clean, lift_height=0.08, seed=seed)
    return inject_jitter(m1, noise_std=0.05, seed=seed + 1000)


def _rollout(
    *, trial_id: str, initial_motion: np.ndarray, reference_motion: np.ndarray,
    target_evals: tuple[str, ...], selector: SequenceImitationSelector,
    tools_by_name: dict[str, CorrectionTool], evaluators: list[Any],
    netgain_weights: dict[str, float], k_max: int,
) -> dict[str, Any]:
    """Generic rollout — target_evals 가 synthetic_A 또는 ALL_EVALUATORS."""
    motion = initial_motion.copy()
    T = motion.shape[0]
    frame_range = (0, T - 1)
    scores_init = _eval_scores(motion, evaluators)
    target_init = float(np.mean([scores_init[n] for n in target_evals]))
    mpjpe_init = _mpjpe(initial_motion, reference_motion) if not np.allclose(initial_motion, reference_motion) else 0.0

    prev_scores: Optional[dict[str, float]] = None
    prev_tool = "NONE"; prev_strength = "NONE"
    sequence = []
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
        total_before = sum(current_scores.values()); total_after = sum(new_scores.values())
        if total_after > total_before + 0.01:
            sequence.append(["SCORE_VIOLATION_ROLLBACK", pred.tool_name, pred.strength])
            stop_reason = "score_violation_rollback"
            break
        motion = new_motion
        sequence.append([pred.tool_name, pred.target_part, pred.strength])
        cum_corr_mag += float(report.correction_magnitude)
        prev_scores = current_scores
        prev_tool = pred.tool_name; prev_strength = pred.strength

    scores_final = _eval_scores(motion, evaluators)
    target_final = float(np.mean([scores_final[n] for n in target_evals]))
    target_delta = target_final - target_init
    fidelity_loss = _mpjpe(motion, reference_motion) - mpjpe_init
    alpha = float(netgain_weights["alpha"])
    netgain = -target_delta - alpha * fidelity_loss
    return {
        "trial_id": trial_id,
        "sequence": sequence,
        "length": len([s for s in sequence if s[0] not in ("STOP", "APPLY_FAIL", "SCORE_VIOLATION_ROLLBACK")]),
        "stop_reason": stop_reason,
        "target_delta": float(target_delta),
        "fidelity_loss": float(fidelity_loss),
        "netgain": float(netgain),
    }


def _paired(arr_a: np.ndarray, arr_b: np.ndarray) -> dict:
    diff = arr_a - arr_b
    nonzero = diff[np.abs(diff) > 1e-12]
    if nonzero.size > 0:
        w_g = stats.wilcoxon(arr_a, arr_b, alternative="greater", zero_method="wilcox")
        p_g = float(w_g.pvalue)
    else:
        p_g = 1.0
    d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 1e-15 else 0.0
    return {
        "n": len(diff), "a_median": float(np.median(arr_a)), "a_mean": float(arr_a.mean()),
        "b_median": float(np.median(arr_b)), "b_mean": float(arr_b.mean()),
        "diff_median": float(np.median(diff)), "diff_mean": float(diff.mean()),
        "n_strict_a": int((diff > 1e-12).sum()), "n_loss_a": int((diff < -1e-12).sum()),
        "wilcoxon_p_greater": p_g, "cohen_d_paired": d,
    }


def _run_one_model(
    *, model_type: str, X_train, y_train, X_eval, y_eval,
    eval_trials_syn: list[str], eval_trials_g2: list[str],
    humanml3d_dir: Path, g2_dir: Path, tools_by_name, evaluators,
    netgain_weights, k_max: int, seed: int, model_random_state: int,
    b2_multi_per_trial: dict, b2_g2_per_trial: dict,
    oracle_multi_per_trial: dict, oracle_g2_per_trial: dict,
) -> dict[str, Any]:
    print(f"\n=== Model: {model_type} ===")
    selector = SequenceImitationSelector(model_type=model_type, random_state=model_random_state)
    metrics = selector.train(X_train, y_train)
    eval_acc = float(selector.model.score(X_eval, y_eval))
    print(f"  train_acc: {metrics['train_accuracy']:.4f}, eval_acc: {eval_acc:.4f}")

    # Synthetic rollout.
    syn_results = []
    for tid in eval_trials_syn:
        path = humanml3d_dir / f"{tid}.npy"
        if not path.exists():
            continue
        clean = np.load(str(path)).astype(np.float64)
        corrupted = _multi_inject(clean, seed=seed)
        res = _rollout(
            trial_id=tid, initial_motion=corrupted, reference_motion=clean,
            target_evals=TARGET_EVALUATORS_SYN_A, selector=selector,
            tools_by_name=tools_by_name, evaluators=evaluators,
            netgain_weights=netgain_weights, k_max=k_max,
        )
        syn_results.append(res)

    # G2 rollout.
    g2_results = []
    for tid in eval_trials_g2:
        path = g2_dir / f"{tid}.npy"
        if not path.exists():
            continue
        motion = np.load(str(path)).astype(np.float64)
        res = _rollout(
            trial_id=tid, initial_motion=motion, reference_motion=motion,  # Protocol B
            target_evals=ALL_EVALUATORS, selector=selector,
            tools_by_name=tools_by_name, evaluators=evaluators,
            netgain_weights=netgain_weights, k_max=k_max,
        )
        g2_results.append(res)

    # Aggregate.
    syn_ng = np.array([r["netgain"] for r in syn_results])
    g2_ng = np.array([r["netgain"] for r in g2_results])
    syn_rollback = sum(1 for r in syn_results if r["stop_reason"] == "score_violation_rollback")
    syn_selector_stop = sum(1 for r in syn_results if r["stop_reason"] == "selector_stop")
    g2_selector_stop = sum(1 for r in g2_results if r["stop_reason"] == "selector_stop")

    # Paired vs B2.
    syn_common = [r["trial_id"] for r in syn_results if r["trial_id"] in b2_multi_per_trial]
    syn_paired = {}
    if syn_common:
        a = np.array([next(r["netgain"] for r in syn_results if r["trial_id"] == t) for t in syn_common])
        b = np.array([b2_multi_per_trial[t] for t in syn_common])
        syn_paired = _paired(a, b)

    g2_common = [r["trial_id"] for r in g2_results if r["trial_id"] in b2_g2_per_trial]
    g2_paired = {}
    if g2_common:
        a = np.array([next(r["netgain"] for r in g2_results if r["trial_id"] == t) for t in g2_common])
        b = np.array([b2_g2_per_trial[t] for t in g2_common])
        g2_paired = _paired(a, b)

    # Oracle gap closure.
    def _gap_closure(rl_dict, b2_dict, oracle_dict) -> float:
        common = sorted(rl_dict.keys() & b2_dict.keys() & oracle_dict.keys())
        if not common: return 0.0
        gap_init = np.mean([oracle_dict[t] - b2_dict[t] for t in common])
        gap_rl = np.mean([rl_dict[t] - b2_dict[t] for t in common])
        return float(gap_rl / gap_init) if gap_init != 0 else 0.0

    syn_rl_per_trial = {r["trial_id"]: r["netgain"] for r in syn_results}
    g2_rl_per_trial = {r["trial_id"]: r["netgain"] for r in g2_results}
    syn_closure = _gap_closure(syn_rl_per_trial, b2_multi_per_trial, oracle_multi_per_trial)
    g2_closure = _gap_closure(g2_rl_per_trial, b2_g2_per_trial, oracle_g2_per_trial)

    return {
        "model_type": model_type,
        "model_random_state": model_random_state,
        "train_accuracy": metrics["train_accuracy"],
        "eval_accuracy": eval_acc,
        "synthetic": {
            "n_eval": len(syn_results),
            "netgain": {"median": float(np.median(syn_ng)) if len(syn_ng) else 0.0,
                        "mean": float(syn_ng.mean()) if len(syn_ng) else 0.0},
            "rollback_count": syn_rollback,
            "rollback_ratio": syn_rollback / max(len(syn_results), 1),
            "selector_stop_count": syn_selector_stop,
            "length_freq": dict(Counter(r["length"] for r in syn_results)),
            "first_action_freq": dict(Counter(r["sequence"][0][0] if r["sequence"] else "n/a" for r in syn_results)),
            "vs_b2_paired": syn_paired,
            "oracle_gap_closure": syn_closure,
        },
        "g2": {
            "n_eval": len(g2_results),
            "netgain": {"median": float(np.median(g2_ng)) if len(g2_ng) else 0.0,
                        "mean": float(g2_ng.mean()) if len(g2_ng) else 0.0},
            "selector_stop_count": g2_selector_stop,
            "length_freq": dict(Counter(r["length"] for r in g2_results)),
            "first_action_freq": dict(Counter(r["sequence"][0][0] if r["sequence"] else "n/a" for r in g2_results)),
            "vs_b2_paired": g2_paired,
            "oracle_gap_closure": g2_closure,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-1 alternative model comparison")
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--b2-multi", type=Path, required=True)
    parser.add_argument("--b2-g2", type=Path, required=True)
    parser.add_argument("--oracle-multi", type=Path, required=True)
    parser.add_argument("--oracle-g2", type=Path, required=True)
    parser.add_argument("--humanml3d-dir", type=Path, default=DEFAULT_HUMANML3D_DIR)
    parser.add_argument("--g2-dir", type=Path, default=DEFAULT_G2_DIR)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-random-state", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1

    with open(args.training_data, encoding="utf-8") as f:
        td = json.load(f)
    pairs = td["pairs"]
    train_pairs, eval_pairs = split_train_eval_by_trial(
        pairs, train_ratio=args.train_ratio, seed=args.seed)
    X_train, y_train = pairs_to_arrays(train_pairs)
    X_eval, y_eval = pairs_to_arrays(eval_pairs)
    eval_trials_syn = sorted({p["trial_id"] for p in eval_pairs if p["distribution"] == "synthetic_multi"})
    eval_trials_g2 = sorted({p["trial_id"] for p in eval_pairs if p["distribution"] == "g2_natural"})
    print(f"[INFO] split: train {len(y_train)} / eval {len(y_eval)} pairs")
    print(f"  synthetic eval trials: {len(eval_trials_syn)}, G2 eval trials: {len(eval_trials_g2)}")

    # Load reference NetGain dicts.
    with open(args.b2_multi, encoding="utf-8") as f:
        b2m = json.load(f)
    b2_multi_per_trial = {r["trial_id"]: r["netgain_provisional"] for r in b2m["results"]}
    with open(args.b2_g2, encoding="utf-8") as f:
        b2g = json.load(f)
    b2_g2_per_trial = {r["trial_id"]: r["netgain"] for r in b2g["results"]}
    with open(args.oracle_multi, encoding="utf-8") as f:
        om = json.load(f)
    oracle_multi_per_trial = {r["trial_id"]: r["best_A"]["netgain_A"] for r in om["per_sample"] if r.get("best_A")}
    with open(args.oracle_g2, encoding="utf-8") as f:
        og = json.load(f)
    oracle_g2_per_trial = {r["trial_id"]: r["best"]["netgain"] for r in og["per_sample"] if r.get("best")}

    tools_by_name: dict[str, CorrectionTool] = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "BoneProjectionTool": BoneProjectionTool(),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
    }
    evaluators = list(DEFAULT_EVALUATORS)

    all_results = []
    for mt in MODELS_TO_COMPARE:
        r = _run_one_model(
            model_type=mt, X_train=X_train, y_train=y_train,
            X_eval=X_eval, y_eval=y_eval,
            eval_trials_syn=eval_trials_syn, eval_trials_g2=eval_trials_g2,
            humanml3d_dir=args.humanml3d_dir, g2_dir=args.g2_dir,
            tools_by_name=tools_by_name, evaluators=evaluators,
            netgain_weights=netgain_weights, k_max=args.k_max,
            seed=args.seed, model_random_state=args.model_random_state,
            b2_multi_per_trial=b2_multi_per_trial, b2_g2_per_trial=b2_g2_per_trial,
            oracle_multi_per_trial=oracle_multi_per_trial,
            oracle_g2_per_trial=oracle_g2_per_trial,
        )
        all_results.append(r)

    summary = {
        "schema_version": "1.0.0",
        "record_type": "rl1_alt_models_comparison",
        "training_data": str(args.training_data),
        "split_seed": args.seed, "train_ratio": args.train_ratio,
        "model_random_state": args.model_random_state, "k_max": args.k_max,
        "n_train_pairs": len(y_train), "n_eval_pairs": len(y_eval),
        "eval_trials_syn": eval_trials_syn, "eval_trials_g2": eval_trials_g2,
        "b2_multi_path": str(args.b2_multi), "b2_g2_path": str(args.b2_g2),
        "oracle_multi_path": str(args.oracle_multi), "oracle_g2_path": str(args.oracle_g2),
        "models_compared": MODELS_TO_COMPARE,
        "per_model_results": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {args.output}")

    # Summary table.
    print(f"\n=== Summary table (5 criteria, n_eval_syn={len(eval_trials_syn)}, n_eval_g2={len(eval_trials_g2)}) ===")
    print(f"  {'model':25s} {'eval_acc':>10s} {'syn_ng_med':>11s} {'syn_rollb':>10s} {'syn_d_vs_b2':>12s} {'syn_closure':>12s} {'g2_d_vs_b2':>11s} {'g2_closure':>11s}")
    for r in all_results:
        syn = r["synthetic"]; g2 = r["g2"]
        syn_d = syn["vs_b2_paired"].get("cohen_d_paired", 0.0)
        g2_d = g2["vs_b2_paired"].get("cohen_d_paired", 0.0)
        print(f"  {r['model_type']:25s} {r['eval_accuracy']:>10.4f} {syn['netgain']['median']:>+11.4f} "
              f"{syn['rollback_ratio']:>9.1%} {syn_d:>+12.3f} {syn['oracle_gap_closure']:>+11.1%} "
              f"{g2_d:>+11.3f} {g2['oracle_gap_closure']:>+10.1%}")


if __name__ == "__main__":
    main()
