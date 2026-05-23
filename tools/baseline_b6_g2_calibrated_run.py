"""Step 5-B: G2 small-calibration — B6 retrained on G2 oracle labels.

사용자 directive: "G2 30개 중 일부로 B6/B7 을 재보정, 나머지에 평가. zero-shot 은
실패했지만 small-calibration 으로 회복 가능한가?"

Pipeline:
  1. G2 oracle training data (build_g2_oracle_training_data.py 산출).
  2. Sample-level disjoint split (seed=42, 0.5: 15 train + 15 eval).
  3. B6 retrain on G2 train 15 (G2 oracle labels).
  4. Eval on G2 eval 15.
  5. 비교 (paired):
     - Zero-shot B6 (synthetic-trained, oracle_training_data_v2): same G2 eval 15 적용.
     - Small-calibration B6 (G2-trained): G2 eval 15 적용.

CLI:
    python -m tools.baseline_b6_g2_calibrated_run \\
        --g2-training-data evals/snapshots/g2_oracle_training_data_v1.json \\
        --synthetic-training-data evals/snapshots/oracle_training_data_v2.json \\
        --g2-batch-dir external_assets/g2_generated_v1 \\
        --seed 42 --train-ratio 0.5 \\
        --task-id baseline_b6_g2_calibrated_v1 \\
        --output evals/snapshots/baseline_b6_g2_calibrated_v1.json
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
from orchestrator.supervised_selector import (
    SupervisedSelector,
    split_train_eval_by_trial,
    tuples_to_arrays,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "baseline_b6_g2_calibrated_sample"
ALL_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
ARTIFACT_KINDS = ["foot_floating", "bone_stretch_right_arm", "global_jitter"]
EVAL_TO_ARTIFACT = {
    "FootFloatingEvaluator": "foot_floating",
    "BoneLengthEvaluator": "bone_stretch_right_arm",
    "VelocityJitterEvaluator": "global_jitter",
}
TOOL_BY_NAME = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _get_severity_version(evaluator: Any) -> str:
    mod = sys.modules.get(type(evaluator).__module__)
    if mod is None:
        return "unversioned"
    return getattr(mod, "SEVERITY_VERSION", "unversioned")


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _target_score_all(reports_dict: dict[str, list[EvaluatorReport]]) -> float:
    return float(np.mean([_max_score(reports_dict.get(n, [])) for n in ALL_EVALUATORS]))


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _load_g2_motions(batch_dir: Path) -> dict[str, tuple[Path, dict[str, Any], np.ndarray]]:
    out: dict[str, tuple[Path, dict[str, Any], np.ndarray]] = {}
    for npy_path in sorted(batch_dir.glob("motion_*.npy")):
        meta_path = npy_path.with_suffix(".json")
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        motion = np.load(str(npy_path)).astype(np.float64)
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            continue
        trial_id = meta.get("trial_id", npy_path.stem)
        out[trial_id] = (npy_path, meta, motion)
    return out


def _apply_b6(motion: np.ndarray, evaluators: list[Any], selector: SupervisedSelector,
              alpha: float, beta: float, gamma: float) -> dict[str, Any]:
    """Single-step B6: primary artifact = max-score → predict → apply."""
    T = motion.shape[0]
    reports_before = {ev.name: ev.evaluate(motion) for ev in evaluators}
    target_before = _target_score_all(reports_before)
    eval_scores = [_max_score(reports_before.get(n, [])) for n in ALL_EVALUATORS]
    primary_ev = max(zip(ALL_EVALUATORS, eval_scores), key=lambda kv: kv[1])[0]
    primary_artifact = EVAL_TO_ARTIFACT[primary_ev]
    artifact_onehot = [1 if k == primary_artifact else 0 for k in ARTIFACT_KINDS]
    state = np.array(artifact_onehot + eval_scores, dtype=np.float64)
    pred = selector.predict(state, primary_artifact)
    tool = TOOL_BY_NAME.get(pred.tool_name)
    if tool is None:
        return {"decision": "skip", "netgain": 0.0, "tool": "NONE", "strength": "n/a",
                "target_delta": 0.0, "fidelity_loss": 0.0, "correction_magnitude": 0.0,
                "tool_call_count": 0,
                "evaluator_before": {n: _max_score(reports_before.get(n, [])) for n in ALL_EVALUATORS},
                "evaluator_after": {n: _max_score(reports_before.get(n, [])) for n in ALL_EVALUATORS}}
    try:
        corrected, report = tool.apply(motion, target_part=pred.target_part, target_joints=[],
                                       frame_range=(0, T - 1), strength=pred.strength)
        reports_after = {ev.name: ev.evaluate(corrected) for ev in evaluators}
        target_after = _target_score_all(reports_after)
        target_delta = target_after - target_before
        fidelity_loss = _mpjpe(corrected, motion)
        cm = float(report.correction_magnitude)
        netgain = -target_delta - alpha * fidelity_loss - beta * cm - gamma * 1.0
        return {
            "decision": "revise", "tool": pred.tool_name, "strength": pred.strength,
            "primary_artifact": primary_artifact, "primary_evaluator": primary_ev,
            "target_score_before": target_before, "target_score_after": target_after,
            "target_delta": target_delta, "fidelity_loss": fidelity_loss,
            "correction_magnitude": cm, "tool_call_count": 1, "netgain": netgain,
            "evaluator_before": {n: _max_score(reports_before.get(n, [])) for n in ALL_EVALUATORS},
            "evaluator_after": {n: _max_score(reports_after.get(n, [])) for n in ALL_EVALUATORS},
        }
    except ValueError as e:
        return {"decision": "skip", "netgain": 0.0, "tool": pred.tool_name, "strength": pred.strength,
                "target_delta": 0.0, "fidelity_loss": 0.0, "correction_magnitude": 0.0,
                "tool_call_count": 0, "skip_reason": str(e),
                "evaluator_before": {n: _max_score(reports_before.get(n, [])) for n in ALL_EVALUATORS},
                "evaluator_after": {n: _max_score(reports_before.get(n, [])) for n in ALL_EVALUATORS}}


def main() -> None:
    parser = argparse.ArgumentParser(description="B6 G2 small-calibration (Step 5-B)")
    parser.add_argument("--g2-training-data", type=Path, required=True,
                        help="G2 oracle training data (build_g2_oracle_training_data.py output).")
    parser.add_argument("--synthetic-training-data", type=Path, required=True,
                        help="Synthetic oracle training data (oracle_training_data_v2.json).")
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--model-random-state", type=int, default=1)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--split-id", default="g2_small_calibration_v1")
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
    alpha = netgain_weights["alpha"]
    beta = netgain_weights["beta"]
    gamma = netgain_weights["gamma"]

    # Load G2 oracle training data.
    with open(args.g2_training_data, encoding="utf-8") as f:
        g2_td = json.load(f)
    g2_tuples = g2_td["tuples"]
    print(f"[INFO] G2 oracle data: {len(g2_tuples)} tuples")

    # Sample-level split.
    g2_train_tuples, g2_eval_tuples = split_train_eval_by_trial(
        g2_tuples, train_ratio=args.train_ratio, seed=args.seed,
    )
    g2_train_ids = {t["trial_id"] for t in g2_train_tuples}
    g2_eval_ids = sorted({t["trial_id"] for t in g2_eval_tuples})
    print(f"[INFO] G2 train: {len(g2_train_ids)} trials | G2 eval: {len(g2_eval_ids)} trials")

    # Train calibrated B6 on G2 train.
    X_g2_train, tool_g2_train, str_g2_train = tuples_to_arrays(g2_train_tuples)
    calibrated_selector = SupervisedSelector(model_type="random_forest", random_state=args.model_random_state)
    cal_train_metrics = calibrated_selector.train(X_g2_train, tool_g2_train, str_g2_train)
    print(f"[INFO] calibrated trained on G2 {len(g2_train_ids)}: {cal_train_metrics}")

    # Train zero-shot B6 on synthetic oracle.
    with open(args.synthetic_training_data, encoding="utf-8") as f:
        syn_td = json.load(f)
    syn_tuples = syn_td["tuples"]
    syn_train_tuples, _ = split_train_eval_by_trial(syn_tuples, train_ratio=0.5, seed=args.seed)
    X_syn, tool_syn, str_syn = tuples_to_arrays(syn_train_tuples)
    zero_shot_selector = SupervisedSelector(model_type="random_forest", random_state=args.model_random_state)
    zs_train_metrics = zero_shot_selector.train(X_syn, tool_syn, str_syn)
    print(f"[INFO] zero-shot trained on synthetic {len(syn_train_tuples)//3} trials: {zs_train_metrics}")

    # Load G2 motions.
    g2_motions = _load_g2_motions(args.g2_batch_dir)
    evaluators = list(DEFAULT_EVALUATORS)
    evaluator_config_hashes = {ev.name: ev.evaluator_class_hash() for ev in evaluators}
    evaluator_severity_versions = {ev.name: _get_severity_version(ev) for ev in evaluators}
    tool_class_hashes = {name: t.tool_class_hash() for name, t in TOOL_BY_NAME.items()}

    raw_dir = Path(args.raw_output_dir).resolve() if args.raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    # Eval on G2 eval set.
    results: dict[str, list[dict[str, Any]]] = {"zero_shot": [], "calibrated": []}
    for trial_id in g2_eval_ids:
        if trial_id not in g2_motions:
            print(f"[WARN] G2 motion not found: {trial_id}")
            continue
        npy_path, g2_meta, motion = g2_motions[trial_id]

        # zero_shot
        zs_result = _apply_b6(motion, evaluators, zero_shot_selector, alpha, beta, gamma)
        zs_result["trial_id"] = trial_id
        zs_result["g2_prompt"] = g2_meta.get("prompt", "")[:80]
        results["zero_shot"].append(zs_result)

        # calibrated
        cal_result = _apply_b6(motion, evaluators, calibrated_selector, alpha, beta, gamma)
        cal_result["trial_id"] = trial_id
        cal_result["g2_prompt"] = g2_meta.get("prompt", "")[:80]
        results["calibrated"].append(cal_result)

        if raw_dir is not None:
            timestamp = _now_iso()
            record = {
                "schema_version": SCHEMA_VERSION, "record_type": RECORD_TYPE,
                "timestamp": timestamp, "task_id": args.task_id, "split_id": args.split_id,
                "trial_id": trial_id, "baseline_type": "b6_g2_calibrated",
                "sample_path": str(npy_path), "generator_id": g2_meta.get("generator_id"),
                "g2_prompt": g2_meta.get("prompt"), "g2_seed": g2_meta.get("seed"),
                "model_random_state": args.model_random_state,
                "train_ratio": args.train_ratio,
                "g2_train_ids_count": len(g2_train_ids),
                "g2_eval_ids_count": len(g2_eval_ids),
                "fidelity_loss_reference": "original_g2_motion",
                "target_score_definition": "mean(FootFloating, BoneLength, VelocityJitter max scores)",
                "netgain_weight_status": "calibrated_protocol_a_v1",
                "evaluator_config_hashes": evaluator_config_hashes,
                "evaluator_severity_versions": evaluator_severity_versions,
                "tool_class_hashes": tool_class_hashes,
                "selections": {"g2_natural_multi_artifact": {
                    "zero_shot": zs_result,
                    "calibrated": cal_result,
                    "netgain": cal_result["netgain"],  # for paired_test compat (oracle field)
                }},
                "transfer_diagnostic_note": "G2 small-calibration: zero-shot vs G2-calibrated B6.",
                "negative_result": False,
            }
            out = raw_dir / f"{timestamp}_{args.task_id}_{trial_id}.json"
            out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    # Aggregate.
    def _agg(rs: list[dict[str, Any]]) -> dict[str, Any]:
        ng = np.array([r["netgain"] for r in rs])
        td = np.array([r["target_delta"] for r in rs])
        fl = np.array([r["fidelity_loss"] for r in rs])
        from collections import Counter
        tool_dist = Counter(r.get("tool") for r in rs)
        strength_dist = Counter(r.get("strength") for r in rs)
        return {
            "n": len(rs),
            "netgain": {"mean": float(ng.mean()), "median": float(np.median(ng)),
                        "min": float(ng.min()), "max": float(ng.max())},
            "target_delta": {"mean": float(td.mean()), "median": float(np.median(td))},
            "fidelity_loss": {"mean": float(fl.mean()), "median": float(np.median(fl))},
            "tool_distribution": dict(tool_dist),
            "strength_distribution": dict(strength_dist),
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "baseline_b6_g2_calibrated_summary",
        "task_id": args.task_id, "split_id": args.split_id,
        "n_g2_train": len(g2_train_ids),
        "n_g2_eval": len(g2_eval_ids),
        "g2_train_ids": sorted(g2_train_ids),
        "g2_eval_ids": g2_eval_ids,
        "fidelity_loss_reference": "original_g2_motion",
        "target_score_definition": "mean(FootFloating, BoneLength, VelocityJitter max scores)",
        "calibrated_train_metrics": cal_train_metrics,
        "zero_shot_train_metrics": zs_train_metrics,
        "results_zero_shot": _agg(results["zero_shot"]),
        "results_calibrated": _agg(results["calibrated"]),
        "per_sample_zero_shot": results["zero_shot"],
        "per_sample_calibrated": results["calibrated"],
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"\n[OK] wrote {args.output}")
    print(f"\n=== Zero-shot B6 (synthetic-trained) on G2 eval {len(g2_eval_ids)} ===")
    print(f"  NetGain median={summary['results_zero_shot']['netgain']['median']:+.5f}, "
          f"mean={summary['results_zero_shot']['netgain']['mean']:+.5f}")
    print(f"  Tools: {summary['results_zero_shot']['tool_distribution']}")
    print(f"  Strengths: {summary['results_zero_shot']['strength_distribution']}")
    print(f"\n=== Calibrated B6 (G2-trained on {len(g2_train_ids)}) on G2 eval {len(g2_eval_ids)} ===")
    print(f"  NetGain median={summary['results_calibrated']['netgain']['median']:+.5f}, "
          f"mean={summary['results_calibrated']['netgain']['mean']:+.5f}")
    print(f"  Tools: {summary['results_calibrated']['tool_distribution']}")
    print(f"  Strengths: {summary['results_calibrated']['strength_distribution']}")


if __name__ == "__main__":
    main()
