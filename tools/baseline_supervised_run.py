"""B6 (supervised selector) measurement — H-2026-205 Stage 0.

Pipeline:
  1. Load training tuples (tools/build_oracle_training_data.py 산출).
  2. Sample-level disjoint split (train/eval).
  3. Train SupervisedSelector (RandomForest default).
  4. Eval split 의 각 trial × artifact 에 대해:
     - inject artifact (same seed as oracle).
     - state vector 계산 (corrupted motion 의 evaluator scores).
     - predict (tool, strength).
     - apply tool → corrected.
     - NetGain 계산 (Protocol A, calibrated_protocol_a_v1).
  5. raw record + summary.

본 baseline 의 NetGain 이 B5 (rule-based) 보다 의미 있게 개선되는지가
[H-2026-205](../evals/hypotheses/H-2026-205.md) 의 정량 검증.

CLI:
    python -m tools.baseline_supervised_run \\
        --training-data evals/snapshots/oracle_training_data_v1.json \\
        --model-type random_forest \\
        --train-ratio 0.7 \\
        --seed 42 \\
        --task-id baseline_b6_supervised_v1 \\
        --split-id supervised_eval_v1 \\
        --raw-output-dir evals/raw \\
        --output evals/snapshots/baseline_b6_supervised_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from correction_tools import (
    BoneProjectionTool,
    FootLockTool,
    VelocitySmoothingTool,
)
from evaluators import DEFAULT_EVALUATORS, EvaluatorReport
from orchestrator.oracle_single_step import (
    CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1,
    DEFAULT_PROVISIONAL_NETGAIN_WEIGHTS,
)
from orchestrator.supervised_selector import (
    ARTIFACT_TO_TARGET_PART,
    SupervisedSelector,
    split_train_eval_by_trial,
    tuples_to_arrays,
)
from tools.synthetic_injection import (
    inject_bone_stretch,
    inject_foot_floating,
    inject_jitter,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "baseline_supervised_sample"
SUMMARY_TYPE = "baseline_supervised_summary"

WEIGHT_PRESETS = {
    "provisional": (DEFAULT_PROVISIONAL_NETGAIN_WEIGHTS, "provisional"),
    "calibrated_protocol_a_v1": (CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1, "calibrated_protocol_a_v1"),
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


def _apply_injection(artifact_kind: str, clean: np.ndarray, seed: int) -> np.ndarray:
    kwargs = {"seed": seed}
    if artifact_kind == "foot_floating":
        return inject_foot_floating(clean, lift_height=0.08, **kwargs)
    if artifact_kind == "bone_stretch_right_arm":
        T = clean.shape[0]
        half = max(1, T // 2)
        stretched = inject_bone_stretch(clean[:half], chain_label="right_arm", stretch_factor=1.30, **kwargs)
        return np.concatenate([stretched, clean[half:]], axis=0)
    if artifact_kind == "global_jitter":
        return inject_jitter(clean, noise_std=0.05, **kwargs)
    raise ValueError(f"unknown artifact_kind {artifact_kind!r}")


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _aggregate_target_score(reports: list[EvaluatorReport]) -> float:
    if not reports:
        return 0.0
    return float(np.mean([r.score for r in reports]))


def _max_score(reports: list[EvaluatorReport]) -> float:
    if not reports:
        return 0.0
    return float(max(r.score for r in reports))


def _measure_one_trial(
    *,
    selector: SupervisedSelector,
    clean: np.ndarray,
    artifact_kind: str,
    seed: int,
    netgain_weights: dict[str, float],
    target_evaluator_name: str,
) -> dict[str, Any]:
    evaluators = list(DEFAULT_EVALUATORS)
    alpha = float(netgain_weights["alpha"])
    beta = float(netgain_weights["beta"])
    gamma = float(netgain_weights["gamma"])

    corrupted = _apply_injection(artifact_kind, clean, seed)
    T = corrupted.shape[0]
    frame_range = (0, T - 1)
    mpjpe_corrupted = _mpjpe(corrupted, clean)

    # State vector: artifact_kind one-hot + evaluator scores.
    artifact_kinds = ["foot_floating", "bone_stretch_right_arm", "global_jitter"]
    artifact_onehot = [1 if k == artifact_kind else 0 for k in artifact_kinds]
    eval_names_ordered = ["FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator"]
    reports_before_dict: dict[str, list[EvaluatorReport]] = {
        ev.name: ev.evaluate(corrupted) for ev in evaluators
    }
    eval_scores = [_max_score(reports_before_dict[name]) for name in eval_names_ordered]
    state_vector = np.array(artifact_onehot + eval_scores, dtype=np.float64)

    # Predict.
    pred = selector.predict(state_vector, artifact_kind)
    decision_trace: dict[str, Any] = {
        "predicted_tool": pred.tool_name,
        "predicted_strength": pred.strength,
        "target_part": pred.target_part,
        "tool_proba": pred.tool_proba,
        "strength_proba": pred.strength_proba,
    }

    # Apply tool.
    tool = TOOL_BY_NAME.get(pred.tool_name)
    if tool is None:
        corrected = corrupted.copy()
        cumulative_mag = 0.0
        n_calls = 0
        decision_trace["applied"] = False
        decision_trace["skip_reason"] = f"unknown_tool_{pred.tool_name}"
    else:
        try:
            corrected, report = tool.apply(
                corrupted,
                target_part=pred.target_part,
                target_joints=[],
                frame_range=frame_range,
                strength=pred.strength,
            )
            cumulative_mag = float(report.correction_magnitude)
            n_calls = 1
            decision_trace["correction_magnitude"] = cumulative_mag
            decision_trace["applied"] = True
        except ValueError as e:
            corrected = corrupted.copy()
            cumulative_mag = 0.0
            n_calls = 0
            decision_trace["applied"] = False
            decision_trace["skip_reason"] = str(e)

    # Evaluate corrected.
    target_score_before = _aggregate_target_score(reports_before_dict.get(target_evaluator_name, []))
    reports_after_dict = {ev.name: ev.evaluate(corrected) for ev in evaluators}
    target_score_after = _aggregate_target_score(reports_after_dict.get(target_evaluator_name, []))
    target_delta = target_score_after - target_score_before
    mpjpe_corrected = _mpjpe(corrected, clean)
    fidelity_loss = mpjpe_corrected - mpjpe_corrupted
    artifact_reduction = -target_delta
    tool_call_cost = float(n_calls)
    netgain = artifact_reduction - alpha * fidelity_loss - beta * cumulative_mag - gamma * tool_call_cost

    cross_delta = {}
    for ev_name in reports_before_dict:
        if ev_name == target_evaluator_name:
            continue
        b = _aggregate_target_score(reports_before_dict[ev_name])
        a = _aggregate_target_score(reports_after_dict.get(ev_name, []))
        cross_delta[ev_name] = float(a - b)

    return {
        "artifact_kind": artifact_kind,
        "target_evaluator": target_evaluator_name,
        "orchestrator": "SupervisedSelector",
        "state_vector": state_vector.tolist(),
        "decision_trace": decision_trace,
        "target_score_before": float(target_score_before),
        "target_score_after": float(target_score_after),
        "target_delta": float(target_delta),
        "fidelity_loss_protocol_a": float(fidelity_loss),
        "correction_magnitude": float(cumulative_mag),
        "tool_call_cost": float(tool_call_cost),
        "cross_evaluator_delta": cross_delta,
        "netgain_provisional": float(netgain),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B6 (supervised selector) measurement")
    parser.add_argument("--training-data", type=Path, required=True,
                        help="oracle_training_data_v1.json (build_oracle_training_data.py output)")
    parser.add_argument("--model-type", type=str, default="random_forest",
                        choices=["random_forest", "logistic_regression", "dummy_most_frequent"])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42, help="split seed + injection seed.")
    parser.add_argument("--model-random-state", type=int, default=None,
                        help="model (RandomForest) random_state. None 이면 --seed 사용. n_seeds ≥ 3 robust 측정 시 별도 지정.")
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--split-id", type=str, default="supervised_eval_v1")
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--netgain-preset", type=str, default="calibrated_protocol_a_v1",
                        choices=list(WEIGHT_PRESETS.keys()))
    args = parser.parse_args()

    netgain_weights, netgain_weight_status = WEIGHT_PRESETS[args.netgain_preset]

    # Load training data.
    with open(args.training_data, encoding="utf-8") as f:
        td = json.load(f)
    tuples = td["tuples"]

    # Split.
    train_tuples, eval_tuples = split_train_eval_by_trial(tuples, train_ratio=args.train_ratio, seed=args.seed)
    print(f"[INFO] train: {len(train_tuples)} tuples, eval: {len(eval_tuples)} tuples")

    # Train.
    model_random_state = args.model_random_state if args.model_random_state is not None else args.seed
    selector = SupervisedSelector(model_type=args.model_type, random_state=model_random_state)
    X_train, tool_train, strength_train = tuples_to_arrays(train_tuples)
    X_eval, tool_eval, strength_eval = tuples_to_arrays(eval_tuples)
    train_metrics = selector.train(X_train, tool_train, strength_train,
                                   feature_names=td.get("feature_names"))
    eval_metrics = selector.evaluate(X_eval, tool_eval, strength_eval)
    print(f"[INFO] train metrics: {train_metrics}")
    print(f"[INFO] eval metrics: {eval_metrics}")

    # Measure NetGain on eval set.
    raw_dir = Path(args.raw_output_dir).resolve() if args.raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    evaluator_config_hashes = {ev.name: ev.evaluator_class_hash() for ev in DEFAULT_EVALUATORS}
    evaluator_severity_versions = {ev.name: _get_severity_version(ev) for ev in DEFAULT_EVALUATORS}
    tool_class_hashes = {name: tool.tool_class_hash() for name, tool in TOOL_BY_NAME.items()}

    # eval_tuples 의 trial_id → group selections by trial.
    selections_by_trial: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    target_eval_by_artifact = {
        "foot_floating": "FootFloatingEvaluator",
        "bone_stretch_right_arm": "BoneLengthEvaluator",
        "global_jitter": "VelocityJitterEvaluator",
    }
    for et in eval_tuples:
        trial_id = et["trial_id"]
        artifact_kind = et["artifact_kind"]
        # Reload clean motion via training data has sample_path? No, training data 안에 없음.
        # → oracle raw record 의 sample_path 를 통해 clean 재load.
        # 본 정보는 training tuple 에 있어야 함. build_oracle_training_data.py 가 sample_path 안 박제 → fallback.
        # Use trial_id 로 sample_path 추정.
        sample_path = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints" / f"{trial_id}.npy"
        if not sample_path.exists():
            print(f"[WARN] sample not found: {sample_path}")
            continue
        clean = np.load(str(sample_path)).astype(np.float64)
        result = _measure_one_trial(
            selector=selector,
            clean=clean,
            artifact_kind=artifact_kind,
            seed=args.seed,
            netgain_weights=netgain_weights,
            target_evaluator_name=target_eval_by_artifact[artifact_kind],
        )
        selections_by_trial[trial_id][artifact_kind] = result

        # Write raw record.
        if raw_dir is not None:
            timestamp = _now_iso()
            motion = np.load(str(sample_path))
            record = {
                "schema_version": SCHEMA_VERSION,
                "record_type": RECORD_TYPE,
                "timestamp": timestamp,
                "task_id": args.task_id,
                "split_id": args.split_id,
                "trial_id": trial_id,
                "baseline_type": "b6_supervised",
                "sample_path": str(sample_path),
                "generator_id": "humanml3d_gt",
                "evaluator_config_hashes": evaluator_config_hashes,
                "evaluator_severity_versions": evaluator_severity_versions,
                "tool_registry_config_hashes": tool_class_hashes,
                "skeleton_normalizer_model_card_hash": None,
                "motion_shape": list(motion.shape),
                "fps": 20,
                "seed": int(args.seed),
                "netgain_weight_status": netgain_weight_status,
                "netgain_weights": dict(netgain_weights),
                "selector_model_type": args.model_type,
                "train_eval_split_seed": int(args.seed),
                "model_random_state": int(model_random_state),
                "train_ratio": float(args.train_ratio),
                "selections": {artifact_kind: result},
                "negative_result": False,
            }
            out_path = raw_dir / f"{timestamp}_{args.task_id}_{trial_id}_{artifact_kind}.json"
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    # Aggregate.
    by_artifact = defaultdict(list)
    for trial_id, sel_dict in selections_by_trial.items():
        for kind, sel in sel_dict.items():
            by_artifact[kind].append(sel)

    artifact_summary = []
    for artifact_kind, sels in sorted(by_artifact.items()):
        netgains = np.array([s["netgain_provisional"] for s in sels], dtype=np.float64)
        td_ = np.array([s["target_delta"] for s in sels], dtype=np.float64)
        fl = np.array([s["fidelity_loss_protocol_a"] for s in sels], dtype=np.float64)
        cm = np.array([s["correction_magnitude"] for s in sels], dtype=np.float64)
        tool_dist: dict[str, int] = defaultdict(int)
        strength_dist: dict[str, int] = defaultdict(int)
        for s in sels:
            tool_dist[s["decision_trace"]["predicted_tool"]] += 1
            strength_dist[s["decision_trace"]["predicted_strength"]] += 1
        artifact_summary.append({
            "artifact_kind": artifact_kind,
            "target_evaluator": sels[0]["target_evaluator"],
            "n_samples": len(sels),
            "tool_distribution": dict(tool_dist),
            "strength_distribution": dict(strength_dist),
            "netgain_provisional": {
                "mean": float(netgains.mean()),
                "median": float(np.median(netgains)),
                "p25": float(np.percentile(netgains, 25)),
                "p75": float(np.percentile(netgains, 75)),
                "min": float(netgains.min()),
                "max": float(netgains.max()),
            },
            "target_delta": {"mean": float(td_.mean()), "median": float(np.median(td_))},
            "fidelity_loss_protocol_a": {"mean": float(fl.mean()), "median": float(np.median(fl))},
            "correction_magnitude": {"mean": float(cm.mean()), "median": float(np.median(cm))},
        })

    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": SUMMARY_TYPE,
        "task_id": args.task_id,
        "split_id": args.split_id,
        "baseline_type": "b6_supervised",
        "orchestrator": "SupervisedSelector",
        "model_type": args.model_type,
        "train_ratio": float(args.train_ratio),
        "seed": int(args.seed),
        "n_train_tuples": len(train_tuples),
        "n_eval_tuples": len(eval_tuples),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "n_samples_evaluated": len(selections_by_trial),
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,
        "tool_class_hashes": tool_class_hashes,
        "raw_records_dir": str(raw_dir) if raw_dir else None,
        "trial_ids": sorted(selections_by_trial),
        "training_data_source": str(args.training_data),
        "netgain_weight_status": netgain_weight_status,
        "netgain_weights": dict(netgain_weights),
        "per_artifact": artifact_summary,
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"\n[OK] wrote summary to {args.output}")
        for ar in summary["per_artifact"]:
            print(
                f"  {ar['artifact_kind']:30s} | n={ar['n_samples']} | "
                f"NetGain median={ar['netgain_provisional']['median']:+.5f} | "
                f"tools={ar['tool_distribution']} | strengths={ar['strength_distribution']}"
            )
    else:
        print(text)


if __name__ == "__main__":
    main()
