"""Fixed-pipeline baseline measurements — H-2026-204 RQ1 의 B1/B2/B3.

본 도구는 명세 §9.1 의 baseline 정의 그대로:
  - **B1**: Base generator only (no refinement) — corrected = corrupted, NetGain = 0.
  - **B2**: Base + global smoothing — VelocitySmoothingTool(full_body, medium) 1회.
  - **B3**: Base + fixed correction pipeline — FootLockTool(both_feet, medium) →
    VelocitySmoothingTool(full_body, medium) → BoneProjectionTool(right_arm, medium).

세 baseline 모두 **artifact-conditioned 선택 없음** — 어떤 artifact 가 들어와도 같은
treatment 를 적용. 본 fixed 처리가 oracle/rule-based 의 conditional 처리보다 얼마나
약한지 정량.

NetGain 계산은 [`oracle_single_step.py`](../orchestrator/oracle_single_step.py) 의
공식 그대로 (Protocol A FidelityLoss = MPJPE(corrected, clean) - MPJPE(corrupted, clean)).

AGENTS.md 의무:
  - §3-15 raw record metadata 박제 (severity_versions / split_id / hashes).
  - §6-11 netgain_weight_status (calibrated_protocol_a_v1 또는 provisional).
  - §6-12 cross-evaluator side effects 박제.

CLI 예:
    python -m tools.baseline_fixed_pipeline_run \\
        --baseline-type b2 \\
        --n-samples 30 --seed 42 \\
        --task-id baseline_fixed_smoothing_v1 \\
        --split-id calibration_v1 \\
        --netgain-preset calibrated_protocol_a_v1 \\
        --raw-output-dir evals/raw \\
        --output evals/snapshots/baseline_fixed_smoothing_v1.json
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
from tools.synthetic_injection import (
    inject_bone_stretch,
    inject_foot_floating,
    inject_jitter,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "baseline_fixed_pipeline_sample"
SUMMARY_TYPE = "baseline_fixed_pipeline_summary"

WEIGHT_PRESETS: dict[str, tuple[dict[str, float], str]] = {
    "provisional": (DEFAULT_PROVISIONAL_NETGAIN_WEIGHTS, "provisional"),
    "calibrated_protocol_a_v1": (CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1, "calibrated_protocol_a_v1"),
}

#: artifact spec — oracle_single_step_run 과 동일 (consistency 의무).
ARTIFACT_SPECS: list[dict[str, Any]] = [
    {
        "kind": "foot_floating",
        "target_evaluator": "FootFloatingEvaluator",
        "inject_kwargs": {"lift_height": 0.08},
        "inject_fn": "inject_foot_floating",
    },
    {
        "kind": "bone_stretch_right_arm",
        "target_evaluator": "BoneLengthEvaluator",
        "inject_kwargs": {"chain_label": "right_arm", "stretch_factor": 1.30},
        "inject_fn": "inject_bone_stretch",
    },
    {
        "kind": "global_jitter",
        "target_evaluator": "VelocityJitterEvaluator",
        "inject_kwargs": {"noise_std": 0.05},
        "inject_fn": "inject_jitter",
    },
]

#: baseline_type → fixed treatment 정의.
#: 각 step: (tool_factory, target_part, strength).
BASELINE_PIPELINES: dict[str, list[tuple[str, str, str]]] = {
    "b1": [],  # no refinement.
    "b2": [
        ("VelocitySmoothingTool", "full_body", "medium"),
    ],
    "b3": [
        ("FootLockTool", "both_feet", "medium"),
        ("VelocitySmoothingTool", "full_body", "medium"),
        ("BoneProjectionTool", "right_arm", "medium"),
    ],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _get_severity_version(evaluator: Any) -> str:
    mod = sys.modules.get(type(evaluator).__module__)
    if mod is None:
        return "unversioned"
    return getattr(mod, "SEVERITY_VERSION", "unversioned")


def _make_tool(tool_name: str) -> Any:
    if tool_name == "FootLockTool":
        return FootLockTool(default_ground_y=0.0)
    if tool_name == "VelocitySmoothingTool":
        return VelocitySmoothingTool()
    if tool_name == "BoneProjectionTool":
        return BoneProjectionTool()
    raise ValueError(f"unknown tool {tool_name!r}")


def _apply_injection(spec: dict[str, Any], clean: np.ndarray, seed: int) -> np.ndarray:
    fn_name = spec["inject_fn"]
    kwargs = dict(spec["inject_kwargs"])
    kwargs["seed"] = seed
    if fn_name == "inject_foot_floating":
        return inject_foot_floating(clean, **kwargs)
    if fn_name == "inject_bone_stretch":
        T = clean.shape[0]
        half = max(1, T // 2)
        stretched = inject_bone_stretch(clean[:half], **kwargs)
        return np.concatenate([stretched, clean[half:]], axis=0)
    if fn_name == "inject_jitter":
        return inject_jitter(clean, **kwargs)
    raise ValueError(f"unknown inject_fn {fn_name!r}")


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _aggregate_target_score(reports: list[EvaluatorReport]) -> float:
    if not reports:
        return 0.0
    return float(np.mean([r.score for r in reports]))


def _apply_pipeline(
    baseline_type: str,
    corrupted: np.ndarray,
) -> tuple[np.ndarray, float, int, list[dict[str, Any]]]:
    """Fixed pipeline 적용. Returns (corrected, cumulative_magnitude, n_tool_calls, trace)."""
    pipeline = BASELINE_PIPELINES[baseline_type]
    if not pipeline:
        return corrupted.copy(), 0.0, 0, []

    T = corrupted.shape[0]
    frame_range = (0, T - 1)
    corrected = corrupted.copy()
    cumulative = 0.0
    n_calls = 0
    trace: list[dict[str, Any]] = []
    for tool_name, target_part, strength in pipeline:
        tool = _make_tool(tool_name)
        try:
            corrected, report = tool.apply(
                corrected,
                target_part=target_part,
                target_joints=[],
                frame_range=frame_range,
                strength=strength,  # type: ignore[arg-type]
            )
        except ValueError as e:
            trace.append({
                "tool": tool_name,
                "target_part": target_part,
                "strength": strength,
                "skipped": True,
                "skip_reason": str(e),
                "correction_magnitude": 0.0,
            })
            continue
        cumulative += float(report.correction_magnitude)
        n_calls += 1
        trace.append({
            "tool": tool_name,
            "target_part": target_part,
            "strength": strength,
            "skipped": False,
            "correction_magnitude": float(report.correction_magnitude),
        })
    return corrected, cumulative, n_calls, trace


def _measure_one_sample(
    *,
    baseline_type: str,
    sample_path: Path,
    seed: int,
    netgain_weights: dict[str, float],
) -> dict[str, Any]:
    clean = np.load(str(sample_path)).astype(np.float64)
    if clean.ndim != 3 or clean.shape[1] != 22 or clean.shape[2] != 3:
        raise ValueError(f"sample {sample_path.name} shape {clean.shape} not [T, 22, 3]")

    evaluators = list(DEFAULT_EVALUATORS)
    alpha = float(netgain_weights["alpha"])
    beta = float(netgain_weights["beta"])
    gamma = float(netgain_weights["gamma"])

    selections: dict[str, dict[str, Any]] = {}
    for spec in ARTIFACT_SPECS:
        corrupted = _apply_injection(spec, clean, seed)
        mpjpe_corrupted = _mpjpe(corrupted, clean)

        reports_before = {ev.name: ev.evaluate(corrupted) for ev in evaluators}
        target_score_before = _aggregate_target_score(
            reports_before.get(spec["target_evaluator"], [])
        )

        corrected, cumulative_mag, n_calls, trace = _apply_pipeline(baseline_type, corrupted)

        reports_after = {ev.name: ev.evaluate(corrected) for ev in evaluators}
        target_score_after = _aggregate_target_score(
            reports_after.get(spec["target_evaluator"], [])
        )
        target_delta = target_score_after - target_score_before

        mpjpe_corrected = _mpjpe(corrected, clean)
        fidelity_loss = mpjpe_corrected - mpjpe_corrupted
        artifact_reduction = -target_delta
        tool_call_cost = float(n_calls)
        netgain = artifact_reduction - alpha * fidelity_loss - beta * cumulative_mag - gamma * tool_call_cost

        cross_delta: dict[str, float] = {}
        for ev_name in reports_before:
            if ev_name == spec["target_evaluator"]:
                continue
            b = _aggregate_target_score(reports_before[ev_name])
            a = _aggregate_target_score(reports_after.get(ev_name, []))
            cross_delta[ev_name] = float(a - b)

        selections[spec["kind"]] = {
            "artifact_kind": spec["kind"],
            "target_evaluator": spec["target_evaluator"],
            "baseline_type": baseline_type,
            "target_score_before": float(target_score_before),
            "target_score_after": float(target_score_after),
            "target_delta": float(target_delta),
            "fidelity_loss_protocol_a": float(fidelity_loss),
            "correction_magnitude": float(cumulative_mag),
            "tool_call_cost": float(tool_call_cost),
            "cross_evaluator_delta": cross_delta,
            "netgain_provisional": float(netgain),
            "tool_call_trace": trace,
        }
    return selections


def _make_raw_record(
    *,
    timestamp: str,
    task_id: str,
    split_id: str,
    trial_id: str,
    baseline_type: str,
    sample_path: Path,
    seed: int,
    motion_shape: tuple[int, ...],
    selections: dict[str, dict[str, Any]],
    evaluator_config_hashes: dict[str, str],
    evaluator_severity_versions: dict[str, str],
    tool_class_hashes: dict[str, str],
    netgain_weight_status: str,
    netgain_weights: dict[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "timestamp": timestamp,
        "task_id": task_id,
        "split_id": split_id,
        "trial_id": trial_id,
        "baseline_type": baseline_type,
        "sample_path": str(sample_path),
        "generator_id": "humanml3d_gt",
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,
        "tool_registry_config_hashes": tool_class_hashes,
        "skeleton_normalizer_model_card_hash": None,
        "motion_shape": list(motion_shape),
        "fps": 20,
        "seed": int(seed),
        "netgain_weight_status": netgain_weight_status,
        "netgain_weights": dict(netgain_weights),
        "selections": selections,
        "negative_result": False,
    }


def _aggregate(
    selections_by_sample: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    by_artifact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _trial, sel_dict in selections_by_sample.items():
        for kind, sel in sel_dict.items():
            by_artifact[kind].append(sel)

    artifact_summary: list[dict[str, Any]] = []
    for artifact_kind, sels in sorted(by_artifact.items()):
        netgains = np.array([s["netgain_provisional"] for s in sels], dtype=np.float64)
        td = np.array([s["target_delta"] for s in sels], dtype=np.float64)
        fl = np.array([s["fidelity_loss_protocol_a"] for s in sels], dtype=np.float64)
        cm = np.array([s["correction_magnitude"] for s in sels], dtype=np.float64)
        artifact_summary.append({
            "artifact_kind": artifact_kind,
            "target_evaluator": sels[0]["target_evaluator"],
            "n_samples": len(sels),
            "netgain_provisional": {
                "mean": float(netgains.mean()),
                "median": float(np.median(netgains)),
                "p25": float(np.percentile(netgains, 25)),
                "p75": float(np.percentile(netgains, 75)),
                "min": float(netgains.min()),
                "max": float(netgains.max()),
            },
            "target_delta": {
                "mean": float(td.mean()),
                "median": float(np.median(td)),
            },
            "fidelity_loss_protocol_a": {
                "mean": float(fl.mean()),
                "median": float(np.median(fl)),
            },
            "correction_magnitude": {
                "mean": float(cm.mean()),
                "median": float(np.median(cm)),
            },
        })
    return {"per_artifact": artifact_summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-pipeline baseline (B1/B2/B3) measurement"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--baseline-type", type=str, required=True, choices=sorted(BASELINE_PIPELINES))
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--netgain-preset", type=str, default="calibrated_protocol_a_v1",
                        choices=list(WEIGHT_PRESETS.keys()))
    args = parser.parse_args()

    netgain_weights, netgain_weight_status = WEIGHT_PRESETS[args.netgain_preset]
    split_id = args.split_id if args.split_id is not None else args.task_id

    rng = np.random.default_rng(args.seed)
    npy_files = sorted(args.data_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"no .npy files in {args.data_dir}")
    n = min(args.n_samples, len(npy_files))
    chosen_idx = rng.choice(len(npy_files), size=n, replace=False)
    chosen = [npy_files[i] for i in chosen_idx]

    # tool registry config hashes — 본 baseline 에서 호출 가능한 tool 들 모두.
    all_tools = [FootLockTool(default_ground_y=0.0), VelocitySmoothingTool(), BoneProjectionTool()]
    tool_class_hashes = {type(t).__name__: t.tool_class_hash() for t in all_tools}

    evaluator_config_hashes = {ev.name: ev.evaluator_class_hash() for ev in DEFAULT_EVALUATORS}
    evaluator_severity_versions = {ev.name: _get_severity_version(ev) for ev in DEFAULT_EVALUATORS}

    raw_dir = Path(args.raw_output_dir).resolve() if args.raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    selections_by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    for path in chosen:
        trial_id = path.stem
        try:
            selections = _measure_one_sample(
                baseline_type=args.baseline_type,
                sample_path=path,
                seed=args.seed,
                netgain_weights=netgain_weights,
            )
        except ValueError as e:
            print(f"[WARN] skipping {trial_id}: {e}", file=sys.stderr)
            continue
        selections_by_sample[trial_id] = selections

        if raw_dir is not None:
            timestamp = _now_iso()
            motion = np.load(str(path))
            record = _make_raw_record(
                timestamp=timestamp,
                task_id=args.task_id,
                split_id=split_id,
                trial_id=trial_id,
                baseline_type=args.baseline_type,
                sample_path=path,
                seed=args.seed,
                motion_shape=motion.shape,
                selections=selections,
                evaluator_config_hashes=evaluator_config_hashes,
                evaluator_severity_versions=evaluator_severity_versions,
                tool_class_hashes=tool_class_hashes,
                netgain_weight_status=netgain_weight_status,
                netgain_weights=netgain_weights,
            )
            out_path = raw_dir / f"{timestamp}_{args.task_id}_{trial_id}.json"
            out_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    aggregate = _aggregate(selections_by_sample)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": SUMMARY_TYPE,
        "task_id": args.task_id,
        "split_id": split_id,
        "baseline_type": args.baseline_type,
        "baseline_pipeline": BASELINE_PIPELINES[args.baseline_type],
        "seed": int(args.seed),
        "n_samples_evaluated": len(selections_by_sample),
        "data_dir": str(args.data_dir),
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,
        "tool_class_hashes": tool_class_hashes,
        "raw_records_dir": str(raw_dir) if raw_dir else None,
        "trial_ids": sorted(selections_by_sample),
        "netgain_weight_status": netgain_weight_status,
        "netgain_weights": dict(netgain_weights),
        **aggregate,
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OK] wrote summary to {args.output}")
        if raw_dir:
            print(f"[OK] wrote {summary['n_samples_evaluated']} raw records to {raw_dir}")
        for ar in summary["per_artifact"]:
            print(
                f"  {ar['artifact_kind']:30s} | "
                f"n={ar['n_samples']} | "
                f"NetGain median={ar['netgain_provisional']['median']:+.5f} | "
                f"target_delta median={ar['target_delta']['median']:+.5f}"
            )
    else:
        print(text)


if __name__ == "__main__":
    main()
