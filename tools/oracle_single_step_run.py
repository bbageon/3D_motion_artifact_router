"""Single-step oracle measurement CLI — HumanML3D × 3 artifact × oracle 선택.

본 도구는 [`select_best_tool_single_step`](../orchestrator/oracle_single_step.py) 을
HumanML3D GT motion 위에서 본격 실행한다. 각 sample × artifact_kind 마다 후보 tool
× strength 의 모든 조합을 평가해 best NetGain (provisional) 을 식별.

산출물:
  (a) sample-level raw record (per-artifact OracleSelection 포함).
  (b) 통합 snapshot — best tool 선택 분포, NetGain 분포, cross-evaluator delta 평균.

AGENTS.md 의무:
  - §3-15 raw record metadata: severity_versions + split_id +
    evaluator_config_hashes + tool_registry_config_hashes 모두 박제.
  - §3-16 oracle type 명시: 각 OracleSelection 의 `oracle_type="single_step"`.
  - §6-11 provisional NetGain: 모든 raw record + snapshot 의
    `netgain_weight_status="provisional"`.
  - §6-12 cross-evaluator side effects: 모든 candidate 의 cross_evaluator_delta
    박제.

명세 §9.1 baseline B8 (Oracle best-tool, upper bound) 의 측정 기반.

CLI 예:
    python -m tools.oracle_single_step_run --n-samples 30 --seed 42 \\
        --task-id oracle_single_step_v1 \\
        --split-id calibration_v1 \\
        --raw-output-dir evals/raw \\
        --output evals/snapshots/oracle_single_step_v1.json
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
    CorrectionTool,
    FootLockTool,
    VelocitySmoothingTool,
)
from evaluators import DEFAULT_EVALUATORS
from orchestrator import (
    OracleSelection,
    select_best_tool_single_step,
)
from orchestrator.oracle_single_step import (
    DEFAULT_PROVISIONAL_NETGAIN_WEIGHTS,
    ORACLE_TYPE,
)
from tools.synthetic_injection import (
    inject_bone_stretch,
    inject_foot_floating,
    inject_jitter,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "oracle_single_step_sample"
SUMMARY_TYPE = "oracle_single_step_summary"

#: 측정할 artifact 종류 + 짝지어진 target evaluator + 생성 함수.
#: tool_effect_matrix 와 동일 — corrupted motion 의 일관된 정의.
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _get_severity_version(evaluator: Any) -> str:
    mod = sys.modules.get(type(evaluator).__module__)
    if mod is None:
        return "unversioned"
    return getattr(mod, "SEVERITY_VERSION", "unversioned")


def _apply_injection(spec: dict[str, Any], clean: np.ndarray, seed: int) -> np.ndarray:
    """tool_effect_matrix 와 동일 — bone stretch 는 partial-frame."""
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


def _measure_one_sample(
    *,
    sample_path: Path,
    seed: int,
    tools_with_target_parts: list[tuple[CorrectionTool, str]],
    strengths: tuple[str, ...],
) -> dict[str, OracleSelection]:
    """한 sample 의 각 artifact_kind 별 OracleSelection 산출."""
    clean = np.load(str(sample_path)).astype(np.float64)
    if clean.ndim != 3 or clean.shape[1] != 22 or clean.shape[2] != 3:
        raise ValueError(f"sample {sample_path.name} shape {clean.shape} not [T, 22, 3]")

    selections: dict[str, OracleSelection] = {}
    for spec in ARTIFACT_SPECS:
        corrupted = _apply_injection(spec, clean, seed)
        sel = select_best_tool_single_step(
            clean_motion=clean,
            corrupted_motion=corrupted,
            artifact_kind=spec["kind"],
            target_evaluator_name=spec["target_evaluator"],
            tools_with_target_parts=tools_with_target_parts,
            evaluators=list(DEFAULT_EVALUATORS),
            strengths=strengths,
        )
        selections[spec["kind"]] = sel
    return selections


def _make_raw_record(
    *,
    timestamp: str,
    task_id: str,
    split_id: str,
    trial_id: str,
    sample_path: Path,
    seed: int,
    motion_shape: tuple[int, int, int],
    selections: dict[str, OracleSelection],
    evaluator_config_hashes: dict[str, str],
    evaluator_severity_versions: dict[str, str],
    tool_class_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "timestamp": timestamp,
        "task_id": task_id,
        "split_id": split_id,
        "trial_id": trial_id,
        "sample_path": str(sample_path),
        "generator_id": "humanml3d_gt",
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,
        "tool_registry_config_hashes": tool_class_hashes,
        "skeleton_normalizer_model_card_hash": None,
        "motion_shape": list(motion_shape),
        "fps": 20,
        "seed": int(seed),
        "oracle_type": ORACLE_TYPE,  # AGENTS.md §3-16
        "netgain_weight_status": "provisional",  # AGENTS.md §6-11
        "netgain_weights": dict(DEFAULT_PROVISIONAL_NETGAIN_WEIGHTS),
        "selections": {kind: sel.to_dict() for kind, sel in selections.items()},
        "negative_result": False,
    }


def _aggregate_selections(
    selections_by_sample: dict[str, dict[str, OracleSelection]],
) -> dict[str, Any]:
    """artifact_kind 별 best 분포 + best NetGain·target_delta 통계."""
    by_artifact: dict[str, list[OracleSelection]] = defaultdict(list)
    for _trial, sel_dict in selections_by_sample.items():
        for kind, sel in sel_dict.items():
            by_artifact[kind].append(sel)

    artifact_summary: list[dict[str, Any]] = []
    for artifact_kind, sels in sorted(by_artifact.items()):
        best_tool_freq: dict[str, int] = defaultdict(int)
        best_strength_freq: dict[str, int] = defaultdict(int)
        best_netgains: list[float] = []
        best_target_deltas: list[float] = []
        best_fidelity_losses: list[float] = []
        best_corr_mags: list[float] = []
        n_all_skipped = 0
        for sel in sels:
            if sel.best_candidate is None:
                n_all_skipped += 1
                continue
            bc = sel.best_candidate
            best_tool_freq[bc.tool_name] += 1
            best_strength_freq[bc.strength] += 1
            best_netgains.append(bc.netgain_provisional)
            best_target_deltas.append(bc.target_delta)
            best_fidelity_losses.append(bc.fidelity_loss_protocol_a)
            best_corr_mags.append(bc.correction_magnitude)

        if best_netgains:
            arr = np.array(best_netgains, dtype=np.float64)
            td = np.array(best_target_deltas, dtype=np.float64)
            fl = np.array(best_fidelity_losses, dtype=np.float64)
            cm = np.array(best_corr_mags, dtype=np.float64)
            artifact_summary.append({
                "artifact_kind": artifact_kind,
                "target_evaluator": sels[0].target_evaluator,
                "n_samples": len(sels),
                "n_all_skipped": n_all_skipped,
                "best_tool_freq": dict(best_tool_freq),
                "best_strength_freq": dict(best_strength_freq),
                "best_netgain_provisional": {
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "p25": float(np.percentile(arr, 25)),
                    "p75": float(np.percentile(arr, 75)),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                },
                "best_target_delta": {
                    "mean": float(td.mean()),
                    "median": float(np.median(td)),
                },
                "best_fidelity_loss_protocol_a": {
                    "mean": float(fl.mean()),
                    "median": float(np.median(fl)),
                },
                "best_correction_magnitude": {
                    "mean": float(cm.mean()),
                    "median": float(np.median(cm)),
                },
            })
        else:
            artifact_summary.append({
                "artifact_kind": artifact_kind,
                "target_evaluator": sels[0].target_evaluator,
                "n_samples": len(sels),
                "n_all_skipped": n_all_skipped,
                "best_tool_freq": {},
                "best_strength_freq": {},
                "best_netgain_provisional": None,
                "best_target_delta": None,
                "best_fidelity_loss_protocol_a": None,
                "best_correction_magnitude": None,
            })
    return {"per_artifact": artifact_summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HumanML3D × 3 artifact × single-step oracle measurement"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-id", type=str, default="oracle_single_step_v1")
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strengths", type=str, nargs="+",
                        default=["small", "medium", "large"])
    args = parser.parse_args()

    split_id = args.split_id if args.split_id is not None else args.task_id

    rng = np.random.default_rng(args.seed)
    npy_files = sorted(args.data_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"no .npy files in {args.data_dir}")
    n = min(args.n_samples, len(npy_files))
    chosen_idx = rng.choice(len(npy_files), size=n, replace=False)
    chosen = [npy_files[i] for i in chosen_idx]

    tools_with_target_parts: list[tuple[CorrectionTool, str]] = [
        (FootLockTool(default_ground_y=0.0), "both_feet"),
        (BoneProjectionTool(), "right_arm"),
        (VelocitySmoothingTool(), "full_body"),
    ]

    evaluator_config_hashes = {
        ev.name: ev.evaluator_class_hash() for ev in DEFAULT_EVALUATORS
    }
    evaluator_severity_versions = {
        ev.name: _get_severity_version(ev) for ev in DEFAULT_EVALUATORS
    }
    tool_class_hashes = {
        type(t).__name__: t.tool_class_hash() for t, _ in tools_with_target_parts
    }

    raw_dir = Path(args.raw_output_dir).resolve() if args.raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    selections_by_sample: dict[str, dict[str, OracleSelection]] = {}
    for path in chosen:
        trial_id = path.stem
        try:
            selections = _measure_one_sample(
                sample_path=path,
                seed=args.seed,
                tools_with_target_parts=tools_with_target_parts,
                strengths=tuple(args.strengths),
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
                sample_path=path,
                seed=args.seed,
                motion_shape=motion.shape,
                selections=selections,
                evaluator_config_hashes=evaluator_config_hashes,
                evaluator_severity_versions=evaluator_severity_versions,
                tool_class_hashes=tool_class_hashes,
            )
            out_path = raw_dir / f"{timestamp}_{args.task_id}_{trial_id}.json"
            out_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    aggregate = _aggregate_selections(selections_by_sample)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": SUMMARY_TYPE,
        "task_id": args.task_id,
        "split_id": split_id,
        "seed": int(args.seed),
        "n_samples_evaluated": len(selections_by_sample),
        "data_dir": str(args.data_dir),
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,
        "tool_class_hashes": tool_class_hashes,
        "strengths": list(args.strengths),
        "raw_records_dir": str(raw_dir) if raw_dir else None,
        "trial_ids": sorted(selections_by_sample),
        "oracle_type": ORACLE_TYPE,  # AGENTS.md §3-16
        "netgain_weight_status": "provisional",  # AGENTS.md §6-11
        "netgain_weights": dict(DEFAULT_PROVISIONAL_NETGAIN_WEIGHTS),
        **aggregate,
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OK] wrote summary to {args.output}")
        if raw_dir:
            print(f"[OK] wrote {summary['n_samples_evaluated']} raw records to {raw_dir}")
    else:
        print(text)


if __name__ == "__main__":
    main()
