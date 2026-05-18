"""Tool effect matrix CLI — HumanML3D GT × 3 inject × 3 tool × 3 strength 측정.

본 도구는 [`orchestrator/rule_based.py`](../orchestrator/rule_based.py) 의
`compute_tool_effect_matrix` 를 HumanML3D GT motion 위에서 본격 실행한다. 산출
물은 (a) sample-level raw record 와 (b) (artifact_kind × tool × strength) 단위
의 aggregate snapshot 두 종으로 박제된다.

본 결과의 의미:

  1. [H-2026-204](../evals/hypotheses/H-2026-204.md) (RQ1+RQ2) 의 Stage 1
     (Week 3) deliverable 의 핵심 — "어떤 tool 이 어떤 artifact 를 줄이는가" 의
     실측.
  2. AGENTS.md §6-12 cross-evaluator side effects 기록 의무 — target evaluator
     의 score 변화뿐 아니라 다른 evaluator 의 score 변동도 모든 entry 에 박제.
  3. 후속 oracle best-tool baseline (single-step, AGENTS.md §3-16) 의 후보 tool
     선택 logic 의 입력.

설계 결정 — tool 별 자연 target_part:
  각 tool 의 target_part 인터페이스가 달라 (FootLockTool→`both_feet`,
  BoneProjectionTool→chain 이름, VelocitySmoothingTool→`full_body`) 단일
  artifact 당 단일 target_part 만 지원하는 low-level 함수에 그대로 못 넘긴다.
  본 CLI 는 각 tool 의 자연 target_part 로 별도 호출해 한 sample 의 27 entry
  (3 artifact × 3 tool × 3 strength) 를 모은다.

CLI 예:
    python -m tools.tool_effect_matrix --n-samples 30 --seed 42 \\
        --task-id tool_effect_matrix_v1 \\
        --split-id calibration_v1 \\
        --raw-output-dir evals/raw \\
        --output evals/snapshots/tool_effect_matrix_v1.json
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
from orchestrator import compute_tool_effect_matrix
from orchestrator.rule_based import ToolEffectEntry
from tools.synthetic_injection import (
    inject_bone_stretch,
    inject_foot_floating,
    inject_jitter,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "tool_effect_matrix_sample"
SUMMARY_TYPE = "tool_effect_matrix_summary"

#: 측정할 artifact 종류 + 짝지어진 target evaluator + corrupted motion 생성 함수.
ARTIFACT_SPECS: list[dict[str, Any]] = [
    {
        "kind": "foot_floating",
        "target_evaluator": "FootFloatingEvaluator",
        "inject_kwargs": {"lift_height": 0.08},  # tau_float 0.05 < 0.08 < tau_contact 0.10
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

#: tool 별 자연 target_part — tool 의 인터페이스에 가장 자연스러운 입력값.
TOOL_NATURAL_TARGET_PART: dict[str, str] = {
    "FootLockTool": "both_feet",
    "BoneProjectionTool": "right_arm",  # bone_stretch_right_arm 와 매칭
    "VelocitySmoothingTool": "full_body",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _get_severity_version(evaluator: Any) -> str:
    """evaluator 모듈의 SEVERITY_VERSION 상수 추출. baseline_smoke 와 동일."""
    mod = sys.modules.get(type(evaluator).__module__)
    if mod is None:
        return "unversioned"
    return getattr(mod, "SEVERITY_VERSION", "unversioned")


def _apply_injection(spec: dict[str, Any], clean: np.ndarray, seed: int) -> np.ndarray:
    """spec 에 따라 corrupted motion 생성.

    `inject_bone_stretch` 는 **frame 절반만** 적용한다 (전 frame uniform stretch
    는 BoneLengthEvaluator 의 median reference 가 같이 이동해 detection 0
    이라는 Week 2 noted-issue 회피).
    """
    fn_name = spec["inject_fn"]
    kwargs = dict(spec["inject_kwargs"])
    kwargs["seed"] = seed
    if fn_name == "inject_foot_floating":
        return inject_foot_floating(clean, **kwargs)
    if fn_name == "inject_bone_stretch":
        # partial-frame: 첫 절반에 stretch, 나머지는 원본 — frame-wise variation 유발.
        T = clean.shape[0]
        half = max(1, T // 2)
        stretched_half = inject_bone_stretch(clean[:half], **kwargs)
        return np.concatenate([stretched_half, clean[half:]], axis=0)
    if fn_name == "inject_jitter":
        return inject_jitter(clean, **kwargs)
    raise ValueError(f"unknown inject_fn {fn_name!r}")


def _build_target_evaluator_map() -> dict[str, str]:
    return {spec["kind"]: spec["target_evaluator"] for spec in ARTIFACT_SPECS}


def _measure_one_sample(
    sample_path: Path,
    seed: int,
    tools: list[CorrectionTool],
    strengths: tuple[str, ...],
) -> tuple[list[ToolEffectEntry], dict[str, Any]]:
    """한 sample 의 모든 (artifact × tool × strength) entry 산출.

    Returns:
        (entries list, motion_metadata dict).
    """
    clean = np.load(str(sample_path)).astype(np.float64)
    if clean.ndim != 3 or clean.shape[1] != 22 or clean.shape[2] != 3:
        raise ValueError(
            f"sample {sample_path.name} shape {clean.shape} not [T, 22, 3]"
        )

    artifact_pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for spec in ARTIFACT_SPECS:
        corrupted = _apply_injection(spec, clean, seed)
        artifact_pairs.append((spec["kind"], clean, corrupted))

    target_eval_map = _build_target_evaluator_map()

    all_entries: list[ToolEffectEntry] = []
    for tool in tools:
        tool_class = type(tool).__name__
        natural_target_part = TOOL_NATURAL_TARGET_PART.get(tool_class, "full_body")
        target_part_by_artifact = {
            spec["kind"]: natural_target_part for spec in ARTIFACT_SPECS
        }
        entries = compute_tool_effect_matrix(
            artifact_pairs=artifact_pairs,
            tools=[tool],
            evaluators=list(DEFAULT_EVALUATORS),
            target_evaluator_by_artifact=target_eval_map,
            target_part_by_artifact=target_part_by_artifact,
            strengths=strengths,
        )
        all_entries.extend(entries)

    motion_metadata = {
        "sample_path": str(sample_path),
        "motion_shape": [int(clean.shape[0]), int(clean.shape[1]), int(clean.shape[2])],
        "fps": 20,
    }
    return all_entries, motion_metadata


def _make_raw_record(
    *,
    timestamp: str,
    task_id: str,
    split_id: str,
    trial_id: str,
    seed: int,
    motion_metadata: dict[str, Any],
    entries: list[ToolEffectEntry],
    evaluator_config_hashes: dict[str, str],
    evaluator_severity_versions: dict[str, str],
    tool_class_hashes: dict[str, str],
) -> dict[str, Any]:
    """Tool effect matrix 의 sample-level raw record.

    AGENTS.md §3-6 (평가 기록 의무) + §3-15 (severity_versions + split_id +
    evaluator_config_hashes) + §6-12 (cross_evaluator_delta — 각 entry 에 박제).
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "timestamp": timestamp,
        "task_id": task_id,
        "split_id": split_id,
        "trial_id": trial_id,
        "sample_path": motion_metadata["sample_path"],
        "generator_id": "humanml3d_gt",
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,
        "tool_registry_config_hashes": tool_class_hashes,
        "skeleton_normalizer_model_card_hash": None,
        "motion_shape": motion_metadata["motion_shape"],
        "fps": motion_metadata["fps"],
        "seed": int(seed),
        "tool_effect_entries": [e.to_dict() for e in entries],
        "metrics_not_applicable": {
            "NetGain": "tool effect matrix is single-step apply; NetGain weights "
                       "(α/β/γ) are provisional — see AGENTS.md §6-11",
            "tool_call_trace": "single-step apply per cell; no sequential trace",
            "netgain_weight_status": "provisional",  # AGENTS.md §6-11
        },
        "negative_result": False,
    }


def _aggregate_entries(
    entries_by_sample: dict[str, list[ToolEffectEntry]],
) -> dict[str, Any]:
    """(artifact_kind × tool × strength) cell 별로 sample 분포 통계.

    각 cell 마다:
      - target_delta: mean/median/p95/min/max.
      - correction_magnitude: mean/median.
      - cross_evaluator_delta: per-evaluator mean.
      - skip_count: cell 안 sample 중 skip 된 (ValueError 등) 개수.
      - n_samples: cell 의 total sample 수.
    """
    buckets: dict[tuple[str, str, str], list[ToolEffectEntry]] = defaultdict(list)
    for sample_id, entries in entries_by_sample.items():
        for e in entries:
            key = (e.artifact_kind, e.tool_name, e.strength)
            buckets[key].append(e)

    cells: list[dict[str, Any]] = []
    for (artifact_kind, tool_name, strength), bucket in sorted(buckets.items()):
        valid = [e for e in bucket if not e.metadata.get("skipped", False)]
        if not valid:
            cells.append({
                "artifact_kind": artifact_kind,
                "tool_name": tool_name,
                "strength": strength,
                "n_samples": len(bucket),
                "skip_count": len(bucket),
                "skipped_reason_sample": (
                    bucket[0].metadata.get("reason") if bucket else None
                ),
                "target_delta": None,
                "correction_magnitude": None,
                "cross_evaluator_delta_mean": None,
            })
            continue

        target_deltas = np.array([e.target_delta for e in valid], dtype=np.float64)
        corr_mags = np.array([e.correction_magnitude for e in valid], dtype=np.float64)

        cross_delta_acc: dict[str, list[float]] = defaultdict(list)
        for e in valid:
            for ev_name, delta in e.cross_evaluator_delta.items():
                cross_delta_acc[ev_name].append(delta)
        cross_delta_mean = {
            ev: float(np.mean(vals)) for ev, vals in cross_delta_acc.items()
        }

        cells.append({
            "artifact_kind": artifact_kind,
            "tool_name": tool_name,
            "strength": strength,
            "target_evaluator": valid[0].target_evaluator,
            "n_samples": len(bucket),
            "skip_count": len(bucket) - len(valid),
            "target_delta": {
                "mean": float(target_deltas.mean()),
                "median": float(np.median(target_deltas)),
                "p5": float(np.percentile(target_deltas, 5)),
                "p95": float(np.percentile(target_deltas, 95)),
                "min": float(target_deltas.min()),
                "max": float(target_deltas.max()),
            },
            "correction_magnitude": {
                "mean": float(corr_mags.mean()),
                "median": float(np.median(corr_mags)),
                "max": float(corr_mags.max()),
            },
            "cross_evaluator_delta_mean": cross_delta_mean,
        })
    return {"cells": cells, "n_cells": len(cells)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HumanML3D GT × 3 inject × 3 tool × 3 strength tool effect matrix"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-id", type=str, default="tool_effect_matrix_v1")
    parser.add_argument("--split-id", type=str, default=None,
                        help="None 이면 task_id 와 동일.")
    parser.add_argument("--raw-output-dir", type=Path, default=None,
                        help="sample-level raw record 디렉토리. None 이면 저장 안 함.")
    parser.add_argument("--output", type=Path, default=None,
                        help="aggregated snapshot JSON. None 이면 stdout.")
    parser.add_argument("--strengths", type=str, nargs="+",
                        default=["small", "medium", "large"],
                        help="시험할 strength list.")
    args = parser.parse_args()

    split_id = args.split_id if args.split_id is not None else args.task_id

    rng = np.random.default_rng(args.seed)
    npy_files = sorted(args.data_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"no .npy files in {args.data_dir}")
    n = min(args.n_samples, len(npy_files))
    chosen_idx = rng.choice(len(npy_files), size=n, replace=False)
    chosen = [npy_files[i] for i in chosen_idx]

    tools: list[CorrectionTool] = [
        FootLockTool(default_ground_y=0.0),
        BoneProjectionTool(),
        VelocitySmoothingTool(),
    ]

    evaluator_config_hashes = {
        ev.name: ev.evaluator_class_hash() for ev in DEFAULT_EVALUATORS
    }
    evaluator_severity_versions = {
        ev.name: _get_severity_version(ev) for ev in DEFAULT_EVALUATORS
    }
    tool_class_hashes = {type(t).__name__: t.tool_class_hash() for t in tools}

    raw_dir = Path(args.raw_output_dir).resolve() if args.raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    entries_by_sample: dict[str, list[ToolEffectEntry]] = {}
    for path in chosen:
        trial_id = path.stem
        try:
            entries, motion_meta = _measure_one_sample(
                path, seed=args.seed, tools=tools, strengths=tuple(args.strengths)
            )
        except ValueError as e:
            print(f"[WARN] skipping {trial_id}: {e}", file=sys.stderr)
            continue
        entries_by_sample[trial_id] = entries

        if raw_dir is not None:
            timestamp = _now_iso()
            record = _make_raw_record(
                timestamp=timestamp,
                task_id=args.task_id,
                split_id=split_id,
                trial_id=trial_id,
                seed=args.seed,
                motion_metadata=motion_meta,
                entries=entries,
                evaluator_config_hashes=evaluator_config_hashes,
                evaluator_severity_versions=evaluator_severity_versions,
                tool_class_hashes=tool_class_hashes,
            )
            out_path = raw_dir / f"{timestamp}_{args.task_id}_{trial_id}.json"
            out_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    aggregate = _aggregate_entries(entries_by_sample)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": SUMMARY_TYPE,
        "task_id": args.task_id,
        "split_id": split_id,
        "seed": int(args.seed),
        "n_samples_evaluated": len(entries_by_sample),
        "data_dir": str(args.data_dir),
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,
        "tool_class_hashes": tool_class_hashes,
        "strengths": list(args.strengths),
        "raw_records_dir": str(raw_dir) if raw_dir else None,
        "trial_ids": sorted(entries_by_sample),
        "netgain_weight_status": "provisional",  # AGENTS.md §6-11
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
