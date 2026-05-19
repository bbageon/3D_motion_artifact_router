"""Sequence oracle measurement CLI — HumanML3D × 3 artifact × max_depth=3 DFS.

본 도구는 [`select_best_sequence_oracle`](../orchestrator/oracle_sequence.py) 을
HumanML3D GT motion 위에서 본격 실행한다. closed-loop refinement 의 fair
upper bound (공정한 상한선) 측정.

용어 풀이:
  - **DFS** (Depth-First Search, 깊이 우선 탐색): 트리 한 가지를 끝까지 따라간 후
    되돌아오는 탐색 방식.
  - **max_depth**: 한 sample 에 tool 을 최대 몇 번까지 순차 적용할지의 한계.
  - **Score 비감소 가지치기**: 매 step 의 TotalArtifactScore 가 직전보다 증가하면
    본 분기 (sequence) 를 잘라낸다 (AGENTS.md §3-4).
  - **calibrated_protocol_a_v1**: Priority 1 단계의 grid search 로 결정된 NetGain
    weight (α=5.0, β=0.0, γ=0.0).

CLI 예:
    python -m tools.oracle_sequence_run --n-samples 30 --seed 42 \\
        --task-id oracle_sequence_v1 --split-id calibration_v1 \\
        --max-depth 3 \\
        --raw-output-dir evals/raw \\
        --output evals/snapshots/oracle_sequence_v1.json
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

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS
from orchestrator import SequenceOracleSelection, select_best_sequence_oracle
from orchestrator.oracle_sequence import ORACLE_TYPE
from orchestrator.oracle_single_step import (
    CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1,
    DEFAULT_PROVISIONAL_NETGAIN_WEIGHTS,
)
from tools.synthetic_injection import inject_bone_stretch, inject_foot_floating, inject_jitter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "oracle_sequence_sample"
SUMMARY_TYPE = "oracle_sequence_summary"

WEIGHT_PRESETS: dict[str, tuple[dict[str, float], str]] = {
    "provisional": (DEFAULT_PROVISIONAL_NETGAIN_WEIGHTS, "provisional"),
    "calibrated_protocol_a_v1": (CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1, "calibrated_protocol_a_v1"),
}

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
    fn_name = spec["inject_fn"]
    kwargs = dict(spec["inject_kwargs"])
    kwargs["seed"] = seed
    if fn_name == "inject_foot_floating":
        return inject_foot_floating(clean, **kwargs)
    if fn_name == "inject_bone_stretch":
        T = clean.shape[0]
        half = max(1, T // 2)
        return np.concatenate(
            [inject_bone_stretch(clean[:half], **kwargs), clean[half:]], axis=0
        )
    if fn_name == "inject_jitter":
        return inject_jitter(clean, **kwargs)
    raise ValueError(f"unknown inject_fn {fn_name!r}")


def _measure_one_sample(
    *,
    sample_path: Path,
    seed: int,
    tools_with_target_parts: list[tuple[CorrectionTool, str]],
    strengths: tuple[str, ...],
    netgain_weights: dict[str, float],
    netgain_weight_status: str,
    max_depth: int,
    top_k: int,
) -> dict[str, SequenceOracleSelection]:
    clean = np.load(str(sample_path)).astype(np.float64)
    if clean.ndim != 3 or clean.shape[1] != 22 or clean.shape[2] != 3:
        raise ValueError(f"sample {sample_path.name} shape {clean.shape} not [T, 22, 3]")

    selections: dict[str, SequenceOracleSelection] = {}
    for spec in ARTIFACT_SPECS:
        corrupted = _apply_injection(spec, clean, seed)
        sel = select_best_sequence_oracle(
            clean_motion=clean,
            corrupted_motion=corrupted,
            artifact_kind=spec["kind"],
            target_evaluator_name=spec["target_evaluator"],
            tools_with_target_parts=tools_with_target_parts,
            evaluators=list(DEFAULT_EVALUATORS),
            strengths=strengths,
            netgain_weights=netgain_weights,
            netgain_weight_status=netgain_weight_status,
            max_depth=max_depth,
            top_k=top_k,
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
    selections: dict[str, SequenceOracleSelection],
    evaluator_config_hashes: dict[str, str],
    evaluator_severity_versions: dict[str, str],
    tool_class_hashes: dict[str, str],
    netgain_weight_status: str,
    netgain_weights: dict[str, float],
    max_depth: int,
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
        "oracle_type": ORACLE_TYPE,
        "netgain_weight_status": netgain_weight_status,
        "netgain_weights": dict(netgain_weights),
        "max_depth": max_depth,
        "selections": {kind: sel.to_dict() for kind, sel in selections.items()},
        "negative_result": False,
    }


def _aggregate_selections(
    selections_by_sample: dict[str, dict[str, SequenceOracleSelection]],
) -> dict[str, Any]:
    by_artifact: dict[str, list[SequenceOracleSelection]] = defaultdict(list)
    for sel_dict in selections_by_sample.values():
        for kind, sel in sel_dict.items():
            by_artifact[kind].append(sel)

    artifact_summary: list[dict[str, Any]] = []
    for artifact_kind, sels in sorted(by_artifact.items()):
        best_sequence_length_freq: dict[int, int] = defaultdict(int)
        best_first_tool_freq: dict[str, int] = defaultdict(int)
        best_netgains: list[float] = []
        best_target_deltas: list[float] = []
        best_fidelity_losses: list[float] = []
        best_lengths: list[int] = []
        n_candidates_explored: list[int] = []
        n_candidates_pruned: list[int] = []

        for sel in sels:
            bc = sel.best_candidate
            n_candidates_explored.append(sel.n_candidates_explored)
            n_candidates_pruned.append(sel.n_candidates_pruned)
            if bc is None:
                continue
            best_lengths.append(bc.length)
            best_sequence_length_freq[bc.length] += 1
            if bc.length >= 1:
                best_first_tool_freq[bc.sequence[0][0]] += 1
            best_netgains.append(bc.netgain_provisional)
            best_target_deltas.append(bc.target_delta_total)
            best_fidelity_losses.append(bc.fidelity_loss_protocol_a)

        if best_netgains:
            ng_arr = np.array(best_netgains, dtype=np.float64)
            td_arr = np.array(best_target_deltas, dtype=np.float64)
            fl_arr = np.array(best_fidelity_losses, dtype=np.float64)
            length_arr = np.array(best_lengths, dtype=np.float64)
            artifact_summary.append({
                "artifact_kind": artifact_kind,
                "target_evaluator": sels[0].target_evaluator,
                "n_samples": len(sels),
                "best_sequence_length_freq": dict(best_sequence_length_freq),
                "best_first_tool_freq": dict(best_first_tool_freq),
                "best_length": {
                    "mean": float(length_arr.mean()),
                    "median": float(np.median(length_arr)),
                },
                "best_netgain_provisional": {
                    "mean": float(ng_arr.mean()),
                    "median": float(np.median(ng_arr)),
                    "p25": float(np.percentile(ng_arr, 25)),
                    "p75": float(np.percentile(ng_arr, 75)),
                    "min": float(ng_arr.min()),
                    "max": float(ng_arr.max()),
                },
                "best_target_delta": {
                    "mean": float(td_arr.mean()),
                    "median": float(np.median(td_arr)),
                },
                "best_fidelity_loss_protocol_a": {
                    "mean": float(fl_arr.mean()),
                    "median": float(np.median(fl_arr)),
                },
                "n_candidates_explored": {
                    "mean": float(np.mean(n_candidates_explored)),
                    "median": float(np.median(n_candidates_explored)),
                    "max": int(max(n_candidates_explored)),
                },
                "n_candidates_pruned": {
                    "mean": float(np.mean(n_candidates_pruned)),
                    "median": float(np.median(n_candidates_pruned)),
                },
            })
    return {"per_artifact": artifact_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequence oracle measurement (closed-loop fair upper bound)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-id", type=str, default="oracle_sequence_v1")
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strengths", type=str, nargs="+",
                        default=["small", "medium", "large"])
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--netgain-preset", type=str, default="calibrated_protocol_a_v1",
                        choices=list(WEIGHT_PRESETS.keys()))
    args = parser.parse_args()

    netgain_weights, netgain_weight_status = WEIGHT_PRESETS[args.netgain_preset]
    split_id = args.split_id if args.split_id is not None else args.task_id

    rng = np.random.default_rng(args.seed)
    npy_files = sorted(args.data_dir.glob("*.npy"))
    n = min(args.n_samples, len(npy_files))
    chosen_idx = rng.choice(len(npy_files), size=n, replace=False)
    chosen = [npy_files[i] for i in chosen_idx]

    tools_with_target_parts: list[tuple[CorrectionTool, str]] = [
        (FootLockTool(default_ground_y=0.0), "both_feet"),
        (BoneProjectionTool(), "right_arm"),
        (VelocitySmoothingTool(), "full_body"),
    ]

    evaluator_config_hashes = {ev.name: ev.evaluator_class_hash() for ev in DEFAULT_EVALUATORS}
    evaluator_severity_versions = {ev.name: _get_severity_version(ev) for ev in DEFAULT_EVALUATORS}
    tool_class_hashes = {type(t).__name__: t.tool_class_hash() for t, _ in tools_with_target_parts}

    raw_dir = Path(args.raw_output_dir).resolve() if args.raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    selections_by_sample: dict[str, dict[str, SequenceOracleSelection]] = {}
    for i, path in enumerate(chosen, 1):
        trial_id = path.stem
        try:
            selections = _measure_one_sample(
                sample_path=path,
                seed=args.seed,
                tools_with_target_parts=tools_with_target_parts,
                strengths=tuple(args.strengths),
                netgain_weights=netgain_weights,
                netgain_weight_status=netgain_weight_status,
                max_depth=args.max_depth,
                top_k=args.top_k,
            )
        except ValueError as e:
            print(f"[WARN] skipping {trial_id}: {e}", file=sys.stderr)
            continue
        selections_by_sample[trial_id] = selections
        print(f"[OK] {i}/{n} done: {trial_id}", file=sys.stderr)

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
                netgain_weight_status=netgain_weight_status,
                netgain_weights=netgain_weights,
                max_depth=args.max_depth,
            )
            out_path = raw_dir / f"{timestamp}_{args.task_id}_{trial_id}.json"
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

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
        "max_depth": args.max_depth,
        "top_k": args.top_k,
        "raw_records_dir": str(raw_dir) if raw_dir else None,
        "trial_ids": sorted(selections_by_sample),
        "oracle_type": ORACLE_TYPE,
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
    else:
        print(text)


if __name__ == "__main__":
    main()
