"""B5 (Rule-based orchestrator) measurement — H-2026-204 RQ1 임계 2 검증.

명세 §9.1 의 baseline B5: artifact-conditioned tool selection via rule-based
orchestrator (`orchestrator/rule_based.py`).

각 (trial, artifact) 에서:
  1. clean motion + corrupted (synthetic injection).
  2. corrupted 에 모든 evaluator 적용 → reports.
  3. RuleBasedOrchestrator.decide(reports) → OrchestratorDecision (STOP / revise / reject).
  4. STOP / reject 이면 corrected = corrupted (NetGain = 0, target_delta = 0).
  5. revise 이면 selected_tool.apply(...) → corrected. NetGain 계산.

B1/B2/B3 와 paired (같은 calibration_v1 sample + 같은 artifact injection seed) 가능.

AGENTS.md 의무:
  - §3-15 raw record metadata.
  - §6-11 netgain_weight_status.
  - §6-12 cross-evaluator side effects (orchestrator decide metadata + tool 적용 후 cross-eval).

CLI 예:
    python -m tools.baseline_rule_based_run \\
        --n-samples 30 --seed 42 \\
        --task-id baseline_b5_rule_based_v1 \\
        --split-id calibration_v1 \\
        --netgain-preset calibrated_protocol_a_v1 \\
        --raw-output-dir evals/raw \\
        --output evals/snapshots/baseline_b5_rule_based_v1.json
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
from orchestrator.rule_based import RuleBasedOrchestrator
from tools.synthetic_injection import (
    inject_bone_stretch,
    inject_foot_floating,
    inject_jitter,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "baseline_rule_based_sample"
SUMMARY_TYPE = "baseline_rule_based_summary"

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


def _measure_one_sample(
    *,
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

    tool_registry = [
        FootLockTool(default_ground_y=0.0),
        BoneProjectionTool(),
        VelocitySmoothingTool(),
    ]
    tool_by_class = {type(t).__name__: t for t in tool_registry}
    orchestrator = RuleBasedOrchestrator(tool_registry=tool_registry)

    selections: dict[str, dict[str, Any]] = {}
    for spec in ARTIFACT_SPECS:
        corrupted = _apply_injection(spec, clean, seed)
        T = corrupted.shape[0]
        frame_range = (0, T - 1)
        mpjpe_corrupted = _mpjpe(corrupted, clean)

        # Evaluate corrupted (all evaluators).
        # For decide(), 모든 evaluator 의 report 를 평탄화하여 전달.
        reports_before_dict: dict[str, list[EvaluatorReport]] = {
            ev.name: ev.evaluate(corrupted) for ev in evaluators
        }
        all_reports_before: list[EvaluatorReport] = []
        for rs in reports_before_dict.values():
            all_reports_before.extend(rs)
        target_score_before = _aggregate_target_score(
            reports_before_dict.get(spec["target_evaluator"], [])
        )

        # Orchestrator decide.
        # artifact_kind_hint 전달: synthetic injection setting 에서 target evaluator
        # 의 report 만 primary 후보로 사용. v1 의 jitter→BoneProjectionTool mis-mapping
        # bug fix (RuleBasedOrchestrator.decide 의 hint 동작 — orchestrator/rule_based.py).
        decision = orchestrator.decide(
            all_reports_before, tool_history=[], artifact_kind_hint=spec["kind"]
        )
        trace_step: dict[str, Any] = {
            "decision": decision.decision,
            "primary_error": decision.primary_error,
            "selected_tool": decision.selected_tool,
            "target_part": decision.target_part,
            "strength": decision.strength,
            "next_step": decision.next_step,
        }

        # Apply tool if revise.
        if decision.decision == "revise" and decision.selected_tool in tool_by_class:
            tool = tool_by_class[decision.selected_tool]
            try:
                corrected, report = tool.apply(
                    corrupted,
                    target_part=decision.target_part or "full_body",
                    target_joints=[],
                    frame_range=frame_range,
                    strength=decision.strength or "medium",
                )
                cumulative_mag = float(report.correction_magnitude)
                n_calls = 1
                trace_step["correction_magnitude"] = cumulative_mag
                trace_step["applied"] = True
            except ValueError as e:
                corrected = corrupted.copy()
                cumulative_mag = 0.0
                n_calls = 0
                trace_step["correction_magnitude"] = 0.0
                trace_step["applied"] = False
                trace_step["skip_reason"] = str(e)
        else:
            # STOP or reject: no application.
            corrected = corrupted.copy()
            cumulative_mag = 0.0
            n_calls = 0
            trace_step["correction_magnitude"] = 0.0
            trace_step["applied"] = False

        # Evaluate corrected.
        reports_after_dict = {ev.name: ev.evaluate(corrected) for ev in evaluators}
        target_score_after = _aggregate_target_score(
            reports_after_dict.get(spec["target_evaluator"], [])
        )
        target_delta = target_score_after - target_score_before
        mpjpe_corrected = _mpjpe(corrected, clean)
        fidelity_loss = mpjpe_corrected - mpjpe_corrupted
        artifact_reduction = -target_delta
        tool_call_cost = float(n_calls)
        netgain = artifact_reduction - alpha * fidelity_loss - beta * cumulative_mag - gamma * tool_call_cost

        cross_delta: dict[str, float] = {}
        for ev_name in reports_before_dict:
            if ev_name == spec["target_evaluator"]:
                continue
            b = _aggregate_target_score(reports_before_dict[ev_name])
            a = _aggregate_target_score(reports_after_dict.get(ev_name, []))
            cross_delta[ev_name] = float(a - b)

        selections[spec["kind"]] = {
            "artifact_kind": spec["kind"],
            "target_evaluator": spec["target_evaluator"],
            "orchestrator": "RuleBasedOrchestrator",
            "decision_trace": trace_step,
            "target_score_before": float(target_score_before),
            "target_score_after": float(target_score_after),
            "target_delta": float(target_delta),
            "fidelity_loss_protocol_a": float(fidelity_loss),
            "correction_magnitude": float(cumulative_mag),
            "tool_call_cost": float(tool_call_cost),
            "cross_evaluator_delta": cross_delta,
            "netgain_provisional": float(netgain),
        }
    return selections


def _make_raw_record(
    *,
    timestamp: str,
    task_id: str,
    split_id: str,
    trial_id: str,
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
        "baseline_type": "b5_rule_based",
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
        # Decision distribution.
        decision_counts: dict[str, int] = defaultdict(int)
        selected_tool_counts: dict[str, int] = defaultdict(int)
        for s in sels:
            d = s["decision_trace"]["decision"]
            decision_counts[d] += 1
            t = s["decision_trace"].get("selected_tool") or "NONE"
            selected_tool_counts[t] += 1
        artifact_summary.append({
            "artifact_kind": artifact_kind,
            "target_evaluator": sels[0]["target_evaluator"],
            "n_samples": len(sels),
            "decision_distribution": dict(decision_counts),
            "selected_tool_distribution": dict(selected_tool_counts),
            "netgain_provisional": {
                "mean": float(netgains.mean()),
                "median": float(np.median(netgains)),
                "p25": float(np.percentile(netgains, 25)),
                "p75": float(np.percentile(netgains, 75)),
                "min": float(netgains.min()),
                "max": float(netgains.max()),
            },
            "target_delta": {"mean": float(td.mean()), "median": float(np.median(td))},
            "fidelity_loss_protocol_a": {"mean": float(fl.mean()), "median": float(np.median(fl))},
            "correction_magnitude": {"mean": float(cm.mean()), "median": float(np.median(cm))},
        })
    return {"per_artifact": artifact_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="B5 (rule-based) baseline measurement")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--split-id", type=str, default=None)
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
                timestamp=timestamp, task_id=args.task_id, split_id=split_id,
                trial_id=trial_id, sample_path=path, seed=args.seed,
                motion_shape=motion.shape, selections=selections,
                evaluator_config_hashes=evaluator_config_hashes,
                evaluator_severity_versions=evaluator_severity_versions,
                tool_class_hashes=tool_class_hashes,
                netgain_weight_status=netgain_weight_status,
                netgain_weights=netgain_weights,
            )
            out_path = raw_dir / f"{timestamp}_{args.task_id}_{trial_id}.json"
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    aggregate = _aggregate(selections_by_sample)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": SUMMARY_TYPE,
        "task_id": args.task_id,
        "split_id": split_id,
        "baseline_type": "b5_rule_based",
        "orchestrator": "RuleBasedOrchestrator",
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
                f"  {ar['artifact_kind']:30s} | n={ar['n_samples']} | "
                f"NetGain median={ar['netgain_provisional']['median']:+.5f} | "
                f"decisions={ar['decision_distribution']} | "
                f"tools={ar['selected_tool_distribution']}"
            )
    else:
        print(text)


if __name__ == "__main__":
    main()
