"""Step 4-5: G2 natural sequence oracle (5-level strength).

사용자 directive (2026-05-25 9-step plan Step 4-5):
> "G2 general n=10 5-level oracle + G2 natural n=50 5-level oracle."

본 도구는 `tools/oracle_sequence_g2_run.py` 의 5-level 변형. action space 가
9 → 15 (3 tool × 5 strength). Protocol B simplified (reference = original_g2,
target = mean(all 3 evaluators)).

5-level strength: xsmall / small5 / medium5 / large5 / xlarge (factor 0.2-1.0).
docs/action_space_provenance.md + AGENTS.md §3-21 박제.

CLI:
    # G2 general n=10:
    python -m tools.oracle_sequence_g2_5level_run \\
        --g2-batch-dir external_assets/g2_general_pilot_v1 \\
        --task-id oracle_sequence_g2_general_5level_v1 \\
        --split-id g2_general_5level_v1 \\
        --output evals/snapshots/oracle_sequence_g2_general_5level_v1.json

    # G2 natural n=50:
    python -m tools.oracle_sequence_g2_5level_run \\
        --g2-batch-dir external_assets/g2_generated_v1 \\
        --task-id oracle_sequence_g2_natural_5level_v1 \\
        --split-id g2_natural_5level_v1 \\
        --output evals/snapshots/oracle_sequence_g2_natural_5level_v1.json
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

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_G2_DIR = REPO_ROOT / "external_assets" / "g2_generated_v1"
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "oracle_sequence_g2_5level_sample"
SUMMARY_TYPE = "oracle_sequence_g2_5level_summary"

ALL_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
TOOLS_WITH_TARGET_PARTS: list[tuple[str, str]] = [
    ("FootLockTool", "both_feet"),
    ("BoneProjectionTool", "right_arm"),
    ("VelocitySmoothingTool", "full_body"),
]
STRENGTHS_5LEVEL = ("xsmall", "small5", "medium5", "large5", "xlarge")
STRENGTH_RANK_5 = {"xsmall": 0, "small5": 1, "medium5": 2, "large5": 3, "xlarge": 4}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _get_severity_version(evaluator: Any) -> str:
    mod = sys.modules.get(type(evaluator).__module__)
    if mod is None:
        return "unversioned"
    return getattr(mod, "SEVERITY_VERSION", "unversioned")


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _evaluate_all(motion: np.ndarray, evaluators: list[Any]) -> dict[str, list[EvaluatorReport]]:
    return {ev.name: ev.evaluate(motion) for ev in evaluators}


def _target_score_full(reports_by_ev: dict[str, list[EvaluatorReport]]) -> float:
    return float(np.mean([_max_score(reports_by_ev.get(n, [])) for n in ALL_EVALUATORS]))


def _total_artifact_score(reports_by_ev: dict[str, list[EvaluatorReport]]) -> float:
    total = 0.0
    for reports in reports_by_ev.values():
        for r in reports:
            total += float(r.score)
    return total


def _eval_candidate(
    *, motion: np.ndarray, original_g2_motion: np.ndarray,
    sequence: list[tuple[str, str, str]], cum_corr_mag: float,
    target_initial_full: float, cross_initial: dict[str, float],
    evaluators: list[Any], netgain_weights: dict[str, float],
) -> dict[str, Any]:
    reports = _evaluate_all(motion, evaluators)
    target_final_full = _target_score_full(reports)
    target_delta_full = target_final_full - target_initial_full
    fidelity_loss_protocol_b = _mpjpe(motion, original_g2_motion)
    alpha = float(netgain_weights["alpha"])
    beta = float(netgain_weights["beta"])
    gamma = float(netgain_weights["gamma"])
    netgain = (
        -target_delta_full - alpha * fidelity_loss_protocol_b - beta * cum_corr_mag - gamma * len(sequence)
    )
    cross_delta = {name: float(_max_score(reports[name]) - cross_initial.get(name, 0.0)) for name in reports}
    return {
        "sequence": [list(s) for s in sequence], "length": len(sequence),
        "target_score_initial_full": target_initial_full,
        "target_score_final_full": target_final_full,
        "target_delta_full": float(target_delta_full),
        "fidelity_loss_protocol_b": float(fidelity_loss_protocol_b),
        "cumulative_correction_magnitude": float(cum_corr_mag),
        "cross_evaluator_delta": cross_delta,
        "netgain": float(netgain),
    }


def _strength_allowed(
    used: dict[tuple[str, str], list[str]], tool_name: str, target_part: str, new_strength: str,
) -> bool:
    prev = used.get((tool_name, target_part), [])
    if not prev:
        return True
    return STRENGTH_RANK_5[new_strength] < min(STRENGTH_RANK_5[s] for s in prev)


def select_best_sequence_g2_5level(
    *, original_g2_motion: np.ndarray,
    tools_by_name: dict[str, CorrectionTool],
    tools_with_target_parts: list[tuple[str, str]],
    strengths: tuple[str, ...], evaluators: list[Any],
    netgain_weights: dict[str, float], max_depth: int,
    score_increase_tolerance: float = 0.01, top_k: int = 10,
) -> dict[str, Any]:
    T = original_g2_motion.shape[0]
    frame_range = (0, T - 1)
    reports_initial = _evaluate_all(original_g2_motion, evaluators)
    target_initial_full = _target_score_full(reports_initial)
    cross_initial = {name: _max_score(reports_initial[name]) for name in reports_initial}
    total_initial = _total_artifact_score(reports_initial)
    actions = [(tn, tp, st) for tn, tp in tools_with_target_parts for st in strengths]
    all_candidates: list[dict[str, Any]] = []
    counters = {"explored": 0, "pruned_score": 0, "pruned_strength": 0, "pruned_apply_error": 0}

    def _record(motion: np.ndarray, sequence: list[tuple[str, str, str]], cum_corr_mag: float) -> None:
        cand = _eval_candidate(
            motion=motion, original_g2_motion=original_g2_motion,
            sequence=sequence, cum_corr_mag=cum_corr_mag,
            target_initial_full=target_initial_full, cross_initial=cross_initial,
            evaluators=evaluators, netgain_weights=netgain_weights,
        )
        all_candidates.append(cand)
        counters["explored"] += 1

    def _dfs(motion: np.ndarray, sequence: list, cum_corr_mag: float, prev_total: float, used: dict) -> None:
        _record(motion, sequence, cum_corr_mag)
        if len(sequence) >= max_depth:
            return
        for tn, tp, st in actions:
            if not _strength_allowed(used, tn, tp, st):
                counters["pruned_strength"] += 1
                continue
            tool = tools_by_name[tn]
            try:
                new_motion, report = tool.apply(motion, target_part=tp, target_joints=[],
                                                  frame_range=frame_range, strength=st)
            except ValueError:
                counters["pruned_apply_error"] += 1
                continue
            new_reports = _evaluate_all(new_motion, evaluators)
            new_total = _total_artifact_score(new_reports)
            if new_total > prev_total + score_increase_tolerance:
                counters["pruned_score"] += 1
                continue
            new_used = dict(used)
            new_used[(tn, tp)] = list(used.get((tn, tp), [])) + [st]
            _dfs(new_motion, sequence + [(tn, tp, st)],
                 cum_corr_mag + float(report.correction_magnitude), new_total, new_used)

    _dfs(original_g2_motion, [], 0.0, total_initial, {})
    if not all_candidates:
        return {"best": None, "top_k": [], "counters": counters,
                "metadata": {"target_initial_full": target_initial_full, "total_initial": total_initial}}
    best = max(all_candidates, key=lambda c: (c["netgain"], -c["length"]))
    top_k_list = sorted(all_candidates, key=lambda c: (c["netgain"], -c["length"]), reverse=True)[:top_k]
    return {
        "best": best, "top_k": top_k_list, "counters": counters,
        "metadata": {
            "target_initial_full": target_initial_full, "total_initial": total_initial,
            "n_candidates_total": len(all_candidates),
        },
    }


def _load_g2_motions(batch_dir: Path) -> list[tuple[str, Path, dict, np.ndarray]]:
    out = []
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
        out.append((trial_id, npy_path, meta, motion))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 5-level Sequence Oracle (RL-2 candidate ablation)")
    parser.add_argument("--g2-batch-dir", type=Path, default=DEFAULT_G2_DIR)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--score-increase-tolerance", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=-1, help="-1 = all motions in dir")
    parser.add_argument("--task-id", type=str, default="oracle_sequence_g2_5level_v1")
    parser.add_argument("--split-id", type=str, default="g2_natural_5level_v1")
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
    tools_by_name: dict[str, CorrectionTool] = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "BoneProjectionTool": BoneProjectionTool(),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
    }
    evaluators = list(DEFAULT_EVALUATORS)
    evaluator_config_hashes = {ev.name: ev.evaluator_class_hash() for ev in evaluators}
    evaluator_severity_versions = {ev.name: _get_severity_version(ev) for ev in evaluators}
    tool_class_hashes = {name: t.tool_class_hash() for name, t in tools_by_name.items()}

    motions = _load_g2_motions(args.g2_batch_dir)
    if args.n_samples > 0:
        motions = motions[: args.n_samples]
    print(f"[INFO] loaded {len(motions)} G2 motions from {args.g2_batch_dir}")

    raw_dir = Path(args.raw_output_dir).resolve() if args.raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    per_sample: list[dict[str, Any]] = []
    for i, (trial_id, npy_path, g2_meta, motion) in enumerate(motions, 1):
        result = select_best_sequence_g2_5level(
            original_g2_motion=motion, tools_by_name=tools_by_name,
            tools_with_target_parts=TOOLS_WITH_TARGET_PARTS,
            strengths=STRENGTHS_5LEVEL, evaluators=evaluators,
            netgain_weights=netgain_weights, max_depth=args.max_depth,
            score_increase_tolerance=args.score_increase_tolerance, top_k=args.top_k,
        )
        sample_record = {
            "trial_id": trial_id, "sample_path": str(npy_path),
            "g2_prompt": g2_meta.get("prompt", "")[:80],
            "best": result["best"], "top_k": result["top_k"],
            "counters": result["counters"], "metadata": result["metadata"],
        }
        per_sample.append(sample_record)
        if result["best"] is not None:
            print(f"[OK] {i}/{len(motions)} {trial_id}: NetGain={result['best']['netgain']:+.4f} "
                  f"(len {result['best']['length']}), explored={result['counters']['explored']}")
        else:
            print(f"[WARN] {i}/{len(motions)} {trial_id}: no candidate")

        if raw_dir is not None:
            ts = _now_iso()
            record = {
                "schema_version": SCHEMA_VERSION, "record_type": RECORD_TYPE,
                "timestamp": ts, "task_id": args.task_id, "split_id": args.split_id,
                "trial_id": trial_id, "sample_path": str(npy_path),
                "generator_id": g2_meta.get("generator_id"),
                "g2_prompt": g2_meta.get("prompt"), "g2_seed": g2_meta.get("seed"),
                "evaluator_config_hashes": evaluator_config_hashes,
                "evaluator_severity_versions": evaluator_severity_versions,
                "tool_class_hashes": tool_class_hashes,
                "motion_shape": list(motion.shape), "fps": 20,
                "oracle_type": "sequence",
                "action_space_grid": "5level_xsmall_small5_medium5_large5_xlarge",
                "strength_factors_used": list(STRENGTHS_5LEVEL),
                "fidelity_loss_reference": "original_g2_motion",
                "target_score_definition": "mean(FootFloating, BoneLength, VelocityJitter max scores)",
                "netgain_weight_status": "calibrated_protocol_a_v1",
                "netgain_weights": dict(netgain_weights),
                "max_depth": args.max_depth,
                "best": result["best"], "top_k": result["top_k"],
                "counters": result["counters"], "metadata": result["metadata"],
                "negative_result": False,
            }
            (raw_dir / f"{ts}_{args.task_id}_{trial_id}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    bests = [s["best"] for s in per_sample if s["best"] is not None]
    if bests:
        ng = np.array([b["netgain"] for b in bests])
        lens = Counter(b["length"] for b in bests)
        first_actions = Counter((b["sequence"][0][0] if b["length"] >= 1 else "STOP") for b in bests)
        first_strengths = Counter((b["sequence"][0][2] if b["length"] >= 1 else "n/a") for b in bests)
        aggregate = {
            "n": len(bests),
            "netgain": {"mean": float(ng.mean()), "median": float(np.median(ng)),
                        "min": float(ng.min()), "max": float(ng.max()),
                        "p25": float(np.percentile(ng, 25)), "p75": float(np.percentile(ng, 75))},
            "best_length_freq": dict(lens),
            "first_action_freq": dict(first_actions),
            "first_strength_freq": dict(first_strengths),
        }
    else:
        aggregate = {"n": 0}

    summary = {
        "schema_version": SCHEMA_VERSION, "record_type": SUMMARY_TYPE,
        "task_id": args.task_id, "split_id": args.split_id,
        "n_samples_evaluated": len(per_sample), "max_depth": args.max_depth,
        "action_space_grid": "5level",
        "strength_factors_used": list(STRENGTHS_5LEVEL),
        "oracle_type": "sequence",
        "fidelity_loss_reference": "original_g2_motion",
        "target_score_definition": "mean(FootFloating, BoneLength, VelocityJitter max scores)",
        "netgain_weight_status": "calibrated_protocol_a_v1",
        "netgain_weights": dict(netgain_weights),
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,
        "tool_class_hashes": tool_class_hashes,
        "trial_ids": [s["trial_id"] for s in per_sample],
        "aggregate": aggregate, "per_sample": per_sample,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[OK] wrote {args.output}")
    print(f"\n=== Aggregate G2 5-level (target=mean(all 3), Protocol B simplified) ===")
    print(f"  n={aggregate['n']}, NetGain median={aggregate['netgain']['median']:+.5f}, mean={aggregate['netgain']['mean']:+.5f}")
    print(f"  best_length: {aggregate['best_length_freq']}")
    print(f"  first_action: {aggregate['first_action_freq']}")
    print(f"  first_strength (5-level): {aggregate['first_strength_freq']}")


if __name__ == "__main__":
    main()
