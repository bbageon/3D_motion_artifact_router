"""Step 3-5: 5-level Sequence Oracle (multi-artifact + G2 natural).

사용자 directive (2026-05-25, action space ablation):
> "5-level oracle 의 비교 — 3-level vs 5-level. RL-2 action space 결정."

본 도구는 `tools/oracle_sequence_multi_run.py` 의 5-level 변형. action space 가
9 → 15 으로 확장 (3 tool × 5 strength). pruning + same-pair strength decay 동일.

5-level tokens: xsmall / small5 / medium5 / large5 / xlarge (factor 0.2-1.0).
docs/action_space_provenance.md + AGENTS.md §3-21 박제.

NOTE: 본 도구 는 synthetic (Protocol A target=mean(foot+jitter)) 만 지원.
G2 natural 의 5-level 은 별도 `oracle_sequence_g2_5level_run.py` (후속).

CLI:
    python -m tools.oracle_sequence_multi_5level_run \\
        --n-samples 60 --seed 42 --max-depth 5 \\
        --task-id oracle_sequence_multi_5level_v1 \\
        --raw-output-dir evals/raw \\
        --output evals/snapshots/oracle_sequence_multi_5level_v1.json
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
from tools.synthetic_injection import inject_foot_floating, inject_jitter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "oracle_sequence_multi_5level_sample"
SUMMARY_TYPE = "oracle_sequence_multi_5level_summary"

ALL_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
TARGET_EVALUATORS_A = ("FootFloatingEvaluator", "VelocityJitterEvaluator")
TOOLS_WITH_TARGET_PARTS: list[tuple[str, str]] = [
    ("FootLockTool", "both_feet"),
    ("BoneProjectionTool", "right_arm"),
    ("VelocitySmoothingTool", "full_body"),
]

#: 5-level strength tokens (docs/action_space_provenance.md).
STRENGTHS_5LEVEL = ("xsmall", "small5", "medium5", "large5", "xlarge")
STRENGTH_RANK_5 = {"xsmall": 0, "small5": 1, "medium5": 2, "large5": 3, "xlarge": 4}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _get_severity_version(evaluator: Any) -> str:
    mod = sys.modules.get(type(evaluator).__module__)
    if mod is None:
        return "unversioned"
    return getattr(mod, "SEVERITY_VERSION", "unversioned")


def _multi_inject(clean: np.ndarray, seed: int) -> np.ndarray:
    m1 = inject_foot_floating(clean, lift_height=0.08, seed=seed)
    return inject_jitter(m1, noise_std=0.05, seed=seed + 1000)


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _max_score(reports: list[EvaluatorReport]) -> float:
    if not reports:
        return 0.0
    return float(max(r.score for r in reports))


def _evaluate_all(motion: np.ndarray, evaluators: list[Any]) -> dict[str, list[EvaluatorReport]]:
    return {ev.name: ev.evaluate(motion) for ev in evaluators}


def _target_score_A(reports_by_ev: dict[str, list[EvaluatorReport]]) -> float:
    return float(np.mean([_max_score(reports_by_ev.get(n, [])) for n in TARGET_EVALUATORS_A]))


def _target_score_full(reports_by_ev: dict[str, list[EvaluatorReport]]) -> float:
    return float(np.mean([_max_score(reports_by_ev.get(n, [])) for n in ALL_EVALUATORS]))


def _total_artifact_score(reports_by_ev: dict[str, list[EvaluatorReport]]) -> float:
    total = 0.0
    for reports in reports_by_ev.values():
        for r in reports:
            total += float(r.score)
    return total


def _eval_candidate(
    *, motion: np.ndarray, clean_motion: np.ndarray, corrupted_motion: np.ndarray,
    sequence: list[tuple[str, str, str]], cum_corr_mag: float,
    target_initial_A: float, target_initial_full: float,
    cross_initial: dict[str, float], mpjpe_corrupted_clean: float,
    evaluators: list[Any], netgain_weights: dict[str, float],
) -> dict[str, Any]:
    reports = _evaluate_all(motion, evaluators)
    target_final_A = _target_score_A(reports)
    target_final_full = _target_score_full(reports)
    target_delta_A = target_final_A - target_initial_A
    target_delta_full = target_final_full - target_initial_full
    mpjpe_final = _mpjpe(motion, clean_motion)
    fidelity_loss_protocol_a = mpjpe_final - mpjpe_corrupted_clean
    alpha = float(netgain_weights["alpha"])
    beta = float(netgain_weights["beta"])
    gamma = float(netgain_weights["gamma"])
    netgain_A = (
        -target_delta_A - alpha * fidelity_loss_protocol_a - beta * cum_corr_mag - gamma * len(sequence)
    )
    netgain_full = (
        -target_delta_full - alpha * fidelity_loss_protocol_a - beta * cum_corr_mag - gamma * len(sequence)
    )
    cross_delta = {name: float(_max_score(reports[name]) - cross_initial.get(name, 0.0)) for name in reports}
    return {
        "sequence": [list(s) for s in sequence], "length": len(sequence),
        "target_score_initial_A": target_initial_A, "target_score_final_A": target_final_A,
        "target_delta_A": float(target_delta_A),
        "target_score_initial_full": target_initial_full, "target_score_final_full": target_final_full,
        "target_delta_full": float(target_delta_full),
        "fidelity_loss_protocol_a": float(fidelity_loss_protocol_a),
        "cumulative_correction_magnitude": float(cum_corr_mag),
        "cross_evaluator_delta": cross_delta,
        "netgain_A": float(netgain_A), "netgain_full": float(netgain_full),
    }


def _strength_allowed(
    used: dict[tuple[str, str], list[str]], tool_name: str, target_part: str, new_strength: str,
) -> bool:
    prev = used.get((tool_name, target_part), [])
    if not prev:
        return True
    return STRENGTH_RANK_5[new_strength] < min(STRENGTH_RANK_5[s] for s in prev)


def select_best_sequence_5level(
    *, clean_motion: np.ndarray, corrupted_motion: np.ndarray,
    tools_by_name: dict[str, CorrectionTool],
    tools_with_target_parts: list[tuple[str, str]],
    strengths: tuple[str, ...], evaluators: list[Any],
    netgain_weights: dict[str, float], max_depth: int,
    score_increase_tolerance: float = 0.01, top_k: int = 10,
) -> dict[str, Any]:
    T = corrupted_motion.shape[0]
    frame_range = (0, T - 1)
    reports_initial = _evaluate_all(corrupted_motion, evaluators)
    target_initial_A = _target_score_A(reports_initial)
    target_initial_full = _target_score_full(reports_initial)
    cross_initial = {name: _max_score(reports_initial[name]) for name in reports_initial}
    total_initial = _total_artifact_score(reports_initial)
    mpjpe_corrupted_clean = _mpjpe(corrupted_motion, clean_motion)

    actions = [(tn, tp, st) for tn, tp in tools_with_target_parts for st in strengths]
    all_candidates: list[dict[str, Any]] = []
    counters = {"explored": 0, "pruned_score": 0, "pruned_strength": 0, "pruned_apply_error": 0}

    def _record(motion: np.ndarray, sequence: list[tuple[str, str, str]], cum_corr_mag: float) -> None:
        cand = _eval_candidate(
            motion=motion, clean_motion=clean_motion, corrupted_motion=corrupted_motion,
            sequence=sequence, cum_corr_mag=cum_corr_mag,
            target_initial_A=target_initial_A, target_initial_full=target_initial_full,
            cross_initial=cross_initial, mpjpe_corrupted_clean=mpjpe_corrupted_clean,
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

    _dfs(corrupted_motion, [], 0.0, total_initial, {})
    if not all_candidates:
        return {"best_A": None, "best_full": None, "top_k_A": [], "top_k_full": [],
                "counters": counters,
                "metadata": {"mpjpe_corrupted_clean": mpjpe_corrupted_clean,
                              "target_initial_A": target_initial_A, "target_initial_full": target_initial_full,
                              "total_initial": total_initial}}
    best_A = max(all_candidates, key=lambda c: (c["netgain_A"], -c["length"]))
    best_full = max(all_candidates, key=lambda c: (c["netgain_full"], -c["length"]))
    top_k_A = sorted(all_candidates, key=lambda c: (c["netgain_A"], -c["length"]), reverse=True)[:top_k]
    top_k_full = sorted(all_candidates, key=lambda c: (c["netgain_full"], -c["length"]), reverse=True)[:top_k]
    return {
        "best_A": best_A, "best_full": best_full, "top_k_A": top_k_A, "top_k_full": top_k_full,
        "counters": counters,
        "metadata": {
            "mpjpe_corrupted_clean": mpjpe_corrupted_clean,
            "target_initial_A": target_initial_A, "target_initial_full": target_initial_full,
            "total_initial": total_initial, "n_candidates_total": len(all_candidates),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="5-level Sequence Oracle (multi-artifact, RL-2 candidate ablation)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--score-increase-tolerance", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--task-id", type=str, default="oracle_sequence_multi_5level_v1")
    parser.add_argument("--split-id", type=str, default="multi_artifact_5level_v1")
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

    rng = np.random.default_rng(args.seed)
    npy_files = sorted(args.data_dir.glob("*.npy"))
    chosen_idx = rng.choice(len(npy_files), size=args.n_samples, replace=False)
    chosen = [npy_files[i] for i in chosen_idx]

    raw_dir = Path(args.raw_output_dir).resolve() if args.raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    per_sample: list[dict[str, Any]] = []
    for i, path in enumerate(chosen, 1):
        trial_id = path.stem
        try:
            clean = np.load(str(path)).astype(np.float64)
        except Exception as e:
            print(f"[WARN] {trial_id}: {e}", file=sys.stderr); continue
        if clean.ndim != 3 or clean.shape[1] != 22 or clean.shape[2] != 3:
            continue
        corrupted = _multi_inject(clean, seed=args.seed)
        result = select_best_sequence_5level(
            clean_motion=clean, corrupted_motion=corrupted,
            tools_by_name=tools_by_name, tools_with_target_parts=TOOLS_WITH_TARGET_PARTS,
            strengths=STRENGTHS_5LEVEL, evaluators=evaluators, netgain_weights=netgain_weights,
            max_depth=args.max_depth, score_increase_tolerance=args.score_increase_tolerance, top_k=args.top_k,
        )
        sample_record = {
            "trial_id": trial_id, "sample_path": str(path),
            "best_A": result["best_A"], "best_full": result["best_full"],
            "top_k_A": result["top_k_A"], "top_k_full": result["top_k_full"],
            "counters": result["counters"], "metadata": result["metadata"],
        }
        per_sample.append(sample_record)
        if result["best_A"] is not None:
            print(f"[OK] {i}/{len(chosen)} {trial_id}: NetGain_A={result['best_A']['netgain_A']:+.4f} "
                  f"(len {result['best_A']['length']}), explored={result['counters']['explored']}, "
                  f"strength_pruned={result['counters']['pruned_strength']}, score_pruned={result['counters']['pruned_score']}")
        else:
            print(f"[WARN] {i}/{len(chosen)} {trial_id}: no candidate")

        if raw_dir is not None:
            ts = _now_iso()
            record = {
                "schema_version": SCHEMA_VERSION, "record_type": RECORD_TYPE,
                "timestamp": ts, "task_id": args.task_id, "split_id": args.split_id,
                "trial_id": trial_id, "sample_path": str(path), "generator_id": "humanml3d_gt",
                "evaluator_config_hashes": evaluator_config_hashes,
                "evaluator_severity_versions": evaluator_severity_versions,
                "tool_class_hashes": tool_class_hashes,
                "motion_shape": list(clean.shape), "fps": 20, "seed": int(args.seed),
                "oracle_type": "sequence",
                "action_space_grid": "5level_xsmall_small5_medium5_large5_xlarge",
                "strength_factors_used": list(STRENGTHS_5LEVEL),
                "netgain_weight_status": "calibrated_protocol_a_v1",
                "netgain_weights": dict(netgain_weights),
                "max_depth": args.max_depth,
                "multi_artifact_recipe": ["foot_floating", "global_jitter"],
                "target_evaluators_A": list(TARGET_EVALUATORS_A),
                "best_A": result["best_A"], "best_full": result["best_full"],
                "counters": result["counters"], "metadata": result["metadata"],
                "negative_result": False,
            }
            (raw_dir / f"{ts}_{args.task_id}_{trial_id}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    # Aggregate.
    def _agg(field_best: str, field_ng: str) -> dict[str, Any]:
        bests = [s[field_best] for s in per_sample if s[field_best] is not None]
        if not bests:
            return {"n": 0}
        ng = np.array([b[field_ng] for b in bests])
        lengths = Counter(b["length"] for b in bests)
        first_actions = Counter(b["sequence"][0][0] if b["length"] >= 1 else "STOP" for b in bests)
        first_strengths = Counter(b["sequence"][0][2] if b["length"] >= 1 else "n/a" for b in bests)
        return {
            "n": len(bests),
            "netgain": {"mean": float(ng.mean()), "median": float(np.median(ng)),
                        "min": float(ng.min()), "max": float(ng.max()),
                        "p25": float(np.percentile(ng, 25)), "p75": float(np.percentile(ng, 75))},
            "best_length_freq": dict(lengths),
            "first_action_freq": dict(first_actions),
            "first_strength_freq": dict(first_strengths),
        }

    summary = {
        "schema_version": SCHEMA_VERSION, "record_type": SUMMARY_TYPE,
        "task_id": args.task_id, "split_id": args.split_id,
        "n_samples_evaluated": len(per_sample),
        "seed": int(args.seed), "max_depth": args.max_depth,
        "action_space_grid": "5level",
        "strength_factors_used": list(STRENGTHS_5LEVEL),
        "multi_artifact_recipe": ["foot_floating", "global_jitter"],
        "target_evaluators_A": list(TARGET_EVALUATORS_A),
        "oracle_type": "sequence",
        "netgain_weight_status": "calibrated_protocol_a_v1",
        "netgain_weights": dict(netgain_weights),
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,
        "tool_class_hashes": tool_class_hashes,
        "trial_ids": [s["trial_id"] for s in per_sample],
        "aggregate_A": _agg("best_A", "netgain_A"),
        "aggregate_full": _agg("best_full", "netgain_full"),
        "per_sample": per_sample,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[OK] wrote {args.output}")
    print(f"\n=== Aggregate 5-level (target=mean(foot+jitter)) ===")
    print(f"  n={summary['aggregate_A']['n']}, NetGain median={summary['aggregate_A']['netgain']['median']:+.5f}, mean={summary['aggregate_A']['netgain']['mean']:+.5f}")
    print(f"  best_length: {summary['aggregate_A']['best_length_freq']}")
    print(f"  first_action: {summary['aggregate_A']['first_action_freq']}")
    print(f"  first_strength (5-level): {summary['aggregate_A']['first_strength_freq']}")


if __name__ == "__main__":
    main()
