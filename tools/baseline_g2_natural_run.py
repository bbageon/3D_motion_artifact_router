"""G2 natural artifact 에 baseline 적용 측정 — Step 5 (transfer diagnostic).

**리포트 의무 (사용자 directive 2026-05-20)**:
> 본 실험은 B6/B7 의 최종 우위 검증이 아니라, synthetic 에서 관측된
> single-step ceiling 이 G2 natural distribution 에서도 재현되는지 확인하는
> transfer diagnostic 이다.

목적 (4 specific question):
  1. G2 natural artifact 에서 B6/B7 > B5 인가? (learned > rule 의 transfer)
  2. G2 에서도 action 이 VS/medium 으로 수렴? (single-step ceiling 재현)
  3. BoneLength artifact 가 많은 G2 에서 B2 smoothing 만으로 충분? (B2 ceiling 검증)
  4. B2 가 못 고치는 bone artifact 에서 learned policy 의 gain 있는가?

비교 baselines:
  - B1 (no refinement) — reference (NetGain ≡ 0).
  - B2 (fixed smoothing) — multi-artifact 의 single-step ceiling.
  - B5 (rule-based, full state) — rule baseline.
  - B6 single (supervised, no hint) — single-step supervised.
  - B6 closed-loop — multi-step supervised.
  - B7 bandit closed-loop — multi-step Q-regression.

NetGain (Protocol B simplified):
  - reference for FidelityLoss = **original G2 motion** (artifact 보유, clean GT 없음).
  - FidelityLoss = MPJPE(refined, original_g2) — modification magnitude proxy.
  - target_score = mean(FootFloating, **BoneLength**, VelocityJitter max scores) — 모든
    evaluator 포함 (Step 4 의 full state + bone 분석 의무).
  - NetGain = -ΔTargetScore - α·FidelityLoss - β·CorrectionMag - γ·ToolCost.
  - α=5.0 calibration 은 synthetic Protocol A 결과 — G2 transfer 시 magnitude 종속 caveat.

CLI:
    python -m tools.baseline_g2_natural_run \\
        --baseline-type b6_closed_loop \\
        --g2-batch-dir external_assets/g2_generated_v1 \\
        --training-data evals/snapshots/oracle_training_data_v2.json \\
        --task-id baseline_b6_cl_g2_v1 \\
        --output evals/snapshots/baseline_b6_cl_g2_v1.json
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
from orchestrator.contextual_bandit import (
    ContextualBandit,
    ContextualBanditOrchestrator,
)
from orchestrator.contextual_bandit import (
    split_train_eval_by_trial as bandit_split,
    tuples_to_arrays as bandit_tuples_to_arrays,
)
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from orchestrator.rule_based import RuleBasedOrchestrator
from orchestrator.supervised_selector import (
    SupervisedSelector,
    SupervisedSelectorOrchestrator,
    split_train_eval_by_trial,
    tuples_to_arrays,
)
from refinement_loop.loop import RefinementLoop

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "baseline_g2_natural_sample"
ALL_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")

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


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _max_score(reports: list[EvaluatorReport]) -> float:
    if not reports:
        return 0.0
    return float(max(r.score for r in reports))


def _target_score_all(reports_dict: dict[str, list[EvaluatorReport]]) -> float:
    """Target = mean(all 3 evaluators max). bone 포함 (Step 4/5 의 full state)."""
    return float(np.mean([_max_score(reports_dict.get(n, [])) for n in ALL_EVALUATORS]))


def _load_g2_motions(batch_dir: Path) -> list[tuple[Path, dict[str, Any], np.ndarray]]:
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
        out.append((npy_path, meta, motion))
    return out


def _apply_b2_fixed_smoothing(motion: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """B2: VelocitySmoothing(full_body, medium) 1회."""
    T = motion.shape[0]
    tool = TOOL_BY_NAME["VelocitySmoothingTool"]
    try:
        corrected, report = tool.apply(motion, target_part="full_body", target_joints=[],
                                       frame_range=(0, T - 1), strength="medium")
        return corrected, {
            "decision": "revise", "tool": "VelocitySmoothingTool",
            "strength": "medium", "target_part": "full_body",
            "correction_magnitude": float(report.correction_magnitude),
            "tool_call_count": 1,
        }
    except ValueError as e:
        return motion.copy(), {
            "decision": "skip", "skip_reason": str(e),
            "correction_magnitude": 0.0, "tool_call_count": 0,
        }


def _apply_b1_noop(motion: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    return motion.copy(), {
        "decision": "STOP", "tool": None, "tool_call_count": 0, "correction_magnitude": 0.0,
    }


def _apply_b5_rule_based(motion: np.ndarray, evaluators: list[Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """B5 single-step: rule_based 의 default (no artifact_kind_hint, 모든 evaluator 후보).

    G2 natural 은 multi-artifact 자연 분포 — hint 없이 severity-based primary 결정.
    """
    T = motion.shape[0]
    all_reports = []
    for ev in evaluators:
        all_reports.extend(ev.evaluate(motion))
    orch = RuleBasedOrchestrator(tool_registry=list(TOOL_BY_NAME.values()))
    decision = orch.decide(all_reports, tool_history=[], artifact_kind_hint=None)
    if decision.decision == "revise" and decision.selected_tool in TOOL_BY_NAME:
        tool = TOOL_BY_NAME[decision.selected_tool]
        try:
            corrected, report = tool.apply(motion, target_part=decision.target_part or "full_body",
                                           target_joints=[], frame_range=(0, T - 1),
                                           strength=decision.strength or "medium")
            return corrected, {
                "decision": "revise", "primary_error": decision.primary_error,
                "tool": decision.selected_tool, "strength": decision.strength,
                "target_part": decision.target_part,
                "correction_magnitude": float(report.correction_magnitude),
                "tool_call_count": 1,
            }
        except ValueError as e:
            return motion.copy(), {"decision": "skip", "skip_reason": str(e),
                                   "tool_call_count": 0, "correction_magnitude": 0.0}
    return motion.copy(), {"decision": decision.decision, "tool": None,
                           "tool_call_count": 0, "correction_magnitude": 0.0}


def _apply_b6_single(motion: np.ndarray, evaluators: list[Any], selector: SupervisedSelector,
                    artifact_kinds_list: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """B6 single-step (no closed-loop): primary artifact = max-score evaluator → predict."""
    T = motion.shape[0]
    reports_by_ev = {ev.name: ev.evaluate(motion) for ev in evaluators}
    # primary artifact = max score among 3 evaluators.
    eval_to_artifact = {
        "FootFloatingEvaluator": "foot_floating",
        "BoneLengthEvaluator": "bone_stretch_right_arm",
        "VelocityJitterEvaluator": "global_jitter",
    }
    scores_3 = {n: _max_score(reports_by_ev.get(n, [])) for n in ALL_EVALUATORS}
    primary_ev = max(scores_3, key=scores_3.get)
    primary_artifact = eval_to_artifact[primary_ev]

    artifact_onehot = [1 if k == primary_artifact else 0 for k in artifact_kinds_list]
    eval_scores_ordered = [scores_3.get(n, 0.0) for n in ALL_EVALUATORS]
    state_vector = np.array(artifact_onehot + eval_scores_ordered, dtype=np.float64)
    pred = selector.predict(state_vector, primary_artifact)
    tool = TOOL_BY_NAME.get(pred.tool_name)
    if tool is None:
        return motion.copy(), {"decision": "skip", "tool_call_count": 0,
                               "correction_magnitude": 0.0}
    try:
        corrected, report = tool.apply(motion, target_part=pred.target_part, target_joints=[],
                                       frame_range=(0, T - 1), strength=pred.strength)
        return corrected, {
            "decision": "revise", "tool": pred.tool_name, "strength": pred.strength,
            "target_part": pred.target_part,
            "primary_evaluator": primary_ev, "primary_artifact": primary_artifact,
            "correction_magnitude": float(report.correction_magnitude),
            "tool_call_count": 1,
        }
    except ValueError as e:
        return motion.copy(), {"decision": "skip", "skip_reason": str(e),
                               "tool_call_count": 0, "correction_magnitude": 0.0}


def _apply_closed_loop(motion: np.ndarray, evaluators: list[Any], orchestrator: Any,
                       max_iterations: int = 4, tolerance: float = 0.01) -> tuple[np.ndarray, dict[str, Any]]:
    """closed-loop with given orchestrator. RefinementLoop wrapper."""
    loop = RefinementLoop(
        evaluators=evaluators, correction_tools=TOOL_BY_NAME, orchestrator=orchestrator,
        max_iterations=max_iterations, score_increase_tolerance=tolerance,
    )
    result = loop.run(motion, fps=20)
    decision_summary = []
    for d in result.decision_history:
        decision_summary.append({
            "decision": d.decision, "tool": d.selected_tool, "strength": d.strength,
            "target_part": d.target_part,
        })
    return result.refined_motion, {
        "decision": "closed_loop",
        "tool_call_count": len(result.tool_history),
        "correction_magnitude": float(sum(r.correction_magnitude for r in result.tool_history)),
        "converged": result.converged, "rolled_back": result.rolled_back,
        "max_iter_reached": result.max_iterations_reached,
        "stop_reason": result.metadata.get("stop_reason"),
        "decision_trace": decision_summary,
        "score_trace": [float(x) for x in result.score_trace],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 natural baseline measurement (transfer diagnostic)")
    parser.add_argument("--baseline-type", required=True,
                        choices=["b1_no_refinement", "b2_fixed_smoothing", "b5_rule_based",
                                 "b6_single", "b6_closed_loop", "b7_bandit"])
    parser.add_argument("--g2-batch-dir", type=Path, default=REPO_ROOT / "external_assets" / "g2_generated_v1")
    parser.add_argument("--training-data", type=Path,
                        help="b6_* 시 oracle_training_data, b7 시 bandit_training_data.")
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--model-random-state", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--stop-score-threshold", type=float, default=0.02)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--split-id", default="g2_natural_v1")
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
    alpha = netgain_weights["alpha"]
    beta = netgain_weights["beta"]
    gamma = netgain_weights["gamma"]

    g2_motions = _load_g2_motions(args.g2_batch_dir)
    print(f"[INFO] loaded {len(g2_motions)} G2 motions from {args.g2_batch_dir}")
    evaluators = list(DEFAULT_EVALUATORS)

    # Setup selectors if needed.
    supervised_selector: Optional[SupervisedSelector] = None
    bandit: Optional[ContextualBandit] = None
    artifact_kinds_list = ["foot_floating", "bone_stretch_right_arm", "global_jitter"]

    if args.baseline_type in ("b6_single", "b6_closed_loop"):
        with open(args.training_data, encoding="utf-8") as f:
            td = json.load(f)
        tuples = td["tuples"]
        train_tuples, _ = split_train_eval_by_trial(tuples, train_ratio=args.train_ratio, seed=args.seed)
        X_tr, tool_tr, str_tr = tuples_to_arrays(train_tuples)
        supervised_selector = SupervisedSelector(model_type="random_forest",
                                                 random_state=args.model_random_state)
        supervised_selector.train(X_tr, tool_tr, str_tr, feature_names=td.get("feature_names"))
        print(f"[INFO] supervised trained on {len(train_tuples)} tuples")

    if args.baseline_type == "b7_bandit":
        with open(args.training_data, encoding="utf-8") as f:
            td = json.load(f)
        tuples = td["tuples"]
        train_tuples, _ = bandit_split(tuples, train_ratio=args.train_ratio, seed=args.seed)
        states_tr, actions_tr, rewards_tr = bandit_tuples_to_arrays(train_tuples)
        bandit = ContextualBandit(model_type="random_forest", random_state=args.model_random_state)
        bandit.train(states_tr, actions_tr, rewards_tr)
        print(f"[INFO] bandit trained on {len(train_tuples)} tuples")

    evaluator_config_hashes = {ev.name: ev.evaluator_class_hash() for ev in evaluators}
    evaluator_severity_versions = {ev.name: _get_severity_version(ev) for ev in evaluators}
    tool_class_hashes = {name: tool.tool_class_hash() for name, tool in TOOL_BY_NAME.items()}

    raw_dir = Path(args.raw_output_dir).resolve() if args.raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for npy_path, g2_meta, motion in g2_motions:
        trial_id = g2_meta.get("trial_id", npy_path.stem)
        reports_before = {ev.name: ev.evaluate(motion) for ev in evaluators}
        target_before = _target_score_all(reports_before)

        # Apply baseline.
        if args.baseline_type == "b1_no_refinement":
            refined, trace = _apply_b1_noop(motion)
        elif args.baseline_type == "b2_fixed_smoothing":
            refined, trace = _apply_b2_fixed_smoothing(motion)
        elif args.baseline_type == "b5_rule_based":
            refined, trace = _apply_b5_rule_based(motion, evaluators)
        elif args.baseline_type == "b6_single":
            refined, trace = _apply_b6_single(motion, evaluators, supervised_selector, artifact_kinds_list)
        elif args.baseline_type == "b6_closed_loop":
            orch = SupervisedSelectorOrchestrator(selector=supervised_selector,
                                                  stop_score_threshold=args.stop_score_threshold,
                                                  target_evaluators=list(ALL_EVALUATORS))
            refined, trace = _apply_closed_loop(motion, evaluators, orch, args.max_iterations, args.tolerance)
        elif args.baseline_type == "b7_bandit":
            orch = ContextualBanditOrchestrator(bandit=bandit)
            refined, trace = _apply_closed_loop(motion, evaluators, orch, args.max_iterations, args.tolerance)
        else:
            raise ValueError(args.baseline_type)

        reports_after = {ev.name: ev.evaluate(refined) for ev in evaluators}
        target_after = _target_score_all(reports_after)
        target_delta = target_after - target_before
        # FidelityLoss: G2 motion 에 clean GT 없음 → reference = original G2.
        fidelity_loss = _mpjpe(refined, motion)
        artifact_reduction = -target_delta
        n_calls = float(trace.get("tool_call_count", 0))
        cumulative_mag = float(trace.get("correction_magnitude", 0.0))
        netgain = artifact_reduction - alpha * fidelity_loss - beta * cumulative_mag - gamma * n_calls

        result = {
            "trial_id": trial_id,
            "g2_prompt": g2_meta.get("prompt", "")[:80],
            "motion_shape": list(motion.shape),
            "target_score_before": target_before,
            "target_score_after": target_after,
            "target_delta": target_delta,
            "fidelity_loss_protocol_b": fidelity_loss,
            "correction_magnitude": cumulative_mag,
            "tool_call_count": n_calls,
            "netgain": netgain,
            "trace": trace,
            "per_evaluator_before_max": {n: _max_score(reports_before.get(n, [])) for n in ALL_EVALUATORS},
            "per_evaluator_after_max": {n: _max_score(reports_after.get(n, [])) for n in ALL_EVALUATORS},
        }
        results.append(result)

        if raw_dir is not None:
            timestamp = _now_iso()
            record = {
                "schema_version": SCHEMA_VERSION, "record_type": RECORD_TYPE,
                "timestamp": timestamp, "task_id": args.task_id, "split_id": args.split_id,
                "trial_id": trial_id, "baseline_type": args.baseline_type,
                "sample_path": str(npy_path), "generator_id": g2_meta.get("generator_id"),
                "g2_prompt": g2_meta.get("prompt"),
                "g2_seed": g2_meta.get("seed"),
                "fidelity_loss_reference": "original_g2_motion",
                "target_score_definition": "mean(FootFloating_max, BoneLength_max, VelocityJitter_max)",
                "netgain_weight_status": "calibrated_protocol_a_v1",
                "netgain_weight_caveat": "α=5.0 calibration is from synthetic Protocol A — transfer to G2 Protocol B magnitudes is calibration-dependent.",
                "netgain_weights": dict(netgain_weights),
                "evaluator_config_hashes": evaluator_config_hashes,
                "evaluator_severity_versions": evaluator_severity_versions,
                "tool_class_hashes": tool_class_hashes,
                "selections": {"g2_natural_multi_artifact": result},
                "transfer_diagnostic_note": "본 실험은 single-step ceiling 의 G2 natural 재현 검증. 최종 우위 검증 아님.",
                "negative_result": False,
            }
            out = raw_dir / f"{timestamp}_{args.task_id}_{trial_id}.json"
            out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    ng = np.array([r["netgain"] for r in results], dtype=np.float64)
    td_ = np.array([r["target_delta"] for r in results], dtype=np.float64)
    fl = np.array([r["fidelity_loss_protocol_b"] for r in results], dtype=np.float64)
    n_calls_arr = np.array([r["tool_call_count"] for r in results], dtype=np.float64)
    # Per-evaluator reduction.
    eval_reduction: dict[str, list[float]] = {n: [] for n in ALL_EVALUATORS}
    for r in results:
        for n in ALL_EVALUATORS:
            eval_reduction[n].append(r["per_evaluator_before_max"][n] - r["per_evaluator_after_max"][n])
    # Tool/strength distribution.
    from collections import Counter
    tool_dist: Counter = Counter()
    strength_dist: Counter = Counter()
    for r in results:
        trace = r["trace"]
        if "decision_trace" in trace:
            # closed-loop: count first step의 tool.
            for d in trace["decision_trace"]:
                if d.get("tool"):
                    tool_dist[d["tool"]] += 1
                if d.get("strength"):
                    strength_dist[d["strength"]] += 1
        elif "tool" in trace and trace["tool"]:
            tool_dist[trace["tool"]] += 1
            strength_dist[trace.get("strength", "n/a")] += 1
        else:
            tool_dist["NONE"] += 1

    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "baseline_g2_natural_summary",
        "task_id": args.task_id, "split_id": args.split_id,
        "baseline_type": args.baseline_type,
        "n_samples": len(results),
        "fidelity_loss_reference": "original_g2_motion",
        "target_score_definition": "mean(FootFloating, BoneLength, VelocityJitter max scores)",
        "netgain_weight_status": "calibrated_protocol_a_v1",
        "netgain_weight_caveat": "α=5.0 calibration is from synthetic Protocol A — G2 Protocol B transfer caveat.",
        "transfer_diagnostic_note": "본 실험은 single-step ceiling 의 G2 natural 재현 검증.",
        "netgain": {
            "mean": float(ng.mean()), "median": float(np.median(ng)),
            "p25": float(np.percentile(ng, 25)), "p75": float(np.percentile(ng, 75)),
            "min": float(ng.min()), "max": float(ng.max()),
        },
        "target_delta": {"mean": float(td_.mean()), "median": float(np.median(td_))},
        "fidelity_loss_protocol_b": {"mean": float(fl.mean()), "median": float(np.median(fl))},
        "tool_call_count": {"mean": float(n_calls_arr.mean()), "median": float(np.median(n_calls_arr))},
        "tool_distribution": dict(tool_dist),
        "strength_distribution": dict(strength_dist),
        "per_evaluator_reduction_median": {
            n: float(np.median(np.array(eval_reduction[n]))) for n in ALL_EVALUATORS
        },
        "per_evaluator_reduction_mean": {
            n: float(np.array(eval_reduction[n]).mean()) for n in ALL_EVALUATORS
        },
        "trial_ids": [r["trial_id"] for r in results],
        "results": results,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"\n[OK] wrote summary to {args.output}")
        print(f"  baseline_type: {args.baseline_type}")
        print(f"  n={len(results)}")
        print(f"  NetGain median={summary['netgain']['median']:+.5f}, mean={summary['netgain']['mean']:+.5f}")
        print(f"  target_delta median={summary['target_delta']['median']:+.5f}")
        print(f"  tool distribution: {dict(tool_dist)}")
        print(f"  strength distribution: {dict(strength_dist)}")
        print(f"  per-evaluator reduction median: {summary['per_evaluator_reduction_median']}")


if __name__ == "__main__":
    main()
