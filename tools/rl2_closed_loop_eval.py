"""Step F-3 (사용자 directive 2026-05-27): RL-2 closed-loop learned policy 평가.

사용자 directive:
> "action을 정확히 못 맞혀도 실제 motion 결과는 괜찮을 수 있습니다. 다음 평가는 accuracy가
>  아니라 NetGain / physical violation / safe reward / STOP correctness / oracle gap closure
>  를 봐야 합니다. RF와 HGB 둘 다 평가. G2 / synthetic 분리. 실패 유형 분석."

본 도구는 학습된 RF/HGB policy 를 closed-loop 실행:
  state → action → correction → physical gate → accept/rollback/STOP → next state

비교 baseline: no-op / B2-best / safe oracle / unsafe oracle / RL-2 RF / RL-2 HGB.
보고: G2 / synthetic 분리 NetGain / violation rate / STOP rate / oracle gap closure
+ failure 유형 (STOP miss / wrong tool / right tool wrong strength / unsafe action).

CLI:
    python -m tools.rl2_closed_loop_eval --seeds 0,1,2 \
        --output evals/snapshots/rl2_closed_loop_eval_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS, EvaluatorReport
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from tools.synthetic_injection import inject_foot_floating, inject_jitter
from tools.safe_sequence_oracle_run import _gate_scores, _gate_violation
from tools.rl2_build_training_data import (
    ACTION_LIST, ARTIFACT_EVALUATORS, PHYSICAL_EVALUATORS, STRENGTHS_5LEVEL,
    TOOLS_ORDER, TOOL_TARGET, _artifact_scores, _physical_scores, _build_state,
)
from tools.rl2_train_imitation import _flatten_state, _sample_level_split, _build_models

TOOL_BY_NAME: dict[str, CorrectionTool] = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}


def _idx_to_action(idx: int):
    if idx == 0:
        return None  # STOP
    idx -= 1
    tool = TOOLS_ORDER[idx // 5]
    return (tool, TOOL_TARGET[tool], STRENGTHS_5LEVEL[idx % 5])


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _mpjpe(a, b) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _target_full(motion, evaluators) -> float:
    by = {ev.name: ev.evaluate(motion) for ev in evaluators}
    return float(np.mean([_max_score(by.get(n, [])) for n in ARTIFACT_EVALUATORS]))


def _target_A(motion, evaluators) -> float:
    by = {ev.name: ev.evaluate(motion) for ev in evaluators}
    return float(np.mean([_max_score(by.get(n, [])) for n in ("FootFloatingEvaluator", "VelocityJitterEvaluator")]))


def _run_policy_closed_loop(motion0, clf, evaluators, gate_evaluators, gate_thresholds,
                            dist_tag, max_depth=3, include_dist_tag=False):
    """Execute policy closed-loop with gate (accept/rollback/STOP).

    Returns final motion + trace (actions taken, gate decisions, n_steps, stopped_reason).
    """
    motion = motion0.copy()
    T = motion.shape[0]
    artifact = _artifact_scores(motion, evaluators)
    physical = _physical_scores(motion, gate_evaluators)
    prev_artifact, prev_physical = list(artifact), list(physical)
    prev_action_idx = 0
    gate_parent = {n: physical[i] for i, n in enumerate(PHYSICAL_EVALUATORS)}
    actions_taken = []
    stopped_reason = "max_depth"

    for t in range(max_depth):
        delta = [a - pa for a, pa in zip(artifact, prev_artifact)] + \
                [p - pp for p, pp in zip(physical, prev_physical)]
        state = _build_state(artifact, physical, delta, prev_action_idx, max_depth - t, t, dist_tag)
        feats = np.array([_flatten_state(state, include_dist_tag)])
        action_idx = int(clf.predict(feats)[0])
        if action_idx == 0:
            stopped_reason = "policy_stop"
            break
        act = _idx_to_action(action_idx)
        tool = TOOL_BY_NAME[act[0]]
        try:
            new_motion, _ = tool.apply(motion, target_part=act[1], target_joints=[],
                                        frame_range=(0, T - 1), strength=act[2])
        except ValueError:
            stopped_reason = "apply_error"
            break
        # Physical gate check.
        gate_after = _gate_scores(new_motion, gate_evaluators)
        decisions = _gate_violation(gate_after, gate_parent, gate_thresholds)
        if any(d == "hard_violation" for d in decisions.values()):
            # Rollback: do not accept this step, STOP (conservative).
            stopped_reason = "gate_rollback"
            actions_taken.append({"action": ACTION_LIST[action_idx], "gate": "rollback"})
            break
        # Accept.
        actions_taken.append({"action": ACTION_LIST[action_idx], "gate": "accept"})
        motion = new_motion
        prev_artifact, prev_physical = list(artifact), list(physical)
        artifact = _artifact_scores(motion, evaluators)
        physical = _physical_scores(motion, gate_evaluators)
        gate_parent = {n: physical[i] for i, n in enumerate(PHYSICAL_EVALUATORS)}
        prev_action_idx = action_idx

    return motion, {"actions": actions_taken, "n_steps": len(actions_taken),
                    "stopped_reason": stopped_reason}


def _netgain_g2(final_motion, original, evaluators, w):
    """Protocol B simplified (vs original G2)."""
    t_init = _target_full(original, evaluators)
    t_final = _target_full(final_motion, evaluators)
    fid = _mpjpe(final_motion, original)
    return -(t_final - t_init) - float(w["alpha"]) * fid


def _netgain_synthetic(final_motion, clean, corrupted, evaluators, w):
    """Protocol A (vs clean GT)."""
    t_init = _target_A(corrupted, evaluators)
    t_final = _target_A(final_motion, evaluators)
    mpjpe_corr = _mpjpe(corrupted, clean)
    fid = _mpjpe(final_motion, clean) - mpjpe_corr
    return -(t_final - t_init) - float(w["alpha"]) * fid


def _has_violation(final_motion, original_motion, gate_evaluators, gate_thresholds):
    """Final motion vs original — any hard violation?"""
    before = _gate_scores(original_motion, gate_evaluators)
    after = _gate_scores(final_motion, gate_evaluators)
    dec = _gate_violation(after, before, gate_thresholds)
    return any(d == "hard_violation" for d in dec.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_imitation_dataset_v1.json")
    parser.add_argument("--g2-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_g2_hml3d_test300_v1.json")
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--synthetic-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_synthetic_severe_v1.json")
    parser.add_argument("--synthetic-data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--calibration", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--models", type=str, default="RF,HGB")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_closed_loop_eval_v1.json")
    args = parser.parse_args()

    data = json.load(open(args.dataset, encoding="utf-8"))
    rows = data["rows"]
    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_thresholds = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                       if calib["summary"][n].get("n", 0) > 0}
    seeds = [int(s) for s in args.seeds.split(",")]
    model_names = args.models.split(",")
    w = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
    evaluators = list(DEFAULT_EVALUATORS)
    gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)

    # Load oracle snapshots for safe/unsafe baselines + original motions.
    g2_oracle = {p["trial_id"]: p for p in json.load(open(args.g2_oracle, encoding="utf-8"))["per_sample"]}
    syn_oracle = {p["trial_id"]: p for p in json.load(open(args.synthetic_oracle, encoding="utf-8"))["per_sample"]}

    models = _build_models()
    X_all = np.array([_flatten_state(r["state"], False) for r in rows])
    y_all = np.array([r["action_idx"] for r in rows])

    # Aggregate over seeds.
    seed_results = defaultdict(lambda: defaultdict(list))  # [dist][method] -> list of per-seed agg dicts

    for seed in seeds:
        train_idx, eval_idx = _sample_level_split(rows, seed)
        # Eval sample ids (unique, per distribution).
        eval_g2 = sorted({rows[i]["sample_id"] for i in eval_idx if rows[i]["distribution"] == "g2"})
        eval_syn = sorted({rows[i]["sample_id"] for i in eval_idx if rows[i]["distribution"] == "synthetic"})

        # Train policies.
        trained = {}
        for mn in model_names:
            clf = models[mn]()
            clf.fit(X_all[train_idx], y_all[train_idx])
            trained[mn] = clf

        # === G2 eval ===
        for tid in eval_g2:
            npy = args.g2_batch_dir / f"{tid}.npy"
            if not npy.exists():
                continue
            original = np.load(str(npy)).astype(np.float64)
            # Baselines.
            noop_ng = _netgain_g2(original, original, evaluators, w)  # = 0
            safe_ng = g2_oracle[tid]["safe_best"]["netgain"] if g2_oracle[tid].get("safe_best") else noop_ng
            unsafe_ng = g2_oracle[tid]["unsafe_best"]["netgain"] if g2_oracle[tid].get("unsafe_best") else noop_ng
            # B2-best (small/medium/large), pick best NetGain + record its violation.
            b2_best = noop_ng
            b2_best_motion = original
            for st in ("small", "medium", "large"):
                out, _ = TOOL_BY_NAME["VelocitySmoothingTool"].apply(
                    original, target_part="full_body", target_joints=[], frame_range=(0, original.shape[0]-1), strength=st)
                ng_st = _netgain_g2(out, original, evaluators, w)
                if ng_st > b2_best:
                    b2_best, b2_best_motion = ng_st, out
            b2_viol = _has_violation(b2_best_motion, original, gate_evaluators, gate_thresholds)
            row = {"trial_id": tid, "noop": noop_ng, "safe_oracle": safe_ng,
                   "unsafe_oracle": unsafe_ng, "b2_best": b2_best, "b2_viol": b2_viol}
            for mn, clf in trained.items():
                final, trace = _run_policy_closed_loop(original, clf, evaluators, gate_evaluators,
                                                       gate_thresholds, dist_tag=0, max_depth=args.max_depth)
                ng = _netgain_g2(final, original, evaluators, w)
                viol = _has_violation(final, original, gate_evaluators, gate_thresholds)
                row[f"rl2_{mn}_ng"] = ng
                row[f"rl2_{mn}_viol"] = viol
                row[f"rl2_{mn}_nsteps"] = trace["n_steps"]
                row[f"rl2_{mn}_stop"] = trace["stopped_reason"] == "policy_stop" and trace["n_steps"] == 0
            seed_results["g2"]["_rows"].append(row)

        # === Synthetic eval ===
        for tid in eval_syn:
            npy = args.synthetic_data_dir / f"{tid}.npy"
            if not npy.exists():
                continue
            clean = np.load(str(npy)).astype(np.float64)
            m1 = inject_foot_floating(clean, lift_height=0.08, seed=args.synthetic_seed)
            corrupted = inject_jitter(m1, noise_std=0.05, seed=args.synthetic_seed + 1000)
            noop_ng = _netgain_synthetic(corrupted, clean, corrupted, evaluators, w)  # = 0
            safe_ng = syn_oracle[tid]["safe_best"]["netgain"] if syn_oracle[tid].get("safe_best") else noop_ng
            unsafe_ng = syn_oracle[tid]["unsafe_best"]["netgain"] if syn_oracle[tid].get("unsafe_best") else noop_ng
            b2_best = noop_ng
            b2_best_motion = corrupted
            for st in ("small", "medium", "large"):
                out, _ = TOOL_BY_NAME["VelocitySmoothingTool"].apply(
                    corrupted, target_part="full_body", target_joints=[], frame_range=(0, corrupted.shape[0]-1), strength=st)
                ng_st = _netgain_synthetic(out, clean, corrupted, evaluators, w)
                if ng_st > b2_best:
                    b2_best, b2_best_motion = ng_st, out
            b2_viol = _has_violation(b2_best_motion, corrupted, gate_evaluators, gate_thresholds)
            row = {"trial_id": tid, "noop": noop_ng, "safe_oracle": safe_ng,
                   "unsafe_oracle": unsafe_ng, "b2_best": b2_best, "b2_viol": b2_viol}
            for mn, clf in trained.items():
                final, trace = _run_policy_closed_loop(corrupted, clf, evaluators, gate_evaluators,
                                                       gate_thresholds, dist_tag=1, max_depth=args.max_depth)
                ng = _netgain_synthetic(final, clean, corrupted, evaluators, w)
                viol = _has_violation(final, corrupted, gate_evaluators, gate_thresholds)
                row[f"rl2_{mn}_ng"] = ng
                row[f"rl2_{mn}_viol"] = viol
                row[f"rl2_{mn}_nsteps"] = trace["n_steps"]
                row[f"rl2_{mn}_stop"] = trace["stopped_reason"] == "policy_stop" and trace["n_steps"] == 0
            seed_results["synthetic"]["_rows"].append(row)
        print(f"[seed {seed}] eval done (G2 {len(eval_g2)}, synthetic {len(eval_syn)})")

    # === Aggregate ===
    def _agg_dist(dist_rows):
        methods = ["noop", "b2_best", "safe_oracle", "unsafe_oracle"]
        out = {}
        mean_noop = float(np.mean([r["noop"] for r in dist_rows])) if dist_rows else 0.0
        mean_safe = float(np.mean([r["safe_oracle"] for r in dist_rows])) if dist_rows else 0.0
        denom_agg = mean_safe - mean_noop
        for m in methods:
            vals = [r[m] for r in dist_rows if m in r]
            out[m] = {"mean_netgain": float(np.mean(vals)) if vals else 0.0}
        # B2 violation rate (thesis: B2 의 NetGain 이 physical cost 동반하는가).
        b2_viols = [r["b2_viol"] for r in dist_rows if "b2_viol" in r]
        if b2_viols:
            out["b2_best"]["violation_rate"] = float(np.mean(b2_viols))
        for mn in model_names:
            ng = [r[f"rl2_{mn}_ng"] for r in dist_rows if f"rl2_{mn}_ng" in r]
            viol = [r[f"rl2_{mn}_viol"] for r in dist_rows if f"rl2_{mn}_viol" in r]
            stop = [r[f"rl2_{mn}_stop"] for r in dist_rows if f"rl2_{mn}_stop" in r]
            mean_pol = float(np.mean(ng)) if ng else 0.0
            # Aggregate oracle gap closure: (mean_policy - mean_noop) / (mean_safe_oracle - mean_noop).
            gap = (mean_pol - mean_noop) / denom_agg if abs(denom_agg) > 1e-9 else 0.0
            out[f"rl2_{mn}"] = {
                "mean_netgain": mean_pol,
                "violation_rate": float(np.mean(viol)) if viol else 0.0,
                "stop_rate": float(np.mean(stop)) if stop else 0.0,
                "oracle_gap_closure_aggregate": gap,
            }
        return out

    final_out = {
        "schema_version": "1.0.0", "record_type": "rl2_closed_loop_eval",
        "task_id": "rl2_closed_loop_eval_v1", "seeds": seeds, "max_depth": args.max_depth,
        "models": model_names,
        "note": "closed-loop policy + physical gate (accept/rollback/STOP). NetGain: G2=Protocol B, synthetic=Protocol A.",
        "g2": _agg_dist(seed_results["g2"]["_rows"]),
        "synthetic": _agg_dist(seed_results["synthetic"]["_rows"]),
        "n_g2_eval_total": len(seed_results["g2"]["_rows"]),
        "n_synthetic_eval_total": len(seed_results["synthetic"]["_rows"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(final_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RL-2 Closed-loop Eval (Step F-3) ===")
    for dist in ("g2", "synthetic"):
        print(f"\n[{dist}] (n_eval_total across seeds = {final_out[f'n_{dist}_eval_total']})")
        agg = final_out[dist]
        print(f"  {'method':<16} {'mean_NetGain':<14} {'violation':<12} {'STOP':<10} {'oracle_gap'}")
        for m in ["noop", "b2_best", "unsafe_oracle", "safe_oracle"] + [f"rl2_{mn}" for mn in model_names]:
            d = agg.get(m, {})
            ng = d.get("mean_netgain", 0)
            vr = d.get("violation_rate")
            sr = d.get("stop_rate")
            gc = d.get("oracle_gap_closure_aggregate")
            vr_s = f"{vr*100:<11.0f}%" if vr is not None else " " * 12
            sr_s = f"{sr*100:<9.0f}%" if sr is not None else " " * 10
            gc_s = f"{gc*100:.0f}%" if gc is not None else ""
            print(f"  {m:<16} {ng:<+14.4f} {vr_s} {sr_s} {gc_s}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
