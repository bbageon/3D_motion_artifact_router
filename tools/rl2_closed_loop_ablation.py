"""Step 6 (사용자 directive 2026-05-31): RL-2 closed-loop real-gate-ON ablation.

사용자 directive 박제:
> "Step 6 은 진행. M0 primary, M1/M2/M3 는 ablation. 핵심 질문: M0+gate 가 random+gate /
>  heuristic+gate 보다 좋은가? gate 가 안전 보장 + candidate_evals / rejection_rate 가
>  낮아야 좋은 정책. group-wise 결과 확인."

본 도구는 4 holdout state 들에 대해 closed-loop policy + real physical gate (regression-based
hard violation → rollback + STOP) 실행. baselines:
  - STOP-only           — 아무것도 안 함
  - random+gate         — uniform (tool, u) 후보 (가능 시 retry K 회), gate 체크
  - heuristic+gate      — dominant artifact 기반 rule (FootFloating → FootLock medium 등)
  - M0/M1/M2/M3 + gate  — Q 정책 reranking (P_safe ≥ 0.5 argmax utility)
  - dense_oracle_step1  — initial-grid 의 best safe (Step 3 transitions lookup, 단일 step ceiling)

metric per (model, holdout):
  physical_violation_rate, mean NetGain (G2=Protocol B, clean/syn=Protocol A),
  over_STOP_rate, candidate_evals_mean, gate_rejection_rate, act_rate,
  group_breakdown (G2 holdouts 의 motion_group).

NOTE: standard metric (FID / R-Prec / MM-Dist) 은 mgpt env 필요 — 별도 도구 (part 2).

CLI:
    python -m tools.rl2_closed_loop_ablation \
        --models evals/models/rl2_q_policies_stage2_v1.joblib \
        --split evals/splits/g2_real_stress_split_v2.json \
        --g2-real evals/snapshots/rl2_transition_g2_real_stage2_v1.json \
        --max-depth 3 --random-retries 3 \
        --output evals/snapshots/rl2_closed_loop_ablation_stage2_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS, EvaluatorReport
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from tools.synthetic_injection import inject_foot_floating, inject_jitter
from tools.safe_sequence_oracle_run import _gate_scores, _gate_violation
from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, REAL_DISTRIBUTION, common_snapshot_metadata
from tools.rl2_build_training_data import ARTIFACT_EVALUATORS, PHYSICAL_EVALUATORS, TOOL_TARGET
from tools.rl2_transition_build import TOOLS_ORDER
from tools.rl2_transition_g2_real_build import _resolve_g2
from tools.rl2_q_policy_train import _cand_features_from_state
from tools.rl2_offline_generalization_eval import _predict_safe

TOOL_BY_NAME: dict[str, CorrectionTool] = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}
TOOL_IDX = {t: i for i, t in enumerate(TOOLS_ORDER)}
HOLDOUTS = ("g2_stress_holdout", "g2_natural_holdout", "clean_noharm_holdout", "synthetic_diag_holdout")
TARGET_EVALUATORS_A = ("FootFloatingEvaluator", "VelocityJitterEvaluator")
PROTOCOL_A_DISTS = {"clean", "synthetic_severe"}


def _max_score(reports): return float(max((r.score for r in reports), default=0.0))


def _scores(motion, evaluators, names):
    by = {ev.name: ev.evaluate(motion) for ev in evaluators}
    return {n: round(_max_score(by.get(n, [])), 6) for n in names}


def _mpjpe(a, b): return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _target_full(scores): return float(np.mean([scores[n] for n in ARTIFACT_EVALUATORS]))
def _target_A(scores): return float(np.mean([scores[n] for n in TARGET_EVALUATORS_A]))


def _u_to_params(tool, u):
    if tool == "VelocitySmoothingTool":
        return {"continuous_sigma": 2.0 * u}
    return {"continuous_factor": u}


def _apply(motion, tool, u, frame_range):
    return TOOL_BY_NAME[tool].apply(motion, target_part=TOOL_TARGET[tool], target_joints=[],
                                    frame_range=frame_range, strength="medium", metadata=_u_to_params(tool, u))


def _state_dict_from_motion(motion, evaluators, gate_evaluators):
    art = _scores(motion, evaluators, ARTIFACT_EVALUATORS)
    phy = _scores(motion, gate_evaluators, PHYSICAL_EVALUATORS)
    return {"artifact_scores": art, "physical_scores": phy}


def _gate_decision(motion_after, gate_parent, gate_evaluators, gate_thresholds):
    after_scores = _gate_scores(motion_after, gate_evaluators)
    dec = _gate_violation(after_scores, gate_parent, gate_thresholds)
    return after_scores, dec, any(d == "hard_violation" for d in dec.values())


def _netgain(motion_before, motion_after, reference, dist, evaluators, alpha):
    """G2=Protocol B (vs motion_before), clean/syn=Protocol A (vs reference clean)."""
    if dist == "g2_natural":
        target_b = _target_full(_scores(motion_before, evaluators, ARTIFACT_EVALUATORS))
        target_a = _target_full(_scores(motion_after, evaluators, ARTIFACT_EVALUATORS))
        fid = _mpjpe(motion_after, motion_before)
    else:
        target_b = _target_A(_scores(motion_before, evaluators, TARGET_EVALUATORS_A))
        target_a = _target_A(_scores(motion_after, evaluators, TARGET_EVALUATORS_A))
        fid = _mpjpe(motion_after, reference) - _mpjpe(motion_before, reference)
    return -(target_a - target_b) - alpha * fid


# === Baselines ===
def _stop_only(motion0, *args, **kwargs):
    return {"final_motion": motion0, "actions": [], "candidate_evals": 0,
            "gate_rejections": 0, "stop_reason": "policy_stop"}


def _random_gate(motion0, evaluators, gate_evaluators, gate_thresholds, u_grid, rng,
                 max_depth=3, retries_per_step=3):
    motion = motion0.copy(); T = motion.shape[0]
    actions, cand_evals, gate_rej = [], 0, 0
    gate_parent = _gate_scores(motion, gate_evaluators)
    stop_reason = "max_depth"
    for step in range(max_depth):
        accepted = False
        for _ in range(retries_per_step):
            tool = TOOLS_ORDER[int(rng.integers(len(TOOLS_ORDER)))]
            u = float(rng.choice(u_grid))
            cand_evals += 1
            try:
                new_motion, _ = _apply(motion, tool, u, (0, T - 1))
            except ValueError:
                continue
            after_scores, decision, violated = _gate_decision(new_motion, gate_parent, gate_evaluators, gate_thresholds)
            if violated:
                gate_rej += 1; continue
            actions.append({"tool": tool, "u": u, "step": step, "policy": "random"})
            motion = new_motion; gate_parent = after_scores
            accepted = True; break
        if not accepted:
            stop_reason = "random_all_rejected"; break
    return {"final_motion": motion, "actions": actions, "candidate_evals": cand_evals,
            "gate_rejections": gate_rej, "stop_reason": stop_reason}


def _heuristic_gate(motion0, evaluators, gate_evaluators, gate_thresholds, u_grid_unused,
                    rng_unused, max_depth=3):
    """Rule: dominant artifact → matching tool at u=0.5; STOP if all artifact small."""
    motion = motion0.copy(); T = motion.shape[0]
    actions, cand_evals, gate_rej = [], 0, 0
    gate_parent = _gate_scores(motion, gate_evaluators)
    rule = {"FootFloatingEvaluator": "FootLockTool", "VelocityJitterEvaluator": "VelocitySmoothingTool",
            "BoneLengthEvaluator": "BoneProjectionTool"}
    stop_reason = "max_depth"
    THR = 0.02  # artifact dominant threshold (heuristic).
    for step in range(max_depth):
        art = _scores(motion, evaluators, ARTIFACT_EVALUATORS)
        dom = max(ARTIFACT_EVALUATORS, key=lambda n: art[n])
        if art[dom] < THR:
            stop_reason = "policy_stop"; break
        tool = rule[dom]; u = 0.5
        cand_evals += 1
        try:
            new_motion, _ = _apply(motion, tool, u, (0, T - 1))
        except ValueError:
            stop_reason = "apply_error"; break
        after_scores, _, violated = _gate_decision(new_motion, gate_parent, gate_evaluators, gate_thresholds)
        if violated:
            gate_rej += 1; stop_reason = "gate_rollback"; break
        actions.append({"tool": tool, "u": u, "step": step, "policy": "heuristic"})
        motion = new_motion; gate_parent = after_scores
    return {"final_motion": motion, "actions": actions, "candidate_evals": cand_evals,
            "gate_rejections": gate_rej, "stop_reason": stop_reason}


def _q_policy_gate(motion0, util_head, safe_head, iso, evaluators, gate_evaluators,
                   gate_thresholds, u_grid, max_depth=3, p_safe_threshold=0.5, stop_epsilon=0.0):
    motion = motion0.copy(); T = motion.shape[0]
    actions, cand_evals, gate_rej = [], 0, 0
    gate_parent = _gate_scores(motion, gate_evaluators)
    stop_reason = "max_depth"
    for step in range(max_depth):
        state = _state_dict_from_motion(motion, evaluators, gate_evaluators)
        # Predict all candidates (3 tools × |u_grid|).
        cand = [(t, u) for t in TOOLS_ORDER for u in u_grid]
        X = np.array([_cand_features_from_state(state, t, u) for t, u in cand])
        pu = util_head.predict(X)
        ps = _predict_safe(safe_head, iso, X)
        safe_mask = ps >= p_safe_threshold
        if not safe_mask.any():
            stop_reason = "all_predicted_unsafe"; break
        cand_idx = np.where(safe_mask)[0]
        best = cand_idx[int(np.argmax(pu[cand_idx]))]
        if pu[best] < stop_epsilon:
            stop_reason = "policy_stop"; break
        tool, u = cand[best]
        cand_evals += 1
        try:
            new_motion, _ = _apply(motion, tool, u, (0, T - 1))
        except ValueError:
            stop_reason = "apply_error"; break
        after_scores, _, violated = _gate_decision(new_motion, gate_parent, gate_evaluators, gate_thresholds)
        if violated:
            gate_rej += 1; stop_reason = "gate_rollback"; break
        actions.append({"tool": tool, "u": u, "step": step, "policy": "Q",
                        "pred_util": float(pu[best]), "pred_safe": float(ps[best])})
        motion = new_motion; gate_parent = after_scores
    return {"final_motion": motion, "actions": actions, "candidate_evals": cand_evals,
            "gate_rejections": gate_rej, "stop_reason": stop_reason}


def _dense_oracle_step1(motion0, state_transitions, dist, reference, evaluators, alpha):
    """Step-1 oracle: pick best safe (real gate) actual utility from initial grid (transitions lookup)."""
    safe = [t for t in state_transitions if t["gate_result"] == "pass" and t["safe_utility"] > 0]
    if not safe:
        return {"final_motion": motion0, "actions": [], "candidate_evals": 0,
                "gate_rejections": 0, "stop_reason": "policy_stop"}
    best = max(safe, key=lambda t: t["safe_utility"])
    # Apply to get final motion.
    new_motion, _ = _apply(motion0, best["tool"], best["u"], (0, motion0.shape[0] - 1))
    return {"final_motion": new_motion, "actions": [{"tool": best["tool"], "u": best["u"], "step": 0, "policy": "oracle_step1"}],
            "candidate_evals": 1, "gate_rejections": 0, "stop_reason": "oracle_single_step"}


def _load_state_motion(state_id, dist, args):
    if dist == "g2_natural":
        pool_dir, stem = _resolve_g2(state_id, args.existing_pool, args.balanced_pool)
        return np.load(str(pool_dir / f"{stem}.npy")).astype(np.float64), None
    elif dist == "clean":
        m = np.load(str(args.data_dir / f"{state_id}.npy")).astype(np.float64)
        return m, m
    elif dist == "synthetic_severe":
        clean = np.load(str(args.data_dir / f"{state_id}.npy")).astype(np.float64)
        m1 = inject_foot_floating(clean, lift_height=0.08, seed=args.synthetic_seed)
        return inject_jitter(m1, noise_std=0.05, seed=args.synthetic_seed + 1000), clean
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path,
                        default=REPO_ROOT / "evals" / "models" / "rl2_q_policies_stage2_v1.joblib")
    parser.add_argument("--split", type=Path,
                        default=REPO_ROOT / "evals" / "splits" / "g2_real_stress_split_v2.json")
    parser.add_argument("--g2-real", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_g2_real_stage2_v1.json")
    parser.add_argument("--existing-pool", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--balanced-pool", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_balanced300_seed20260531")
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--calibration", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    parser.add_argument("--u-grid", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--random-retries", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=20260531)
    parser.add_argument("--p-safe-threshold", type=float, default=0.5)
    parser.add_argument("--stop-epsilon", type=float, default=0.0)
    parser.add_argument("--save-final-motions", action="store_true",
                        help="closed-loop final motions 저장 (Step 6 part 2 standard metric 입력).")
    parser.add_argument("--final-motion-dir", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "closed_loop_final_motions_v1")
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_closed_loop_ablation_stage2_v1.json")
    args = parser.parse_args()

    u_grid = [round(float(x), 4) for x in args.u_grid.split(",")]
    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_thresholds = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                       if calib["summary"][n].get("n", 0) > 0}
    alpha = float(CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1["alpha"])
    evaluators = list(DEFAULT_EVALUATORS); gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)

    import joblib
    models_bundle = joblib.load(str(args.models))
    print(f"[INFO] models loaded: {list(models_bundle.keys())}")

    split = json.load(open(args.split, encoding="utf-8"))
    g2_data = json.load(open(args.g2_real, encoding="utf-8"))
    state_map = g2_data["states"]
    transitions_by_state = defaultdict(list)
    for t in g2_data["transitions"]:
        transitions_by_state[t["state_id"]].append(t)

    # Holdout state lists.
    holdout_states = {ho: [sid for sid in state_map if state_map[sid]["split"] == ho] for ho in HOLDOUTS}
    print(f"[INFO] holdouts: { {k: len(v) for k, v in holdout_states.items()} }")

    # Policies dict (M0~M3 + baselines).
    baselines = {"STOP_only", "random_gate", "heuristic_gate", "dense_oracle_step1"}
    all_policies = list(models_bundle.keys()) + ["STOP_only", "random_gate", "heuristic_gate", "dense_oracle_step1"]

    if args.save_final_motions:
        args.final_motion_dir.mkdir(parents=True, exist_ok=True)

    results = {p: {} for p in all_policies}

    for ho in HOLDOUTS:
        print(f"\n[INFO] running {ho} ({len(holdout_states[ho])} states)")
        per_state_outcomes = {p: [] for p in all_policies}
        for i, sid in enumerate(holdout_states[ho], 1):
            sinfo = state_map[sid]
            dist = sinfo["distribution"]
            motion0, reference = _load_state_motion(sid, dist, args)
            if motion0 is None:
                continue
            mg = sinfo.get("motion_group", "?")
            common_ref = reference if dist in PROTOCOL_A_DISTS else motion0

            # Baselines.
            outs = {}
            outs["STOP_only"] = _stop_only(motion0)
            rng = np.random.default_rng(args.random_seed + hash(sid) % (2**31))
            outs["random_gate"] = _random_gate(motion0, evaluators, gate_evaluators, gate_thresholds,
                                                u_grid, rng, args.max_depth, args.random_retries)
            outs["heuristic_gate"] = _heuristic_gate(motion0, evaluators, gate_evaluators, gate_thresholds,
                                                     None, None, args.max_depth)
            outs["dense_oracle_step1"] = _dense_oracle_step1(motion0, transitions_by_state[sid], dist,
                                                              common_ref, evaluators, alpha)
            # Q policies.
            for mname, (util, safe, iso) in models_bundle.items():
                outs[mname] = _q_policy_gate(motion0, util, safe, iso, evaluators, gate_evaluators,
                                              gate_thresholds, u_grid, args.max_depth,
                                              args.p_safe_threshold, args.stop_epsilon)
            # Compute metrics per policy.
            for pname, out in outs.items():
                # NetGain (final vs initial).
                ng = _netgain(motion0, out["final_motion"], common_ref, dist, evaluators, alpha)
                # Physical violation (final motion vs initial gate).
                init_gate = _gate_scores(motion0, gate_evaluators)
                _, _, final_violated = _gate_decision(out["final_motion"], init_gate,
                                                       gate_evaluators, gate_thresholds)
                # over_STOP: oracle has positive utility but policy never acted.
                oracle_pos = max((t["safe_utility"] for t in transitions_by_state[sid]
                                  if t["gate_result"] == "pass" and t["safe_utility"] > 0), default=0.0)
                over_stop = (len(out["actions"]) == 0) and (oracle_pos > 1e-6)
                per_state_outcomes[pname].append({
                    "state_id": sid, "motion_group": mg, "distribution": dist,
                    "netgain": ng, "physical_violation": final_violated,
                    "n_actions": len(out["actions"]), "acted": len(out["actions"]) > 0,
                    "candidate_evals": out["candidate_evals"], "gate_rejections": out["gate_rejections"],
                    "stop_reason": out["stop_reason"], "over_stop": over_stop,
                    "oracle_step1_util_safe": oracle_pos,
                })
                if args.save_final_motions:
                    np.save(str(args.final_motion_dir / f"{ho}__{pname}__{sid.replace('/', '_')}.npy"),
                            out["final_motion"].astype(np.float32))
            if i % 20 == 0:
                print(f"   {ho} {i}/{len(holdout_states[ho])}")

        # Aggregate per holdout.
        for pname, rows in per_state_outcomes.items():
            if not rows: continue
            def _ms(field, default=0.0):
                vals = [r[field] for r in rows]
                if isinstance(vals[0], bool):
                    vals = [float(v) for v in vals]
                return float(np.mean(vals)) if vals else default
            agg = {
                "n_states": len(rows),
                "mean_netgain": _ms("netgain"),
                "physical_violation_rate": _ms("physical_violation"),
                "act_rate": _ms("acted"),
                "over_stop_rate": _ms("over_stop"),
                "mean_candidate_evals": _ms("candidate_evals"),
                "mean_gate_rejections": _ms("gate_rejections"),
                "gate_rejection_rate_per_eval": (
                    float(np.sum([r["gate_rejections"] for r in rows])) /
                    max(float(np.sum([r["candidate_evals"] for r in rows])), 1.0)
                ),
                "mean_oracle_step1_util": _ms("oracle_step1_util_safe"),
            }
            # Group-wise (G2).
            if ho.startswith("g2_"):
                by_g = defaultdict(list)
                for r in rows:
                    by_g[r["motion_group"]].append(r)
                agg["group_breakdown"] = {
                    g: {"n": len(grows),
                        "mean_netgain": float(np.mean([r["netgain"] for r in grows])),
                        "physical_violation_rate": float(np.mean([r["physical_violation"] for r in grows])),
                        "act_rate": float(np.mean([r["acted"] for r in grows])),
                        "over_stop_rate": float(np.mean([r["over_stop"] for r in grows])),
                        }
                    for g, grows in by_g.items()
                }
            results[pname][ho] = agg

    out = {
        "schema_version": "1.0.0", "record_type": "rl2_closed_loop_ablation",
        "task_id": "rl2_closed_loop_ablation_stage2_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "rl2_closed_loop_ablation_stage2_v1",
            oracle_type="closed_loop_single_step_dense",
            action_grid="continuous-dense-u11", stage="Stage-2G-step6-closed-loop-ablation",
            evidence_tier=[REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
            evaluators=evaluators, gate_evaluators=gate_evaluators,
        ),
        "directive": "M0 primary, M1/M2/M3 ablation, random+gate / heuristic+gate / dense_oracle_step1 비교. real-gate-ON closed-loop.",
        "max_depth": args.max_depth, "random_retries": args.random_retries,
        "p_safe_threshold": args.p_safe_threshold, "stop_epsilon": args.stop_epsilon,
        "u_grid": u_grid, "alpha": alpha,
        "policies": all_policies,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Step 6 — Closed-Loop Real-Gate-ON Ablation ===")
    for ho in HOLDOUTS:
        print(f"\n[{ho}]")
        print(f"  {'policy':<24} {'NetGain':<10} {'phys_viol%':<11} {'act%':<7} {'over_STOP%':<11} {'cand_evals':<11} {'reject%'}")
        for pname in all_policies:
            r = results[pname].get(ho)
            if r is None: continue
            ng = r["mean_netgain"]; pv = r["physical_violation_rate"]; act = r["act_rate"]
            ost = r["over_stop_rate"]; ce = r["mean_candidate_evals"]; gr = r["gate_rejection_rate_per_eval"]
            print(f"  {pname:<24} {ng:<+10.4f} {pv*100:<10.1f}% {act*100:<6.0f}% {ost*100:<10.0f}% {ce:<11.2f} {gr*100:.1f}%")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
