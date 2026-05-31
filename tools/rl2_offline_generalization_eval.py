"""Step 5 (사용자 directive 2026-05-31): RL-2 offline generalization evaluation.

closed-loop 전에 held-out transition 에서 먼저 보는 단계 (사용자 명시).

평가 split (4 holdout):
  g2_stress_holdout     — 실제 generator artifact 를 고칠 수 있나?
  g2_natural_holdout    — 일반 생성 모션을 과보정하지 않나?
  clean_noharm_holdout  — 깨끗한 모션을 망치지 않나?
  synthetic_diag_holdout — appendix stress mechanism 재현?

핵심 metric (사용자 spec):
  topk_safe_recall      — 좋은 safe 후보 (oracle top-k) 를 찾는가
  unsafe_topk_rate      — 위험 후보를 상위에 올리는가 (false-safe)
  safe_utility_recovery — dense oracle 대비 회수율
  argmax_regret         — oracle 대비 손실
  over_stop_rate        — 너무 보수적인가 (oracle 은 act, policy 는 STOP)

평가 대상 model: M0 (Stage A) / M1 (G2 real) / M2 (G2+syn) / M3 (M2+isotonic).
G2 holdout 은 motion_group 별 break-down 동반.

CLI:
    python -m tools.rl2_offline_generalization_eval \
        --models evals/models/rl2_q_policies_stage2_v1.joblib \
        --g2-real evals/snapshots/rl2_transition_g2_real_stage2_v1.json \
        --output evals/snapshots/rl2_offline_gen_eval_stage2_v1.json
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

from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, REAL_DISTRIBUTION, common_snapshot_metadata
from tools.rl2_build_training_data import ARTIFACT_EVALUATORS, PHYSICAL_EVALUATORS
from tools.rl2_transition_build import TOOLS_ORDER
from tools.rl2_q_policy_train import _cand_features_from_state
from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS

HOLDOUTS = ("g2_stress_holdout", "g2_natural_holdout", "clean_noharm_holdout", "synthetic_diag_holdout")
STOP_UTILITY = 0.0  # u=0 / STOP baseline.


def _predict_safe(safe_head, iso, X):
    """P_safe predicted (with optional isotonic post-cal)."""
    if hasattr(safe_head, "predict_proba"):
        raw = safe_head.predict_proba(X)[:, 1]
    else:
        raw = safe_head.predict(X).astype(float)
    if iso is not None:
        return np.clip(iso.predict(raw), 0.0, 1.0)
    return raw


def _eval_state(state_dict, state_transitions, util_head, safe_head, iso,
                stop_epsilon=0.0, p_safe_threshold=0.5, topk=3):
    """state 의 모든 candidates 의 (Q,P) 예측 + oracle 대비 metric."""
    # Candidate features.
    cand = state_transitions  # already (tool, u) per row.
    X = np.array([_cand_features_from_state(state_dict, t["tool"], t["u"]) for t in cand])
    pu = util_head.predict(X)
    ps = _predict_safe(safe_head, iso, X)
    # Actual.
    actual_util = np.array([t["safe_utility"] for t in cand])
    actual_safe = np.array([t["gate_result"] == "pass" for t in cand])

    # Oracle = best safe actual utility (or STOP=0).
    safe_idx = np.where(actual_safe)[0]
    if len(safe_idx) > 0:
        oracle_util = max(STOP_UTILITY, float(actual_util[safe_idx].max()))
    else:
        oracle_util = STOP_UTILITY

    # Policy argmax: among P_safe>=threshold, max predicted util; if max < epsilon → STOP.
    pred_safe_mask = ps >= p_safe_threshold
    policy_acted = False; policy_pick = None; policy_actual_util = STOP_UTILITY
    policy_pick_actually_unsafe = False
    if pred_safe_mask.any():
        cand_idx = np.where(pred_safe_mask)[0]
        best = cand_idx[int(np.argmax(pu[cand_idx]))]
        if pu[best] >= stop_epsilon:
            policy_acted = True; policy_pick = int(best)
            policy_actual_util = float(actual_util[best])
            policy_pick_actually_unsafe = bool(not actual_safe[best])

    # Top-k predicted-safe by predicted utility.
    if pred_safe_mask.any():
        cand_idx = np.where(pred_safe_mask)[0]
        order = cand_idx[np.argsort(-pu[cand_idx])][:topk]
        topk_pred = list(order)
    else:
        topk_pred = []
    # Top-k true safe by actual utility.
    if len(safe_idx) > 0:
        topk_true = list(safe_idx[np.argsort(-actual_util[safe_idx])][:topk])
    else:
        topk_true = []
    # topk_safe_recall = |pred top-k ∩ true top-k| / max(|true top-k|, 1).
    intersect = len(set(topk_pred) & set(topk_true))
    topk_safe_recall = intersect / max(len(topk_true), 1)
    # unsafe in pred top-k: how many of pred top-k are actually unsafe.
    n_unsafe_in_pred_topk = sum(1 for i in topk_pred if not actual_safe[i])
    unsafe_topk_rate = n_unsafe_in_pred_topk / max(len(topk_pred), 1)

    # over-STOP: policy STOPs but oracle has positive utility > epsilon.
    over_stop = (not policy_acted) and (oracle_util > stop_epsilon + 1e-9)

    # Recovery: policy_actual_util / oracle_util (guarded).
    if oracle_util > 1e-9:
        recovery = policy_actual_util / oracle_util
    elif abs(policy_actual_util) < 1e-9:
        recovery = 1.0  # both STOP / 0 utility.
    else:
        recovery = 0.0

    regret = oracle_util - policy_actual_util

    return {
        "oracle_util": oracle_util, "policy_util": policy_actual_util,
        "regret": float(regret), "recovery": float(recovery),
        "topk_safe_recall": float(topk_safe_recall),
        "unsafe_topk_rate": float(unsafe_topk_rate),
        "n_unsafe_in_pred_topk": int(n_unsafe_in_pred_topk),
        "policy_acted": bool(policy_acted),
        "policy_pick_unsafe": bool(policy_pick_actually_unsafe),
        "over_stop": bool(over_stop),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path,
                        default=REPO_ROOT / "evals" / "models" / "rl2_q_policies_stage2_v1.joblib")
    parser.add_argument("--g2-real", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_g2_real_stage2_v1.json")
    parser.add_argument("--stop-epsilon", type=float, default=0.0)
    parser.add_argument("--p-safe-threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_offline_gen_eval_stage2_v1.json")
    args = parser.parse_args()

    import joblib
    models_bundle = joblib.load(str(args.models))
    g2_data = json.load(open(args.g2_real, encoding="utf-8"))
    states_map = g2_data["states"]
    # Group transitions by state.
    by_state = defaultdict(list)
    for t in g2_data["transitions"]:
        by_state[t["state_id"]].append(t)

    holdout_states = defaultdict(list)
    for sid, info in states_map.items():
        if info["split"] in HOLDOUTS:
            holdout_states[info["split"]].append(sid)

    print(f"[INFO] holdouts: { {k: len(v) for k, v in holdout_states.items()} }")

    # For each model × holdout, compute metrics.
    results = {}
    for mname, (util, safe, iso) in models_bundle.items():
        results[mname] = {}
        for ho_name, sids in holdout_states.items():
            per_state = []
            for sid in sids:
                sinfo = states_map[sid]
                cand = by_state[sid]
                metrics = _eval_state(sinfo, cand, util, safe, iso,
                                      stop_epsilon=args.stop_epsilon,
                                      p_safe_threshold=args.p_safe_threshold,
                                      topk=args.topk)
                metrics["state_id"] = sid; metrics["motion_group"] = sinfo.get("motion_group", "?")
                per_state.append(metrics)

            # Aggregate.
            def _ms(field, default=0.0):
                vals = [s[field] for s in per_state if field in s]
                return float(np.mean(vals)) if vals else default

            agg = {
                "n_states": len(per_state),
                "mean_oracle_util": _ms("oracle_util"),
                "mean_policy_util": _ms("policy_util"),
                "mean_regret": _ms("regret"),
                "mean_recovery": _ms("recovery"),
                "mean_topk_safe_recall": _ms("topk_safe_recall"),
                "mean_unsafe_topk_rate": _ms("unsafe_topk_rate"),
                "act_rate": _ms("policy_acted"),
                "policy_pick_unsafe_rate": _ms("policy_pick_unsafe"),
                "over_stop_rate": _ms("over_stop"),
            }
            # Group-wise (G2 only, motion_group != 'clean'/'synthetic').
            group_xtab = {}
            if any(s.get("motion_group", "") not in ("clean", "synthetic") for s in per_state):
                by_group = defaultdict(list)
                for s in per_state:
                    by_group[s["motion_group"]].append(s)
                for g, rows in by_group.items():
                    group_xtab[g] = {
                        "n_states": len(rows),
                        "mean_regret": float(np.mean([r["regret"] for r in rows])),
                        "mean_recovery": float(np.mean([r["recovery"] for r in rows])),
                        "mean_unsafe_topk_rate": float(np.mean([r["unsafe_topk_rate"] for r in rows])),
                        "act_rate": float(np.mean([r["policy_acted"] for r in rows])),
                        "policy_pick_unsafe_rate": float(np.mean([r["policy_pick_unsafe"] for r in rows])),
                    }
            agg["group_breakdown"] = group_xtab
            results[mname][ho_name] = agg

    out = {
        "schema_version": "1.0.0", "record_type": "rl2_offline_gen_eval",
        "task_id": "rl2_offline_gen_eval_stage2_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "rl2_offline_gen_eval_stage2_v1",
            oracle_type="action_effect_transition",
            action_grid="continuous-dense-u11", stage="Stage-2G-step5-offline-gen-eval",
            evidence_tier=[REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
            evaluators=list(DEFAULT_EVALUATORS), gate_evaluators=list(DEFAULT_PHYSICAL_GATE_EVALUATORS),
        ),
        "directive": "M0/M1/M2/M3 의 4 holdout offline generalization. closed-loop 전 단계 — 사용자 명시.",
        "stop_epsilon": args.stop_epsilon, "p_safe_threshold": args.p_safe_threshold, "topk": args.topk,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Step 5 — Offline Generalization Eval ===")
    for ho_name in HOLDOUTS:
        print(f"\n[{ho_name}]")
        print(f"  {'model':<6} {'n':<5} {'regret':<10} {'recovery':<10} {'unsafe_topk':<13} {'act%':<7} {'pick_unsafe%':<14} {'over_STOP%'}")
        for mname in ("M0", "M1", "M2", "M3"):
            r = results[mname].get(ho_name)
            if r is None: continue
            print(f"  {mname:<6} {r['n_states']:<5} {r['mean_regret']:<+10.4f} {r['mean_recovery']:<10.2f} "
                  f"{r['mean_unsafe_topk_rate']*100:<12.1f}% {r['act_rate']*100:<6.0f}% "
                  f"{r['policy_pick_unsafe_rate']*100:<13.1f}% {r['over_stop_rate']*100:.0f}%")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
