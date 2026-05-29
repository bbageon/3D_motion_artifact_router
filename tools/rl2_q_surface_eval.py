"""RL-2 Stage 1 (사용자 directive 2026-05-29): Q_safe two-head model + reranking closed-loop 평가.

사용자 directive 박제:
> "모델: utility head (safe utility 예측) + risk head (physical violation 확률). 추론:
>  for tool, for u: predict utility, risk; remove risk > threshold; choose max utility;
>  STOP if max utility < epsilon. Q 가 높다고 바로 믿으면 안 된다 — 추론 시에도 physical
>  gate 를 실제로 한 번 더 돌려야 한다."

Stage 1 목표 (사용자):
  (1) 기존 3-level 결과를 Q/reranking formulation 으로 재현.
  (2) imitation policy 보다 Q surface 가 더 잘 설명/작동 하는지 확인.
  (3) STOP threshold calibration.

본 도구:
  - rl2_q_surface_dataset 에서 (state, tool, u) → (safe_utility, is_violation) 회귀/분류 학습.
  - utility head (HGB regressor) + risk head (HGB classifier), sample-level split, multi-seed.
  - reranking closed-loop: predict → risk filter → max-utility → STOP threshold → REAL gate 재검증.
  - 비교: noop / b2_netgain_best / safe oracle / imitation HGB (3-level) / Q-rerank.

CLI:
    python -m tools.rl2_q_surface_eval --seeds 0,1,2 \
        --q-dataset evals/snapshots/rl2_q_surface_dataset_stage1_v1.json \
        --imitation-dataset evals/snapshots/rl2_imitation_dataset_3level_v1.json \
        --output evals/snapshots/rl2_q_surface_eval_stage1_v1.json

근거 (AGENTS.md §3-22): reranking 우선 (offline RL OOD value overestimation, Kumar CQL
NeurIPS 2020). safe_utility = Category C (metric_provenance §4-1-1).
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
from tools.rl2_build_training_data import (
    ARTIFACT_EVALUATORS, PHYSICAL_EVALUATORS, TOOLS_ORDER, TOOL_TARGET,
    _artifact_scores, _physical_scores, _build_state, build_action_list,
)
from tools.rl2_train_imitation import _flatten_state, _sample_level_split
from tools.rl2_q_surface_build import (
    U_TO_STRENGTH_3LEVEL, TARGET_EVALUATORS_A, _target_full, _target_A, _mpjpe, _u_strength,
    ACTION_LIST_3, N_ACTIONS_3,
)

TOOL_BY_NAME: dict[str, CorrectionTool] = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}
TOOL_IDX = {t: i for i, t in enumerate(TOOLS_ORDER)}


def _candidate_features(state_feats: list[float], tool: str, u: float) -> list[float]:
    onehot = [0.0] * len(TOOLS_ORDER)
    onehot[TOOL_IDX[tool]] = 1.0
    return list(state_feats) + onehot + [float(u)]


def _build_q_xy(rows, u_grid):
    """Q-surface rows → (X, y_utility, y_violation, sample_keys)."""
    X, y_u, y_v, keys, dists = [], [], [], [], []
    for r in rows:
        sfeat = _flatten_state(r["state"], include_dist_tag=False)
        for e in r["effects"]:
            X.append(_candidate_features(sfeat, e["tool"], e["u"]))
            y_u.append(e["safe_utility"])
            y_v.append(1 if e["physical_violation"] else 0)
            keys.append((r["distribution"], r["sample_id"]))
            dists.append(r["distribution"])
    return np.array(X), np.array(y_u), np.array(y_v), keys, dists


def _build_heads():
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    util = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_depth=6, random_state=0)
    risk = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_depth=6, random_state=0)
    return util, risk


def _q_split_idx(keys, train_sample_keys):
    tr = [i for i, k in enumerate(keys) if k in train_sample_keys]
    ev = [i for i, k in enumerate(keys) if k not in train_sample_keys]
    return tr, ev


def _run_q_reranking(motion0, util_head, risk_head, evaluators, gate_evaluators,
                     gate_thresholds, dist_tag, u_grid, max_depth, risk_threshold, stop_epsilon):
    """Q reranking closed-loop: predict → risk filter → max-utility → STOP → REAL gate."""
    motion = motion0.copy()
    T = motion.shape[0]
    artifact = _artifact_scores(motion, evaluators)
    physical = _physical_scores(motion, gate_evaluators)
    prev_artifact, prev_physical = list(artifact), list(physical)
    prev_action_idx = 0
    gate_parent = {n: physical[i] for i, n in enumerate(PHYSICAL_EVALUATORS)}
    actions = []
    stopped = "max_depth"
    for t in range(max_depth):
        delta = [a - pa for a, pa in zip(artifact, prev_artifact)] + \
                [p - pp for p, pp in zip(physical, prev_physical)]
        state = _build_state(artifact, physical, delta, prev_action_idx, max_depth - t, t,
                             dist_tag, n_actions=N_ACTIONS_3)
        sfeat = _flatten_state(state, include_dist_tag=False)
        cand = [(tool, u) for tool in TOOLS_ORDER for u in u_grid]
        Xc = np.array([_candidate_features(sfeat, tool, u) for tool, u in cand])
        util_pred = util_head.predict(Xc)
        risk_pred = risk_head.predict_proba(Xc)[:, 1] if hasattr(risk_head, "predict_proba") else risk_head.predict(Xc)
        # Filter risky candidates; choose max predicted utility among safe.
        safe_mask = risk_pred <= risk_threshold
        if not safe_mask.any():
            stopped = "all_risky"; break
        safe_idx = np.where(safe_mask)[0]
        best_local = safe_idx[np.argmax(util_pred[safe_idx])]
        if util_pred[best_local] < stop_epsilon:
            stopped = "policy_stop"; break
        tool, u = cand[best_local]
        st = _u_strength(u)
        try:
            new_motion, _ = TOOL_BY_NAME[tool].apply(motion, target_part=TOOL_TARGET[tool],
                                                     target_joints=[], frame_range=(0, T-1), strength=st)
        except ValueError:
            stopped = "apply_error"; break
        # REAL physical gate re-check (사용자 risk: Q 만 믿지 말 것).
        gate_after = _gate_scores(new_motion, gate_evaluators)
        decisions = _gate_violation(gate_after, gate_parent, gate_thresholds)
        if any(d == "hard_violation" for d in decisions.values()):
            actions.append({"tool": tool, "u": u, "gate": "rollback",
                            "pred_util": float(util_pred[best_local]), "pred_risk": float(risk_pred[best_local])})
            stopped = "gate_rollback"; break
        actions.append({"tool": tool, "u": u, "gate": "accept",
                        "pred_util": float(util_pred[best_local]), "pred_risk": float(risk_pred[best_local])})
        motion = new_motion
        prev_artifact, prev_physical = list(artifact), list(physical)
        artifact = _artifact_scores(motion, evaluators)
        physical = _physical_scores(motion, gate_evaluators)
        gate_parent = {n: physical[i] for i, n in enumerate(PHYSICAL_EVALUATORS)}
        key = f"{tool}|{st}"
        prev_action_idx = ACTION_LIST_3.index(key) if key in ACTION_LIST_3 else 0
    return motion, {"actions": actions, "n_steps": len(actions), "stopped_reason": stopped}


def _netgain_g2(final, original, evaluators, w):
    return -(_target_full(final, evaluators) - _target_full(original, evaluators)) - float(w["alpha"]) * _mpjpe(final, original)


def _netgain_synthetic(final, clean, corrupted, evaluators, w):
    mpjpe_corr = _mpjpe(corrupted, clean)
    return -(_target_A(final, evaluators) - _target_A(corrupted, evaluators)) - float(w["alpha"]) * (_mpjpe(final, clean) - mpjpe_corr)


def _has_violation(final, original, gate_evaluators, gate_thresholds):
    before = _gate_scores(original, gate_evaluators)
    after = _gate_scores(final, gate_evaluators)
    dec = _gate_violation(after, before, gate_thresholds)
    return any(d == "hard_violation" for d in dec.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-dataset", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_q_surface_dataset_stage1_v1.json")
    parser.add_argument("--imitation-dataset", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_imitation_dataset_3level_v1.json")
    parser.add_argument("--g2-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_g2_hml3d_test300_3level_v1.json")
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--synthetic-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_synthetic_severe_3level_v1.json")
    parser.add_argument("--synthetic-data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--calibration", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--stop-epsilon", type=float, default=0.0)
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_q_surface_eval_stage1_v1.json")
    args = parser.parse_args()

    qdata = json.load(open(args.q_dataset, encoding="utf-8"))
    q_rows = qdata["rows"]
    u_grid = qdata["u_grid"]
    imdata = json.load(open(args.imitation_dataset, encoding="utf-8"))
    im_rows = imdata["rows"]
    im_action_list = imdata.get("action_list")
    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_thresholds = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                       if calib["summary"][n].get("n", 0) > 0}
    seeds = [int(s) for s in args.seeds.split(",")]
    w = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
    evaluators = list(DEFAULT_EVALUATORS)
    gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)
    g2_oracle = {p["trial_id"]: p for p in json.load(open(args.g2_oracle, encoding="utf-8"))["per_sample"]}
    syn_oracle = {p["trial_id"]: p for p in json.load(open(args.synthetic_oracle, encoding="utf-8"))["per_sample"]}

    # Imitation policy infra (3-level).
    from tools.rl2_train_imitation import _build_models
    from tools.rl2_closed_loop_eval import _run_policy_closed_loop, _idx_to_action  # 3-level via action_list
    from tools.rl2_build_training_data import STRENGTHS_3LEVEL
    im_X = np.array([_flatten_state(r["state"], False) for r in im_rows])
    im_y = np.array([r["action_idx"] for r in im_rows])

    X, y_u, y_v, q_keys, q_dists = _build_q_xy(q_rows, u_grid)
    util_factory_risk_factory = _build_heads  # noqa

    head_metrics = defaultdict(list)
    seed_rows = defaultdict(lambda: defaultdict(list))  # [dist][_rows]

    for seed in seeds:
        # Sample-level split via imitation rows (deterministic by sample keys + seed).
        tr_im, ev_im = _sample_level_split(im_rows, seed)
        train_keys = {(im_rows[i]["distribution"], im_rows[i]["sample_id"]) for i in tr_im}
        eval_g2 = sorted({im_rows[i]["sample_id"] for i in ev_im if im_rows[i]["distribution"] == "g2"})
        eval_syn = sorted({im_rows[i]["sample_id"] for i in ev_im if im_rows[i]["distribution"] == "synthetic"})

        # Train Q heads on train samples.
        q_tr, q_ev = _q_split_idx(q_keys, train_keys)
        util_head, risk_head = _build_heads()
        util_head.fit(X[q_tr], y_u[q_tr])
        risk_head.fit(X[q_tr], y_v[q_tr])
        # Head metrics on eval candidates.
        from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, accuracy_score
        u_pred = util_head.predict(X[q_ev])
        head_metrics["util_r2"].append(float(r2_score(y_u[q_ev], u_pred)))
        head_metrics["util_mae"].append(float(mean_absolute_error(y_u[q_ev], u_pred)))
        rp = risk_head.predict_proba(X[q_ev])[:, 1]
        if len(np.unique(y_v[q_ev])) > 1:
            head_metrics["risk_auc"].append(float(roc_auc_score(y_v[q_ev], rp)))
        head_metrics["risk_acc"].append(float(accuracy_score(y_v[q_ev], (rp > args.risk_threshold).astype(int))))

        # Train imitation HGB on same train samples.
        im_clf = _build_models()["HGB"]()
        im_clf.fit(im_X[tr_im], im_y[tr_im])

        for dist, eval_ids in (("g2", eval_g2), ("synthetic", eval_syn)):
            for tid in eval_ids:
                if dist == "g2":
                    npy = args.g2_batch_dir / f"{tid}.npy"
                    if not npy.exists():
                        continue
                    original = np.load(str(npy)).astype(np.float64)
                    motion_in, clean, dist_tag = original, original, 0
                    noop_ng = 0.0
                    oc = g2_oracle.get(tid, {})
                    ng_fn = lambda m: _netgain_g2(m, original, evaluators, w)
                else:
                    npy = args.synthetic_data_dir / f"{tid}.npy"
                    if not npy.exists():
                        continue
                    clean = np.load(str(npy)).astype(np.float64)
                    m1 = inject_foot_floating(clean, lift_height=0.08, seed=args.synthetic_seed)
                    corrupted = inject_jitter(m1, noise_std=0.05, seed=args.synthetic_seed + 1000)
                    motion_in, dist_tag = corrupted, 1
                    noop_ng = 0.0
                    oc = syn_oracle.get(tid, {})
                    ng_fn = lambda m: _netgain_synthetic(m, clean, corrupted, evaluators, w)
                safe_ng = oc["safe_best"]["netgain"] if oc.get("safe_best") else noop_ng
                row = {"trial_id": tid, "noop": noop_ng, "safe_oracle": safe_ng}
                # Q-rerank.
                q_final, q_trace = _run_q_reranking(motion_in, util_head, risk_head, evaluators,
                                                    gate_evaluators, gate_thresholds, dist_tag, u_grid,
                                                    args.max_depth, args.risk_threshold, args.stop_epsilon)
                row["q_rerank_ng"] = ng_fn(q_final)
                row["q_rerank_viol"] = _has_violation(q_final, motion_in, gate_evaluators, gate_thresholds)
                row["q_rerank_stop"] = q_trace["stopped_reason"] == "policy_stop" and q_trace["n_steps"] == 0
                row["q_rerank_nsteps"] = q_trace["n_steps"]
                # Imitation HGB.
                im_final, im_trace = _run_policy_closed_loop(
                    motion_in, im_clf, evaluators, gate_evaluators, gate_thresholds, dist_tag=dist_tag,
                    max_depth=args.max_depth, strengths=STRENGTHS_3LEVEL, action_list=im_action_list)
                row["imitation_ng"] = ng_fn(im_final)
                row["imitation_viol"] = _has_violation(im_final, motion_in, gate_evaluators, gate_thresholds)
                row["imitation_stop"] = im_trace["stopped_reason"] == "policy_stop" and im_trace["n_steps"] == 0
                seed_rows[dist]["_rows"].append(row)
        print(f"[seed {seed}] eval done (G2 {len(eval_g2)}, synthetic {len(eval_syn)}) | "
              f"util_r2={head_metrics['util_r2'][-1]:.3f}")

    def _agg(dist_rows):
        if not dist_rows:
            return {}
        mean_noop = float(np.mean([r["noop"] for r in dist_rows]))
        mean_safe = float(np.mean([r["safe_oracle"] for r in dist_rows]))
        denom = mean_safe - mean_noop
        out = {"noop": {"mean_netgain": mean_noop}, "safe_oracle": {"mean_netgain": mean_safe}}
        for m in ("q_rerank", "imitation"):
            ng = [r[f"{m}_ng"] for r in dist_rows]
            viol = [r[f"{m}_viol"] for r in dist_rows]
            stop = [r[f"{m}_stop"] for r in dist_rows]
            steps = [r[f"{m}_nsteps"] for r in dist_rows if f"{m}_nsteps" in r]
            mp = float(np.mean(ng))
            out[m] = {"mean_netgain": mp,
                      "violation_rate": float(np.mean(viol)),
                      "stop_rate": float(np.mean(stop)),
                      "oracle_gap_closure_aggregate": (mp - mean_noop) / denom if abs(denom) > 1e-9 else 0.0}
            if steps:
                out[m]["mean_steps"] = float(np.mean(steps))
        return out

    final_out = {
        "schema_version": "1.0.0", "record_type": "rl2_q_surface_eval",
        "task_id": "rl2_q_surface_eval_stage1_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "rl2_q_surface_eval_stage1_v1",
            oracle_type="sequence", action_grid="3-level", stage="RL-2-Q-surface-stage1",
            evidence_tier=[REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
            evaluators=evaluators, gate_evaluators=gate_evaluators,
        ),
        "formulation": "Q_safe(s, tool, u) reranking (Stage 1, u_grid 3-level)",
        "u_grid": u_grid, "seeds": seeds, "max_depth": args.max_depth,
        "risk_threshold": args.risk_threshold, "stop_epsilon": args.stop_epsilon,
        "safe_utility_metric": "Category C internal routing reward (metric_provenance §4-1-1)",
        "head_metrics": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in head_metrics.items()},
        "g2": _agg(seed_rows["g2"]["_rows"]),
        "synthetic": _agg(seed_rows["synthetic"]["_rows"]),
        "n_g2_eval_total": len(seed_rows["g2"]["_rows"]),
        "n_synthetic_eval_total": len(seed_rows["synthetic"]["_rows"]),
        "note": "Q-rerank vs imitation HGB (same 3-level samples). NetGain: G2=Protocol B, synthetic=Protocol A. "
                "REAL physical gate re-check at inference (action_space_provenance §5-2-6).",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(final_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RL-2 Q-surface Reranking Eval (Stage 1) ===")
    hm = final_out["head_metrics"]
    print(f"  Q heads: util R2={hm.get('util_r2',{}).get('mean',0):.3f}  util MAE={hm.get('util_mae',{}).get('mean',0):.4f}  "
          f"risk AUC={hm.get('risk_auc',{}).get('mean',float('nan')):.3f}  risk acc={hm.get('risk_acc',{}).get('mean',0):.3f}")
    for dist in ("g2", "synthetic"):
        print(f"\n[{dist}] (n_eval_total across seeds = {final_out[f'n_{dist}_eval_total']})")
        agg = final_out[dist]
        print(f"  {'method':<14} {'mean_NetGain':<14} {'violation':<11} {'STOP':<9} {'oracle_gap'}")
        for m in ("noop", "safe_oracle", "imitation", "q_rerank"):
            d = agg.get(m, {})
            ng = d.get("mean_netgain", 0); vr = d.get("violation_rate"); sr = d.get("stop_rate"); gc = d.get("oracle_gap_closure_aggregate")
            vr_s = f"{vr*100:<10.0f}%" if vr is not None else " " * 11
            sr_s = f"{sr*100:<8.0f}%" if sr is not None else " " * 9
            gc_s = f"{gc*100:.0f}%" if gc is not None else ""
            print(f"  {m:<14} {ng:<+14.4f} {vr_s} {sr_s} {gc_s}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
