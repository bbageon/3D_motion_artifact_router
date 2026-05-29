"""RL-2 Stage A+ (사용자 directive 2026-05-29): Continuous Argmax Confirmation.

사용자 directive 박제:
> "Q_safe(s, tool, u) 에서 continuous argmax 를 실제 정책으로 써도 되는지 확인한다.
>  coarse u 일부로 Q surface 학습 → holdout/dense u 로 실제 평가 → predicted best u 실제
>  적용 → dense oracle 대비 regret 측정 → gate boundary 검증 (unsafe 를 safe 로 안 고르는지)."

핵심 질문: Q 를 **coarse u 만** 보고 학습해도, **보지 않은 u** 에서 utility/safety 를
맞히고 continuous argmax 가 dense oracle 에 근접하는가?

설계:
  - 학습: coarse u (default {0.0, 0.5, 1.0}) 의 transition (train-split state) 로 Q_utility +
    P_safe 학습.
  - dense ground truth: eval-split state 에 fine u-grid (default step 0.05 = 21점) 의 tool 적용
    → 실제 utility + real gate 측정 (continuous oracle).
  - continuous argmax policy: Q 로 (tool, fine u) utility/P_safe 예측 → P_safe>=0.5 인 후보 중
    argmax utility → (tool*, u*); STOP if max < epsilon. 선택된 u* 의 **실제** 측정값 lookup.
  - regret = dense_oracle_utility − policy_actual_utility.

Primary metrics: argmax_regret / utility_recovery / unsafe_as_safe_rate.
Secondary: holdout_u_R2 / boundary_error_u / safe_top1_match.

CLI:
    python -m tools.rl2_continuous_argmax_eval --seeds 0,1,2 \
        --coarse-u 0.0,0.5,1.0 --fine-step 0.05 \
        --dataset evals/snapshots/rl2_transition_dataset_stageA_v1.json \
        --output evals/snapshots/rl2_continuous_argmax_stageAplus_v1.json

근거 (AGENTS.md §3-22): continuous action OOD value overestimation → conservative/holdout
검증 (Kumar CQL NeurIPS 2020); parameterized action (Masson AAAI 2016). safe_utility =
Category C (metric_provenance §4-1-1).
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

from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from tools.synthetic_injection import inject_foot_floating, inject_jitter
from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, REAL_DISTRIBUTION, common_snapshot_metadata
from tools.rl2_build_training_data import ARTIFACT_EVALUATORS, PHYSICAL_EVALUATORS
from tools.rl2_transition_build import (
    TOOLS_ORDER, PROTOCOL_A_DISTS, _sweep_state, TOOL_BY_NAME,
)

TOOL_IDX = {t: i for i, t in enumerate(TOOLS_ORDER)}


def _state_feats(state: dict) -> list[float]:
    """transition state dict → 8-dim feature (artifact 3 + physical 5)."""
    a = [state["artifact_scores"][n] for n in ARTIFACT_EVALUATORS]
    p = [state["physical_scores"][n] for n in PHYSICAL_EVALUATORS]
    return a + p


def _cand_feats(sfeat: list[float], tool: str, u: float) -> list[float]:
    oh = [0.0] * len(TOOLS_ORDER); oh[TOOL_IDX[tool]] = 1.0
    return list(sfeat) + oh + [float(u)]


def _build_heads():
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    util = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_depth=6, random_state=0)
    safe = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_depth=6, random_state=0)
    return util, safe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_dataset_stageA_v1.json")
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--synthetic-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_synthetic_severe_3level_v1.json")
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--calibration", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    parser.add_argument("--coarse-u", type=str, default="0.0,0.5,1.0")
    parser.add_argument("--fine-step", type=float, default=0.05)
    parser.add_argument("--stop-epsilon", type=float, default=0.0)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--eval-frac", type=float, default=0.3)
    parser.add_argument("--max-eval-per-dist", type=int, default=60,
                        help="eval state 수 상한 per distribution (dense 적용 비용 제한)")
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_continuous_argmax_stageAplus_v1.json")
    args = parser.parse_args()

    coarse_u = set(round(float(x), 3) for x in args.coarse_u.split(","))
    fine_u = [round(x, 3) for x in np.arange(0.0, 1.0 + 1e-9, args.fine_step)]
    print(f"[INFO] coarse train u: {sorted(coarse_u)} | fine eval u ({len(fine_u)}): step {args.fine_step}")
    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_thresholds = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                       if calib["summary"][n].get("n", 0) > 0}
    alpha = float(CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1["alpha"])
    evaluators = list(DEFAULT_EVALUATORS)
    gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)

    data = json.load(open(args.dataset, encoding="utf-8"))
    rows = data["rows"]
    # Index rows by (distribution, sample_id) → {u: {tool: row}} for fast train/holdout lookup.
    by_sample = defaultdict(lambda: defaultdict(dict))  # [(dist,sid)][u][tool] = row
    sample_keys = set()
    for r in rows:
        key = (r["distribution"], r["sample_id"])
        by_sample[key][r["u"]][r["tool"]] = r
        sample_keys.add(key)
    sample_keys = sorted(sample_keys)

    seeds = [int(s) for s in args.seeds.split(",")]
    agg = defaultdict(lambda: defaultdict(list))  # [dist][metric] -> per-seed values
    head_r2 = []

    def _load_state_motion(dist, sid):
        if dist == "g2_natural":
            p = args.g2_batch_dir / f"{sid}.npy"
            m = np.load(str(p)).astype(np.float64)
            return m, m
        # Protocol A dists: reconstruct from clean.
        clean = np.load(str(args.data_dir / f"{sid}.npy")).astype(np.float64)
        if dist == "synthetic_severe":
            m1 = inject_foot_floating(clean, lift_height=0.08, seed=args.synthetic_seed)
            return inject_jitter(m1, noise_std=0.05, seed=args.synthetic_seed + 1000), clean
        if dist == "near_boundary":
            return inject_jitter(clean, noise_std=0.025, seed=args.synthetic_seed + 2000), clean
        return clean, clean  # clean

    for seed in seeds:
        rng = np.random.default_rng(seed)
        by_dist = defaultdict(list)
        for (d, s) in sample_keys:
            by_dist[d].append((d, s))
        train_keys, eval_keys = set(), []
        for d, keys in by_dist.items():
            keys = sorted(keys); rng.shuffle(keys)
            n_eval = min(int(round(len(keys) * args.eval_frac)), args.max_eval_per_dist)
            eval_keys += keys[:n_eval]
            train_keys |= set(keys[n_eval:])

        # Train Q on coarse-u rows from train states.
        Xtr, ytr_u, ytr_safe = [], [], []
        for key in train_keys:
            for u, tool_rows in by_sample[key].items():
                if u not in coarse_u:
                    continue
                for tool, r in tool_rows.items():
                    Xtr.append(_cand_feats(_state_feats(r["state"]), tool, u))
                    ytr_u.append(r["safe_utility"])
                    ytr_safe.append(1 if r["gate_result"] == "pass" else 0)
        Xtr = np.array(Xtr); ytr_u = np.array(ytr_u); ytr_safe = np.array(ytr_safe)
        util_head, safe_head = _build_heads()
        util_head.fit(Xtr, ytr_u)
        safe_head.fit(Xtr, ytr_safe)

        # holdout_u_R2: eval-state rows at NON-coarse grid u (from dataset 11-grid).
        from sklearn.metrics import r2_score
        Xho, yho = [], []
        for key in eval_keys:
            for u, tool_rows in by_sample[key].items():
                if u in coarse_u:
                    continue
                for tool, r in tool_rows.items():
                    Xho.append(_cand_feats(_state_feats(r["state"]), tool, u)); yho.append(r["safe_utility"])
        if Xho:
            head_r2.append(float(r2_score(yho, util_head.predict(np.array(Xho)))))

        # Continuous argmax on eval states (dense fine-u ground truth via tool application).
        per_dist_rows = defaultdict(list)
        for (dist, sid) in eval_keys:
            motion, clean = _load_state_motion(dist, sid)
            dense = _sweep_state(motion, dist, clean, evaluators, gate_evaluators,
                                 gate_thresholds, fine_u, alpha, sid)
            # dense[i] = row for (tool, u). Build lookup + state feat.
            sfeat = _state_feats({"artifact_scores": {n: dense[0]["state"]["artifact_scores"][n] for n in ARTIFACT_EVALUATORS},
                                  "physical_scores": {n: dense[0]["state"]["physical_scores"][n] for n in PHYSICAL_EVALUATORS}}) if dense else None
            if not dense:
                continue
            # dense oracle (real gate) + STOP=0 baseline.
            safe_dense = [d for d in dense if d["gate_result"] == "pass"]
            oracle_util = max([0.0] + [d["safe_utility"] for d in safe_dense])
            # policy: predict over all dense candidates, pick argmax predicted-safe utility.
            Xc = np.array([_cand_feats(sfeat, d["tool"], d["u"]) for d in dense])
            pu = util_head.predict(Xc)
            ps = safe_head.predict_proba(Xc)[:, 1]
            safe_mask = ps >= 0.5
            if safe_mask.any():
                cand_idx = np.where(safe_mask)[0]
                best = cand_idx[int(np.argmax(pu[cand_idx]))]
                if pu[best] < args.stop_epsilon:
                    policy_util, acted, picked = 0.0, False, None  # STOP
                else:
                    policy_util = dense[best]["safe_utility"]  # ACTUAL measured utility at picked (tool*,u*).
                    acted, picked = True, dense[best]
            else:
                policy_util, acted, picked = 0.0, False, None
            unsafe_as_safe = bool(acted and picked["gate_result"] == "hard_violation")
            regret = oracle_util - policy_util
            # boundary error (per tool): true vs predicted first-violation u.
            berr = []
            for tool in TOOLS_ORDER:
                trows = sorted([d for d in dense if d["tool"] == tool], key=lambda x: x["u"])
                true_b = next((d["u"] for d in trows if d["gate_result"] == "hard_violation"), 1.0)
                Xt = np.array([_cand_feats(sfeat, tool, d["u"]) for d in trows])
                pst = safe_head.predict_proba(Xt)[:, 1]
                pred_b = next((trows[i]["u"] for i in range(len(trows)) if pst[i] < 0.5), 1.0)
                berr.append(abs(true_b - pred_b))
            # safe top1 match: policy (tool*,u*) vs oracle argmax (tool,u).
            top1 = False
            if safe_dense and acted:
                o = max(safe_dense, key=lambda d: d["safe_utility"])
                top1 = (o["tool"] == picked["tool"] and abs(o["u"] - picked["u"]) <= args.fine_step + 1e-9)
            per_dist_rows[dist].append({
                "regret": regret, "oracle_util": oracle_util, "policy_util": policy_util,
                "unsafe_as_safe": unsafe_as_safe, "acted": acted,
                "boundary_error_u": float(np.mean(berr)), "top1_match": top1,
            })

        for dist, rws in per_dist_rows.items():
            agg[dist]["argmax_regret"].append(float(np.mean([r["regret"] for r in rws])))
            mo = float(np.mean([r["oracle_util"] for r in rws]))
            mp = float(np.mean([r["policy_util"] for r in rws]))
            agg[dist]["mean_oracle_util"].append(mo)
            agg[dist]["mean_policy_util"].append(mp)
            agg[dist]["utility_recovery"].append(mp / mo if abs(mo) > 1e-6 else (1.0 if abs(mp) < 1e-6 else 0.0))
            agg[dist]["unsafe_as_safe_rate"].append(float(np.mean([r["unsafe_as_safe"] for r in rws])))
            agg[dist]["boundary_error_u"].append(float(np.mean([r["boundary_error_u"] for r in rws])))
            agg[dist]["safe_top1_match"].append(float(np.mean([r["top1_match"] for r in rws])))
            agg[dist]["act_rate"].append(float(np.mean([r["acted"] for r in rws])))
            agg[dist]["n_eval"].append(len(rws))
        print(f"[seed {seed}] holdout_u_R2={head_r2[-1]:.3f} | eval states={sum(len(v) for v in per_dist_rows.values())}")

    def _ms(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None
    results = {}
    for dist in agg:
        results[dist] = {k: _ms(v) for k, v in agg[dist].items()}

    out = {
        "schema_version": "1.0.0", "record_type": "rl2_continuous_argmax_confirmation",
        "task_id": "rl2_continuous_argmax_stageAplus_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "rl2_continuous_argmax_stageAplus_v1",
            oracle_type="action_effect_transition", action_grid="continuous-u",
            stage="RL-2-Q-surface-stageA+",
            evidence_tier=[REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
            evaluators=evaluators, gate_evaluators=gate_evaluators,
        ),
        "question": "coarse-u 학습 Q 의 continuous argmax 가 dense oracle 에 근접 + unsafe 회피 하는가?",
        "coarse_train_u": sorted(coarse_u), "fine_eval_u_step": args.fine_step, "n_fine_u": len(fine_u),
        "seeds": seeds, "stop_epsilon": args.stop_epsilon,
        "holdout_u_R2": _ms(head_r2),
        "safe_utility_metric": "Category C internal routing reward (metric_provenance §4-1-1)",
        "per_distribution": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RL-2 Stage A+ Continuous Argmax Confirmation ===")
    print(f"  holdout_u_R2 (unseen u 예측력): {out['holdout_u_R2']['mean']:.3f} ± {out['holdout_u_R2']['std']:.3f}")
    print(f"\n  {'dist':<18} {'regret':<10} {'recovery':<10} {'unsafe_as_safe':<15} {'boundary_err_u':<15} {'top1':<8} {'act%'}")
    for dist in ("clean", "near_boundary", "g2_natural", "synthetic_severe"):
        if dist not in results:
            continue
        r = results[dist]
        print(f"  {dist:<18} {r['argmax_regret']['mean']:<10.4f} {r['utility_recovery']['mean']:<10.2f} "
              f"{r['unsafe_as_safe_rate']['mean']*100:<14.1f}% {r['boundary_error_u']['mean']:<15.3f} "
              f"{r['safe_top1_match']['mean']*100:<7.0f}% {r['act_rate']['mean']*100:.0f}%")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
