"""RL-2 Stage B-0 (사용자 directive 2026-05-30): P_safe Bias Audit.

사용자 directive 박제:
> "46.3% 는 단순 성능 부족이 아니라 safety head 의 낙관 편향 신호. mining 전에 P_safe
>  bias audit 을 먼저. unsafe 를 positive class 로 보는 것이 핵심 (safe 정확도 보다 unsafe
>  recall 이 중요). P_safe >= 0.5 는 일반 분류 기준, safety 기준 아님 — 보수적 threshold
>  필요."

7 진단 (사용자 directive):
  (1) tool 별 unsafe_as_safe — 어떤 tool 이 문제인지.
  (2) u-bin 별 unsafe_as_safe — 어떤 strength 구간이 문제인지.
  (3) distribution 별 safe/unsafe label ratio — class imbalance.
  (4) P_safe calibration curve — 확률 ↔ 실제 safety 일치.
  (5) PR-AUC (unsafe = positive) — unsafe 탐지 성능.
  (6) Confusion matrix (unsafe = positive) — safety 관점 평가.
  (7) Selected action vs all candidate — argmax 가 위험 후보 끌어올리는지.

추가: P_safe threshold sweep (0.5 → 0.95 → 0.99) — 보수적 기준의 trade-off (unsafe-as-safe
vs act_rate). 사용자 framing 박제: "execute only if P_safe ≥ 0.95".

근거 (AGENTS.md §3-22): offline RL OOD action value overestimation (CQL Kumar NeurIPS 2020;
IQL Kostrikov ICLR 2022). safety calibration: Naeini et al. AAAI 2015 (calibration curves).

CLI:
    python -m tools.rl2_psafe_bias_audit --seeds 0,1,2 \
        --coarse-u 0.0,0.5,1.0 \
        --dataset evals/snapshots/rl2_transition_dataset_stageA_v1.json \
        --output evals/snapshots/rl2_psafe_bias_audit_stageB0_v1.json \
        --output-fig reports/figures/2026-05-30/psafe_bias_audit.png
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, REAL_DISTRIBUTION, common_snapshot_metadata
from tools.rl2_build_training_data import ARTIFACT_EVALUATORS, PHYSICAL_EVALUATORS
from tools.rl2_transition_build import TOOLS_ORDER
from tools.rl2_continuous_argmax_eval import _state_feats, _cand_feats, _build_heads
from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS


def _calibration_bins(y_true_safe, p_safe, n_bins=10):
    """bin P_safe predictions, return per-bin (mean_pred, actual_safe_rate, n)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p_safe >= lo) & (p_safe < hi if i < n_bins - 1 else p_safe <= hi)
        if not mask.any():
            out.append({"bin": [float(lo), float(hi)], "n": 0,
                        "mean_predicted": None, "actual_safe_rate": None})
            continue
        out.append({"bin": [float(lo), float(hi)], "n": int(mask.sum()),
                    "mean_predicted": float(np.mean(p_safe[mask])),
                    "actual_safe_rate": float(np.mean(y_true_safe[mask]))})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_dataset_stageA_v1.json")
    parser.add_argument("--coarse-u", type=str, default="0.0,0.5,1.0")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--eval-frac", type=float, default=0.3)
    parser.add_argument("--thresholds", type=str, default="0.5,0.7,0.8,0.9,0.95,0.99")
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_psafe_bias_audit_stageB0_v1.json")
    parser.add_argument("--output-fig", type=Path,
                        default=REPO_ROOT / "reports" / "figures" / "2026-05-30" / "psafe_bias_audit.png")
    args = parser.parse_args()

    coarse_u = set(round(float(x), 3) for x in args.coarse_u.split(","))
    thresholds = [float(x) for x in args.thresholds.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    data = json.load(open(args.dataset, encoding="utf-8"))
    rows = data["rows"]
    print(f"[INFO] {len(rows)} transitions, coarse train u={sorted(coarse_u)}, seeds={seeds}")

    # Group rows by (dist, sample_id) for sample-level split.
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["distribution"], r["sample_id"])].append(r)
    keys_by_dist = defaultdict(list)
    for k in by_key:
        keys_by_dist[k[0]].append(k)

    # Aggregate across seeds.
    seed_audits = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        train_keys, eval_keys = set(), []
        for d, keys in keys_by_dist.items():
            keys = sorted(keys); rng.shuffle(keys)
            n_eval = int(round(len(keys) * args.eval_frac))
            eval_keys += keys[:n_eval]
            train_keys |= set(keys[n_eval:])

        # Build train/eval feature matrices.
        def _flatten_rows(keys, u_filter=None):
            Xs, y_u, y_safe, meta = [], [], [], []
            for k in keys:
                for r in by_key[k]:
                    if u_filter is not None and r["u"] not in u_filter:
                        continue
                    Xs.append(_cand_feats(_state_feats(r["state"]), r["tool"], r["u"]))
                    y_u.append(r["safe_utility"])
                    y_safe.append(1 if r["gate_result"] == "pass" else 0)
                    meta.append({"dist": r["distribution"], "tool": r["tool"], "u": r["u"],
                                 "sample_id": r["sample_id"]})
            return np.array(Xs), np.array(y_u), np.array(y_safe), meta

        Xtr, ytr_u, ytr_safe, _ = _flatten_rows(train_keys, u_filter=coarse_u)
        Xev, yev_u, yev_safe, ev_meta = _flatten_rows(eval_keys, u_filter=None)  # eval at ALL 11 u

        util_head, safe_head = _build_heads()
        util_head.fit(Xtr, ytr_u)
        safe_head.fit(Xtr, ytr_safe)
        p_safe = safe_head.predict_proba(Xev)[:, 1]
        is_safe_true = yev_safe.astype(bool)
        is_unsafe_true = ~is_safe_true

        from sklearn.metrics import average_precision_score, roc_auc_score
        # PR-AUC with unsafe = positive class (label=1 means unsafe).
        y_unsafe = (1 - yev_safe).astype(int)
        score_unsafe = 1.0 - p_safe
        pr_auc_unsafe = float(average_precision_score(y_unsafe, score_unsafe))
        roc_auc_unsafe = float(roc_auc_score(y_unsafe, score_unsafe)) if len(np.unique(y_unsafe)) > 1 else None

        # Confusion matrix at P_safe >= 0.5 (unsafe = positive).
        pred_unsafe = (p_safe < 0.5).astype(int)
        TP = int(np.sum((y_unsafe == 1) & (pred_unsafe == 1)))
        FN = int(np.sum((y_unsafe == 1) & (pred_unsafe == 0)))  # false safe (unsafe-as-safe)
        FP = int(np.sum((y_unsafe == 0) & (pred_unsafe == 1)))
        TN = int(np.sum((y_unsafe == 0) & (pred_unsafe == 0)))
        prec_unsafe = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec_unsafe = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_unsafe = 2 * prec_unsafe * rec_unsafe / (prec_unsafe + rec_unsafe) if (prec_unsafe + rec_unsafe) > 0 else 0.0

        # (1) per-tool unsafe-as-safe (predicted safe but actually unsafe).
        per_tool = defaultdict(lambda: {"n_pred_safe": 0, "n_unsafe_among_pred_safe": 0})
        for i, m in enumerate(ev_meta):
            if p_safe[i] >= 0.5:
                per_tool[m["tool"]]["n_pred_safe"] += 1
                if is_unsafe_true[i]:
                    per_tool[m["tool"]]["n_unsafe_among_pred_safe"] += 1
        per_tool_out = {t: {**v, "unsafe_as_safe_rate": (v["n_unsafe_among_pred_safe"] / v["n_pred_safe"] if v["n_pred_safe"] else 0.0)} for t, v in per_tool.items()}

        # (2) per-u unsafe-as-safe.
        per_u = defaultdict(lambda: {"n_pred_safe": 0, "n_unsafe_among_pred_safe": 0})
        for i, m in enumerate(ev_meta):
            if p_safe[i] >= 0.5:
                per_u[m["u"]]["n_pred_safe"] += 1
                if is_unsafe_true[i]:
                    per_u[m["u"]]["n_unsafe_among_pred_safe"] += 1
        per_u_out = {str(u): {**v, "unsafe_as_safe_rate": (v["n_unsafe_among_pred_safe"] / v["n_pred_safe"] if v["n_pred_safe"] else 0.0)} for u, v in sorted(per_u.items())}

        # (3) per-distribution safe/unsafe label ratio (train AND eval).
        Xtr2, _, ytr_safe2, tr_meta = _flatten_rows(train_keys, u_filter=coarse_u)
        per_dist_label = {}
        for dist in ("clean", "near_boundary", "g2_natural", "synthetic_severe"):
            tr_n = sum(1 for m in tr_meta if m["dist"] == dist)
            tr_s = sum(1 for i, m in enumerate(tr_meta) if m["dist"] == dist and ytr_safe2[i] == 1)
            ev_n = sum(1 for m in ev_meta if m["dist"] == dist)
            ev_s = sum(1 for i, m in enumerate(ev_meta) if m["dist"] == dist and yev_safe[i] == 1)
            per_dist_label[dist] = {
                "train": {"n": tr_n, "safe_rate": tr_s / tr_n if tr_n else 0.0, "unsafe_rate": 1 - (tr_s / tr_n) if tr_n else 0.0},
                "eval":  {"n": ev_n, "safe_rate": ev_s / ev_n if ev_n else 0.0, "unsafe_rate": 1 - (ev_s / ev_n) if ev_n else 0.0},
            }
        # Per-distribution unsafe-as-safe (eval).
        per_dist_unsafe = {}
        for dist in ("clean", "near_boundary", "g2_natural", "synthetic_severe"):
            mask = np.array([m["dist"] == dist for m in ev_meta])
            if not mask.any(): continue
            pmask = p_safe[mask] >= 0.5
            true_unsafe = is_unsafe_true[mask]
            n_ps = int(pmask.sum())
            n_fs = int(np.sum(pmask & true_unsafe))
            per_dist_unsafe[dist] = {"n_pred_safe": n_ps, "n_false_safe": n_fs,
                                      "unsafe_as_safe_rate": (n_fs / n_ps if n_ps else 0.0)}

        # (4) Calibration curve.
        calib = _calibration_bins(yev_safe, p_safe, n_bins=10)

        # (7) Selected action vs all candidate — does argmax pull up risky cands?
        # Group by (sample_id, dist), for each sample pick top-1 (predicted-safe argmax utility).
        by_state = defaultdict(list)
        u_pred = util_head.predict(Xev)
        for i, m in enumerate(ev_meta):
            by_state[(m["dist"], m["sample_id"])].append((i, m["tool"], m["u"]))
        sel_vs_all = {}
        for dist in ("clean", "near_boundary", "g2_natural", "synthetic_severe"):
            cands_unsafe = 0; cands_n = 0
            top1_unsafe = 0; top1_n = 0
            for key, cands in by_state.items():
                if key[0] != dist: continue
                idxs = [c[0] for c in cands]
                # All-candidate unsafe rate.
                cands_n += len(idxs)
                cands_unsafe += int(np.sum(is_unsafe_true[idxs]))
                # Top-1: max predicted utility among P_safe>=0.5 (predicted-safe set).
                safe_idxs = [i for i in idxs if p_safe[i] >= 0.5]
                if not safe_idxs: continue
                best = safe_idxs[int(np.argmax([u_pred[i] for i in safe_idxs]))]
                top1_n += 1
                if is_unsafe_true[best]:
                    top1_unsafe += 1
            sel_vs_all[dist] = {
                "all_cand_unsafe_rate": cands_unsafe / cands_n if cands_n else 0.0,
                "top1_unsafe_rate": top1_unsafe / top1_n if top1_n else 0.0,
                "lift": (top1_unsafe / top1_n if top1_n else 0.0) - (cands_unsafe / cands_n if cands_n else 0.0),
                "n_states": top1_n,
            }

        # P_safe threshold sweep — overall + synthetic.
        sweep = []
        for thr in thresholds:
            pred_safe_set = p_safe >= thr
            ovr_n = int(pred_safe_set.sum())
            ovr_fs = int(np.sum(pred_safe_set & is_unsafe_true))
            syn_mask = np.array([m["dist"] == "synthetic_severe" for m in ev_meta]) & pred_safe_set
            syn_n = int(syn_mask.sum())
            syn_fs = int(np.sum(syn_mask & is_unsafe_true))
            sweep.append({
                "p_safe_threshold": thr,
                "act_rate_overall": ovr_n / len(p_safe),
                "unsafe_as_safe_overall": ovr_fs / ovr_n if ovr_n else 0.0,
                "act_rate_synthetic": syn_n / int(np.sum([m["dist"] == "synthetic_severe" for m in ev_meta])) if any(m["dist"] == "synthetic_severe" for m in ev_meta) else 0.0,
                "unsafe_as_safe_synthetic": syn_fs / syn_n if syn_n else 0.0,
            })

        seed_audits.append({
            "per_tool": per_tool_out, "per_u": per_u_out,
            "per_dist_label_ratio": per_dist_label, "per_dist_unsafe_as_safe": per_dist_unsafe,
            "calibration": calib,
            "pr_auc_unsafe": pr_auc_unsafe, "roc_auc_unsafe": roc_auc_unsafe,
            "confusion_unsafe_pos": {"TP": TP, "FP": FP, "FN_unsafe_as_safe": FN, "TN": TN,
                                      "precision_unsafe": prec_unsafe, "recall_unsafe": rec_unsafe, "f1_unsafe": f1_unsafe},
            "selected_vs_all": sel_vs_all,
            "threshold_sweep": sweep,
        })

    # Aggregate over seeds.
    def _agg_dict(key_chain, fn=lambda v: v):
        out = {}
        per_seed = [eval("a" + key_chain) for a in seed_audits]
        if isinstance(per_seed[0], dict):
            keys = set().union(*[d.keys() for d in per_seed])
            for k in keys:
                vals = [d.get(k) for d in per_seed if k in d]
                if isinstance(vals[0], dict):
                    sub_keys = set().union(*[v.keys() for v in vals])
                    out[k] = {sk: float(np.mean([v.get(sk, 0) for v in vals])) if all(isinstance(v.get(sk), (int, float)) for v in vals) else vals[0].get(sk) for sk in sub_keys}
                else:
                    out[k] = float(np.mean(vals)) if isinstance(vals[0], (int, float)) else vals[0]
        return out

    # Build aggregate output (means across seeds for scalar metrics).
    def _ms(vs):
        return {"mean": float(np.mean(vs)), "std": float(np.std(vs))}
    aggregate = {
        "pr_auc_unsafe": _ms([a["pr_auc_unsafe"] for a in seed_audits]),
        "roc_auc_unsafe": _ms([a["roc_auc_unsafe"] for a in seed_audits if a["roc_auc_unsafe"] is not None]),
        "confusion_unsafe_pos": {
            "TP": float(np.mean([a["confusion_unsafe_pos"]["TP"] for a in seed_audits])),
            "FP": float(np.mean([a["confusion_unsafe_pos"]["FP"] for a in seed_audits])),
            "FN_unsafe_as_safe": float(np.mean([a["confusion_unsafe_pos"]["FN_unsafe_as_safe"] for a in seed_audits])),
            "TN": float(np.mean([a["confusion_unsafe_pos"]["TN"] for a in seed_audits])),
            "precision_unsafe": _ms([a["confusion_unsafe_pos"]["precision_unsafe"] for a in seed_audits]),
            "recall_unsafe": _ms([a["confusion_unsafe_pos"]["recall_unsafe"] for a in seed_audits]),
            "f1_unsafe": _ms([a["confusion_unsafe_pos"]["f1_unsafe"] for a in seed_audits]),
        },
        "per_tool_unsafe_as_safe": {t: _ms([a["per_tool"].get(t, {"unsafe_as_safe_rate": 0})["unsafe_as_safe_rate"] for a in seed_audits])
                                     for t in TOOLS_ORDER},
        "per_u_unsafe_as_safe": {u: _ms([a["per_u"].get(u, {"unsafe_as_safe_rate": 0})["unsafe_as_safe_rate"] for a in seed_audits])
                                  for u in sorted({u for a in seed_audits for u in a["per_u"]})},
        "per_dist_unsafe_as_safe": {d: _ms([a["per_dist_unsafe_as_safe"].get(d, {"unsafe_as_safe_rate": 0})["unsafe_as_safe_rate"] for a in seed_audits])
                                     for d in ("clean", "near_boundary", "g2_natural", "synthetic_severe")},
        "per_dist_label_ratio_train_safe": {d: _ms([a["per_dist_label_ratio"][d]["train"]["safe_rate"] for a in seed_audits])
                                              for d in ("clean", "near_boundary", "g2_natural", "synthetic_severe")},
        "selected_vs_all": {d: {"all_cand_unsafe_rate": _ms([a["selected_vs_all"][d]["all_cand_unsafe_rate"] for a in seed_audits]),
                                "top1_unsafe_rate": _ms([a["selected_vs_all"][d]["top1_unsafe_rate"] for a in seed_audits]),
                                "lift": _ms([a["selected_vs_all"][d]["lift"] for a in seed_audits])}
                            for d in ("clean", "near_boundary", "g2_natural", "synthetic_severe")},
        "threshold_sweep": [
            {"p_safe_threshold": s,
             "act_rate_overall": _ms([a["threshold_sweep"][i]["act_rate_overall"] for a in seed_audits]),
             "unsafe_as_safe_overall": _ms([a["threshold_sweep"][i]["unsafe_as_safe_overall"] for a in seed_audits]),
             "act_rate_synthetic": _ms([a["threshold_sweep"][i]["act_rate_synthetic"] for a in seed_audits]),
             "unsafe_as_safe_synthetic": _ms([a["threshold_sweep"][i]["unsafe_as_safe_synthetic"] for a in seed_audits])}
            for i, s in enumerate(thresholds)
        ],
        # Per-seed calibration (mean over seeds at each bin).
        "calibration_curve": [
            {"bin": seed_audits[0]["calibration"][i]["bin"],
             "n_mean": float(np.mean([a["calibration"][i]["n"] for a in seed_audits])),
             "mean_predicted": (float(np.mean([a["calibration"][i]["mean_predicted"] for a in seed_audits if a["calibration"][i]["mean_predicted"] is not None]))
                                if any(a["calibration"][i]["mean_predicted"] is not None for a in seed_audits) else None),
             "actual_safe_rate": (float(np.mean([a["calibration"][i]["actual_safe_rate"] for a in seed_audits if a["calibration"][i]["actual_safe_rate"] is not None]))
                                   if any(a["calibration"][i]["actual_safe_rate"] is not None for a in seed_audits) else None)}
            for i in range(len(seed_audits[0]["calibration"]))
        ],
    }

    out = {
        "schema_version": "1.0.0", "record_type": "rl2_psafe_bias_audit",
        "task_id": "rl2_psafe_bias_audit_stageB0_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "rl2_psafe_bias_audit_stageB0_v1",
            oracle_type="action_effect_transition", action_grid="continuous-u",
            stage="RL-2-stageB0-psafe-audit",
            evidence_tier=[REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
            evaluators=list(DEFAULT_EVALUATORS), gate_evaluators=list(DEFAULT_PHYSICAL_GATE_EVALUATORS),
        ),
        "directive": "unsafe = positive class. safety: false-safe (unsafe_as_safe) 가 가장 위험. recall(unsafe) > acc.",
        "coarse_train_u": sorted(coarse_u),
        "seeds": seeds, "eval_frac": args.eval_frac,
        "thresholds": thresholds,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # === Figure: (a) calibration curve, (b) threshold sweep (synthetic + overall) ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Calibration.
    cc = aggregate["calibration_curve"]
    xs = [c["mean_predicted"] for c in cc if c["mean_predicted"] is not None]
    ys = [c["actual_safe_rate"] for c in cc if c["actual_safe_rate"] is not None]
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    axes[0].plot(xs, ys, "-o", color="#3182bd", linewidth=2, label="P_safe predicted vs actual")
    axes[0].set_xlabel("predicted P_safe (bin mean)"); axes[0].set_ylabel("actual safe rate")
    axes[0].set_title("(a) P_safe calibration curve"); axes[0].grid(alpha=0.3); axes[0].legend()
    # Threshold sweep.
    sw = aggregate["threshold_sweep"]
    ts = [s["p_safe_threshold"] for s in sw]
    uas_ovr = [s["unsafe_as_safe_overall"]["mean"] for s in sw]
    uas_syn = [s["unsafe_as_safe_synthetic"]["mean"] for s in sw]
    act_ovr = [s["act_rate_overall"]["mean"] for s in sw]
    act_syn = [s["act_rate_synthetic"]["mean"] for s in sw]
    axes[1].plot(ts, uas_syn, "-o", color="#e6550d", linewidth=2, label="unsafe_as_safe (synthetic)")
    axes[1].plot(ts, uas_ovr, "-o", color="#fdae6b", linewidth=2, label="unsafe_as_safe (overall)")
    axes[1].plot(ts, act_syn, "--s", color="#31a354", linewidth=2, label="act_rate (synthetic)")
    axes[1].plot(ts, act_ovr, "--s", color="#a1d99b", linewidth=2, label="act_rate (overall)")
    axes[1].set_xlabel("P_safe threshold"); axes[1].set_ylabel("rate")
    axes[1].set_title("(b) Threshold sweep — conservative safety trade-off")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)
    fig.suptitle("RL-2 Stage B-0 — P_safe Bias Audit (unsafe = positive class)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    args.output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output_fig), dpi=120); plt.close(fig)

    # Console summary.
    print("\n=== Stage B-0 P_safe Bias Audit (aggregate, 3 seeds) ===")
    print(f"  PR-AUC (unsafe=pos): {aggregate['pr_auc_unsafe']['mean']:.3f} ± {aggregate['pr_auc_unsafe']['std']:.3f}")
    cm = aggregate["confusion_unsafe_pos"]
    print(f"  Confusion @ P_safe>=0.5 (unsafe=pos): TP={cm['TP']:.0f} FP={cm['FP']:.0f} FN(unsafe-as-safe)={cm['FN_unsafe_as_safe']:.0f} TN={cm['TN']:.0f}")
    print(f"     precision_unsafe={cm['precision_unsafe']['mean']:.3f}  recall_unsafe={cm['recall_unsafe']['mean']:.3f}  F1={cm['f1_unsafe']['mean']:.3f}")
    print(f"\n  per-tool unsafe_as_safe (pred-safe→실제-unsafe 비율):")
    for t, v in aggregate["per_tool_unsafe_as_safe"].items():
        print(f"     {t:<26} {v['mean']*100:5.1f}% ± {v['std']*100:.1f}")
    print(f"\n  per-u unsafe_as_safe:")
    for u in sorted(aggregate["per_u_unsafe_as_safe"]):
        v = aggregate["per_u_unsafe_as_safe"][u]
        print(f"     u={u:<5} {v['mean']*100:5.1f}% ± {v['std']*100:.1f}")
    print(f"\n  per-dist unsafe_as_safe + train safe label ratio:")
    for d in ("clean", "near_boundary", "g2_natural", "synthetic_severe"):
        u = aggregate["per_dist_unsafe_as_safe"][d]
        l = aggregate["per_dist_label_ratio_train_safe"][d]
        print(f"     {d:<18} unsafe_as_safe={u['mean']*100:5.1f}%   train_safe_label_rate={l['mean']*100:5.1f}%")
    print(f"\n  selected-vs-all (argmax 가 위험 후보 끌어올리는가?):")
    for d, v in aggregate["selected_vs_all"].items():
        print(f"     {d:<18} all_cand_unsafe={v['all_cand_unsafe_rate']['mean']*100:5.1f}%  top1_unsafe={v['top1_unsafe_rate']['mean']*100:5.1f}%  lift={v['lift']['mean']*100:+5.1f}%")
    print(f"\n  threshold sweep (synthetic 영역 unsafe_as_safe vs act):")
    for s in aggregate["threshold_sweep"]:
        print(f"     P_safe>={s['p_safe_threshold']:.2f}  syn unsafe_as_safe={s['unsafe_as_safe_synthetic']['mean']*100:5.1f}%  syn act_rate={s['act_rate_synthetic']['mean']*100:5.1f}%  "
              f"ovr unsafe_as_safe={s['unsafe_as_safe_overall']['mean']*100:5.1f}%")
    print(f"\n[OK] audit -> {args.output}")
    print(f"[OK] fig   -> {args.output_fig}")


if __name__ == "__main__":
    main()
