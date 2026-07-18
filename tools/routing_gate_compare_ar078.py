"""AR-078: routing gate 5종 비교 (held-out benefit prediction) — 사전등록 spec 준수.

데이터 = 기존 박제물 2개 join (새 모델 실행 없음):
  state (pre-action): preaction_state_ar065_v1.csv
  label (ΔMM, split): routing_benefit_permotion_ar077_v1.csv

Gate: G1 generator-only / G2 skate-only / G3 mismatch-only /
      G4 richer-state(no-gen, LR) / G5 richer-state(+gen, LR).
LR 은 calibration rows 로만 학습 (표준화 통계 포함). holdout 평가 전용.

지표: benefit-AUC (prompt-bootstrap B=1000, multiplicity 보존, G4/G5−G2 paired diff CI)
      + harmful-apply@적용률 {10,20,30,33,40,50}% (selective prediction).

판정 (사전 고정): 개선 지지 = (G4 or G5) vs G2 — AUC diff CI 하한>0 AND
적용률 30% harmful-apply 감소 (bootstrap diff CI). 아니면 한계 확정.

CLI (motion3d env):
    python tools/routing_gate_compare_ar078.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAP = REPO_ROOT / "evals" / "snapshots"
STATE_CSV = SNAP / "preaction_state_ar065_v1.csv"
LABEL_CSV = SNAP / "routing_benefit_permotion_ar077_v1.csv"
OUT = SNAP / "routing_gate_compare_ar078_v1.json"

APPLY_RATES = (0.10, 0.20, 0.30, 0.33, 0.40, 0.50)
N_BOOT = 1000
JITTER_SEED = 20260725   # 동률 순서 결정 (사전 고정)
GENS = ("mdm", "motiongpt", "momask")
INTENTS = ("locomotion", "in_place", "ambiguous")


def _auc(score, label):
    score = np.asarray(score, float); label = np.asarray(label).astype(int)
    if label.sum() == 0 or label.sum() == len(label):
        return float("nan")
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label == 1].sum() - label.sum() * (label.sum() + 1) / 2)
                 / (label.sum() * (label == 0).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()

    # ---- join (gen,sid,seed).
    state = {(r["gen"], r["sid"], r["seed"]): r
             for r in csv.DictReader(open(STATE_CSV, encoding="utf-8"))}
    rows = []
    n_unjoined = 0
    for r in csv.DictReader(open(LABEL_CSV, encoding="utf-8")):
        k = (r["gen"], r["sid"], r["seed"])
        if k not in state:
            n_unjoined += 1
            continue
        s = state[k]
        rows.append({
            "gen": r["gen"], "sid": r["sid"], "seed": r["seed"],
            "dmm": float(r["dmm"]), "split": r["split"],
            "foot_skate": float(s["foot_skate"]),
            "mismatch": float(s["root_gait_mismatch"]),
            "mismatch_ratio": min(float(s["mismatch_ratio"]), 10.0),  # clip (분모≈0 폭주 방지)
            "contact_run_mean": float(s["contact_run_mean"]),
            "contact_fraction": float(s["contact_fraction"]),
            "path_length": float(s["path_length"]),
            "straightness": float(s["straightness"]),
            "intent": s["locomotion_intent"],
        })
    N = len(rows)
    dmm = np.array([r["dmm"] for r in rows])
    gens = np.array([r["gen"] for r in rows])
    sids = np.array([r["sid"] for r in rows])
    is_calib = np.array([r["split"] == "calib" for r in rows])
    ho = ~is_calib

    # ---- δ = calibration VQ |ΔMM| median (AR-077 재현).
    vq = np.isin(gens, ["motiongpt", "momask"])
    delta = float(np.median(np.abs(dmm[vq & is_calib])))
    benefit = (dmm < -delta).astype(int)
    harm = (dmm > delta).astype(int)

    # ---- feature matrix (전부 pre-action).
    geo_cols = ("foot_skate", "mismatch", "mismatch_ratio", "contact_run_mean",
                "contact_fraction", "path_length", "straightness")
    X_geo = np.array([[r[c] for c in geo_cols] for r in rows])
    X_int = np.array([[1.0 if r["intent"] == b else 0.0 for b in INTENTS] for r in rows])
    X_gen = np.array([[1.0 if r["gen"] == g else 0.0 for g in GENS] for r in rows])
    X4 = np.hstack([X_geo, X_int])            # richer, no-gen
    X5 = np.hstack([X_geo, X_int, X_gen])     # richer, +gen

    def _fit_lr(X):
        mu = X[is_calib].mean(0); sd = X[is_calib].std(0) + 1e-9
        Z = (X - mu) / sd
        lr = LogisticRegression(max_iter=2000)
        lr.fit(Z[is_calib], benefit[is_calib])
        return lr.predict_proba(Z)[:, 1], lr

    score4, lr4 = _fit_lr(X4)
    score5, lr5 = _fit_lr(X5)
    scores = {
        "G1_generator_only": (gens == "mdm").astype(float),
        "G2_skate_only": np.array([r["foot_skate"] for r in rows]),
        "G3_mismatch_only": np.array([r["mismatch"] for r in rows]),
        "G4_richer_nogen": score4,
        "G5_richer_gen": score5,
    }

    # ---- holdout AUC + prompt-bootstrap (multiplicity 보존, 전 gate 동일 draw = paired).
    ho_sids = np.array(sorted(set(sids[ho])))
    rows_by_sid = {s: np.where((sids == s) & ho)[0] for s in ho_sids}
    rng = np.random.default_rng(20260726)
    boot_auc = {g: [] for g in scores}
    for _ in range(N_BOOT):
        draw = rng.choice(ho_sids, len(ho_sids), replace=True)
        idx = np.concatenate([rows_by_sid[s] for s in draw])
        if not (0 < benefit[idx].sum() < len(idx)):
            continue
        for g, sc in scores.items():
            boot_auc[g].append(_auc(sc[idx], benefit[idx]))
    def _ci(a):
        return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    auc_res = {g: {"auc": round(_auc(sc[ho], benefit[ho]), 3), "ci95": _ci(boot_auc[g])}
               for g, sc in scores.items()}
    # paired diff vs G2 (동일 draw).
    diff_res = {}
    for g in ("G4_richer_nogen", "G5_richer_gen", "G3_mismatch_only", "G1_generator_only"):
        d = np.array(boot_auc[g]) - np.array(boot_auc["G2_skate_only"])
        diff_res[f"{g}_minus_G2"] = {"mean": round(float(d.mean()), 3), "ci95": _ci(d)}

    # ---- harmful-apply@적용률 (holdout; 동률 = 고정 jitter).
    jit = np.random.default_rng(JITTER_SEED).random(N) * 1e-9
    n_ho = int(ho.sum())
    ho_idx = np.where(ho)[0]
    rate_res = {}
    for g, sc in scores.items():
        order = ho_idx[np.argsort(-(sc[ho] + jit[ho]))]
        rate_res[g] = {}
        for rate in APPLY_RATES:
            k = max(1, int(round(rate * n_ho)))
            ap_ = order[:k]
            rate_res[g][f"{int(rate*100)}%"] = {
                "harmful_apply": round(float(harm[ap_].mean()), 3),
                "non_beneficial": round(float(1 - benefit[ap_].mean()), 3),
                "benefit_capture": round(float(benefit[ap_].sum() / max(benefit[ho].sum(), 1)), 3),
            }
    # 적용률 30% harmful-apply bootstrap diff (G4/G5 − G2, 동일 draw).
    harm30_diff = {}
    for g in ("G4_richer_nogen", "G5_richer_gen"):
        ds = []
        for _ in range(N_BOOT):
            draw = rng.choice(ho_sids, len(ho_sids), replace=True)
            idx = np.concatenate([rows_by_sid[s] for s in draw])
            k = max(1, int(round(0.30 * len(idx))))
            vals = {}
            for gg in (g, "G2_skate_only"):
                o = idx[np.argsort(-(scores[gg][idx] + jit[idx]))][:k]
                vals[gg] = float(harm[o].mean())
            ds.append(vals[g] - vals["G2_skate_only"])
        harm30_diff[f"{g}_minus_G2"] = {"mean": round(float(np.mean(ds)), 3), "ci95": _ci(np.array(ds))}

    # ---- 판정 (사전 고정 기준).
    def _supported(g):
        a = diff_res[f"{g}_minus_G2"]["ci95"][0] > 0
        b = harm30_diff[f"{g}_minus_G2"]["ci95"][1] < 0
        return a and b, {"auc_diff_ci_lower_gt_0": bool(a), "harm30_diff_ci_upper_lt_0": bool(b)}
    sup4, det4 = _supported("G4_richer_nogen")
    sup5, det5 = _supported("G5_richer_gen")
    verdict = ("개선 지지 (richer state)" if (sup4 or sup5)
               else "한계 확정 — 현 state 로 per-motion 결정 불충분 (generator-level rule 이 현재 한계)")

    out = {
        "schema_version": "1.0.0", "record_type": "routing_gate_compare", "board_id": "AR-078",
        "n_joined": N, "n_unjoined": n_unjoined, "n_holdout": n_ho,
        "delta": round(delta, 4),
        "base_rates_holdout": {"benefit": round(float(benefit[ho].mean()), 3),
                               "harm": round(float(harm[ho].mean()), 3)},
        "holdout_benefit_auc": auc_res,
        "paired_auc_diff_vs_G2": diff_res,
        "harmful_apply_at_rate": rate_res,
        "harm30_bootstrap_diff_vs_G2": harm30_diff,
        "preregistered_verdict": {"verdict": verdict,
                                  "G4_detail": det4, "G5_detail": det5,
                                  "criterion": "AUC diff CI 하한>0 AND 적용률30% harmful-apply diff CI 상한<0"},
        "lr_coef_calib": {"G4": dict(zip(list(geo_cols) + list(INTENTS),
                                         [round(float(c), 3) for c in lr4.coef_[0]])),
                          "G5": dict(zip(list(geo_cols) + list(INTENTS) + list(GENS),
                                         [round(float(c), 3) for c in lr5.coef_[0]]))},
        "claim_boundary": "δ 조건부 exploratory (AR-077 승계). MM-Dist proxy — 지각 아님. LR=투명 결합기 "
                          "(정책 성능 일반 주장 금지). 단일 벤치마크. '라우팅 작동' 표현은 개선 지지 + AR-073 후.",
        "grounding": ["Gangrade AISTATS 2021 (selective prediction)", "Guo HumanML3D CVPR2022"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    import sys as _s; _s.stdout.reconfigure(encoding="utf-8")
    print(f"join {N} (unjoined {n_unjoined}) | holdout {n_ho} | δ={delta:.4f} | base benefit {benefit[ho].mean():.3f} harm {harm[ho].mean():.3f}")
    for g in scores:
        r30 = rate_res[g]["30%"]
        print(f"{g:<20} AUC {auc_res[g]['auc']} ci{auc_res[g]['ci95']} | @30%: harm {r30['harmful_apply']} capture {r30['benefit_capture']}")
    for k, v in diff_res.items():
        print(f"  AUCdiff {k}: {v['mean']} ci{v['ci95']}")
    for k, v in harm30_diff.items():
        print(f"  harm@30 diff {k}: {v['mean']} ci{v['ci95']}")
    print(f"VERDICT: {verdict}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
