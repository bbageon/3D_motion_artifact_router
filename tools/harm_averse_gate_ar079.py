"""AR-079: harm-averse selective gate — harm 을 직접 예측·회피 (사전등록 spec 준수).

AR-078 발견(순위≠안전 축 분리)의 후속: benefit 분류 대신 **harm 라벨(ΔMM>+δ)을 직접
예측**하는 H-LR 을 만들고, HA score = P(benefit) − λ·P(harm) 로 결합한다.
λ 는 **calibration 에서만** 선택 (calib harm@30 최소화 s.t. calib capture@30 ≥ 0.8×G5-calib).

판정 (사전 고정): 지지 = holdout 에서 harm@30(HA)−harm@30(G5) diff CI 상한<0
AND capture@30(HA) ≥ 0.8×capture@30(G5). 아니면 기각/한계 (그 자체로 유효).

CLI (motion3d env):
    python tools/harm_averse_gate_ar079.py
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
OUT = SNAP / "harm_averse_gate_ar079_v1.json"

LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)
APPLY_RATES = (0.10, 0.20, 0.30, 0.40, 0.50)
N_BOOT = 1000
JITTER_SEED = 20260725
GENS = ("mdm", "motiongpt", "momask")
INTENTS = ("locomotion", "in_place", "ambiguous")


def _auc(score, label):
    score = np.asarray(score, float); label = np.asarray(label).astype(int)
    if label.sum() == 0 or label.sum() == len(label):
        return float("nan")
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label == 1].sum() - label.sum() * (label.sum() + 1) / 2)
                 / (label.sum() * (label == 0).sum()))


def _at_rate(score, jit, idx, rate, harm, benefit):
    k = max(1, int(round(rate * len(idx))))
    ap = idx[np.argsort(-(score[idx] + jit[idx]))][:k]
    return (float(harm[ap].mean()), float(benefit[ap].sum() / max(benefit[idx].sum(), 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()

    state = {(r["gen"], r["sid"], r["seed"]): r
             for r in csv.DictReader(open(STATE_CSV, encoding="utf-8"))}
    rows = []
    for r in csv.DictReader(open(LABEL_CSV, encoding="utf-8")):
        k = (r["gen"], r["sid"], r["seed"])
        if k not in state:
            continue
        s = state[k]
        rows.append({
            "gen": r["gen"], "sid": r["sid"], "dmm": float(r["dmm"]), "split": r["split"],
            "f": [float(s["foot_skate"]), float(s["root_gait_mismatch"]),
                  min(float(s["mismatch_ratio"]), 10.0), float(s["contact_run_mean"]),
                  float(s["contact_fraction"]), float(s["path_length"]), float(s["straightness"])],
            "intent": s["locomotion_intent"],
        })
    N = len(rows)
    dmm = np.array([r["dmm"] for r in rows])
    gens = np.array([r["gen"] for r in rows])
    sids = np.array([r["sid"] for r in rows])
    is_calib = np.array([r["split"] == "calib" for r in rows])
    ho = ~is_calib

    vq = np.isin(gens, ["motiongpt", "momask"])
    delta = float(np.median(np.abs(dmm[vq & is_calib])))
    benefit = (dmm < -delta).astype(int)
    harm = (dmm > delta).astype(int)

    X = np.hstack([
        np.array([r["f"] for r in rows]),
        np.array([[1.0 if r["intent"] == b else 0.0 for b in INTENTS] for r in rows]),
        np.array([[1.0 if r["gen"] == g else 0.0 for g in GENS] for r in rows]),
    ])
    mu = X[is_calib].mean(0); sd = X[is_calib].std(0) + 1e-9
    Z = (X - mu) / sd

    lr_b = LogisticRegression(max_iter=2000); lr_b.fit(Z[is_calib], benefit[is_calib])
    lr_h = LogisticRegression(max_iter=2000); lr_h.fit(Z[is_calib], harm[is_calib])
    p_b = lr_b.predict_proba(Z)[:, 1]
    p_h = lr_h.predict_proba(Z)[:, 1]
    skate = X[:, 0]  # G2 참조 (표준화 전 foot_skate 는 첫 feature — 순위 동일하므로 Z 첫 열도 무방)

    jit = np.random.default_rng(JITTER_SEED).random(N) * 1e-9

    # ---- λ 선택: calibration 에서만.
    calib_idx = np.where(is_calib)[0]
    g5_calib_harm, g5_calib_cap = _at_rate(p_b, jit, calib_idx, 0.30, harm, benefit)
    best_lam, best_harm = None, None
    lam_table = {}
    for lam in LAMBDAS:
        sc = p_b - lam * p_h
        h30, c30 = _at_rate(sc, jit, calib_idx, 0.30, harm, benefit)
        ok = c30 >= 0.8 * g5_calib_cap
        lam_table[str(lam)] = {"calib_harm30": round(h30, 3), "calib_capture30": round(c30, 3),
                               "capture_constraint_ok": bool(ok)}
        if ok and (best_harm is None or h30 < best_harm):
            best_harm, best_lam = h30, lam
    if best_lam is None:   # 제약 만족 λ 없음 → 제약 무시 최소 harm (기록)
        best_lam = min(LAMBDAS, key=lambda l: lam_table[str(l)]["calib_harm30"])
    score_ha = p_b - best_lam * p_h

    # ---- holdout 평가.
    ho_idx = np.where(ho)[0]
    gates = {"G2_skate": skate, "G5_benefit_LR": p_b, "HA_harm_averse": score_ha,
             "Hinv_low_harm_only": -p_h}
    rate_res = {g: {f"{int(r*100)}%": None for r in APPLY_RATES} for g in gates}
    for g, sc in gates.items():
        for r in APPLY_RATES:
            h, c = _at_rate(sc, jit, ho_idx, r, harm, benefit)
            rate_res[g][f"{int(r*100)}%"] = {"harm": round(h, 3), "capture": round(c, 3)}

    harm_auc_ho = round(_auc(p_h[ho], harm[ho]), 3)
    benefit_auc_ha = round(_auc(score_ha[ho], benefit[ho]), 3)

    # bootstrap: harm@30 diff (HA − G5, HA − G2), 동일 draw.
    ho_sids = np.array(sorted(set(sids[ho])))
    rows_by_sid = {s: np.where((sids == s) & ho)[0] for s in ho_sids}
    rng = np.random.default_rng(20260727)
    d_g5, d_g2, harm_auc_boot = [], [], []
    for _ in range(N_BOOT):
        draw = rng.choice(ho_sids, len(ho_sids), replace=True)
        idx = np.concatenate([rows_by_sid[s] for s in draw])
        h_ha, _ = _at_rate(score_ha, jit, idx, 0.30, harm, benefit)
        h_g5, _ = _at_rate(p_b, jit, idx, 0.30, harm, benefit)
        h_g2, _ = _at_rate(skate, jit, idx, 0.30, harm, benefit)
        d_g5.append(h_ha - h_g5); d_g2.append(h_ha - h_g2)
        if 0 < harm[idx].sum() < len(idx):
            harm_auc_boot.append(_auc(p_h[idx], harm[idx]))
    def _ci(a):
        return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]

    ha30 = rate_res["HA_harm_averse"]["30%"]; g530 = rate_res["G5_benefit_LR"]["30%"]
    cond_a = _ci(np.array(d_g5))[1] < 0
    cond_b = ha30["capture"] >= 0.8 * g530["capture"]
    verdict = ("지지 (harm-averse gate 가 harm 을 유의하게 낮춤, capture 유지)"
               if (cond_a and cond_b) else
               "기각/한계 — " + ("harm 유의 감소 실패" if not cond_a else "capture 대가 과다"))

    out = {
        "schema_version": "1.0.0", "record_type": "harm_averse_gate", "board_id": "AR-079",
        "n": N, "n_holdout": int(ho.sum()), "delta": round(delta, 4),
        "harm_predictability": {"harm_auc_holdout": harm_auc_ho, "ci95": _ci(np.array(harm_auc_boot)),
                                "note": "≤0.55 면 harm 자체가 현 state 로 예측 곤란"},
        "lambda_selection_calib_only": {"table": lam_table, "chosen": best_lam,
                                        "g5_calib_ref": {"harm30": round(g5_calib_harm, 3),
                                                          "capture30": round(g5_calib_cap, 3)}},
        "holdout_at_rate": rate_res,
        "ha_benefit_auc_holdout": benefit_auc_ha,
        "harm30_diff_bootstrap": {"HA_minus_G5": {"mean": round(float(np.mean(d_g5)), 3), "ci95": _ci(np.array(d_g5))},
                                   "HA_minus_G2": {"mean": round(float(np.mean(d_g2)), 3), "ci95": _ci(np.array(d_g2))}},
        "preregistered_verdict": {"verdict": verdict,
                                  "cond_a_harm_sig_lower_vs_G5": bool(cond_a),
                                  "cond_b_capture_ge_0.8xG5": bool(cond_b)},
        "claim_boundary": "δ 조건부 exploratory. MM-Dist proxy (지각 아님). λ 는 calib 전용 선택. 단일 벤치마크.",
        "grounding": ["Gangrade AISTATS 2021 (selective prediction)", "Guo HumanML3D CVPR2022"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    import sys as _s; _s.stdout.reconfigure(encoding="utf-8")
    print(f"harm-AUC(holdout) {harm_auc_ho} ci{out['harm_predictability']['ci95']} | λ={best_lam}")
    for g in gates:
        r30 = rate_res[g]["30%"]
        print(f"{g:<20} @30%: harm {r30['harm']} capture {r30['capture']}")
    print(f"harm@30 diff HA−G5 {out['harm30_diff_bootstrap']['HA_minus_G5']} | HA−G2 {out['harm30_diff_bootstrap']['HA_minus_G2']}")
    print(f"VERDICT: {verdict}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
