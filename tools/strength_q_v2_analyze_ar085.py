"""AR-085 2단계: H-A (over-correction 효과) + H-B (v2 정책 vs v1) — 사전등록 판정.

H-A: u_grid {0..2.0} 에서 per-motion argmax-u 분포 + improvement(u) 곡선.
     지지 = locomotion holdout argmax-u>1 비율>=15% AND best-u improvement 가
     u=1 대비 유의 우위 (prompt-bootstrap CI 하한>0).
H-B: 확장 grid additive Q(s,u) 재학습 → v2 정책 vs v1 정책(u≤1) holdout 비교.

전 수치 δ 조건부 exploratory (MM proxy).

CLI (motion3d env):
    python tools/strength_q_v2_analyze_ar085.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))
from tools.strength_q_train_ar081 import AdditiveQ, GROUPS  # additive 구조 재사용

DATA = REPO_ROOT / "evals" / "snapshots" / "strength_q_v2_dataset_ar085_v1.csv"
OUT = REPO_ROOT / "evals" / "snapshots" / "strength_q_v2_result_ar085_v1.json"
U_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
U_COL = {0.0: "mm_u000", 0.25: "mm_u025", 0.5: "mm_u050", 0.75: "mm_u075",
         1.0: "mm_u100", 1.25: "mm_u125", 1.5: "mm_u150", 2.0: "mm_u200"}
U_GRID_V1 = (0.0, 0.25, 0.5, 0.75, 1.0)
LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0)
N_BOOT = 1000
LOCO = ("wants_to_travel",)  # intent one-hot (직관 이름)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    N = len(rows)
    gen = np.array([r["gen"] for r in rows]); sid = np.array([r["sid"] for r in rows])
    calib = np.array([r["split"] == "calib" for r in rows]); ho = ~calib
    is_loco = np.array([r["wants_to_travel"] == "1" for r in rows])
    mm = {u: np.array([float(r[c]) for r in rows]) for u, c in U_COL.items()}
    imp = {u: mm[0.0] - mm[u] for u in U_GRID}
    vq = np.isin(gen, ["motiongpt", "momask"])
    delta = float(np.median(np.abs(mm[1.0] - mm[0.0])[vq & calib]))
    harm = {u: ((mm[u] - mm[0.0]) > delta).astype(float) for u in U_GRID}

    # ---- H-A: argmax-u (grid), improvement 곡선. (전체 유틸 = improvement 만; harm 은 별도 정책서 처리)
    imp_mat = np.column_stack([imp[u] for u in U_GRID])          # [N, 8]
    argmax_u = np.array(U_GRID)[np.argmax(imp_mat, axis=1)]
    ho_loco = ho & is_loco
    # δ 노이즈 위: u=1 대비 유의 개선인 u 만 인정하는 tie-break (동률 시 작은 u).
    frac_over1 = float((argmax_u[ho_loco] > 1.0).mean())
    curve = {str(u): {"mean_imp": round(float(imp[u][ho_loco].mean()), 4),
                      "median_imp": round(float(np.median(imp[u][ho_loco])), 4),
                      "harm_rate": round(float(harm[u][ho_loco].mean()), 4)} for u in U_GRID}
    # best-u(per-motion, holdout-loco) 실현 improvement vs 고정 u=1.0 — paired bootstrap (sid).
    best_imp = imp_mat[np.arange(N), np.argmax(imp_mat, axis=1)]
    ho_l_idx = np.where(ho_loco)[0]
    ho_sids = np.array(sorted(set(sid[ho_loco])))
    rows_by_sid = {s: np.where((sid == s) & ho_loco)[0] for s in ho_sids}
    rng = np.random.default_rng(20260730)
    d_best = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rows_by_sid[s] for s in rng.choice(ho_sids, len(ho_sids), replace=True)])
        d_best.append(float(best_imp[idx].mean() - imp[1.0][idx].mean()))
    def _ci(a): return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    best_ci = _ci(np.array(d_best))
    HA = {"frac_argmax_u_over_1_holdout_loco": round(frac_over1, 3),
          "best_u_minus_u1_improvement": {"mean": round(float(np.mean(d_best)), 4), "ci95": best_ci},
          "improvement_curve": curve,
          "argmax_u_dist_holdout_loco": {str(u): round(float((argmax_u[ho_loco] == u).mean()), 3) for u in U_GRID},
          "by_gen_frac_over1_holdout": {g: round(float((argmax_u[ho & is_loco & (gen == g)] > 1.0).mean()), 3)
                                         for g in ("mdm", "motiongpt", "momask")},
          "verdict": ("지지 (u>1 이득 존재)" if (frac_over1 >= 0.15 and best_ci[0] > 0)
                      else "기각 (u=1 이 ceiling)" if frac_over1 < 0.05
                      else "부분")}

    # ---- H-B: 확장 grid Q vs v1 grid Q. (train_ar081 의 additive 구조 사용, u_grid 만 교체)
    feat_cols = sum(GROUPS.values(), [])
    raw = {c: np.array([float(r[c]) for r in rows]) for c in feat_cols}
    mu = {c: raw[c][calib].mean() for c in feat_cols}; sd = {c: raw[c][calib].std() + 1e-9 for c in feat_cols}
    F = {c: (raw[c] - mu[c]) / sd[c] for c in feat_cols}

    def train_and_eval(grid):
        ci = np.where(calib)[0]
        Fx = {c: np.concatenate([F[c][ci] for _ in grid]) for c in feat_cols}
        uc = np.concatenate([np.full(len(ci), u) for u in grid])
        yi = np.concatenate([imp[u][ci] for u in grid]); yh = np.concatenate([harm[u][ci] for u in grid])
        qi = AdditiveQ(GROUPS).fit(Fx, uc, yi); qh = AdditiveQ(GROUPS, seed=1).fit(Fx, uc, yh)
        # λ 선택 (calib): harmful<=고정u1(calib) 중 improvement 최대.
        def choose(mask, lam):
            idx = np.where(mask)[0]; qs = np.zeros((len(idx), len(grid)))
            for j, u in enumerate(grid):
                Fi = {c: F[c][idx] for c in feat_cols}; uv = np.full(len(idx), u)
                qs[:, j] = qi.predict(Fi, uv) - lam * np.clip(qh.predict(Fi, uv), 0, 1)
            ch = np.array(grid)[np.argmax(qs, axis=1)]
            return idx, ch, np.array([imp[u][i] for u, i in zip(ch, idx)]), np.array([harm[u][i] for u, i in zip(ch, idx)])
        base_h = float(harm[1.0][calib].mean()); best = None
        for lam in LAMBDAS:
            _, _, ri, rh = choose(calib, lam)
            if rh.mean() <= base_h and (best is None or ri.mean() > best[1]):
                best = (lam, ri.mean())
        lam = best[0] if best else 4.0
        idx, ch, ri, rh = choose(ho, lam)
        return {"lam": lam, "idx": idx, "imp": ri, "harm": rh,
                "u_dist": {str(u): round(float((ch == u).mean()), 3) for u in grid}}

    v2 = train_and_eval(U_GRID); v1 = train_and_eval(U_GRID_V1)
    # paired bootstrap (holdout sid) v2-v1.
    ho_all_sids = np.array(sorted(set(sid[ho])))
    pos = {i: k for k, i in enumerate(np.where(ho)[0])}
    rbs = {s: np.array([pos[i] for i in np.where((sid == s) & ho)[0]]) for s in ho_all_sids}
    di, dh = [], []
    for _ in range(N_BOOT):
        p = np.concatenate([rbs[s] for s in rng.choice(ho_all_sids, len(ho_all_sids), replace=True)])
        di.append(float(v2["imp"][p].mean() - v1["imp"][p].mean())); dh.append(float(v2["harm"][p].mean() - v1["harm"][p].mean()))
    ci_i, ci_h = _ci(np.array(di)), _ci(np.array(dh))
    HB = {"v2_grid": {"mean_imp": round(float(v2["imp"].mean()), 4), "harm": round(float(v2["harm"].mean()), 4),
                      "lam": v2["lam"], "u_dist": v2["u_dist"]},
          "v1_grid": {"mean_imp": round(float(v1["imp"].mean()), 4), "harm": round(float(v1["harm"].mean()), 4),
                      "lam": v1["lam"], "u_dist": v1["u_dist"]},
          "v2_minus_v1": {"improvement": {"mean": round(float(np.mean(di)), 4), "ci95": ci_i},
                          "harm": {"mean": round(float(np.mean(dh)), 4), "ci95": ci_h}},
          "verdict": ("지지 (v2 정책 개선)" if (ci_i[0] > 0 and ci_h[1] <= 0.005) else "한계")}

    out = {"schema_version": "1.0.0", "record_type": "strength_q_v2_result", "board_id": "AR-085",
           "u_grid": list(U_GRID), "u_max": 2.0, "delta": round(delta, 4), "n": N, "n_holdout_loco": int(ho_loco.sum()),
           "H_A_over_correction": HA, "H_B_policy": HB,
           "claim_boundary": "δ 조건부 exploratory·MM proxy·one-shot. 지각/GT-최적 주장 없음. v1(u≤1) 무수정.",
           "grounding": ["Agarwal NeurIPS 2021", "Gangrade AISTATS 2021", "Guo CVPR2022"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    import sys as _s; _s.stdout.reconfigure(encoding="utf-8")
    print("=== H-A over-correction ===")
    print(f"argmax-u>1 (holdout loco): {HA['frac_argmax_u_over_1_holdout_loco']} | by gen {HA['by_gen_frac_over1_holdout']}")
    print(f"best-u vs u=1 imp: {HA['best_u_minus_u1_improvement']}")
    print("imp curve:", {u: c["mean_imp"] for u, c in HA["improvement_curve"].items()})
    print(f"H-A VERDICT: {HA['verdict']}")
    print("=== H-B policy ===")
    print(f"v2 imp {HB['v2_grid']['mean_imp']} harm {HB['v2_grid']['harm']} | v1 imp {HB['v1_grid']['mean_imp']} harm {HB['v1_grid']['harm']}")
    print(f"v2-v1 imp {HB['v2_minus_v1']['improvement']} harm {HB['v2_minus_v1']['harm']}")
    print(f"v2 u_dist {HB['v2_grid']['u_dist']}")
    print(f"H-B VERDICT: {HB['verdict']}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
