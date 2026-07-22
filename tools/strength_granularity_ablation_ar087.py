"""AR-087: strength 세분화가 필요한가 — binary{STOP,full} vs graded 정책 ablation.

사용자 질문 "굳이 strength 조절할 필요 있나"에 직접 답. 동일 additive Q 기계로
grid 만 교체해 holdout 성과 비교:
  - binary : u ∈ {0, 1.0}          (고칠까 말까 = 결정 A 만)
  - v1     : u ∈ {0,.25,.5,.75,1}  (+ 중간 강도)
  - v2     : u ∈ {0..2.0}          (+ over-correction)

판정 (사전 고정): graded 가 binary 대비 (improvement CI-clean 우위) OR (harmful
CI-clean 감소) 면 "세분화 유의미"; 둘 다 CI 0 포함이면 "binary 충분 (강도 조절 불필요,
가치는 apply/STOP 에 있음)".

전 수치 δ 조건부 exploratory·MM proxy·one-shot.
CLI (motion3d env):
    python tools/strength_granularity_ablation_ar087.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))
from tools.strength_q_train_ar081 import AdditiveQ, GROUPS

DATA = REPO_ROOT / "evals" / "snapshots" / "strength_q_v2_dataset_ar085_v1.csv"
OUT = REPO_ROOT / "evals" / "snapshots" / "strength_granularity_ablation_ar087_v1.json"
U_COL = {0.0: "mm_u000", 0.25: "mm_u025", 0.5: "mm_u050", 0.75: "mm_u075",
         1.0: "mm_u100", 1.25: "mm_u125", 1.5: "mm_u150", 2.0: "mm_u200"}
GRIDS = {"binary_stop_full": (0.0, 1.0),
         "graded_v1": (0.0, 0.25, 0.5, 0.75, 1.0),
         "graded_v2_overcorr": (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)}
LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0)
N_BOOT = 1000


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    gen = np.array([r["gen"] for r in rows]); sid = np.array([r["sid"] for r in rows])
    calib = np.array([r["split"] == "calib" for r in rows]); ho = ~calib
    mm = {u: np.array([float(r[c]) for r in rows]) for u, c in U_COL.items()}
    imp = {u: mm[0.0] - mm[u] for u in U_COL}
    vq = np.isin(gen, ["motiongpt", "momask"])
    delta = float(np.median(np.abs(mm[1.0] - mm[0.0])[vq & calib]))
    harm = {u: ((mm[u] - mm[0.0]) > delta).astype(float) for u in U_COL}
    feat_cols = sum(GROUPS.values(), [])
    raw = {c: np.array([float(r[c]) for r in rows]) for c in feat_cols}
    mu = {c: raw[c][calib].mean() for c in feat_cols}; sd = {c: raw[c][calib].std() + 1e-9 for c in feat_cols}
    F = {c: (raw[c] - mu[c]) / sd[c] for c in feat_cols}

    def train_eval(grid):
        ci = np.where(calib)[0]
        Fx = {c: np.concatenate([F[c][ci] for _ in grid]) for c in feat_cols}
        uc = np.concatenate([np.full(len(ci), u) for u in grid])
        qi = AdditiveQ(GROUPS).fit(Fx, uc, np.concatenate([imp[u][ci] for u in grid]))
        qh = AdditiveQ(GROUPS, seed=1).fit(Fx, uc, np.concatenate([harm[u][ci] for u in grid]))
        def choose(mask, lam):
            idx = np.where(mask)[0]; qs = np.zeros((len(idx), len(grid)))
            for j, u in enumerate(grid):
                Fi = {c: F[c][idx] for c in feat_cols}; uv = np.full(len(idx), u)
                qs[:, j] = qi.predict(Fi, uv) - lam * np.clip(qh.predict(Fi, uv), 0, 1)
            ch = np.array(grid)[np.argmax(qs, axis=1)]
            return idx, ch, np.array([imp[u][i] for u, i in zip(ch, idx)]), np.array([harm[u][i] for u, i in zip(ch, idx)])
        base = float(harm[1.0][calib].mean()); best = None
        for lam in LAMBDAS:
            _, _, ri, rh = choose(calib, lam)
            if rh.mean() <= base and (best is None or ri.mean() > best[1]):
                best = (lam, ri.mean())
        lam = best[0] if best else 4.0
        idx, ch, ri, rh = choose(ho, lam)
        return {"lam": lam, "idx": idx, "imp": ri, "harm": rh,
                "u_dist": {str(u): round(float((ch == u).mean()), 3) for u in grid}}

    res = {name: train_eval(grid) for name, grid in GRIDS.items()}
    # paired bootstrap vs binary (holdout sid).
    ho_sids = np.array(sorted(set(sid[ho]))); pos = {i: k for k, i in enumerate(np.where(ho)[0])}
    rbs = {s: np.array([pos[i] for i in np.where((sid == s) & ho)[0]]) for s in ho_sids}
    rng = np.random.default_rng(20260801)
    def _ci(a): return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    diffs = {}
    b = res["binary_stop_full"]
    for name in ("graded_v1", "graded_v2_overcorr"):
        g = res[name]; di, dh = [], []
        for _ in range(N_BOOT):
            p = np.concatenate([rbs[s] for s in rng.choice(ho_sids, len(ho_sids), replace=True)])
            di.append(float(g["imp"][p].mean() - b["imp"][p].mean())); dh.append(float(g["harm"][p].mean() - b["harm"][p].mean()))
        ci_i, ci_h = _ci(np.array(di)), _ci(np.array(dh))
        useful = (ci_i[0] > 0) or (ci_h[1] < 0)
        diffs[f"{name}_minus_binary"] = {
            "improvement": {"mean": round(float(np.mean(di)), 4), "ci95": ci_i},
            "harm": {"mean": round(float(np.mean(dh)), 4), "ci95": ci_h},
            "graded_useful_over_binary": bool(useful)}

    v1_useful = diffs["graded_v1_minus_binary"]["graded_useful_over_binary"]
    verdict = ("세분화(graded) 유의미 — 강도 조절이 binary 위에 가치 추가" if v1_useful
               else "binary 충분 — 연속 강도 조절 불필요; 가치는 apply/STOP(결정 A)에 있음")

    out = {"schema_version": "1.0.0", "record_type": "strength_granularity_ablation", "board_id": "AR-087",
           "question": "굳이 strength(연속 강도) 조절이 필요한가 — binary{STOP,full} vs graded",
           "delta": round(delta, 4), "n_holdout": int(ho.sum()),
           "policies": {name: {"mean_improvement": round(float(r["imp"].mean()), 4),
                               "harmful": round(float(r["harm"].mean()), 4),
                               "lam": r["lam"], "u_dist": r["u_dist"]} for name, r in res.items()},
           "paired_diff_vs_binary": diffs,
           "preregistered_verdict": verdict,
           "claim_boundary": "δ 조건부·MM proxy·one-shot. apply/STOP(결정 A) 가치는 AR-081/082 별도 확립. 지각 아님.",
           "grounding": ["Agarwal NeurIPS 2021", "Gangrade AISTATS 2021", "Guo CVPR2022"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    import sys as _s; _s.stdout.reconfigure(encoding="utf-8")
    for name, r in res.items():
        print(f"{name:<22} imp {r['imp'].mean():.4f} harm {r['harm'].mean():.4f} | u_dist {r['u_dist']}")
    for k, d in diffs.items():
        print(f"  {k}: imp {d['improvement']['mean']} ci{d['improvement']['ci95']} | harm {d['harm']['mean']} ci{d['harm']['ci95']} | useful={d['graded_useful_over_binary']}")
    print(f"VERDICT: {verdict}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
