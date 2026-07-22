"""AR-086: u>1 이득의 물리–semantic 분해 (사전등록 판정).

각 holdout locomotion motion × u_grid: root_ratio(GT 참조)·foot_skate·MM-Dist(CSV).
MM-argmax-u 에서 물리 특성으로 "물리 under-correction 복구" vs "MM proxy 선호" 판정.

물리 지표는 tool 재적용(numpy, GPU 불요). MM 은 AR-085 dataset 재사용.

CLI (motion3d env):
    python tools/u_decomposition_ar086.py
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

from correction_tools import RootGaitConsistencyTool
from tools.coords_protocol import PELVIS
from tools.representative_pool_measure import foot_skate_world

POOL_ROOT = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608"
GT_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
MM_CSV = REPO_ROOT / "evals" / "snapshots" / "strength_q_v2_dataset_ar085_v1.csv"
OUT = REPO_ROOT / "evals" / "snapshots" / "u_decomposition_ar086_v1.json"
U_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
U_COL = {0.0: "mm_u000", 0.25: "mm_u025", 0.5: "mm_u050", 0.75: "mm_u075",
         1.0: "mm_u100", 1.25: "mm_u125", 1.5: "mm_u150", 2.0: "mm_u200"}
LOCO_THRESH = 0.010


def _rs(m):
    return float(np.linalg.norm(np.diff(m[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean())


def _boot_ci(a, rng, n=1000):
    a = np.asarray(a, float); m = len(a)
    bs = [float(np.mean(a[rng.integers(0, m, m)])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", type=Path, default=OUT); args = ap.parse_args()
    rng = np.random.default_rng(20260731)
    tool = RootGaitConsistencyTool()

    mm_rows = {(r["gen"], r["sid"], r["seed"]): r for r in csv.DictReader(open(MM_CSV, encoding="utf-8"))}
    meta_ix, gt_speed = {}, {}
    for gen in ("mdm", "motiongpt", "momask"):
        for p in sorted((POOL_ROOT / gen).glob("*.json")):
            if p.name == "_pool_summary.json":
                continue
            m = json.load(open(p, encoding="utf-8")); meta_ix[(gen, m["sample_id"], str(m["seed"]))] = m

    recs = []
    keys = list(mm_rows)[: args.limit] if args.limit else list(mm_rows)
    for key in keys:
        r = mm_rows[key]
        if r["split"] != "holdout":
            continue
        m = meta_ix.get(key)
        if m is None:
            continue
        sid = key[1]
        if sid not in gt_speed:
            gt = np.load(GT_DIR / f"{sid}.npy").astype(np.float64)
            gt_speed[sid] = _rs(gt)
        gs = gt_speed[sid]
        if gs <= LOCO_THRESH:
            continue
        traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        T = traj.shape[0]
        ratio, fs = {}, {}
        for u in U_GRID:
            if u == 0.0:
                corr = traj
            else:
                corr, _ = tool.apply(traj, "root", [], (0, T - 1),
                                     metadata={"coord_space": "trajectory", "continuous_u": u})
            ratio[u] = _rs(corr) / max(gs, 1e-9)
            fs[u] = foot_skate_world(corr)
        mm = {u: float(r[U_COL[u]]) for u in U_GRID}
        imp = {u: mm[0.0] - mm[u] for u in U_GRID}
        amu = max(U_GRID, key=lambda u: imp[u])   # MM-argmax-u
        recs.append({"gen": key[0], "sid": sid, "amu": amu,
                     "ratio": ratio, "fs": fs, "imp": imp, "gt_speed": gs})

    n = len(recs)
    over = [rc for rc in recs if rc["amu"] > 1.0]
    # ---- 판정: MM-argmax-u 에서 물리 특성.
    ratio_at_amu = np.array([rc["ratio"][rc["amu"]] for rc in recs])
    fs_at_amu = np.array([rc["fs"][rc["amu"]] for rc in recs])
    fs_at_u1 = np.array([rc["fs"][1.0] for rc in recs])
    ratio_at_amu_over = np.array([rc["ratio"][rc["amu"]] for rc in over]) if over else np.array([])

    med_ratio_amu = float(np.median(ratio_at_amu))
    med_ratio_amu_over = float(np.median(ratio_at_amu_over)) if len(ratio_at_amu_over) else None
    fs_delta_amu = fs_at_amu - fs_at_u1            # >0 = argmax 에서 skate 악화
    fs_worse_frac = float((fs_delta_amu > 1e-4).mean())

    # 곡선 (mean over holdout loco).
    curve = {str(u): {"root_ratio": round(float(np.mean([rc["ratio"][u] for rc in recs])), 4),
                      "foot_skate": round(float(np.mean([rc["fs"][u] for rc in recs])), 5),
                      "mm_improvement": round(float(np.mean([rc["imp"][u] for rc in recs])), 4)}
             for u in U_GRID}

    # 사전등록 판정.
    fs_delta_ci = _boot_ci(fs_delta_amu, rng)
    phys = (0.85 <= med_ratio_amu <= 1.20) and (fs_delta_ci[1] <= 1e-4)
    proxy = (med_ratio_amu > 1.35) or (fs_delta_ci[0] > 1e-4)
    verdict = ("물리 under-correction 복구" if phys and not proxy
               else "MM proxy 선호" if proxy and not phys
               else "혼합")

    out = {"schema_version": "1.0.0", "record_type": "u_decomposition", "board_id": "AR-086",
           "n_holdout_loco": n, "n_argmax_over1": len(over),
           "at_MM_argmax_u": {
               "median_root_ratio_all": round(med_ratio_amu, 3),
               "median_root_ratio_argmax_over1": round(med_ratio_amu_over, 3) if med_ratio_amu_over else None,
               "foot_skate_delta_vs_u1": {"mean": round(float(fs_delta_amu.mean()), 5), "ci95": fs_delta_ci,
                                          "frac_worse": round(fs_worse_frac, 3)}},
           "curves_mean": curve,
           "preregistered_verdict": {"verdict": verdict,
                                     "physical_ok": bool(phys), "proxy_ok": bool(proxy),
                                     "criterion": "물리=median ratio∈[.85,1.2] AND fs 비악화 / proxy=ratio>1.35 OR fs 유의악화"},
           "interpretation": "root_ratio→1 수렴하며 fs 비악화면 물리 복구; ratio 과증 또는 fs 악화하며 MM만 개선이면 proxy.",
           "claim_boundary": "GT 참조 진단. GT_root_speed 를 유일 정답으로 단정 금지 (텍스트가 GT보다 큰 이동 요구 가능). 지각은 A/B 몫.",
           "grounding": ["Guo HumanML3D CVPR2022", "AR-071/076 GT-참조 root 진단"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    import sys as _s; _s.stdout.reconfigure(encoding="utf-8")
    print(f"n={n} (argmax>1: {len(over)}) | median root_ratio @MM-argmax: all {med_ratio_amu:.3f}, over1-only {med_ratio_amu_over}")
    print(f"foot_skate @argmax − @u1: mean {fs_delta_amu.mean():+.5f} ci{fs_delta_ci} worse_frac {fs_worse_frac:.3f}")
    print("curves (u: root_ratio / foot_skate / mm_imp):")
    for u in U_GRID:
        c = curve[str(u)]
        print(f"  u={u}: ratio {c['root_ratio']:.3f}  fs {c['foot_skate']:.5f}  mmimp {c['mm_improvement']:+.4f}")
    print(f"VERDICT: {verdict}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
