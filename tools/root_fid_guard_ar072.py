"""AR-072 ③-Cat-A 보강: FID guard (분포 수준) + over-correction 꼬리 점검.

①-보강 (외부 피드백 채택): equal-protocol FID(corrected, GT) vs FID(original, GT) —
유의 악화 시 기각. + ratio_after (v_root/GT) 의 꼬리 (over-correction: "walk in place"
류에서 root 를 밀어버린 sample) 점검.

NOTE: mgpt env.
CLI:
    C:/Users/geonu/anaconda3/envs/mgpt/python.exe tools/root_fid_guard_ar072.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

import tools.standard_metric_fid as G

POOL = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608" / "mdm"
CORR = REPO_ROOT / "external_assets" / "ar072_root_corrected_mdm"
GT_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
PELVIS = 0
LOCO_THRESH = 0.010
N_BOOT = 200


def _fid(a: np.ndarray, b: np.ndarray) -> float:
    """Frechet distance between Gaussians fit to embeddings a, b."""
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False)
    s2 = np.cov(b, rowvar=False)
    diff = mu1 - mu2
    import scipy.linalg as sla
    covmean, _ = sla.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "root_fid_guard_ar072_v1.json")
    args = ap.parse_args()

    G._setup_motion_process_globals()
    mean = np.load(str(G.MEAN_PATH)); std = np.load(str(G.STD_PATH))
    move, motion_enc = G._load_encoders()
    print("[INFO] encoders loaded")

    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(POOL.glob("*.json"))
             if p.name != "_pool_summary.json"]

    f_orig, f_corr, f_gt = [], [], []
    gt_done = set()
    ratios_after = []   # (sid, gt_speed, ratio_after) — 꼬리 점검
    per_prompt_va = {}
    gt_speed = {}
    for i, m in enumerate(metas, 1):
        sid = m["sample_id"]
        orig = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        corr = np.load(CORR / f"{sid}__seed{m['seed']}.trajectory.npy").astype(np.float64)
        fo, fc = G._to_features(orig), G._to_features(corr)
        if fo is not None and fc is not None:
            f_orig.append(fo); f_corr.append(fc)
        if sid not in gt_done:
            gt = np.load(GT_DIR / f"{sid}.npy").astype(np.float64)
            fg = G._to_features(gt)
            if fg is not None:
                f_gt.append(fg)
            gt_done.add(sid)
            gt_speed[sid] = float(np.linalg.norm(
                np.diff(gt[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean())
        va = float(np.linalg.norm(np.diff(corr[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean())
        per_prompt_va.setdefault(sid, []).append(va)
        if i % 200 == 0:
            print(f"  features {i}/{len(metas)}")

    e_orig = G._embed(f_orig, mean, std, move, motion_enc)
    e_corr = G._embed(f_corr, mean, std, move, motion_enc)
    e_gt = G._embed(f_gt, mean, std, move, motion_enc)
    print(f"[INFO] embeddings: orig {len(e_orig)}, corr {len(e_corr)}, gt {len(e_gt)}")

    fid_o = _fid(e_orig, e_gt)
    fid_c = _fid(e_corr, e_gt)
    # bootstrap ΔFID (embedding resample, paired index for orig/corr).
    rng = np.random.default_rng(20260715)
    deltas = []
    n = len(e_orig)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        gidx = rng.integers(0, len(e_gt), len(e_gt))
        deltas.append(_fid(e_corr[idx], e_gt[gidx]) - _fid(e_orig[idx], e_gt[gidx]))
    deltas = np.array(deltas)
    d_ci = [round(float(np.percentile(deltas, 2.5)), 3), round(float(np.percentile(deltas, 97.5)), 3)]
    fail_fid = d_ci[0] > 0  # 유의 악화

    # over-correction 꼬리: locomotion 외 prompt 포함 전체에서 ratio_after 분포.
    rows = [(sid, gt_speed[sid], float(np.mean(v)) / max(gt_speed[sid], 1e-9))
            for sid, v in per_prompt_va.items()]
    loco = [r for r in rows if r[1] > LOCO_THRESH]
    nonloco = [r for r in rows if r[1] <= LOCO_THRESH]
    ra_loco = np.array([r[2] for r in loco])
    over_15 = float((ra_loco > 1.5).mean()) if len(ra_loco) else 0.0
    # 비-locomotion (정지·제자리 류): 보정 후 root 속도 절대값 (밀어버렸는지).
    va_nonloco = np.array([float(np.mean(per_prompt_va[r[0]])) for r in nonloco])

    out = {
        "schema_version": "1.0.0", "record_type": "root_fid_guard", "board_id": "AR-072",
        "fid": {"original_vs_GT": round(fid_o, 3), "corrected_vs_GT": round(fid_c, 3),
                "delta": round(fid_c - fid_o, 3), "delta_boot_ci95": d_ci, "n_boot": N_BOOT,
                "verdict": "기각" if fail_fid else ("통과 (개선)" if d_ci[1] < 0 else "통과")},
        "over_correction_tail": {
            "n_locomotion": len(loco),
            "ratio_after_p50": round(float(np.percentile(ra_loco, 50)), 3),
            "ratio_after_p95": round(float(np.percentile(ra_loco, 95)), 3),
            "frac_ratio_gt_1.5": round(over_15, 4),
            "n_nonlocomotion": len(nonloco),
            "nonloco_root_speed_after_mean": round(float(va_nonloco.mean()), 5) if len(va_nonloco) else None,
            "nonloco_note": "정지/제자리 류에서 보정 후 root 속도가 크면 'walk in place' 위험 실현"},
        "claim_boundary": "분포 guard + 꼬리 점검 — 지각 판정 아님.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"FID: {fid_o:.3f} -> {fid_c:.3f} (delta {fid_c-fid_o:+.3f}, boot CI {d_ci}) "
          f"{'FAIL' if fail_fid else 'PASS'}")
    print(f"tail: loco ratio p50={np.percentile(ra_loco,50):.2f} p95={np.percentile(ra_loco,95):.2f} "
          f">1.5 frac={over_15*100:.1f}% | nonloco n={len(nonloco)} "
          f"root_speed_after={va_nonloco.mean() if len(va_nonloco) else 0:.4f}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
