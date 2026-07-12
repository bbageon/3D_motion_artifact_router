"""AR-077 Cat-A v2 (피드백 반영): locomotion 165 한정 + prompt-bootstrap + FID + equal-N.

수정 (외부 피드백 2026-07-12):
  1. 표본: **locomotion 한정** (GT root speed>0.01) — 전체 300(non-loco 오염) 대신
     사전등록대로 MDM 과 동일 필터. non-loco 결과는 별도 보조분석(superseded).
  2. 통계 단위: **prompt** (seed-avg embedding), permutation 20개를 독립실험 취급하던 것
     폐기 → **prompt-bootstrap** (resample prompts) + 내부 batch-32 permutation.
  3. **FID 추가** (corrected/original vs GT-ref), prompt-bootstrap CI.
  4. **equal-N**: orig·corr·gt 모두 존재하는 prompt 만 (누락 count 명시).

원자 단위 = prompt (seed 3 평균). R-Prec/MM-Dist/FID 전부 prompt-bootstrap.

NOTE: mgpt env.
CLI:
    C:/Users/geonu/anaconda3/envs/mgpt/python.exe tools/vq_root_catA_v2_ar077.py --gen motiongpt
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

import tools.standard_metric_fid as G
from tools.standard_metric_rprec import GLOVE_DIR, _embed_text, _euclidean_matrix, _load_text_encoder
from mGPT.data.humanml.utils.word_vectorizer import WordVectorizer  # noqa: E402
import scipy.linalg as sla

POOL_ROOT = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608"
CORR_ROOT = REPO_ROOT / "external_assets" / "ar077_root_corrected"
GT_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
PELVIS = 0
BATCH = 32
N_PERM = 8         # bootstrap 내부 batch permutation (R-Prec 안정)
N_BOOT = 300       # prompt-bootstrap
LOCO_THRESH = 0.010


def _rprec_mm(text_e, mot_e, rng):
    n = len(text_e); top = np.zeros(3); mm = 0.0; c = 0
    for _ in range(N_PERM):
        perm = rng.permutation(n)
        for s in range(0, n - BATCH + 1, BATCH):
            b = perm[s:s + BATCH]
            d = _euclidean_matrix(text_e[b], mot_e[b])
            mm += float(np.mean(np.diag(d)))
            ranks = np.argsort(d, axis=1)
            for k in range(3):
                top[k] += np.mean([(i in ranks[i, :k + 1]) for i in range(BATCH)])
            c += 1
    return (top / c), (mm / c)


def _fid(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cov, _ = sla.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float((mu1 - mu2).dot(mu1 - mu2) + np.trace(s1) + np.trace(s2) - 2 * np.trace(cov))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, choices=["mdm", "motiongpt", "momask"])
    args = ap.parse_args()
    pool = POOL_ROOT / args.gen
    corr = CORR_ROOT / args.gen
    out_path = REPO_ROOT / "evals" / "snapshots" / f"vq_root_catA_v2_ar077_{args.gen}_v1.json"

    G._setup_motion_process_globals()
    mean = np.load(str(G.MEAN_PATH)); std = np.load(str(G.STD_PATH))
    move, motion_enc = G._load_encoders()
    text_enc = _load_text_encoder()
    wv = WordVectorizer(str(GLOVE_DIR), "our_vab")
    print(f"[INFO] {args.gen}: encoders loaded")

    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(pool.glob("*.json"))
             if p.name != "_pool_summary.json"]
    # prompt 별 seed 묶기 + locomotion 필터 + equal-N (orig/corr/gt 모두 존재).
    by_prompt = defaultdict(list)
    for m in metas:
        by_prompt[m["sample_id"]].append(m)
    gt_speed, gt_feat = {}, {}
    n_missing = 0
    caps, e_o, e_c, e_g = [], [], [], []
    for sid, ms in by_prompt.items():
        gp = GT_DIR / f"{sid}.npy"
        if not gp.exists():
            n_missing += 1; continue
        gt = np.load(gp).astype(np.float64)
        gspeed = float(np.linalg.norm(np.diff(gt[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean())
        if gspeed <= LOCO_THRESH:
            continue  # locomotion 한정
        gf = G._to_features(gt)
        if gf is None:
            n_missing += 1; continue
        fo, fc = [], []
        for m in ms:
            cp = corr / f"{sid}__seed{m['seed']}.trajectory.npy"
            if not cp.exists():
                continue
            o = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
            c = np.load(cp).astype(np.float64)
            a, b = G._to_features(o), G._to_features(c)
            if a is not None and b is not None:
                fo.append(a); fc.append(b)
        if not fo:
            n_missing += 1; continue
        # seed-avg embedding (prompt 단위).
        e_o.append(G._embed(fo, mean, std, move, motion_enc).mean(0))
        e_c.append(G._embed(fc, mean, std, move, motion_enc).mean(0))
        e_g.append(G._embed([gf], mean, std, move, motion_enc)[0])
        caps.append(ms[0]["prompt"])
    e_o, e_c, e_g = map(np.array, (e_o, e_c, e_g))
    e_t = _embed_text(caps, wv, text_enc)
    N = len(caps)
    print(f"[INFO] {args.gen}: locomotion prompts N={N} (missing/skip {n_missing})")

    # point estimate.
    rng0 = np.random.default_rng(0)
    top_o, mm_o = _rprec_mm(e_t, e_o, rng0)
    top_c, mm_c = _rprec_mm(e_t, e_c, np.random.default_rng(0))
    fid_o, fid_c = _fid(e_o, e_g), _fid(e_c, e_g)

    # prompt-bootstrap.
    d_r1, d_mm, d_fid = [], [], []
    rng = np.random.default_rng(20260721)
    for _ in range(N_BOOT):
        idx = rng.integers(0, N, N)
        et, eo, ec, eg = e_t[idx], e_o[idx], e_c[idx], e_g[idx]
        rp = np.random.default_rng(int(rng.integers(0, 1 << 30)))
        to, mo = _rprec_mm(et, eo, rp)
        tc, mc = _rprec_mm(et, ec, np.random.default_rng(0))  # same batches via fixed seed within
        d_r1.append(tc[0] - to[0]); d_mm.append(mc - mo)
        d_fid.append(_fid(ec, eg) - _fid(eo, eg))
    def _ci(a):
        return [round(float(np.percentile(a, 2.5)), 5), round(float(np.percentile(a, 97.5)), 5)]
    r1c, mmc, fidc = _ci(np.array(d_r1)), _ci(np.array(d_mm)), _ci(np.array(d_fid))
    harm = (r1c[1] < 0) or (mmc[0] > 0) or (fidc[0] > 0)

    out = {
        "schema_version": "1.0.0", "record_type": "vq_root_catA_v2", "board_id": "AR-077",
        "generator": args.gen, "N_locomotion_prompts": N, "n_missing_or_skip": n_missing,
        "protocol": "locomotion 한정, prompt 단위(seed-avg emb), prompt-bootstrap(B=300, 내부 batch-32 perm), equal-N",
        "stat_unit": "prompt (seed 평균) — permutation 은 R-Prec 추정 내부 반복 (독립실험 아님)",
        "original": {"R@1": round(float(top_o[0]), 4), "R@2": round(float(top_o[1]), 4),
                     "R@3": round(float(top_o[2]), 4), "MM_Dist": round(mm_o, 4), "FID_vs_GT": round(fid_o, 3)},
        "corrected": {"R@1": round(float(top_c[0]), 4), "R@2": round(float(top_c[1]), 4),
                      "R@3": round(float(top_c[2]), 4), "MM_Dist": round(mm_c, 4), "FID_vs_GT": round(fid_c, 3)},
        "paired_diff_prompt_bootstrap": {
            "R@1": {"mean": round(float(np.mean(d_r1)), 5), "ci95": r1c},
            "MM_Dist": {"mean": round(float(np.mean(d_mm)), 5), "ci95": mmc},
            "FID": {"mean": round(float(np.mean(d_fid)), 3), "ci95": fidc}},
        "no_harm_verdict": ("harm (Cat-A 유의 악화 → apply 는 no-harm 위반)" if harm
                            else "무해 (Cat-A 유지)"),
        "routing": f"{args.gen} locomotion: root-aware correction 이 Cat-A 를 {'악화' if harm else '유지'}. "
                   "root deficit 여부(diagnostic)와 결합해 apply/STOP 판정.",
        "claim_boundary": "locomotion 한정 공정 비교. generator 인과(VQ 구조) 금지. 지각은 A/B 별도. "
                          "'GT 와 동일' 금지 → '평균 root deficit 증거 없음'.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"{args.gen} loco N={N}: R@1 {top_o[0]:.4f}->{top_c[0]:.4f} (d CI {r1c}) | "
          f"MM {mm_o:.3f}->{mm_c:.3f} (d CI {mmc}) | FID {fid_o:.2f}->{fid_c:.2f} (d CI {fidc}) -> {out['no_harm_verdict']}")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
