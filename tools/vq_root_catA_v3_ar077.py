"""AR-077 Cat-A v3 (피드백 2차 — 통계 버그 수정): locomotion, prompt-bootstrap 정합.

v2 버그 수정:
  - orig/corr **같은 batch permutation** (v2 는 corr 에 고정 rng(0), orig 에 변동 rng → paired 깨짐).
  - **seed embedding 미평균** — FID 분포 인위 축소 방지. prompt bootstrap 하되 seed 3개 유지.
  - **엄격 3-seed equal-N** — 세 seed 모두 존재하는 prompt 만.
  - **bootstrap 1000** (프로젝트 규칙).

단위 = motion (prompt bootstrap 시 그 prompt 의 3 seed 모두 포함). R-Prec/MM/FID 전부
같은 prompt-resample + 같은 permutation 으로 orig/corr paired.

NOTE: mgpt env.
CLI:
    C:/Users/geonu/anaconda3/envs/mgpt/python.exe tools/vq_root_catA_v3_ar077.py --gen motiongpt
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
BATCH, N_PERM, N_BOOT = 32, 8, 1000
LOCO_THRESH = 0.010
N_SEED_REQ = 3


def _fid(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cov, _ = sla.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float((mu1 - mu2).dot(mu1 - mu2) + np.trace(s1) + np.trace(s2) - 2 * np.trace(cov))


def _rprec_mm_paired(text_e, orig_e, corr_e, rng):
    """orig/corr 에 **동일 permutation** — paired. 반환 (top3_o, mm_o, top3_c, mm_c)."""
    n = len(text_e)
    to = np.zeros(3); tc = np.zeros(3); mo = mc = 0.0; c = 0
    for _ in range(N_PERM):
        perm = rng.permutation(n)
        for s in range(0, n - BATCH + 1, BATCH):
            b = perm[s:s + BATCH]
            te = text_e[b]
            for eset, top, mm_acc in ((orig_e, to, "o"), (corr_e, tc, "c")):
                d = _euclidean_matrix(te, eset[b])
                dm = float(np.mean(np.diag(d)))
                ranks = np.argsort(d, axis=1)
                for k in range(3):
                    top[k] += np.mean([(i in ranks[i, :k + 1]) for i in range(BATCH)])
                if mm_acc == "o":
                    nonlocal_o = dm
                else:
                    nonlocal_c = dm
            mo += nonlocal_o; mc += nonlocal_c; c += 1
    return to / c, mo / c, tc / c, mc / c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, choices=["mdm", "motiongpt", "momask"])
    ap.add_argument("--pool-root", type=Path, default=POOL_ROOT)
    ap.add_argument("--corr-root", type=Path, default=CORR_ROOT)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    pool = args.pool_root / args.gen; corr = args.corr_root / args.gen
    out_path = REPO_ROOT / "evals" / "snapshots" / f"vq_root_catA_v3_ar077_{args.gen}{args.out_suffix}_v1.json"

    G._setup_motion_process_globals()
    mean = np.load(str(G.MEAN_PATH)); std = np.load(str(G.STD_PATH))
    move, motion_enc = G._load_encoders()
    text_enc = _load_text_encoder()
    wv = WordVectorizer(str(GLOVE_DIR), "our_vab")
    print(f"[INFO] {args.gen}: encoders loaded")

    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(pool.glob("*.json"))
             if p.name != "_pool_summary.json"]
    by_prompt = defaultdict(list)
    for m in metas:
        by_prompt[m["sample_id"]].append(m)

    # prompt 단위 그룹 (locomotion + 엄격 3-seed equal-N). embedding 은 seed 별 유지.
    prompts = []  # 각: {"text": emb, "orig": [3 emb], "corr": [3 emb], "gt": emb}
    n_skip = 0
    for sid, ms in by_prompt.items():
        gp = GT_DIR / f"{sid}.npy"
        if not gp.exists():
            n_skip += 1; continue
        gt = np.load(gp).astype(np.float64)
        gspeed = float(np.linalg.norm(np.diff(gt[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean())
        if gspeed <= LOCO_THRESH:
            continue
        gf = G._to_features(gt)
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
        if gf is None or len(fo) < N_SEED_REQ:   # 엄격 3-seed
            n_skip += 1; continue
        eo = G._embed(fo[:N_SEED_REQ], mean, std, move, motion_enc)
        ec = G._embed(fc[:N_SEED_REQ], mean, std, move, motion_enc)
        eg = G._embed([gf], mean, std, move, motion_enc)[0]
        et = _embed_text([ms[0]["prompt"]], wv, text_enc)[0]
        prompts.append({"text": et, "orig": eo, "corr": ec, "gt": eg})
    P = len(prompts)
    print(f"[INFO] {args.gen}: locomotion 3-seed prompts P={P} (skip {n_skip})")

    def _assemble(idxs):
        """prompt index list → motion-level (seed 유지) text/orig/corr/gt arrays."""
        t, o, c, g = [], [], [], []
        for i in idxs:
            p = prompts[i]
            for s in range(N_SEED_REQ):
                t.append(p["text"]); o.append(p["orig"][s]); c.append(p["corr"][s])
            g.append(p["gt"])
        return np.array(t), np.array(o), np.array(c), np.array(g)

    # point estimate (전 prompt).
    et, eo, ec, eg = _assemble(range(P))
    to, mo, tc, mc = _rprec_mm_paired(et, eo, ec, np.random.default_rng(0))
    fid_o, fid_c = _fid(eo, eg), _fid(ec, eg)

    # prompt-bootstrap (B=1000).
    rng = np.random.default_rng(20260723)
    d_r1, d_mm, d_fid = [], [], []
    for _ in range(N_BOOT):
        idxs = rng.integers(0, P, P)
        et_b, eo_b, ec_b, eg_b = _assemble(idxs)
        rp = np.random.default_rng(int(rng.integers(0, 1 << 30)))
        to_b, mo_b, tc_b, mc_b = _rprec_mm_paired(et_b, eo_b, ec_b, rp)  # 같은 perm orig/corr
        d_r1.append(tc_b[0] - to_b[0]); d_mm.append(mc_b - mo_b)
        d_fid.append(_fid(ec_b, eg_b) - _fid(eo_b, eg_b))
    def _ci(a):
        return [round(float(np.percentile(a, 2.5)), 5), round(float(np.percentile(a, 97.5)), 5)]
    r1c, mmc, fidc = _ci(np.array(d_r1)), _ci(np.array(d_mm)), _ci(np.array(d_fid))
    harm = (r1c[1] < 0) or (mmc[0] > 0) or (fidc[0] > 0)

    out = {
        "schema_version": "1.0.0", "record_type": "vq_root_catA_v3", "board_id": "AR-077",
        "generator": args.gen, "P_locomotion_prompts_3seed": P, "n_skip": n_skip,
        "protocol": "locomotion + 엄격 3-seed equal-N; prompt-bootstrap B=1000; orig/corr 동일 permutation(paired); "
                    "seed embedding 미평균(FID 분포 보존)",
        "fixes_over_v2": ["같은 permutation(paired)", "seed 미평균", "엄격 3-seed", "bootstrap 1000"],
        "original": {"R@1": round(float(to[0]), 4), "MM_Dist": round(mo, 4), "FID_vs_GT": round(fid_o, 3)},
        "corrected": {"R@1": round(float(tc[0]), 4), "MM_Dist": round(mc, 4), "FID_vs_GT": round(fid_c, 3)},
        "paired_diff_prompt_bootstrap": {
            "R@1": {"mean": round(float(np.mean(d_r1)), 5), "ci95": r1c},
            "MM_Dist": {"mean": round(float(np.mean(d_mm)), 5), "ci95": mmc},
            "FID": {"mean": round(float(np.mean(d_fid)), 3), "ci95": fidc}},
        "no_harm_verdict": "harm (유의 악화)" if harm else "무해/개선",
        "claim_boundary": "locomotion 한정 공정 비교, 통계 정합. generator 인과 금지. 지각은 A/B. MDM·단일 벤치마크.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"{args.gen} P={P}: R@1 {to[0]:.4f}->{tc[0]:.4f}(d{r1c}) MM {mo:.3f}->{mc:.3f}(d{mmc}) "
          f"FID {fid_o:.2f}->{fid_c:.2f}(d{fidc}) -> {out['no_harm_verdict']}")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
