"""AR-077 Cat-A R-Precision 수정 (2차 피드백 4): one-seed-per-prompt per retrieval.

v3 버그: 같은 prompt 의 text embedding 을 seed 3개에 반복해 한 batch 에 넣음 → 같은 문장의
다른 seed 모션이 정답 후보인데 diagonal 아니라고 오답 처리 (R-Precision 왜곡).
수정: **retrieval batch 는 prompt 당 seed 하나만** (permutation 마다 seed 무작위 선택),
N_PERM 반복 평균. MM-Dist/FID 는 v3 유지 (영향 없음) — 본 도구는 R@1/2/3 만 재산출.

단위 = prompt (locomotion 3-seed). orig/corr 동일 permutation·동일 seed 선택 (paired).
prompt-bootstrap B=1000.

NOTE: mgpt env.
CLI:
    C:/Users/geonu/anaconda3/envs/mgpt/python.exe tools/catA_rprec_fix_ar077.py --gen mdm
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

POOL_ROOT = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608"
CORR_ROOT = REPO_ROOT / "external_assets" / "ar077_root_corrected"
GT_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
PELVIS, BATCH, N_PERM, N_BOOT, LOCO_THRESH, N_SEED = 0, 32, 12, 1000, 0.010, 3


def _rprec_oneseed(text_p, orig_p, corr_p, rng):
    """text_p/orig_p/corr_p: [P] prompt 단위, 각 orig/corr 은 [P][N_SEED] embedding.
    permutation 마다 prompt 당 seed 1개 무작위 선택 → batch-32 R-Prec. orig/corr 동일 선택."""
    P = len(text_p)
    to = np.zeros(3); tc = np.zeros(3); c = 0
    for _ in range(N_PERM):
        sel = rng.integers(0, N_SEED, P)          # prompt 당 seed 선택 (orig/corr 공통)
        o = np.array([orig_p[i][sel[i]] for i in range(P)])
        cc = np.array([corr_p[i][sel[i]] for i in range(P)])
        perm = rng.permutation(P)
        for s in range(0, P - BATCH + 1, BATCH):
            b = perm[s:s + BATCH]
            te = text_p[b]
            for eset, top in ((o, to), (cc, tc)):
                d = _euclidean_matrix(te, eset[b])
                ranks = np.argsort(d, axis=1)
                for k in range(3):
                    top[k] += np.mean([(i in ranks[i, :k + 1]) for i in range(BATCH)])
            c += 1
    return to / c, tc / c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, choices=["mdm", "motiongpt", "momask"])
    args = ap.parse_args()
    pool = POOL_ROOT / args.gen; corr = CORR_ROOT / args.gen
    out_path = REPO_ROOT / "evals" / "snapshots" / f"catA_rprec_fix_ar077_{args.gen}_v1.json"

    G._setup_motion_process_globals()
    mean = np.load(str(G.MEAN_PATH)); std = np.load(str(G.STD_PATH))
    move, motion_enc = G._load_encoders()
    text_enc = _load_text_encoder()
    wv = WordVectorizer(str(GLOVE_DIR), "our_vab")
    print(f"[INFO] {args.gen}: encoders loaded")

    by_prompt = defaultdict(list)
    for p in sorted(pool.glob("*.json")):
        if p.name == "_pool_summary.json":
            continue
        m = json.load(open(p, encoding="utf-8")); by_prompt[m["sample_id"]].append(m)

    tp, op, cp_, n_skip = [], [], [], 0
    for sid, ms in by_prompt.items():
        gp = GT_DIR / f"{sid}.npy"
        if not gp.exists():
            n_skip += 1; continue
        gt = np.load(gp).astype(np.float64)
        if float(np.linalg.norm(np.diff(gt[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean()) <= LOCO_THRESH:
            continue
        fo, fc = [], []
        for m in ms:
            cf = corr / f"{sid}__seed{m['seed']}.trajectory.npy"
            if not cf.exists():
                continue
            a = G._to_features(np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64))
            b = G._to_features(np.load(cf).astype(np.float64))
            if a is not None and b is not None:
                fo.append(a); fc.append(b)
        if len(fo) < N_SEED:
            n_skip += 1; continue
        op.append(G._embed(fo[:N_SEED], mean, std, move, motion_enc))
        cp_.append(G._embed(fc[:N_SEED], mean, std, move, motion_enc))
        tp.append(_embed_text([ms[0]["prompt"]], wv, text_enc)[0])
    tp = np.array(tp); P = len(tp)
    print(f"[INFO] {args.gen}: P={P} (skip {n_skip})")

    to, tc = _rprec_oneseed(tp, op, cp_, np.random.default_rng(0))
    d_r1 = []
    rng = np.random.default_rng(20260724)
    for _ in range(N_BOOT):
        idx = rng.integers(0, P, P)
        rp = np.random.default_rng(int(rng.integers(0, 1 << 30)))
        to_b, tc_b = _rprec_oneseed(tp[idx], [op[i] for i in idx], [cp_[i] for i in idx], rp)
        d_r1.append(tc_b[0] - to_b[0])
    d_r1 = np.array(d_r1)
    r1c = [round(float(np.percentile(d_r1, 2.5)), 5), round(float(np.percentile(d_r1, 97.5)), 5)]

    out = {
        "schema_version": "1.0.0", "record_type": "catA_rprec_fix", "board_id": "AR-077",
        "generator": args.gen, "P_locomotion_prompts_3seed": P,
        "fix": "one-seed-per-prompt per retrieval (v3 는 같은 caption seed 3개를 batch 에 넣어 오답 처리 → 수정)",
        "R@1_original": round(float(to[0]), 4), "R@1_corrected": round(float(tc[0]), 4),
        "R@2_original": round(float(to[1]), 4), "R@2_corrected": round(float(tc[1]), 4),
        "R@3_original": round(float(to[2]), 4), "R@3_corrected": round(float(tc[2]), 4),
        "R@1_diff_prompt_bootstrap": {"mean": round(float(d_r1.mean()), 5), "ci95": r1c},
        "note": "MM-Dist/FID 는 vq_root_catA_v3 유지 (R-Prec seed 문제와 무관).",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"{args.gen} P={P}: R@1 {to[0]:.4f}->{tc[0]:.4f} (d CI {r1c})")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
