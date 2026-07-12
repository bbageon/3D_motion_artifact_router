"""AR-077 Cat-A: VQ 에 root correction 적용 시 no-harm 확인 (R-Prec/MM-Dist paired).

VQ(MotionGPT/MoMask) 는 root deficit 없음 (AR-077 diagnostic: ratio≈1.0) → correction 은
near-no-op. 본 도구는 그 near-no-op 이 semantic 을 **악화시키지 않는지**(no-harm) 확정.
악화하면 "VQ 에 적용하면 오히려 해로움 → STOP" 근거; 무해하면 "이득 없음 → STOP" 근거.
어느 쪽이든 generator-conditioned routing (VQ=STOP) 지지.

NOTE: mgpt env.
CLI:
    C:/Users/geonu/anaconda3/envs/mgpt/python.exe tools/vq_root_catA_ar077.py --gen motiongpt
    C:/Users/geonu/anaconda3/envs/mgpt/python.exe tools/vq_root_catA_ar077.py --gen momask
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
from tools.standard_metric_rprec import GLOVE_DIR, _embed_text, _euclidean_matrix, _load_text_encoder
from mGPT.data.humanml.utils.word_vectorizer import WordVectorizer  # noqa: E402

POOL_ROOT = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608"
CORR_ROOT = REPO_ROOT / "external_assets" / "ar077_root_corrected"
BATCH, N_REP = 32, 20


def _per_rep(text_e, mot_e, perm):
    n = len(text_e); top = np.zeros(3); mm = 0.0; c = 0
    for s in range(0, n - BATCH + 1, BATCH):
        b = perm[s:s + BATCH]
        d = _euclidean_matrix(text_e[b], mot_e[b])
        mm += float(np.mean(np.diag(d)))
        ranks = np.argsort(d, axis=1)
        for k in range(3):
            top[k] += np.mean([(i in ranks[i, :k + 1]) for i in range(BATCH)])
        c += 1
    return top / c, mm / c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, choices=["motiongpt", "momask", "mdm"])
    args = ap.parse_args()
    pool = POOL_ROOT / args.gen
    corr = CORR_ROOT / args.gen
    out_path = REPO_ROOT / "evals" / "snapshots" / f"vq_root_catA_ar077_{args.gen}_v1.json"

    G._setup_motion_process_globals()
    mean = np.load(str(G.MEAN_PATH)); std = np.load(str(G.STD_PATH))
    move, motion_enc = G._load_encoders()
    text_enc = _load_text_encoder()
    wv = WordVectorizer(str(GLOVE_DIR), "our_vab")
    print(f"[INFO] {args.gen}: encoders loaded")

    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(pool.glob("*.json"))
             if p.name != "_pool_summary.json"]
    caps, fo, fc = [], [], []
    for m in metas:
        cp = corr / f"{m['sample_id']}__seed{m['seed']}.trajectory.npy"
        if not cp.exists():
            continue
        o = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        c = np.load(cp).astype(np.float64)
        a, b = G._to_features(o), G._to_features(c)
        if a is None or b is None:
            continue
        caps.append(m["prompt"]); fo.append(a); fc.append(b)
    n = len(caps)
    print(f"[INFO] n={n}; embedding...")
    eo = G._embed(fo, mean, std, move, motion_enc)
    ec = G._embed(fc, mean, std, move, motion_enc)
    et = _embed_text(caps, wv, text_enc)

    ot, ct, om, cm = [], [], [], []
    for rep in range(N_REP):
        perm = np.random.default_rng(rep).permutation(n)
        to, mo = _per_rep(et, eo, perm); tc, mc = _per_rep(et, ec, perm)
        ot.append(to); ct.append(tc); om.append(mo); cm.append(mc)
    ot, ct, om, cm = map(np.array, (ot, ct, om, cm))
    d_r1 = ct[:, 0] - ot[:, 0]; d_mm = cm - om

    def _ci(a):
        return [round(float(np.percentile(a, 2.5)), 5), round(float(np.percentile(a, 97.5)), 5)]
    r1_ci, mm_ci = _ci(d_r1), _ci(d_mm)
    harm = (r1_ci[1] < 0) or (mm_ci[0] > 0)   # R@1 유의 하락 또는 MM 유의 악화

    out = {
        "schema_version": "1.0.0", "record_type": "vq_root_catA", "board_id": "AR-077",
        "generator": args.gen, "n_pairs": n, "protocol": f"paired batch={BATCH} reps={N_REP}",
        "original": {"R@1": round(float(ot[:, 0].mean()), 4), "MM_Dist": round(float(om.mean()), 4)},
        "corrected": {"R@1": round(float(ct[:, 0].mean()), 4), "MM_Dist": round(float(cm.mean()), 4)},
        "paired_diff": {"R@1": {"mean": round(float(d_r1.mean()), 5), "ci95": r1_ci},
                        "MM_Dist": {"mean": round(float(d_mm.mean()), 5), "ci95": mm_ci}},
        "no_harm_verdict": "harm (악화 → STOP)" if harm else "무해 (near-no-op → 이득 없어 STOP)",
        "routing": f"{args.gen}: root deficit 없음(ratio≈1.0) + correction {'악화' if harm else 'no-op'} → STOP 정답",
        "claim_boundary": "Cat-A no-harm — 지각 단정 아님. generator-conditioned routing (VQ=STOP) 지지.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"{args.gen}: R@1 {ot[:,0].mean():.4f}->{ct[:,0].mean():.4f} (d CI {r1_ci}) | "
          f"MM {om.mean():.4f}->{cm.mean():.4f} (d CI {mm_ci}) -> {out['no_harm_verdict']}")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
