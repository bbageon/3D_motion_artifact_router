"""AR-072 ③-Cat-A: semantic guard — R-Precision/MM-Dist equal-N paired (기각 조건 관문).

사전등록 (spec ①): root 보정은 이동 거리(=prompt 의미의 일부)를 바꾸므로,
**R@1 이 유의하게 하락하거나 MM-Dist 가 유의하게 악화되면 처방 기각** (물리가 아무리 좋아도).

설계 (paired):
  - 같은 caption, 같은 sample 의 원본 vs 보정 trajectory 를 tm2t co-embedding 으로 평가.
  - rep 20회: 각 rep 마다 **동일한 permutation/batch(32)** 를 양 arm 에 사용 → rep 별
    paired 차이 (corrected − original) → percentile CI.
  - 기각: R@1 diff CI 상한 < 0 (유의 하락) 또는 MM-Dist diff CI 하한 > 0 (유의 악화).

NOTE: mgpt env 필요 (torch + tm2t + spacy + GloVe).
CLI:
    C:/Users/geonu/anaconda3/envs/mgpt/python.exe tools/root_semantic_guard_ar072.py --limit 60
    C:/Users/geonu/anaconda3/envs/mgpt/python.exe tools/root_semantic_guard_ar072.py
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
from tools.standard_metric_rprec import (
    GLOVE_DIR, _embed_text, _euclidean_matrix, _load_text_encoder,
)
from mGPT.data.humanml.utils.word_vectorizer import WordVectorizer  # noqa: E402 (mgpt env)

POOL = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608" / "mdm"
CORR = REPO_ROOT / "external_assets" / "ar072_root_corrected_mdm"
BATCH = 32
N_REP = 20


def _per_rep_metrics(text_embs, motion_embs, perm):
    """한 permutation 에 대한 top1/2/3 + mm (batch 32)."""
    n = len(text_embs)
    top = np.zeros(3); mm = 0.0; count = 0
    for s in range(0, n - BATCH + 1, BATCH):
        b = perm[s:s + BATCH]
        d = _euclidean_matrix(text_embs[b], motion_embs[b])
        mm += float(np.mean(np.diag(d)))
        ranks = np.argsort(d, axis=1)
        for k in range(3):
            top[k] += np.mean([(i in ranks[i, :k + 1]) for i in range(BATCH)])
        count += 1
    return top / count, mm / count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "root_semantic_guard_ar072_v1.json")
    args = ap.parse_args()

    G._setup_motion_process_globals()
    mean = np.load(str(G.MEAN_PATH)); std = np.load(str(G.STD_PATH))
    move, motion_enc = G._load_encoders()
    text_enc = _load_text_encoder()
    wv = WordVectorizer(str(GLOVE_DIR), "our_vab")
    print("[INFO] encoders loaded")

    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(POOL.glob("*.json"))
             if p.name != "_pool_summary.json"]
    if args.limit:
        metas = metas[:args.limit]

    caps, f_orig, f_corr = [], [], []
    n_skip = 0
    for i, m in enumerate(metas, 1):
        corr_p = CORR / f"{m['sample_id']}__seed{m['seed']}.trajectory.npy"
        if not corr_p.exists():
            n_skip += 1
            continue
        orig = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        corr = np.load(corr_p).astype(np.float64)
        fo = G._to_features(orig)
        fc = G._to_features(corr)
        if fo is None or fc is None:
            n_skip += 1
            continue
        caps.append(m["prompt"]); f_orig.append(fo); f_corr.append(fc)
        if i % 150 == 0:
            print(f"  features {i}/{len(metas)}")

    print(f"[INFO] n={len(caps)} (skip {n_skip}); embedding...")
    e_orig = G._embed(f_orig, mean, std, move, motion_enc)
    e_corr = G._embed(f_corr, mean, std, move, motion_enc)
    e_text = _embed_text(caps, wv, text_enc)
    n = len(caps)

    reps = {"orig_top": [], "corr_top": [], "orig_mm": [], "corr_mm": []}
    for rep in range(N_REP):
        perm = np.random.default_rng(rep).permutation(n)
        to, mo = _per_rep_metrics(e_text, e_orig, perm)
        tc, mc = _per_rep_metrics(e_text, e_corr, perm)
        reps["orig_top"].append(to); reps["corr_top"].append(tc)
        reps["orig_mm"].append(mo); reps["corr_mm"].append(mc)
    ot = np.array(reps["orig_top"]); ct = np.array(reps["corr_top"])
    om = np.array(reps["orig_mm"]); cm = np.array(reps["corr_mm"])
    d_top = ct - ot                    # [20, 3] paired diff
    d_mm = cm - om                     # [20]

    def _ci(a):
        return [round(float(np.percentile(a, 2.5)), 5), round(float(np.percentile(a, 97.5)), 5)]

    r1_ci = _ci(d_top[:, 0]); mm_ci = _ci(d_mm)
    fail_r1 = r1_ci[1] < 0             # R@1 유의 하락
    fail_mm = mm_ci[0] > 0             # MM-Dist 유의 악화
    verdict = "기각" if (fail_r1 or fail_mm) else "통과"

    out = {
        "schema_version": "1.0.0", "record_type": "root_semantic_guard", "board_id": "AR-072",
        "n_pairs": n, "n_skip": n_skip, "protocol": f"paired, batch={BATCH}, reps={N_REP} (동일 permutation 양 arm)",
        "original": {"R@1": round(float(ot[:, 0].mean()), 4), "R@2": round(float(ot[:, 1].mean()), 4),
                     "R@3": round(float(ot[:, 2].mean()), 4), "MM_Dist": round(float(om.mean()), 4)},
        "corrected": {"R@1": round(float(ct[:, 0].mean()), 4), "R@2": round(float(ct[:, 1].mean()), 4),
                      "R@3": round(float(ct[:, 2].mean()), 4), "MM_Dist": round(float(cm.mean()), 4)},
        "paired_diff": {"R@1": {"mean": round(float(d_top[:, 0].mean()), 5), "ci95": r1_ci},
                        "R@2": {"mean": round(float(d_top[:, 1].mean()), 5), "ci95": _ci(d_top[:, 1])},
                        "R@3": {"mean": round(float(d_top[:, 2].mean()), 5), "ci95": _ci(d_top[:, 2])},
                        "MM_Dist": {"mean": round(float(d_mm.mean()), 5), "ci95": mm_ci}},
        "preregistered_criterion": "기각 = R@1 diff CI 상한 < 0 또는 MM-Dist diff CI 하한 > 0",
        "verdict": verdict,
        "claim_boundary": "semantic guard 판정 — 지각 판정 아님 (통과 시 A/B 로).",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"R@1 {ot[:,0].mean():.4f} -> {ct[:,0].mean():.4f} (diff CI {r1_ci})")
    print(f"MM  {om.mean():.4f} -> {cm.mean():.4f} (diff CI {mm_ci})")
    print(f"VERDICT: {'FAIL' if verdict == '기각' else 'PASS'}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
