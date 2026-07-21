"""AR-085 1단계: over-correction u-grid 라벨 — u∈{1.25,1.5,2.0} 신규 산출 후 v1 과 join.

사전등록 (AR-085 spec): u_grid = {0,.25,.5,.75,1,1.25,1.5,2.0}. u≤1 라벨은
strength_q_dataset_ar081_v1.csv 재사용 (state 13차원·mm_u000..mm_u100 포함) —
본 도구는 **u∈{1.25,1.5,2.0} 의 MM-Dist 만** 신규 계산해 확장 컬럼으로 붙인다.

state·split·δ 는 v1 그대로 (append). tool U_MAX=2.0 개방분 사용.

NOTE: mgpt env. ~8,100 correction+embed (신규 3 u × 2,699). ~40min.
CLI:
    C:/Users/geonu/anaconda3/envs/mgpt/python.exe tools/strength_q_v2_dataset_ar085.py
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

import tools.standard_metric_fid as G
from tools.standard_metric_rprec import GLOVE_DIR, _embed_text, _load_text_encoder
from correction_tools import RootGaitConsistencyTool
from mGPT.data.humanml.utils.word_vectorizer import WordVectorizer  # noqa: E402

POOL_ROOT = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608"
V1_CSV = REPO_ROOT / "evals" / "snapshots" / "strength_q_dataset_ar081_v1.csv"
OUT_CSV = REPO_ROOT / "evals" / "snapshots" / "strength_q_v2_dataset_ar085_v1.csv"
OUT_META = REPO_ROOT / "evals" / "snapshots" / "strength_q_v2_dataset_ar085_v1.json"
NEW_U = (1.25, 1.5, 2.0)
NEW_COL = {1.25: "mm_u125", 1.5: "mm_u150", 2.0: "mm_u200"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    G._setup_motion_process_globals()
    mean = np.load(str(G.MEAN_PATH)); std = np.load(str(G.STD_PATH))
    move, motion_enc = G._load_encoders()
    text_enc = _load_text_encoder()
    wv = WordVectorizer(str(GLOVE_DIR), "our_vab")
    tool = RootGaitConsistencyTool()
    print("[INFO] encoders loaded")

    v1 = list(csv.DictReader(open(V1_CSV, encoding="utf-8")))
    v1_key = {(r["gen"], r["sid"], r["seed"]): r for r in v1}
    meta_ix = {}
    for gen in ("mdm", "motiongpt", "momask"):
        for p in sorted((POOL_ROOT / gen).glob("*.json")):
            if p.name == "_pool_summary.json":
                continue
            m = json.load(open(p, encoding="utf-8"))
            meta_ix[(gen, m["sample_id"], str(m["seed"]))] = m

    rows_out, n_skip = [], 0
    items = v1[: args.limit] if args.limit else v1
    for i, r in enumerate(items, 1):
        key = (r["gen"], r["sid"], r["seed"])
        m = meta_ix.get(key)
        if m is None:
            n_skip += 1; continue
        traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        T = traj.shape[0]
        et = _embed_text([m["prompt"]], wv, text_enc)[0]
        mm_new, ok = {}, True
        for u in NEW_U:
            corr, _ = tool.apply(traj, "root", [], (0, T - 1),
                                 metadata={"coord_space": "trajectory", "continuous_u": u})
            f = G._to_features(corr)
            if f is None:
                ok = False; break
            e = G._embed([f], mean, std, move, motion_enc)[0]
            mm_new[u] = float(np.linalg.norm(et - e))
        if not ok:
            n_skip += 1; continue
        out = dict(r)  # v1 전 컬럼 (state + mm_u000..mm_u100)
        for u in NEW_U:
            out[NEW_COL[u]] = round(mm_new[u], 4)
        rows_out.append(out)
        if i % 200 == 0:
            print(f"  {i}/{len(items)}")

    cols = list(v1[0].keys()) + [NEW_COL[u] for u in NEW_U]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_out:
            w.writerow({c: r.get(c, "") for c in cols})

    meta = {
        "schema_version": "1.0.0", "record_type": "strength_q_v2_dataset", "board_id": "AR-085",
        "transition_dataset_id": "strength_q_ar085_v2",
        "u_grid": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0], "u_max": 2.0,
        "new_u_computed": list(NEW_U), "reuse": "u<=1 = strength_q_dataset_ar081_v1 (동일 파이프라인)",
        "mining_reason": "none (natural pool)", "n_rows": len(rows_out), "n_skip": n_skip,
        "claim_boundary": "학습 데이터 — 정책/최적성 claim 없음. MM proxy. u>1=extrapolation (offset 선형).",
    }
    json.dump(meta, open(OUT_META, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    import sys as _s; _s.stdout.reconfigure(encoding="utf-8")
    print(f"[OK] {len(rows_out)} rows (skip {n_skip}) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
