"""AR-081 1단계: strength-Q 학습 데이터셋 — 13차원 직관 이름 state + u-grid ΔMM 라벨.

사전등록 spec (AR-081) 준수:
  - state 13차원, **직관적 이름** (naming 단일 출처 = spec §1 표; 구 이름 매핑 명시).
  - u_grid = {0, 0.25, 0.5, 0.75, 1.0}. u=0(무보정)·u=1.0 라벨은 기존
    routing_benefit_permotion CSV 의 mm_orig/mm_corr 재사용 (동일 파이프라인) —
    u=0.25/0.5/0.75 만 신규 산출 (tool 적용 → tm2t embedding → per-motion MM).
  - split = 기존 sample_id 단독 split 승계 (permotion CSV 의 split 열).
  - §3-26 transition metadata: dataset_id·u_grid·mining_reason 기록.

NOTE: mgpt env (embedding). ~8,100 feature+embed — 수십 분.
CLI:
    C:/Users/geonu/anaconda3/envs/mgpt/python.exe tools/strength_q_dataset_ar081.py
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
from tools.preaction_state_ar065 import extract_state, _intent_bucket
from correction_tools import RootGaitConsistencyTool
from mGPT.data.humanml.utils.word_vectorizer import WordVectorizer  # noqa: E402

POOL_ROOT = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608"
PERMOTION = REPO_ROOT / "evals" / "snapshots" / "routing_benefit_permotion_ar077_v1.csv"
OUT_CSV = REPO_ROOT / "evals" / "snapshots" / "strength_q_dataset_ar081_v1.csv"
OUT_META = REPO_ROOT / "evals" / "snapshots" / "strength_q_dataset_ar081_v1.json"
GENS = ("mdm", "motiongpt", "momask")
NEW_U = (0.25, 0.5, 0.75)
FPS = 20.0

#: spec §1 naming 표 — 구 이름 → 직관 이름 (단일 출처는 spec; 여기선 구현 매핑).
def intuitive_state(traj, prompt: str, gen: str) -> dict:
    old = extract_state(traj)
    intent = _intent_bucket(prompt)
    return {
        # A. 보정 필요도
        "stepping_speed": old["gait_implied_speed"],            # 다리가 요구하는 몸통 속도
        "body_speed_shortfall": old["root_gait_mismatch"],      # 몸통이 모자란 속도 (음수=과속)
        "foot_sliding": old["foot_skate"],                      # 접지발 미끄러짐
        # B. 측정 신뢰도
        "ground_contact_ratio": old["contact_fraction"],        # 접지 프레임 비율
        "contact_streak": round(old["contact_run_mean"] / max(old["n_frames"], 1), 5),  # 정규화 접지 지속
        # C. 동작 문맥
        "path_straightness": old["straightness"],
        "duration_sec": round(old["n_frames"] / FPS, 3),
        "wants_to_travel": 1 if intent == "locomotion" else 0,
        "wants_to_stay": 1 if intent == "in_place" else 0,
        "intent_unclear": 1 if intent == "ambiguous" else 0,
        # D. generator prior
        "is_mdm": 1 if gen == "mdm" else 0,
        "is_motiongpt": 1 if gen == "motiongpt" else 0,
        "is_momask": 1 if gen == "momask" else 0,
    }


STATE_COLS = ["stepping_speed", "body_speed_shortfall", "foot_sliding",
              "ground_contact_ratio", "contact_streak",
              "path_straightness", "duration_sec",
              "wants_to_travel", "wants_to_stay", "intent_unclear",
              "is_mdm", "is_motiongpt", "is_momask"]


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

    # 기존 라벨 (u=0, u=1.0) + split 승계.
    prev = {(r["gen"], r["sid"], r["seed"]): r
            for r in csv.DictReader(open(PERMOTION, encoding="utf-8"))}

    rows_out = []
    n_skip = 0
    for gen in GENS:
        metas = [json.load(open(p, encoding="utf-8")) for p in sorted((POOL_ROOT / gen).glob("*.json"))
                 if p.name != "_pool_summary.json"]
        if args.limit:
            metas = metas[:args.limit]
        done = 0
        for m in metas:
            k = (gen, m["sample_id"], str(m["seed"]))
            if k not in prev:
                n_skip += 1
                continue
            traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
            T = traj.shape[0]
            st = intuitive_state(traj, m["prompt"], gen)
            et = _embed_text([m["prompt"]], wv, text_enc)[0]
            mm_u = {}
            ok = True
            for u in NEW_U:
                corr, _ = tool.apply(traj, target_part="root", target_joints=[],
                                     frame_range=(0, T - 1), strength="large",
                                     metadata={"coord_space": "trajectory", "continuous_u": u})
                f = G._to_features(corr)
                if f is None:
                    ok = False
                    break
                e = G._embed([f], mean, std, move, motion_enc)[0]
                mm_u[u] = float(np.linalg.norm(et - e))
            if not ok:
                n_skip += 1
                continue
            p = prev[k]
            row = {"gen": gen, "sid": m["sample_id"], "seed": m["seed"], "split": p["split"], **st,
                   "mm_u000": float(p["mm_orig"]), "mm_u025": round(mm_u[0.25], 4),
                   "mm_u050": round(mm_u[0.5], 4), "mm_u075": round(mm_u[0.75], 4),
                   "mm_u100": float(p["mm_corr"])}
            rows_out.append(row)
            done += 1
            if done % 150 == 0:
                print(f"  [{gen}] {done}")
        print(f"[INFO] {gen}: {done} motions")

    cols = ["gen", "sid", "seed", "split"] + STATE_COLS + ["mm_u000", "mm_u025", "mm_u050", "mm_u075", "mm_u100"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_out:
            w.writerow({c: r[c] for c in cols})

    meta = {
        "schema_version": "1.0.0", "record_type": "strength_q_dataset", "board_id": "AR-081",
        "transition_dataset_id": "strength_q_ar081_v1",
        "u_grid": [0.0, 0.25, 0.5, 0.75, 1.0], "u0_is_stop": True,
        "mining_reason": "none (natural pool — hard-mining 없음)",
        "n_rows": len(rows_out), "n_skip": n_skip,
        "state_naming": "AR-081 spec §1 표 단일 출처 (직관 이름; 구 AR-065 이름 매핑 명시)",
        "labels": "mm_uXXX = MM-Dist(text, motion@u). improvement(u) = mm_u000 − mm_uXXX (양수=개선)",
        "split": "routing_benefit_permotion (sample_id 단독, seed 20260722) 승계",
        "reuse": "u=0/1.0 라벨 = 기존 permotion CSV (동일 tm2t 파이프라인)",
        "claim_boundary": "학습 데이터 산출 — 정책 성능 claim 없음. MM proxy (지각 아님).",
    }
    json.dump(meta, open(OUT_META, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    import sys as _s; _s.stdout.reconfigure(encoding="utf-8")
    print(f"[OK] {len(rows_out)} rows (skip {n_skip}) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
