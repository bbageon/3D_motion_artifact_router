"""AR-065: pre-action routing state 재구축 — AR-078 gate 비교의 재료.

전 feature = **원본 trajectory + prompt 만으로, tool 적용 전** 계산 (GT-free,
after-action leakage 금지 — spec ⚠️ Counterfactual 원칙). spec 목록 (결과 보기 전 고정):

  1 generator_id        배포 시 알려진 메타데이터
  2 foot_skate          v2 정의 (원본 관측)
  3 root_gait_mismatch  counterfactual: 접지발 골반-상대 후류 속도(gait 함의 속도)
                        − 실제 root 속도. induced_disp/path_gain 의 pre-action 대체물
  4 contact_persistence 접지 run 평균 길이 + 접지 비율
  5 trajectory_shape    경로 길이 + 직진도 (net displacement / path length)
  6 locomotion_intent   prompt keyword bucket (경량 rule)

출력: per-motion CSV (routing_benefit_permotion 과 gen/sid/seed 로 join 가능)
+ per-generator 분포 요약 JSON. label(ΔMM 등 보정 후 값)은 본 파일에 **없음**
(AR-078 에서 CSV join — state 산출과 label 산출의 물리적 분리).

CLI (motion3d env):
    python tools/preaction_state_ar065.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from correction_tools.coordinate_footskate_cleanup_tool import (
    DEFAULT_CONTACT_H, DEFAULT_CONTACT_VY, DEFAULT_GROUND_PERCENTILE,
    _estimate_ground, _v2_flags,
)
from tools.coords_protocol import PELVIS, LEFT_FOOT, RIGHT_FOOT
from tools.representative_pool_measure import foot_skate_world

POOL_ROOT = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608"
GENS = ("mdm", "motiongpt", "momask")

# locomotion intent 경량 keyword rule (AR-066 계열 — 사전 고정).
LOCO_KW = ("walk", "run", "jog", "stride", "march", "pace", "step forward", "moves forward",
           "walks", "runs", "jogs", "sprint", "crawl", "climb")
INPLACE_KW = ("in place", "in-place", "stand", "stands", "sit", "sits", "still",
              "stationary", "poses", "balance")


def _intent_bucket(prompt: str) -> str:
    p = prompt.lower()
    inplace = any(k in p for k in INPLACE_KW)
    loco = any(k in p for k in LOCO_KW)
    if loco and not inplace:
        return "locomotion"
    if inplace and not loco:
        return "in_place"
    return "ambiguous"


def extract_state(traj: np.ndarray) -> dict:
    """원본 trajectory [T,22,3] → pre-action geometry features (GT-free)."""
    T = traj.shape[0]
    ground = _estimate_ground(traj, DEFAULT_GROUND_PERCENTILE)
    pel_xz = traj[:, PELVIS, :][:, [0, 2]]
    v_root = np.linalg.norm(np.diff(pel_xz, axis=0), axis=1)      # [T-1]
    path_len = float(v_root.sum())
    net_disp = float(np.linalg.norm(pel_xz[-1] - pel_xz[0]))
    straightness = net_disp / max(path_len, 1e-6)

    # 접지 + gait 함의 속도 (counterfactual — root solve 가 회복할 벨트 속도).
    rel_speeds, runs, contact_frames = [], [], 0
    for f in (LEFT_FOOT, RIGHT_FOOT):
        contact, _ = _v2_flags(traj, f, ground, DEFAULT_CONTACT_H, DEFAULT_CONTACT_VY, 1e9)
        contact_frames += int(contact.sum())
        rel = traj[:, f, :][:, [0, 2]] - pel_xz
        rel_v = np.linalg.norm(np.diff(rel, axis=0), axis=1)      # 골반-상대 발 속도
        m = contact[:-1] & contact[1:]
        rel_speeds.extend(rel_v[m].tolist())
        # 접지 run 길이.
        run = 0
        for c in contact:
            if c:
                run += 1
            elif run:
                runs.append(run); run = 0
        if run:
            runs.append(run)
    gait_implied = float(np.mean(rel_speeds)) if rel_speeds else 0.0
    root_actual = float(v_root.mean())
    return {
        "foot_skate": foot_skate_world(traj),
        "gait_implied_speed": round(gait_implied, 6),
        "root_speed_actual": round(root_actual, 6),
        "root_gait_mismatch": round(gait_implied - root_actual, 6),
        "mismatch_ratio": round(gait_implied / max(root_actual, 1e-6), 3),
        "contact_run_mean": round(float(np.mean(runs)), 2) if runs else 0.0,
        "contact_fraction": round(contact_frames / (2 * T), 4),
        "path_length": round(path_len, 4),
        "straightness": round(straightness, 4),
        "n_frames": T,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-csv", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "preaction_state_ar065_v1.csv")
    ap.add_argument("--out-json", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "preaction_state_ar065_v1.json")
    args = ap.parse_args()

    rows = []
    for gen in GENS:
        metas = [json.load(open(p, encoding="utf-8")) for p in sorted((POOL_ROOT / gen).glob("*.json"))
                 if p.name != "_pool_summary.json"]
        if args.limit:
            metas = metas[:args.limit]
        for m in metas:
            traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
            st = extract_state(traj)
            st.update({"gen": gen, "sid": m["sample_id"], "seed": m["seed"],
                       "locomotion_intent": _intent_bucket(m["prompt"])})
            rows.append(st)
        print(f"[INFO] {gen}: {sum(1 for r in rows if r['gen'] == gen)} motions")

    cols = ["gen", "sid", "seed", "foot_skate", "gait_implied_speed", "root_speed_actual",
            "root_gait_mismatch", "mismatch_ratio", "contact_run_mean", "contact_fraction",
            "path_length", "straightness", "locomotion_intent", "n_frames"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})

    # per-generator 분포 요약 (informational).
    summary = {}
    for gen in GENS:
        sub = [r for r in rows if r["gen"] == gen]
        summary[gen] = {
            "n": len(sub),
            **{k: {"mean": round(float(np.mean([r[k] for r in sub])), 5),
                   "p50": round(float(np.median([r[k] for r in sub])), 5)}
               for k in ("foot_skate", "root_gait_mismatch", "mismatch_ratio",
                         "contact_fraction", "straightness")},
            "intent_dist": {b: sum(1 for r in sub if r["locomotion_intent"] == b)
                            for b in ("locomotion", "in_place", "ambiguous")},
        }
    out = {
        "schema_version": "1.0.0", "record_type": "preaction_state", "board_id": "AR-065",
        "leakage_audit": "전 feature = 원본 trajectory + prompt 만 사용 (correction 미실행, GT 미참조). "
                         "root_gait_mismatch 는 counterfactual (접지발 상대 후류 속도 − root 속도) — "
                         "induced_disp/path_gain 의 pre-action 대체물. label 은 본 파일에 없음 (AR-078 join).",
        "per_generator": summary,
        "csv": str(args.out_csv.relative_to(REPO_ROOT)) if args.out_csv.is_relative_to(REPO_ROOT) else str(args.out_csv),
        "claim_boundary": "state 재료 산출 — gate 성능 claim 없음 (AR-078). 기존 learned ranker (AR-040/052) "
                          "feature 재산출·re-train 판단은 AR-065 잔여 scope.",
    }
    json.dump(out, open(args.out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    import sys as _s; _s.stdout.reconfigure(encoding="utf-8")
    for gen in GENS:
        s = summary[gen]
        print(f"[{gen}] mismatch mean {s['root_gait_mismatch']['mean']} ratio p50 {s['mismatch_ratio']['p50']} "
              f"| skate {s['foot_skate']['mean']} | intent {s['intent_dist']}")
    print(f"[OK] wrote {args.out_csv} + {args.out_json}")


if __name__ == "__main__":
    main()
