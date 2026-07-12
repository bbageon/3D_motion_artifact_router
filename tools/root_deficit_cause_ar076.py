"""AR-076: root deficit 원인 3-way 진단 (원본 MDM vs GT, GPU 불필요).

A (root 채널): 발-골반 상대 stride 진폭·cadence 가 GT 수준인가 (관절 gait 정상, root 만 문제).
B (평균 회귀): v_root_MDM vs v_root_GT 회귀 기울기 < 1 + 이동거리 분산 GT 대비 축소.
C (적분 오차): 부족의 시간 누적 — 전반부 vs 후반부 speed ratio.

per-prompt (seed 평균), MDM locomotion prompt (GT root speed > 0.01). 통계: bootstrap CI + OLS.

CLI (motion3d env):
    python tools/root_deficit_cause_ar076.py
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

from tools.coords_protocol import PELVIS, LEFT_FOOT, RIGHT_FOOT, estimate_ground
from correction_tools.coordinate_footskate_cleanup_tool import _v2_flags

POOL = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608" / "mdm"
GT_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
LOCO_THRESH = 0.010
V2 = (0.05, 0.035, 1e9)


def _root_speed(m):
    return float(np.linalg.norm(np.diff(m[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean())


def _half_speeds(m):
    v = np.linalg.norm(np.diff(m[:, PELVIS, :][:, [0, 2]], axis=0), axis=1)
    h = len(v) // 2
    return float(v[:h].mean()), float(v[h:].mean())


def _stride_metrics(m, ground):
    """발-골반 상대 stride: 접지 중 발이 골반 기준 뒤로 흐르는 진폭(=보폭) + 접지 횟수(=cadence)."""
    amps, contacts = [], 0
    T = m.shape[0]
    for f in (LEFT_FOOT, RIGHT_FOOT):
        contact, _ = _v2_flags(m, f, ground, *V2)
        rel_x = m[:, f, 0] - m[:, PELVIS, 0]         # 골반-상대 발 x
        # 접지 run 별 진폭 (뒤로 흐른 총량) + run 개수.
        run = []
        for i, c in enumerate(contact):
            if c:
                run.append(rel_x[i])
            elif run:
                amps.append(max(run) - min(run)); contacts += 1; run = []
        if run:
            amps.append(max(run) - min(run)); contacts += 1
    stride = float(np.median(amps)) if amps else 0.0
    cadence = contacts / (T / 20.0)                   # 접지/초 (20fps)
    return stride, cadence


def _boot_ci(a, rng, n=1000):
    a = np.asarray(a); m = len(a)
    bs = [float(np.mean(a[rng.integers(0, m, m)])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "root_deficit_cause_ar076_v1.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(POOL.glob("*.json"))
             if p.name != "_pool_summary.json"]
    if args.limit:
        metas = metas[:args.limit]

    per = defaultdict(lambda: defaultdict(list))
    gt_cache = {}
    for m in metas:
        traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        g = estimate_ground(traj)
        sid = m["sample_id"]
        per[sid]["vr"].append(_root_speed(traj))
        fh, sh = _half_speeds(traj)
        per[sid]["fh"].append(fh); per[sid]["sh"].append(sh)
        st, cad = _stride_metrics(traj, g)
        per[sid]["stride"].append(st); per[sid]["cadence"].append(cad)
        if sid not in gt_cache:
            gt = np.load(GT_DIR / f"{sid}.npy").astype(np.float64)
            gg = float(np.percentile(np.minimum(gt[:, LEFT_FOOT, 1], gt[:, RIGHT_FOOT, 1]), 10))
            gst, gcad = _stride_metrics(gt, gg)
            gt_cache[sid] = {"vr": _root_speed(gt), "stride": gst, "cadence": gcad}

    rows = []
    for sid, d in per.items():
        gt = gt_cache[sid]
        if gt["vr"] <= LOCO_THRESH:
            continue
        rows.append({
            "sid": sid, "vr": float(np.mean(d["vr"])), "gt_vr": gt["vr"],
            "ratio": float(np.mean(d["vr"])) / gt["vr"],
            "fh": float(np.mean(d["fh"])), "sh": float(np.mean(d["sh"])),
            "stride": float(np.mean(d["stride"])), "gt_stride": gt["stride"],
            "cadence": float(np.mean(d["cadence"])), "gt_cadence": gt["cadence"],
        })
    n = len(rows)
    vr = np.array([r["vr"] for r in rows]); gvr = np.array([r["gt_vr"] for r in rows])

    # ---- A: 발-골반 상대 stride/cadence 가 GT 수준인가 (root 만 문제면 gait 는 정상).
    stride_ratio = np.array([r["stride"] / r["gt_stride"] for r in rows if r["gt_stride"] > 1e-6])
    cad_ratio = np.array([r["cadence"] / r["gt_cadence"] for r in rows if r["gt_cadence"] > 1e-6])
    A = {
        "stride_ratio_MDM_over_GT": {"mean": round(float(stride_ratio.mean()), 4), "ci95": _boot_ci(stride_ratio, rng)},
        "cadence_ratio_MDM_over_GT": {"mean": round(float(cad_ratio.mean()), 4), "ci95": _boot_ci(cad_ratio, rng)},
        "verdict": None,  # 아래 채움
    }

    # ---- B: v_root_MDM = a + b·v_root_GT 회귀. B 지지 = 기울기 b < 1 (큰 이동일수록 더 축소) + 분산 축소.
    b, a = np.polyfit(gvr, vr, 1)
    # bootstrap 기울기 CI.
    slopes = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        slopes.append(np.polyfit(gvr[idx], vr[idx], 1)[0])
    slope_ci = [round(float(np.percentile(slopes, 2.5)), 4), round(float(np.percentile(slopes, 97.5)), 4)]
    B = {
        "ols_slope": round(float(b), 4), "ols_intercept": round(float(a), 5), "slope_ci95": slope_ci,
        "std_MDM": round(float(vr.std()), 5), "std_GT": round(float(gvr.std()), 5),
        "std_ratio_MDM_over_GT": round(float(vr.std() / gvr.std()), 4),
        "verdict": None,
    }

    # ---- C: 전반부 vs 후반부 speed ratio (부족의 시간 누적).
    fh = np.array([r["fh"] for r in rows]); sh = np.array([r["sh"] for r in rows])
    half_ratio = sh / np.maximum(fh, 1e-9)  # 후반/전반 (C 지지 = < 1, 후반 더 느려짐)
    C = {
        "first_half_speed_mean": round(float(fh.mean()), 5), "second_half_speed_mean": round(float(sh.mean()), 5),
        "second_over_first_ratio": {"mean": round(float(half_ratio.mean()), 4), "ci95": _boot_ci(half_ratio, rng)},
        "verdict": None,
    }

    # verdicts (CI 기반).
    A["verdict"] = ("root-only 지지 (gait 정상)" if A["stride_ratio_MDM_over_GT"]["ci95"][0] > 0.8
                    else "gait 도 축소 (root-only 아님)" if A["stride_ratio_MDM_over_GT"]["ci95"][1] < 0.8
                    else "부분")
    B["verdict"] = ("지지 (기울기<1 + 분산 축소)" if (slope_ci[1] < 1.0 and B["std_ratio_MDM_over_GT"] < 0.9)
                    else "부분" if slope_ci[1] < 1.0 or B["std_ratio_MDM_over_GT"] < 0.9 else "기각")
    C["verdict"] = ("지지 (후반 감속)" if C["second_over_first_ratio"]["ci95"][1] < 0.95
                    else "기각 (누적 편향 없음)" if C["second_over_first_ratio"]["ci95"][0] > 1.05 else "불확정")

    out = {
        "schema_version": "1.0.0", "record_type": "root_deficit_cause", "board_id": "AR-076",
        "n_locomotion_prompts": n, "split_id": "protocol_rep_pool_seed20260608 mdm + HumanML3D GT",
        "candidate_A_root_channel": A,
        "candidate_B_mean_regression": B,
        "candidate_C_integration_drift": C,
        "overall_ratio_mean": round(float((vr / gvr).mean()), 4),
        "claim_boundary": "관측 진단 (재학습/ablation 아님) — 확정 원인 주장 금지. MDM·단일 벤치마크 한정.",
        "grounding": ["Guo HumanML3D CVPR2022 (표현)", "Tevet MDM ICLR2023 (x0 예측)", "Guo MoMask CVPR2024 (VQ 대비)"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[A root채널] stride ratio {A['stride_ratio_MDM_over_GT']['mean']} ci{A['stride_ratio_MDM_over_GT']['ci95']} "
          f"cadence {A['cadence_ratio_MDM_over_GT']['mean']} -> {A['verdict']}")
    print(f"[B 평균회귀] slope {B['ols_slope']} ci{slope_ci} std_ratio {B['std_ratio_MDM_over_GT']} -> {B['verdict']}")
    print(f"[C 적분오차] 후반/전반 {C['second_over_first_ratio']['mean']} ci{C['second_over_first_ratio']['ci95']} -> {C['verdict']}")
    print(f"[OK] n={n}, overall ratio {(vr/gvr).mean():.3f}; wrote {args.output}")


if __name__ == "__main__":
    main()
