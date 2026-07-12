"""AR-071: root/gait mismatch 진단 — 사용자 가설 검정 (사전 확정 순서 ②).

가설 (사용자, 2026-07-12): "foot skating 은 root trajectory 와 gait speed mismatch 의
결과일 수 있음" — 다리는 걷는데 root 전진이 부족 → 접지발이 world 에서 **뒤로** 끌림.

검정 3종 (비순환 설계):
  T1. **방향 검정** (직접): skate frame(v2 정의)에서 발 world 속도를 진행방향(heading =
      골반 수평속도 방향)에 투영한 부호. 가설 예측 = **음수(뒤로/문워크) 우세**.
      (노이즈성 skate 라면 방향 무작위 ≈ 50%.)
  T2. **GT 대비 root 전진 부족**: 같은 prompt 의 HumanML3D GT 와 골반 평균 수평속도 비교.
      speed_ratio = v_root(MDM)/v_root(GT). 가설 예측 = **ratio < 1** (locomotion prompt).
  T3. **부족 ↔ fs 상관**: deficit = 1 − speed_ratio 와 foot_skate_world 의 Spearman.
      가설 예측 = **양(+)의 상관**.

설계 노트 (spec 정밀화): spec 초안의 v_gait_implied(접지 중 발-골반 상대속도) − v_root 는
fs 와 수학적으로 얽혀(접지 중 상대속도 = root속도 + 미끄럼) tautology 위험 → **fs 정의와
독립인 골반 속도만 쓰는 GT 기준 deficit** 으로 교체. 원 정의는 참고 지표로만 병기.

CLI (motion3d env):
    python tools/root_gait_mismatch_ar071.py --limit 12   # preflight
    python tools/root_gait_mismatch_ar071.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from correction_tools.coordinate_footskate_cleanup_tool import _v2_flags
from tools.coords_protocol import estimate_ground, PELVIS, LEFT_FOOT, RIGHT_FOOT
from tools.representative_pool_measure import foot_skate_world

POOL = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608" / "mdm"
GT_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"

MOVE_THRESH = 0.005   # m/frame — heading 정의 최소 골반 속도 (0.1 m/s)
LOCO_THRESH = 0.010   # m/frame — GT 기준 locomotion prompt 판정 (0.2 m/s)
V2 = (0.05, 0.035, 0.025)


def _horiz_vel(arr_xz):
    v = np.diff(arr_xz, axis=0)
    return np.concatenate([v, v[-1:]], axis=0)  # [T,2], last pad


def analyze_sample(traj):
    ground = estimate_ground(traj)
    pel = traj[:, PELVIS, :][:, [0, 2]]
    pv = _horiz_vel(pel)                        # 골반 수평 속도 벡터 [T,2]
    ps = np.linalg.norm(pv, axis=1)             # 속도 크기
    moving = ps > MOVE_THRESH
    heading = np.zeros_like(pv)
    heading[moving] = pv[moving] / ps[moving, None]

    signed_list, rel_speed_list = [], []
    n_skate_frames = 0
    for foot in (LEFT_FOOT, RIGHT_FOOT):
        contact, skate = _v2_flags(traj, foot, ground, *V2)
        fxz = traj[:, foot, :][:, [0, 2]]
        fv = _horiz_vel(fxz)                    # 발 world 속도
        mask = skate & moving
        n_skate_frames += int(skate.sum())
        if mask.any():
            signed_list.append(np.einsum("ij,ij->i", fv[mask], heading[mask]))
        cm = contact & moving
        if cm.any():
            rel_speed_list.append(np.linalg.norm(fv[cm] - pv[cm], axis=1))  # 참고용 (circular-risk)

    signed = np.concatenate(signed_list) if signed_list else np.array([])
    rel = np.concatenate(rel_speed_list) if rel_speed_list else np.array([])
    return {
        "v_root_mean": float(ps.mean()),
        "moving_frac": float(moving.mean()),
        "n_signed": int(signed.size),
        "backward_frac": float((signed < 0).mean()) if signed.size else np.nan,
        "mean_signed": float(signed.mean()) if signed.size else np.nan,
        "v_gait_implied_ref": float(rel.mean()) if rel.size else np.nan,  # 참고용
        "fs": foot_skate_world(traj),
    }


def _boot_ci(vals, rng, n=1000):
    nv = len(vals)
    bs = [float(np.mean(vals[rng.integers(0, nv, nv)])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "root_gait_mismatch_ar071_v1.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(POOL.glob("*.json"))
             if p.name != "_pool_summary.json"]
    if args.limit:
        metas = metas[:args.limit]

    per_prompt = defaultdict(lambda: defaultdict(list))
    gt_speed_cache: dict[str, float] = {}
    n_no_gt = 0
    for m in metas:
        traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        r = analyze_sample(traj)
        sid = m["sample_id"]
        for k, v in r.items():
            per_prompt[sid][k].append(v)
        if sid not in gt_speed_cache:
            gt_p = GT_DIR / f"{sid}.npy"
            if gt_p.exists():
                gt = np.load(gt_p).astype(np.float64)
                gt_speed_cache[sid] = float(np.linalg.norm(
                    np.diff(gt[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean())
            else:
                gt_speed_cache[sid] = np.nan
                n_no_gt += 1

    # prompt 단위 집계.
    rows = []
    for sid, d in per_prompt.items():
        gt_s = gt_speed_cache[sid]
        bf = [x for x in d["backward_frac"] if not np.isnan(x)]
        ms = [x for x in d["mean_signed"] if not np.isnan(x)]
        rows.append({
            "sid": sid,
            "v_root": float(np.mean(d["v_root_mean"])),
            "gt_speed": gt_s,
            "fs": float(np.mean(d["fs"])),
            "backward_frac": float(np.mean(bf)) if bf else np.nan,
            "mean_signed": float(np.mean(ms)) if ms else np.nan,
            "n_signed": int(np.sum(d["n_signed"])),
        })

    # ---- T1: 방향 검정 (skate frame 있는 prompt).
    t1 = [r for r in rows if r["n_signed"] >= 5]
    bfrac = np.array([r["backward_frac"] for r in t1])
    msigned = np.array([r["mean_signed"] for r in t1])
    t1_res = {
        "n_prompts": len(t1),
        "backward_frac_mean": round(float(bfrac.mean()), 4),
        "backward_frac_ci95": _boot_ci(bfrac, rng),
        "prompts_majority_backward": round(float((bfrac > 0.5).mean()), 4),
        "mean_signed_slide_m_per_frame": round(float(msigned.mean()), 5),
        "wilcoxon_p_signed_vs_0": (round(float(wilcoxon(msigned[msigned != 0]).pvalue), 8)
                                    if (msigned != 0).sum() >= 10 else None),
    }

    # ---- T2/T3: locomotion prompt (GT 존재 + GT speed > LOCO_THRESH).
    loco = [r for r in rows if not np.isnan(r["gt_speed"]) and r["gt_speed"] > LOCO_THRESH]
    ratio = np.array([r["v_root"] / r["gt_speed"] for r in loco])
    deficit = 1.0 - ratio
    fs_arr = np.array([r["fs"] for r in loco])
    rho, pval = spearmanr(deficit, fs_arr)
    t2_res = {
        "n_locomotion_prompts": len(loco),
        "speed_ratio_mean": round(float(ratio.mean()), 4),
        "speed_ratio_ci95": _boot_ci(ratio, rng),
        "speed_ratio_p50": round(float(np.percentile(ratio, 50)), 4),
        "prompts_ratio_below_1": round(float((ratio < 1.0).mean()), 4),
    }
    t3_res = {"spearman_rho_deficit_vs_fs": round(float(rho), 4),
              "p_value": float(pval),
              "n": len(loco)}

    # verdict 규칙 (CI 기반 — 점추정만으로 지지 판정 금지):
    #   T1 지지 = backward_frac CI 하한 > 0.5 AND Wilcoxon p < 0.05. CI 가 0.5 포함 → 불확정.
    #   T2 지지 = ratio CI 상한 < 1.0. T3 지지 = rho > 0 AND p < 0.05.
    t1_ci_lo = t1_res["backward_frac_ci95"][0]
    t1_p = t1_res["wilcoxon_p_signed_vs_0"]
    verdict = {
        "T1_direction": ("지지" if (t1_ci_lo > 0.5 and t1_p is not None and t1_p < 0.05)
                          else ("불확정" if t1_res["backward_frac_mean"] > 0.5 else "기각")),
        "T2_deficit_exists": "지지" if t2_res["speed_ratio_ci95"][1] < 1.0 else "기각",
        "T3_deficit_explains_fs": "지지" if (rho > 0 and pval < 0.05) else "기각",
    }
    n_support = sum(1 for v in verdict.values() if v == "지지")
    verdict["overall"] = ("지지" if n_support == 3 else
                          "부분 지지" if n_support >= 1 else "기각")

    out = {
        "schema_version": "1.0.0", "record_type": "root_gait_mismatch_diagnostic",
        "board_id": "AR-071",
        "hypothesis": "foot skating 은 root trajectory 와 gait speed mismatch 의 결과일 수 있음 (사용자 2026-07-12)",
        "split_id": "protocol_rep_pool_seed20260608 mdm (300 prompts x 3 seeds) + HumanML3D GT",
        "design_note": "T2/T3 deficit 은 fs 정의와 독립인 골반 속도만 사용 (GT 기준) — spec 초안의 "
                       "접지 상대속도 기반 mismatch 는 fs 와 수학적 얽힘(tautology 위험)으로 참고 지표로 강등",
        "thresholds": {"MOVE_THRESH": MOVE_THRESH, "LOCO_THRESH": LOCO_THRESH, "v2": V2,
                       "t1_min_skate_frames": 5},
        "T1_direction_test": t1_res,
        "T2_root_deficit_vs_GT": t2_res,
        "T3_deficit_fs_correlation": t3_res,
        "coverage": {"n_prompts_total": len(rows), "n_no_gt": n_no_gt},
        "preregistered_predictions": {"T1": "backward 우세 (>0.5)", "T2": "ratio < 1", "T3": "rho > 0, p<0.05"},
        "verdict": verdict,
        "claim_boundary": "기전 진단 (상관+방향) — 인과 확증은 root-rescale 개입 실험으로만. "
                          "지지 시 root-aware correction 은 새 pair (새 사전등록 + §3-11 게이트).",
        "grounding": ["GMD ICCV2023 / EDGE CVPR2023 (contact 일관성·trajectory 유도 동기) — "
                      "본 가설 자체는 internal hypothesis"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"T1 방향:   skate 미끄럼의 뒤(-)방향 비율 mean={t1_res['backward_frac_mean']} "
          f"ci{t1_res['backward_frac_ci95']} | 과반이 뒤인 prompt {t1_res['prompts_majority_backward']*100:.1f}% "
          f"(n={t1_res['n_prompts']})")
    print(f"T2 부족:   v_root(MDM)/v_root(GT) mean={t2_res['speed_ratio_mean']} "
          f"ci{t2_res['speed_ratio_ci95']} | ratio<1 prompt {t2_res['prompts_ratio_below_1']*100:.1f}% "
          f"(locomotion n={t2_res['n_locomotion_prompts']})")
    print(f"T3 상관:   Spearman rho={t3_res['spearman_rho_deficit_vs_fs']} p={t3_res['p_value']:.2e}")
    print(f"VERDICT: {verdict}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
