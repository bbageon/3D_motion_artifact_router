"""AR-058-3i: A/B v2 preference pack — combo(cleanup→BoneProjection) vs 원본 (마지막 재검정).

v1 (AR-058-3h) 결과: cleanup 단독 보정본 선호 11/20 (우연) → fail 분기. 사용자 지각
관찰("발이 짧아져 땅으로 들어간 느낌") → AR-070 이 처방을 검증: **combo = cleanup(u=0.75)
→ BoneProjection(양 다리, large)** 가 legCV 완전 제거(fi 100%) + fs 3.5배 증폭(fi 90%).
본 pack 은 그 combo 의 **지각 가치**를 판정하는 사전등록 A/B — **마지막 재검정**
(repair 1회 소진; 무한 재시도 = p-hacking 차단).

v1 과의 차이 (사전 등록):
  - 처방: cleanup 단독 → **combo** (기존 tool 조합, AR-070 결정 규칙 충족)
  - 표본: **rank 21~40 신규 20 prompt** (v1 의 20개 제외 — 기억 오염 방지)
  - 공시: **float +≈1.5cm trade** 를 pair metadata 에 박제 (그로 인한 기각도 유효 결과)
  - 판정 기준 (동일 구조): 보정본 선호 >=15/20 → 지지 (이항 양측 p≈0.041) /
    **<=12/20 → Stop 방향** (마지막 재검정 소진) / 13~14 → inconclusive (b2 검토)

렌더 protocol = v1 재사용 (side-by-side 동기, marker 없음, world 좌표, Y-up §3-19, blind).

CLI (motion3d env):
    python tools/ab_preference_pack_v2_ar058_3i.py --n-pairs 2   # preflight
    python tools/ab_preference_pack_v2_ar058_3i.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import BoneProjectionTool, CoordinateFootSkateCleanupTool
from evaluators import BoneLengthCVEvaluator
from tools.coords_protocol import estimate_ground, PELVIS, LEFT_FOOT
from tools.representative_pool_measure import foot_skate_world
from tools.physical_metric_g2_stress import float_mag
from tools.footskate_segment_gif_ar058_3f import _top_skate_segment
from tools.ab_preference_pack_ar058_3h import render_pair_gif  # v1 렌더 protocol 재사용

POOL = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608" / "mdm"
V1_KEY = REPO_ROOT / "reports" / "figures" / "2026-07-08" / "ar058_3h_ab_preference_pack" / "answer_key_RESEARCHER_ONLY.csv"
OUT = REPO_ROOT / "reports" / "figures" / "2026-07-12" / "ar058_3i_ab_v2_pack"
SNAP = REPO_ROOT / "evals" / "snapshots" / "ab_preference_pack_v2_ar058_3i_v1.json"

U_CLEANUP = 0.75
PAIR_RNG_SEED = 20260712
N_LEG_BONES = 8
#: 재생 실속도 (canonical 20fps, §3-1). Amendment (2026-07-12, 응답 수집 전):
#: v1/초기 v2 는 8fps(2.5배 슬로모) 렌더 — 사용자 지적 "느려서 발이 끌리는 것처럼
#: 보임" → 슬로모는 동적 결함(slide) 지각을 약화하고 정적 결함(hover)은 그대로
#: 보여 보정본에 비대칭 불리. v2 판정은 실속도로.
PLAYBACK_FPS = 20.0


def apply_combo(traj, cleanup, bone_tool):
    T = traj.shape[0]
    out, cleanup_report = cleanup.apply(traj, target_part="both_feet", target_joints=[],
                                        frame_range=(0, T - 1), strength="medium",
                                        metadata={"continuous_u": U_CLEANUP,
                                                  "ground_y": estimate_ground(traj),
                                                  "coord_space": "trajectory"})
    for part in ("left_leg", "right_leg"):
        out, _rep = bone_tool.apply(out, target_part=part, target_joints=[],
                                    frame_range=(0, T - 1), strength="large")
    return out, cleanup_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    rng = np.random.default_rng(PAIR_RNG_SEED)
    cleanup = CoordinateFootSkateCleanupTool()
    bone_tool = BoneProjectionTool()
    bone_ev = BoneLengthCVEvaluator()

    # v1 표본 제외 목록.
    v1_sids = {r["sample_id"] for r in csv.DictReader(open(V1_KEY, encoding="utf-8"))}
    print(f"[INFO] v1 제외 sample_id {len(v1_sids)}개")

    # 선정: prompt 별 max-seed foot_skate_world 순위 → v1 제외 후 상위 n.
    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(POOL.glob("*.json"))
             if p.name != "_pool_summary.json"]
    by_prompt: dict[str, tuple[float, dict]] = {}
    for m in metas:
        traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        fs = foot_skate_world(traj)
        if m["sample_id"] not in by_prompt or fs > by_prompt[m["sample_id"]][0]:
            by_prompt[m["sample_id"]] = (fs, m)
    ranked = [x for x in sorted(by_prompt.values(), key=lambda x: -x[0])
              if x[1]["sample_id"] not in v1_sids]
    print(f"[INFO] 후보 {len(ranked)} prompts (v1 제외 후), top fs={ranked[0][0]:.4f}")

    def leg_cv(local):
        r = bone_ev.evaluate(local)[0]
        return float(np.asarray(r.metadata["all_bone_cvs"], dtype=float)[:N_LEG_BONES].max())

    pairs, skipped = [], 0
    for fs_before, m in ranked:
        if len(pairs) >= args.n_pairs:
            break
        traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        ground = estimate_ground(traj)
        seg = _top_skate_segment(traj, ground)
        combo, rep_cleanup = apply_combo(traj, cleanup, bone_tool)
        if seg is None or rep_cleanup.metadata["n_correct"] < 1:
            skipped += 1
            continue
        foot, s, e, nsk = seg
        lb = traj - traj[:, PELVIS:PELVIS + 1, :]
        la = combo - combo[:, PELVIS:PELVIS + 1, :]
        k = len(pairs) + 1
        left_is = "corrected" if rng.random() < 0.5 else "original"
        m_left, m_right = (combo, traj) if left_is == "corrected" else (traj, combo)
        gif = args.out_dir / f"pair_{k:02d}.gif"
        render_pair_gif(m_left, m_right, foot, s, e, ground, gif, fps=PLAYBACK_FPS)
        pairs.append({
            "pair_id": f"pair_{k:02d}", "sample_id": m["sample_id"], "seed": m["seed"],
            "prompt": m["prompt"], "left_is": left_is,
            "fs_before": round(fs_before, 5), "fs_after": round(foot_skate_world(combo), 5),
            "leg_cv_before": round(leg_cv(lb), 5), "leg_cv_after": round(leg_cv(la), 5),
            "float_before": round(float_mag(lb), 5), "float_after": round(float_mag(la), 5),
            "skate_seg": [int(s), int(e)], "n_skate_frames": int(nsk),
            "foot": "left" if foot == LEFT_FOOT else "right",
        })
        p = pairs[-1]
        print(f"[OK] pair_{k:02d} {p['sample_id']} fs {p['fs_before']:.4f}->{p['fs_after']:.4f} "
              f"legCV {p['leg_cv_before']:.3f}->{p['leg_cv_after']:.3f} "
              f"float {p['float_before']:.3f}->{p['float_after']:.3f} left={left_is}")

    # rater 파일 (blind).
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "rater_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "which_more_natural_A_or_B", "confidence_0_3", "notes"])
        for p in pairs:
            w.writerow([p["pair_id"], "", "", ""])
    idx = ["# AR-058-3i A/B v2 Preference Test (blind) — 마지막 재검정", "",
           "> **재생 = 실제 속도 (20fps)** — 지난 pack 과 달리 슬로모가 아닙니다.",
           "> 각 쌍에서 **발과 지면의 접촉, 다리의 자연스러움**을 보세요.",
           "> 왼쪽(A)과 오른쪽(B) 중 **어느 쪽이 더 자연스럽습니까?** 반드시 하나 선택 (강제선택)",
           "> + 확신도(0~3) → [rater_sheet.csv](rater_sheet.csv). 두 패널은 같은 구간 동기 재생.",
           "> 정답은 시트에 없습니다. answer_key 는 판정 전 열람 금지.", "",
           "| Pair | GIF |", "|---|---|"]
    for p in pairs:
        idx.append(f"| {p['pair_id']} | [gif]({p['pair_id']}.gif) |")
    (args.out_dir / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    with open(args.out_dir / "answer_key_RESEARCHER_ONLY.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(pairs[0].keys()))
        for p in pairs:
            w.writerow(list(p.values()))

    # 사전등록 snapshot.
    fs_d = [p["fs_after"] - p["fs_before"] for p in pairs]
    fl_d = [p["float_after"] - p["float_before"] for p in pairs]
    cv_d = [p["leg_cv_after"] - p["leg_cv_before"] for p in pairs]
    snap = {
        "schema_version": "1.0.0", "record_type": "ab_preference_pack_v2_preregistration",
        "board_id": "AR-058-3i", "status": "pack_built_awaiting_ratings",
        "final_retest_notice": "마지막 재검정 — repair-then-retest 1회 소진 (AR-069). 추가 재시도 없음 사전 명시.",
        "treatment": {"combo": "CoordinateFootSkateCleanupTool(u=0.75, trajectory) -> BoneProjectionTool(left_leg+right_leg, large)",
                      "rationale": "AR-070 P4: legCV 완전 제거(fi 100%) + fs 3.5배(fi 90%); 사전 정의 결정 규칙 충족"},
        "selection_rule": "MDM pool prompt별 max-seed foot_skate_world 순위에서 v1 20개 제외 후 상위 (rank 21~40 상당)",
        "blinding": {"pair_rng_seed": PAIR_RNG_SEED,
                     "left_corrected_count": sum(1 for p in pairs if p["left_is"] == "corrected")},
        "stimulus": {"playback_fps": PLAYBACK_FPS, "native_fps": 20,
                     "amendment_2026_07_12": "응답 수집 전 fps 8→20 (실속도). 사유: 슬로모(2.5x)가 "
                                             "동적 결함(slide) 지각을 약화, 정적 결함(hover)은 유지 — "
                                             "보정본에 비대칭 불리 (사용자 지적). v1 결과에는 caveat 주석."},
        "preregistered_criterion": {
            "rater_n": "1 (b1; 외부 claim 은 3+ 필요 — AR-023)",
            "support": ">=15/20 (이항 양측 p≈0.041) → P5 어조 '지각 선호 확인(b1)' + H-204 경로 개방",
            "fail": "<=12/20 → **Stop 방향** — 연구를 측정·감사·STOP 계층으로 재정의 (§10 재소집)",
            "inconclusive": "13~14 → 평가자 추가(b2) 검토"},
        "honesty_disclosure": {
            "float_trade": {"mean_delta": round(float(np.mean(fl_d)), 5),
                            "note": "combo 는 float 를 평균 +1.5cm 수준 올림 (AR-070) — 이로 인한 기각도 유효 결과 (후속: gate-filtered 적용 또는 AR-064)"},
            "fs_delta_mean": round(float(np.mean(fs_d)), 5),
            "leg_cv_delta_mean": round(float(np.mean(cv_d)), 5),
            "fs_improved_pairs": int(sum(1 for d in fs_d if d < 0)),
            "leg_cv_improved_pairs": int(sum(1 for d in cv_d if d < 0))},
        "pack_stats": {"n_pairs": len(pairs), "skipped": skipped},
        "pairs": pairs,
        "evidence_tier": "b1 예정 (§3-17)",
        "claim_boundary": "primary pair 전용. 보조 pair 일반화 금지. v1 과 표본 비중복 (기억 오염 방지).",
    }
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    json.dump(snap, open(SNAP, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[DONE] {len(pairs)} pairs (skipped {skipped}) -> {args.out_dir}")
    print(f"  fs mean {np.mean(fs_d):+.5f} ({sum(1 for d in fs_d if d<0)}/{len(pairs)} improved) | "
          f"legCV mean {np.mean(cv_d):+.4f} | float mean {np.mean(fl_d):+.4f}")
    print(f"  snapshot: {SNAP}")


if __name__ == "__main__":
    main()
