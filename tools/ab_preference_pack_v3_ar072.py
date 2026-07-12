"""AR-072 ④: A/B v3 — 원본 vs root-aware 보정 (locomotion 한정, 새 pair 의 첫 지각 검정).

전 관문 통과 후 (기전 0.42→1.02 · R@1/MM-Dist/FID 전부 유의 개선 · 불변성 0/900).
본 pack 은 **root pair 의 지각 판정** — anchoring 계열의 "마지막 재검정 소진"과는
별개 pair 이나, 동일 규율: **사전 기준·blind·1회 원칙**.

사전 등록:
  - 처방: RootGaitConsistencyTool u=1.0 단독 (다리 무수정 — combo 아님).
  - 표본: **locomotion prompt 한정** (GT 평균 root 속도 > 0.01 — spec ①-보강 3:
    "walk in place" 류 급소 회피) 중 foot_skate_world worst-tail, **v1∪v2 의 40개
    sample_id 제외** (기억 오염 방지), 상위 20.
  - 재생: 실속도 20fps. blind: 좌우 무작위 (rng 20260716), answer_key 분리.
  - 판정 기준: 보정본 선호 ≥15/20 → 지지 (이항 양측 p≈0.041) → "intervention
    evidence + 지각 회복(b1)" / **≤12/20 → root pair 도 지각 기각** (1회 원칙 —
    추가 재시도 없음) / 13~14 → inconclusive (b2 검토).
  - 공시: over-correction 꼬리 (ratio_after>1.5 = 18.2%) — 해당 sample 이 표본에
    들어올 수 있으며 그로 인한 기각도 유효 결과.

CLI (motion3d env):
    python tools/ab_preference_pack_v3_ar072.py --n-pairs 2   # preflight
    python tools/ab_preference_pack_v3_ar072.py
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

from correction_tools import RootGaitConsistencyTool
from tools.coords_protocol import estimate_ground, PELVIS, LEFT_FOOT
from tools.representative_pool_measure import foot_skate_world
from tools.footskate_segment_gif_ar058_3f import _top_skate_segment
from tools.ab_preference_pack_ar058_3h import render_pair_gif

POOL = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608" / "mdm"
GT_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
V1_KEY = REPO_ROOT / "reports" / "figures" / "2026-07-08" / "ar058_3h_ab_preference_pack" / "answer_key_RESEARCHER_ONLY.csv"
V2_KEY = REPO_ROOT / "reports" / "figures" / "2026-07-12" / "ar058_3i_ab_v2_pack" / "answer_key_RESEARCHER_ONLY.csv"
OUT = REPO_ROOT / "reports" / "figures" / "2026-07-12" / "ar072_ab_v3_pack"
SNAP = REPO_ROOT / "evals" / "snapshots" / "ab_preference_pack_v3_ar072_v1.json"

PAIR_RNG_SEED = 20260716
PLAYBACK_FPS = 20.0
LOCO_THRESH = 0.010


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    rng = np.random.default_rng(PAIR_RNG_SEED)
    tool = RootGaitConsistencyTool()
    excl = set()
    for key in (V1_KEY, V2_KEY):
        excl |= {r["sample_id"] for r in csv.DictReader(open(key, encoding="utf-8"))}
    print(f"[INFO] v1+v2 제외 sample_id {len(excl)}개")

    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(POOL.glob("*.json"))
             if p.name != "_pool_summary.json"]
    by_prompt: dict[str, tuple[float, dict]] = {}
    gt_speed: dict[str, float] = {}
    for m in metas:
        sid = m["sample_id"]
        if sid in excl:
            continue
        if sid not in gt_speed:
            gt = np.load(GT_DIR / f"{sid}.npy").astype(np.float64)
            gt_speed[sid] = float(np.linalg.norm(
                np.diff(gt[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean())
        if gt_speed[sid] <= LOCO_THRESH:
            continue  # locomotion 한정 (spec ①-보강 3)
        traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        fs = foot_skate_world(traj)
        if sid not in by_prompt or fs > by_prompt[sid][0]:
            by_prompt[sid] = (fs, m)
    ranked = sorted(by_prompt.values(), key=lambda x: -x[0])
    print(f"[INFO] locomotion 후보 {len(ranked)} prompts, top fs={ranked[0][0]:.4f}")

    pairs, skipped = [], 0
    for fs_before, m in ranked:
        if len(pairs) >= args.n_pairs:
            break
        traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        T = traj.shape[0]
        ground = estimate_ground(traj)
        seg = _top_skate_segment(traj, ground)
        corr, rep = tool.apply(traj, target_part="root", target_joints=[],
                               frame_range=(0, T - 1), strength="large",
                               metadata={"coord_space": "trajectory"})
        if seg is None or rep.metadata["constrained_frame_frac"] < 0.3:
            skipped += 1
            continue
        foot, s, e, nsk = seg
        sid = m["sample_id"]
        va = float(np.linalg.norm(np.diff(corr[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean())
        k = len(pairs) + 1
        left_is = "corrected" if rng.random() < 0.5 else "original"
        m_left, m_right = (corr, traj) if left_is == "corrected" else (traj, corr)
        render_pair_gif(m_left, m_right, foot, s, e, ground,
                        args.out_dir / f"pair_{k:02d}.gif", fps=PLAYBACK_FPS)
        pairs.append({
            "pair_id": f"pair_{k:02d}", "sample_id": sid, "seed": m["seed"],
            "prompt": m["prompt"], "left_is": left_is,
            "fs_before": round(fs_before, 5), "fs_after": round(foot_skate_world(corr), 5),
            "ratio_before": round(float(np.linalg.norm(
                np.diff(traj[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean()) / gt_speed[sid], 3),
            "ratio_after": round(va / gt_speed[sid], 3),
            "skate_seg": [int(s), int(e)], "n_skate_frames": int(nsk),
            "foot": "left" if foot == LEFT_FOOT else "right",
        })
        p = pairs[-1]
        print(f"[OK] pair_{k:02d} {sid} fs {p['fs_before']:.4f}->{p['fs_after']:.4f} "
              f"ratio {p['ratio_before']:.2f}->{p['ratio_after']:.2f} left={left_is}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "rater_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "which_more_natural_A_or_B", "confidence_0_3", "notes"])
        for p in pairs:
            w.writerow([p["pair_id"], "", "", ""])
    idx = ["# AR-072 A/B v3 Preference Test (blind) — root-aware 처방의 첫 지각 검정", "",
           "> **재생 = 실제 속도 (20fps).** 각 쌍에서 **전체적인 자연스러움**을 보세요 —",
           "> 발-지면 접촉, 그리고 **몸이 걸음에 맞게 전진하는가**.",
           "> 왼쪽(A)/오른쪽(B) 중 더 자연스러운 쪽 강제선택 + 확신도(0~3) → [rater_sheet.csv](rater_sheet.csv)",
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

    fs_d = [p["fs_after"] - p["fs_before"] for p in pairs]
    snap = {
        "schema_version": "1.0.0", "record_type": "ab_preference_pack_v3_preregistration",
        "board_id": "AR-072", "status": "pack_built_awaiting_ratings",
        "treatment": "RootGaitConsistencyTool u=1.0 단독 (다리 무수정)",
        "selection_rule": f"MDM locomotion(GT speed>{LOCO_THRESH}) worst-tail fs, v1+v2 40개 제외, 상위 20",
        "stimulus": {"playback_fps": PLAYBACK_FPS},
        "blinding": {"pair_rng_seed": PAIR_RNG_SEED,
                     "left_corrected_count": sum(1 for p in pairs if p["left_is"] == "corrected")},
        "preregistered_criterion": {
            "rater_n": "1 (b1)",
            "support": ">=15/20 → root pair 'intervention evidence + 지각 회복(b1)'",
            "fail": "<=12/20 → root pair 지각 기각 (1회 원칙 — 추가 재시도 없음)",
            "inconclusive": "13~14 → b2 검토"},
        "honesty_disclosure": {
            "fs_delta_mean": round(float(np.mean(fs_d)), 5),
            "fs_improved_pairs": int(sum(1 for d in fs_d if d < 0)),
            "ratio_after_over_1.5_pairs": int(sum(1 for p in pairs if p["ratio_after"] > 1.5)),
            "note": "over-correction 꼬리 (pool 18.2%) 가 표본에 포함될 수 있음 — 그로 인한 기각도 유효"},
        "pack_stats": {"n_pairs": len(pairs), "skipped": skipped},
        "pairs": pairs,
        "evidence_tier": "b1 예정",
        "claim_boundary": "root pair 전용. 성공해도 'intervention evidence' 표현까지 (인과 증명 아님).",
    }
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    json.dump(snap, open(SNAP, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[DONE] {len(pairs)} pairs (skipped {skipped}) -> {args.out_dir}")
    print(f"  fs mean {np.mean(fs_d):+.5f} | ratio>1.5 pairs: {sum(1 for p in pairs if p['ratio_after']>1.5)}")
    print(f"  snapshot: {SNAP}")


if __name__ == "__main__":
    main()
