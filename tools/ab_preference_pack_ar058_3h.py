"""AR-058-3h: A/B preference pack — MDM worst-tail, 원본 vs coord 보정본 (blind, 강제선택).

목적: "artifact 가 사람 눈에 명확한 문제로 안 보인다"(사용자 관찰, 절대평가) 를
**비교평가(A/B 강제선택, 검안 방식)** 로 전환해 결정적으로 검정한다.
대상 = 확정 연구 가닥 §0-0 의 primary pair (foot skating ↔ coord cleanup) 만.

사전 등록 (결과 보기 전 고정 — snapshot 에 박제):
  - 선정 규칙: MDM representative pool 에서 prompt 별 max-seed foot_skate_world 상위 20.
  - 보정: CoordinateFootSkateCleanupTool, **u=0.75 고정** — 근거: AR-061 pool 평균에서
    max 효과(u=1.0)의 ~65% fs 감소를 유지하면서 bone_cv 상승은 절반 이하
    (+0.031 vs +0.047) → "skate 개선" 과 "propagation bone 왜곡" 의 지각 혼입 최소화.
  - 판정 기준 (단일 평가자 b1 pilot): 보정본 선호 ≥15/20 → 유의 지지 (이항 양측 p≈0.041)
    → P5 어조 "지각 선호 확인(b1)". ≤12/20 → primary pair 지각 claim 불성립
    → §10 Go/Stop 정식 소집. 13~14 → inconclusive → 평가자 추가(b2, AR-023).
  - 단일 평가자 결과는 §3-17 b1 tier — 외부 공개 claim 은 3+ 평가자(b2/b3) 필요.

Blinding: pair 별 좌우 배치 무작위(rng 20260708), rater 파일에 원본/보정 표기 없음.
정답은 answer_key.csv (연구자 전용) 에만.

CLI (motion3d env):
    python tools/ab_preference_pack_ar058_3h.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import CoordinateFootSkateCleanupTool
from evaluators import BoneLengthCVEvaluator
from skeleton_normalizer.canonical_smpl_22 import T2M_KINEMATIC_CHAIN
from tools.coords_protocol import estimate_ground, PELVIS, LEFT_FOOT, RIGHT_FOOT
from tools.representative_pool_measure import foot_skate_world
from tools.footskate_segment_gif_ar058_3f import _top_skate_segment

POOL = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608" / "mdm"
OUT = REPO_ROOT / "reports" / "figures" / "2026-07-08" / "ar058_3h_ab_preference_pack"
SNAP = REPO_ROOT / "evals" / "snapshots" / "ab_preference_pack_ar058_3h_v1.json"

N_PAIRS = 20
U_FIXED = 0.75          # 사전 등록 (모듈 docstring 근거)
PAIR_RNG_SEED = 20260708
FPS = 8
PAD = 5


def _draw_panel(ax, motion, t, foot, seg_start, ground, lims):
    (cx, cy, cz, mx) = lims
    ax.clear()
    ax.view_init(elev=12, azim=-70, vertical_axis="y")  # §3-19 Y-up
    ax.set_xlim(cx - mx, cx + mx); ax.set_ylim(cy - mx, cy + mx); ax.set_zlim(cz - mx, cz + mx)
    for chain in T2M_KINEMATIC_CHAIN:
        for a, b in zip(chain[:-1], chain[1:]):
            ax.plot([motion[t, a, 0], motion[t, b, 0]], [motion[t, a, 1], motion[t, b, 1]],
                    [motion[t, a, 2], motion[t, b, 2]], color="#333", lw=1.6)
    gx = np.array([cx - mx, cx + mx]); gz = np.array([cz - mx, cz + mx])
    GX, GZ = np.meshgrid(gx, gz)
    ax.plot_surface(GX, np.full_like(GX, ground), GZ, alpha=0.12, color="gray")
    # 발 marker + trail — 양 패널 동일 스타일 (blind: 색·문구로 정체 힌트 금지).
    ax.scatter([motion[t, foot, 0]], [motion[t, foot, 1]], [motion[t, foot, 2]],
               color="#1f77b4", s=80, edgecolors="k", zorder=5)
    tr = motion[seg_start:t + 1, foot]
    ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="#1f77b4", lw=2.5, alpha=0.85)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def render_pair_gif(m_left, m_right, foot, s, e, ground, out_path):
    """좌/우 동기 재생 side-by-side GIF. 축 범위 = 두 motion 의 segment 합집합 (동일 스케일)."""
    T = m_left.shape[0]
    s0, e0 = max(0, s - PAD), min(T - 1, e + PAD)
    seg = np.concatenate([m_left[s0:e0 + 1], m_right[s0:e0 + 1]], axis=0)
    xmin, xmax = seg[:, :, 0].min(), seg[:, :, 0].max()
    ymin, ymax = seg[:, :, 1].min(), seg[:, :, 1].max()
    zmin, zmax = seg[:, :, 2].min(), seg[:, :, 2].max()
    mx = max(xmax - xmin, zmax - zmin, ymax - ymin) * 0.55 + 0.1
    lims = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, mx)

    fig = plt.figure(figsize=(11, 5.8))
    axL = fig.add_subplot(121, projection="3d")
    axR = fig.add_subplot(122, projection="3d")

    def draw(t):
        _draw_panel(axL, m_left, t, foot, s0, ground, lims)
        _draw_panel(axR, m_right, t, foot, s0, ground, lims)
        axL.set_title("A", fontsize=14)
        axR.set_title("B", fontsize=14)
        fig.suptitle(f"frame {t}", fontsize=10, color="#666")

    anim = FuncAnimation(fig, draw, frames=range(s0, e0 + 1), interval=1000 / FPS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=FPS))
    # §3-19 첫 frame 검사용 PNG.
    draw(s0)
    fig.savefig(str(out_path.with_suffix("")) + "_f0.png", dpi=70)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(PAIR_RNG_SEED)
    tool = CoordinateFootSkateCleanupTool()
    bone_ev = BoneLengthCVEvaluator()

    # ---- 1) worst-tail 선정: prompt 별 max-seed fs 상위 N_PAIRS.
    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(POOL.glob("*.json"))
             if p.name != "_pool_summary.json"]
    by_prompt: dict[str, tuple[float, dict]] = {}
    for m in metas:
        traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        fs = foot_skate_world(traj)
        if m["sample_id"] not in by_prompt or fs > by_prompt[m["sample_id"]][0]:
            by_prompt[m["sample_id"]] = (fs, m)
    ranked = sorted(by_prompt.values(), key=lambda x: -x[0])
    print(f"[INFO] MDM prompts={len(by_prompt)}  worst fs={ranked[0][0]:.4f}  20th={ranked[19][0]:.4f}")

    # ---- 2) pair 생성.
    pairs, skipped = [], 0
    for fs_before, m in ranked:
        if len(pairs) >= N_PAIRS:
            break
        traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        T = traj.shape[0]
        ground = estimate_ground(traj)
        corr, rep = tool.apply(traj, target_part="both_feet", target_joints=[],
                               frame_range=(0, T - 1), strength="medium",
                               metadata={"continuous_u": U_FIXED, "ground_y": ground,
                                         "coord_space": "trajectory"})
        seg = _top_skate_segment(traj, ground)
        if rep.metadata["n_correct"] < 1 or seg is None:
            skipped += 1
            continue
        foot, s, e, nsk = seg
        fs_after = foot_skate_world(corr)
        lb = traj - traj[:, PELVIS:PELVIS + 1, :]
        la = corr - corr[:, PELVIS:PELVIS + 1, :]
        cv_b = max((r.score for r in bone_ev.evaluate(lb)), default=0.0)
        cv_a = max((r.score for r in bone_ev.evaluate(la)), default=0.0)

        k = len(pairs) + 1
        left_is = "corrected" if rng.random() < 0.5 else "original"
        m_left, m_right = (corr, traj) if left_is == "corrected" else (traj, corr)
        gif = OUT / f"pair_{k:02d}.gif"
        render_pair_gif(m_left, m_right, foot, s, e, ground, gif)
        pairs.append({
            "pair_id": f"pair_{k:02d}", "sample_id": m["sample_id"], "seed": m["seed"],
            "prompt": m["prompt"], "left_is": left_is,
            "fs_before": round(fs_before, 5), "fs_after": round(fs_after, 5),
            "bone_cv_before": round(cv_b, 5), "bone_cv_after": round(cv_a, 5),
            "skate_seg": [int(s), int(e)], "n_skate_frames": int(nsk),
            "foot": "left" if foot == LEFT_FOOT else "right",
        })
        print(f"[OK] pair_{k:02d} {m['sample_id']} fs {fs_before:.4f}->{fs_after:.4f} "
              f"boneCV {cv_b:.4f}->{cv_a:.4f} left={left_is}")

    # ---- 3) rater 파일 (blind).
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "rater_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "which_more_natural_A_or_B", "confidence_0_3", "notes"])
        for p in pairs:
            w.writerow([p["pair_id"], "", "", ""])
    idx = ["# AR-058-3h A/B Preference Test (blind)", "",
           "> 각 쌍에서 **발과 지면의 접촉**을 보세요. 왼쪽(A)과 오른쪽(B) 중",
           "> **어느 쪽이 더 자연스럽습니까?** 반드시 하나를 선택하고 (강제선택),",
           "> 확신도(0=모르겠음 ~ 3=확실)를 함께 적어주세요 → [rater_sheet.csv](rater_sheet.csv)",
           "> 두 패널은 같은 구간을 동기 재생합니다. 정답(어느 쪽이 무엇인지)은 시트에 없습니다.", "",
           "| Pair | GIF |", "|---|---|"]
    for p in pairs:
        idx.append(f"| {p['pair_id']} | [gif]({p['pair_id']}.gif) |")
    (OUT / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    # 정답 key (연구자 전용 — 판정 전 열람 금지).
    with open(OUT / "answer_key_RESEARCHER_ONLY.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(pairs[0].keys()))
        for p in pairs:
            w.writerow(list(p.values()))

    # ---- 4) 사전 등록 snapshot (응답 수집 전 박제).
    fs_deltas = [p["fs_after"] - p["fs_before"] for p in pairs]
    snap = {
        "schema_version": "1.0.0", "record_type": "ab_preference_pack_preregistration",
        "board_id": "AR-058-3h",
        "status": "pack_built_awaiting_ratings",
        "selection_rule": "MDM representative pool, prompt별 max-seed foot_skate_world 상위 20 (n_correct>=1)",
        "correction": {"tool": "CoordinateFootSkateCleanupTool", "u_fixed": U_FIXED,
                       "u_rationale": "AR-061: u=1.0 대비 fs 효과 ~65% 유지, bone_cv 상승 절반 이하 — 왜곡 혼입 최소화"},
        "blinding": {"pair_rng_seed": PAIR_RNG_SEED, "left_corrected_count":
                     sum(1 for p in pairs if p["left_is"] == "corrected")},
        "preregistered_criterion": {
            "rater_n": "1 (b1 pilot; 외부 claim 은 3+ 필요 — AR-023)",
            "support": "보정본 선호 >=15/20 (이항 양측 p~0.041) → P5 어조 '지각 선호 확인(b1)'",
            "fail": "<=12/20 → primary pair 지각 claim 불성립 → AGENTS §10 Go/Stop 정식 소집",
            "inconclusive": "13~14 → 평가자 추가 (b2)"
        },
        "pack_stats": {"n_pairs": len(pairs), "skipped": skipped,
                       "fs_delta_mean": round(float(np.mean(fs_deltas)), 5),
                       "fs_improved_pairs": int(sum(1 for d in fs_deltas if d < 0))},
        "pairs": pairs,
        "evidence_tier": "b1 (단일 평가자) 예정 — §3-17 quality-validated 는 b2/b3 부터",
        "claim_boundary": "본 pack 은 primary pair 전용. 보조 pair 로 일반화 금지.",
    }
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    json.dump(snap, open(SNAP, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[DONE] {len(pairs)} pairs -> {OUT}")
    print(f"  fs delta mean {np.mean(fs_deltas):+.5f} | improved {sum(1 for d in fs_deltas if d<0)}/{len(pairs)}")
    print(f"  snapshot: {SNAP}")


if __name__ == "__main__":
    main()
