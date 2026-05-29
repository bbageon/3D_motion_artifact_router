"""Step G-4 (사용자 directive 2026-05-28): perceptual distortion validation GIF.

사용자 directive:
> "B2가 metric상 악화되는 것이 사람 눈에도 over-smoothing / distortion으로 보이는가?
>  safe oracle은 정말 원본 quality를 보존하는가?"

본 도구는 b2_artifact_best 의 bone CV 증가 가 큰 top-N G2 sample 에 대해 3-panel
side-by-side GIF 생성: Original | b2_artifact_best (B2-large over-smoothing) | safe_oracle.

사람 평가 질문: panel 2 (B2) 가 distortion (다리 길이 / 자세) 으로 보이는가?
panel 3 (safe oracle) 이 panel 1 (original) 의 quality 를 보존하는가?

Y-up convention 의무 (AGENTS.md §3-19).

CLI:
    python -m tools.g4_perceptual_distortion_gif \
        --output-dir reports/figures/2026-05-28/g4_perceptual_distortion
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external_assets" / "code"))
from plot_3d_motion import T2M_KINEMATIC_CHAIN, _valid_chain  # type: ignore

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS

TOOL_BY_NAME: dict[str, CorrectionTool] = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}
# Top-6 b2_artifact_best bone CV distortion samples (computed 2026-05-28).
TOP_SAMPLES = ["motion_159", "motion_044", "motion_185", "motion_197", "motion_154", "motion_208"]


def _b2_artifact_best(m):
    best = None
    for st in ("small", "medium", "large"):
        out, _ = TOOL_BY_NAME["VelocitySmoothingTool"].apply(
            m, target_part="full_body", target_joints=[], frame_range=(0, m.shape[0]-1), strength=st)
        art = sum(max((r.score for r in ev.evaluate(out)), default=0.0) for ev in DEFAULT_EVALUATORS)
        if best is None or art < best[0]:
            best = (art, out, st)
    return best[1], best[2]


def _apply_seq(m, seq):
    T = m.shape[0]; out = m.copy()
    for s in seq:
        if s[0] not in TOOL_BY_NAME: break
        out, _ = TOOL_BY_NAME[s[0]].apply(out, target_part=s[1], target_joints=[], frame_range=(0, T-1), strength=s[2])
    return out


def _setup_axis(ax, xlim, ylim, zlim, title):
    ax.set_xlim3d(*xlim); ax.set_ylim3d(*ylim); ax.set_zlim3d(*zlim)
    ax.view_init(elev=15.0, azim=-70.0, vertical_axis="y")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_title(title, fontsize=10)
    verts = [[[xlim[0], ylim[0], zlim[0]], [xlim[0], ylim[0], zlim[1]],
              [xlim[1], ylim[0], zlim[1]], [xlim[1], ylim[0], zlim[0]]]]
    pl = Poly3DCollection(verts); pl.set_facecolor((0.5, 0.5, 0.5, 0.2)); ax.add_collection3d(pl)


def _draw(ax, frame, color):
    for chain in T2M_KINEMATIC_CHAIN:
        v = _valid_chain(chain, frame)
        if len(v) < 2: continue
        ax.plot3D(frame[v, 0], frame[v, 1], frame[v, 2], linewidth=2.4, color=color)
        ax.scatter(frame[v, 0], frame[v, 1], frame[v, 2], c=color, s=18, depthshade=False)


def _common_lims(ms, pad=0.2):
    cat = np.concatenate(ms, 0).reshape(-1, 3)
    return ((cat[:,0].min()-pad, cat[:,0].max()+pad), (cat[:,1].min()-pad, cat[:,1].max()+pad),
            (cat[:,2].min()-pad, cat[:,2].max()+pad))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_g2_hml3d_test300_3level_v1.json")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "reports" / "figures" / "2026-05-28" / "g4_perceptual_distortion")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    args = parser.parse_args()

    oracle = {p["trial_id"]: p for p in json.load(open(args.oracle, encoding="utf-8"))["per_sample"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    panel_labels = ["Original", "B2-artifact-best (over-smoothing)", "safe oracle (gate-aware)"]

    for tid in TOP_SAMPLES:
        npy = args.g2_batch_dir / f"{tid}.npy"
        if not npy.exists():
            print(f"[WARN] {tid}: missing"); continue
        m = np.load(str(npy)).astype(np.float64)
        b2, b2_st = _b2_artifact_best(m)
        sb = oracle.get(tid, {}).get("safe_best")
        so = _apply_seq(m, sb["sequence"]) if (sb and sb["length"] > 0) else m
        so_label = "STOP(=orig)" if not (sb and sb["length"] > 0) else "corrected"
        mm = min(m.shape[0], b2.shape[0], so.shape[0])
        m_s = m[:mm:args.frame_stride].astype(np.float32)
        b2_s = b2[:mm:args.frame_stride].astype(np.float32)
        so_s = so[:mm:args.frame_stride].astype(np.float32)
        Teff = m_s.shape[0]
        xlim, ylim, zlim = _common_lims([m_s, b2_s, so_s])
        motions = [m_s, b2_s, so_s]; colors = ["#888888", "#ff8c00", "#3273dc"]
        suptitle = f"{tid} — Original | B2-{b2_st} (over-smooth) | safe oracle ({so_label})"
        frames = []
        for idx in range(Teff):
            fig = plt.figure(figsize=(16, 6))
            for i in range(3):
                ax = fig.add_subplot(1, 3, i+1, projection="3d")
                _setup_axis(ax, xlim, ylim, zlim, panel_labels[i])
                if idx < motions[i].shape[0]:
                    _draw(ax, motions[i][idx], colors[i])
            fig.suptitle(f"{suptitle}\nframe {idx+1}/{Teff}", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=95); plt.close(fig); buf.seek(0)
            frames.append(Image.open(buf).convert("RGB"))
        dur = int(1000/args.fps); durations = [dur]*Teff; durations[-1] = 800
        out = args.output_dir / f"{tid}_3panel.gif"
        frames[0].save(str(out), save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
        manifest.append({"trial_id": tid, "gif": str(out.relative_to(REPO_ROOT)),
                         "b2_strength": b2_st, "safe_oracle": so_label,
                         "panels": ["Original", f"B2-{b2_st}", f"safe_oracle({so_label})"]})
        print(f"[OK] {tid}: B2-{b2_st}, safe={so_label} ({Teff} frames)")

    # INSTRUCTIONS + manifest.
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "task": "G-4 perceptual distortion validation",
        "question": "Panel 2 (B2 over-smoothing) 가 distortion 으로 보이는가? Panel 3 (safe oracle) 이 Panel 1 (Original) quality 보존하는가?",
        "samples": manifest,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    instructions = """\
﻿# Step G-4 Perceptual Distortion Validation — 사용자 작업 안내

## 질문 (사용자 directive 2026-05-28)
1. **Panel 2 (B2-artifact-best, over-smoothing)** 가 Panel 1 (Original) 대비 distortion
   (다리 길이 변형 / 자세 부자연 / 리듬 손실) 으로 보이는가?
2. **Panel 3 (safe oracle, gate-aware)** 가 Panel 1 (Original) 의 quality 를 보존하는가?

## GIF (top-6 b2_artifact_best bone CV distortion samples)
reports/figures/2026-05-28/g4_perceptual_distortion/*_3panel.gif

각 GIF 3-panel: **Original (gray) | B2-artifact-best (orange) | safe oracle (blue)**.

## 평가 (각 sample 별)
| 질문 | 응답 |
|---|---|
| B2 (panel 2) distortion 보이는가? | yes / no / subtle |
| safe oracle (panel 3) original 보존하는가? | yes / no |

## 해석 분기
- B2 distortion 다수 visible + safe oracle 보존 → **standard metric (FID/bone CV) 악화 가 perceptual 로 confirm** (quality-validated evidence).
- B2 distortion 안 보임 → metric 악화 가 perceptual threshold 미만 (caveat: G2 low/moderate severity).

본 평가 = b1 sub-tier (1명 internal sanity, AGENTS.md §3-17). 정식 evidence 는 3명+ (b2/b3).
"""
    (args.output_dir / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    print(f"\n[OK] {len(manifest)} GIFs + manifest + INSTRUCTIONS in {args.output_dir}")


if __name__ == "__main__":
    main()
