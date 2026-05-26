"""Step B (사용자 directive 2026-05-26): G2 top correction side-by-side 3-panel GIF.

사용자 directive (2026-05-26):
> "지금 제일 중요한 sample은 synthetic이 아니라 G2 입니다. 우선 이 4개만 봐야 합니다:
>  motion_006, motion_007, motion_008, motion_028. 각각 original / B2-small / 5-level oracle
>  을 side-by-side로 만들어야 합니다. overlay 말고 좌우 비교가 좋습니다.
>  목적: NetGain이 높은 G2 보정이 실제로 좋아 보이는지, 아니면 다리 길이/자세/리듬 왜곡이
>  생기는지 확인."

본 도구는 4 G2 natural top-correction sample 의 3-panel side-by-side GIF 생성:
- Panel 1: Original (no correction)
- Panel 2: B2-val-best (= B2-small for G2, per AGENTS.md sec3-18 + 부록 V)
- Panel 3: 5-level oracle (oracle_sequence_g2_natural_5level_v1 의 best sequence)

Y-up convention 의무 (AGENTS.md sec3-19). 동일 xlim/ylim/zlim, 동일 frame index 동기화.

CLI:
    python -m tools.g2_top_correction_side_by_side_gif \
        --output-dir reports/figures/2026-05-26/g2_top_correction_side_by_side
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
sys.path.insert(0, str(REPO_ROOT / "external_assets" / "code"))
from plot_3d_motion import T2M_KINEMATIC_CHAIN, _valid_chain  # type: ignore

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool

TOOL_BY_NAME: dict[str, CorrectionTool] = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}

TOP4_SAMPLES = ["motion_006", "motion_007", "motion_008", "motion_028"]


def _apply_sequence(motion: np.ndarray, sequence: list) -> np.ndarray:
    T = motion.shape[0]
    out = motion.copy()
    for step in sequence:
        if step[0] in ("STOP", "APPLY_FAIL", "SCORE_VIOLATION_ROLLBACK"):
            break
        tool = TOOL_BY_NAME[step[0]]
        out, _ = tool.apply(out, target_part=step[1], target_joints=[],
                             frame_range=(0, T - 1), strength=step[2])
    return out


def _apply_b2(motion: np.ndarray, strength: str = "small") -> np.ndarray:
    """G2 의 B2-val-best = B2-small (부록 V, 100%)."""
    T = motion.shape[0]
    out, _ = TOOL_BY_NAME["VelocitySmoothingTool"].apply(
        motion, target_part="full_body", target_joints=[],
        frame_range=(0, T - 1), strength=strength,
    )
    return out


def _draw_floor(ax, xlim, ylim, zlim):
    verts = [
        [xlim[0], ylim[0], zlim[0]],
        [xlim[0], ylim[0], zlim[1]],
        [xlim[1], ylim[0], zlim[1]],
        [xlim[1], ylim[0], zlim[0]],
    ]
    plane = Poly3DCollection([verts])
    plane.set_facecolor((0.5, 0.5, 0.5, 0.20))
    ax.add_collection3d(plane)


def _draw_skeleton(ax, frame_xyz, color, lw=2.6, marker_size=18):
    for chain in T2M_KINEMATIC_CHAIN:
        valid = _valid_chain(chain, frame_xyz)
        if len(valid) < 2:
            continue
        ax.plot3D(frame_xyz[valid, 0], frame_xyz[valid, 1], frame_xyz[valid, 2],
                  linewidth=lw, color=color)
        ax.scatter(frame_xyz[valid, 0], frame_xyz[valid, 1], frame_xyz[valid, 2],
                   c=color, s=marker_size, depthshade=False)


def _setup_axis(ax, xlim, ylim, zlim, panel_title):
    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(*ylim)
    ax.set_zlim3d(*zlim)
    ax.view_init(elev=15.0, azim=-70.0, vertical_axis="y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(panel_title, fontsize=10)
    _draw_floor(ax, xlim, ylim, zlim)


def _render_3panel_frame(
    idx: int, motions: dict[str, np.ndarray], panel_titles: list[str],
    xlim, ylim, zlim, suptitle: str, color: str = "#ff8c00",
) -> Image.Image:
    """3 panel side-by-side (Original / B2-small / 5-level oracle)."""
    fig = plt.figure(figsize=(16, 6))
    keys = list(motions.keys())
    for i, key in enumerate(keys):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        _setup_axis(ax, xlim, ylim, zlim, panel_titles[i])
        motion = motions[key]
        if idx < motion.shape[0]:
            _draw_skeleton(ax, motion[idx], color=color, lw=2.4)
    fig.suptitle(f"{suptitle}\nframe {idx + 1}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _common_lims(motions: list[np.ndarray], pad: float = 0.20):
    cat = np.concatenate(motions, axis=0)
    cat = cat[~np.isnan(cat).any(axis=-1)]
    cat = cat.reshape(-1, 3)
    return (
        (float(cat[:, 0].min()) - pad, float(cat[:, 0].max()) + pad),
        (float(cat[:, 1].min()) - pad, float(cat[:, 1].max()) + pad),
        (float(cat[:, 2].min()) - pad, float(cat[:, 2].max()) + pad),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 top correction side-by-side 3-panel GIF (Step B)")
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_v1")
    parser.add_argument("--snapshot", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "oracle_sequence_g2_natural_5level_v1.json")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "reports" / "figures" / "2026-05-26" / "g2_top_correction_side_by_side")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    args = parser.parse_args()

    with open(args.snapshot, encoding="utf-8") as f:
        snap = json.load(f)
    snap_by = {s["trial_id"]: s for s in snap["per_sample"]}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, tid in enumerate(TOP4_SAMPLES, 1):
        npy_path = args.data_dir / f"{tid}.npy"
        if not npy_path.exists():
            print(f"[WARN] [{i}/{len(TOP4_SAMPLES)}] {tid}: missing {npy_path}")
            continue
        if tid not in snap_by:
            print(f"[WARN] [{i}/{len(TOP4_SAMPLES)}] {tid}: missing in snapshot")
            continue
        motion = np.load(str(npy_path)).astype(np.float64)
        s = snap_by[tid]
        prompt = s.get("g2_prompt", "")[:80]
        seq = s["best"]["sequence"]
        seq_len = s["best"]["length"]
        netgain = s["best"]["netgain"]
        seq_summary = " → ".join(f"{a[0].replace('Tool','')}/{a[2]}" for a in seq)
        print(f"[{i}/{len(TOP4_SAMPLES)}] {tid} (T={motion.shape[0]}, ng={netgain:+.4f}, len={seq_len})")
        print(f"   prompt: '{prompt}...'")
        print(f"   oracle: {seq_summary}")

        b2 = _apply_b2(motion, "small")
        oracle = _apply_sequence(motion, seq)

        # Synchronize and downsample.
        m = min(motion.shape[0], b2.shape[0], oracle.shape[0])
        mot_s = motion[:m:args.frame_stride].astype(np.float32)
        b2_s = b2[:m:args.frame_stride].astype(np.float32)
        oracle_s = oracle[:m:args.frame_stride].astype(np.float32)
        T_eff = mot_s.shape[0]

        xlim, ylim, zlim = _common_lims([mot_s, b2_s, oracle_s])
        motions = {"original": mot_s, "b2_small": b2_s, "oracle_5level": oracle_s}
        panel_titles = [
            "Original (no correction)",
            "B2-val-best (= B2-small for G2)",
            f"5-level oracle (NetGain={netgain:+.4f}, len={seq_len})",
        ]
        suptitle = f"{tid} — '{prompt}' — Oracle sequence: {seq_summary}"

        frames: list[Image.Image] = []
        for idx in range(T_eff):
            frames.append(_render_3panel_frame(idx, motions, panel_titles,
                                                xlim, ylim, zlim, suptitle))
        dur = int(1000 / args.fps)
        durations = [dur] * T_eff
        durations[-1] = 800
        out = args.output_dir / f"{tid}_side_by_side_3panel.gif"
        frames[0].save(str(out), save_all=True, append_images=frames[1:],
                        duration=durations, loop=0, optimize=False)
        print(f"   [OK] saved: {out}  ({T_eff} frames)")

    print(f"\n[OK] all 4 side-by-side GIFs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
