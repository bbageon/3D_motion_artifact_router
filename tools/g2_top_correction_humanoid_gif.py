"""Humanoid side-by-side GIFs for G2 top correction cases.

This renderer is a fast qualitative-view helper. It does not fit a true SMPL
mesh; instead, it draws the canonical 22-joint motion as thick body-like limbs
with larger head/torso/foot markers. The goal is to make visual inspection less
ambiguous than thin skeleton lines while keeping the dependency surface small.

For true SMPL rendering, use MotionGPT's official joints2smpl + render path
(`external_assets/MotionGPT/fit.py` and `render.py`) as a slower follow-up.

CLI:
    python -m tools.g2_top_correction_humanoid_gif \
        --output-dir reports/figures/2026-05-26/g2_top_correction_humanoid
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
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.g2_top_correction_side_by_side_gif import (  # noqa: E402
    TOP4_SAMPLES,
    _apply_b2,
    _apply_sequence,
    _common_lims,
)

# Canonical SMPL-22 edges. The widths are intentionally non-uniform to make
# torso/legs read as a body instead of a pure stick figure.
EDGES: list[tuple[int, int, float]] = [
    # legs
    (0, 1, 8.0), (1, 4, 8.0), (4, 7, 7.5), (7, 10, 7.0),
    (0, 2, 8.0), (2, 5, 8.0), (5, 8, 7.5), (8, 11, 7.0),
    # torso / head
    (0, 3, 10.0), (3, 6, 10.0), (6, 9, 10.0), (9, 12, 8.0), (12, 15, 7.0),
    # arms
    (9, 13, 6.5), (13, 16, 6.0), (16, 18, 5.5), (18, 20, 5.0),
    (9, 14, 6.5), (14, 17, 6.0), (17, 19, 5.5), (19, 21, 5.0),
]

JOINT_SIZES = {
    0: 70,   # pelvis
    9: 85,   # chest
    12: 65,  # neck
    15: 155, # head
    10: 90, 11: 90, # feet
    20: 70, 21: 70, # hands
}


def _draw_floor(ax, xlim, ylim, zlim) -> None:
    verts = [
        [xlim[0], ylim[0], zlim[0]],
        [xlim[0], ylim[0], zlim[1]],
        [xlim[1], ylim[0], zlim[1]],
        [xlim[1], ylim[0], zlim[0]],
    ]
    plane = Poly3DCollection([verts])
    plane.set_facecolor((0.42, 0.42, 0.42, 0.18))
    plane.set_edgecolor((0.30, 0.30, 0.30, 0.25))
    ax.add_collection3d(plane)


def _setup_axis(ax, xlim, ylim, zlim, title: str) -> None:
    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(*ylim)
    ax.set_zlim3d(*zlim)
    ax.view_init(elev=14.0, azim=-72.0, vertical_axis="y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title, fontsize=10)
    _draw_floor(ax, xlim, ylim, zlim)


def _draw_humanoid(ax, frame: np.ndarray, color: str) -> None:
    # Body-like thick limbs.
    for a, b, width in EDGES:
        seg = frame[[a, b]]
        ax.plot3D(
            seg[:, 0], seg[:, 1], seg[:, 2],
            color=color, linewidth=width, alpha=0.93,
        )

    # Joint/head/feet markers help identify deformation and contact.
    sizes = np.array([JOINT_SIZES.get(i, 38) for i in range(frame.shape[0])])
    ax.scatter(
        frame[:, 0], frame[:, 1], frame[:, 2],
        s=sizes, c=color, alpha=0.96, depthshade=False,
        edgecolors="#1f1f1f", linewidths=0.35,
    )

    # Foot pads: dark markers make foot contact easier to inspect.
    feet = frame[[10, 11]]
    ax.scatter(
        feet[:, 0], feet[:, 1], feet[:, 2],
        s=145, c="#202020", marker="s", alpha=0.80, depthshade=False,
    )


def _render_frame(
    idx: int,
    motions: dict[str, np.ndarray],
    titles: list[str],
    xlim,
    ylim,
    zlim,
    suptitle: str,
) -> Image.Image:
    fig = plt.figure(figsize=(16, 6))
    colors = ["#2f80ed", "#6c757d", "#ff8c00"]
    for i, (key, motion) in enumerate(motions.items()):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        _setup_axis(ax, xlim, ylim, zlim, titles[i])
        if idx < motion.shape[0]:
            _draw_humanoid(ax, motion[idx], colors[i])
    fig.suptitle(f"{suptitle}\nframe {idx + 1}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 top correction humanoid 3-panel GIF renderer")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "external_assets" / "g2_generated_v1",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=REPO_ROOT / "evals" / "snapshots" / "oracle_sequence_g2_natural_5level_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "figures" / "2026-05-26" / "g2_top_correction_humanoid",
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    args = parser.parse_args()

    with open(args.snapshot, encoding="utf-8") as f:
        snap = json.load(f)
    by_trial = {s["trial_id"]: s for s in snap["per_sample"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, tid in enumerate(TOP4_SAMPLES, 1):
        npy_path = args.data_dir / f"{tid}.npy"
        if not npy_path.exists() or tid not in by_trial:
            print(f"[WARN] [{i}/{len(TOP4_SAMPLES)}] skip {tid}")
            continue

        original = np.load(str(npy_path)).astype(np.float64)
        rec = by_trial[tid]
        seq = rec["best"]["sequence"]
        netgain = rec["best"]["netgain"]
        prompt = rec.get("g2_prompt", "")[:80]
        seq_summary = " -> ".join(f"{a[0].replace('Tool','')}/{a[2]}" for a in seq) or "STOP"

        b2_small = _apply_b2(original, "small")
        oracle = _apply_sequence(original, seq)
        m = min(original.shape[0], b2_small.shape[0], oracle.shape[0])
        original = original[:m:args.frame_stride].astype(np.float32)
        b2_small = b2_small[:m:args.frame_stride].astype(np.float32)
        oracle = oracle[:m:args.frame_stride].astype(np.float32)

        xlim, ylim, zlim = _common_lims([original, b2_small, oracle], pad=0.26)
        motions = {"original": original, "b2_small": b2_small, "oracle": oracle}
        titles = [
            "Original",
            "B2-small",
            f"5-level oracle (NetGain={netgain:+.3f})",
        ]
        suptitle = f"{tid} - {prompt} - oracle: {seq_summary}"
        frames = [
            _render_frame(idx, motions, titles, xlim, ylim, zlim, suptitle)
            for idx in range(original.shape[0])
        ]
        durations = [int(1000 / args.fps)] * len(frames)
        durations[-1] = 850
        out = args.output_dir / f"{tid}_humanoid_3panel.gif"
        frames[0].save(
            str(out),
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            optimize=False,
        )
        print(f"[OK] {tid}: {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
