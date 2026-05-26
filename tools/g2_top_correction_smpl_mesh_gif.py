"""Step B-2-3: SMPL mesh 3-panel side-by-side GIF (motion-router env).

mgpt env 의 _smpl_fit_inline.py 가 생성한 {tid}_{variant}_mesh.npy 와 _smpl_faces.npy
를 읽고, matplotlib Poly3DCollection 으로 mesh rendering. Y-up convention (AGENTS.md §3-19).

각 GIF = 3-panel (Original / B2-small / 5-level oracle) 동기화 frame.

CLI:
    python -m tools.g2_top_correction_smpl_mesh_gif \
        --staging-dir external_assets/MotionGPT/staging/g2_top4 \
        --output-dir reports/figures/2026-05-26/g2_top_correction_smpl_mesh
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

TOP4_SAMPLES = ["motion_006", "motion_007", "motion_008", "motion_028"]
VARIANTS = ["original", "b2small", "oracle5level"]
PANEL_LABELS = {
    "original": "Original (no correction)",
    "b2small": "B2-val-best (= B2-small for G2)",
    "oracle5level": "5-level oracle",
}


def _setup_axis(ax, xlim, ylim, zlim, title):
    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(*ylim)
    ax.set_zlim3d(*zlim)
    ax.view_init(elev=10.0, azim=-70.0, vertical_axis="y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title, fontsize=10)
    # Floor (Y-up: y=ymin).
    verts = [[
        [xlim[0], ylim[0], zlim[0]], [xlim[0], ylim[0], zlim[1]],
        [xlim[1], ylim[0], zlim[1]], [xlim[1], ylim[0], zlim[0]],
    ]]
    plane = Poly3DCollection(verts)
    plane.set_facecolor((0.5, 0.5, 0.5, 0.20))
    ax.add_collection3d(plane)


def _draw_mesh(ax, verts: np.ndarray, faces: np.ndarray, color, alpha=0.85):
    """Draw SMPL mesh as Poly3DCollection."""
    # faces.shape = (13776, 3) int indices into verts (6890, 3).
    tris = verts[faces]  # (13776, 3, 3)
    mesh = Poly3DCollection(tris, alpha=alpha, linewidths=0.05)
    mesh.set_facecolor(color)
    mesh.set_edgecolor((0.15, 0.15, 0.15, 0.3))
    ax.add_collection3d(mesh)


def _render_frame(
    idx: int, mesh_data: dict[str, np.ndarray], faces: np.ndarray,
    panel_titles: list[str], xlim, ylim, zlim, suptitle: str, frame_total: int,
) -> Image.Image:
    fig = plt.figure(figsize=(16, 6))
    colors = ["#a0a0a0", "#7ab2e8", "#ff8c00"]
    for i, key in enumerate(VARIANTS):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        _setup_axis(ax, xlim, ylim, zlim, panel_titles[i])
        verts = mesh_data[key][idx] if idx < mesh_data[key].shape[0] else mesh_data[key][-1]
        _draw_mesh(ax, verts, faces, color=colors[i])
    fig.suptitle(f"{suptitle}\nframe {idx + 1}/{frame_total}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _common_lims(meshes: list[np.ndarray], pad: float = 0.10):
    cat = np.concatenate(meshes, axis=0).reshape(-1, 3)
    return (
        (float(cat[:, 0].min()) - pad, float(cat[:, 0].max()) + pad),
        (float(cat[:, 1].min()) - pad, float(cat[:, 1].max()) + pad),
        (float(cat[:, 2].min()) - pad, float(cat[:, 2].max()) + pad),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "MotionGPT" / "staging" / "g2_top4")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "reports" / "figures" / "2026-05-26" / "g2_top_correction_smpl_mesh")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    faces_path = args.staging_dir / "_smpl_faces.npy"
    if not faces_path.exists():
        print(f"[ERROR] faces file missing: {faces_path}"); sys.exit(1)
    faces = np.load(str(faces_path)).astype(np.int32)
    print(f"[INFO] faces: {faces.shape}")

    manifest_path = args.staging_dir / "manifest.json"
    manifest = json.load(open(manifest_path, encoding="utf-8")) if manifest_path.exists() else {}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for tid in TOP4_SAMPLES:
        mesh_data = {}
        missing = False
        for variant in VARIANTS:
            mp = args.staging_dir / f"{tid}_{variant}_mesh.npy"
            if not mp.exists():
                print(f"[WARN] {tid}/{variant}: missing {mp.name}"); missing = True; break
            mesh_data[variant] = np.load(str(mp)).astype(np.float32)
        if missing:
            print(f"[SKIP] {tid}: incomplete mesh data")
            continue

        T_per = {k: v.shape[0] for k, v in mesh_data.items()}
        T_eff = min(T_per.values())
        for k in mesh_data:
            mesh_data[k] = mesh_data[k][:T_eff]
        print(f"[{tid}] T_eff={T_eff}, mesh shapes: {[(k, v.shape) for k, v in mesh_data.items()]}")

        xlim, ylim, zlim = _common_lims(list(mesh_data.values()))
        info = manifest.get(tid, {})
        prompt = info.get("g2_prompt", "")[:80]
        seq = info.get("oracle_sequence", [])
        seq_summary = " → ".join(f"{a[0].replace('Tool','')}/{a[2]}" for a in seq) if seq else "STOP"
        suptitle = f"{tid} — '{prompt}' — Oracle: {seq_summary}"
        panel_titles = [PANEL_LABELS[v] for v in VARIANTS]

        frames = []
        for idx in range(T_eff):
            frames.append(_render_frame(idx, mesh_data, faces, panel_titles,
                                          xlim, ylim, zlim, suptitle, T_eff))
            if (idx + 1) % 10 == 0:
                print(f"   rendered {idx + 1}/{T_eff}")
        dur = int(1000 / args.fps)
        durations = [dur] * T_eff
        durations[-1] = 800
        out = args.output_dir / f"{tid}_smpl_mesh_3panel.gif"
        frames[0].save(str(out), save_all=True, append_images=frames[1:],
                        duration=durations, loop=0, optimize=False)
        print(f"   [OK] saved: {out}\n")

    print(f"\n[OK] all SMPL mesh GIFs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
