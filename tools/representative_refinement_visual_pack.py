"""AR-051 visual pack for representative refinement effects.

Creates before/after side-by-side GIFs from the same tool application protocol
used by ``representative_refinement_effect.py``. The goal is qualitative audit:
one clear failure case and one improvement candidate, not a new metric.

CLI:
    python tools/representative_refinement_visual_pack.py \
        --output-dir reports/figures/2026-06-19/ar051_refinement_visual_pack
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external_assets" / "code"))
from plot_3d_motion import T2M_KINEMATIC_CHAIN, _valid_chain  # type: ignore

sys.path.insert(0, str(REPO_ROOT))
from correction_tools import BoneProjectionTool, FootLockTool, VelocitySmoothingTool
from tools.representative_refinement_effect import _apply_variant, _measure


def _load_json(path: Path) -> dict[str, Any]:
    return json.load(open(path, encoding="utf-8"))


def _tools() -> dict[str, Any]:
    return {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
        "BoneProjectionTool": BoneProjectionTool(),
    }


def _meta_path(pool_root: Path, generator: str, sample_id: str, seed: int) -> Path:
    return pool_root / generator / f"{sample_id}__seed{seed}.json"


def _load_motion_pair(pool_root: Path, generator: str, sample_id: str, seed: int, variant: str):
    meta = _load_json(_meta_path(pool_root, generator, sample_id, seed))
    traj = np.load(REPO_ROOT / meta["trajectory_npy"]).astype(np.float64)
    local = np.load(REPO_ROOT / meta["local_npy"]).astype(np.float64)
    out_traj, out_local, prov = _apply_variant(variant, traj, local, meta, _tools())
    before = _measure(traj, local)
    after = _measure(out_traj, out_local)
    delta = {k: float(after[k] - before[k]) for k in before}
    delta["fidelity_mpjpe"] = float(np.mean(np.linalg.norm(out_local - local, axis=-1)))
    return meta, traj, out_traj, delta, prov


def _find_improvement_candidate(pool_root: Path, generator: str, variant: str) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for meta_file in sorted((pool_root / generator).glob("*.json")):
        if meta_file.name.startswith("_"):
            continue
        meta = _load_json(meta_file)
        traj = np.load(REPO_ROOT / meta["trajectory_npy"]).astype(np.float64)
        local = np.load(REPO_ROOT / meta["local_npy"]).astype(np.float64)
        out_traj, out_local, _ = _apply_variant(variant, traj, local, meta, _tools())
        before = _measure(traj, local)
        after = _measure(out_traj, out_local)
        d_skate = float(after["foot_skate_world"] - before["foot_skate_world"])
        d_art = float(after["artifact_total"] - before["artifact_total"])
        fidelity = float(np.mean(np.linalg.norm(out_local - local, axis=-1)))
        if d_skate >= 0.0 or d_art > 0.0:
            continue
        score = d_skate - 0.2 * fidelity
        cand = {
            "generator": generator,
            "sample_id": meta["sample_id"],
            "seed": meta["seed"],
            "variant": variant,
            "delta_foot_skate_world": d_skate,
            "delta_artifact_total": d_art,
            "fidelity_mpjpe": fidelity,
            "score": score,
        }
        if best is None or cand["score"] < best["score"]:
            best = cand
    if best is None:
        raise RuntimeError(f"no improvement candidate found for {generator}/{variant}")
    return best


def _common_lims(motions: list[np.ndarray], pad: float = 0.25):
    cat = np.concatenate(motions, axis=0).reshape(-1, 3)
    cat = cat[~np.isnan(cat).any(axis=1)]
    return (
        (float(cat[:, 0].min()) - pad, float(cat[:, 0].max()) + pad),
        (float(cat[:, 1].min()) - pad, float(cat[:, 1].max()) + pad),
        (float(cat[:, 2].min()) - pad, float(cat[:, 2].max()) + pad),
    )


def _draw_floor(ax, xlim, ylim, zlim):
    verts = [
        [xlim[0], ylim[0], zlim[0]],
        [xlim[0], ylim[0], zlim[1]],
        [xlim[1], ylim[0], zlim[1]],
        [xlim[1], ylim[0], zlim[0]],
    ]
    plane = Poly3DCollection([verts])
    plane.set_facecolor((0.5, 0.5, 0.5, 0.18))
    ax.add_collection3d(plane)


def _draw_skeleton(ax, frame_xyz: np.ndarray, color: str):
    for chain in T2M_KINEMATIC_CHAIN:
        valid = _valid_chain(chain, frame_xyz)
        if len(valid) < 2:
            continue
        ax.plot3D(
            frame_xyz[valid, 0],
            frame_xyz[valid, 1],
            frame_xyz[valid, 2],
            linewidth=2.2,
            color=color,
        )
        ax.scatter(
            frame_xyz[valid, 0],
            frame_xyz[valid, 1],
            frame_xyz[valid, 2],
            c=color,
            s=12,
            depthshade=False,
        )


def _setup_axis(ax, xlim, ylim, zlim, title: str):
    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(*ylim)
    ax.set_zlim3d(*zlim)
    ax.view_init(elev=15.0, azim=-70.0, vertical_axis="y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title, fontsize=10)
    _draw_floor(ax, xlim, ylim, zlim)


def _render_frame(idx: int, before: np.ndarray, after: np.ndarray, title: str, xlim, ylim, zlim) -> Image.Image:
    fig = plt.figure(figsize=(11, 5))
    for pos, (motion, panel, color) in enumerate(
        ((before, "Before", "#2b7bbb"), (after, "After", "#ef8a17")),
        start=1,
    ):
        ax = fig.add_subplot(1, 2, pos, projection="3d")
        _setup_axis(ax, xlim, ylim, zlim, panel)
        _draw_skeleton(ax, motion[idx], color)
    fig.suptitle(f"{title}\nframe {idx + 1}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _save_case_gif(
    *,
    output: Path,
    before: np.ndarray,
    after: np.ndarray,
    title: str,
    frame_stride: int,
    max_frames: int,
    fps: int,
) -> None:
    n = min(before.shape[0], after.shape[0])
    before = before[:n:frame_stride].astype(np.float32)
    after = after[:n:frame_stride].astype(np.float32)
    if before.shape[0] > max_frames:
        idx = np.linspace(0, before.shape[0] - 1, max_frames).round().astype(int)
        before = before[idx]
        after = after[idx]
    xlim, ylim, zlim = _common_lims([before, after])
    frames = [_render_frame(i, before, after, title, xlim, ylim, zlim) for i in range(before.shape[0])]
    output.parent.mkdir(parents=True, exist_ok=True)
    dur = int(1000 / fps)
    durations = [dur] * len(frames)
    durations[-1] = 900
    frames[0].save(str(output), save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path, default=REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608")
    ap.add_argument("--effect-snapshot", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "representative_refinement_effect_v1.json")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "figures" / "2026-06-19" / "ar051_refinement_visual_pack")
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    snap = _load_json(args.effect_snapshot)
    failure = snap["examples"]["mdm"]["artifact_improved_foot_skate_worse"][0]
    improvement = _find_improvement_candidate(args.pool_root, "mdm", "velocity_smoothing_medium")

    manifest = {
        "board_id": "AR-051",
        "visual_claim_boundary": "Qualitative audit only. Uses same fixed tool protocols as AR-051 physical/proxy pass.",
        "cases": [],
    }

    cases = [
        ("failure_artifact_down_skate_up", failure),
        ("candidate_skate_and_artifact_down", improvement),
    ]
    for label, case in cases:
        gen = case.get("generator", "mdm")
        sample_id = case["sample_id"]
        seed = int(case["seed"])
        variant = case["variant"]
        meta, before, after, delta, prov = _load_motion_pair(args.pool_root, gen, sample_id, seed, variant)
        title = (
            f"{label} | {gen} {sample_id} seed {seed} | {variant} | "
            f"d_skate={delta['foot_skate_world']:+.4f}, d_art={delta['artifact_total']:+.4f}"
        )
        out = args.output_dir / f"{label}_{gen}_{sample_id}_seed{seed}_{variant}.gif"
        _save_case_gif(
            output=out,
            before=before,
            after=after,
            title=title,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            fps=args.fps,
        )
        try:
            gif_rel = str(out.resolve().relative_to(REPO_ROOT))
        except ValueError:
            gif_rel = str(out)
        manifest["cases"].append({
            "label": label,
            "generator": gen,
            "sample_id": sample_id,
            "seed": seed,
            "variant": variant,
            "prompt": meta["prompt"],
            "gif": gif_rel,
            "delta": {k: round(float(v), 6) for k, v in delta.items()},
            "provenance": prov,
        })
        print(f"[OK] {label}: {out}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
