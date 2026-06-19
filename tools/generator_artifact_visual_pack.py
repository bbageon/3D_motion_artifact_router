"""AR-053: visual pack for generator-specific artifacts.

Selects representative artifact cases from the representative-300 pool and
renders Y-up skeleton GIFs:

  - MotionGPT: FootFloating
  - MDM: world foot-skate
  - MoMask: FootFloating

The GIFs are qualitative audit evidence only. They do not modify generator
outputs and do not claim refinement performance.

CLI:
    python tools/generator_artifact_visual_pack.py \
        --output-dir reports/figures/2026-06-19/ar053_generator_artifacts
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
from PIL import Image, ImageSequence
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external_assets" / "code"))
from plot_3d_motion import T2M_KINEMATIC_CHAIN, _valid_chain  # type: ignore

sys.path.insert(0, str(REPO_ROOT))
from evaluators import DEFAULT_EVALUATORS
from tools.coords_protocol import LEFT_FOOT, RIGHT_FOOT, estimate_ground
from tools.representative_pool_measure import foot_skate_world


FOOT_FLOAT_THRESH = 0.05
CONTACT_H = 0.05


def _load_json(path: Path) -> dict[str, Any]:
    return json.load(open(path, encoding="utf-8"))


def _max_score(reports) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _footfloating_score(local: np.ndarray) -> float:
    ev = next(ev for ev in DEFAULT_EVALUATORS if ev.name == "FootFloatingEvaluator")
    return _max_score(ev.evaluate(local))


def _iter_pool(pool_root: Path, generator: str):
    for meta_path in sorted((pool_root / generator).glob("*.json")):
        if meta_path.name.startswith("_"):
            continue
        meta = _load_json(meta_path)
        traj = np.load(REPO_ROOT / meta["trajectory_npy"]).astype(np.float64)
        local = np.load(REPO_ROOT / meta["local_npy"]).astype(np.float64)
        yield meta, traj, local


def _select_cases(pool_root: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    specs = {
        "motiongpt_footfloating": ("motiongpt", "footfloating"),
        "mdm_world_footskate": ("mdm", "footskate"),
        "momask_footfloating": ("momask", "footfloating"),
    }
    for label, (gen, metric) in specs.items():
        best: dict[str, Any] | None = None
        for meta, traj, local in _iter_pool(pool_root, gen):
            ff = _footfloating_score(local)
            skate = foot_skate_world(traj)
            score = ff if metric == "footfloating" else skate
            cand = {
                "label": label,
                "generator": gen,
                "artifact": metric,
                "sample_id": meta["sample_id"],
                "seed": int(meta["seed"]),
                "prompt": meta["prompt"],
                "target_length": int(meta["target_length"]),
                "actual_generated": int(meta["actual_generated"]),
                "footfloating_score": round(float(ff), 6),
                "foot_skate_world": round(float(skate), 6),
                "ground_y": round(float(meta.get("ground_y", estimate_ground(traj))), 6),
                "trajectory_npy": meta["trajectory_npy"],
                "local_npy": meta["local_npy"],
                "selection_score": round(float(score), 6),
            }
            if best is None or cand["selection_score"] > best["selection_score"]:
                best = cand
        if best is None:
            raise RuntimeError(f"no case selected for {label}")
        cases[label] = best
    return cases


def _common_lims(motion: np.ndarray, pad: float = 0.35):
    cat = motion.reshape(-1, 3)
    cat = cat[~np.isnan(cat).any(axis=1)]
    return (
        (float(cat[:, 0].min()) - pad, float(cat[:, 0].max()) + pad),
        (float(cat[:, 1].min()) - pad, float(cat[:, 1].max()) + pad),
        (float(cat[:, 2].min()) - pad, float(cat[:, 2].max()) + pad),
    )


def _draw_floor(ax, xlim, ylim, zlim, ground_y: float):
    verts = [
        [xlim[0], ground_y, zlim[0]],
        [xlim[0], ground_y, zlim[1]],
        [xlim[1], ground_y, zlim[1]],
        [xlim[1], ground_y, zlim[0]],
    ]
    plane = Poly3DCollection([verts])
    plane.set_facecolor((0.45, 0.45, 0.45, 0.18))
    ax.add_collection3d(plane)


def _draw_skeleton(ax, frame_xyz: np.ndarray, color: str = "#2b7bbb"):
    for chain in T2M_KINEMATIC_CHAIN:
        valid = _valid_chain(chain, frame_xyz)
        if len(valid) < 2:
            continue
        ax.plot3D(
            frame_xyz[valid, 0],
            frame_xyz[valid, 1],
            frame_xyz[valid, 2],
            linewidth=2.1,
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


def _setup_axis(ax, xlim, ylim, zlim, ground_y: float):
    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(min(ylim[0], ground_y - 0.1), max(ylim[1], ground_y + 1.4))
    ax.set_zlim3d(*zlim)
    ax.view_init(elev=15.0, azim=-70.0, vertical_axis="y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    _draw_floor(ax, xlim, ylim, zlim, ground_y)


def _render_frame(
    idx: int,
    motion: np.ndarray,
    title: str,
    xlim,
    ylim,
    zlim,
    ground_y: float,
    artifact: str,
) -> Image.Image:
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _setup_axis(ax, xlim, ylim, zlim, ground_y)
    frame = motion[idx]
    _draw_skeleton(ax, frame)

    feet = frame[[LEFT_FOOT, RIGHT_FOOT]]
    heights = feet[:, 1] - ground_y
    colors = []
    for h in heights:
        if artifact == "footfloating":
            colors.append("#d62728" if h > CONTACT_H else "#2ca02c")
        else:
            colors.append("#d62728" if h <= CONTACT_H else "#2ca02c")
    ax.scatter(
        feet[:, 0],
        feet[:, 1],
        feet[:, 2],
        c=colors,
        s=70,
        depthshade=False,
        edgecolors="black",
        linewidths=0.8,
    )

    start = max(0, idx - 25)
    for j, c in ((LEFT_FOOT, "#ff7f0e"), (RIGHT_FOOT, "#9467bd")):
        trail = motion[start : idx + 1, j, :]
        ax.plot3D(trail[:, 0], np.full(len(trail), ground_y + 0.01), trail[:, 2], color=c, linewidth=2.0, alpha=0.75)

    ax.set_title(f"{title}\nframe {idx + 1} | foot h={heights[0]:+.3f}/{heights[1]:+.3f}m", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _save_gif(
    *,
    motion: np.ndarray,
    title: str,
    artifact: str,
    ground_y: float,
    output: Path,
    frame_stride: int,
    max_frames: int,
    fps: int,
) -> None:
    motion = motion[::frame_stride].astype(np.float32)
    if motion.shape[0] > max_frames:
        idx = np.linspace(0, motion.shape[0] - 1, max_frames).round().astype(int)
        motion = motion[idx]
    xlim, ylim, zlim = _common_lims(motion)
    frames = [
        _render_frame(i, motion, title, xlim, ylim, zlim, ground_y, artifact)
        for i in range(motion.shape[0])
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    dur = int(1000 / fps)
    durations = [dur] * len(frames)
    durations[-1] = 900
    frames[0].save(str(output), save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)


def _gif_nonblank(path: Path) -> dict[str, Any]:
    im = Image.open(path)
    extrema = []
    for i, frame in enumerate(ImageSequence.Iterator(im)):
        if i >= 3:
            break
        extrema.append(frame.convert("L").getextrema())
    return {
        "size": list(im.size),
        "n_frames": getattr(im, "n_frames", None),
        "first_frame_extrema": extrema,
        "nonblank": any(lo != hi for lo, hi in extrema),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path, default=REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "figures" / "2026-06-19" / "ar053_generator_artifacts")
    ap.add_argument("--snapshot", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "generator_artifact_visual_pack_v1.json")
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=90)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    cases = _select_cases(args.pool_root)
    manifest = {
        "schema_version": "1.0.0",
        "record_type": "generator_artifact_visual_pack",
        "board_id": "AR-053",
        "pool": args.pool_root.name,
        "claim_boundary": "Qualitative artifact visualization only. Not a generator ranking or refinement-performance claim.",
        "case_selection": {
            "motiongpt_footfloating": "max FootFloatingEvaluator score",
            "mdm_world_footskate": "max foot_skate_world",
            "momask_footfloating": "max FootFloatingEvaluator score",
        },
        "cases": [],
    }

    for label, case in cases.items():
        motion = np.load(REPO_ROOT / case["trajectory_npy"]).astype(np.float64)
        title = (
            f"{case['generator']} | {case['artifact']} | {case['sample_id']} seed {case['seed']}\n"
            f"FF={case['footfloating_score']:.3f}, skate={case['foot_skate_world']:.4f} | {case['prompt'][:80]}"
        )
        out = args.output_dir / f"{label}_{case['sample_id']}_seed{case['seed']}.gif"
        _save_gif(
            motion=motion,
            title=title,
            artifact=case["artifact"],
            ground_y=float(case["ground_y"]),
            output=out,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            fps=args.fps,
        )
        check = _gif_nonblank(out)
        case_out = {
            **case,
            "gif": str(out.resolve().relative_to(REPO_ROOT)),
            "gif_check": check,
        }
        manifest["cases"].append(case_out)
        print(f"[OK] {label}: {out} nonblank={check['nonblank']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] snapshot: {args.snapshot}")


if __name__ == "__main__":
    main()
