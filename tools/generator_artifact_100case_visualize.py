"""AR-055: render all AR-054 generator artifact cases as compact GIFs.

The output is a qualitative visualization pack, not a generator ranking or
refinement-performance claim.

CLI:
    python tools/generator_artifact_100case_visualize.py
"""
from __future__ import annotations

import argparse
import html
import io
import json
import re
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
from tools.coords_protocol import LEFT_FOOT, RIGHT_FOOT


CONTACT_H = 0.05


def _load_json(path: Path) -> dict[str, Any]:
    return json.load(open(path, encoding="utf-8"))


def _slug(text: str, max_len: int = 48) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "case"


def _common_lims(motion: np.ndarray, pad: float = 0.35):
    cat = motion.reshape(-1, 3)
    cat = cat[~np.isnan(cat).any(axis=1)]
    return (
        (float(cat[:, 0].min()) - pad, float(cat[:, 0].max()) + pad),
        (float(cat[:, 1].min()) - pad, float(cat[:, 1].max()) + pad),
        (float(cat[:, 2].min()) - pad, float(cat[:, 2].max()) + pad),
    )


def _draw_floor(ax, xlim, zlim, ground_y: float):
    verts = [
        [xlim[0], ground_y, zlim[0]],
        [xlim[0], ground_y, zlim[1]],
        [xlim[1], ground_y, zlim[1]],
        [xlim[1], ground_y, zlim[0]],
    ]
    plane = Poly3DCollection([verts])
    plane.set_facecolor((0.45, 0.45, 0.45, 0.16))
    ax.add_collection3d(plane)


def _draw_skeleton(ax, frame_xyz: np.ndarray, color: str = "#2b7bbb"):
    for chain in T2M_KINEMATIC_CHAIN:
        valid = _valid_chain(chain, frame_xyz)
        if len(valid) < 2:
            continue
        ax.plot3D(frame_xyz[valid, 0], frame_xyz[valid, 1], frame_xyz[valid, 2], linewidth=1.8, color=color)
        ax.scatter(frame_xyz[valid, 0], frame_xyz[valid, 1], frame_xyz[valid, 2], c=color, s=8, depthshade=False)


def _setup_axis(ax, xlim, ylim, zlim, ground_y: float):
    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(min(ylim[0], ground_y - 0.08), max(ylim[1], ground_y + 1.35))
    ax.set_zlim3d(*zlim)
    ax.view_init(elev=15.0, azim=-70.0, vertical_axis="y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    _draw_floor(ax, xlim, zlim, ground_y)


def _artifact_mode(bucket: str) -> str:
    if "float" in bucket or "contact" in bucket or "length" in bucket:
        return "footfloating"
    return "footskate"


def _render_frame(
    idx: int,
    motion: np.ndarray,
    title: str,
    xlim,
    ylim,
    zlim,
    ground_y: float,
    artifact_mode: str,
) -> Image.Image:
    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _setup_axis(ax, xlim, ylim, zlim, ground_y)
    frame = motion[idx]
    _draw_skeleton(ax, frame)

    feet = frame[[LEFT_FOOT, RIGHT_FOOT]]
    heights = feet[:, 1] - ground_y
    colors = []
    for h in heights:
        if artifact_mode == "footfloating":
            colors.append("#d62728" if h > CONTACT_H else "#2ca02c")
        else:
            colors.append("#d62728" if h <= CONTACT_H else "#2ca02c")
    ax.scatter(
        feet[:, 0],
        feet[:, 1],
        feet[:, 2],
        c=colors,
        s=52,
        depthshade=False,
        edgecolors="black",
        linewidths=0.7,
    )

    start = max(0, idx - 18)
    for j, c in ((LEFT_FOOT, "#ff7f0e"), (RIGHT_FOOT, "#9467bd")):
        trail = motion[start : idx + 1, j, :]
        ax.plot3D(
            trail[:, 0],
            np.full(len(trail), ground_y + 0.008),
            trail[:, 2],
            color=c,
            linewidth=1.7,
            alpha=0.72,
        )

    ax.set_title(f"{title}\nframe {idx + 1} | foot h={heights[0]:+.3f}/{heights[1]:+.3f}m", fontsize=7)
    fig.tight_layout(pad=0.25)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE, colors=128)


def _save_gif(
    *,
    motion: np.ndarray,
    title: str,
    artifact_bucket: str,
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
    if motion.shape[0] == 0:
        raise ValueError("empty motion after stride")
    xlim, ylim, zlim = _common_lims(motion)
    mode = _artifact_mode(artifact_bucket)
    frames = [_render_frame(i, motion, title, xlim, ylim, zlim, ground_y, mode) for i in range(motion.shape[0])]
    output.parent.mkdir(parents=True, exist_ok=True)
    dur = int(1000 / fps)
    durations = [dur] * len(frames)
    durations[-1] = max(700, dur)
    frames[0].save(str(output), save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def _gif_check(path: Path) -> dict[str, Any]:
    im = Image.open(path)
    extrema = []
    for i, frame in enumerate(ImageSequence.Iterator(im)):
        if i >= 3:
            break
        extrema.append(frame.convert("L").getextrema())
    return {
        "size": list(im.size),
        "n_frames": int(getattr(im, "n_frames", 0)),
        "bytes": int(path.stat().st_size),
        "nonblank": any(lo != hi for lo, hi in extrema),
    }


def _load_cases(input_dir: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for gen in ("motiongpt", "mdm", "momask"):
        manifest = _load_json(input_dir / f"artifact_100_manifest_{gen}.json")
        for case in manifest["cases"]:
            cases.append(case)
    return cases


def _write_index_html(output_dir: Path, rendered: list[dict[str, Any]]) -> None:
    by_gen: dict[str, list[dict[str, Any]]] = {}
    for r in rendered:
        by_gen.setdefault(r["generator"], []).append(r)

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>AR-055 Generator Artifact GIF Pack</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f7;color:#222}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}"
        ".card{background:white;border:1px solid #ddd;border-radius:8px;padding:10px}"
        "img{width:100%;height:auto;border:1px solid #ddd}.cap{font-size:12px;line-height:1.35}"
        "code{font-size:11px}</style>",
        "<h1>AR-055 Generator Artifact GIF Pack</h1>",
        "<p>Qualitative artifact visualization only. Not generator ranking or refinement-performance evidence.</p>",
    ]
    for gen in ("motiongpt", "mdm", "momask"):
        rows = by_gen.get(gen, [])
        parts.append(f"<h2>{html.escape(gen)} ({len(rows)})</h2><div class='grid'>")
        for r in rows:
            rel = Path(r["gif"]).relative_to(output_dir)
            prompt = html.escape(r["prompt"])
            parts.append(
                "<div class='card'>"
                f"<img src='{html.escape(rel.as_posix())}' loading='lazy'>"
                f"<div class='cap'><b>#{r['selection_index']:03d}</b> "
                f"{html.escape(r['artifact_bucket'])} / {html.escape(r['motion_group'])}<br>"
                f"<code>{html.escape(r['sample_id'])} seed {r['seed']}</code><br>{prompt}</div>"
                "</div>"
            )
        parts.append("</div>")
    (output_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def _write_index_md(output_dir: Path, rendered: list[dict[str, Any]]) -> None:
    lines = [
        "# AR-055 Generator Artifact 100-Case GIF Export",
        "",
        "이 폴더는 AR-054 artifact 100-case library 전체를 GIF로 시각화한 결과다.",
        "",
        "Claim boundary: 정성 시각화 자료이며, generator ranking이나 refinement 성능 claim이 아니다.",
        "",
        "| generator | n | total MB |",
        "|---|---:|---:|",
    ]
    for gen in ("motiongpt", "mdm", "momask"):
        rows = [r for r in rendered if r["generator"] == gen]
        mb = sum(r["gif_check"]["bytes"] for r in rows) / (1024 * 1024)
        lines.append(f"| {gen} | {len(rows)} | {mb:.1f} |")
    lines += [
        "",
        "브라우저에서 전체를 보려면 `index.html`을 여세요.",
        "",
        "폴더 구조:",
        "",
        "- `motiongpt/*.gif`",
        "- `mdm/*.gif`",
        "- `momask/*.gif`",
    ]
    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "figures" / "2026-06-19" / "ar054_generator_artifact_100cases",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "figures" / "2026-06-21" / "ar055_generator_artifact_100case_gifs",
    )
    ap.add_argument(
        "--snapshot",
        type=Path,
        default=REPO_ROOT / "evals" / "snapshots" / "generator_artifact_100case_gif_export_v1.json",
    )
    ap.add_argument("--frame-stride", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=36)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--limit-per-generator", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cases = _load_cases(args.input_dir)
    if args.limit_per_generator is not None:
        limited = []
        for gen in ("motiongpt", "mdm", "momask"):
            limited.extend([c for c in cases if c["generator"] == gen][: args.limit_per_generator])
        cases = limited

    rendered: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        gen = case["generator"]
        bucket = case["artifact_bucket"]
        group = case["motion_group"]
        filename = (
            f"{case['selection_index']:03d}_{_slug(bucket, 28)}_{_slug(group, 24)}_"
            f"{case['sample_id']}_seed{case['seed']}.gif"
        )
        out = args.output_dir / gen / filename
        try:
            if not out.exists() or args.overwrite:
                motion = np.load(REPO_ROOT / case["trajectory_npy"]).astype(np.float64)
                title = (
                    f"{gen} | #{case['selection_index']:03d} | {bucket} | {group}\n"
                    f"FF={case['metrics']['footfloating_score']:.3f}, "
                    f"skate={case['metrics']['foot_skate_world']:.4f}, "
                    f"len={case['metrics']['length_ratio']:.2f}"
                )
                _save_gif(
                    motion=motion,
                    title=title,
                    artifact_bucket=bucket,
                    ground_y=float(case.get("ground_y", 0.0)),
                    output=out,
                    frame_stride=args.frame_stride,
                    max_frames=args.max_frames,
                    fps=args.fps,
                )
            check = _gif_check(out)
            record = {
                "generator": gen,
                "selection_index": int(case["selection_index"]),
                "sample_id": case["sample_id"],
                "seed": int(case["seed"]),
                "prompt": case["prompt"],
                "artifact_bucket": bucket,
                "motion_group": group,
                "metrics": case["metrics"],
                "gif": str(out.resolve()),
                "gif_rel": str(out.resolve().relative_to(REPO_ROOT)),
                "gif_check": check,
            }
            rendered.append(record)
            if idx % 10 == 0 or idx == len(cases):
                print(f"[{idx:03d}/{len(cases):03d}] rendered {gen}/{filename} bytes={check['bytes']}")
        except Exception as exc:  # keep long batch resumable
            errors.append({
                "generator": gen,
                "selection_index": case.get("selection_index"),
                "sample_id": case.get("sample_id"),
                "seed": case.get("seed"),
                "error": repr(exc),
            })
            print(f"[ERR] {gen} #{case.get('selection_index')} {case.get('sample_id')}: {exc}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_index_html(args.output_dir, rendered)
    _write_index_md(args.output_dir, rendered)
    manifest = {
        "schema_version": "1.0.0",
        "record_type": "generator_artifact_100case_gif_export",
        "board_id": "AR-055",
        "input_dir": str(args.input_dir.resolve().relative_to(REPO_ROOT)),
        "output_dir": str(args.output_dir.resolve().relative_to(REPO_ROOT)),
        "claim_boundary": "Qualitative artifact visualization only. Not generator ranking, prevalence estimate, or refinement-performance evidence.",
        "render_profile": {
            "frame_stride": args.frame_stride,
            "max_frames": args.max_frames,
            "fps": args.fps,
            "figsize": [5.2, 4.4],
            "dpi": 80,
            "palette_colors": 128,
        },
        "n_requested": len(cases),
        "n_rendered": len(rendered),
        "n_errors": len(errors),
        "total_bytes": int(sum(r["gif_check"]["bytes"] for r in rendered)),
        "per_generator_counts": {
            gen: sum(1 for r in rendered if r["generator"] == gen)
            for gen in ("motiongpt", "mdm", "momask")
        },
        "per_generator_bytes": {
            gen: int(sum(r["gif_check"]["bytes"] for r in rendered if r["generator"] == gen))
            for gen in ("motiongpt", "mdm", "momask")
        },
        "errors": errors,
        "cases": rendered,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] rendered={len(rendered)} errors={len(errors)}")
    print(f"[OK] output={args.output_dir}")
    print(f"[OK] snapshot={args.snapshot}")


if __name__ == "__main__":
    main()
