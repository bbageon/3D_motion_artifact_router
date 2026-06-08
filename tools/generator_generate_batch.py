"""Multi-generator batch inference for ArtifactRouter.

Creates a canonical motion pool from a shared prompt file. This generalizes the
older MotionGPT-only batch runner so MDM and MoMask can be profiled with the
same prompt bank.

Examples:
    python -m tools.generator_generate_batch \
        --generator mdm \
        --prompt-file evals/prompts/humanml3d_test_300_seed20260527.txt \
        --n-samples 50 \
        --n-frames 40 \
        --base-seed 20260603 \
        --conda-env momask \
        --output-dir external_assets/mdm_generated_hml3d_test50_seed20260603

    python -m tools.generator_generate_batch \
        --generator momask \
        --prompt-file evals/prompts/humanml3d_test_300_seed20260527.txt \
        --n-samples 50 \
        --n-frames 40 \
        --base-seed 20260603 \
        --conda-env momask \
        --gpu-id -1 \
        --output-dir external_assets/momask_generated_hml3d_test50_seed20260603
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from generators.motiongpt_wrapper import MotionGPT_G2
from generators.mdm_wrapper import MDM_G1
from generators.momask_wrapper import MoMask_G2

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_FILE = REPO_ROOT / "evals" / "prompts" / "humanml3d_test_300_seed20260527.txt"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _load_prompts(prompt_file: Path) -> list[str]:
    with open(prompt_file, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _build_generator(args: argparse.Namespace):
    bk = args.backend
    su = args.service_url
    if args.generator == "motiongpt":
        return MotionGPT_G2(backend=bk, service_url=su)
    if args.generator == "mdm":
        return MDM_G1(conda_env=args.conda_env, device=args.device, backend=bk, service_url=su)
    if args.generator == "momask":
        return MoMask_G2(conda_env=args.conda_env, gpu_id=args.gpu_id, backend=bk, service_url=su)
    raise ValueError(f"Unsupported generator: {args.generator}")


def _root_abs_max(motion: np.ndarray) -> float:
    if motion.ndim != 3 or motion.shape[1] < 1:
        return float("nan")
    return float(np.abs(motion[:, 0, :]).max())


def _sample_metadata(
    *,
    idx: int,
    prompt: str,
    seed: int,
    n_frames: int,
    out: Any,
    out_npy: Path,
) -> dict[str, Any]:
    return {
        "sample_index": idx,
        "trial_id": f"motion_{idx:03d}",
        "prompt": prompt,
        "seed": seed,
        "n_frames_hint": n_frames,
        "motion_shape": list(out.motion.shape),
        "fps": out.fps,
        "root_abs_max": _root_abs_max(out.motion),
        "generator_id": out.generator_id,
        "generator_class_hash": out.generator_class_hash,
        "metadata": out.metadata,
        "npy_path": str(out_npy),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate a canonical motion pool.")
    parser.add_argument("--generator", choices=["motiongpt", "mdm", "momask"], required=True)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-frames", type=int, default=40)
    parser.add_argument("--base-seed", type=int, default=20260603)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--conda-env", type=str, default="momask",
                        help="External env for MDM/MoMask (conda backend). Ignored for MotionGPT/http.")
    parser.add_argument("--device", type=int, default=0, help="MDM device id (conda backend).")
    parser.add_argument("--gpu-id", type=int, default=-1, help="MoMask gpu id (conda backend); -1 for CPU.")
    parser.add_argument("--backend", choices=["conda", "http"], default=None,
                        help="conda (default) | http (docker GPU microservice).")
    parser.add_argument("--service-url", type=str, default=None,
                        help="http backend base URL (default localhost:800x per generator).")
    args = parser.parse_args()

    prompts = _load_prompts(args.prompt_file)
    if len(prompts) < args.n_samples:
        raise ValueError(
            f"prompt file has only {len(prompts)} prompts but --n-samples={args.n_samples}"
        )
    prompts = prompts[: args.n_samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gen = _build_generator(args)
    batch_meta: dict[str, Any] = {
        "batch_id": f"{args.generator}_generated_{_now_iso()}",
        "generator": args.generator,
        "backend": args.backend or "conda",
        "service_url": args.service_url,
        "prompt_file": str(args.prompt_file),
        "n_samples": args.n_samples,
        "n_frames_hint": args.n_frames,
        "base_seed": args.base_seed,
        "output_dir": str(args.output_dir),
        "skip_existing": bool(args.skip_existing),
        "samples": [],
    }

    print(f"[INFO] generator={args.generator}")
    print(f"[INFO] prompt_file={args.prompt_file}")
    print(f"[INFO] n_samples={args.n_samples}, n_frames={args.n_frames}, base_seed={args.base_seed}")
    print(f"[INFO] output_dir={args.output_dir}")

    n_success = 0
    n_failed = 0
    for i, prompt in enumerate(prompts):
        idx = i + 1
        seed = args.base_seed + i
        out_npy = args.output_dir / f"motion_{idx:03d}.npy"
        out_meta = args.output_dir / f"motion_{idx:03d}.json"
        if args.skip_existing and out_npy.exists() and out_meta.exists():
            print(f"[SKIP {idx:3d}/{len(prompts)}] {out_npy.name}")
            with open(out_meta, encoding="utf-8") as f:
                batch_meta["samples"].append(json.load(f))
            n_success += 1
            continue

        print(f"[GEN {idx:3d}/{len(prompts)}] seed={seed}, prompt[:70]={prompt[:70]!r}")
        try:
            out = gen.generate(prompt=prompt, n_frames=args.n_frames, seed=seed)
        except Exception as exc:
            print(f"  [FAIL] {exc}", file=sys.stderr)
            n_failed += 1
            continue

        np.save(out_npy, out.motion.astype(np.float32))
        meta = _sample_metadata(
            idx=idx,
            prompt=prompt,
            seed=seed,
            n_frames=args.n_frames,
            out=out,
            out_npy=out_npy,
        )
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        batch_meta["samples"].append(meta)
        n_success += 1
        print(f"  [OK] shape={out.motion.shape}, root_abs_max={meta['root_abs_max']:.3g}")

    summary_path = args.output_dir / "_batch_summary.json"
    batch_meta["n_success"] = n_success
    batch_meta["n_failed"] = n_failed
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_meta, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] {n_success} succeeded, {n_failed} failed.")
    print(f"[OK] batch summary: {summary_path}")


if __name__ == "__main__":
    main()
