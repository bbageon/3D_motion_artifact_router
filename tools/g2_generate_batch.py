"""G2 (MotionGPT) batch inference — natural artifact 분포 측정용 motion batch.

본 도구는 [`external_assets/MotionGPT/demos/t2m.txt`](../external_assets/MotionGPT/demos/t2m.txt)
의 50 detailed prompt (60+ 단어급, MotionGPT 의 length control issue 회피) 를 사용해
N motion 을 batch 생성. 각 motion 은 npy + per-sample metadata (prompt, generator_id,
checkpoint hash, length_generated 등) 로 저장.

저장 구조:
    <output_dir>/
      motion_001.npy
      motion_001.json   # metadata
      motion_002.npy
      ...
      _batch_summary.json   # 전체 batch metadata

AGENTS.md 의무:
  - §3-5 generator quality-tier 분리: generator_id 에 "G2_motiongpt" 포함.
  - §3-6 평가 기록: 모든 npy 와 짝 짓는 json metadata.
  - §5-2 stochasticity: per-sample seed 고정 (base_seed + index).

CLI 예:
    python -m tools.g2_generate_batch \\
        --n-samples 30 \\
        --n-frames 40 \\
        --base-seed 42 \\
        --output-dir external_assets/g2_generated_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from generators.motiongpt_wrapper import MotionGPT_G2

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_FILE = REPO_ROOT / "external_assets" / "MotionGPT" / "demos" / "t2m.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "external_assets" / "g2_generated_v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _load_prompts(prompt_file: Path) -> list[str]:
    with open(prompt_file, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 (MotionGPT) batch generation")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--n-samples", type=int, default=30,
                        help="생성할 motion 수 (<= prompt 수).")
    parser.add_argument("--n-frames", type=int, default=40,
                        help="n_frames hint (MotionGPT 가 ignore 가능).")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="per-sample seed = base_seed + index.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-existing", action="store_true",
                        help="이미 .npy 가 있으면 skip (resume).")
    args = parser.parse_args()

    prompts = _load_prompts(args.prompt_file)
    if len(prompts) < args.n_samples:
        raise ValueError(
            f"prompt file has only {len(prompts)} prompts but --n-samples={args.n_samples}"
        )
    prompts = prompts[: args.n_samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gen = MotionGPT_G2()
    generator_class_hash = gen._generator_class_hash()
    print(f"[INFO] generator_id base: G2_motiongpt_{gen.checkpoint_path.stem}")
    print(f"[INFO] generator_class_hash: {generator_class_hash}")
    print(f"[INFO] n_prompts: {len(prompts)}, n_frames hint: {args.n_frames}, base_seed: {args.base_seed}")
    print(f"[INFO] output_dir: {args.output_dir}")

    batch_meta = {
        "batch_id": f"g2_generated_v1_{_now_iso()}",
        "generator_id": f"G2_motiongpt_{gen.checkpoint_path.stem}",
        "generator_class_hash": generator_class_hash,
        "prompt_file": str(args.prompt_file),
        "n_samples": args.n_samples,
        "n_frames_hint": args.n_frames,
        "base_seed": args.base_seed,
        "output_dir": str(args.output_dir),
        "checkpoint_path": str(gen.checkpoint_path),
        "samples": [],
    }

    n_success = 0
    n_failed = 0
    for i, prompt in enumerate(prompts):
        idx = i + 1
        out_npy = args.output_dir / f"motion_{idx:03d}.npy"
        out_meta = args.output_dir / f"motion_{idx:03d}.json"
        if args.skip_existing and out_npy.exists() and out_meta.exists():
            print(f"[SKIP] {out_npy.name} already exists")
            with open(out_meta, encoding="utf-8") as f:
                existing_meta = json.load(f)
            batch_meta["samples"].append(existing_meta)
            n_success += 1
            continue
        seed = args.base_seed + i
        print(f"[GEN {idx:3d}/{len(prompts)}] seed={seed}, prompt[:60]={prompt[:60]!r}")
        try:
            out = gen.generate(prompt=prompt, n_frames=args.n_frames, seed=seed)
        except Exception as e:
            print(f"  [FAIL] {e}", file=sys.stderr)
            n_failed += 1
            continue

        np.save(out_npy, out.motion.astype(np.float32))
        sample_meta = {
            "sample_index": idx,
            "trial_id": f"motion_{idx:03d}",
            "prompt": prompt,
            "seed": seed,
            "n_frames_hint": args.n_frames,
            "motion_shape": list(out.motion.shape),
            "fps": out.fps,
            "generator_id": out.generator_id,
            "generator_class_hash": out.generator_class_hash,
            "metadata": out.metadata,
            "npy_path": str(out_npy),
        }
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(sample_meta, f, indent=2, ensure_ascii=False)
        batch_meta["samples"].append(sample_meta)
        n_success += 1
        print(f"  [OK] shape={out.motion.shape}, length_generated={out.metadata.get('length_generated', '?')}")

    summary_path = args.output_dir / "_batch_summary.json"
    batch_meta["n_success"] = n_success
    batch_meta["n_failed"] = n_failed
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_meta, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] {n_success} succeeded, {n_failed} failed.")
    print(f"[OK] batch summary: {summary_path}")


if __name__ == "__main__":
    main()
