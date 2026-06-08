"""Build a deterministic plain-text prompt file from HumanML3D captions.

This is used for larger G2 (MotionGPT) natural-distribution batches when the
official MotionGPT demo prompt file is too small.

Example:
    python -m tools.build_humanml3d_prompt_file \
        --split test \
        --n-prompts 300 \
        --seed 20260527 \
        --output evals/prompts/humanml3d_test_300_seed20260527.txt
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMANML3D_ROOT = REPO_ROOT / "external_assets" / "HumanML3D"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_split_ids(root: Path, split: str) -> list[str]:
    split_path = root / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    with open(split_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _first_caption(root: Path, sample_id: str) -> str | None:
    text_path = root / "texts" / f"{sample_id}.txt"
    if not text_path.exists():
        return None
    with open(text_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            caption = line.split("#", 1)[0].strip()
            if caption:
                return caption
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HumanML3D prompt file")
    parser.add_argument("--humanml3d-root", type=Path, default=DEFAULT_HUMANML3D_ROOT)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--n-prompts", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    args = parser.parse_args()

    ids = _load_split_ids(args.humanml3d_root, args.split)
    rng = random.Random(args.seed)
    rng.shuffle(ids)

    rows: list[dict[str, str]] = []
    for sample_id in ids:
        caption = _first_caption(args.humanml3d_root, sample_id)
        if caption is None:
            continue
        rows.append({"sample_id": sample_id, "prompt": caption})
        if len(rows) >= args.n_prompts:
            break

    if len(rows) < args.n_prompts:
        raise RuntimeError(f"Only found {len(rows)} valid prompts; requested {args.n_prompts}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(row["prompt"] for row in rows) + "\n", encoding="utf-8")

    metadata_output = args.metadata_output or args.output.with_suffix(".json")
    metadata = {
        "created_at_utc": _now_iso(),
        "humanml3d_root": str(args.humanml3d_root),
        "split": args.split,
        "n_prompts": len(rows),
        "seed": args.seed,
        "output": str(args.output),
        "rows": rows,
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote {len(rows)} prompts: {args.output}")
    print(f"[OK] metadata: {metadata_output}")


if __name__ == "__main__":
    main()
