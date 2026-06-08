"""AR-048: corrected cross-generator prompt bank builder.

Fixes (AR-048 spec A):
  1. HumanML3D test split.
  2. full-motion caption만 (texts/<id>.txt 의 frame-range start==0.0 & end==0.0).
  3. 원본·미러 중복 제거 (M-prefix=좌우반전 → non-M 만 유지).
  4. GT 파일 존재 확인 (new_joints/<id>.npy).
  5. 동일 문장 중복 제거 (정규화 후).
  6. GT 길이 filter [40,200) + target_length = min(4*floor(GT_len/4), 196).
  7. sample_id/prompt/gt_length/target_length/gt_path 기록.

통계 단위 = prompt (seed 는 generation 반복). 본 bank 는 generator 무관 — 같은 prompt 를
3 generator 에 동일 target_length 로 넣어 paired 비교(§6-10).

CLI (motion3d env):
    python tools/build_protocol_prompt_bank.py --n-prompts 30 --seed 20260607 \
        --output evals/prompts/protocol_v1_test_30_seed20260607.txt
"""
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HML3D = REPO_ROOT / "external_assets" / "HumanML3D"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_caption(c: str) -> str:
    """동일 문장 dedup 용 정규화: 소문자 + 공백 정리 + 끝 구두점 제거."""
    return re.sub(r"\s+", " ", c.strip().lower()).rstrip(". ")


def _first_full_motion_caption(texts_dir: Path, sid: str) -> str | None:
    """texts/<id>.txt 에서 frame-range 가 전체(0.0/0.0)인 첫 캡션. 부분 캡션 제외."""
    p = texts_dir / f"{sid}.txt"
    if not p.exists():
        return None
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) < 4:
            continue
        cap = parts[0].strip()
        try:
            start, end = float(parts[2]), float(parts[3])
        except ValueError:
            continue
        if start == 0.0 and end == 0.0 and cap:
            return cap
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--humanml3d-root", type=Path, default=DEFAULT_HML3D)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260607)
    ap.add_argument("--min-gt-len", type=int, default=40, help="inclusive lower bound on GT frame count")
    ap.add_argument("--max-gt-len", type=int, default=200, help="exclusive upper bound")
    ap.add_argument("--cap-len", type=int, default=196, help="max target_length (frames)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    root = args.humanml3d_root
    ids = [l.strip() for l in open(root / f"{args.split}.txt", encoding="utf-8") if l.strip()]
    base_ids = [x for x in ids if not x.startswith("M")]  # mirror dedup: keep non-M
    rng = random.Random(args.seed)
    rng.shuffle(base_ids)

    new_joints = root / "new_joints"
    texts = root / "texts"
    rows: list[dict] = []
    seen_caps: set[str] = set()
    stats = {"total_base": len(base_ids), "no_gt": 0, "len_filtered": 0,
             "no_full_caption": 0, "dup_sentence": 0, "accepted": 0, "examined": 0}

    for sid in base_ids:
        stats["examined"] += 1
        gt = new_joints / f"{sid}.npy"
        if not gt.exists():
            stats["no_gt"] += 1
            continue
        gt_len = int(np.load(gt, mmap_mode="r").shape[0])
        if not (args.min_gt_len <= gt_len < args.max_gt_len):
            stats["len_filtered"] += 1
            continue
        cap = _first_full_motion_caption(texts, sid)
        if not cap:
            stats["no_full_caption"] += 1
            continue
        nc = _normalize_caption(cap)
        if nc in seen_caps:
            stats["dup_sentence"] += 1
            continue
        seen_caps.add(nc)
        target_length = min(4 * (gt_len // 4), args.cap_len)
        rows.append({
            "sample_id": sid, "prompt": cap, "gt_length": gt_len,
            "target_length": target_length, "gt_path": str(gt.relative_to(REPO_ROOT)),
        })
        stats["accepted"] += 1
        if len(rows) >= args.n_prompts:
            break

    if len(rows) < args.n_prompts:
        raise RuntimeError(f"only {len(rows)} valid prompts, requested {args.n_prompts}. stats={stats}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(r["prompt"] for r in rows) + "\n", encoding="utf-8")

    meta = {
        "created_at_utc": _now_iso(), "board_id": "AR-048", "record_type": "protocol_prompt_bank",
        "humanml3d_root": str(args.humanml3d_root), "split": args.split, "seed": args.seed,
        "n_prompts": len(rows),
        "filters": {
            "mirror_dedup": "keep non-M (drop M-prefixed mirror)",
            "full_motion_only": "texts frame-range start==0.0 and end==0.0",
            "gt_exists": "new_joints/<id>.npy",
            "sentence_dedup": "normalized (lower+whitespace+trailing-punct)",
            "gt_len_filter": [args.min_gt_len, args.max_gt_len],
        },
        "target_length_rule": f"min(4*floor(gt_len/4), {args.cap_len})",
        "stat_unit": "prompt (seed = generation repetition, not independent unit)",
        "stats": stats,
        "rows": rows,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] {len(rows)} prompts -> {args.output}")
    print(f"[stats] {json.dumps(stats, ensure_ascii=False)}")
    print("[sample] sample_id / gt_len -> target_len / prompt[:55]:")
    for r in rows[:5]:
        print(f"   {r['sample_id']:<8} {r['gt_length']:>3} -> {r['target_length']:>3}  {r['prompt'][:55]!r}")


if __name__ == "__main__":
    main()
