"""AR-049: hard-tier prompt bank builder (a-priori hardness, generator-independent).

사전 등록(HARKing 방지): generator 결과를 보기 **전에** 언어/GT 특징만으로 hardness 를 정의·고정하고
HumanML3D test 에서 top-N hard prompt 를 선별한다. AR-048 protocol 재사용(같은 builder 필터).

Hardness = 4 축의 z-score 등가중 합 (모두 caption 텍스트/POS/GT길이에서만 계산, generator 무관):
  1. compositional = #VERB(POS) + 2·#sequence_marker(then/while/before/after/until/and then)
  2. fine_grained  = #body_part_term + #laterality(left/right)
  3. long          = GT_length (frame)
  4. rare          = mean over content-word 의 inverse document-frequency (corpus = 후보 캡션 전체)

AR-048 필터 동일: test split · full-motion(0.0/0.0) caption · mirror dedup(non-M) · GT 존재 ·
문장 dedup · GT 길이 [40,200) · target_length = min(4·floor(L/4),196).

CLI (motion3d env):
    python tools/build_hard_prompt_bank.py --n-prompts 300 --seed 20260608 \
        --output evals/prompts/protocol_hard_test_300_seed20260608.txt
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HML3D = REPO_ROOT / "external_assets" / "HumanML3D"

SEQ_MARKERS = ("then", "while", "before", "after", "until", "once")
BODY_PARTS = {
    "hand", "hands", "arm", "arms", "leg", "legs", "knee", "knees", "foot", "feet",
    "shoulder", "shoulders", "elbow", "elbows", "head", "hip", "hips", "torso", "wrist",
    "wrists", "finger", "fingers", "chest", "back", "waist", "ankle", "ankles", "neck",
    "thigh", "thighs", "palm", "palms", "toe", "toes", "heel", "heels", "fist",
}
LATERAL = {"left", "right"}
STOPWORDS = {
    "a", "an", "the", "is", "are", "to", "of", "and", "in", "on", "with", "his", "her",
    "their", "they", "he", "she", "it", "as", "at", "for", "then", "person", "man",
    "woman", "figure", "someone", "while", "from", "up", "down", "out", "into",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize(c: str) -> str:
    return re.sub(r"\s+", " ", c.strip().lower()).rstrip(". ")


def _first_full_motion(texts_dir: Path, sid: str) -> tuple[str, str] | None:
    """(caption, pos_tags) of first full-motion(0.0/0.0) line; None if none."""
    p = texts_dir / f"{sid}.txt"
    if not p.exists():
        return None
    for line in open(p, encoding="utf-8"):
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        cap, pos = parts[0].strip(), parts[1].strip()
        try:
            if float(parts[2]) == 0.0 and float(parts[3]) == 0.0 and cap:
                return cap, pos
        except ValueError:
            continue
    return None


def _words(c: str) -> list[str]:
    return re.findall(r"[a-z]+", c.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--humanml3d-root", type=Path, default=DEFAULT_HML3D)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-prompts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--min-gt-len", type=int, default=40)
    ap.add_argument("--max-gt-len", type=int, default=200)
    ap.add_argument("--cap-len", type=int, default=196)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    root = args.humanml3d_root
    ids = [l.strip() for l in open(root / f"{args.split}.txt", encoding="utf-8") if l.strip()]
    base_ids = [x for x in ids if not x.startswith("M")]
    new_joints, texts = root / "new_joints", root / "texts"

    # pass 1: collect passing candidates (AR-048 filters) + raw hardness components.
    cands = []
    seen = set()
    stats = {"total_base": len(base_ids), "no_gt": 0, "len_filtered": 0, "no_full_caption": 0, "dup_sentence": 0}
    for sid in base_ids:
        gt = new_joints / f"{sid}.npy"
        if not gt.exists():
            stats["no_gt"] += 1
            continue
        L = int(np.load(gt, mmap_mode="r").shape[0])
        if not (args.min_gt_len <= L < args.max_gt_len):
            stats["len_filtered"] += 1
            continue
        fm = _first_full_motion(texts, sid)
        if not fm:
            stats["no_full_caption"] += 1
            continue
        cap, pos = fm
        nc = _normalize(cap)
        if nc in seen:
            stats["dup_sentence"] += 1
            continue
        seen.add(nc)
        cw = _words(cap)
        n_verb = pos.count("/VERB")
        n_seq = sum(cap.lower().count(m) for m in SEQ_MARKERS)
        n_body = sum(1 for w in cw if w in BODY_PARTS)
        n_lat = sum(1 for w in cw if w in LATERAL)
        cands.append({
            "sample_id": sid, "prompt": cap, "gt_length": L,
            "target_length": min(4 * (L // 4), args.cap_len),
            "comp_raw": n_verb + 2 * n_seq, "fine_raw": n_body + n_lat, "long_raw": L,
            "content_words": [w for w in cw if w not in STOPWORDS and len(w) > 2],
        })
    if len(cands) < args.n_prompts:
        raise RuntimeError(f"only {len(cands)} candidates, need {args.n_prompts}. stats={stats}")

    # rare: document-frequency over candidate corpus → mean inverse-df of content words.
    N = len(cands)
    df = Counter()
    for c in cands:
        for w in set(c["content_words"]):
            df[w] += 1
    for c in cands:
        cwu = set(c["content_words"])
        c["rare_raw"] = (float(np.mean([math.log(N / df[w]) for w in cwu])) if cwu else 0.0)

    # z-normalize 4 axes, hardness = equal-weight sum.
    def _z(key):
        v = np.array([c[key] for c in cands], dtype=np.float64)
        mu, sd = v.mean(), v.std()
        return (v - mu) / sd if sd > 1e-9 else v * 0.0
    zc, zf, zl, zr = _z("comp_raw"), _z("fine_raw"), _z("long_raw"), _z("rare_raw")
    for i, c in enumerate(cands):
        c["z_comp"], c["z_fine"], c["z_long"], c["z_rare"] = float(zc[i]), float(zf[i]), float(zl[i]), float(zr[i])
        c["hardness"] = c["z_comp"] + c["z_fine"] + c["z_long"] + c["z_rare"]

    # select top-N by hardness (deterministic; tie-break by sample_id for reproducibility).
    cands.sort(key=lambda c: (-c["hardness"], c["sample_id"]))
    selected = cands[: args.n_prompts]

    rows = [{"sample_id": c["sample_id"], "prompt": c["prompt"], "gt_length": c["gt_length"],
             "target_length": c["target_length"], "gt_path": str((new_joints / f"{c['sample_id']}.npy").relative_to(REPO_ROOT)),
             "hardness": round(c["hardness"], 3),
             "hardness_breakdown": {"comp_raw": c["comp_raw"], "fine_raw": c["fine_raw"],
                                    "long_raw": c["long_raw"], "rare_raw": round(c["rare_raw"], 3),
                                    "z": [round(c["z_comp"], 2), round(c["z_fine"], 2),
                                          round(c["z_long"], 2), round(c["z_rare"], 2)]}}
            for c in selected]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(r["prompt"] for r in rows) + "\n", encoding="utf-8")
    meta = {
        "created_at_utc": _now(), "board_id": "AR-049", "record_type": "hard_tier_prompt_bank",
        "split": args.split, "seed": args.seed, "n_prompts": len(rows), "n_candidates": N,
        "hardness_definition": {
            "axes": ["compositional = #VERB + 2*#seq_marker", "fine_grained = #body_part + #laterality",
                     "long = GT_length", "rare = mean inverse-df of content words (candidate corpus)"],
            "combination": "equal-weight sum of per-axis z-scores; select top-N (a-priori, generator-independent)",
            "seq_markers": list(SEQ_MARKERS), "n_body_part_terms": len(BODY_PARTS),
            "pre_registration": "generator 결과 확인 전 고정 (HARKing 방지, AGENTS §3-11 정신)",
        },
        "ar048_filters": "test split · full-motion(0.0/0.0) · mirror dedup(non-M) · GT exists · sentence dedup · GT len [%d,%d) · target=min(4floor(L/4),%d)" % (args.min_gt_len, args.max_gt_len, args.cap_len),
        "filter_stats": stats, "rows": rows,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # console summary.
    print(f"[OK] {len(rows)} hard prompts -> {args.output}  (from {N} candidates)")
    hv = np.array([r["hardness"] for r in rows])
    allh = np.array([c["hardness"] for c in cands])
    print(f"[hardness] selected min/median/max = {hv.min():.2f}/{np.median(hv):.2f}/{hv.max():.2f}  "
          f"(all candidates median {np.median(allh):.2f})")
    print(f"[axes mean(selected raw)] comp={np.mean([r['hardness_breakdown']['comp_raw'] for r in rows]):.2f} "
          f"fine={np.mean([r['hardness_breakdown']['fine_raw'] for r in rows]):.2f} "
          f"long={np.mean([r['hardness_breakdown']['long_raw'] for r in rows]):.0f} "
          f"rare={np.mean([r['hardness_breakdown']['rare_raw'] for r in rows]):.2f}")
    print("[hardest 8]")
    for r in rows[:8]:
        b = r["hardness_breakdown"]
        print(f"   {r['sample_id']:<8} h={r['hardness']:5.2f} (v{b['comp_raw']} body{b['fine_raw']} L{b['long_raw']} r{b['rare_raw']:.1f})  {r['prompt'][:58]!r}")


if __name__ == "__main__":
    main()
