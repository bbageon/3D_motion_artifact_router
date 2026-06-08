"""AR-049: representative prompt bank + complexity annotation (NOT selection).

설계 변경(spec AR-049-representative): hardness top-300 은 길이 편향(180-199 frame 82.7%)으로
HumanML3D test 대표성을 잃음 → **complexity 로 선별하지 않는다**. AR-048 적격 후보에서 고정 seed
**단순 무작위 추출 300**(representative)을 primary 로 하고, 4축 complexity 는 **조건부 분석용
annotation(raw/z + quartile membership)** 으로만 부착한다. (selection ≠ annotation)

complexity 정의·필터는 [build_hard_prompt_bank](build_hard_prompt_bank.py) 재사용 (동일 a-priori 축).

CLI (motion3d env):
    python tools/build_representative_prompt_bank.py --n-prompts 300 --seed 20260608 \
        --output evals/prompts/protocol_rep_test_300_seed20260608.txt
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))
from tools.build_hard_prompt_bank import (
    SEQ_MARKERS, BODY_PARTS, LATERAL, STOPWORDS, _first_full_motion, _words, _normalize,
)

DEFAULT_HML3D = REPO_ROOT / "external_assets" / "HumanML3D"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    base_ids = sorted(x for x in ids if not x.startswith("M"))  # deterministic order
    new_joints, texts = root / "new_joints", root / "texts"

    # ---- eligibility pass (AR-048 filters) → full eligible corpus (for rarity df) ----
    eligible = []
    seen = set()
    stats = {"total_base": len(base_ids), "no_gt": 0, "len_filtered": 0, "no_full_caption": 0, "dup_sentence": 0}
    for sid in base_ids:
        gt = new_joints / f"{sid}.npy"
        if not gt.exists():
            stats["no_gt"] += 1; continue
        L = int(np.load(gt, mmap_mode="r").shape[0])
        if not (args.min_gt_len <= L < args.max_gt_len):
            stats["len_filtered"] += 1; continue
        fm = _first_full_motion(texts, sid)
        if not fm:
            stats["no_full_caption"] += 1; continue
        cap, pos = fm
        nc = _normalize(cap)
        if nc in seen:
            stats["dup_sentence"] += 1; continue
        seen.add(nc)
        cw = _words(cap)
        eligible.append({
            "sample_id": sid, "prompt": cap, "gt_length": L,
            "target_length": min(4 * (L // 4), args.cap_len),
            "comp_raw": pos.count("/VERB") + 2 * sum(cap.lower().count(m) for m in SEQ_MARKERS),
            "fine_raw": sum(1 for w in cw if w in BODY_PARTS) + sum(1 for w in cw if w in LATERAL),
            "long_raw": L,
            "content_words": sorted({w for w in cw if w not in STOPWORDS and len(w) > 2}),
        })
    if len(eligible) < args.n_prompts:
        raise RuntimeError(f"only {len(eligible)} eligible, need {args.n_prompts}. stats={stats}")

    # rarity over full eligible corpus.
    Ne = len(eligible)
    df = Counter()
    for c in eligible:
        for w in c["content_words"]:
            df[w] += 1
    for c in eligible:
        c["rare_raw"] = (float(np.mean([math.log(Ne / df[w]) for w in c["content_words"]]))
                         if c["content_words"] else 0.0)

    # ---- representative simple random sample (fixed seed; NOT complexity-selected) ----
    rng = random.Random(args.seed)
    sel = rng.sample(eligible, args.n_prompts)
    sel.sort(key=lambda c: c["sample_id"])  # stable order for the bank

    # ---- annotation: z-score within the 300 + quartile membership ----
    def _z(key):
        v = np.array([c[key] for c in sel], dtype=np.float64)
        mu, sd = v.mean(), v.std()
        return (v - mu) / sd if sd > 1e-9 else v * 0.0
    zc, zf, zl, zr = _z("comp_raw"), _z("fine_raw"), _z("long_raw"), _z("rare_raw")
    overall = zc + zf + zl + zr
    # per-axis top-25% thresholds + overall top/bottom 25% (within the 300).
    thr = {k: float(np.percentile([c[k] for c in sel], 75)) for k in ("comp_raw", "fine_raw", "long_raw", "rare_raw")}
    ov_hi, ov_lo = float(np.percentile(overall, 75)), float(np.percentile(overall, 25))

    rows = []
    for i, c in enumerate(sel):
        ov = float(overall[i])
        membership = {
            "comp_top25": bool(c["comp_raw"] >= thr["comp_raw"]),
            "fine_top25": bool(c["fine_raw"] >= thr["fine_raw"]),
            "long_top25": bool(c["long_raw"] >= thr["long_raw"]),
            "rare_top25": bool(c["rare_raw"] >= thr["rare_raw"]),
            "overall_top25": bool(ov >= ov_hi),
            "overall_bottom25": bool(ov <= ov_lo),
        }
        rows.append({
            "sample_id": c["sample_id"], "prompt": c["prompt"], "gt_length": c["gt_length"],
            "target_length": c["target_length"],
            "gt_path": str((new_joints / f"{c['sample_id']}.npy").relative_to(REPO_ROOT)),
            "complexity": {
                "raw": {"comp": c["comp_raw"], "fine": c["fine_raw"], "long": c["long_raw"],
                        "rare": round(c["rare_raw"], 3)},
                "z": {"comp": round(float(zc[i]), 2), "fine": round(float(zf[i]), 2),
                      "long": round(float(zl[i]), 2), "rare": round(float(zr[i]), 2)},
                "overall": round(ov, 3), "membership": membership,
            },
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(r["prompt"] for r in rows) + "\n", encoding="utf-8")
    meta = {
        "created_at_utc": _now(), "board_id": "AR-049", "record_type": "representative_prompt_bank",
        "split": args.split, "seed": args.seed, "n_prompts": len(rows), "n_eligible": Ne,
        "sampling": "AR-048 적격 후보 전체에서 고정 seed 단순 무작위 추출 (complexity 선별/길이 quota 없음)",
        "ar048_filters": "full-motion(0.0/0.0) · mirror dedup(non-M) · GT exists · sentence dedup · GT len [%d,%d) · target=min(4floor(L/4),%d)" % (args.min_gt_len, args.max_gt_len, args.cap_len),
        "complexity_annotation": {
            "role": "조건부 분석용 라벨 (선별 기준 아님). 검증된 절대 난이도 아님 — a-priori linguistic/GT descriptor.",
            "axes": ["comp=#VERB+2*#seq_marker", "fine=#body_part+#laterality", "long=GT_length",
                     "rare=mean inverse-df(content words, eligible corpus)"],
            "z_and_quartile": "z-score·quartile membership 은 representative 300 내부 기준",
            "overall_complexity": "4 z-score 합 (내부 composite label, primary 아님)",
            "pre_registration": "generator 결과 보기 전 고정 (HARKing 방지)",
        },
        "filter_stats": stats, "rows": rows,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # console summary.
    n_long = sum(1 for r in rows if 180 <= r["gt_length"] <= 199)
    print(f"[OK] {len(rows)} representative prompts -> {args.output}  (from {Ne} eligible)")
    print(f"[representativeness] GT len: min={min(r['gt_length'] for r in rows)} "
          f"median={int(np.median([r['gt_length'] for r in rows]))} max={max(r['gt_length'] for r in rows)}; "
          f"180-199 frame = {n_long}/{len(rows)} ({n_long/len(rows):.0%})  (hard-300 은 82.7% 였음)")
    for ax in ("comp", "fine", "long", "rare"):
        hi = sum(1 for r in rows if r["complexity"]["membership"][f"{ax}_top25"])
        print(f"   {ax:<5} top25% subgroup n={hi}")
    print("[sample 5]")
    for r in rows[:5]:
        m = r["complexity"]
        print(f"   {r['sample_id']:<8} L{r['gt_length']:<3} ov={m['overall']:+.1f} "
              f"(v{m['raw']['comp']} body{m['raw']['fine']} r{m['raw']['rare']:.1f})  {r['prompt'][:52]!r}")


if __name__ == "__main__":
    main()
