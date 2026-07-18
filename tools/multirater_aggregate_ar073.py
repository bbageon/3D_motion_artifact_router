"""AR-073: 다중 평가자 응답 집계 — per-rater 이항 + Fleiss κ + pooled/majority.

입력 (기입된 시트만 자동 인식):
  rater1: ar072_ab_v3_pack/rater_sheet_filled_rater1.csv (+ v3 answer_key — pair 순서 그대로)
  rater2/3/...: ar073_multirater_pack/<rater>/rater_sheet.csv (which 열 기입 시)
                + 같은 폴더 answer_key_RESEARCHER_ONLY.csv (display→source_pair, rater_A_is)

출력: per-rater corrected-선호 count + 이항 양측 p / Fleiss κ (공통 source_pair, 범주
{corrected, original}) / pooled 선호율 / pair 별 majority. 미기입 rater 는 건너뜀 (부분 집계 가능).

CLI (motion3d env):
    python tools/multirater_aggregate_ar073.py
"""
from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
V3_DIR = REPO_ROOT / "reports" / "figures" / "2026-07-12" / "ar072_ab_v3_pack"
MR_DIR = REPO_ROOT / "reports" / "figures" / "2026-07-12" / "ar073_multirater_pack"
OUT = REPO_ROOT / "evals" / "snapshots" / "multirater_result_ar073_v1.json"


def _binom_two_sided(k, n):
    return sum(comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= abs(k - n / 2)) / 2 ** n


def _load_rater1():
    key = {r["pair_id"]: r["left_is"]
           for r in csv.DictReader(open(V3_DIR / "answer_key_RESEARCHER_ONLY.csv", encoding="utf-8"))}
    sheet = list(csv.DictReader(open(V3_DIR / "rater_sheet_filled_rater1.csv", encoding="utf-8")))
    choices = {}
    for r in sheet:
        pick = r["which_more_natural_A_or_B"].strip().upper()
        if pick not in ("A", "B"):
            return None
        left = key[r["pair_id"]]
        chosen = left if pick == "A" else ("original" if left == "corrected" else "corrected")
        choices[r["pair_id"]] = chosen
    return choices


def _load_multirater(rdir: Path):
    key = {r["display_id"]: r for r in csv.DictReader(open(rdir / "answer_key_RESEARCHER_ONLY.csv", encoding="utf-8"))}
    sheet = list(csv.DictReader(open(rdir / "rater_sheet.csv", encoding="utf-8")))
    choices = {}
    for r in sheet:
        pick = r.get("which_more_natural_A_or_B", "").strip().upper()
        if pick not in ("A", "B"):
            return None    # 미기입 → 전체 skip
        k = key[r["item_id"]]
        a_is = k["rater_A_is"]
        chosen = a_is if pick == "A" else ("original" if a_is == "corrected" else "corrected")
        choices[k["source_pair"]] = chosen
    return choices


def _fleiss_kappa(mat):
    """mat: [n_items, n_categories] count matrix (raters per item 동일)."""
    n_rat = mat.sum(1)[0]
    p_j = mat.sum(0) / mat.sum()
    P_i = ((mat ** 2).sum(1) - n_rat) / (n_rat * (n_rat - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()

    raters = {}
    r1 = _load_rater1()
    if r1:
        raters["rater1"] = r1
    if MR_DIR.exists():
        for rdir in sorted(MR_DIR.iterdir()):
            if rdir.is_dir() and (rdir / "rater_sheet.csv").exists():
                ch = _load_multirater(rdir)
                if ch:
                    raters[rdir.name] = ch
                else:
                    print(f"[SKIP] {rdir.name}: 시트 미기입")
    if not raters:
        print("[FAIL] 기입된 rater 없음")
        return

    per_rater = {}
    for name, ch in raters.items():
        n = len(ch); k = sum(1 for v in ch.values() if v == "corrected")
        per_rater[name] = {"corrected_preferred": k, "n": n,
                           "binom_two_sided_p": round(_binom_two_sided(k, n), 5)}

    common = sorted(set.intersection(*(set(c) for c in raters.values())))
    result = {"schema_version": "1.0.0", "record_type": "multirater_result", "board_id": "AR-073",
              "raters_included": list(raters), "n_common_pairs": len(common),
              "per_rater": per_rater}
    if len(raters) >= 2:
        mat = np.array([[sum(1 for c in raters.values() if c[p] == cat) for cat in ("corrected", "original")]
                        for p in common])
        pooled_k = int(mat[:, 0].sum()); pooled_n = int(mat.sum())
        result["fleiss_kappa"] = round(_fleiss_kappa(mat), 3)
        result["pooled"] = {"corrected_preferred": pooled_k, "n": pooled_n,
                            "rate": round(pooled_k / pooled_n, 3),
                            "binom_two_sided_p": round(_binom_two_sided(pooled_k, pooled_n), 6)}
        result["majority_per_pair"] = {p: ("corrected" if mat[i, 0] > mat[i, 1]
                                           else "original" if mat[i, 1] > mat[i, 0] else "tie")
                                       for i, p in enumerate(common)}
        maj = list(result["majority_per_pair"].values())
        result["majority_summary"] = {v: maj.count(v) for v in ("corrected", "original", "tie")}
        result["tier"] = "b2" if len(raters) >= 3 else "b1+ (2 raters)"
    else:
        result["note"] = "rater 1명뿐 — κ/pooled 는 2명+ 기입 후"
        result["tier"] = "b1"
    result["claim_boundary"] = "지각 선호 집계. 외부 공개급(b2/b3)은 3+ rater + κ 보고 후."

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    import sys as _s; _s.stdout.reconfigure(encoding="utf-8")
    for n, pr in per_rater.items():
        print(f"{n}: {pr['corrected_preferred']}/{pr['n']} (p={pr['binom_two_sided_p']})")
    if "fleiss_kappa" in result:
        print(f"Fleiss κ={result['fleiss_kappa']} | pooled {result['pooled']['corrected_preferred']}/{result['pooled']['n']} "
              f"(p={result['pooled']['binom_two_sided_p']}) | majority {result['majority_summary']}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
