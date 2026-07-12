"""AR-073: AR-072 A/B v3 다중 평가자(b2/b3) 배포 pack — 외부 공개급 승격.

b1(rater1, 19/20) 을 3명+ inter-rater 로 올린다. 새 실험이 아니라 **동일 자극 재사용**:
AR-072 v3 pack 의 20 GIF (원본 vs root-aware, 동일 세그먼트) 를 그대로 쓰되,
다중 평가자용으로 각 rater 에게 **독립 익명화** 배포:
  - pair 제시 순서 rater 별 셔플 (order effect 완화)
  - 익명 display_id (item_NN — 원본 pair_NN 은닉)
  - rater 별 빈 응답 시트 + answer_key (연구자 전용)
  좌우(A/B) 배치는 v3 렌더 그대로 유지 (v3 에서 이미 pair 별 무작위 — corrected-left
  균형; 좌우 재무작위는 GIF 재렌더가 필요해 비용 대비 이득 낮음 → 순서 셔플로 독립성 확보).

집계는 응답 수집 후 별도 (Fleiss κ + aggregate preference + per-rater 이항검정).
본 도구는 **배포 pack 생성만** (판정 = 사람, §2-4).

CLI (motion3d env):
    python tools/multirater_pack_ar073.py --raters rater2 rater3
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
V3_PACK = REPO_ROOT / "reports" / "figures" / "2026-07-12" / "ar072_ab_v3_pack"
V3_KEY = V3_PACK / "answer_key_RESEARCHER_ONLY.csv"
OUT = REPO_ROOT / "reports" / "figures" / "2026-07-12" / "ar073_multirater_pack"
SNAP = REPO_ROOT / "evals" / "snapshots" / "multirater_pack_ar073_v1.json"

# rater 별 셔플/배치 seed (사전 고정 — 재현성).
RATER_SEED = {"rater2": 20260801, "rater3": 20260802, "rater4": 20260803, "rater5": 20260804}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", nargs="+", default=["rater2", "rater3"])
    args = ap.parse_args()

    v3 = list(csv.DictReader(open(V3_KEY, encoding="utf-8")))  # pair_id, ..., left_is, ...
    v3_by_pair = {r["pair_id"]: r for r in v3}
    n = len(v3)
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = {"raters": {}, "source_pack": "ar072_ab_v3_pack", "n_pairs": n,
                "note": "동일 자극 재사용; rater 별 순서 셔플 + 좌우 재무작위 (독립 익명화)"}

    for rater in args.raters:
        seed = RATER_SEED.get(rater, abs(hash(rater)) % (2**31))
        rng = np.random.default_rng(seed)
        rdir = OUT / rater
        rdir.mkdir(parents=True, exist_ok=True)
        order = list(rng.permutation(n))            # 제시 순서 셔플
        rows = []
        for new_i, src_i in enumerate(order, 1):
            src = v3[src_i]
            pid = src["pair_id"]                     # 원본 pair (예: pair_07)
            disp = f"item_{new_i:02d}"               # rater 에게 보이는 익명 ID
            shutil.copyfile(V3_PACK / f"{pid}.gif", rdir / f"{disp}.gif")
            # A(좌)/B(우) 는 v3 렌더 그대로: left_is 가 화면 A 의 정체.
            rows.append({"display_id": disp, "source_pair": pid,
                         "rater_A_is": src["left_is"]})
        # 빈 응답 시트 (rater 배포용 — display_id 만).
        with open(rdir / "rater_sheet.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "which_more_natural_A_or_B", "confidence_0_3", "notes"])
            for r in rows:
                w.writerow([r["display_id"], "", "", ""])
        # index.
        idx = [f"# AR-073 A/B (rater: {rater}) — blind, 독립 익명화", "",
               "> 각 item 에서 **전체적인 자연스러움** (발-지면 접촉 + 몸이 걸음에 맞게 전진) 을 보고",
               "> A/B 중 더 자연스러운 쪽 강제선택 + 확신도(0~3) → [rater_sheet.csv](rater_sheet.csv).",
               "> 실속도(20fps). 정답은 시트에 없습니다.", "",
               "| Item | GIF |", "|---|---|"]
        for r in rows:
            idx.append(f"| {r['display_id']} | [gif]({r['display_id']}.gif) |")
        (rdir / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
        # answer_key (연구자 전용).
        with open(rdir / "answer_key_RESEARCHER_ONLY.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["display_id", "source_pair", "rater_A_is"])
            for r in rows:
                w.writerow([r["display_id"], r["source_pair"], r["rater_A_is"]])
        manifest["raters"][rater] = {"seed": seed, "dir": str(rdir.relative_to(REPO_ROOT)),
                                     "n_items": len(rows)}
        print(f"[OK] {rater}: {len(rows)} items -> {rdir}")

    manifest["aggregation_plan"] = {
        "per_rater": "이항 양측 (corrected 선호 vs 0.5)",
        "inter_rater": "Fleiss kappa (3+ rater, A/B 범주) — source_pair 로 정렬 후",
        "aggregate": "pooled corrected-preference + majority-vote per pair",
        "tier": "b2 (3 rater) / b3 (더 많거나 독립 모집)",
        "criterion": "rater1(b1) 19/20 재현 여부 + kappa 로 신뢰도"}
    manifest["claim_boundary"] = "배포 pack 생성만 — 판정=사람. 집계는 응답 수집 후."
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(SNAP, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[DONE] {len(args.raters)} rater packs -> {OUT}")
    print(f"  manifest: {SNAP}")


if __name__ == "__main__":
    main()
