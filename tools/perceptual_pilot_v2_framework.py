"""Step 8-C: Perceptual Pilot v2 stratified manifest + response template + analyzer.

사용자 directive (2026-05-26) 박제:
- Group A (G2 STOP-best): 7 samples × 2 pair types = 14 pairs.
- Group B (G2 correction-best, pilot n=3): 3 samples × 2 pair types = 6 pairs.
- Group C (Synthetic 5-level, high-Δ enriched): 10 samples × (10+10+5 split) = 25 pairs.
- Group D (Negative controls, correlation 제외): 5 pairs.
Total = 50 pairs. 2-session 평가 권장 (§6-3 of DESIGN.md).

본 framework 는 두 mode:
  1. **manifest** mode — 50-pair stratified manifest + response template + INSTRUCTIONS 생성.
  2. **analyze** mode — response CSV → group-aware analysis (Group D 는 correlation 분석에서 제외).

사용자 작업 흐름:
  1. python -m tools.perceptual_pilot_v2_framework manifest --output-dir evals/perceptual/pilot_v2
  2. 사용자가 GIF 보면서 response_v2.csv 의 user_choice column 채움 (50 row)
  3. python -m tools.perceptual_pilot_v2_framework analyze \
         --manifest evals/perceptual/pilot_v2/pairs_manifest_v2.json \
         --response evals/perceptual/pilot_v2/response_v2.csv \
         --output evals/snapshots/perceptual_pilot_v2.json

Design 단일 출처: [`evals/perceptual/pilot_v2/DESIGN.md`](../evals/perceptual/pilot_v2/DESIGN.md).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- Group A/B: G2 general 의 sample 분리 ----
GROUP_A_STOP_BEST_SAMPLES = ["motion_003", "motion_005", "motion_006", "motion_007",
                              "motion_008", "motion_009", "motion_010"]
GROUP_B_CORRECTION_BEST_SAMPLES = ["motion_001", "motion_002", "motion_004"]

# ---- Group C: synthetic top-10 high-Δ + top-5 PC ----
GROUP_C_TOP_10 = ["014552", "M007995", "012798", "M000741", "012631",
                   "M008358", "001885", "010382", "012943", "M007140"]
GROUP_C_TOP_5_PC = ["014552", "M007995", "012798", "M000741", "012631"]

# ---- Group D: negative controls (identical GIF × 2) ----
GROUP_D_NC_DEFS = [
    {"nc_id": "NC1", "trial_id": "motion_001", "method": "5-level oracle"},
    {"nc_id": "NC2", "trial_id": "014552", "method": "5-level oracle"},
    {"nc_id": "NC3", "trial_id": "motion_003", "method": "Original (STOP)"},
    {"nc_id": "NC4", "trial_id": "M007995", "method": "5-level oracle"},
    {"nc_id": "NC5", "trial_id": "motion_010", "method": "B2-small"},
]


def _gif_path_g2(trial_id: str, pair_type: str) -> str:
    """G2 general 의 method 별 GIF path 매핑 (v1/v2 yup_fix dir 자동 선택).

    pair_type: 'original_vs_b2_large', 'original_vs_oracle', 'original_vs_b2_valbest',
                'b2_valbest_vs_oracle'.
    """
    # v1 yup_fix: motion_001, 004, 007, 008, 010.
    # v2 perceptual_v2 yup_fix: motion_002, 003, 005, 006, 009 (신규 생성).
    v1_dir = "reports/figures/2026-05-25/g2_general_gif_yup_fix"
    v2_dir = "reports/figures/2026-05-26/perceptual_v2_gif_yup_fix"
    base_dir = v1_dir if trial_id in {"motion_001", "motion_004", "motion_007", "motion_008", "motion_010"} else v2_dir
    if pair_type == "original_vs_b2_large":
        return f"{base_dir}/{trial_id}_vs_b2_large.gif"
    if pair_type == "original_vs_oracle":
        return f"{base_dir}/{trial_id}_vs_oracle.gif"
    if pair_type == "original_vs_b2_valbest":
        # G2 의 B2-val-best = B2-small (부록 V, 100%).
        return f"{base_dir}/{trial_id}_vs_b2_small.gif"
    if pair_type == "b2_valbest_vs_oracle":
        # 별도 overlay GIF 없으면 — 두 단일 GIF 를 reference. v2 시점 에서는 vs_b2_small + vs_oracle 의
        # 두 GIF 를 함께 본다 (rater 가 두 motion 의 차이를 mental compose).
        # 따라서 본 pair 는 매우 미세 — Group B 의 핵심 pair 이지만 GIF rendering 한계.
        # 실제 implementation 에서 별도 b2_valbest_vs_oracle.gif 를 추후 생성 가능.
        return f"{base_dir}/{trial_id}_vs_oracle.gif"  # reference, gray=original, but read as B2-small
    return ""


def _gif_path_synthetic(trial_id: str, pair_type: str) -> str:
    """Synthetic GIF path 매핑.

    pair_type: '3level_vs_5level', 'b2valbest_vs_5level', 'corrupted_vs_5level'.
    """
    base = "reports/figures/2026-05-26/perceptual_v2_gif_yup_fix"
    return f"{base}/synthetic_{trial_id}_{pair_type}.gif"


def _build_group_a_pairs(seed_rng: np.random.Generator) -> list[dict[str, Any]]:
    """Group A: G2 STOP-best (7 × 2 = 14 pairs)."""
    pairs = []
    for tid in GROUP_A_STOP_BEST_SAMPLES:
        # Pair 1: Original vs B2-large (over-modification visible).
        a, b = "Original", "B2-large"
        if seed_rng.random() < 0.5:
            a, b = b, a
        pairs.append({
            "group": "A_STOP_best",
            "category_pair": "Original vs B2-large",
            "trial_id": tid,
            "blind_label_A": a, "blind_label_B": b,
            "gif_A": _gif_path_g2(tid, "original_vs_b2_large") if a == "Original" else _gif_path_g2(tid, "original_vs_b2_large"),
            "gif_B": _gif_path_g2(tid, "original_vs_b2_large"),
            "netgain_winner_hidden": "Original",  # B2-large 의 NetGain << 0 → Original 우월.
            "note": "overlay GIF: gray = original, orange = B2-large. blind_label 에 따라 A/B 의 의미 결정.",
        })
        # Pair 2: Original vs oracle(STOP) — identity comparison (sanity check).
        a, b = "Original", "oracle (STOP)"
        if seed_rng.random() < 0.5:
            a, b = b, a
        pairs.append({
            "group": "A_STOP_best",
            "category_pair": "Original vs oracle(STOP)",
            "trial_id": tid,
            "blind_label_A": a, "blind_label_B": b,
            "gif_A": _gif_path_g2(tid, "original_vs_oracle"),
            "gif_B": _gif_path_g2(tid, "original_vs_oracle"),
            "netgain_winner_hidden": "tie",  # NetGain=0 (STOP identity).
            "note": "STOP 의 sanity check — tie 가 dominant 이어야 정상.",
        })
    return pairs


def _build_group_b_pairs(seed_rng: np.random.Generator) -> list[dict[str, Any]]:
    """Group B: G2 correction-best (3 × 2 = 6 pairs, pilot n=3)."""
    pairs = []
    for tid in GROUP_B_CORRECTION_BEST_SAMPLES:
        # Pair 1: Original vs B2-val-best (= B2-small for G2).
        a, b = "Original", "B2-val-best"
        if seed_rng.random() < 0.5:
            a, b = b, a
        pairs.append({
            "group": "B_correction_best",
            "category_pair": "Original vs B2-val-best",
            "trial_id": tid,
            "blind_label_A": a, "blind_label_B": b,
            "gif_A": _gif_path_g2(tid, "original_vs_b2_valbest"),
            "gif_B": _gif_path_g2(tid, "original_vs_b2_valbest"),
            "netgain_winner_hidden": "B2-val-best",  # NetGain 가까운 0, but slightly > 0.
            "note": "overlay GIF: gray = original, orange = B2-small (= val-best for G2).",
        })
        # Pair 2: B2-val-best vs oracle.
        a, b = "B2-val-best", "oracle"
        if seed_rng.random() < 0.5:
            a, b = b, a
        pairs.append({
            "group": "B_correction_best",
            "category_pair": "B2-val-best vs oracle",
            "trial_id": tid,
            "blind_label_A": a, "blind_label_B": b,
            "gif_A": _gif_path_g2(tid, "original_vs_b2_valbest"),  # reference: gray=orig, orange=B2-small.
            "gif_B": _gif_path_g2(tid, "original_vs_oracle"),  # reference: gray=orig, orange=oracle.
            "netgain_winner_hidden": "oracle",
            "note": "두 GIF 를 함께 봐서 B2-small vs oracle 의 차이 비교 — pilot limit (n=3).",
        })
    return pairs


def _build_group_c_pairs(
    seq3_by: dict[str, list], seq5_by: dict[str, list], seed_rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Group C: synthetic 5-level high-Δ enriched (10 × 2 + 5 × 1 = 25 pairs)."""
    pairs = []
    for tid in GROUP_C_TOP_10:
        len3 = len(seq3_by.get(tid, []))
        len5 = len(seq5_by.get(tid, []))
        # Pair 1: 3-level oracle vs 5-level oracle (핵심).
        a, b = "3-level oracle", "5-level oracle"
        if seed_rng.random() < 0.5:
            a, b = b, a
        pairs.append({
            "group": "C_synthetic_5level",
            "category_pair": "3-level oracle vs 5-level oracle",
            "trial_id": tid,
            "blind_label_A": a, "blind_label_B": b,
            "gif_A": _gif_path_synthetic(tid, "3level_vs_5level"),
            "gif_B": _gif_path_synthetic(tid, "3level_vs_5level"),
            "netgain_winner_hidden": "5-level oracle",  # Case A 정량 evidence.
            "note": f"overlay GIF: gray = 3-level (len {len3}), orange = 5-level (len {len5}). Case A 의 perceptual 검증.",
        })
        # Pair 2: B2-val-best vs 5-level oracle.
        a, b = "B2-val-best", "5-level oracle"
        if seed_rng.random() < 0.5:
            a, b = b, a
        pairs.append({
            "group": "C_synthetic_5level",
            "category_pair": "B2-val-best vs 5-level oracle",
            "trial_id": tid,
            "blind_label_A": a, "blind_label_B": b,
            "gif_A": _gif_path_synthetic(tid, "b2valbest_vs_5level"),
            "gif_B": _gif_path_synthetic(tid, "b2valbest_vs_5level"),
            "netgain_winner_hidden": "5-level oracle",
            "note": "overlay GIF: gray = B2-medium (synthetic val-best proxy), orange = 5-level oracle.",
        })
    # Pair 3 (positive control, top-5 only): corrupted vs 5-level oracle.
    for tid in GROUP_C_TOP_5_PC:
        a, b = "corrupted", "5-level oracle"
        if seed_rng.random() < 0.5:
            a, b = b, a
        pairs.append({
            "group": "C_positive_control",
            "category_pair": "corrupted vs 5-level oracle [PC]",
            "trial_id": tid,
            "blind_label_A": a, "blind_label_B": b,
            "gif_A": _gif_path_synthetic(tid, "corrupted_vs_5level"),
            "gif_B": _gif_path_synthetic(tid, "corrupted_vs_5level"),
            "netgain_winner_hidden": "5-level oracle",
            "note": "POSITIVE CONTROL — overlay GIF: gray = corrupted, orange = 5-level oracle. 너무 쉬운 비교 — sanity check only.",
        })
    return pairs


def _build_group_d_pairs() -> list[dict[str, Any]]:
    """Group D: negative controls (5 identical pairs, correlation 제외)."""
    pairs = []
    for nc in GROUP_D_NC_DEFS:
        tid = nc["trial_id"]
        # identical GIF for both A and B — choose representative GIF for the method.
        if nc["method"] == "5-level oracle":
            gif = _gif_path_synthetic(tid, "3level_vs_5level") if tid in GROUP_C_TOP_10 else _gif_path_g2(tid, "original_vs_oracle")
        elif nc["method"] == "Original (STOP)":
            gif = _gif_path_g2(tid, "original_vs_oracle")
        elif nc["method"] == "B2-small":
            gif = _gif_path_g2(tid, "original_vs_b2_valbest")
        else:
            gif = "(missing)"
        pairs.append({
            "group": "D_negative_control",
            "category_pair": f"NC: {nc['method']} (identical)",
            "trial_id": tid,
            "blind_label_A": nc["method"], "blind_label_B": nc["method"],
            "gif_A": gif, "gif_B": gif,
            "netgain_winner_hidden": "tie",  # identical → must be tie.
            "note": f"NEGATIVE CONTROL ({nc['nc_id']}) — identical GIF, expected tie. Correlation 분석에서 제외.",
        })
    return pairs


def _build_manifest(*, snap_3level: Path, snap_5level: Path, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    # Load oracle sequences for Group C.
    with open(snap_3level, encoding="utf-8") as f:
        snap3 = json.load(f)
    with open(snap_5level, encoding="utf-8") as f:
        snap5 = json.load(f)
    seq3_by = {s["trial_id"]: s["best_A"]["sequence"] for s in snap3["per_sample"]}
    seq5_by = {s["trial_id"]: s["best_A"]["sequence"] for s in snap5["per_sample"]}

    # Build groups.
    group_a = _build_group_a_pairs(rng)
    group_b = _build_group_b_pairs(rng)
    group_c = _build_group_c_pairs(seq3_by, seq5_by, rng)
    group_d = _build_group_d_pairs()

    all_pairs = group_a + group_b + group_c + group_d
    # Shuffle order (group pattern 노출 차단).
    indices = np.arange(len(all_pairs))
    rng.shuffle(indices)
    shuffled = [all_pairs[i] for i in indices]
    # Pair IDs after shuffle.
    for i, p in enumerate(shuffled, 1):
        p["pair_id"] = f"P{i:03d}"

    # Session split (per DESIGN.md §6-3).
    session1, session2 = [], []
    for p in shuffled:
        g = p["group"]
        if g in ("A_STOP_best", "B_correction_best"):
            session1.append(p["pair_id"])
        elif g in ("C_synthetic_5level", "C_positive_control"):
            session2.append(p["pair_id"])
        elif g == "D_negative_control":
            # NC1/3/5 (G2-related) → session 1, NC2/4 (synthetic) → session 2.
            if p["trial_id"] in {"014552", "M007995"}:
                session2.append(p["pair_id"])
            else:
                session1.append(p["pair_id"])

    return {
        "schema_version": "2.0.0",
        "record_type": "perceptual_pilot_manifest_v2",
        "task_id": "perceptual_pilot_v2",
        "seed": seed,
        "design_doc": "evals/perceptual/pilot_v2/DESIGN.md",
        "n_pairs": len(shuffled),
        "group_counts": {
            "A_STOP_best": sum(1 for p in shuffled if p["group"] == "A_STOP_best"),
            "B_correction_best": sum(1 for p in shuffled if p["group"] == "B_correction_best"),
            "C_synthetic_5level": sum(1 for p in shuffled if p["group"] == "C_synthetic_5level"),
            "C_positive_control": sum(1 for p in shuffled if p["group"] == "C_positive_control"),
            "D_negative_control": sum(1 for p in shuffled if p["group"] == "D_negative_control"),
        },
        "sessions": {
            "session1_pair_ids": session1,
            "session2_pair_ids": session2,
        },
        "pairs": shuffled,
    }


def _build_response_template(manifest: dict[str, Any]) -> str:
    lines = ["pair_id,session,group,category_pair,trial_id,gif_A,gif_B,user_choice,confidence,comments"]
    s1 = set(manifest["sessions"]["session1_pair_ids"])
    for p in manifest["pairs"]:
        session = "1" if p["pair_id"] in s1 else "2"
        lines.append(
            f"{p['pair_id']},{session},{p['group']},"
            f"\"{p['category_pair']}\",{p['trial_id']},"
            f"{p['gif_A']},{p['gif_B']},,,"
        )
    return "\n".join(lines) + "\n"


def _build_instructions(manifest: dict[str, Any]) -> str:
    n = manifest["n_pairs"]
    s1 = len(manifest["sessions"]["session1_pair_ids"])
    s2 = len(manifest["sessions"]["session2_pair_ids"])
    gc = manifest["group_counts"]
    return f"""\
﻿# Perceptual Pilot v2 — 사용자 작업 안내 (Step 8-C)

> 본 문서는 [`DESIGN.md`](DESIGN.md) 의 stratified design 의 사용자 작업 절차. AGENTS.md §3-17 quality-validated evidence (b1 internal sanity sub-tier).

---

## 1. 본 pilot 의 목적

**5-level RL-2 action space 의 perceptual 검증** (synthetic Case A) + **G2 STOP/weak correction 의 perceptual 검증**.

- **Group A (G2 STOP-best, n=7×2=14)**: G2 의 over-modification 검증.
- **Group B (G2 correction-best, n=3×2=6, pilot)**: G2 의 adaptive routing 의 lower-bound evidence.
- **Group C (Synthetic 5-level high-Δ enriched, n=25)**: RL-2 = 5-level 의 perceptual supports.
- **Group D (Negative controls, n=5)**: rater sanity check (correlation 분석 제외).

본 pilot 은 **1명 internal sanity check (b1 sub-tier)**. 정식 evidence (b3) 는 10-20명 user study 의 후속.

---

## 2. 사용자 작업 절차

### Step 1: 2-session 분할 평가 (권장)

총 {n} pairs ~ 35-50 분. **2-session 분할 권장**:

| Session | pair 수 | 시간 | 포함 |
|---|---|---|---|
| **Session 1** | {s1} | ~ 15-20 분 | Group A/B (G2) + NC1/3/5 (G2 NC) |
| **Session 2** | {s2} | ~ 20-30 분 | Group C (synthetic) + NC2/4 (synthetic NC) |

각 session 사이 휴식 권장. session 1/2 순서는 자유.

### Step 2: GIF browse

[`reports/figures/2026-05-25/g2_general_gif_yup_fix/`](../../../reports/figures/2026-05-25/g2_general_gif_yup_fix/) (motion_001,004,007,008,010)
[`reports/figures/2026-05-26/perceptual_v2_gif_yup_fix/`](../../../reports/figures/2026-05-26/perceptual_v2_gif_yup_fix/) (motion_002,003,005,006,009 + synthetic)

### Step 3: pair_id 별 평가

각 row (`response_template_v2.csv`) 의 다음 column 채움:

| Field | 의무 | 값 |
|---|---|---|
| `user_choice` | 의무 | `A`, `B`, 또는 `tie` |
| `confidence` | 옵션 | 1-5 (1=very unsure, 5=very sure) |
| `comments` | 옵션 | 자유 텍스트 |

**평가 기준**: "어느 motion 이 더 자연스러운가?". GIF 의 gray vs orange 는 method_A vs method_B (manifest 에 매핑 hidden).

### Step 4: response_v2.csv 저장

작성 완료 후 [`response_v2.csv`](response_v2.csv).

### Step 5: Analysis 실행

```
python -m tools.perceptual_pilot_v2_framework analyze \\
    --manifest evals/perceptual/pilot_v2/pairs_manifest_v2.json \\
    --response evals/perceptual/pilot_v2/response_v2.csv \\
    --output evals/snapshots/perceptual_pilot_v2.json
```

---

## 3. Group counts (manifest 의 분포)

| Group | Count |
|---|---|
| A_STOP_best | {gc.get('A_STOP_best', 0)} |
| B_correction_best | {gc.get('B_correction_best', 0)} |
| C_synthetic_5level | {gc.get('C_synthetic_5level', 0)} |
| C_positive_control | {gc.get('C_positive_control', 0)} |
| D_negative_control | {gc.get('D_negative_control', 0)} |
| **Total** | **{n}** |

---

## 4. 본 pilot 의 한계 (DESIGN.md §8 박제)

1. n=50, 1명 evaluator — **b1 sub-tier internal sanity** only.
2. Group C 는 high-Δ enriched subset (전체 representative 아님).
3. Group B 는 n=3 pilot (G2 natural n=50 보강 candidate 별도 등록).
4. Group D 는 correlation 분석에서 제외 (rater sanity only).

---

## 5. Cross-link

- Design: [`DESIGN.md`](DESIGN.md).
- v1 (참고): [`../pilot_v1/`](../pilot_v1/).
- Manifest: [`pairs_manifest_v2.json`](pairs_manifest_v2.json).
- Response template: [`response_template_v2.csv`](response_template_v2.csv).
- Framework: [`tools/perceptual_pilot_v2_framework.py`](../../../tools/perceptual_pilot_v2_framework.py).
"""


def _analyze(manifest_path: Path, response_path: Path, output_path: Path) -> None:
    with open(manifest_path, encoding="utf-8-sig") as f:
        manifest = json.load(f)
    pairs_by_id = {p["pair_id"]: p for p in manifest["pairs"]}

    responses = []
    with open(response_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            choice = row.get("user_choice", "").strip()
            if choice in ("A", "B", "tie"):
                responses.append(row)
    print(f"[INFO] loaded {len(responses)} responses")
    if not responses:
        print("[WARN] no responses — analysis skipped.")
        return

    group_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"agree": 0, "disagree": 0, "tie": 0})
    pair_type_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"agree": 0, "disagree": 0, "tie": 0})
    nc_results: list[dict[str, Any]] = []
    correlation_data_ng: list[float] = []
    correlation_data_human: list[int] = []
    disagreements: list[dict[str, Any]] = []

    for r in responses:
        pid = r["pair_id"]
        if pid not in pairs_by_id:
            continue
        p = pairs_by_id[pid]
        choice = r["user_choice"].strip()
        group = p["group"]
        # Negative control 의 분석 분리.
        if group == "D_negative_control":
            nc_results.append({"pair_id": pid, "choice": choice, "expected": "tie"})
            continue
        # Group A/B/C 의 analysis.
        netgain_winner = p["netgain_winner_hidden"]
        human_winner = p["blind_label_A"] if choice == "A" else (p["blind_label_B"] if choice == "B" else "tie")
        category = p["category_pair"]
        if choice == "tie" or netgain_winner == "tie":
            group_stats[group]["tie"] += 1
            pair_type_stats[category]["tie"] += 1
            agree_flag = "tie"
        elif human_winner == netgain_winner:
            group_stats[group]["agree"] += 1
            pair_type_stats[category]["agree"] += 1
            agree_flag = "agree"
        else:
            group_stats[group]["disagree"] += 1
            pair_type_stats[category]["disagree"] += 1
            agree_flag = "disagree"
            disagreements.append({
                "pair_id": pid, "group": group, "category": category,
                "trial_id": p["trial_id"],
                "netgain_winner": netgain_winner, "human_winner": human_winner,
            })
        # Correlation data — Group A/B/C only (D 제외).
        # Sign: blind_label_A 가 "5-level oracle" 또는 NetGain winner 면 +1, B 면 -1.
        winner_sign = 0
        if netgain_winner == p["blind_label_A"]:
            winner_sign = 1
        elif netgain_winner == p["blind_label_B"]:
            winner_sign = -1
        # Skip pairs where NetGain winner is "tie" (no sign).
        if winner_sign != 0:
            human_sign = 1 if choice == "A" else (-1 if choice == "B" else 0)
            correlation_data_ng.append(float(winner_sign))
            correlation_data_human.append(int(human_sign))

    # NC analysis.
    nc_tie = sum(1 for r in nc_results if r["choice"] == "tie")
    nc_total = len(nc_results)
    nc_pass = nc_tie >= 4  # threshold per DESIGN.md §5-4.

    # Correlation (excluding D and ties).
    from scipy import stats as sp_stats
    spearman_r, spearman_p, kendall_tau, kendall_p = None, None, None, None
    if len(correlation_data_ng) > 1:
        sr = sp_stats.spearmanr(correlation_data_ng, correlation_data_human)
        spearman_r, spearman_p = float(sr.statistic), float(sr.pvalue)
        kt = sp_stats.kendalltau(correlation_data_ng, correlation_data_human)
        kendall_tau, kendall_p = float(kt.statistic), float(kt.pvalue)

    # Per-group agreement rate.
    group_agreement_rates = {}
    for g, s in group_stats.items():
        n_nontie = s["agree"] + s["disagree"]
        group_agreement_rates[g] = {
            "agree": s["agree"], "disagree": s["disagree"], "tie": s["tie"],
            "agreement_rate_excl_tie": s["agree"] / max(n_nontie, 1) if n_nontie > 0 else 0.0,
        }

    summary = {
        "schema_version": "2.0.0",
        "record_type": "perceptual_pilot_analysis_v2",
        "manifest_path": str(manifest_path),
        "response_path": str(response_path),
        "n_responses": len(responses),
        "negative_controls": {
            "n_total": nc_total, "n_tie": nc_tie,
            "pass_threshold_tie_>=4": nc_pass,
            "results": nc_results,
        },
        "by_group": dict(group_agreement_rates),
        "by_pair_type": {k: dict(v) for k, v in pair_type_stats.items()},
        "correlation_excluding_D_and_ties": {
            "n": len(correlation_data_ng),
            "spearman_r": spearman_r, "spearman_p": spearman_p,
            "kendall_tau": kendall_tau, "kendall_p": kendall_p,
        },
        "disagreements": disagreements,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== Perceptual Pilot v2 Analysis ===")
    print(f"  n_responses: {len(responses)}")
    print(f"  NC sanity (tie {nc_tie}/{nc_total} >= 4): {'PASS' if nc_pass else 'FAIL'}")
    for g, s in group_agreement_rates.items():
        print(f"  [{g}] agree {s['agree']}, disagree {s['disagree']}, tie {s['tie']}, agreement_rate={s['agreement_rate_excl_tie']:.3f}")
    if spearman_r is not None:
        print(f"  Spearman r: {spearman_r:.3f} (p={spearman_p:.4g})")
        print(f"  Kendall tau: {kendall_tau:.3f} (p={kendall_p:.4g})")
    print(f"[OK] wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Perceptual pilot v2 stratified framework (Step 8-C)")
    sub = parser.add_subparsers(dest="mode", required=True)

    mp = sub.add_parser("manifest")
    mp.add_argument("--snapshot-3level", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "oracle_sequence_multi_v2_n60.json")
    mp.add_argument("--snapshot-5level", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "oracle_sequence_multi_5level_v1.json")
    mp.add_argument("--output-dir", type=Path, default=REPO_ROOT / "evals" / "perceptual" / "pilot_v2")
    mp.add_argument("--seed", type=int, default=42)

    ap = sub.add_parser("analyze")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--response", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.mode == "manifest":
        manifest = _build_manifest(
            snap_3level=args.snapshot_3level, snap_5level=args.snapshot_5level, seed=args.seed,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Write manifest (with BOM for Windows readability).
        mpath = args.output_dir / "pairs_manifest_v2.json"
        mpath.write_text("﻿" + json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] wrote {mpath} ({manifest['n_pairs']} pairs)")
        # Response template (no BOM, CSV).
        tpath = args.output_dir / "response_template_v2.csv"
        tpath.write_text(_build_response_template(manifest), encoding="utf-8")
        print(f"[OK] wrote {tpath}")
        # INSTRUCTIONS (with BOM).
        ipath = args.output_dir / "INSTRUCTIONS.md"
        ipath.write_text(_build_instructions(manifest), encoding="utf-8")
        print(f"[OK] wrote {ipath}")
        # Group counts.
        print(f"\n  Group counts: {manifest['group_counts']}")
        print(f"  Session 1: {len(manifest['sessions']['session1_pair_ids'])} pairs")
        print(f"  Session 2: {len(manifest['sessions']['session2_pair_ids'])} pairs")
    elif args.mode == "analyze":
        _analyze(args.manifest, args.response, args.output)


if __name__ == "__main__":
    main()
