"""ad-hoc: 3-level vs 5-level Sequence Oracle 비교 분석.

사용자 directive (2026-05-25 Step 6): "3-level vs 5-level 분석. RL-2 action
space 결정 의 정량 evidence (Case A/B/C 분기)."

본 도구는 두 oracle snapshot (3-level + 5-level) 의 per-sample NetGain 의 paired
test + 분포 비교 + RL-2 결정 분기 기준 적용.

비교 지표 (사용자 directive):
  - NetGain median / mean (paired)
  - fidelity loss
  - correction magnitude
  - tool call count (best length)
  - STOP rate
  - strength distribution (3-level vs 5-level)
  - oracle gap vs B2-family-best

CLI:
    # Synthetic 비교:
    python -m tools._compare_3level_5level_oracle \\
        --3level evals/snapshots/oracle_sequence_multi_v2_n60.json \\
        --5level evals/snapshots/oracle_sequence_multi_5level_v1.json \\
        --b2-family evals/snapshots/baseline_b2_family_sweep_v1.json \\
        --domain synthetic \\
        --output evals/snapshots/compare_3level_5level_synthetic.json

    # G2 비교:
    python -m tools._compare_3level_5level_oracle \\
        --3level evals/snapshots/oracle_sequence_g2_v1.json \\
        --5level evals/snapshots/oracle_sequence_g2_natural_5level_v1.json \\
        --b2-family evals/snapshots/baseline_b2_family_sweep_v1.json \\
        --domain g2 \\
        --output evals/snapshots/compare_3level_5level_g2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def _paired(arr_a: np.ndarray, arr_b: np.ndarray) -> dict:
    diff = arr_a - arr_b
    nonzero = diff[np.abs(diff) > 1e-12]
    if nonzero.size > 0:
        w_g = stats.wilcoxon(arr_a, arr_b, alternative="greater", zero_method="wilcox")
        w_two = stats.wilcoxon(arr_a, arr_b, alternative="two-sided", zero_method="wilcox")
        p_g, p_two = float(w_g.pvalue), float(w_two.pvalue)
    else:
        p_g, p_two = 1.0, 1.0
    d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 1e-15 else 0.0
    rng = np.random.default_rng(42)
    boot_med = np.empty(1000); boot_mean = np.empty(1000)
    for i in range(1000):
        idx = rng.integers(0, diff.size, size=diff.size)
        boot_med[i] = np.median(diff[idx]); boot_mean[i] = diff[idx].mean()
    return {
        "n": len(diff),
        "a_median": float(np.median(arr_a)), "a_mean": float(arr_a.mean()),
        "b_median": float(np.median(arr_b)), "b_mean": float(arr_b.mean()),
        "diff_median": float(np.median(diff)), "diff_mean": float(diff.mean()),
        "n_strict_a": int((diff > 1e-12).sum()), "n_tie": int((np.abs(diff) <= 1e-12).sum()),
        "n_loss_a": int((diff < -1e-12).sum()),
        "wilcoxon_p_greater": p_g, "wilcoxon_p_two_sided": p_two,
        "cohen_d_paired": d,
        "boot_ci_median": [float(np.percentile(boot_med, 2.5)), float(np.percentile(boot_med, 97.5))],
        "boot_ci_mean": [float(np.percentile(boot_mean, 2.5)), float(np.percentile(boot_mean, 97.5))],
    }


def _extract_per_sample(snap_path: Path, domain: str) -> dict[str, dict[str, Any]]:
    """oracle snapshot → trial_id → {netgain, length, fidelity_loss, correction_mag, ...}."""
    with open(snap_path, encoding="utf-8") as f:
        s = json.load(f)
    out = {}
    for ps in s["per_sample"]:
        tid = ps["trial_id"]
        if domain == "synthetic":
            best = ps.get("best_A") or ps.get("best")
            if best is None: continue
            out[tid] = {
                "netgain": best.get("netgain_A", best.get("netgain", 0.0)),
                "length": best.get("length", 0),
                "fidelity_loss": best.get("fidelity_loss_protocol_a", 0.0),
                "correction_mag": best.get("cumulative_correction_magnitude", 0.0),
                "first_action": best["sequence"][0][0] if best.get("sequence") else "STOP",
                "first_strength": best["sequence"][0][2] if best.get("sequence") else "n/a",
                "sequence": best.get("sequence", []),
            }
        else:  # g2
            best = ps.get("best")
            if best is None: continue
            out[tid] = {
                "netgain": best.get("netgain", 0.0),
                "length": best.get("length", 0),
                "fidelity_loss": best.get("fidelity_loss_protocol_b", 0.0),
                "correction_mag": best.get("cumulative_correction_magnitude", 0.0),
                "first_action": best["sequence"][0][0] if best.get("sequence") else "STOP",
                "first_strength": best["sequence"][0][2] if best.get("sequence") else "n/a",
                "sequence": best.get("sequence", []),
            }
    return out


def _b2_val_best_per_sample(sweep_path: Path, domain: str) -> dict[str, float]:
    with open(sweep_path, encoding="utf-8") as f:
        s = json.load(f)
    key = "per_sample_synthetic" if domain == "synthetic" else "per_sample_g2"
    return {ps["trial_id"]: ps["per_variant"]["val_best"]["netgain"] for ps in s.get(key, [])}


def main() -> None:
    parser = argparse.ArgumentParser(description="3-level vs 5-level Sequence Oracle 비교")
    parser.add_argument("--three-level", type=Path, required=True, dest="three_level",
                        help="3-level oracle snapshot")
    parser.add_argument("--five-level", type=Path, required=True, dest="five_level",
                        help="5-level oracle snapshot")
    parser.add_argument("--b2-family", type=Path, required=True)
    parser.add_argument("--domain", choices=["synthetic", "g2"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"[INFO] domain: {args.domain}")
    s3 = _extract_per_sample(args.three_level, args.domain)
    s5 = _extract_per_sample(args.five_level, args.domain)
    print(f"  3-level: {len(s3)} samples")
    print(f"  5-level: {len(s5)} samples")

    common = sorted(s3.keys() & s5.keys())
    print(f"  common: {len(common)} samples")
    if not common:
        print("[WARN] no common samples — comparison skipped"); return

    # NetGain paired (5-level vs 3-level).
    ng_5 = np.array([s5[t]["netgain"] for t in common])
    ng_3 = np.array([s3[t]["netgain"] for t in common])
    netgain_paired = _paired(ng_5, ng_3)

    # Fidelity loss paired (lower is better → 5-level - 3-level, negative = 5-level 가 더 작음).
    fl_5 = np.array([s5[t]["fidelity_loss"] for t in common])
    fl_3 = np.array([s3[t]["fidelity_loss"] for t in common])
    fl_paired = _paired(fl_5, fl_3)

    # Correction magnitude paired.
    cm_5 = np.array([s5[t]["correction_mag"] for t in common])
    cm_3 = np.array([s3[t]["correction_mag"] for t in common])
    cm_paired = _paired(cm_5, cm_3)

    # Best length comparison.
    len_5_dist = Counter(s5[t]["length"] for t in common)
    len_3_dist = Counter(s3[t]["length"] for t in common)
    len_5_mean = float(np.mean([s5[t]["length"] for t in common]))
    len_3_mean = float(np.mean([s3[t]["length"] for t in common]))

    # STOP rate.
    stop_5 = sum(1 for t in common if s5[t]["length"] == 0)
    stop_3 = sum(1 for t in common if s3[t]["length"] == 0)

    # First-action distribution.
    first_action_5 = Counter(s5[t]["first_action"] for t in common)
    first_action_3 = Counter(s3[t]["first_action"] for t in common)

    # First-strength distribution.
    first_strength_5 = Counter(s5[t]["first_strength"] for t in common)
    first_strength_3 = Counter(s3[t]["first_strength"] for t in common)

    # B2-val-best gap closure.
    b2_val = _b2_val_best_per_sample(args.b2_family, args.domain)
    common_b2 = [t for t in common if t in b2_val]
    if common_b2:
        b2_arr = np.array([b2_val[t] for t in common_b2])
        ng_5_b2 = np.array([s5[t]["netgain"] for t in common_b2])
        ng_3_b2 = np.array([s3[t]["netgain"] for t in common_b2])
        gap_5 = float((ng_5_b2 - b2_arr).mean())
        gap_3 = float((ng_3_b2 - b2_arr).mean())
    else:
        gap_5 = gap_3 = None

    # === Print summary ===
    print(f"\n=== NetGain paired (5-level vs 3-level, n={netgain_paired['n']}) ===")
    print(f"  5-level median={netgain_paired['a_median']:+.5f}, mean={netgain_paired['a_mean']:+.5f}")
    print(f"  3-level median={netgain_paired['b_median']:+.5f}, mean={netgain_paired['b_mean']:+.5f}")
    print(f"  Delta (5-3) median={netgain_paired['diff_median']:+.5f}, mean={netgain_paired['diff_mean']:+.5f}")
    print(f"  n_5-strict / n_tie / n_loss: {netgain_paired['n_strict_a']} / {netgain_paired['n_tie']} / {netgain_paired['n_loss_a']}")
    print(f"  Wilcoxon p (5 > 3): {netgain_paired['wilcoxon_p_greater']:.5g}")
    print(f"  Cohen's d (paired): {netgain_paired['cohen_d_paired']:+.3f}")
    print(f"  Boot CI median: [{netgain_paired['boot_ci_median'][0]:+.5f}, {netgain_paired['boot_ci_median'][1]:+.5f}]")

    print(f"\n=== Fidelity loss paired (5-level - 3-level, lower=better) ===")
    print(f"  Delta median={fl_paired['diff_median']:+.5f}, mean={fl_paired['diff_mean']:+.5f}")
    print(f"  (negative = 5-level fidelity loss 가 더 작음)")

    print(f"\n=== Correction magnitude paired ===")
    print(f"  Delta median={cm_paired['diff_median']:+.5f}, mean={cm_paired['diff_mean']:+.5f}")

    print(f"\n=== Best length distribution ===")
    print(f"  5-level: {dict(sorted(len_5_dist.items()))} | mean={len_5_mean:.2f}")
    print(f"  3-level: {dict(sorted(len_3_dist.items()))} | mean={len_3_mean:.2f}")

    print(f"\n=== STOP rate ===")
    print(f"  5-level: {stop_5}/{len(common)} ({100*stop_5/len(common):.1f}%)")
    print(f"  3-level: {stop_3}/{len(common)} ({100*stop_3/len(common):.1f}%)")

    print(f"\n=== First action distribution ===")
    print(f"  5-level: {dict(first_action_5)}")
    print(f"  3-level: {dict(first_action_3)}")

    print(f"\n=== First strength distribution ===")
    print(f"  5-level: {dict(first_strength_5)}")
    print(f"  3-level: {dict(first_strength_3)}")

    if gap_5 is not None:
        print(f"\n=== Oracle gap vs B2-val-best (mean, n={len(common_b2)}) ===")
        print(f"  5-level gap: {gap_5:+.5f}")
        print(f"  3-level gap: {gap_3:+.5f}")
        print(f"  Delta (5-3): {gap_5 - gap_3:+.5f}")

    # RL-2 결정 분기 판단.
    branch = "TBD"
    branch_reasoning = ""
    if netgain_paired["wilcoxon_p_greater"] < 0.05 and netgain_paired["cohen_d_paired"] > 0.2:
        branch = "A: 5-level 의미 있게 우월 → RL-2 = 5-level"
        branch_reasoning = f"Wilcoxon p (5>3) = {netgain_paired['wilcoxon_p_greater']:.5g} < 0.05, Cohen's d = {netgain_paired['cohen_d_paired']:+.3f} > 0.2"
    elif fl_paired["diff_mean"] < -0.001 or cm_paired["diff_mean"] < -0.001:
        branch = "B: NetGain 비슷, fidelity/correction 감소 → 5-level (over-mod 감소용)"
        branch_reasoning = f"NetGain non-significant, fidelity Δmean = {fl_paired['diff_mean']:+.5f}, correction Δmean = {cm_paired['diff_mean']:+.5f}"
    elif abs(netgain_paired["diff_median"]) < 0.005 and abs(netgain_paired["cohen_d_paired"]) < 0.2:
        branch = "C: 차이 거의 없음 → 3-level 유지, 5-level appendix"
        branch_reasoning = f"NetGain Δmedian = {netgain_paired['diff_median']:+.5f} (negligible), Cohen's d = {netgain_paired['cohen_d_paired']:+.3f}"
    else:
        branch = "Inconclusive — manual inspection 필요"
        branch_reasoning = f"NetGain Δmedian = {netgain_paired['diff_median']:+.5f}, p = {netgain_paired['wilcoxon_p_greater']:.5g}, d = {netgain_paired['cohen_d_paired']:+.3f}"

    print(f"\n=== RL-2 action space 결정 분기 (사용자 directive Case A/B/C) ===")
    print(f"  Branch: {branch}")
    print(f"  Reasoning: {branch_reasoning}")

    summary = {
        "schema_version": "1.0.0",
        "record_type": "compare_3level_5level_oracle",
        "domain": args.domain,
        "three_level_source": str(args.three_level),
        "five_level_source": str(args.five_level),
        "b2_family_source": str(args.b2_family),
        "n_common": len(common),
        "netgain_paired_5_vs_3": netgain_paired,
        "fidelity_loss_paired_5_minus_3": fl_paired,
        "correction_magnitude_paired_5_minus_3": cm_paired,
        "best_length": {
            "five_level_dist": dict(sorted(len_5_dist.items())), "five_level_mean": len_5_mean,
            "three_level_dist": dict(sorted(len_3_dist.items())), "three_level_mean": len_3_mean,
        },
        "stop_rate": {
            "five_level": {"count": stop_5, "ratio": stop_5 / len(common)},
            "three_level": {"count": stop_3, "ratio": stop_3 / len(common)},
        },
        "first_action_dist": {
            "five_level": dict(first_action_5), "three_level": dict(first_action_3),
        },
        "first_strength_dist": {
            "five_level": dict(first_strength_5), "three_level": dict(first_strength_3),
        },
        "oracle_gap_vs_b2_val_best_mean": {
            "five_level": gap_5, "three_level": gap_3,
            "delta_5_minus_3": (gap_5 - gap_3) if gap_5 is not None and gap_3 is not None else None,
            "n_b2_common": len(common_b2) if common_b2 else 0,
        },
        "rl2_action_space_branch": branch,
        "rl2_branch_reasoning": branch_reasoning,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
