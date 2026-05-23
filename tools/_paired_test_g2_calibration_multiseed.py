"""ad-hoc: G2 small-calibration multi-seed paired test (v2).

3 model_random_state seeds × same eval split (seed=42, train_ratio=0.4, n_eval=30).
For each seed: paired Wilcoxon (calibrated > zero_shot), Cohen's d, bootstrap CI.
Aggregate: per-seed result + variance.

CLI:
    python -m tools._paired_test_g2_calibration_multiseed \\
        evals/snapshots/baseline_b6_g2_calibrated_v2_seed1.json \\
        evals/snapshots/baseline_b6_g2_calibrated_v2_seed2.json \\
        evals/snapshots/baseline_b6_g2_calibrated_v2_seed3.json
"""
import json
import sys
import numpy as np
from scipy import stats


def _paired_stats(zs_ng: np.ndarray, cal_ng: np.ndarray, seed_label: str) -> dict:
    diff = cal_ng - zs_ng
    nonzero = diff[np.abs(diff) > 1e-12]
    if nonzero.size > 0:
        w_greater = stats.wilcoxon(cal_ng, zs_ng, alternative="greater", zero_method="wilcox")
        w_two = stats.wilcoxon(cal_ng, zs_ng, alternative="two-sided", zero_method="wilcox")
        p_g = float(w_greater.pvalue)
        p_two = float(w_two.pvalue)
    else:
        p_g = 1.0
        p_two = 1.0
    cohen_d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 1e-15 else 0.0
    rng = np.random.default_rng(42)
    boot_med = np.empty(1000)
    boot_mean = np.empty(1000)
    for i in range(1000):
        idx = rng.integers(0, diff.size, size=diff.size)
        boot_med[i] = np.median(diff[idx])
        boot_mean[i] = diff[idx].mean()
    return {
        "seed_label": seed_label,
        "n_pairs": len(zs_ng),
        "zs_ng_median": float(np.median(zs_ng)),
        "zs_ng_mean": float(zs_ng.mean()),
        "cal_ng_median": float(np.median(cal_ng)),
        "cal_ng_mean": float(cal_ng.mean()),
        "diff_median": float(np.median(diff)),
        "diff_mean": float(diff.mean()),
        "n_strict_cal": int((diff > 1e-12).sum()),
        "n_tie": int((np.abs(diff) <= 1e-12).sum()),
        "n_loss_cal": int((diff < -1e-12).sum()),
        "wilcoxon_p_greater": p_g,
        "wilcoxon_p_two_sided": p_two,
        "cohen_d_paired": cohen_d,
        "boot_ci_median": [float(np.percentile(boot_med, 2.5)), float(np.percentile(boot_med, 97.5))],
        "boot_ci_mean": [float(np.percentile(boot_mean, 2.5)), float(np.percentile(boot_mean, 97.5))],
    }


def main() -> None:
    paths = sys.argv[1:]
    print(f"=== Multi-seed paired test (G2 small-calibration v2) ===")
    print(f"  inputs: {paths}\n")
    per_seed = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        zs = s["per_sample_zero_shot"]
        cal = s["per_sample_calibrated"]
        zs_by_trial = {r["trial_id"]: r for r in zs}
        cal_by_trial = {r["trial_id"]: r for r in cal}
        trials = sorted(zs_by_trial.keys() & cal_by_trial.keys())
        zs_ng = np.array([zs_by_trial[t]["netgain"] for t in trials])
        cal_ng = np.array([cal_by_trial[t]["netgain"] for t in trials])
        seed_label = s.get("task_id", p)
        result = _paired_stats(zs_ng, cal_ng, seed_label)
        per_seed.append(result)
        print(f"--- {seed_label} (n={result['n_pairs']}) ---")
        print(f"  ZS NetGain: median={result['zs_ng_median']:+.5f}, mean={result['zs_ng_mean']:+.5f}")
        print(f"  CAL NetGain: median={result['cal_ng_median']:+.5f}, mean={result['cal_ng_mean']:+.5f}")
        print(f"  Δ (CAL-ZS): median={result['diff_median']:+.5f}, mean={result['diff_mean']:+.5f}")
        print(f"  n_strict_cal/n_tie/n_loss: {result['n_strict_cal']}/{result['n_tie']}/{result['n_loss_cal']}")
        print(f"  Wilcoxon p (greater): {result['wilcoxon_p_greater']:.5g} | two-sided: {result['wilcoxon_p_two_sided']:.5g}")
        print(f"  Cohen's d: {result['cohen_d_paired']:+.3f}")
        print(f"  boot CI median: [{result['boot_ci_median'][0]:+.5f}, {result['boot_ci_median'][1]:+.5f}]")
        print(f"  boot CI mean:   [{result['boot_ci_mean'][0]:+.5f}, {result['boot_ci_mean'][1]:+.5f}]")
        print()

    # Cross-seed aggregate.
    diff_medians = np.array([r["diff_median"] for r in per_seed])
    diff_means = np.array([r["diff_mean"] for r in per_seed])
    p_values = np.array([r["wilcoxon_p_greater"] for r in per_seed])
    cohen_ds = np.array([r["cohen_d_paired"] for r in per_seed])
    n_strict = np.array([r["n_strict_cal"] for r in per_seed])
    print(f"=== Cross-seed aggregate ({len(per_seed)} model_random_state) ===")
    print(f"  Δ median: {diff_medians.mean():+.5f} ± {diff_medians.std(ddof=1):.5f} (range [{diff_medians.min():+.5f}, {diff_medians.max():+.5f}])")
    print(f"  Δ mean:   {diff_means.mean():+.5f} ± {diff_means.std(ddof=1):.5f} (range [{diff_means.min():+.5f}, {diff_means.max():+.5f}])")
    print(f"  Wilcoxon p (greater): mean={p_values.mean():.5g}, range [{p_values.min():.5g}, {p_values.max():.5g}], all < 0.05: {bool((p_values < 0.05).all())}")
    print(f"  Cohen's d: {cohen_ds.mean():+.3f} ± {cohen_ds.std(ddof=1):.3f} (range [{cohen_ds.min():+.3f}, {cohen_ds.max():+.3f}])")
    print(f"  n_strict_cal/n_pairs: {n_strict.mean():.1f}/{per_seed[0]['n_pairs']} (across seeds: {list(n_strict)})")

    # Combined Fisher's method (for joint p-value across seeds, treating seeds as independent — note: same eval split, so paired diffs are correlated; this is informational).
    chi2 = -2 * np.sum(np.log(p_values))
    df = 2 * len(per_seed)
    fisher_p = float(stats.chi2.sf(chi2, df))
    print(f"  Fisher's combined p (informational, seeds not strictly independent): chi2={chi2:.3f}, df={df}, p={fisher_p:.5g}")

    summary = {
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
        "aggregate": {
            "diff_median_mean": float(diff_medians.mean()),
            "diff_median_std": float(diff_medians.std(ddof=1)),
            "diff_median_range": [float(diff_medians.min()), float(diff_medians.max())],
            "diff_mean_mean": float(diff_means.mean()),
            "diff_mean_std": float(diff_means.std(ddof=1)),
            "diff_mean_range": [float(diff_means.min()), float(diff_means.max())],
            "wilcoxon_p_mean": float(p_values.mean()),
            "wilcoxon_p_range": [float(p_values.min()), float(p_values.max())],
            "wilcoxon_p_all_lt_005": bool((p_values < 0.05).all()),
            "cohen_d_mean": float(cohen_ds.mean()),
            "cohen_d_std": float(cohen_ds.std(ddof=1)),
            "cohen_d_range": [float(cohen_ds.min()), float(cohen_ds.max())],
            "n_strict_cal_per_seed": [int(x) for x in n_strict],
            "fisher_combined_p_informational": fisher_p,
            "fisher_combined_chi2": float(chi2),
        },
    }
    out_path = "evals/snapshots/baseline_b6_g2_calibrated_v2_multiseed_paired.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
