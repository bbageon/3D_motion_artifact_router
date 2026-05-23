"""ad-hoc: G2 small-calibration paired test (zero-shot vs calibrated, same eval set)."""
import json
import sys
import numpy as np
from scipy import stats

with open(sys.argv[1], encoding="utf-8") as f:
    s = json.load(f)

zs = s["per_sample_zero_shot"]
cal = s["per_sample_calibrated"]

# Same trial_id ordering (already aligned).
zs_by_trial = {r["trial_id"]: r for r in zs}
cal_by_trial = {r["trial_id"]: r for r in cal}
trials = sorted(zs_by_trial.keys() & cal_by_trial.keys())

zs_ng = np.array([zs_by_trial[t]["netgain"] for t in trials])
cal_ng = np.array([cal_by_trial[t]["netgain"] for t in trials])
diff = cal_ng - zs_ng

# Wilcoxon (calibrated > zero_shot).
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

print(f"n_pairs: {len(trials)}")
print(f"zero_shot NetGain median: {np.median(zs_ng):+.5f}, mean: {zs_ng.mean():+.5f}")
print(f"calibrated NetGain median: {np.median(cal_ng):+.5f}, mean: {cal_ng.mean():+.5f}")
print(f"diff (cal - zs) median: {np.median(diff):+.5f}, mean: {diff.mean():+.5f}")
print(f"n_strict_advantage_cal: {int((diff > 1e-12).sum())}/{len(diff)}")
print(f"n_tie: {int((np.abs(diff) <= 1e-12).sum())}")
print(f"n_loss_cal: {int((diff < -1e-12).sum())}")
print(f"paired Wilcoxon p (greater, cal > zs): {p_g:.5g}")
print(f"paired Wilcoxon p (two-sided): {p_two:.5g}")
print(f"Cohen's d_paired: {cohen_d:+.3f}")
print(f"bootstrap 95% CI median: [{np.percentile(boot_med, 2.5):+.5f}, {np.percentile(boot_med, 97.5):+.5f}]")
print(f"bootstrap 95% CI mean:   [{np.percentile(boot_mean, 2.5):+.5f}, {np.percentile(boot_mean, 97.5):+.5f}]")
