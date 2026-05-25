"""ad-hoc: G2 sequence oracle (Step RL-0 보강) vs B2 G2 paired test.

사용자 directive (2026-05-25 roadmap): "Order 1: G2 natural sequence oracle K=5 —
G2에서도 sequence headroom 있는지 확인"

본 도구는 oracle_sequence_g2_v1.json 과 baseline_b2_fixed_smoothing_g2_v2.json 의
per-sample NetGain 을 trial_id 로 paired join 후 Wilcoxon signed-rank + Cohen's d
+ bootstrap CI 통계 측정.

CLI:
    python -m tools._paired_test_g2_sequence_vs_b2 \\
        evals/snapshots/oracle_sequence_g2_v1.json \\
        evals/snapshots/baseline_b2_fixed_smoothing_g2_v2.json
"""
import json
import sys
import numpy as np
from scipy import stats


def _paired(arr_a: np.ndarray, arr_b: np.ndarray) -> dict:
    diff = arr_a - arr_b
    nonzero = diff[np.abs(diff) > 1e-12]
    if nonzero.size > 0:
        w_g = stats.wilcoxon(arr_a, arr_b, alternative="greater", zero_method="wilcox")
        w_two = stats.wilcoxon(arr_a, arr_b, alternative="two-sided", zero_method="wilcox")
        p_g = float(w_g.pvalue); p_two = float(w_two.pvalue)
    else:
        p_g = 1.0; p_two = 1.0
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


def main():
    seq_path, b2_path = sys.argv[1], sys.argv[2]
    with open(seq_path, encoding="utf-8") as f:
        seq = json.load(f)
    with open(b2_path, encoding="utf-8") as f:
        b2 = json.load(f)

    seq_by_trial = {}
    for s in seq["per_sample"]:
        if s["best"] is not None:
            seq_by_trial[s["trial_id"]] = s["best"]["netgain"]

    b2_by_trial = {}
    for r in b2["results"]:
        b2_by_trial[r["trial_id"]] = r["netgain"]

    common = sorted(seq_by_trial.keys() & b2_by_trial.keys())
    print(f"=== G2 Sequence Oracle (Step RL-0 G2) vs B2 G2 — paired (n={len(common)}) ===")
    print(f"  seq input: {seq_path}")
    print(f"  B2 input:  {b2_path}\n")

    seq_arr = np.array([seq_by_trial[t] for t in common])
    b2_arr = np.array([b2_by_trial[t] for t in common])

    r = _paired(seq_arr, b2_arr)
    print(f"--- Sequence oracle (G2 natural) > B2 G2 ---")
    print(f"  oracle median={r['a_median']:+.5f}, mean={r['a_mean']:+.5f}")
    print(f"  B2 median={r['b_median']:+.5f}, mean={r['b_mean']:+.5f}")
    print(f"  Δ (oracle - B2): median={r['diff_median']:+.5f}, mean={r['diff_mean']:+.5f}")
    print(f"  n_strict_oracle/n_tie/n_loss: {r['n_strict_a']}/{r['n_tie']}/{r['n_loss_a']}")
    print(f"  Wilcoxon p (greater): {r['wilcoxon_p_greater']:.5g} | two-sided: {r['wilcoxon_p_two_sided']:.5g}")
    print(f"  Cohen's d: {r['cohen_d_paired']:+.3f}")
    print(f"  bootstrap CI median: [{r['boot_ci_median'][0]:+.5f}, {r['boot_ci_median'][1]:+.5f}]")
    print(f"  bootstrap CI mean:   [{r['boot_ci_mean'][0]:+.5f}, {r['boot_ci_mean'][1]:+.5f}]")

    # Best length / first action distribution.
    print(f"\n--- Sequence oracle best-path distribution ---")
    seq_per_sample = {s["trial_id"]: s for s in seq["per_sample"] if s["best"] is not None}
    from collections import Counter
    lengths = Counter(seq_per_sample[t]["best"]["length"] for t in common)
    first_actions = Counter((seq_per_sample[t]["best"]["sequence"][0][0] if seq_per_sample[t]["best"]["length"] >= 1 else "STOP") for t in common)
    print(f"  best_length: {dict(lengths)}")
    print(f"  first_action: {dict(first_actions)}")
    stop_count = lengths.get(0, 0)
    print(f"  STOP-best (do-nothing) ratio: {stop_count}/{len(common)} ({100*stop_count/len(common):.1f}%)")

    summary = {
        "n_common": len(common),
        "common_trial_ids": common,
        "paired": r,
        "best_length_freq": dict(lengths),
        "first_action_freq": dict(first_actions),
        "stop_best_ratio": f"{stop_count}/{len(common)}",
        "interpretation": {
            "g2_RL_value_threshold_questioned": (
                f"G2 sequence oracle > B2 G2? = {r['diff_median'] > 0 and r['wilcoxon_p_greater'] < 0.05}. "
                f"Δ median = {r['diff_median']:+.5f}, p = {r['wilcoxon_p_greater']:.5g}, d = {r['cohen_d_paired']:+.3f}. "
                "If True: G2 에서도 multi-step headroom 존재. If False/marginal: G2 natural 의 over-modification 한계."
            ),
        },
    }
    out_path = "evals/snapshots/_paired_g2_seq_oracle_vs_b2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
