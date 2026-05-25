"""ad-hoc: Sequence oracle (Step RL-0) vs B2 multi paired test.

핵심 분기점 (사용자 directive 2026-05-24): "B2 를 넘는 action sequence 가 실제로
존재하는가?"

  - Sequence oracle > B2: multi-step headroom 존재 → multi-step RL 가치 있음.
  - Sequence oracle ≈ B2 또는 < B2: action space 천장 → tool/reward 재설계 필요.

본 도구는 oracle_sequence_multi_v1.json (Step RL-0) 과 B2 multi (baseline_b2_
fixed_smoothing_multi_v1.json) 의 per-sample NetGain 을 trial_id 로 paired join 후
Wilcoxon signed-rank + Cohen's d + bootstrap CI 통계 측정.

두 NetGain metric 모두 박제:
  - NetGain_A (target = mean(foot + jitter)): B2 multi 와 같은 metric.
  - NetGain_full (target = mean(all 3)): full evaluator metric.

CLI:
    python -m tools._paired_test_sequence_oracle_vs_b2 \\
        evals/snapshots/oracle_sequence_multi_v1.json \\
        evals/snapshots/baseline_b2_fixed_smoothing_multi_v1.json
"""
import json
import sys
import numpy as np
from scipy import stats


def _paired(arr_a: np.ndarray, arr_b: np.ndarray, label: str) -> dict:
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
        "label": label, "n": len(diff),
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

    # Build per-sample dicts.
    seq_per_sample_A = {}
    seq_per_sample_full = {}
    for s in seq["per_sample"]:
        if s["best_A"] is not None:
            seq_per_sample_A[s["trial_id"]] = s["best_A"]["netgain_A"]
        if s["best_full"] is not None:
            seq_per_sample_full[s["trial_id"]] = s["best_full"]["netgain_full"]

    # B2 multi per-sample.
    b2_per_sample = {}
    for s in b2.get("results", []):
        b2_per_sample[s["trial_id"]] = s["netgain_provisional"]

    common = sorted(seq_per_sample_A.keys() & b2_per_sample.keys())
    print(f"=== Sequence Oracle (Step RL-0) vs B2 multi — paired (n={len(common)}) ===")
    print(f"  seq input: {seq_path}")
    print(f"  B2 input:  {b2_path}\n")

    seq_A_arr = np.array([seq_per_sample_A[t] for t in common])
    seq_full_arr = np.array([seq_per_sample_full[t] for t in common])
    b2_arr = np.array([b2_per_sample[t] for t in common])

    # Comparison 1: Sequence oracle (NetGain_A) vs B2 (same metric).
    r_A = _paired(seq_A_arr, b2_arr, "seq_A_vs_B2")
    print(f"--- Sequence oracle (target=mean(foot+jitter)) > B2 multi (same metric) ---")
    print(f"  oracle median={r_A['a_median']:+.5f}, mean={r_A['a_mean']:+.5f}")
    print(f"  B2 median={r_A['b_median']:+.5f}, mean={r_A['b_mean']:+.5f}")
    print(f"  Δ (oracle - B2): median={r_A['diff_median']:+.5f}, mean={r_A['diff_mean']:+.5f}")
    print(f"  n_strict_oracle/n_tie/n_loss: {r_A['n_strict_a']}/{r_A['n_tie']}/{r_A['n_loss_a']}")
    print(f"  Wilcoxon p (greater): {r_A['wilcoxon_p_greater']:.5g} | two-sided: {r_A['wilcoxon_p_two_sided']:.5g}")
    print(f"  Cohen's d: {r_A['cohen_d_paired']:+.3f}")
    print(f"  bootstrap CI median: [{r_A['boot_ci_median'][0]:+.5f}, {r_A['boot_ci_median'][1]:+.5f}]")
    print(f"  bootstrap CI mean:   [{r_A['boot_ci_mean'][0]:+.5f}, {r_A['boot_ci_mean'][1]:+.5f}]")
    print()

    # Comparison 2: Sequence oracle (NetGain_full) vs B2 (caveat: different target metric).
    r_full = _paired(seq_full_arr, b2_arr, "seq_full_vs_B2")
    print(f"--- Sequence oracle (target=mean(all 3)) vs B2 multi (informational — different metric) ---")
    print(f"  oracle median={r_full['a_median']:+.5f}, mean={r_full['a_mean']:+.5f}")
    print(f"  B2 median={r_full['b_median']:+.5f}, mean={r_full['b_mean']:+.5f}")
    print(f"  Δ: median={r_full['diff_median']:+.5f}, mean={r_full['diff_mean']:+.5f}")
    print(f"  n_strict_oracle/n_tie/n_loss: {r_full['n_strict_a']}/{r_full['n_tie']}/{r_full['n_loss_a']}")
    print(f"  Wilcoxon p (greater): {r_full['wilcoxon_p_greater']:.5g}")
    print(f"  Cohen's d: {r_full['cohen_d_paired']:+.3f}")
    print()

    summary = {
        "n_common": len(common),
        "common_trial_ids": common,
        "seq_oracle_A_vs_b2": r_A,
        "seq_oracle_full_vs_b2_informational": r_full,
        "interpretation": {
            "RL_value_threshold_questioned": (
                "Sequence oracle > B2 (target=A metric)? "
                f"= {r_A['diff_median'] > 0 and r_A['wilcoxon_p_greater'] < 0.05}. "
                "If true: multi-step RL has headroom. "
                "If false: tool/reward redesign needed (not RL problem)."
            ),
            "oracle_A_vs_b2_median_gap": r_A["diff_median"],
            "oracle_A_vs_b2_wilcoxon_p_greater": r_A["wilcoxon_p_greater"],
            "oracle_A_vs_b2_cohen_d": r_A["cohen_d_paired"],
        },
    }
    out_path = "evals/snapshots/_paired_seq_oracle_vs_b2_multi.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
