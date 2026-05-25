"""ad-hoc: RL-1 imitation vs baselines paired test.

사용자 directive (2026-05-25): 4 평가 goal —
  1. synthetic 에서 oracle 의 multi-step sequence 를 얼마나 따라가는가
  2. G2 에서 STOP 을 얼마나 잘 배우는가
  3. B2/B5/B6/B7 보다 NetGain 이 좋아지는가
  4. oracle gap 을 얼마나 닫는가

본 도구는 RL-1 eval trial 9 (synthetic) + 15 (G2) 의 NetGain 을 다른 baseline 의
같은 trial_id NetGain 과 paired 비교 (Wilcoxon + Cohen's d + bootstrap CI).

CLI:
    python -m tools._paired_test_rl1_vs_baselines \\
        evals/snapshots/baseline_rl1_imitation_v1.json
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


def _extract_b2_multi(path: str) -> dict[str, float]:
    """B2 multi (synthetic) per-trial NetGain."""
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    return {r["trial_id"]: r["netgain_provisional"] for r in s["results"]}


def _extract_b2_g2(path: str) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    return {r["trial_id"]: r["netgain"] for r in s["results"]}


def _extract_oracle_synthetic(path: str) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    out = {}
    for r in s["per_sample"]:
        if r.get("best_A"):
            out[r["trial_id"]] = r["best_A"]["netgain_A"]
    return out


def _extract_oracle_g2(path: str) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    out = {}
    for r in s["per_sample"]:
        if r.get("best"):
            out[r["trial_id"]] = r["best"]["netgain"]
    return out


def main():
    rl1_path = sys.argv[1]
    with open(rl1_path, encoding="utf-8") as f:
        rl1 = json.load(f)
    rl1_syn = {r["trial_id"]: r["netgain_A"] for r in rl1["per_sample_synthetic"]}
    rl1_g2 = {r["trial_id"]: r["netgain"] for r in rl1["per_sample_g2"]}

    print(f"=== RL-1 imitation vs baselines — paired ===")
    print(f"  RL-1 eval synthetic: {len(rl1_syn)} trials")
    print(f"  RL-1 eval G2: {len(rl1_g2)} trials\n")

    # Synthetic comparisons — try v2 first (n=60), fall back to v1 (n=30).
    import os
    b2_v2_path = "evals/snapshots/baseline_b2_fixed_smoothing_multi_v2_n60.json"
    oracle_v2_path = "evals/snapshots/oracle_sequence_multi_v2_n60.json"
    b2_syn_path = b2_v2_path if os.path.exists(b2_v2_path) else "evals/snapshots/baseline_b2_fixed_smoothing_multi_v1.json"
    oracle_syn_path = oracle_v2_path if os.path.exists(oracle_v2_path) else "evals/snapshots/oracle_sequence_multi_v1.json"
    print(f"  synthetic B2: {b2_syn_path}")
    print(f"  synthetic oracle: {oracle_syn_path}")
    b2_syn = _extract_b2_multi(b2_syn_path)
    oracle_syn = _extract_oracle_synthetic(oracle_syn_path)

    common_syn_b2 = sorted(rl1_syn.keys() & b2_syn.keys())
    common_syn_oracle = sorted(rl1_syn.keys() & oracle_syn.keys())
    print(f"--- SYNTHETIC ---")
    print(f"  RL-1 ∩ B2: {len(common_syn_b2)} | RL-1 ∩ oracle: {len(common_syn_oracle)}")

    results = {}
    if common_syn_b2:
        a = np.array([rl1_syn[t] for t in common_syn_b2])
        b = np.array([b2_syn[t] for t in common_syn_b2])
        r = _paired(a, b)
        results["synthetic_rl1_vs_b2"] = r
        print(f"\n  RL-1 vs B2 (synthetic, n={r['n']}):")
        print(f"    RL-1 median={r['a_median']:+.5f}, mean={r['a_mean']:+.5f}")
        print(f"    B2   median={r['b_median']:+.5f}, mean={r['b_mean']:+.5f}")
        print(f"    Δ: median={r['diff_median']:+.5f}, mean={r['diff_mean']:+.5f}")
        print(f"    n_strict_RL1/n_tie/n_loss: {r['n_strict_a']}/{r['n_tie']}/{r['n_loss_a']}")
        print(f"    Wilcoxon p (greater): {r['wilcoxon_p_greater']:.5g} | two-sided: {r['wilcoxon_p_two_sided']:.5g}")
        print(f"    Cohen's d: {r['cohen_d_paired']:+.3f}")

    if common_syn_oracle:
        a = np.array([rl1_syn[t] for t in common_syn_oracle])
        b = np.array([oracle_syn[t] for t in common_syn_oracle])
        r = _paired(a, b)
        results["synthetic_rl1_vs_oracle_gap"] = r
        # Gap closure ratio.
        gap_init_to_b2 = (b - np.array([b2_syn.get(t, 0.0) for t in common_syn_oracle]))
        gap_rl1_to_b2 = (a - np.array([b2_syn.get(t, 0.0) for t in common_syn_oracle]))
        ratio = (gap_rl1_to_b2.mean() / gap_init_to_b2.mean()) if gap_init_to_b2.mean() != 0 else 0.0
        results["synthetic_oracle_gap_closure_ratio"] = float(ratio)
        print(f"\n  RL-1 vs sequence oracle (synthetic ceiling, n={r['n']}):")
        print(f"    RL-1 median={r['a_median']:+.5f}, oracle median={r['b_median']:+.5f}")
        print(f"    RL-1/oracle median ratio: {r['a_median']/r['b_median']*100 if r['b_median']!=0 else 0:.1f}%")
        print(f"    Δ (RL-1 - oracle): median={r['diff_median']:+.5f}, mean={r['diff_mean']:+.5f}")
        print(f"    Oracle gap closure ratio (RL-1/oracle gain vs B2): {ratio*100:.1f}%")

    # G2 comparisons.
    b2_g2 = _extract_b2_g2("evals/snapshots/baseline_b2_fixed_smoothing_g2_v2.json")
    oracle_g2 = _extract_oracle_g2("evals/snapshots/oracle_sequence_g2_v1.json")

    common_g2_b2 = sorted(rl1_g2.keys() & b2_g2.keys())
    common_g2_oracle = sorted(rl1_g2.keys() & oracle_g2.keys())
    print(f"\n--- G2 ---")
    print(f"  RL-1 ∩ B2 G2: {len(common_g2_b2)} | RL-1 ∩ oracle G2: {len(common_g2_oracle)}")

    if common_g2_b2:
        a = np.array([rl1_g2[t] for t in common_g2_b2])
        b = np.array([b2_g2[t] for t in common_g2_b2])
        r = _paired(a, b)
        results["g2_rl1_vs_b2"] = r
        print(f"\n  RL-1 vs B2 (G2, n={r['n']}):")
        print(f"    RL-1 median={r['a_median']:+.5f}, mean={r['a_mean']:+.5f}")
        print(f"    B2   median={r['b_median']:+.5f}, mean={r['b_mean']:+.5f}")
        print(f"    Δ: median={r['diff_median']:+.5f}, mean={r['diff_mean']:+.5f}")
        print(f"    n_strict_RL1/n_tie/n_loss: {r['n_strict_a']}/{r['n_tie']}/{r['n_loss_a']}")
        print(f"    Wilcoxon p (greater): {r['wilcoxon_p_greater']:.5g} | two-sided: {r['wilcoxon_p_two_sided']:.5g}")
        print(f"    Cohen's d: {r['cohen_d_paired']:+.3f}")

    if common_g2_oracle:
        a = np.array([rl1_g2[t] for t in common_g2_oracle])
        b = np.array([oracle_g2[t] for t in common_g2_oracle])
        r = _paired(a, b)
        results["g2_rl1_vs_oracle_gap"] = r
        gap_init_to_b2 = (b - np.array([b2_g2.get(t, 0.0) for t in common_g2_oracle]))
        gap_rl1_to_b2 = (a - np.array([b2_g2.get(t, 0.0) for t in common_g2_oracle]))
        ratio = (gap_rl1_to_b2.mean() / gap_init_to_b2.mean()) if gap_init_to_b2.mean() != 0 else 0.0
        results["g2_oracle_gap_closure_ratio"] = float(ratio)
        print(f"\n  RL-1 vs sequence oracle (G2 ceiling, n={r['n']}):")
        print(f"    RL-1 median={r['a_median']:+.5f}, oracle median={r['b_median']:+.5f}")
        print(f"    Δ (RL-1 - oracle): median={r['diff_median']:+.5f}, mean={r['diff_mean']:+.5f}")
        print(f"    Oracle gap closure ratio (RL-1/oracle gain vs B2): {ratio*100:.1f}%")

    # Goal 1 + 2 (action-level): synthetic length distribution + G2 STOP ratio.
    print(f"\n--- Goals 1 & 2 — action-level mimicry ---")
    syn_lens = [r["length"] for r in rl1["per_sample_synthetic"]]
    print(f"  Synthetic RL-1 length distribution: {dict(sorted({l: syn_lens.count(l) for l in set(syn_lens)}.items()))}")
    g2_first_stop = sum(1 for r in rl1["per_sample_g2"] if r["sequence"] and r["sequence"][0][0] == "STOP")
    g2_total = len(rl1["per_sample_g2"])
    print(f"  G2 RL-1 first_action=STOP: {g2_first_stop}/{g2_total} ({100*g2_first_stop/g2_total:.1f}%)")
    print(f"  (G2 oracle STOP-best ratio reference: 27/50 = 54%)")

    out_path = "evals/snapshots/_paired_rl1_vs_baselines.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
