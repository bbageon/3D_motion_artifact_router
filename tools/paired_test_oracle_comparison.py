"""Paired Wilcoxon + Cohen's d + bootstrap CI on sequence vs single_step oracle.

H-2026-204 RQ2 (closed-loop ≥ single-step) 의 정식 통계 평가.

본 도구는 oracle_single_step_v2 와 oracle_sequence_v1_1_tiebreak 의 same (trial,
artifact) pairs 의 NetGain 을 비교, paired Wilcoxon signed-rank · Cohen's d · bootstrap
95% CI 산출. eval-compare SKILL §5 의 통계 절차에 부합.

CLI 예:
    python -m tools.paired_test_oracle_comparison \\
        --single-step-prefix oracle_single_step_v2 \\
        --sequence-prefix oracle_sequence_v1_1_tiebreak \\
        --output evals/snapshots/paired_test_oracle_v1.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "evals" / "raw"


def _load_best_netgains(
    raw_dir: Path, prefix: str
) -> dict[tuple[str, str], float]:
    """raw record 에서 (trial_id, artifact_kind) → best NetGain."""
    out: dict[tuple[str, str], float] = {}
    for p in sorted(raw_dir.glob(f"*{prefix}*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("record_type") not in (
            "oracle_single_step_sample",
            "oracle_sequence_sample",
        ):
            continue
        trial = r["trial_id"]
        for art, sel in r.get("selections", {}).items():
            best = sel.get("best_candidate")
            if best is not None:
                out[(trial, art)] = float(best["netgain_provisional"])
    return out


def _bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_iter: int = 1000, seed: int = 42
) -> dict[str, float]:
    """Paired bootstrap CI for median(a - b) and mean(a - b)."""
    rng = np.random.default_rng(seed)
    n = a.size
    diffs = a - b
    med_boot = np.empty(n_iter)
    mean_boot = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        d_sample = diffs[idx]
        med_boot[i] = float(np.median(d_sample))
        mean_boot[i] = float(d_sample.mean())
    return {
        "median_ci_lo_95": float(np.percentile(med_boot, 2.5)),
        "median_ci_hi_95": float(np.percentile(med_boot, 97.5)),
        "mean_ci_lo_95": float(np.percentile(mean_boot, 2.5)),
        "mean_ci_hi_95": float(np.percentile(mean_boot, 97.5)),
    }


def _cohens_d_paired(diffs: np.ndarray) -> float:
    """Paired Cohen's d = mean(diff) / std(diff). 음수면 effect direction 반대."""
    if diffs.std(ddof=1) < 1e-15:
        return float("inf") if diffs.mean() > 0 else float("-inf") if diffs.mean() < 0 else 0.0
    return float(diffs.mean() / diffs.std(ddof=1))


def paired_test(
    seq_ng: dict[tuple[str, str], float],
    ss_ng: dict[tuple[str, str], float],
) -> dict[str, Any]:
    common_keys = sorted(set(seq_ng) & set(ss_ng))
    by_art: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for k in common_keys:
        by_art[k[1]].append((seq_ng[k], ss_ng[k]))

    per_artifact: list[dict[str, Any]] = []
    all_seq: list[float] = []
    all_ss: list[float] = []
    for artifact_kind, pairs in sorted(by_art.items()):
        seq_arr = np.array([p[0] for p in pairs], dtype=np.float64)
        ss_arr = np.array([p[1] for p in pairs], dtype=np.float64)
        all_seq.extend(seq_arr.tolist())
        all_ss.extend(ss_arr.tolist())
        diffs = seq_arr - ss_arr

        # Paired Wilcoxon signed-rank (one-sided: seq > ss).
        # 모든 diff = 0 인 경우 scipy 가 ValueError → catch.
        nonzero = diffs[np.abs(diffs) > 1e-12]
        if nonzero.size == 0:
            wilcoxon_stat = 0.0
            wilcoxon_p_two = 1.0
            wilcoxon_p_greater = 1.0
        else:
            try:
                w_two = stats.wilcoxon(seq_arr, ss_arr, alternative="two-sided", zero_method="wilcox")
                w_greater = stats.wilcoxon(seq_arr, ss_arr, alternative="greater", zero_method="wilcox")
                wilcoxon_stat = float(w_two.statistic)
                wilcoxon_p_two = float(w_two.pvalue)
                wilcoxon_p_greater = float(w_greater.pvalue)
            except ValueError as e:
                wilcoxon_stat = float("nan")
                wilcoxon_p_two = float("nan")
                wilcoxon_p_greater = float("nan")

        d = _cohens_d_paired(diffs)
        ci = _bootstrap_ci(seq_arr, ss_arr)

        per_artifact.append({
            "artifact_kind": artifact_kind,
            "n_pairs": int(diffs.size),
            "sequence_netgain_mean": float(seq_arr.mean()),
            "sequence_netgain_median": float(np.median(seq_arr)),
            "single_step_netgain_mean": float(ss_arr.mean()),
            "single_step_netgain_median": float(np.median(ss_arr)),
            "delta_mean": float(diffs.mean()),
            "delta_median": float(np.median(diffs)),
            "delta_percent_median": (
                float(np.median(diffs) / abs(np.median(ss_arr)) * 100.0)
                if abs(np.median(ss_arr)) > 1e-12 else float("inf")
            ),
            "n_strict_advantage": int((diffs > 1e-12).sum()),
            "n_tie": int((np.abs(diffs) <= 1e-12).sum()),
            "n_loss": int((diffs < -1e-12).sum()),
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_p_two_sided": wilcoxon_p_two,
            "wilcoxon_p_greater_one_sided": wilcoxon_p_greater,
            "cohens_d_paired": d,
            "bootstrap_ci_95": ci,
        })

    # Pooled across artifacts.
    all_seq_arr = np.array(all_seq)
    all_ss_arr = np.array(all_ss)
    all_diffs = all_seq_arr - all_ss_arr
    try:
        w_all_two = stats.wilcoxon(all_seq_arr, all_ss_arr, alternative="two-sided", zero_method="wilcox")
        w_all_greater = stats.wilcoxon(all_seq_arr, all_ss_arr, alternative="greater", zero_method="wilcox")
        w_all_stat = float(w_all_two.statistic)
        w_all_p_two = float(w_all_two.pvalue)
        w_all_p_greater = float(w_all_greater.pvalue)
    except ValueError:
        w_all_stat = float("nan")
        w_all_p_two = float("nan")
        w_all_p_greater = float("nan")

    pooled = {
        "n_pairs": int(all_diffs.size),
        "delta_mean": float(all_diffs.mean()),
        "delta_median": float(np.median(all_diffs)),
        "n_strict_advantage": int((all_diffs > 1e-12).sum()),
        "n_tie": int((np.abs(all_diffs) <= 1e-12).sum()),
        "n_loss": int((all_diffs < -1e-12).sum()),
        "wilcoxon_statistic": w_all_stat,
        "wilcoxon_p_two_sided": w_all_p_two,
        "wilcoxon_p_greater_one_sided": w_all_p_greater,
        "cohens_d_paired": _cohens_d_paired(all_diffs),
        "bootstrap_ci_95": _bootstrap_ci(all_seq_arr, all_ss_arr),
    }

    return {
        "schema_version": "1.0.0",
        "record_type": "paired_test_oracle_comparison",
        "n_keys_compared": len(common_keys),
        "per_artifact": per_artifact,
        "pooled": pooled,
        "interpretation": {
            "test": "Wilcoxon signed-rank paired test. Alternative 'greater' = H1: sequence > single_step.",
            "effect_size_convention": (
                "Cohen's d_paired = mean(seq - ss) / std(seq - ss). |d| > 0.2 small, "
                "0.5 medium, 0.8 large (Cohen 1988)."
            ),
            "ci_method": "Paired bootstrap (N=1000) of median and mean differences, percentile method.",
            "h_2026_204_rq2_threshold": (
                "NetGain median Delta >= +3% on artifact-rich input, paired Wilcoxon p < 0.05, "
                "Cohen's d >= 0.1 (보수적). 본 측정은 calibration_v1 (synthetic injection Protocol A)."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired Wilcoxon + Cohen's d + bootstrap CI (sequence vs single_step oracle)"
    )
    parser.add_argument("--raw-records-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--single-step-prefix", type=str, default="oracle_single_step_v2")
    parser.add_argument("--sequence-prefix", type=str, default="oracle_sequence_v1_1_tiebreak")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ss = _load_best_netgains(args.raw_records_dir, args.single_step_prefix)
    seq = _load_best_netgains(args.raw_records_dir, args.sequence_prefix)
    summary = paired_test(seq, ss)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OK] wrote to {args.output}")
        for ar in summary["per_artifact"]:
            print(
                f"  {ar['artifact_kind']:30s} | "
                f"n={ar['n_pairs']:3d} | "
                f"d_med={ar['delta_median']:+.5f} ({ar['delta_percent_median']:+.1f}%) | "
                f"p_gt={ar['wilcoxon_p_greater_one_sided']:.4f} | "
                f"d={ar['cohens_d_paired']:+.3f}"
            )
        p = summary["pooled"]
        print(
            f"  {'POOLED':30s} | "
            f"n={p['n_pairs']:3d} | "
            f"d_med={p['delta_median']:+.5f} | "
            f"p_gt={p['wilcoxon_p_greater_one_sided']:.4f} | "
            f"d={p['cohens_d_paired']:+.3f}"
        )
    else:
        print(text)


if __name__ == "__main__":
    main()
