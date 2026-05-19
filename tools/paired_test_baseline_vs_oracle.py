"""Paired Wilcoxon + Cohen's d + bootstrap CI for baseline (B1/B2/B3) vs oracle (B8).

H-2026-204 RQ1 (artifact-conditioned routing 이 fixed post-processing 보다 NetGain
우위) 의 통계 평가.

본 도구는:
  - baseline raw record (`baseline_fixed_pipeline_sample`) 에서 NetGain 추출.
  - oracle raw record (`oracle_single_step_sample`) 에서 best NetGain 추출.
  - same (trial, artifact) pair 로 paired Wilcoxon (greater alternative: oracle > baseline)
    + Cohen's d_paired + bootstrap 95% CI.

CLI 예:
    python -m tools.paired_test_baseline_vs_oracle \\
        --baseline-prefix baseline_b2_fixed_smoothing_v1 \\
        --oracle-prefix oracle_single_step_v2 \\
        --output evals/snapshots/paired_test_b2_vs_oracle_v1.json
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


BASELINE_RECORD_TYPES = (
    "baseline_fixed_pipeline_sample",
    "baseline_rule_based_sample",
)
ORACLE_RECORD_TYPES = (
    "oracle_single_step_sample",
    "oracle_sequence_sample",
)


def _load_netgains_any(raw_dir: Path, prefix: str) -> dict[tuple[str, str], float]:
    """raw record 의 record_type 에 따라 자동 분기.
       baseline 은 selections[art].netgain_provisional, oracle 은 best_candidate.netgain_provisional.
    """
    out: dict[tuple[str, str], float] = {}
    for p in sorted(raw_dir.glob(f"*{prefix}*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        rec_type = r.get("record_type")
        trial = r.get("trial_id")
        if rec_type in BASELINE_RECORD_TYPES:
            for art, sel in r.get("selections", {}).items():
                if "netgain_provisional" in sel:
                    out[(trial, art)] = float(sel["netgain_provisional"])
        elif rec_type in ORACLE_RECORD_TYPES:
            for art, sel in r.get("selections", {}).items():
                best = sel.get("best_candidate")
                if best is not None:
                    out[(trial, art)] = float(best["netgain_provisional"])
    return out


def _load_baseline_netgains(raw_dir: Path, prefix: str) -> dict[tuple[str, str], float]:
    return _load_netgains_any(raw_dir, prefix)


def _load_oracle_netgains(raw_dir: Path, prefix: str) -> dict[tuple[str, str], float]:
    return _load_netgains_any(raw_dir, prefix)


def _bootstrap_ci(a: np.ndarray, b: np.ndarray, n_iter: int = 1000, seed: int = 42) -> dict[str, float]:
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
    if diffs.std(ddof=1) < 1e-15:
        return float("inf") if diffs.mean() > 0 else float("-inf") if diffs.mean() < 0 else 0.0
    return float(diffs.mean() / diffs.std(ddof=1))


def paired_test(
    oracle_ng: dict[tuple[str, str], float],
    baseline_ng: dict[tuple[str, str], float],
) -> dict[str, Any]:
    common_keys = sorted(set(oracle_ng) & set(baseline_ng))
    by_art: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for k in common_keys:
        by_art[k[1]].append((oracle_ng[k], baseline_ng[k]))

    per_artifact: list[dict[str, Any]] = []
    all_or: list[float] = []
    all_bl: list[float] = []
    for artifact_kind, pairs in sorted(by_art.items()):
        or_arr = np.array([p[0] for p in pairs], dtype=np.float64)
        bl_arr = np.array([p[1] for p in pairs], dtype=np.float64)
        all_or.extend(or_arr.tolist())
        all_bl.extend(bl_arr.tolist())
        diffs = or_arr - bl_arr

        nonzero = diffs[np.abs(diffs) > 1e-12]
        if nonzero.size == 0:
            wilcoxon_stat = 0.0
            wilcoxon_p_two = 1.0
            wilcoxon_p_greater = 1.0
        else:
            try:
                w_two = stats.wilcoxon(or_arr, bl_arr, alternative="two-sided", zero_method="wilcox")
                w_greater = stats.wilcoxon(or_arr, bl_arr, alternative="greater", zero_method="wilcox")
                wilcoxon_stat = float(w_two.statistic)
                wilcoxon_p_two = float(w_two.pvalue)
                wilcoxon_p_greater = float(w_greater.pvalue)
            except ValueError:
                wilcoxon_stat = float("nan")
                wilcoxon_p_two = float("nan")
                wilcoxon_p_greater = float("nan")

        per_artifact.append({
            "artifact_kind": artifact_kind,
            "n_pairs": int(diffs.size),
            "oracle_netgain_mean": float(or_arr.mean()),
            "oracle_netgain_median": float(np.median(or_arr)),
            "baseline_netgain_mean": float(bl_arr.mean()),
            "baseline_netgain_median": float(np.median(bl_arr)),
            "delta_mean": float(diffs.mean()),
            "delta_median": float(np.median(diffs)),
            "delta_percent_median": (
                float(np.median(diffs) / abs(np.median(bl_arr)) * 100.0)
                if abs(np.median(bl_arr)) > 1e-12 else float("inf")
            ),
            "n_oracle_advantage": int((diffs > 1e-12).sum()),
            "n_tie": int((np.abs(diffs) <= 1e-12).sum()),
            "n_baseline_advantage": int((diffs < -1e-12).sum()),
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_p_two_sided": wilcoxon_p_two,
            "wilcoxon_p_greater_one_sided": wilcoxon_p_greater,
            "cohens_d_paired": _cohens_d_paired(diffs),
            "bootstrap_ci_95": _bootstrap_ci(or_arr, bl_arr),
        })

    all_or_arr = np.array(all_or)
    all_bl_arr = np.array(all_bl)
    all_diffs = all_or_arr - all_bl_arr
    try:
        w_all_two = stats.wilcoxon(all_or_arr, all_bl_arr, alternative="two-sided", zero_method="wilcox")
        w_all_greater = stats.wilcoxon(all_or_arr, all_bl_arr, alternative="greater", zero_method="wilcox")
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
        "n_oracle_advantage": int((all_diffs > 1e-12).sum()),
        "n_tie": int((np.abs(all_diffs) <= 1e-12).sum()),
        "n_baseline_advantage": int((all_diffs < -1e-12).sum()),
        "wilcoxon_statistic": w_all_stat,
        "wilcoxon_p_two_sided": w_all_p_two,
        "wilcoxon_p_greater_one_sided": w_all_p_greater,
        "cohens_d_paired": _cohens_d_paired(all_diffs),
        "bootstrap_ci_95": _bootstrap_ci(all_or_arr, all_bl_arr),
    }

    return {
        "schema_version": "1.0.0",
        "record_type": "paired_test_baseline_vs_oracle",
        "n_keys_compared": len(common_keys),
        "per_artifact": per_artifact,
        "pooled": pooled,
        "interpretation": {
            "test": "Wilcoxon signed-rank paired test. Alternative 'greater' = H1: oracle > baseline.",
            "effect_size_convention": "Cohen's d_paired = mean(oracle - baseline) / std. |d|>0.2 small, 0.5 medium, 0.8 large.",
            "h_2026_204_rq1_threshold": (
                "Oracle (B8) vs fixed smoothing (B2): NetGain median Delta >= +10% (명세 §11 Go 5). "
                "Rule-based (B5) vs fixed smoothing (B2): median Delta >= +5%, p<0.05, Cohen's d >= 0.3."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired Wilcoxon: baseline (B1/B2/B3) vs oracle (B8)"
    )
    parser.add_argument("--raw-records-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--baseline-prefix", type=str, required=True)
    parser.add_argument("--oracle-prefix", type=str, default="oracle_single_step_v2")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    baseline_ng = _load_baseline_netgains(args.raw_records_dir, args.baseline_prefix)
    oracle_ng = _load_oracle_netgains(args.raw_records_dir, args.oracle_prefix)
    summary = paired_test(oracle_ng, baseline_ng)
    summary["baseline_prefix"] = args.baseline_prefix
    summary["oracle_prefix"] = args.oracle_prefix

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OK] wrote to {args.output}")
        for ar in summary["per_artifact"]:
            print(
                f"  {ar['artifact_kind']:30s} | "
                f"n={ar['n_pairs']:3d} | "
                f"oracle={ar['oracle_netgain_median']:+.5f} - bl={ar['baseline_netgain_median']:+.5f} | "
                f"d_med={ar['delta_median']:+.5f} ({ar['delta_percent_median']:+.1f}%) | "
                f"p_gt={ar['wilcoxon_p_greater_one_sided']:.4g} | "
                f"d={ar['cohens_d_paired']:+.3f}"
            )
        p = summary["pooled"]
        print(
            f"  {'POOLED':30s} | "
            f"n={p['n_pairs']:3d} | "
            f"d_med={p['delta_median']:+.5f} | "
            f"p_gt={p['wilcoxon_p_greater_one_sided']:.4g} | "
            f"d={p['cohens_d_paired']:+.3f}"
        )
    else:
        print(text)


if __name__ == "__main__":
    main()
