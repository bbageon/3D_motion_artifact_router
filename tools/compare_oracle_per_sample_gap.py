"""Single-step vs Sequence oracle 의 per-sample gap (격차) 분석.

본 도구는 두 oracle snapshot 의 raw record 들을 읽어 **같은 sample × artifact**
의 best NetGain 을 비교하고, per-sample gap (= sequence - single_step) 의 분포
통계를 산출한다.

[H-2026-204](../evals/hypotheses/H-2026-204.md) (RQ1+RQ2) 의 RQ2 (closed-loop vs
single-step) 검증의 정확한 척도. **median of medians** (snapshot 의 median NetGain
끼리 비교) 는 per-sample 분포를 가리는 information loss 가 발생하므로 본 도구는
**per-sample gap** 을 명시 측정.

산출:
  - 각 (artifact_kind) 별 gap 분포 통계 (mean / median / min / max / quantile).
  - gap > 0 (strict 우위) count, gap == 0 (tie) count.

CLI 예:
    python -m tools.compare_oracle_per_sample_gap \\
        --single-step-prefix oracle_single_step_v2 \\
        --sequence-prefix oracle_sequence_v1_1_tiebreak \\
        --output evals/snapshots/oracle_sequence_vs_single_step_v1.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "evals" / "raw"


def _load_best_netgains(
    raw_dir: Path, prefix: str
) -> dict[tuple[str, str], float]:
    """raw record 들에서 (trial_id, artifact_kind) → best NetGain 추출.

    raw_record schema: oracle_single_step / oracle_sequence 모두 selections[artifact]
    .best_candidate.netgain_provisional 위치.
    """
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


def compute_per_sample_gaps(
    raw_dir: Path,
    single_step_prefix: str,
    sequence_prefix: str,
) -> dict[str, Any]:
    ss = _load_best_netgains(raw_dir, single_step_prefix)
    seq = _load_best_netgains(raw_dir, sequence_prefix)

    common_keys = sorted(set(ss) & set(seq))
    gaps_by_art: dict[str, list[float]] = defaultdict(list)
    for k in common_keys:
        gaps_by_art[k[1]].append(seq[k] - ss[k])

    per_artifact: list[dict[str, Any]] = []
    for artifact_kind, gaps in sorted(gaps_by_art.items()):
        arr = np.array(gaps, dtype=np.float64)
        per_artifact.append({
            "artifact_kind": artifact_kind,
            "n_samples": int(arr.size),
            "n_strict_advantage": int((arr > 0).sum()),
            "n_tie": int((arr == 0).sum()),
            "n_loss": int((arr < 0).sum()),  # 이론상 0 이어야 함 (수학적 lower bound).
            "gap": {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            },
        })

    return {
        "schema_version": "1.0.0",
        "record_type": "oracle_per_sample_gap_summary",
        "single_step_prefix": single_step_prefix,
        "sequence_prefix": sequence_prefix,
        "n_keys_compared": len(common_keys),
        "per_artifact": per_artifact,
        "interpretation": {
            "gap_definition": "sequence_best_netgain - single_step_best_netgain (per sample, per artifact)",
            "mathematical_lower_bound": (
                "gap >= 0 (sequence oracle 의 후보 set 이 single-step 의 superset, "
                "즉 sequence 의 max NetGain >= single-step 의 max NetGain)"
            ),
            "strict_advantage_meaning": (
                "n_strict_advantage = closed-loop refinement (반복 적용) 가 "
                "single-step 으로는 도달 불가능한 NetGain 을 달성한 sample 수"
            ),
            "tie_meaning": (
                "n_tie = sequence 의 best 가 length-1 sequence (= single-step) 와 "
                "동일 NetGain. closed-loop 의 추가 가치 없음"
            ),
            "h_2026_204_rq2_implication": (
                "RQ2 (closed-loop vs single-step) 의 정확한 척도는 본 per-sample "
                "gap 분포이며, 'closed-loop 가 항상 single-step 보다 낫다' 는 "
                "수학적으로 (gap >= 0) 성립하나 empirically n_strict_advantage / "
                "n_samples 비율로 판정해야 한다"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-sample gap (sequence - single_step) analysis"
    )
    parser.add_argument("--raw-records-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--single-step-prefix", type=str, default="oracle_single_step_v2")
    parser.add_argument("--sequence-prefix", type=str, default="oracle_sequence_v1_1_tiebreak")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = compute_per_sample_gaps(
        args.raw_records_dir, args.single_step_prefix, args.sequence_prefix
    )
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OK] wrote to {args.output}")
        for ar in summary["per_artifact"]:
            print(
                f"  {ar['artifact_kind']:30s} | "
                f"n={ar['n_samples']} | "
                f"strict_advantage={ar['n_strict_advantage']}, tie={ar['n_tie']}, loss={ar['n_loss']} | "
                f"gap median={ar['gap']['median']:+.5f}, max={ar['gap']['max']:+.5f}"
            )
    else:
        print(text)


if __name__ == "__main__":
    main()
