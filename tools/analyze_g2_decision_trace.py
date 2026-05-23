"""Step 5-A: G2 decision trace 정량 해부.

사용자 directive (2026-05-23): "G2 에서 왜 B5 가 가장 robust 했는지를 먼저 해부하는
게 좋다. B5 가 어떤 sample 에서 어떤 tool 을 골랐는지, BoneProjection 이 실제로 bone
score 를 줄였는지, B6 가 BP 를 많이 골랐는데 왜 bone reduction 이 제한적이었는지."

본 도구는 G2 raw record (Step 5) 에서 다음 정량 추출:
  1. **per-sample decision trace**: 각 sample 의 baseline 별 selected_tool/strength.
  2. **per-tool effectiveness**: 같은 tool/strength 를 적용한 sample 들의 bone/foot/jitter
     score reduction 분포.
  3. **B5 vs B6 same-tool comparison**: 같은 BP 적용된 sample 의 B5/B6 결과 차이.
  4. **B5 의 robust 원인 isolation**: small strength (B5 majority) vs large strength
     (B6 majority) 의 G2 motion 효과 차이.

CLI:
    python -m tools.analyze_g2_decision_trace \\
        --raw-records-dir evals/raw \\
        --output evals/snapshots/g2_decision_trace_analysis_v1.json
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

#: G2 baseline raw record prefixes.
G2_BASELINES = {
    "B1": "baseline_b1_no_refinement_g2_v1",
    "B2": "baseline_b2_fixed_smoothing_g2_v1",
    "B5": "baseline_b5_rule_based_g2_v1",
    "B6_single": "baseline_b6_single_g2_v1",
    "B6_cl": "baseline_b6_cl_g2_v1",
    "B7": "baseline_b7_bandit_g2_v1",
}

ALL_EVALUATORS = ["FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator"]


def _load_g2_records(raw_dir: Path, prefix: str) -> dict[str, dict[str, Any]]:
    """trial_id → record dict."""
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(raw_dir.glob(f"*{prefix}*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("record_type") != "baseline_g2_natural_sample":
            continue
        trial_id = r["trial_id"]
        sel = list(r["selections"].values())[0]
        out[trial_id] = {
            "trial_id": trial_id,
            "g2_prompt": sel.get("g2_prompt", ""),
            "motion_shape": sel.get("motion_shape"),
            "trace": sel.get("trace", {}),
            "target_score_before": sel.get("target_score_before"),
            "target_score_after": sel.get("target_score_after"),
            "target_delta": sel.get("target_delta"),
            "fidelity_loss_protocol_b": sel.get("fidelity_loss_protocol_b"),
            "correction_magnitude": sel.get("correction_magnitude"),
            "tool_call_count": sel.get("tool_call_count"),
            "netgain": sel.get("netgain"),
            "per_evaluator_before_max": sel.get("per_evaluator_before_max", {}),
            "per_evaluator_after_max": sel.get("per_evaluator_after_max", {}),
        }
    return out


def _extract_tool_strength(rec: dict[str, Any], baseline_key: str) -> tuple[str, str]:
    """trace 에서 (tool, strength) 추출. closed-loop 는 첫 step."""
    trace = rec["trace"]
    if "decision_trace" in trace:
        # closed-loop: list of decisions.
        steps = trace["decision_trace"]
        if not steps:
            return ("NONE", "n/a")
        first = steps[0]
        return (first.get("tool", "NONE") or "NONE", first.get("strength", "n/a") or "n/a")
    else:
        return (trace.get("tool") or "NONE", trace.get("strength", "n/a") or "n/a")


def _per_evaluator_delta(rec: dict[str, Any]) -> dict[str, float]:
    """before - after (positive = reduction)."""
    before = rec["per_evaluator_before_max"]
    after = rec["per_evaluator_after_max"]
    return {
        n: float(before.get(n, 0.0) - after.get(n, 0.0))
        for n in ALL_EVALUATORS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 decision trace 분석 (Step 5-A)")
    parser.add_argument("--raw-records-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Load all baselines.
    baseline_records: dict[str, dict[str, dict[str, Any]]] = {}
    for key, prefix in G2_BASELINES.items():
        records = _load_g2_records(args.raw_records_dir, prefix)
        baseline_records[key] = records
        print(f"[INFO] {key} ({prefix}): n={len(records)}")

    # Common trial_ids.
    common_trials = set.intersection(*[set(r.keys()) for r in baseline_records.values()])
    common_trials = sorted(common_trials)
    print(f"[INFO] common trials across all baselines: {len(common_trials)}")

    # --- Per-baseline tool/strength distribution ---
    per_baseline_dist: dict[str, dict[str, Any]] = {}
    for key, records in baseline_records.items():
        tool_count: defaultdict = defaultdict(int)
        strength_count: defaultdict = defaultdict(int)
        for trial in common_trials:
            tn, st = _extract_tool_strength(records[trial], key)
            tool_count[tn] += 1
            strength_count[st] += 1
        per_baseline_dist[key] = {
            "tool": dict(tool_count),
            "strength": dict(strength_count),
        }

    # --- Per-sample comparison: B5 vs B6_single vs B6_cl ---
    per_sample_comparison: list[dict[str, Any]] = []
    for trial in common_trials:
        row: dict[str, Any] = {
            "trial_id": trial,
            "g2_prompt": baseline_records["B5"][trial].get("g2_prompt", "")[:60],
            "evaluator_before": baseline_records["B1"][trial]["per_evaluator_before_max"],
        }
        for key in ["B2", "B5", "B6_single", "B6_cl", "B7"]:
            rec = baseline_records[key][trial]
            tn, st = _extract_tool_strength(rec, key)
            delta = _per_evaluator_delta(rec)
            row[key] = {
                "tool": tn, "strength": st,
                "netgain": float(rec["netgain"]) if rec["netgain"] is not None else None,
                "target_delta": rec["target_delta"],
                "fidelity_loss": rec["fidelity_loss_protocol_b"],
                "evaluator_reduction": delta,  # 각 evaluator 의 reduction (positive=감소).
            }
        per_sample_comparison.append(row)

    # --- Per-tool effectiveness aggregation ---
    # tool (B5 의 selection) 별로 bone/foot/jitter reduction 통계.
    per_tool_effectiveness_b5: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in common_trials:
        rec = baseline_records["B5"][trial]
        tn, st = _extract_tool_strength(rec, "B5")
        delta = _per_evaluator_delta(rec)
        per_tool_effectiveness_b5[(tn, st)].append({
            "trial_id": trial,
            "netgain": float(rec["netgain"]) if rec["netgain"] is not None else None,
            "evaluator_reduction": delta,
        })

    # B6_cl 도 같이.
    per_tool_effectiveness_b6cl: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in common_trials:
        rec = baseline_records["B6_cl"][trial]
        tn, st = _extract_tool_strength(rec, "B6_cl")
        delta = _per_evaluator_delta(rec)
        per_tool_effectiveness_b6cl[(tn, st)].append({
            "trial_id": trial,
            "netgain": float(rec["netgain"]) if rec["netgain"] is not None else None,
            "evaluator_reduction": delta,
        })

    def _summarize_tool_eff(eff_dict: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for (tn, st), records in eff_dict.items():
            arr_netgain = np.array([r["netgain"] for r in records if r["netgain"] is not None])
            foot_red = np.array([r["evaluator_reduction"]["FootFloatingEvaluator"] for r in records])
            bone_red = np.array([r["evaluator_reduction"]["BoneLengthEvaluator"] for r in records])
            jitter_red = np.array([r["evaluator_reduction"]["VelocityJitterEvaluator"] for r in records])
            out[f"{tn}/{st}"] = {
                "n_samples": len(records),
                "netgain_median": float(np.median(arr_netgain)) if len(arr_netgain) else None,
                "netgain_mean": float(arr_netgain.mean()) if len(arr_netgain) else None,
                "foot_reduction_median": float(np.median(foot_red)),
                "bone_reduction_median": float(np.median(bone_red)),
                "jitter_reduction_median": float(np.median(jitter_red)),
                "trial_ids": [r["trial_id"] for r in records],
            }
        return out

    # --- B5 vs B6_cl same-tool sample comparison ---
    # 같은 sample 에 B5 와 B6_cl 가 어떤 tool 골랐는지, 그리고 같은 tool 골랐을 때 NetGain 차이.
    same_tool_comparison: list[dict[str, Any]] = []
    different_tool_comparison: list[dict[str, Any]] = []
    for trial in common_trials:
        b5_rec = baseline_records["B5"][trial]
        b6_rec = baseline_records["B6_cl"][trial]
        b5_tool, b5_str = _extract_tool_strength(b5_rec, "B5")
        b6_tool, b6_str = _extract_tool_strength(b6_rec, "B6_cl")
        row = {
            "trial_id": trial,
            "B5_tool": b5_tool, "B5_strength": b5_str,
            "B6_cl_tool": b6_tool, "B6_cl_strength": b6_str,
            "B5_netgain": b5_rec["netgain"],
            "B6_cl_netgain": b6_rec["netgain"],
            "B5_bone_reduction": _per_evaluator_delta(b5_rec)["BoneLengthEvaluator"],
            "B6_cl_bone_reduction": _per_evaluator_delta(b6_rec)["BoneLengthEvaluator"],
        }
        if b5_tool == b6_tool:
            same_tool_comparison.append(row)
        else:
            different_tool_comparison.append(row)

    # --- Build summary ---
    summary = {
        "schema_version": "1.0.0",
        "record_type": "g2_decision_trace_analysis",
        "n_common_trials": len(common_trials),
        "per_baseline_distribution": per_baseline_dist,
        "per_tool_effectiveness_B5": _summarize_tool_eff(per_tool_effectiveness_b5),
        "per_tool_effectiveness_B6_cl": _summarize_tool_eff(per_tool_effectiveness_b6cl),
        "n_same_tool_B5_vs_B6_cl": len(same_tool_comparison),
        "n_different_tool_B5_vs_B6_cl": len(different_tool_comparison),
        "same_tool_comparison": same_tool_comparison,
        "different_tool_comparison": different_tool_comparison,
        "per_sample_comparison": per_sample_comparison,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"\n[OK] wrote {args.output}")

    # Console summary print.
    print(f"\n=== per-baseline (tool, strength) distribution ===")
    for key, dist in per_baseline_dist.items():
        print(f"  {key}: tools={dist['tool']}, strengths={dist['strength']}")

    print(f"\n=== B5 의 selected (tool, strength) per-tool 효과 ===")
    for k, v in sorted(_summarize_tool_eff(per_tool_effectiveness_b5).items()):
        print(f"  {k:35s} n={v['n_samples']:3d} | NetGain_med={v['netgain_median']:+.5f} | "
              f"foot_red={v['foot_reduction_median']:+.4f} | bone_red={v['bone_reduction_median']:+.4f} | "
              f"jitter_red={v['jitter_reduction_median']:+.4f}")

    print(f"\n=== B6_cl 의 selected (tool, strength) per-tool 효과 (first step) ===")
    for k, v in sorted(_summarize_tool_eff(per_tool_effectiveness_b6cl).items()):
        print(f"  {k:35s} n={v['n_samples']:3d} | NetGain_med={v['netgain_median']:+.5f} | "
              f"foot_red={v['foot_reduction_median']:+.4f} | bone_red={v['bone_reduction_median']:+.4f} | "
              f"jitter_red={v['jitter_reduction_median']:+.4f}")

    print(f"\n=== B5 vs B6_cl tool selection diff ===")
    print(f"  same tool selected: {len(same_tool_comparison)}")
    print(f"  different tool selected: {len(different_tool_comparison)}")
    if different_tool_comparison:
        print(f"  sample of different selections (first 5):")
        for row in different_tool_comparison[:5]:
            print(f"    {row['trial_id']}: B5={row['B5_tool']}/{row['B5_strength']} ng={row['B5_netgain']:+.4f} | "
                  f"B6_cl={row['B6_cl_tool']}/{row['B6_cl_strength']} ng={row['B6_cl_netgain']:+.4f}")


if __name__ == "__main__":
    main()
