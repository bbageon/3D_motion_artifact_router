"""Item 3: NetGain α threshold sensitivity 분석.

사용자 directive (2026-05-25 6 항목 Item 3):
> "Threshold sensitivity 분석 — calibrated_protocol_a_v1 의 α=5.0 의 robustness."

본 도구는 α (FidelityLoss weight) 의 다른 값에서 baseline ranking 의 변동 분석.
- α ∈ {0, 1, 3, 5, 10, 20} sweep.
- Synthetic 의 비교: B2-medium / B2-val-best / sequence oracle / RF / MLP / HGB.
- G2 의 비교: 동일.

각 α 에서:
  - Mean / median NetGain 계산.
  - Baseline rank order (변동 시 NetGain proxy 의 robustness 의문 제기).

본 분석은 controlled diagnostic — 단일 α 의 결과가 다른 α 에서 reverse 되는지
확인. NetGain proxy 의 robustness 진단.

NOTE: 다른 α 측정 은 per-sample 의 target_delta + fidelity_loss 만 있으면 가능
(NetGain = -target_delta - α·fidelity_loss). 기존 snapshot 의 per-sample 결과
재사용.

CLI:
    python -m tools.netgain_sensitivity_analysis --output evals/snapshots/netgain_sensitivity_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _recompute_netgain(target_delta: float, fidelity_loss: float, alpha: float,
                       correction_mag: float = 0.0, tool_calls: float = 0.0,
                       beta: float = 0.0, gamma: float = 0.0) -> float:
    return -target_delta - alpha * fidelity_loss - beta * correction_mag - gamma * tool_calls


def _aggregate_per_alpha(per_sample: list[dict[str, float]], alphas: list[float]) -> dict[float, dict[str, float]]:
    out = {}
    for a in alphas:
        ngs = [_recompute_netgain(s["target_delta"], s["fidelity_loss"], a) for s in per_sample]
        arr = np.array(ngs)
        out[a] = {"mean": float(arr.mean()), "median": float(np.median(arr)),
                  "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75))}
    return out


def _load_b2_family_synthetic(path: Path) -> dict[str, list[dict]]:
    """B2-family sweep 의 synthetic per-sample 결과 → variant 별 (target_delta, fidelity_loss)."""
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    out = {}
    for variant in ["small", "medium", "large", "val_best"]:
        out[f"B2-{variant.replace('_', '-')}"] = [
            {"trial_id": ps["trial_id"],
             "target_delta": ps["per_variant"][variant]["target_delta"],
             "fidelity_loss": ps["per_variant"][variant]["fidelity_loss"]}
            for ps in s["per_sample_synthetic"]
        ]
    return out


def _load_b2_family_g2(path: Path) -> dict[str, list[dict]]:
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    out = {}
    for variant in ["small", "medium", "large", "val_best"]:
        out[f"B2-{variant.replace('_', '-')}"] = [
            {"trial_id": ps["trial_id"],
             "target_delta": ps["per_variant"][variant]["target_delta"],
             "fidelity_loss": ps["per_variant"][variant]["fidelity_loss"]}
            for ps in s["per_sample_g2"]
        ]
    return out


def _load_oracle_synthetic(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    return [
        {"trial_id": r["trial_id"],
         "target_delta": r["best_A"]["target_delta_A"],
         "fidelity_loss": r["best_A"]["fidelity_loss_protocol_a"]}
        for r in s["per_sample"] if r.get("best_A")
    ]


def _load_oracle_g2(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    return [
        {"trial_id": r["trial_id"],
         "target_delta": r["best"]["target_delta_full"],
         "fidelity_loss": r["best"]["fidelity_loss_protocol_b"]}
        for r in s["per_sample"] if r.get("best")
    ]


def _load_rl1_g2_hgb(path: Path) -> list[dict]:
    """HGB G2 best policy 의 per-sample (target_delta, fidelity_loss)."""
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    cl = s["closed_loop_g2"]["per_sample"]
    out = []
    for r in cl:
        out.append({"trial_id": r["trial_id"],
                    "target_delta": r["target_delta_full"],
                    "fidelity_loss": r["fidelity_loss_protocol_b"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="NetGain α threshold sensitivity (Item 3)")
    parser.add_argument("--b2-family", type=Path, default=Path("evals/snapshots/baseline_b2_family_sweep_v1.json"))
    parser.add_argument("--oracle-synthetic", type=Path, default=Path("evals/snapshots/oracle_sequence_multi_v2_n60.json"))
    parser.add_argument("--oracle-g2", type=Path, default=Path("evals/snapshots/oracle_sequence_g2_v1.json"))
    parser.add_argument("--hgb-g2", type=Path, default=Path("evals/snapshots/hgb_g2_best_policy_v1.json"))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0, 3.0, 5.0, 10.0, 20.0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures/2026-05-25"))
    args = parser.parse_args()

    # === Synthetic baselines ===
    print("[INFO] Loading synthetic data...")
    syn_b2 = _load_b2_family_synthetic(args.b2_family)
    syn_oracle = _load_oracle_synthetic(args.oracle_synthetic)
    print(f"  B2 family: {[(k, len(v)) for k,v in syn_b2.items()]}")
    print(f"  Sequence oracle: n={len(syn_oracle)}")

    syn_data = {"sequence_oracle": syn_oracle}
    syn_data.update(syn_b2)

    syn_agg = {label: _aggregate_per_alpha(samples, args.alphas) for label, samples in syn_data.items()}

    # === G2 baselines ===
    print("\n[INFO] Loading G2 data...")
    g2_b2 = _load_b2_family_g2(args.b2_family)
    g2_oracle = _load_oracle_g2(args.oracle_g2)
    g2_hgb = _load_rl1_g2_hgb(args.hgb_g2)
    print(f"  B2 family: {[(k, len(v)) for k,v in g2_b2.items()]}")
    print(f"  Sequence oracle G2: n={len(g2_oracle)}")
    print(f"  HGB G2: n={len(g2_hgb)}")

    g2_data = {"sequence_oracle": g2_oracle, "HGB_RL1": g2_hgb}
    g2_data.update(g2_b2)

    g2_agg = {label: _aggregate_per_alpha(samples, args.alphas) for label, samples in g2_data.items()}

    # === Rank order analysis ===
    def _rank_order(agg: dict[str, dict[float, dict]], alpha: float) -> list[tuple[str, float]]:
        return sorted([(label, info[alpha]["median"]) for label, info in agg.items()],
                      key=lambda kv: kv[1], reverse=True)

    print(f"\n=== Synthetic ranking by NetGain median (per α) ===")
    syn_ranks = {}
    for a in args.alphas:
        order = _rank_order(syn_agg, a)
        syn_ranks[a] = order
        print(f"  α={a:5.1f}:", " > ".join(f"{lbl}({v:+.3f})" for lbl, v in order))

    print(f"\n=== G2 ranking by NetGain median (per α) ===")
    g2_ranks = {}
    for a in args.alphas:
        order = _rank_order(g2_agg, a)
        g2_ranks[a] = order
        print(f"  α={a:5.1f}:", " > ".join(f"{lbl}({v:+.3f})" for lbl, v in order))

    # Check rank stability (top-1 변동).
    syn_top1 = [order[0][0] for order in syn_ranks.values()]
    g2_top1 = [order[0][0] for order in g2_ranks.values()]
    syn_stable = len(set(syn_top1)) == 1
    g2_stable = len(set(g2_top1)) == 1
    print(f"\n=== Rank stability ===")
    print(f"  Synthetic top-1 stable across α: {syn_stable} (set={set(syn_top1)})")
    print(f"  G2 top-1 stable across α: {g2_stable} (set={set(g2_top1)})")

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for label, agg in syn_agg.items():
        meds = [agg[a]["median"] for a in args.alphas]
        axes[0].plot(args.alphas, meds, marker="o", label=label, linewidth=1.5)
    axes[0].set_xlabel("α (FidelityLoss weight)")
    axes[0].set_ylabel("NetGain median")
    axes[0].set_title("Synthetic — NetGain median vs α")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].axvline(5.0, color="red", linestyle="--", alpha=0.5, label="default α=5.0")

    for label, agg in g2_agg.items():
        meds = [agg[a]["median"] for a in args.alphas]
        axes[1].plot(args.alphas, meds, marker="o", label=label, linewidth=1.5)
    axes[1].set_xlabel("α (FidelityLoss weight)")
    axes[1].set_ylabel("NetGain median")
    axes[1].set_title("G2 — NetGain median vs α")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].axvline(5.0, color="red", linestyle="--", alpha=0.5)

    fig.suptitle("NetGain α threshold sensitivity — baselines vs proposed", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    fig_path = args.figure_dir / "netgain_sensitivity_v1.png"
    plt.savefig(str(fig_path), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[PNG] {fig_path}")

    summary = {
        "schema_version": "1.0.0", "record_type": "netgain_sensitivity_v1",
        "alphas_swept": args.alphas,
        "default_alpha": 5.0,
        "synthetic": {
            "per_baseline_per_alpha": {
                label: {str(a): agg[a] for a in args.alphas} for label, agg in syn_agg.items()
            },
            "rank_per_alpha": {str(a): [{"label": l, "median": v} for l, v in order]
                               for a, order in syn_ranks.items()},
            "top1_stable_across_alpha": syn_stable,
            "top1_set": list(set(syn_top1)),
        },
        "g2": {
            "per_baseline_per_alpha": {
                label: {str(a): agg[a] for a in args.alphas} for label, agg in g2_agg.items()
            },
            "rank_per_alpha": {str(a): [{"label": l, "median": v} for l, v in order]
                               for a, order in g2_ranks.items()},
            "top1_stable_across_alpha": g2_stable,
            "top1_set": list(set(g2_top1)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
