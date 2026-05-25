"""Item 2: 3 분포 분리 분석 — HumanML3D clean / G2 natural / synthetic corrupted.

사용자 directive (2026-05-25 6-항목 피드백, Item 2):
> "HumanML3D clean / G2 natural / synthetic 분포 를 분리 보고."

본 도구는 3 분포 의 evaluator score (foot/bone/jitter max) distribution 측정 +
histogram + summary statistics. AGENTS.md §3-17 framing 정정 (synthetic 의
corruption 강도 vs G2 natural 의 정량 차이) 의 evidence.

3 분포:
  1. HumanML3D clean — new_joints/*.npy (random 100 sample).
  2. G2 natural — g2_generated_v1/motion_*.npy (50 sample).
  3. Synthetic corrupted — HumanML3D clean + multi-artifact recipe (foot+jitter).

CLI:
    python -m tools.distribution_analysis \\
        --n-humanml3d 100 --seed 42 \\
        --output evals/snapshots/distribution_analysis_v1.json \\
        --figure-dir reports/figures/2026-05-25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluators import DEFAULT_EVALUATORS, EvaluatorReport
from tools.synthetic_injection import inject_foot_floating, inject_jitter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMANML3D_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
DEFAULT_G2_DIR = REPO_ROOT / "external_assets" / "g2_generated_v1"

ALL_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _eval_scores(motion: np.ndarray, evaluators: list[Any]) -> dict[str, float]:
    return {n: _max_score(ev.evaluate(motion)) for n, ev in zip(ALL_EVALUATORS, evaluators)}


def _multi_inject(clean: np.ndarray, seed: int) -> np.ndarray:
    m1 = inject_foot_floating(clean, lift_height=0.08, seed=seed)
    return inject_jitter(m1, noise_std=0.05, seed=seed + 1000)


def _measure_distribution(
    motions: list[tuple[str, np.ndarray]], evaluators: list[Any], label: str,
) -> dict[str, Any]:
    per_sample = []
    for trial_id, motion in motions:
        scores = _eval_scores(motion, evaluators)
        per_sample.append({"trial_id": trial_id, **scores})
    n = len(per_sample)
    summary = {"label": label, "n_samples": n, "per_sample": per_sample}
    for ev_name in ALL_EVALUATORS:
        arr = np.array([s[ev_name] for s in per_sample])
        summary[ev_name] = {
            "mean": float(arr.mean()), "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max()),
            "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75)),
        }
    return summary


def _plot_histograms(
    dists: dict[str, dict[str, Any]], save_path: Path,
) -> None:
    """3 분포 의 evaluator score histogram + boxplot."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = {"humanml3d_clean": "tab:green", "g2_natural": "tab:orange", "synthetic_corrupted": "tab:red"}

    for col, ev_name in enumerate(ALL_EVALUATORS):
        # Top row: histograms.
        ax = axes[0, col]
        for label, dist in dists.items():
            arr = np.array([s[ev_name] for s in dist["per_sample"]])
            ax.hist(arr, bins=30, alpha=0.55, label=f"{label} (n={len(arr)})", color=colors.get(label, "gray"))
        ax.set_title(f"{ev_name}\nhistogram")
        ax.set_xlabel("max score")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Bottom row: boxplot.
        ax2 = axes[1, col]
        data = []
        labels = []
        box_colors = []
        for label, dist in dists.items():
            arr = np.array([s[ev_name] for s in dist["per_sample"]])
            data.append(arr)
            labels.append(f"{label}\n(n={len(arr)})")
            box_colors.append(colors.get(label, "gray"))
        bp = ax2.boxplot(data, labels=labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        ax2.set_title(f"{ev_name}\nboxplot (log scale)")
        ax2.set_ylabel("max score")
        # Use log scale if min > 0.
        all_pos = all(arr.min() > 0 for arr in data)
        if all_pos:
            ax2.set_yscale("log")
        ax2.grid(alpha=0.3)

    fig.suptitle("Distribution analysis — HumanML3D clean / G2 natural / synthetic corrupted", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG] {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="3 분포 분리 분석 (Item 2)")
    parser.add_argument("--humanml3d-dir", type=Path, default=DEFAULT_HUMANML3D_DIR)
    parser.add_argument("--g2-dir", type=Path, default=DEFAULT_G2_DIR)
    parser.add_argument("--n-humanml3d", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures/2026-05-25"))
    args = parser.parse_args()

    evaluators = list(DEFAULT_EVALUATORS)

    # === Distribution 1: HumanML3D clean (random N samples) ===
    rng = np.random.default_rng(args.seed)
    npy_files = sorted(args.humanml3d_dir.glob("*.npy"))
    chosen_idx = rng.choice(len(npy_files), size=args.n_humanml3d, replace=False)
    print(f"[INFO] HumanML3D clean: {args.n_humanml3d} samples...")
    clean_motions = []
    clean_arrays = []  # for synthetic.
    for i, idx in enumerate(chosen_idx):
        path = npy_files[idx]
        motion = np.load(str(path)).astype(np.float64)
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            continue
        clean_motions.append((path.stem, motion))
        clean_arrays.append((path.stem, motion))
        if (i + 1) % 25 == 0:
            print(f"  [HumanML3D] {i+1}/{args.n_humanml3d}")
    dist_clean = _measure_distribution(clean_motions, evaluators, "humanml3d_clean")

    # === Distribution 2: G2 natural ===
    print(f"\n[INFO] G2 natural samples...")
    g2_files = sorted(args.g2_dir.glob("motion_*.npy"))
    g2_motions = []
    for path in g2_files:
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        motion = np.load(str(path)).astype(np.float64)
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            continue
        g2_motions.append((meta.get("trial_id", path.stem), motion))
    print(f"  G2: {len(g2_motions)} samples")
    dist_g2 = _measure_distribution(g2_motions, evaluators, "g2_natural")

    # === Distribution 3: Synthetic corrupted (clean + multi-inject) ===
    print(f"\n[INFO] Synthetic corrupted samples...")
    syn_motions = []
    for trial_id, clean in clean_arrays:
        corrupted = _multi_inject(clean, seed=args.seed)
        syn_motions.append((trial_id, corrupted))
    dist_synthetic = _measure_distribution(syn_motions, evaluators, "synthetic_corrupted")
    print(f"  synthetic corrupted: {len(syn_motions)} samples")

    # === Comparison ===
    print(f"\n=== 3 distribution summary (max score by evaluator) ===")
    for ev_name in ALL_EVALUATORS:
        print(f"\n[{ev_name}]")
        for label, dist in [("humanml3d_clean", dist_clean), ("g2_natural", dist_g2),
                            ("synthetic_corrupted", dist_synthetic)]:
            s = dist[ev_name]
            print(f"  {label:24s} (n={dist['n_samples']:3d}): median={s['median']:.5f}, "
                  f"mean={s['mean']:.5f}, p25={s['p25']:.5f}, p75={s['p75']:.5f}")

    # Magnitude ratio: synthetic vs G2.
    print(f"\n=== Synthetic vs G2 magnitude ratio (median) ===")
    for ev_name in ALL_EVALUATORS:
        g2_med = dist_g2[ev_name]["median"]
        syn_med = dist_synthetic[ev_name]["median"]
        ratio = (syn_med / g2_med) if g2_med > 1e-12 else float("inf")
        print(f"  {ev_name:30s}: synthetic/g2 ratio = {ratio:.2f}x")

    # Save snapshot.
    summary = {
        "schema_version": "1.0.0",
        "record_type": "distribution_analysis",
        "task_id": "distribution_analysis_v1",
        "seed": args.seed,
        "n_humanml3d": args.n_humanml3d,
        "evaluator_names": list(ALL_EVALUATORS),
        "distributions": {
            "humanml3d_clean": dist_clean,
            "g2_natural": dist_g2,
            "synthetic_corrupted": dist_synthetic,
        },
        "synthetic_vs_g2_magnitude_ratio_median": {
            ev: (dist_synthetic[ev]["median"] / dist_g2[ev]["median"]
                 if dist_g2[ev]["median"] > 1e-12 else None)
            for ev in ALL_EVALUATORS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {args.output}")

    # Plot.
    _plot_histograms(
        {"humanml3d_clean": dist_clean, "g2_natural": dist_g2,
         "synthetic_corrupted": dist_synthetic},
        args.figure_dir / "distribution_analysis_v1.png",
    )


if __name__ == "__main__":
    main()
