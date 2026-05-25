"""B2-family sweep — 사용자 directive (2026-05-25, AGENTS.md §3-18).

B2-family 4 variant 측정:
  B2-small / B2-medium / B2-large / B2-val-best.

B2-val-best = 각 sample 의 NetGain best strength (sample-level oracle within
fixed smoothing family). fixed smoothing 의 진짜 ceiling.

대상: synthetic multi-artifact (n=60, same split as oracle_sequence_multi_v2_n60)
+ G2 natural (n=50).

NetGain:
  - Synthetic: Protocol A (FidelityLoss = MPJPE(refined, clean_gt) -
    MPJPE(corrupted, clean_gt)). target = mean(foot + jitter) (multi recipe).
  - G2: Protocol B simplified (FidelityLoss = MPJPE(refined, original_g2)).
    target = mean(all 3 evaluators).

α=5.0 (calibrated_protocol_a_v1).

CLI:
    python -m tools.baseline_b2_family_sweep \\
        --output evals/snapshots/baseline_b2_family_sweep_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from correction_tools import VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, EvaluatorReport
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from tools.synthetic_injection import inject_foot_floating, inject_jitter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMANML3D_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
DEFAULT_G2_DIR = REPO_ROOT / "external_assets" / "g2_generated_v1"

ALL_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
SYN_TARGET = ("FootFloatingEvaluator", "VelocityJitterEvaluator")
STRENGTHS = ("small", "medium", "large")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _eval_scores(motion: np.ndarray, evaluators: list[Any]) -> dict[str, float]:
    return {n: _max_score(ev.evaluate(motion)) for n, ev in zip(ALL_EVALUATORS, evaluators)}


def _multi_inject(clean: np.ndarray, seed: int) -> np.ndarray:
    m1 = inject_foot_floating(clean, lift_height=0.08, seed=seed)
    return inject_jitter(m1, noise_std=0.05, seed=seed + 1000)


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _apply_b2(motion: np.ndarray, strength: str, vs: VelocitySmoothingTool) -> np.ndarray:
    T = motion.shape[0]
    out, _ = vs.apply(motion, target_part="full_body", target_joints=[],
                       frame_range=(0, T - 1), strength=strength)
    return out


def _compute_synthetic_netgain(
    clean: np.ndarray, corrupted: np.ndarray, refined: np.ndarray,
    evaluators: list[Any], alpha: float,
) -> dict[str, float]:
    scores_init = _eval_scores(corrupted, evaluators)
    scores_final = _eval_scores(refined, evaluators)
    target_init = float(np.mean([scores_init[n] for n in SYN_TARGET]))
    target_final = float(np.mean([scores_final[n] for n in SYN_TARGET]))
    target_delta = target_final - target_init
    fidelity_loss = _mpjpe(refined, clean) - _mpjpe(corrupted, clean)
    return {
        "target_delta": target_delta, "fidelity_loss": fidelity_loss,
        "netgain": -target_delta - alpha * fidelity_loss,
    }


def _compute_g2_netgain(
    original_g2: np.ndarray, refined: np.ndarray, evaluators: list[Any], alpha: float,
) -> dict[str, float]:
    scores_init = _eval_scores(original_g2, evaluators)
    scores_final = _eval_scores(refined, evaluators)
    target_init = float(np.mean([scores_init[n] for n in ALL_EVALUATORS]))
    target_final = float(np.mean([scores_final[n] for n in ALL_EVALUATORS]))
    target_delta = target_final - target_init
    fidelity_loss = _mpjpe(refined, original_g2)
    return {
        "target_delta": target_delta, "fidelity_loss": fidelity_loss,
        "netgain": -target_delta - alpha * fidelity_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B2-family sweep (small/medium/large/val-best)")
    parser.add_argument("--humanml3d-dir", type=Path, default=DEFAULT_HUMANML3D_DIR)
    parser.add_argument("--g2-dir", type=Path, default=DEFAULT_G2_DIR)
    parser.add_argument("--n-syn-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
    alpha = float(netgain_weights["alpha"])
    vs = VelocitySmoothingTool()
    evaluators = list(DEFAULT_EVALUATORS)

    # === Synthetic multi-artifact (n=60, same split as oracle_sequence_multi_v2_n60) ===
    rng = np.random.default_rng(args.seed)
    npy_files = sorted(args.humanml3d_dir.glob("*.npy"))
    chosen_idx = rng.choice(len(npy_files), size=args.n_syn_samples, replace=False)
    chosen_syn = [npy_files[i] for i in chosen_idx]
    print(f"[INFO] synthetic samples: {len(chosen_syn)}")

    syn_per_sample = []
    for i, path in enumerate(chosen_syn, 1):
        trial_id = path.stem
        clean = np.load(str(path)).astype(np.float64)
        if clean.ndim != 3 or clean.shape[1] != 22 or clean.shape[2] != 3:
            continue
        corrupted = _multi_inject(clean, seed=args.seed)
        per_strength = {}
        for st in STRENGTHS:
            refined = _apply_b2(corrupted, st, vs)
            ng = _compute_synthetic_netgain(clean, corrupted, refined, evaluators, alpha)
            per_strength[st] = ng
        # val-best = best NetGain across strengths.
        best_strength = max(STRENGTHS, key=lambda s: per_strength[s]["netgain"])
        per_strength["val_best"] = dict(per_strength[best_strength])
        per_strength["val_best"]["best_strength"] = best_strength
        syn_per_sample.append({"trial_id": trial_id, "per_variant": per_strength})
        if i % 20 == 0:
            print(f"  [SYN] {i}/{len(chosen_syn)} done")

    # === G2 natural (n=50, all motion_*.npy in batch) ===
    g2_files = sorted(args.g2_dir.glob("motion_*.npy"))
    print(f"[INFO] G2 samples: {len(g2_files)}")
    g2_per_sample = []
    for i, path in enumerate(g2_files, 1):
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        motion = np.load(str(path)).astype(np.float64)
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            continue
        trial_id = meta.get("trial_id", path.stem)
        per_strength = {}
        for st in STRENGTHS:
            refined = _apply_b2(motion, st, vs)
            ng = _compute_g2_netgain(motion, refined, evaluators, alpha)
            per_strength[st] = ng
        best_strength = max(STRENGTHS, key=lambda s: per_strength[s]["netgain"])
        per_strength["val_best"] = dict(per_strength[best_strength])
        per_strength["val_best"]["best_strength"] = best_strength
        g2_per_sample.append({"trial_id": trial_id, "per_variant": per_strength})
        if i % 20 == 0:
            print(f"  [G2] {i}/{len(g2_files)} done")

    # Aggregate.
    def _agg(per_sample, variant: str) -> dict[str, Any]:
        ng = np.array([s["per_variant"][variant]["netgain"] for s in per_sample])
        td = np.array([s["per_variant"][variant]["target_delta"] for s in per_sample])
        fl = np.array([s["per_variant"][variant]["fidelity_loss"] for s in per_sample])
        out = {
            "n": len(per_sample),
            "netgain": {"mean": float(ng.mean()), "median": float(np.median(ng)),
                        "min": float(ng.min()), "max": float(ng.max()),
                        "p25": float(np.percentile(ng, 25)), "p75": float(np.percentile(ng, 75))},
            "target_delta": {"mean": float(td.mean()), "median": float(np.median(td))},
            "fidelity_loss": {"mean": float(fl.mean()), "median": float(np.median(fl))},
        }
        if variant == "val_best":
            best_strength_dist = Counter(s["per_variant"]["val_best"]["best_strength"] for s in per_sample)
            out["best_strength_distribution"] = dict(best_strength_dist)
        return out

    summary = {
        "schema_version": "1.0.0",
        "record_type": "baseline_b2_family_sweep_summary",
        "task_id": "baseline_b2_family_sweep_v1",
        "seed": args.seed,
        "n_syn_samples": len(syn_per_sample),
        "n_g2_samples": len(g2_per_sample),
        "netgain_weight_status": "calibrated_protocol_a_v1",
        "netgain_weights": dict(netgain_weights),
        "synthetic_aggregate": {
            "B2-small": _agg(syn_per_sample, "small"),
            "B2-medium": _agg(syn_per_sample, "medium"),
            "B2-large": _agg(syn_per_sample, "large"),
            "B2-val-best": _agg(syn_per_sample, "val_best"),
        },
        "g2_aggregate": {
            "B2-small": _agg(g2_per_sample, "small"),
            "B2-medium": _agg(g2_per_sample, "medium"),
            "B2-large": _agg(g2_per_sample, "large"),
            "B2-val-best": _agg(g2_per_sample, "val_best"),
        },
        "per_sample_synthetic": syn_per_sample,
        "per_sample_g2": g2_per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {args.output}")

    # Summary table.
    print(f"\n=== Synthetic (n={len(syn_per_sample)}) ===")
    for var in ["B2-small", "B2-medium", "B2-large", "B2-val-best"]:
        a = summary["synthetic_aggregate"][var]
        print(f"  {var:14s}: NetGain median={a['netgain']['median']:+.5f} mean={a['netgain']['mean']:+.5f}")
    print(f"  best_strength dist (val-best): {summary['synthetic_aggregate']['B2-val-best']['best_strength_distribution']}")

    print(f"\n=== G2 (n={len(g2_per_sample)}) ===")
    for var in ["B2-small", "B2-medium", "B2-large", "B2-val-best"]:
        a = summary["g2_aggregate"][var]
        print(f"  {var:14s}: NetGain median={a['netgain']['median']:+.5f} mean={a['netgain']['mean']:+.5f}")
    print(f"  best_strength dist (val-best): {summary['g2_aggregate']['B2-val-best']['best_strength_distribution']}")


if __name__ == "__main__":
    main()
