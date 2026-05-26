"""Step C (사용자 directive 2026-05-26): PhysicalGateV0 의 HumanML3D clean calibration.

각 PhysicalGateV0 evaluator (Penetrate / Float / Skate / JerkSpike / BoneLengthCV) 의
HumanML3D clean motion N=500 sample distribution → p50/p90/p95/p99 추출.

본 calibration 결과는 evals/snapshots/physical_gate_clean_calibration_v1.json 에 저장.
Gate threshold 의 정량 근거 — unsafe_threshold = p99 (보수적) 또는 p95 (loose).

CLI:
    python -m tools.physical_gate_clean_calibration \
        --n-samples 500 --seed 42 \
        --output evals/snapshots/physical_gate_clean_calibration_v1.json

근거 (AGENTS.md §3-22): 사용자 directive 2026-05-26 — "HumanML3D clean p95/p99 calibration".
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluators import (
    BoneLengthCVEvaluator, FloatEvaluator, JerkSpikeEvaluator,
    PenetrateEvaluator, SkateEvaluator,
)


def _utcnow_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    parser.add_argument("--min-frames", type=int, default=20,
                        help="Skip motion if T < min_frames (jerk needs ≥3, but stable stats need ≥20).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    evaluators = [
        PenetrateEvaluator(),
        FloatEvaluator(),
        SkateEvaluator(),
        JerkSpikeEvaluator(),
        BoneLengthCVEvaluator(),
    ]
    eval_names = [e.name for e in evaluators]
    print(f"[INFO] evaluators: {eval_names}")

    npy_files = sorted(args.data_dir.glob("*.npy"))
    print(f"[INFO] total HumanML3D npy: {len(npy_files)}, sampling n={args.n_samples} (seed={args.seed})")
    chosen_idx = rng.choice(len(npy_files), size=min(args.n_samples, len(npy_files)), replace=False)
    chosen = [npy_files[i] for i in chosen_idx]

    # Accumulate per-evaluator scores (each evaluator may return multiple reports per motion).
    per_eval_scores: dict[str, list[float]] = {n: [] for n in eval_names}
    per_eval_metadata: dict[str, list[dict]] = {n: [] for n in eval_names}
    n_processed = 0
    n_skipped = 0

    for i, path in enumerate(chosen, 1):
        try:
            motion = np.load(str(path)).astype(np.float64)
        except Exception as e:
            n_skipped += 1
            continue
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            n_skipped += 1
            continue
        if motion.shape[0] < args.min_frames:
            n_skipped += 1
            continue
        for ev in evaluators:
            reports = ev.evaluate(motion)
            if not reports:
                # 0 reports — score 0 for evaluators that return per-violation reports
                # (Penetrate / Float / Skate). For JerkSpike / BoneLengthCV (always one report),
                # this should not happen unless motion too short.
                if ev.name in ("PenetrateEvaluator", "FloatEvaluator", "SkateEvaluator"):
                    per_eval_scores[ev.name].append(0.0)
                continue
            for r in reports:
                per_eval_scores[ev.name].append(r.score)
                if i <= 3:  # store metadata for first few only.
                    per_eval_metadata[ev.name].append({k: v for k, v in r.metadata.items()
                                                       if isinstance(v, (int, float, str, list))})
        n_processed += 1
        if i % 50 == 0:
            print(f"   processed {i}/{len(chosen)} (n_skipped={n_skipped})", flush=True)

    # Compute percentiles.
    summary = {}
    for name, scores in per_eval_scores.items():
        if not scores:
            summary[name] = {"n": 0, "note": "no scores collected"}
            continue
        arr = np.array(scores, dtype=np.float64)
        summary[name] = {
            "n": len(arr),
            "min": float(arr.min()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }

    out = {
        "schema_version": "1.0.0",
        "record_type": "physical_gate_clean_calibration",
        "task_id": "physical_gate_clean_calibration_v1",
        "timestamp": _utcnow_stamp(),
        "seed": args.seed,
        "data_dir": str(args.data_dir),
        "n_requested": args.n_samples,
        "n_processed": n_processed,
        "n_skipped": n_skipped,
        "min_frames": args.min_frames,
        "evaluator_severity_versions": {
            ev.name: getattr(sys.modules[type(ev).__module__], "SEVERITY_VERSION", "unversioned")
            for ev in evaluators
        },
        "summary": summary,
        "sample_metadata_first3": per_eval_metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== PhysicalGateV0 HumanML3D clean calibration (n={n_processed}) ===")
    for name in eval_names:
        s = summary[name]
        if s.get("n", 0) == 0:
            print(f"  {name}: NO DATA"); continue
        print(f"  {name}: n={s['n']:5d}  p50={s['p50']:.5f}  p90={s['p90']:.5f}  p95={s['p95']:.5f}  p99={s['p99']:.5f}  max={s['max']:.5f}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
