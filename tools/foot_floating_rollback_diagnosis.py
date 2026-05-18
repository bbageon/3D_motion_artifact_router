"""Foot_floating × FootLockTool 의 rollback 원인 진단.

Task 3 (loop integration smoke) 의 foot_floating 케이스에서 RefinementLoop 가
rollback 한 사실을 발견 — AGENTS.md §3-4 Score 비감소 의무 작동. 그러나
**어떤 evaluator 가 증가했는지** 는 score_trace 에 박제되지 않았다.

본 도구는 30 sample × 3 strength 의 FootLockTool 적용 결과를 per-evaluator
score 로 분해 박제. 결과는:
  - "rollback 이 정당한가" (다른 evaluator 가 실제 큰 폭 악화) 또는
  - "rollback 이 false alarm 인가" (rule_based 의 score 합산 정의 자체가 부적합)
  의 정량 판별 근거.

본 결과는 NetGain weight grid search (Sub-task 1B) 의 motivation 데이터.

CLI 예:
    python -m tools.foot_floating_rollback_diagnosis --n-samples 30 --seed 42 \\
        --output evals/snapshots/foot_floating_rollback_diagnosis_v1.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from correction_tools import FootLockTool
from evaluators import DEFAULT_EVALUATORS
from tools.synthetic_injection import inject_foot_floating

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"


def _aggregate_score(reports) -> float:
    if not reports:
        return 0.0
    return float(np.mean([r.score for r in reports]))


def diagnose(
    data_dir: Path,
    n_samples: int,
    seed: int,
    lift_height: float,
    strengths: tuple[str, ...],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    npy_files = sorted(data_dir.glob("*.npy"))
    n = min(n_samples, len(npy_files))
    chosen_idx = rng.choice(len(npy_files), size=n, replace=False)
    chosen = [npy_files[i] for i in chosen_idx]

    tool = FootLockTool(default_ground_y=0.0)
    evaluators = list(DEFAULT_EVALUATORS)

    # 결과 누적: (strength, evaluator_name) → list of delta (after - before).
    deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
    # 또한 total score (raw sum) 변화도.
    total_deltas: dict[str, list[float]] = defaultdict(list)
    # rollback 사례 — total score 가 tolerance 초과로 증가한 sample.
    rollback_count: dict[str, int] = defaultdict(int)
    # sample-level diagnostic detail.
    per_sample: list[dict[str, Any]] = []

    for path in chosen:
        clean = np.load(str(path)).astype(np.float64)
        if clean.ndim != 3 or clean.shape[1] != 22 or clean.shape[2] != 3:
            continue
        corrupted = inject_foot_floating(clean, lift_height=lift_height, seed=seed)
        T = corrupted.shape[0]

        # before — per-evaluator score.
        before_per_eval: dict[str, float] = {}
        for ev in evaluators:
            reports = ev.evaluate(corrupted)
            before_per_eval[ev.name] = _aggregate_score(reports)
        total_before = float(sum(before_per_eval.values()))

        sample_rec: dict[str, Any] = {
            "trial_id": path.stem,
            "T": T,
            "before_per_eval": before_per_eval,
            "total_before": total_before,
            "applies": [],
        }

        for strength in strengths:
            try:
                corrected, report = tool.apply(
                    corrupted,
                    target_part="both_feet",
                    target_joints=[],
                    frame_range=(0, T - 1),
                    strength=strength,  # type: ignore[arg-type]
                )
            except ValueError as e:
                sample_rec["applies"].append({
                    "strength": strength,
                    "error": str(e),
                })
                continue

            after_per_eval: dict[str, float] = {}
            for ev in evaluators:
                reports = ev.evaluate(corrected)
                after_per_eval[ev.name] = _aggregate_score(reports)
            total_after = float(sum(after_per_eval.values()))
            total_delta = total_after - total_before

            for ev in evaluators:
                deltas[(strength, ev.name)].append(after_per_eval[ev.name] - before_per_eval[ev.name])
            total_deltas[strength].append(total_delta)
            if total_delta > 0.01:  # default loop tolerance
                rollback_count[strength] += 1

            sample_rec["applies"].append({
                "strength": strength,
                "after_per_eval": after_per_eval,
                "total_after": total_after,
                "total_delta": total_delta,
                "correction_magnitude": float(report.correction_magnitude),
                "would_trigger_rollback": total_delta > 0.01,
            })
        per_sample.append(sample_rec)

    summary: dict[str, Any] = {
        "n_samples": len(per_sample),
        "lift_height_m": lift_height,
        "strengths": list(strengths),
        "per_strength_evaluator_delta_mean": {
            f"{s}__{ev_name}": float(np.mean(vals)) if vals else None
            for (s, ev_name), vals in sorted(deltas.items())
        },
        "per_strength_evaluator_delta_median": {
            f"{s}__{ev_name}": float(np.median(vals)) if vals else None
            for (s, ev_name), vals in sorted(deltas.items())
        },
        "per_strength_total_delta_summary": {
            s: {
                "mean": float(np.mean(vals)) if vals else None,
                "median": float(np.median(vals)) if vals else None,
                "p25": float(np.percentile(vals, 25)) if vals else None,
                "p75": float(np.percentile(vals, 75)) if vals else None,
            }
            for s, vals in sorted(total_deltas.items())
        },
        "per_strength_rollback_count": dict(rollback_count),
        "per_strength_rollback_rate": {
            s: rollback_count[s] / len(per_sample) for s in strengths if per_sample
        },
        "per_sample": per_sample,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="foot_floating × FootLockTool rollback diagnosis")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lift-height", type=float, default=0.08)
    parser.add_argument("--strengths", type=str, nargs="+", default=["small", "medium", "large"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = diagnose(
        data_dir=args.data_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        lift_height=args.lift_height,
        strengths=tuple(args.strengths),
    )

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OK] wrote diagnosis to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
