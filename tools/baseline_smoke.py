"""HumanML3D GT motion baseline smoke run — evaluator score 분포 측정 도구.

본 스크립트는 evaluator 3 종 ([`FootFloatingEvaluator`](../evaluators/foot_floating_evaluator.py),
[`BoneLengthEvaluator`](../evaluators/bone_length_evaluator.py),
[`VelocityJitterEvaluator`](../evaluators/velocity_jitter_evaluator.py)) 를
HumanML3D 의 GT motion (`external_assets/HumanML3D/new_joints/*.npy`) 에 적용해
"clean motion 의 score 분포" 를 측정한다. 본 분포는:

  1. severity threshold 의 보정 근거 (Week 2 의 임계값은 보수적 default 였음).
  2. [`H-2026-203`](../evals/hypotheses/H-2026-203.md) (no-harm) 의 baseline —
     refined motion 의 score 가 clean baseline 분포 안에 머무는지의 기준.

본 도구는 평가 결과를 git 추적 가능 한 작은 JSON 으로 stdout 또는 지정 경로에
출력하며, evals/raw 의 실제 raw record 는 아니므로 별도 metadata 등록은 안 함.

CLI 예:
    python -m tools.baseline_smoke --n-samples 20 --output baseline_smoke.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluators import DEFAULT_EVALUATORS, EvaluatorReport

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"


def run_baseline(
    data_dir: Path = DEFAULT_DATA_DIR,
    n_samples: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Sample n 개 motion 에 evaluator 3 종을 적용해 score 통계 산출."""
    rng = np.random.default_rng(seed)
    npy_files = sorted(data_dir.glob("*.npy"))
    if len(npy_files) == 0:
        raise FileNotFoundError(f"no .npy files in {data_dir}")
    if n_samples < len(npy_files):
        chosen = rng.choice(npy_files, size=n_samples, replace=False)
    else:
        chosen = np.array(npy_files)

    per_evaluator: dict[str, list[float]] = {}
    n_reports: dict[str, int] = {}
    severity_counts: dict[str, dict[str, int]] = {}

    for path in chosen:
        motion = np.load(str(path)).astype(np.float64)
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            continue
        for evaluator in DEFAULT_EVALUATORS:
            name = evaluator.name
            reports: list[EvaluatorReport] = evaluator.evaluate(motion)
            per_evaluator.setdefault(name, [])
            n_reports.setdefault(name, 0)
            severity_counts.setdefault(name, {"low": 0, "medium": 0, "high": 0})
            n_reports[name] += len(reports)
            for r in reports:
                per_evaluator[name].append(float(r.score))
                severity_counts[name][r.severity] += 1

    summary: dict[str, Any] = {
        "n_samples_evaluated": int(len(chosen)),
        "data_dir": str(data_dir),
        "per_evaluator_stats": {},
    }
    for name, scores in per_evaluator.items():
        if not scores:
            summary["per_evaluator_stats"][name] = {
                "n_reports": n_reports[name],
                "severity_counts": severity_counts[name],
                "score_mean": None,
                "score_median": None,
                "score_p95": None,
                "score_max": None,
            }
            continue
        arr = np.asarray(scores, dtype=np.float64)
        summary["per_evaluator_stats"][name] = {
            "n_reports": n_reports[name],
            "severity_counts": severity_counts[name],
            "score_mean": float(arr.mean()),
            "score_median": float(np.median(arr)),
            "score_p95": float(np.percentile(arr, 95)),
            "score_max": float(arr.max()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="HumanML3D GT motion baseline smoke run")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None,
                        help="output JSON path. None 이면 stdout 출력.")
    args = parser.parse_args()

    summary = run_baseline(data_dir=args.data_dir, n_samples=args.n_samples, seed=args.seed)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OK] wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
