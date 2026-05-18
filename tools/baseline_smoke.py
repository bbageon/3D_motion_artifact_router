"""HumanML3D GT motion baseline smoke run — evaluator score 분포 측정 + raw record 저장.

본 스크립트는 evaluator 3 종 ([`FootFloatingEvaluator`](../evaluators/foot_floating_evaluator.py),
[`BoneLengthEvaluator`](../evaluators/bone_length_evaluator.py),
[`VelocityJitterEvaluator`](../evaluators/velocity_jitter_evaluator.py)) 를
HumanML3D 의 GT motion (`external_assets/HumanML3D/new_joints/*.npy`) 에 적용해
"clean motion 의 score 분포" 를 측정한다. 본 분포는:

  1. severity threshold 의 보정 근거 (Week 2 의 임계값은 보수적 default 였음).
  2. [`H-2026-203`](../evals/hypotheses/H-2026-203.md) (no-harm) 의 baseline —
     refined motion 의 score 가 clean baseline 분포 안에 머무는지의 기준.

본 도구는 두 종류 산출물을 생성한다:

- **Summary**: 통계 (n_reports / severity counts / score mean·median·p95·max) 를
  stdout 또는 `--output` 에 JSON 으로.
- **Raw records (sample-level)**: 각 sample 별로 evaluator report 와 motion meta 를
  포함한 record 를 `--raw-output-dir/<timestamp>_<task_id>_<trial_id>.json` 으로
  저장 (AGENTS.md §3-6 평가 기록 의무 준수).

CLI 예:
    python -m tools.baseline_smoke --n-samples 100 \\
        --raw-output-dir evals/raw \\
        --output evals/snapshots/baseline_smoke_100.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluators import DEFAULT_EVALUATORS, EvaluatorReport

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "baseline_smoke"


def _now_iso() -> str:
    """결정성 위해 microsecond 단위 timestamp (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _report_to_dict(r: EvaluatorReport) -> dict[str, Any]:
    d = asdict(r)
    # tuple 은 JSON list 로 변환됨
    d["frames"] = [int(d["frames"][0]), int(d["frames"][1])]
    return d


def _make_raw_record(
    *,
    timestamp: str,
    task_id: str,
    trial_id: str,
    split_id: str,
    sample_path: Path,
    motion: np.ndarray,
    fps: int,
    ground_y_estimated: float,
    evaluator_config_hashes: dict[str, str],
    evaluator_severity_versions: dict[str, str],
    reports_by_evaluator: dict[str, list[EvaluatorReport]],
    seed: int,
) -> dict[str, Any]:
    """AGENTS.md §3-6 평가 기록 의무 + §3-15 raw record metadata 의무 항목을 충족하는
    baseline smoke raw record.

    correction tool 미적용·refinement loop 미실행이므로 NetGain / FidelityLoss /
    tool call trace 는 N/A. metadata 에 명시.

    `severity_versions` 와 `split_id` 는 AGENTS.md §3-15 의무 — 누락 시 후속 비교 평가
    에서 본 record 의 score·severity 해석 불가.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "timestamp": timestamp,
        "task_id": task_id,
        "trial_id": trial_id,
        "split_id": split_id,  # AGENTS.md §3-15 의무
        "sample_path": str(sample_path),
        "generator_id": "humanml3d_gt",  # clean GT, generator 없음
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,  # AGENTS.md §3-15 의무
        "tool_registry_config_hash": None,  # baseline 은 tool 미적용
        "skeleton_normalizer_model_card_hash": None,  # normalize 미사용 (이미 canonical)
        "motion_shape": [int(motion.shape[0]), int(motion.shape[1]), int(motion.shape[2])],
        "fps": fps,
        "ground_y_estimated": float(ground_y_estimated),
        "seed": int(seed),
        "evaluator_reports": {
            name: [_report_to_dict(r) for r in reports]
            for name, reports in reports_by_evaluator.items()
        },
        "metrics_not_applicable": {
            "NetGain": "baseline; no refinement",
            "FidelityLoss": "baseline; no refinement",
            "ToolCallCost": "baseline; no tool invocation",
            "tool_call_trace": "baseline; no tool invocation",
            "netgain_weight_status": "n/a (baseline)",  # AGENTS.md §6-11 — netgain 미계산 시 명시
        },
        "negative_result": False,
    }


def _get_severity_version(evaluator: Any) -> str:
    """evaluator 모듈의 SEVERITY_VERSION 상수 추출. 부재 시 'unversioned' 반환.

    AGENTS.md §3-15 raw record metadata 의무 — 모든 evaluator 가 SEVERITY_VERSION
    상수를 노출해야 본 record 가 후속 비교에서 해석 가능.
    """
    import sys
    mod = sys.modules.get(type(evaluator).__module__)
    if mod is None:
        return "unversioned"
    return getattr(mod, "SEVERITY_VERSION", "unversioned")


def run_baseline(
    data_dir: Path = DEFAULT_DATA_DIR,
    n_samples: int = 20,
    seed: int = 42,
    raw_output_dir: Path | None = None,
    task_id: str = "baseline_smoke_humanml3d",
    split_id: str | None = None,
) -> dict[str, Any]:
    """Sample n 개 motion 에 evaluator 3 종을 적용해 score 통계 + raw record 산출.

    raw_output_dir 가 명시되면 각 sample 별로 raw record 를 저장한다.

    `split_id` 가 None 이면 `task_id` 와 동일하게 설정한다 (AGENTS.md §3-15).
    """
    rng = np.random.default_rng(seed)
    npy_files = sorted(data_dir.glob("*.npy"))
    if len(npy_files) == 0:
        raise FileNotFoundError(f"no .npy files in {data_dir}")
    if n_samples < len(npy_files):
        chosen = rng.choice(npy_files, size=n_samples, replace=False)
    else:
        chosen = np.array(npy_files)

    if split_id is None:
        split_id = task_id

    # evaluator config hashes + severity versions — 본 run 전체 공통.
    # AGENTS.md §3-15: 모든 raw record 에 두 항목 모두 박제 의무.
    evaluator_config_hashes: dict[str, str] = {
        ev.name: ev.evaluator_class_hash() for ev in DEFAULT_EVALUATORS
    }
    evaluator_severity_versions: dict[str, str] = {
        ev.name: _get_severity_version(ev) for ev in DEFAULT_EVALUATORS
    }

    # 통계 누적
    per_evaluator: dict[str, list[float]] = {}
    n_reports: dict[str, int] = {}
    severity_counts: dict[str, dict[str, int]] = {}
    trial_ids: list[str] = []

    raw_dir = Path(raw_output_dir).resolve() if raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    for path in chosen:
        path = Path(str(path))  # numpy array 경유 시 PosixPath 보장
        motion = np.load(str(path)).astype(np.float64)
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            continue
        trial_id = path.stem
        trial_ids.append(trial_id)

        # ground_y heuristic (motion 전체 최저 y) — 본 record 에 박제
        ground_y_estimated = float(np.min(motion[:, :, 1]))

        reports_by_evaluator: dict[str, list[EvaluatorReport]] = {}
        for evaluator in DEFAULT_EVALUATORS:
            name = evaluator.name
            reports: list[EvaluatorReport] = evaluator.evaluate(motion)
            reports_by_evaluator[name] = reports
            per_evaluator.setdefault(name, [])
            n_reports.setdefault(name, 0)
            severity_counts.setdefault(name, {"low": 0, "medium": 0, "high": 0})
            n_reports[name] += len(reports)
            for r in reports:
                per_evaluator[name].append(float(r.score))
                severity_counts[name][r.severity] += 1

        if raw_dir is not None:
            timestamp = _now_iso()
            record = _make_raw_record(
                timestamp=timestamp,
                task_id=task_id,
                trial_id=trial_id,
                split_id=split_id,
                sample_path=path,
                motion=motion,
                fps=20,
                ground_y_estimated=ground_y_estimated,
                evaluator_config_hashes=evaluator_config_hashes,
                evaluator_severity_versions=evaluator_severity_versions,
                reports_by_evaluator=reports_by_evaluator,
                seed=seed,
            )
            out_path = raw_dir / f"{timestamp}_{task_id}_{trial_id}.json"
            out_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "baseline_smoke_summary",
        "n_samples_evaluated": int(len(trial_ids)),
        "data_dir": str(data_dir),
        "task_id": task_id,
        "split_id": split_id,
        "seed": int(seed),
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,
        "raw_records_dir": str(raw_dir) if raw_dir else None,
        "per_evaluator_stats": {},
        "trial_ids": trial_ids,
    }
    for name, scores in per_evaluator.items():
        if not scores:
            summary["per_evaluator_stats"][name] = {
                "n_reports": n_reports[name],
                "severity_counts": severity_counts[name],
                "score_mean": None,
                "score_median": None,
                "score_p50": None,
                "score_p75": None,
                "score_p90": None,
                "score_p95": None,
                "score_p99": None,
                "score_max": None,
                "score_min": None,
            }
            continue
        arr = np.asarray(scores, dtype=np.float64)
        summary["per_evaluator_stats"][name] = {
            "n_reports": n_reports[name],
            "severity_counts": severity_counts[name],
            "score_mean": float(arr.mean()),
            "score_median": float(np.median(arr)),
            "score_p50": float(np.percentile(arr, 50)),
            "score_p75": float(np.percentile(arr, 75)),
            "score_p90": float(np.percentile(arr, 90)),
            "score_p95": float(np.percentile(arr, 95)),
            "score_p99": float(np.percentile(arr, 99)),
            "score_max": float(arr.max()),
            "score_min": float(arr.min()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="HumanML3D GT motion baseline smoke run")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None,
                        help="summary JSON path. None 이면 stdout.")
    parser.add_argument("--raw-output-dir", type=Path, default=None,
                        help="sample-level raw record 디렉토리 (보통 evals/raw/). "
                             "None 이면 raw record 저장 안 함.")
    parser.add_argument("--task-id", type=str, default="baseline_smoke_humanml3d",
                        help="raw record 의 task_id (run 식별자).")
    parser.add_argument("--split-id", type=str, default=None,
                        help="raw record 의 split_id (AGENTS.md §3-15 의무). "
                             "None 이면 task_id 와 동일하게 설정.")
    args = parser.parse_args()

    summary = run_baseline(
        data_dir=args.data_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        raw_output_dir=args.raw_output_dir,
        task_id=args.task_id,
        split_id=args.split_id,
    )
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OK] wrote summary to {args.output}")
        if args.raw_output_dir:
            print(f"[OK] wrote {summary['n_samples_evaluated']} raw records to {args.raw_output_dir}")
    else:
        print(text)


if __name__ == "__main__":
    main()
