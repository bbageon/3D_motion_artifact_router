"""G2 (MotionGPT) generated motion 의 evaluator 분포 profiling — Action 2.

G2 batch (tools/g2_generate_batch.py) 의 motion 들에 모든 evaluator 를 적용해 자연
artifact 분포를 측정한다. 사용자 framing ("큰 이득은 multi-artifact case 또는 G2
natural artifact 에서") 의 검증 기반.

비교 대상: HumanML3D GT 의 baseline_holdout_v2 (이미 측정 완료) 와 분포 비교.

산출물:
  - per-sample raw record (motion shape + evaluator severity + score).
  - aggregate snapshot: per-evaluator severity histogram + score percentiles.
  - HumanML3D GT vs G2 natural 의 분포 차이.

AGENTS.md 의무:
  - §3-5 generator quality-tier 분리: generator_id="G2_motiongpt_*".
  - §3-15 raw record metadata 박제 (severity_versions / split_id / evaluator_config_hashes).

CLI 예:
    python -m tools.g2_evaluator_profile \\
        --g2-batch-dir external_assets/g2_generated_v1 \\
        --task-id g2_natural_artifact_v1 \\
        --raw-output-dir evals/raw \\
        --output evals/snapshots/g2_natural_artifact_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluators import DEFAULT_EVALUATORS

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "g2_natural_artifact_sample"
SUMMARY_TYPE = "g2_natural_artifact_summary"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _get_severity_version(evaluator: Any) -> str:
    mod = sys.modules.get(type(evaluator).__module__)
    if mod is None:
        return "unversioned"
    return getattr(mod, "SEVERITY_VERSION", "unversioned")


def _load_g2_motions(batch_dir: Path) -> list[tuple[Path, dict[str, Any], np.ndarray]]:
    """G2 batch 의 motion + metadata load."""
    out = []
    for npy_path in sorted(batch_dir.glob("motion_*.npy")):
        meta_path = npy_path.with_suffix(".json")
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        motion = np.load(str(npy_path)).astype(np.float64)
        # MotionGPT output 가 [T, 22, 3] 인지 확인.
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            print(f"[WARN] {npy_path.name} unexpected shape {motion.shape}, skipping", file=sys.stderr)
            continue
        out.append((npy_path, meta, motion))
    return out


def _evaluate_motion(motion: np.ndarray) -> dict[str, Any]:
    """모든 evaluator 의 report 를 dict 로 집계 (target_part 별)."""
    out: dict[str, Any] = {}
    for ev in DEFAULT_EVALUATORS:
        reports = ev.evaluate(motion)
        ev_data: list[dict[str, Any]] = []
        for r in reports:
            ev_data.append({
                "body_part": r.body_part,
                "frames": list(r.frames) if r.frames else None,
                "score": float(r.score),
                "severity": r.severity,
                "error_type": r.error_type,
            })
        # Aggregate per-evaluator: mean score, max severity.
        if ev_data:
            scores = [d["score"] for d in ev_data]
            sevs = [d["severity"] for d in ev_data]
            sev_priority = {"low": 0, "medium": 1, "high": 2}
            max_sev = max(sevs, key=lambda s: sev_priority.get(s, -1))
            out[ev.name] = {
                "n_reports": len(ev_data),
                "score_mean": float(np.mean(scores)),
                "score_max": float(np.max(scores)),
                "max_severity": max_sev,
                "reports": ev_data,
            }
        else:
            out[ev.name] = {"n_reports": 0, "score_mean": None, "score_max": None, "max_severity": None, "reports": []}
    return out


def _make_raw_record(
    *,
    timestamp: str,
    task_id: str,
    split_id: str,
    trial_id: str,
    sample_path: Path,
    g2_meta: dict[str, Any],
    motion_shape: tuple[int, ...],
    eval_data: dict[str, Any],
    evaluator_config_hashes: dict[str, str],
    evaluator_severity_versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "timestamp": timestamp,
        "task_id": task_id,
        "split_id": split_id,
        "trial_id": trial_id,
        "sample_path": str(sample_path),
        "generator_id": g2_meta.get("generator_id", "G2_motiongpt_unknown"),
        "generator_class_hash": g2_meta.get("generator_class_hash"),
        "generator_seed": g2_meta.get("seed"),
        "generator_prompt": g2_meta.get("prompt"),
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,
        "skeleton_normalizer_model_card_hash": None,
        "motion_shape": list(motion_shape),
        "fps": g2_meta.get("fps", 20),
        "evaluator_data": eval_data,
        "negative_result": False,
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """per-evaluator severity histogram + score distribution."""
    per_evaluator: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "n_samples": 0,
        "severity_counts": defaultdict(int),
        "scores": [],
    })
    motion_lengths: list[int] = []
    for rec in records:
        motion_lengths.append(rec["motion_shape"][0])
        for ev_name, ev_data in rec["evaluator_data"].items():
            stats = per_evaluator[ev_name]
            stats["n_samples"] += 1
            if ev_data["max_severity"] is not None:
                stats["severity_counts"][ev_data["max_severity"]] += 1
            if ev_data["score_max"] is not None:
                stats["scores"].append(ev_data["score_max"])

    per_evaluator_summary: list[dict[str, Any]] = []
    for ev_name, stats in sorted(per_evaluator.items()):
        scores = np.array(stats["scores"], dtype=np.float64) if stats["scores"] else np.array([])
        per_evaluator_summary.append({
            "evaluator_name": ev_name,
            "n_samples": stats["n_samples"],
            "severity_counts": dict(stats["severity_counts"]),
            "score_distribution": {
                "n": int(scores.size),
                "mean": float(scores.mean()) if scores.size else None,
                "median": float(np.median(scores)) if scores.size else None,
                "p25": float(np.percentile(scores, 25)) if scores.size else None,
                "p75": float(np.percentile(scores, 75)) if scores.size else None,
                "p90": float(np.percentile(scores, 90)) if scores.size else None,
                "p95": float(np.percentile(scores, 95)) if scores.size else None,
                "min": float(scores.min()) if scores.size else None,
                "max": float(scores.max()) if scores.size else None,
            },
        })

    lengths = np.array(motion_lengths, dtype=np.int64) if motion_lengths else np.array([])
    return {
        "n_samples": len(records),
        "motion_length_distribution": {
            "min": int(lengths.min()) if lengths.size else None,
            "max": int(lengths.max()) if lengths.size else None,
            "mean": float(lengths.mean()) if lengths.size else None,
            "median": float(np.median(lengths)) if lengths.size else None,
        },
        "per_evaluator": per_evaluator_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 natural artifact profiling")
    parser.add_argument("--g2-batch-dir", type=Path, required=True,
                        help="G2 batch output dir (tools/g2_generate_batch.py).")
    parser.add_argument("--task-id", type=str, default="g2_natural_artifact_v1")
    parser.add_argument("--split-id", type=str, default="g2_natural_v1")
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    motions = _load_g2_motions(args.g2_batch_dir)
    if not motions:
        raise FileNotFoundError(f"no G2 motions in {args.g2_batch_dir}")
    print(f"[INFO] loaded {len(motions)} G2 motions from {args.g2_batch_dir}")

    evaluator_config_hashes = {ev.name: ev.evaluator_class_hash() for ev in DEFAULT_EVALUATORS}
    evaluator_severity_versions = {ev.name: _get_severity_version(ev) for ev in DEFAULT_EVALUATORS}

    raw_dir = Path(args.raw_output_dir).resolve() if args.raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for npy_path, g2_meta, motion in motions:
        trial_id = g2_meta.get("trial_id", npy_path.stem)
        eval_data = _evaluate_motion(motion)
        timestamp = _now_iso()
        record = _make_raw_record(
            timestamp=timestamp,
            task_id=args.task_id,
            split_id=args.split_id,
            trial_id=trial_id,
            sample_path=npy_path,
            g2_meta=g2_meta,
            motion_shape=motion.shape,
            eval_data=eval_data,
            evaluator_config_hashes=evaluator_config_hashes,
            evaluator_severity_versions=evaluator_severity_versions,
        )
        records.append(record)
        if raw_dir is not None:
            out_path = raw_dir / f"{timestamp}_{args.task_id}_{trial_id}.json"
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        # Console progress.
        print(f"  {trial_id}: shape={motion.shape}, foot_sev={eval_data.get('FootFloatingEvaluator', {}).get('max_severity')}, "
              f"bone_sev={eval_data.get('BoneLengthEvaluator', {}).get('max_severity')}, "
              f"jitter_sev={eval_data.get('VelocityJitterEvaluator', {}).get('max_severity')}")

    agg = _aggregate(records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": SUMMARY_TYPE,
        "task_id": args.task_id,
        "split_id": args.split_id,
        "g2_batch_dir": str(args.g2_batch_dir),
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,
        "raw_records_dir": str(raw_dir) if raw_dir else None,
        **agg,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"\n[OK] wrote summary to {args.output}")
        if raw_dir:
            print(f"[OK] wrote {len(records)} raw records to {raw_dir}")
        print(f"\nmotion length distribution: {agg['motion_length_distribution']}")
        for pe in summary["per_evaluator"]:
            print(f"\n{pe['evaluator_name']}: n={pe['n_samples']}, severity={pe['severity_counts']}")
            sd = pe["score_distribution"]
            if sd["n"] > 0:
                print(f"  score: median={sd['median']:.4f}, p75={sd['p75']:.4f}, p90={sd['p90']:.4f}, max={sd['max']:.4f}")
    else:
        print(text)


if __name__ == "__main__":
    main()
