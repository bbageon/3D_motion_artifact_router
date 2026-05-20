"""H-2026-205 Stage 0 — oracle raw record → supervised selector training data.

각 oracle_single_step_v2 raw record 에서 (state, label) tuple 추출.

State (feature):
  - artifact_kind one-hot (3): foot_floating / bone_stretch_right_arm / global_jitter.
  - corrupted motion 의 evaluator scores (3): FootFloating, BoneLength, VelocityJitter
    의 max score (per-sample, aggregated over reports).

Label:
  - best_tool (3-class): FootLockTool / BoneProjectionTool / VelocitySmoothingTool.
  - best_strength (3-class): small / medium / large.
  - 또는 joint (tool × strength = 9-class) — RankingPolicy 의 직접 학습.

corrupted motion 재평가:
  - oracle raw 에 corrupted motion 의 evaluator scores 가 직접 박제되지 않음 (target
    evaluator 의 target_score_before 만 있음). 모든 evaluator 의 scores 가 state 에
    필요하므로 corrupted motion 을 다시 build (synthetic injection 재현, seed 동일).

CLI:
    python -m tools.build_oracle_training_data \\
        --oracle-prefix oracle_single_step_v2 \\
        --output evals/snapshots/oracle_training_data_v1.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from evaluators import DEFAULT_EVALUATORS
from tools.synthetic_injection import (
    inject_bone_stretch,
    inject_foot_floating,
    inject_jitter,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "evals" / "raw"

ARTIFACT_SPECS = {
    "foot_floating": {
        "inject_fn": "inject_foot_floating",
        "inject_kwargs": {"lift_height": 0.08},
    },
    "bone_stretch_right_arm": {
        "inject_fn": "inject_bone_stretch",
        "inject_kwargs": {"chain_label": "right_arm", "stretch_factor": 1.30},
    },
    "global_jitter": {
        "inject_fn": "inject_jitter",
        "inject_kwargs": {"noise_std": 0.05},
    },
}

ARTIFACT_KINDS = list(ARTIFACT_SPECS.keys())
TOOL_NAMES = ["FootLockTool", "BoneProjectionTool", "VelocitySmoothingTool"]
STRENGTHS = ["small", "medium", "large"]
EVALUATOR_NAMES = ["FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator"]


def _apply_injection(artifact_kind: str, clean: np.ndarray, seed: int) -> np.ndarray:
    spec = ARTIFACT_SPECS[artifact_kind]
    fn_name = spec["inject_fn"]
    kwargs = dict(spec["inject_kwargs"])
    kwargs["seed"] = seed
    if fn_name == "inject_foot_floating":
        return inject_foot_floating(clean, **kwargs)
    if fn_name == "inject_bone_stretch":
        T = clean.shape[0]
        half = max(1, T // 2)
        stretched = inject_bone_stretch(clean[:half], **kwargs)
        return np.concatenate([stretched, clean[half:]], axis=0)
    if fn_name == "inject_jitter":
        return inject_jitter(clean, **kwargs)
    raise ValueError(f"unknown inject_fn {fn_name!r}")


def _max_score(reports: list) -> float:
    if not reports:
        return 0.0
    return float(max(r.score for r in reports))


def _extract_state(motion: np.ndarray, artifact_kind: str) -> dict[str, Any]:
    """corrupted motion + artifact_kind → state feature."""
    evaluators = list(DEFAULT_EVALUATORS)
    eval_scores = {}
    for ev in evaluators:
        eval_scores[ev.name] = _max_score(ev.evaluate(motion))
    # artifact_kind one-hot.
    artifact_onehot = [1 if k == artifact_kind else 0 for k in ARTIFACT_KINDS]
    # evaluator scores (named tuple).
    scores_ordered = [eval_scores[name] for name in EVALUATOR_NAMES]
    return {
        "artifact_kind": artifact_kind,
        "artifact_onehot": artifact_onehot,
        "evaluator_scores": dict(zip(EVALUATOR_NAMES, scores_ordered)),
        "feature_vector": artifact_onehot + scores_ordered,
        "feature_names": [f"is_{k}" for k in ARTIFACT_KINDS] + [f"score_{n}" for n in EVALUATOR_NAMES],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build supervised selector training data from oracle raw records")
    parser.add_argument("--raw-records-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--oracle-prefix", type=str, nargs="+", default=["oracle_single_step_v2"],
                        help="하나 이상의 prefix. 본 prefix 의 모든 raw record 합쳐 training data.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    tuples: list[dict[str, Any]] = []
    raw_paths: list[Path] = []
    for prefix in args.oracle_prefix:
        raw_paths.extend(sorted(args.raw_records_dir.glob(f"*{prefix}*.json")))
    for path in raw_paths:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("record_type") != "oracle_single_step_sample":
            continue
        trial_id = r["trial_id"]
        sample_path = Path(r["sample_path"])
        seed = int(r["seed"])
        if not sample_path.exists():
            print(f"[WARN] sample path not found: {sample_path}")
            continue
        clean = np.load(str(sample_path)).astype(np.float64)
        for art, sel in r["selections"].items():
            best = sel.get("best_candidate")
            if best is None or best.get("skipped"):
                continue
            corrupted = _apply_injection(art, clean, seed)
            state = _extract_state(corrupted, art)
            label = {
                "best_tool": best["tool_name"],
                "best_strength": best["strength"],
                "best_target_part": best.get("target_part"),
                "best_netgain_provisional": float(best["netgain_provisional"]),
            }
            # joint label encoding (tool, strength) → 9-class index.
            joint_idx = TOOL_NAMES.index(best["tool_name"]) * len(STRENGTHS) + STRENGTHS.index(best["strength"])
            tuples.append({
                "trial_id": trial_id,
                "artifact_kind": art,
                "state": state,
                "label": label,
                "joint_class_idx": joint_idx,
                "tool_class_idx": TOOL_NAMES.index(best["tool_name"]),
                "strength_class_idx": STRENGTHS.index(best["strength"]),
            })

    # Label distribution.
    tool_dist = Counter(t["label"]["best_tool"] for t in tuples)
    strength_dist = Counter(t["label"]["best_strength"] for t in tuples)
    joint_dist = Counter((t["label"]["best_tool"], t["label"]["best_strength"]) for t in tuples)
    per_artifact_dist = {}
    for art in ARTIFACT_KINDS:
        per_artifact_dist[art] = Counter(
            (t["label"]["best_tool"], t["label"]["best_strength"])
            for t in tuples if t["artifact_kind"] == art
        )

    summary = {
        "schema_version": "1.0.0",
        "record_type": "oracle_supervised_training_data",
        "oracle_prefix": args.oracle_prefix,
        "split_ids_included": sorted(set(t.get("trial_id", "") for t in tuples)),
        "n_unique_trials": len({t["trial_id"] for t in tuples}),
        "n_tuples": len(tuples),
        "feature_names": tuples[0]["state"]["feature_names"] if tuples else [],
        "n_features": len(tuples[0]["state"]["feature_vector"]) if tuples else 0,
        "n_classes_joint": len(TOOL_NAMES) * len(STRENGTHS),
        "label_distribution": {
            "tool": dict(tool_dist),
            "strength": dict(strength_dist),
            "joint": {f"{k[0]}/{k[1]}": v for k, v in joint_dist.items()},
            "per_artifact": {
                art: {f"{k[0]}/{k[1]}": v for k, v in dist.items()}
                for art, dist in per_artifact_dist.items()
            },
        },
        "tuples": tuples,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OK] wrote {len(tuples)} tuples to {args.output}")
        print(f"\nLabel distribution:")
        print(f"  tools: {dict(tool_dist)}")
        print(f"  strengths: {dict(strength_dist)}")
        print(f"\nJoint (tool, strength) distribution:")
        for k, v in sorted(joint_dist.items()):
            print(f"  {k[0]:25s} {k[1]:8s}: {v}")
        print(f"\nPer artifact label distribution:")
        for art in ARTIFACT_KINDS:
            print(f"  {art}: {dict(per_artifact_dist[art])}")
    else:
        print(text)


if __name__ == "__main__":
    main()
