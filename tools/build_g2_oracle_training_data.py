"""Step 5-B: G2 small-calibration training data.

각 G2 motion 에 모든 9 action 적용 → best (tool, strength) label 추출. G2 distribution
의 oracle. supervised retrain 의 training data.

State (6-dim, Step 1 의 oracle_training_data 와 호환):
  - artifact_kind one-hot (3): primary artifact = max-score evaluator → 매핑.
  - evaluator scores (3): FootFloating, BoneLength, VelocityJitter max scores.

Label:
  - best_tool / best_strength (per-sample, single-step best NetGain over 9 actions).
  - NetGain = -ΔTarget - α·FidelityLoss - β·CorrectionMag - γ·ToolCost (Protocol B,
    target=mean(all 3), reference=original G2).

NOTE: Step 5-B 는 single-step best 만 측정 (closed-loop 가 아닌 single-step oracle).
Step 1 의 oracle_training_data 와 같은 schema → B6 supervised retrain 가능.

CLI:
    python -m tools.build_g2_oracle_training_data \\
        --g2-batch-dir external_assets/g2_generated_v1 \\
        --output evals/snapshots/g2_oracle_training_data_v1.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from correction_tools import BoneProjectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, EvaluatorReport
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_BY_NAME = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}
STRENGTHS = ["small", "medium", "large"]
ALL_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
ARTIFACT_KINDS = ["foot_floating", "bone_stretch_right_arm", "global_jitter"]
EVAL_TO_ARTIFACT = {
    "FootFloatingEvaluator": "foot_floating",
    "BoneLengthEvaluator": "bone_stretch_right_arm",
    "VelocityJitterEvaluator": "global_jitter",
}
TOOL_TO_TARGET_PART = {
    "FootLockTool": "both_feet",
    "BoneProjectionTool": "right_arm",
    "VelocitySmoothingTool": "full_body",
}


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _target_score_all(reports_dict: dict[str, list[EvaluatorReport]]) -> float:
    return float(np.mean([_max_score(reports_dict.get(n, [])) for n in ALL_EVALUATORS]))


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _load_g2_motions(batch_dir: Path) -> list[tuple[Path, dict[str, Any], np.ndarray]]:
    out = []
    for npy_path in sorted(batch_dir.glob("motion_*.npy")):
        meta_path = npy_path.with_suffix(".json")
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        motion = np.load(str(npy_path)).astype(np.float64)
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            continue
        out.append((npy_path, meta, motion))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 oracle training data (Step 5-B)")
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
    alpha = netgain_weights["alpha"]
    beta = netgain_weights["beta"]
    gamma = netgain_weights["gamma"]

    motions = _load_g2_motions(args.g2_batch_dir)
    print(f"[INFO] loaded {len(motions)} G2 motions")
    evaluators = list(DEFAULT_EVALUATORS)

    tuples: list[dict[str, Any]] = []
    for npy_path, g2_meta, motion in motions:
        trial_id = g2_meta.get("trial_id", npy_path.stem)
        T = motion.shape[0]
        frame_range = (0, T - 1)

        reports_before = {ev.name: ev.evaluate(motion) for ev in evaluators}
        target_before = _target_score_all(reports_before)
        eval_scores = [_max_score(reports_before.get(n, [])) for n in ALL_EVALUATORS]
        # Primary artifact = max-score evaluator → artifact_kind.
        primary_ev = max(zip(ALL_EVALUATORS, eval_scores), key=lambda kv: kv[1])[0]
        primary_artifact = EVAL_TO_ARTIFACT[primary_ev]
        artifact_onehot = [1 if k == primary_artifact else 0 for k in ARTIFACT_KINDS]
        state = artifact_onehot + eval_scores  # 6-dim, Step 1 schema 와 호환.

        # 9 actions의 NetGain 측정 (Protocol B simplified, reference = original G2).
        candidates: list[dict[str, Any]] = []
        for tn, tool in TOOL_BY_NAME.items():
            for st in STRENGTHS:
                try:
                    corrected, report = tool.apply(motion, target_part=TOOL_TO_TARGET_PART[tn],
                                                   target_joints=[], frame_range=frame_range,
                                                   strength=st)
                    reports_after = {ev.name: ev.evaluate(corrected) for ev in evaluators}
                    target_after = _target_score_all(reports_after)
                    target_delta = target_after - target_before
                    fidelity_loss = _mpjpe(corrected, motion)
                    cm = float(report.correction_magnitude)
                    netgain = -target_delta - alpha * fidelity_loss - beta * cm - gamma * 1.0
                    candidates.append({
                        "tool_name": tn, "strength": st,
                        "target_delta": float(target_delta),
                        "fidelity_loss_protocol_b": float(fidelity_loss),
                        "correction_magnitude": cm,
                        "netgain": float(netgain),
                    })
                except ValueError:
                    candidates.append({
                        "tool_name": tn, "strength": st,
                        "netgain": float("-inf"), "skipped": True,
                    })

        valid = [c for c in candidates if not c.get("skipped")]
        best = max(valid, key=lambda c: c["netgain"]) if valid else None
        if best is None:
            continue

        tuples.append({
            "trial_id": trial_id,
            "g2_prompt": g2_meta.get("prompt", "")[:80],
            "artifact_kind": primary_artifact,
            "state": {
                "artifact_onehot": artifact_onehot,
                "evaluator_scores": dict(zip(ALL_EVALUATORS, eval_scores)),
                "feature_vector": state,
                "feature_names": [f"is_{k}" for k in ARTIFACT_KINDS] + [f"score_{n}" for n in ALL_EVALUATORS],
            },
            "label": {
                "best_tool": best["tool_name"],
                "best_strength": best["strength"],
                "best_target_part": TOOL_TO_TARGET_PART[best["tool_name"]],
                "best_netgain": best["netgain"],
            },
            "candidates": candidates,  # 9 actions all NetGains.
        })

    # Stats.
    tool_dist = Counter(t["label"]["best_tool"] for t in tuples)
    strength_dist = Counter(t["label"]["best_strength"] for t in tuples)
    joint_dist = Counter((t["label"]["best_tool"], t["label"]["best_strength"]) for t in tuples)

    summary = {
        "schema_version": "1.0.0",
        "record_type": "g2_oracle_training_data",
        "n_tuples": len(tuples),
        "feature_names": tuples[0]["state"]["feature_names"] if tuples else [],
        "fidelity_loss_reference": "original_g2_motion (Protocol B simplified)",
        "target_score_definition": "mean(FootFloating, BoneLength, VelocityJitter max scores)",
        "netgain_weight_status": "calibrated_protocol_a_v1",
        "netgain_weight_caveat": "α=5.0 from synthetic Protocol A — G2 transfer caveat.",
        "label_distribution": {
            "tool": dict(tool_dist),
            "strength": dict(strength_dist),
            "joint": {f"{k[0]}/{k[1]}": v for k, v in joint_dist.items()},
        },
        "tuples": tuples,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"\n[OK] wrote {args.output}")
    print(f"\nLabel distribution (n={len(tuples)}):")
    print(f"  tools: {dict(tool_dist)}")
    print(f"  strengths: {dict(strength_dist)}")
    print(f"\nJoint (tool, strength) distribution:")
    for k, v in sorted(joint_dist.items()):
        print(f"  {k[0]:25s} {k[1]:8s}: {v}")


if __name__ == "__main__":
    main()
