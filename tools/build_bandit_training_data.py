"""H-2026-205 Step 4 — Contextual bandit training data generation (offline simulator).

각 multi-artifact sample 에 모든 9 action (3 tool × 3 strength) + STOP (action=do nothing)
적용 → (state, action, reward) tuples. Per-step Q-learning training data.

State (full evaluator state):
  - FootFloatingEvaluator max score
  - BoneLengthEvaluator max score   ← Step 3.5 와 달리 포함! (사용자 directive)
  - VelocityJitterEvaluator max score
  → 3-dim feature vector.

Action (10 classes):
  - 'STOP': do nothing (corrected = motion).
  - 'FootLockTool_small', 'FootLockTool_medium', 'FootLockTool_large'
  - 'BoneProjectionTool_small/medium/large'
  - 'VelocitySmoothingTool_small/medium/large'
  → 10 actions.

Reward (per-step immediate):
  - target_score = mean(FootFloating max, VelocityJitter max).  ← Step 3, 3.5 와 동일 target.
  - ΔNetGain = (target_before - target_after) - α·FidelityLoss_step - β·CorrectionMag_step - γ·1.
  - alpha=5.0, beta=0.0, gamma=0.0 (calibrated_protocol_a_v1).
  - STOP: ΔNetGain = 0 (no change).

Trajectory 구조:
  - Initial state = inject(clean, multi-artifact).
  - 각 sample 에 9 actions 각각 적용 → 9 trajectories of 1 step.
  - 또는 multi-step trajectories: sample × random action sequences (depth ≤ 3).

본 도구는 **per-step (state, action, reward) tuples only** (1-step bandit, offline).
Multi-step value learning 은 후속 (Step 4-2 또는 H-2026-205 Stage 2).

CLI:
    python -m tools.build_bandit_training_data \\
        --n-samples 60 --seed 42 \\
        --output evals/snapshots/bandit_training_data_v1.json
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
from orchestrator.supervised_selector import ARTIFACT_TO_TARGET_PART
from tools.synthetic_injection import inject_foot_floating, inject_jitter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"

TOOL_BY_NAME = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}
TARGET_EVALUATORS = ("FootFloatingEvaluator", "VelocityJitterEvaluator")
STATE_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
TOOL_NAMES = ["FootLockTool", "BoneProjectionTool", "VelocitySmoothingTool"]
STRENGTHS = ["small", "medium", "large"]

#: action_id (0..9) ↔ (tool, strength). 0 = STOP, 1..9 = 3 tool × 3 strength.
ACTIONS: list[tuple[str, str]] = [("STOP", "none")]
for tn in TOOL_NAMES:
    for st in STRENGTHS:
        ACTIONS.append((tn, st))


def _multi_inject(clean: np.ndarray, seed: int) -> np.ndarray:
    m1 = inject_foot_floating(clean, lift_height=0.08, seed=seed)
    return inject_jitter(m1, noise_std=0.05, seed=seed + 1000)


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _max_score(reports: list[EvaluatorReport]) -> float:
    if not reports:
        return 0.0
    return float(max(r.score for r in reports))


def _state(reports_dict: dict[str, list[EvaluatorReport]]) -> list[float]:
    """Full evaluator state (3-dim, BoneLength 포함)."""
    return [_max_score(reports_dict.get(name, [])) for name in STATE_EVALUATORS]


def _target_score(reports_dict: dict[str, list[EvaluatorReport]]) -> float:
    """Target = mean(FootFloating max, VelocityJitter max). BoneLength 미포함 (Step 3/3.5 와 일관)."""
    return float(np.mean([_max_score(reports_dict.get(name, [])) for name in TARGET_EVALUATORS]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bandit training data generation (offline simulator)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
    alpha = netgain_weights["alpha"]
    beta = netgain_weights["beta"]
    gamma = netgain_weights["gamma"]

    rng = np.random.default_rng(args.seed)
    npy_files = sorted(args.data_dir.glob("*.npy"))
    n = min(args.n_samples, len(npy_files))
    chosen_idx = rng.choice(len(npy_files), size=n, replace=False)
    chosen = [npy_files[i] for i in chosen_idx]
    evaluators = list(DEFAULT_EVALUATORS)

    tuples: list[dict[str, Any]] = []
    for path in chosen:
        trial_id = path.stem
        try:
            clean = np.load(str(path)).astype(np.float64)
        except Exception as e:
            print(f"[WARN] {trial_id}: {e}")
            continue
        if clean.ndim != 3 or clean.shape[1] != 22 or clean.shape[2] != 3:
            continue

        corrupted = _multi_inject(clean, args.seed)
        T = corrupted.shape[0]
        frame_range = (0, T - 1)
        reports_before = {ev.name: ev.evaluate(corrupted) for ev in evaluators}
        state = _state(reports_before)
        target_before = _target_score(reports_before)
        mpjpe_corrupted = _mpjpe(corrupted, clean)

        for action_idx, (tool_name, strength) in enumerate(ACTIONS):
            if tool_name == "STOP":
                # No-op: reward = 0.
                reward = 0.0
                tuples.append({
                    "trial_id": trial_id,
                    "state": state,
                    "action_id": action_idx,
                    "tool_name": tool_name,
                    "strength": strength,
                    "reward_netgain": reward,
                    "target_before": target_before,
                    "target_after": target_before,
                    "target_delta": 0.0,
                    "fidelity_loss_protocol_a": 0.0,
                    "correction_magnitude": 0.0,
                })
                continue

            tool = TOOL_BY_NAME[tool_name]
            target_part = ARTIFACT_TO_TARGET_PART.get(
                {"FootLockTool": "foot_floating", "BoneProjectionTool": "bone_stretch_right_arm",
                 "VelocitySmoothingTool": "global_jitter"}[tool_name],
                "full_body",
            )
            try:
                corrected, report = tool.apply(corrupted, target_part=target_part, target_joints=[],
                                               frame_range=frame_range, strength=strength)
                correction_mag = float(report.correction_magnitude)
            except ValueError as e:
                # 적용 불가 — skip.
                tuples.append({
                    "trial_id": trial_id,
                    "state": state,
                    "action_id": action_idx,
                    "tool_name": tool_name,
                    "strength": strength,
                    "reward_netgain": float("-inf"),
                    "target_before": target_before,
                    "target_after": target_before,
                    "target_delta": 0.0,
                    "fidelity_loss_protocol_a": 0.0,
                    "correction_magnitude": 0.0,
                    "skipped": True,
                    "skip_reason": str(e),
                })
                continue

            reports_after = {ev.name: ev.evaluate(corrected) for ev in evaluators}
            target_after = _target_score(reports_after)
            target_delta = target_after - target_before
            mpjpe_corrected = _mpjpe(corrected, clean)
            fidelity_loss = mpjpe_corrected - mpjpe_corrupted
            artifact_reduction = -target_delta
            reward = artifact_reduction - alpha * fidelity_loss - beta * correction_mag - gamma * 1.0

            tuples.append({
                "trial_id": trial_id,
                "state": state,
                "action_id": action_idx,
                "tool_name": tool_name,
                "strength": strength,
                "reward_netgain": float(reward),
                "target_before": target_before,
                "target_after": target_after,
                "target_delta": float(target_delta),
                "fidelity_loss_protocol_a": float(fidelity_loss),
                "correction_magnitude": float(correction_mag),
            })

    # Stats.
    n_skip = sum(1 for t in tuples if t.get("skipped"))
    action_counts = Counter(t["action_id"] for t in tuples)
    best_action_per_trial: dict[str, tuple[int, float]] = {}
    for t in tuples:
        if t.get("skipped"):
            continue
        trial = t["trial_id"]
        if trial not in best_action_per_trial or t["reward_netgain"] > best_action_per_trial[trial][1]:
            best_action_per_trial[trial] = (t["action_id"], t["reward_netgain"])
    best_action_dist = Counter(v[0] for v in best_action_per_trial.values())

    summary = {
        "schema_version": "1.0.0",
        "record_type": "bandit_training_data_v1",
        "n_tuples": len(tuples),
        "n_unique_trials": len({t["trial_id"] for t in tuples}),
        "n_actions": len(ACTIONS),
        "n_state_features": len(STATE_EVALUATORS),
        "action_index_map": {i: f"{a[0]}/{a[1]}" for i, a in enumerate(ACTIONS)},
        "state_features": list(STATE_EVALUATORS),
        "target_evaluators": list(TARGET_EVALUATORS),
        "target_score_definition": "mean(FootFloating_max, VelocityJitter_max)",
        "netgain_weight_status": "calibrated_protocol_a_v1",
        "netgain_weights": dict(netgain_weights),
        "n_skipped": n_skip,
        "best_action_distribution": {
            ACTIONS[k][0] + "/" + ACTIONS[k][1]: v for k, v in best_action_dist.items()
        },
        "action_counts_per_trial": {f"{ACTIONS[k][0]}/{ACTIONS[k][1]}": v for k, v in action_counts.items()},
        "tuples": tuples,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"[OK] wrote {len(tuples)} tuples to {args.output}")
    print(f"  n_unique_trials: {summary['n_unique_trials']}")
    print(f"  n_skipped: {n_skip}")
    print(f"\nbest-action-per-trial distribution:")
    for k, v in sorted(best_action_dist.items()):
        print(f"  action {k} ({ACTIONS[k][0]}/{ACTIONS[k][1]}): {v}")


if __name__ == "__main__":
    main()
