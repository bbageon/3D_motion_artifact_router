"""RL-1 sequence-oracle imitation training data builder.

사용자 directive (2026-05-25 RL-1 design):
> "제안된 6-dim state만으로는 약함. state에 최소한 이것들을 넣어야:
>  current evaluator scores / score delta from previous step / step index /
>  previous tool / previous strength / remaining budget"
>
> "RL-1 = sequence-oracle imitation policy (지도학습 sequence imitation, NOT
>  Q-learning, NOT 진짜 RL)."

본 도구는 oracle 의 best path 를 step-by-step 으로 re-simulate 하여 매 step 의
(state_t, action_t) pair 를 추출. 마지막 step 뒤에는 (state_L, STOP) 추가.

State (16-dim):
  1-3: current evaluator scores (FootFloating, BoneLength, VelocityJitter max).
  4-6: score delta from previous step (t=0 에서는 모두 0).
  7  : step index (0..K_max-1).
  8-11: previous tool one-hot (NONE / FootLockTool / BoneProjectionTool / VelocitySmoothingTool).
  12-15: previous strength one-hot (NONE / small / medium / large).
  16 : remaining budget (= K_max - step).

Action (10 actions):
  0: STOP
  1-3: FootLockTool / (small, medium, large)
  4-6: BoneProjectionTool / (small, medium, large)
  7-9: VelocitySmoothingTool / (small, medium, large)

Source oracle paths:
  - Synthetic multi-artifact: evals/snapshots/oracle_sequence_multi_v1.json (n=30).
  - G2 natural: evals/snapshots/oracle_sequence_g2_v1.json (n=50).

CLI:
    python -m tools.rl1_build_imitation_data \\
        --synthetic-oracle evals/snapshots/oracle_sequence_multi_v1.json \\
        --g2-oracle evals/snapshots/oracle_sequence_g2_v1.json \\
        --k-max 5 \\
        --output evals/snapshots/rl1_imitation_training_data_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, EvaluatorReport
from tools.synthetic_injection import inject_foot_floating, inject_jitter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMANML3D_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
DEFAULT_G2_DIR = REPO_ROOT / "external_assets" / "g2_generated_v1"

ALL_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
TOOL_NAMES = ["NONE", "FootLockTool", "BoneProjectionTool", "VelocitySmoothingTool"]
STRENGTH_NAMES = ["NONE", "small", "medium", "large"]
TOOL_TO_TARGET_PART = {
    "FootLockTool": "both_feet",
    "BoneProjectionTool": "right_arm",
    "VelocitySmoothingTool": "full_body",
}

#: 10-action joint encoding.
ACTIONS = [
    ("STOP", "NONE"),
    ("FootLockTool", "small"), ("FootLockTool", "medium"), ("FootLockTool", "large"),
    ("BoneProjectionTool", "small"), ("BoneProjectionTool", "medium"), ("BoneProjectionTool", "large"),
    ("VelocitySmoothingTool", "small"), ("VelocitySmoothingTool", "medium"), ("VelocitySmoothingTool", "large"),
]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}

STATE_FEATURE_NAMES = (
    [f"score_{n}" for n in ALL_EVALUATORS]
    + [f"delta_score_{n}" for n in ALL_EVALUATORS]
    + ["step_index"]
    + [f"prev_tool_is_{t}" for t in TOOL_NAMES]
    + [f"prev_strength_is_{s}" for s in STRENGTH_NAMES]
    + ["remaining_budget"]
)
assert len(STATE_FEATURE_NAMES) == 16, f"expected 16-dim state, got {len(STATE_FEATURE_NAMES)}"


def _max_score(reports: list[EvaluatorReport]) -> float:
    if not reports:
        return 0.0
    return float(max(r.score for r in reports))


def _eval_scores(motion: np.ndarray, evaluators: list[Any]) -> dict[str, float]:
    """3 evaluator max scores."""
    return {n: _max_score(ev.evaluate(motion)) for n, ev in zip(ALL_EVALUATORS, evaluators)}


def _make_state(
    *,
    current_scores: dict[str, float],
    prev_scores: Optional[dict[str, float]],
    step_index: int,
    prev_tool: str,
    prev_strength: str,
    k_max: int,
) -> list[float]:
    """16-dim state vector."""
    score_vec = [current_scores[n] for n in ALL_EVALUATORS]
    if prev_scores is None:
        delta_vec = [0.0] * 3
    else:
        delta_vec = [current_scores[n] - prev_scores[n] for n in ALL_EVALUATORS]
    step_vec = [float(step_index)]
    tool_onehot = [1.0 if t == prev_tool else 0.0 for t in TOOL_NAMES]
    strength_onehot = [1.0 if s == prev_strength else 0.0 for s in STRENGTH_NAMES]
    budget_vec = [float(k_max - step_index)]
    return score_vec + delta_vec + step_vec + tool_onehot + strength_onehot + budget_vec


def _multi_inject(clean: np.ndarray, seed: int) -> np.ndarray:
    """Step 3 multi-artifact recipe."""
    m1 = inject_foot_floating(clean, lift_height=0.08, seed=seed)
    return inject_jitter(m1, noise_std=0.05, seed=seed + 1000)


def _load_humanml3d_sample(trial_id: str, data_dir: Path) -> np.ndarray:
    path = data_dir / f"{trial_id}.npy"
    if not path.exists():
        raise FileNotFoundError(f"HumanML3D sample not found: {path}")
    return np.load(str(path)).astype(np.float64)


def _load_g2_sample(trial_id: str, batch_dir: Path) -> np.ndarray:
    path = batch_dir / f"{trial_id}.npy"
    if not path.exists():
        raise FileNotFoundError(f"G2 sample not found: {path}")
    return np.load(str(path)).astype(np.float64)


def _simulate_oracle_path(
    *,
    initial_motion: np.ndarray,
    oracle_sequence: list[list[str]],  # [[tool_name, target_part, strength], ...]
    tools_by_name: dict[str, CorrectionTool],
    evaluators: list[Any],
    k_max: int,
) -> list[dict[str, Any]]:
    """Oracle path 를 step-by-step 으로 simulate 하여 (state, action) pairs 추출.

    Returns: list of {"state": [16 floats], "action_id": int, "action": (tool, strength)}.
    Last entry 는 (state_L, STOP).
    """
    pairs = []
    motion = initial_motion.copy()
    T = motion.shape[0]
    frame_range = (0, T - 1)
    prev_scores: Optional[dict[str, float]] = None
    prev_tool = "NONE"
    prev_strength = "NONE"

    for step_idx in range(len(oracle_sequence) + 1):
        current_scores = _eval_scores(motion, evaluators)
        state = _make_state(
            current_scores=current_scores, prev_scores=prev_scores,
            step_index=step_idx, prev_tool=prev_tool, prev_strength=prev_strength,
            k_max=k_max,
        )
        if step_idx < len(oracle_sequence):
            tn, _, st = oracle_sequence[step_idx]
            action = (tn, st)
        else:
            action = ("STOP", "NONE")
        action_id = ACTION_TO_ID[action]
        pairs.append({
            "state": state,
            "action_id": action_id,
            "action": list(action),
            "step_index": step_idx,
            "current_scores": dict(current_scores),
        })
        if action[0] == "STOP":
            break
        # Apply action.
        tool = tools_by_name[action[0]]
        tp = TOOL_TO_TARGET_PART[action[0]]
        try:
            motion, _ = tool.apply(motion, target_part=tp, target_joints=[],
                                   frame_range=frame_range, strength=action[1])
        except ValueError as e:
            print(f"[WARN] apply failed at step {step_idx}: {e}", file=sys.stderr)
            break
        prev_scores = current_scores
        prev_tool = action[0]
        prev_strength = action[1]
    return pairs


def build_from_oracle_snapshot(
    *,
    oracle_path: Path,
    distribution: str,  # "synthetic_multi" or "g2_natural"
    k_max: int,
    tools_by_name: dict[str, CorrectionTool],
    evaluators: list[Any],
    humanml3d_dir: Path,
    g2_dir: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Oracle snapshot → list of (sample_id, list of (state, action) pairs)."""
    with open(oracle_path, encoding="utf-8") as f:
        oracle = json.load(f)
    per_sample = oracle.get("per_sample", [])

    all_pairs: list[dict[str, Any]] = []
    n_samples_processed = 0
    seq_lengths_seen: list[int] = []
    for s in per_sample:
        trial_id = s["trial_id"]
        if distribution == "synthetic_multi":
            best = s.get("best_A") or s.get("best")
            if best is None:
                continue
            oracle_seq = best["sequence"]
            try:
                clean = _load_humanml3d_sample(trial_id, humanml3d_dir)
            except FileNotFoundError:
                print(f"[WARN] missing humanml3d sample: {trial_id}", file=sys.stderr)
                continue
            initial_motion = _multi_inject(clean, seed=seed)
        elif distribution == "g2_natural":
            best = s.get("best")
            if best is None:
                continue
            oracle_seq = best["sequence"]
            try:
                initial_motion = _load_g2_sample(trial_id, g2_dir)
            except FileNotFoundError:
                print(f"[WARN] missing G2 sample: {trial_id}", file=sys.stderr)
                continue
        else:
            raise ValueError(distribution)

        pairs = _simulate_oracle_path(
            initial_motion=initial_motion, oracle_sequence=oracle_seq,
            tools_by_name=tools_by_name, evaluators=evaluators, k_max=k_max,
        )
        for p in pairs:
            p["trial_id"] = trial_id
            p["distribution"] = distribution
            all_pairs.append(p)
        seq_lengths_seen.append(len(oracle_seq))
        n_samples_processed += 1

    seq_len_dist = Counter(seq_lengths_seen)
    return all_pairs, {
        "n_samples_processed": n_samples_processed,
        "n_pairs_total": len(all_pairs),
        "sequence_length_distribution": dict(seq_len_dist),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-1 sequence imitation training data builder")
    parser.add_argument("--synthetic-oracle", type=Path, required=True)
    parser.add_argument("--g2-oracle", type=Path, required=True)
    parser.add_argument("--humanml3d-dir", type=Path, default=DEFAULT_HUMANML3D_DIR)
    parser.add_argument("--g2-dir", type=Path, default=DEFAULT_G2_DIR)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="synthetic injection seed (oracle 측정 시와 동일)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tools_by_name: dict[str, CorrectionTool] = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "BoneProjectionTool": BoneProjectionTool(),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
    }
    evaluators = list(DEFAULT_EVALUATORS)
    print(f"[INFO] state dimension: {len(STATE_FEATURE_NAMES)} (sanity 16 confirmed)")
    print(f"[INFO] action count: {len(ACTIONS)} (STOP + 9 tool×strength)")

    print(f"[INFO] processing synthetic oracle: {args.synthetic_oracle}")
    syn_pairs, syn_stats = build_from_oracle_snapshot(
        oracle_path=args.synthetic_oracle, distribution="synthetic_multi",
        k_max=args.k_max, tools_by_name=tools_by_name, evaluators=evaluators,
        humanml3d_dir=args.humanml3d_dir, g2_dir=args.g2_dir, seed=args.seed,
    )
    print(f"  synthetic: {syn_stats}")

    print(f"[INFO] processing G2 oracle: {args.g2_oracle}")
    g2_pairs, g2_stats = build_from_oracle_snapshot(
        oracle_path=args.g2_oracle, distribution="g2_natural",
        k_max=args.k_max, tools_by_name=tools_by_name, evaluators=evaluators,
        humanml3d_dir=args.humanml3d_dir, g2_dir=args.g2_dir, seed=args.seed,
    )
    print(f"  G2: {g2_stats}")

    all_pairs = syn_pairs + g2_pairs
    action_dist = Counter(p["action_id"] for p in all_pairs)
    action_dist_named = {f"{ACTIONS[i][0]}/{ACTIONS[i][1]}": v for i, v in action_dist.items()}

    summary = {
        "schema_version": "1.0.0",
        "record_type": "rl1_imitation_training_data",
        "k_max": args.k_max,
        "state_dim": len(STATE_FEATURE_NAMES),
        "state_feature_names": STATE_FEATURE_NAMES,
        "action_count": len(ACTIONS),
        "action_names": [f"{a[0]}/{a[1]}" for a in ACTIONS],
        "synthetic_stats": syn_stats,
        "g2_stats": g2_stats,
        "total_pairs": len(all_pairs),
        "action_distribution": action_dist_named,
        "pairs": all_pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {args.output}")
    print(f"  total pairs: {len(all_pairs)}")
    print(f"  action distribution:")
    for k, v in sorted(action_dist_named.items()):
        print(f"    {k:30s}: {v}")


if __name__ == "__main__":
    main()
