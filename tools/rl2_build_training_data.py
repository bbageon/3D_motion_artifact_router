"""Step F-1 (사용자 directive 2026-05-27): RL-2 safe imitation training data builder.

사용자 directive:
> "Safe oracle 결과에서 state-action pair를 뽑습니다. G2 n=300 safe_best + synthetic
>  severe n=60 safe_best. 각 step마다 state_t → action_t 저장. State: artifact scores 3
>  + physical gate scores 5 + score deltas + prev action one-hot 16 + remaining budget 1
>  + step index 1 + distribution tag optional."

본 도구는 두 safe oracle snapshot 의 safe_best sequence 를 replay 하며 각 step 의
state-action pair 추출 → RL-2 imitation 의 training dataset.

Action (16-class): STOP + 3 tool × 5 strength (target_part 고정).
  0 = STOP
  1-5  = FootLockTool|{xsmall,small5,medium5,large5,xlarge} (both_feet)
  6-10 = BoneProjectionTool|... (right_arm)
  11-15= VelocitySmoothingTool|... (full_body)

State (per step):
  artifact_scores 3 (FootFloating / BoneLength / VelocityJitter max)
  physical_scores 5 (Penetrate / Float / Skate / JerkSpike / BoneLengthCV)
  score_delta 8 (artifact 3 + physical 5, prev step 대비, init 0)
  prev_action_onehot 16
  remaining_budget 1
  step_index 1
  distribution_tag 1 (0=G2, 1=synthetic)  ← optional column (분석 시 include/exclude)

CLI:
    python -m tools.rl2_build_training_data \
        --g2-oracle evals/snapshots/safe_sequence_oracle_g2_hml3d_test300_v1.json \
        --g2-batch-dir external_assets/g2_generated_hml3d_test300_seed20260527 \
        --synthetic-oracle evals/snapshots/safe_sequence_oracle_synthetic_severe_v1.json \
        --output evals/snapshots/rl2_imitation_dataset_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS, EvaluatorReport
from tools.synthetic_injection import inject_foot_floating, inject_jitter

ARTIFACT_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
PHYSICAL_EVALUATORS = ("PenetrateEvaluator", "FloatEvaluator", "SkateEvaluator",
                       "JerkSpikeEvaluator", "BoneLengthCVEvaluator")
STRENGTHS_5LEVEL = ("xsmall", "small5", "medium5", "large5", "xlarge")
TOOLS_ORDER = ("FootLockTool", "BoneProjectionTool", "VelocitySmoothingTool")
TOOL_TARGET = {"FootLockTool": "both_feet", "BoneProjectionTool": "right_arm",
               "VelocitySmoothingTool": "full_body"}

# Action index mapping (16-class).
ACTION_LIST = ["STOP"]
for _tool in TOOLS_ORDER:
    for _st in STRENGTHS_5LEVEL:
        ACTION_LIST.append(f"{_tool}|{_st}")
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}

TOOL_BY_NAME: dict[str, CorrectionTool] = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _artifact_scores(motion: np.ndarray, evaluators: list) -> list[float]:
    by = {ev.name: ev.evaluate(motion) for ev in evaluators}
    return [_max_score(by.get(n, [])) for n in ARTIFACT_EVALUATORS]


def _physical_scores(motion: np.ndarray, gate_evaluators: list) -> list[float]:
    by = {}
    for ev in gate_evaluators:
        try:
            by[ev.name] = _max_score(ev.evaluate(motion))
        except Exception:
            by[ev.name] = 0.0
    return [by.get(n, 0.0) for n in PHYSICAL_EVALUATORS]


def _action_to_idx(step: list) -> int:
    """oracle sequence step [tool, target, strength] → action index."""
    tool, _target, strength = step[0], step[1], step[2]
    key = f"{tool}|{strength}"
    return ACTION_TO_IDX.get(key, 0)


def _build_state(artifact: list[float], physical: list[float], delta: list[float],
                 prev_action_idx: int, remaining_budget: int, step_index: int,
                 dist_tag: int) -> dict:
    prev_onehot = [0] * 16
    prev_onehot[prev_action_idx] = 1
    return {
        "artifact_scores": artifact,        # 3
        "physical_scores": physical,        # 5
        "score_delta": delta,               # 8
        "prev_action_onehot": prev_onehot,  # 16
        "remaining_budget": remaining_budget,
        "step_index": step_index,
        "distribution_tag": dist_tag,
    }


def _replay_sample(motion0: np.ndarray, safe_seq: list, max_depth: int,
                   dist_tag: int, evaluators: list, gate_evaluators: list) -> list[dict]:
    """Replay safe_best sequence → list of (state, action_idx) pairs."""
    pairs = []
    motion = motion0.copy()
    T = motion.shape[0]
    artifact = _artifact_scores(motion, evaluators)
    physical = _physical_scores(motion, gate_evaluators)
    prev_artifact = list(artifact)
    prev_physical = list(physical)
    prev_action_idx = 0  # STOP as "no previous action".
    n_steps = len(safe_seq)

    for t in range(n_steps + 1):
        delta = [a - pa for a, pa in zip(artifact, prev_artifact)] + \
                [p - pp for p, pp in zip(physical, prev_physical)]
        remaining_budget = max_depth - t
        if t < n_steps:
            action_idx = _action_to_idx(safe_seq[t])
        else:
            action_idx = 0  # STOP (terminal).
        state = _build_state(artifact, physical, delta, prev_action_idx,
                             remaining_budget, t, dist_tag)
        pairs.append({"state": state, "action_idx": action_idx,
                      "action_name": ACTION_LIST[action_idx]})
        if t >= n_steps:
            break
        # Apply action to advance state.
        step = safe_seq[t]
        tool = TOOL_BY_NAME[step[0]]
        motion, _ = tool.apply(motion, target_part=step[1], target_joints=[],
                                frame_range=(0, T - 1), strength=step[2])
        prev_artifact, prev_physical = list(artifact), list(physical)
        artifact = _artifact_scores(motion, evaluators)
        physical = _physical_scores(motion, gate_evaluators)
        prev_action_idx = action_idx
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_g2_hml3d_test300_v1.json")
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--synthetic-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_synthetic_severe_v1.json")
    parser.add_argument("--synthetic-data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_imitation_dataset_v1.json")
    args = parser.parse_args()

    evaluators = list(DEFAULT_EVALUATORS)
    gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)

    all_rows = []  # each: {sample_id, distribution, step, state, action_idx, action_name}

    # === G2 n=300 ===
    g2 = json.load(open(args.g2_oracle, encoding="utf-8"))
    print(f"[INFO] G2 oracle: {len(g2['per_sample'])} samples")
    for i, ps in enumerate(g2["per_sample"], 1):
        tid = ps["trial_id"]
        sb = ps.get("safe_best")
        if sb is None:
            continue
        npy = args.g2_batch_dir / f"{tid}.npy"
        if not npy.exists():
            continue
        motion0 = np.load(str(npy)).astype(np.float64)
        safe_seq = sb["sequence"] if sb["length"] > 0 else []
        pairs = _replay_sample(motion0, safe_seq, args.max_depth, dist_tag=0,
                               evaluators=evaluators, gate_evaluators=gate_evaluators)
        for step_i, pr in enumerate(pairs):
            all_rows.append({"sample_id": tid, "distribution": "g2", "step": step_i, **pr})
        if i % 50 == 0:
            print(f"   G2 {i}/{len(g2['per_sample'])} processed")

    # === Synthetic severe n=60 ===
    syn = json.load(open(args.synthetic_oracle, encoding="utf-8"))
    print(f"[INFO] synthetic oracle: {len(syn['per_sample'])} samples")
    for i, ps in enumerate(syn["per_sample"], 1):
        tid = ps["trial_id"]
        sb = ps.get("safe_best")
        if sb is None:
            continue
        npy = args.synthetic_data_dir / f"{tid}.npy"
        if not npy.exists():
            continue
        clean = np.load(str(npy)).astype(np.float64)
        m1 = inject_foot_floating(clean, lift_height=0.08, seed=args.synthetic_seed)
        corrupted = inject_jitter(m1, noise_std=0.05, seed=args.synthetic_seed + 1000)
        safe_seq = sb["sequence"] if sb["length"] > 0 else []
        pairs = _replay_sample(corrupted, safe_seq, args.max_depth, dist_tag=1,
                               evaluators=evaluators, gate_evaluators=gate_evaluators)
        for step_i, pr in enumerate(pairs):
            all_rows.append({"sample_id": tid, "distribution": "synthetic", "step": step_i, **pr})
        if i % 20 == 0:
            print(f"   synthetic {i}/{len(syn['per_sample'])} processed")

    # Action distribution.
    from collections import Counter
    action_dist = Counter(r["action_name"] for r in all_rows)
    g2_actions = Counter(r["action_name"] for r in all_rows if r["distribution"] == "g2")
    syn_actions = Counter(r["action_name"] for r in all_rows if r["distribution"] == "synthetic")

    out = {
        "schema_version": "1.0.0",
        "record_type": "rl2_imitation_dataset",
        "task_id": "rl2_imitation_dataset_v1",
        "action_list": ACTION_LIST,
        "state_spec": {
            "artifact_scores": ARTIFACT_EVALUATORS, "physical_scores": PHYSICAL_EVALUATORS,
            "score_delta_dim": 8, "prev_action_onehot_dim": 16,
            "scalars": ["remaining_budget", "step_index", "distribution_tag"],
        },
        "max_depth": args.max_depth,
        "n_rows": len(all_rows),
        "n_g2_rows": sum(1 for r in all_rows if r["distribution"] == "g2"),
        "n_synthetic_rows": sum(1 for r in all_rows if r["distribution"] == "synthetic"),
        "action_distribution_all": dict(action_dist),
        "action_distribution_g2": dict(g2_actions),
        "action_distribution_synthetic": dict(syn_actions),
        "rows": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== RL-2 Imitation Dataset (Step F-1) ===")
    print(f"  total rows: {len(all_rows)} (G2 {out['n_g2_rows']}, synthetic {out['n_synthetic_rows']})")
    print(f"  G2 action dist (top): {dict(sorted(g2_actions.items(), key=lambda x: -x[1])[:6])}")
    print(f"  synthetic action dist (top): {dict(sorted(syn_actions.items(), key=lambda x: -x[1])[:6])}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
