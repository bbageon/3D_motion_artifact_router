"""RL-1 sanity check 시각화 — 사용자 directive (2026-05-25) 의 4 question evidence.

사용자 4 question:
  1. G2 STOP-best sample: 눈으로도 STOP 이 맞는가?
  2. G2 sample 에 B2 강제 적용 시 motion 이 뭉개지거나 이상해지는가? (over-modification visual)
  3. Synthetic RL-1 실패 sample 의 남은 artifact 시각?
  4. Synthetic sequence oracle 의 output 이 눈으로도 더 좋은가?

본 도구는 3 sample 의 motion 위에 4-way 비교 (corrupted/input / RL-1 / oracle /
B2) 의 metric time-series 를 정적 PNG 로 저장. plot_3d_motion 의 GIF 는 compute
heavy 라 skip — metric 시각화 우선.

Outputs (reports/figures/2026-05-25/):
  - g2_motion_004_sanity.png: G2 STOP-best, original vs B2 forced 의 evaluator
    score time-series + foot height + bone length variation + jitter.
  - synthetic_002652_fail_sanity.png: Synthetic RL-1 fail, clean vs corrupted vs
    RL-1 (rolled back) vs oracle 의 4-way time-series.
  - synthetic_002806_success_sanity.png: Synthetic RL-1 success (length=5),
    clean vs corrupted vs RL-1 vs oracle 의 4-way time-series.

CLI:
    python -m tools.rl1_sanity_visualize \\
        --rl1-snapshot evals/snapshots/baseline_rl1_imitation_v1.json \\
        --syn-oracle evals/snapshots/oracle_sequence_multi_v1.json \\
        --g2-oracle evals/snapshots/oracle_sequence_g2_v1.json \\
        --output-dir reports/figures/2026-05-25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, EvaluatorReport
from tools.synthetic_injection import inject_foot_floating, inject_jitter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMANML3D_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
DEFAULT_G2_DIR = REPO_ROOT / "external_assets" / "g2_generated_v1"

# SMPL 22 joint indices.
PELVIS = 0
LEFT_FOOT = 10
RIGHT_FOOT = 11
RIGHT_SHOULDER = 17
RIGHT_ELBOW = 19
RIGHT_WRIST = 21

TOOL_BY_NAME = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}
TOOL_TO_TARGET_PART = {
    "FootLockTool": "both_feet",
    "BoneProjectionTool": "right_arm",
    "VelocitySmoothingTool": "full_body",
}


def _multi_inject(clean: np.ndarray, seed: int) -> np.ndarray:
    m1 = inject_foot_floating(clean, lift_height=0.08, seed=seed)
    return inject_jitter(m1, noise_std=0.05, seed=seed + 1000)


def _apply_sequence(motion: np.ndarray, sequence: list[list[str]]) -> np.ndarray:
    """Sequence 의 (tool, target_part, strength) 를 순차 적용."""
    T = motion.shape[0]
    frame_range = (0, T - 1)
    out = motion.copy()
    for step in sequence:
        tn, tp, st = step[0], step[1], step[2]
        if tn in ("STOP", "APPLY_FAIL", "SCORE_VIOLATION_ROLLBACK"):
            break
        tool = TOOL_BY_NAME[tn]
        out, _ = tool.apply(out, target_part=tp, target_joints=[], frame_range=frame_range, strength=st)
    return out


def _apply_b2_g2(motion: np.ndarray) -> np.ndarray:
    """B2 G2: VelocitySmoothing(full_body, medium) 1회."""
    T = motion.shape[0]
    tool = TOOL_BY_NAME["VelocitySmoothingTool"]
    out, _ = tool.apply(motion, target_part="full_body", target_joints=[],
                         frame_range=(0, T - 1), strength="medium")
    return out


def _apply_b2_synthetic(motion: np.ndarray) -> np.ndarray:
    """B2 synthetic: same as G2."""
    return _apply_b2_g2(motion)


def _foot_height_time(motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """좌/우 foot 의 ground 위 height time series. Pelvis-relative Y."""
    left = motion[:, LEFT_FOOT, 1] - motion[:, PELVIS, 1]
    right = motion[:, RIGHT_FOOT, 1] - motion[:, PELVIS, 1]
    return left, right


def _right_arm_bone_lengths(motion: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """오른팔 의 3 bone length time series."""
    upper = np.linalg.norm(motion[:, RIGHT_ELBOW] - motion[:, RIGHT_SHOULDER], axis=-1)
    fore = np.linalg.norm(motion[:, RIGHT_WRIST] - motion[:, RIGHT_ELBOW], axis=-1)
    full = np.linalg.norm(motion[:, RIGHT_WRIST] - motion[:, RIGHT_SHOULDER], axis=-1)
    return upper, fore, full


def _jitter_time(motion: np.ndarray) -> np.ndarray:
    """Frame 별 mean acceleration norm (proxy for jitter)."""
    if motion.shape[0] < 3:
        return np.zeros(motion.shape[0])
    vel = np.diff(motion, axis=0)
    acc = np.diff(vel, axis=0)
    acc_norm = np.linalg.norm(acc, axis=-1).mean(axis=-1)
    # Pad to T frames.
    return np.concatenate([[0.0, 0.0], acc_norm])


def _evaluator_scores_summary(motion: np.ndarray, evaluators: list[Any]) -> dict[str, float]:
    out = {}
    for ev in evaluators:
        reports = ev.evaluate(motion)
        out[ev.name] = float(max((r.score for r in reports), default=0.0))
    return out


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _plot_4way(
    *,
    title: str, save_path: Path, motions: dict[str, np.ndarray],
    reference: Optional[np.ndarray] = None,
    extra_text: Optional[str] = None,
) -> None:
    """4-way time-series plot: foot height + bone length variation + jitter."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    colors = {"clean": "black", "corrupted": "red", "original_g2": "red",
              "RL-1": "tab:blue", "oracle": "tab:green", "B2": "tab:orange"}

    # Foot heights.
    ax = axes[0]
    for name, m in motions.items():
        c = colors.get(name, "gray")
        l, r = _foot_height_time(m)
        ax.plot(l, color=c, linestyle="--", alpha=0.7, label=f"{name} (L foot)")
        ax.plot(r, color=c, linestyle="-", alpha=0.7, label=f"{name} (R foot)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("foot height (Pelvis-relative Y)")
    ax.set_title("Foot height over time (artifact = foot floating)", fontsize=10)
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # Right arm bone length.
    ax = axes[1]
    for name, m in motions.items():
        c = colors.get(name, "gray")
        upper, fore, _ = _right_arm_bone_lengths(m)
        ax.plot(upper, color=c, linestyle="--", alpha=0.7, label=f"{name} (upper arm)")
        ax.plot(fore, color=c, linestyle="-", alpha=0.7, label=f"{name} (forearm)")
    ax.set_ylabel("right-arm bone length")
    ax.set_title("Right-arm bone length variation over time (artifact = bone stretch)", fontsize=10)
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # Jitter.
    ax = axes[2]
    for name, m in motions.items():
        c = colors.get(name, "gray")
        j = _jitter_time(m)
        ax.plot(j, color=c, alpha=0.7, label=name)
    ax.set_ylabel("mean acceleration norm")
    ax.set_xlabel("frame")
    ax.set_title("Velocity jitter over time (artifact = global jitter)", fontsize=10)
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    if extra_text:
        fig.text(0.5, 0.005, extra_text, ha="center", fontsize=8, family="monospace")
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-1 sanity check visualization")
    parser.add_argument("--rl1-snapshot", type=Path, required=True)
    parser.add_argument("--syn-oracle", type=Path, required=True)
    parser.add_argument("--g2-oracle", type=Path, required=True)
    parser.add_argument("--humanml3d-dir", type=Path, default=DEFAULT_HUMANML3D_DIR)
    parser.add_argument("--g2-dir", type=Path, default=DEFAULT_G2_DIR)
    parser.add_argument("--seed", type=int, default=42, help="synthetic injection seed (oracle 측정 시와 동일)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--g2-sample", type=str, default="motion_004",
                        help="G2 STOP-best sample trial_id (RL-1 selected STOP).")
    parser.add_argument("--syn-fail-sample", type=str, default="002652",
                        help="Synthetic sample where RL-1 score_violation_rollback.")
    parser.add_argument("--syn-success-sample", type=str, default="002806",
                        help="Synthetic sample where RL-1 length=5 (mimic oracle).")
    args = parser.parse_args()

    evaluators = list(DEFAULT_EVALUATORS)

    # Load snapshots.
    with open(args.rl1_snapshot, encoding="utf-8") as f:
        rl1 = json.load(f)
    with open(args.syn_oracle, encoding="utf-8") as f:
        syn_oracle = json.load(f)
    with open(args.g2_oracle, encoding="utf-8") as f:
        g2_oracle = json.load(f)

    syn_oracle_by_trial = {r["trial_id"]: r for r in syn_oracle["per_sample"] if r.get("best_A")}
    g2_oracle_by_trial = {r["trial_id"]: r for r in g2_oracle["per_sample"] if r.get("best")}
    rl1_syn_by_trial = {r["trial_id"]: r for r in rl1["per_sample_synthetic"]}
    rl1_g2_by_trial = {r["trial_id"]: r for r in rl1["per_sample_g2"]}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # === G2 STOP-best sample ===
    g2_id = args.g2_sample
    print(f"\n[1/3] G2 STOP-best sample {g2_id} (question: 눈으로도 STOP 이 맞는가? B2 가 over-modification?)")
    g2_path = args.g2_dir / f"{g2_id}.npy"
    g2_motion = np.load(str(g2_path)).astype(np.float64)
    rl1_g2 = rl1_g2_by_trial.get(g2_id)
    oracle_g2 = g2_oracle_by_trial.get(g2_id)

    # RL-1 output = STOP, so just original.
    # B2 forced = VS/medium 1회.
    b2_g2_motion = _apply_b2_g2(g2_motion)
    # Oracle = best sequence.
    if oracle_g2 and oracle_g2["best"]["length"] > 0:
        oracle_g2_motion = _apply_sequence(g2_motion, oracle_g2["best"]["sequence"])
    else:
        oracle_g2_motion = g2_motion.copy()  # STOP best.

    # Scores.
    s_orig = _evaluator_scores_summary(g2_motion, evaluators)
    s_b2 = _evaluator_scores_summary(b2_g2_motion, evaluators)
    s_oracle = _evaluator_scores_summary(oracle_g2_motion, evaluators)
    mpjpe_b2 = _mpjpe(b2_g2_motion, g2_motion)
    mpjpe_oracle = _mpjpe(oracle_g2_motion, g2_motion)

    extra = (
        f"original_g2 scores: FF={s_orig['FootFloatingEvaluator']:.4f}, BL={s_orig['BoneLengthEvaluator']:.4f}, VJ={s_orig['VelocityJitterEvaluator']:.4f}\n"
        f"B2 (VS/medium) scores: FF={s_b2['FootFloatingEvaluator']:.4f}, BL={s_b2['BoneLengthEvaluator']:.4f}, VJ={s_b2['VelocityJitterEvaluator']:.4f} | MPJPE_to_g2={mpjpe_b2:.4f}\n"
        f"oracle ({oracle_g2['best']['length']}-step) scores: FF={s_oracle['FootFloatingEvaluator']:.4f}, BL={s_oracle['BoneLengthEvaluator']:.4f}, VJ={s_oracle['VelocityJitterEvaluator']:.4f} | MPJPE_to_g2={mpjpe_oracle:.4f}\n"
        f"RL-1 decision: STOP (selector_stop). NetGain(RL-1)={rl1_g2['netgain']:+.5f} | NetGain(oracle)={oracle_g2['best']['netgain']:+.5f}"
    )
    _plot_4way(
        title=f"G2 STOP-best sample {g2_id} — RL-1 STOP vs B2 vs oracle",
        save_path=args.output_dir / f"g2_{g2_id}_sanity.png",
        motions={"original_g2": g2_motion, "B2": b2_g2_motion, "oracle": oracle_g2_motion},
        extra_text=extra,
    )

    # === Synthetic FAIL sample ===
    syn_fail_id = args.syn_fail_sample
    print(f"\n[2/3] Synthetic RL-1 FAIL sample {syn_fail_id} (question: 남은 artifact?)")
    syn_path = args.humanml3d_dir / f"{syn_fail_id}.npy"
    clean = np.load(str(syn_path)).astype(np.float64)
    corrupted = _multi_inject(clean, seed=args.seed)
    rl1_syn = rl1_syn_by_trial.get(syn_fail_id)
    oracle_syn = syn_oracle_by_trial.get(syn_fail_id)

    # Apply RL-1 sequence (rollback handled — apply until special marker).
    rl1_motion = _apply_sequence(corrupted, rl1_syn["sequence"]) if rl1_syn else corrupted.copy()
    oracle_motion = _apply_sequence(corrupted, oracle_syn["best_A"]["sequence"]) if oracle_syn else corrupted.copy()
    b2_motion = _apply_b2_synthetic(corrupted)

    s_clean = _evaluator_scores_summary(clean, evaluators)
    s_corr = _evaluator_scores_summary(corrupted, evaluators)
    s_rl1 = _evaluator_scores_summary(rl1_motion, evaluators)
    s_oracle_syn = _evaluator_scores_summary(oracle_motion, evaluators)
    s_b2_syn = _evaluator_scores_summary(b2_motion, evaluators)
    extra = (
        f"clean scores: FF={s_clean['FootFloatingEvaluator']:.4f}, BL={s_clean['BoneLengthEvaluator']:.4f}, VJ={s_clean['VelocityJitterEvaluator']:.4f}\n"
        f"corrupted scores: FF={s_corr['FootFloatingEvaluator']:.4f}, BL={s_corr['BoneLengthEvaluator']:.4f}, VJ={s_corr['VelocityJitterEvaluator']:.4f}\n"
        f"RL-1 (len {rl1_syn['length']}, {rl1_syn['stop_reason']}) scores: FF={s_rl1['FootFloatingEvaluator']:.4f}, BL={s_rl1['BoneLengthEvaluator']:.4f}, VJ={s_rl1['VelocityJitterEvaluator']:.4f} | NetGain={rl1_syn['netgain_A']:+.4f}\n"
        f"oracle (len {oracle_syn['best_A']['length']}) scores: FF={s_oracle_syn['FootFloatingEvaluator']:.4f}, BL={s_oracle_syn['BoneLengthEvaluator']:.4f}, VJ={s_oracle_syn['VelocityJitterEvaluator']:.4f} | NetGain={oracle_syn['best_A']['netgain_A']:+.4f}\n"
        f"B2 (VS/medium) scores: FF={s_b2_syn['FootFloatingEvaluator']:.4f}, BL={s_b2_syn['BoneLengthEvaluator']:.4f}, VJ={s_b2_syn['VelocityJitterEvaluator']:.4f}"
    )
    _plot_4way(
        title=f"Synthetic RL-1 FAIL {syn_fail_id} — clean / corrupted / RL-1 (rolled back) / oracle / B2",
        save_path=args.output_dir / f"synthetic_{syn_fail_id}_fail_sanity.png",
        motions={"clean": clean, "corrupted": corrupted, "RL-1": rl1_motion, "oracle": oracle_motion, "B2": b2_motion},
        extra_text=extra,
    )

    # === Synthetic SUCCESS sample ===
    syn_succ_id = args.syn_success_sample
    print(f"\n[3/3] Synthetic RL-1 SUCCESS sample {syn_succ_id} (question: 눈으로도 oracle 만큼 좋은가?)")
    syn_path = args.humanml3d_dir / f"{syn_succ_id}.npy"
    clean = np.load(str(syn_path)).astype(np.float64)
    corrupted = _multi_inject(clean, seed=args.seed)
    rl1_syn = rl1_syn_by_trial.get(syn_succ_id)
    oracle_syn = syn_oracle_by_trial.get(syn_succ_id)

    rl1_motion = _apply_sequence(corrupted, rl1_syn["sequence"]) if rl1_syn else corrupted.copy()
    oracle_motion = _apply_sequence(corrupted, oracle_syn["best_A"]["sequence"]) if oracle_syn else corrupted.copy()
    b2_motion = _apply_b2_synthetic(corrupted)

    s_clean = _evaluator_scores_summary(clean, evaluators)
    s_corr = _evaluator_scores_summary(corrupted, evaluators)
    s_rl1 = _evaluator_scores_summary(rl1_motion, evaluators)
    s_oracle_syn = _evaluator_scores_summary(oracle_motion, evaluators)
    s_b2_syn = _evaluator_scores_summary(b2_motion, evaluators)
    extra = (
        f"clean scores: FF={s_clean['FootFloatingEvaluator']:.4f}, BL={s_clean['BoneLengthEvaluator']:.4f}, VJ={s_clean['VelocityJitterEvaluator']:.4f}\n"
        f"corrupted scores: FF={s_corr['FootFloatingEvaluator']:.4f}, BL={s_corr['BoneLengthEvaluator']:.4f}, VJ={s_corr['VelocityJitterEvaluator']:.4f}\n"
        f"RL-1 (len {rl1_syn['length']}, {rl1_syn['stop_reason']}) scores: FF={s_rl1['FootFloatingEvaluator']:.4f}, BL={s_rl1['BoneLengthEvaluator']:.4f}, VJ={s_rl1['VelocityJitterEvaluator']:.4f} | NetGain={rl1_syn['netgain_A']:+.4f}\n"
        f"oracle (len {oracle_syn['best_A']['length']}) scores: FF={s_oracle_syn['FootFloatingEvaluator']:.4f}, BL={s_oracle_syn['BoneLengthEvaluator']:.4f}, VJ={s_oracle_syn['VelocityJitterEvaluator']:.4f} | NetGain={oracle_syn['best_A']['netgain_A']:+.4f}\n"
        f"B2 (VS/medium) scores: FF={s_b2_syn['FootFloatingEvaluator']:.4f}, BL={s_b2_syn['BoneLengthEvaluator']:.4f}, VJ={s_b2_syn['VelocityJitterEvaluator']:.4f}"
    )
    _plot_4way(
        title=f"Synthetic RL-1 SUCCESS {syn_succ_id} — clean / corrupted / RL-1 (length=5) / oracle / B2",
        save_path=args.output_dir / f"synthetic_{syn_succ_id}_success_sanity.png",
        motions={"clean": clean, "corrupted": corrupted, "RL-1": rl1_motion, "oracle": oracle_motion, "B2": b2_motion},
        extra_text=extra,
    )

    print(f"\n[OK] Visualization PNGs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
