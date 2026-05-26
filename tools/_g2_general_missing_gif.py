"""ad-hoc: Generate the 5 missing G2 general overlay GIFs for perceptual v2 Group A/B.

Existing v1 GIFs (in g2_general_gif_yup_fix/) cover motion_001, 004, 007, 008, 010.
v2 needs the remaining 5: motion_002, 003, 005, 006, 009 — each × 3 (vs_b2_small, vs_b2_large, vs_oracle).

Output dir: reports/figures/2026-05-26/perceptual_v2_gif_yup_fix/
File naming consistent with v1: {trial_id}_vs_{method}.gif
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external_assets" / "code"))
from plot_3d_motion import plot_3d_motion  # type: ignore

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool

TOOL_BY_NAME: dict[str, CorrectionTool] = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}

MISSING_SAMPLES = ["motion_002", "motion_003", "motion_005", "motion_006", "motion_009"]


def _apply_vs(motion: np.ndarray, strength: str) -> np.ndarray:
    T = motion.shape[0]
    out, _ = TOOL_BY_NAME["VelocitySmoothingTool"].apply(
        motion, target_part="full_body", target_joints=[],
        frame_range=(0, T - 1), strength=strength,
    )
    return out


def _apply_sequence(motion: np.ndarray, sequence: list) -> np.ndarray:
    T = motion.shape[0]
    out = motion.copy()
    for step in sequence:
        if step[0] in ("STOP", "APPLY_FAIL", "SCORE_VIOLATION_ROLLBACK"):
            break
        tool = TOOL_BY_NAME[step[0]]
        out, _ = tool.apply(out, target_part=step[1], target_joints=[],
                             frame_range=(0, T - 1), strength=step[2])
    return out


def _save_overlay(save_path: Path, a: np.ndarray, b: np.ndarray, title: str) -> None:
    a_s = a[::2].astype(np.float32)
    b_s = b[::2].astype(np.float32)
    m = min(a_s.shape[0], b_s.shape[0])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_3d_motion(
        save_path=str(save_path),
        obs_arr=np.zeros((0, 22, 3), dtype=np.float32),
        pred_gt_arr=a_s[:m], pred_model_arr=b_s[:m],
        title=title, fps=8, figsize=(6, 6), hold_last_frame_ms=600,
    )


def main() -> None:
    pilot_dir = REPO_ROOT / "external_assets" / "g2_general_pilot_v1"
    snap_path = REPO_ROOT / "evals" / "snapshots" / "g2_general_pilot_v1.json"
    out_dir = REPO_ROOT / "reports" / "figures" / "2026-05-26" / "perceptual_v2_gif_yup_fix"

    with open(snap_path, encoding="utf-8") as f:
        snap = json.load(f)
    snap_by = {s["trial_id"]: s for s in snap["per_sample"]}

    for i, tid in enumerate(MISSING_SAMPLES, 1):
        npy_path = pilot_dir / f"{tid}.npy"
        if not npy_path.exists():
            print(f"[WARN] {tid}: missing {npy_path}")
            continue
        motion = np.load(str(npy_path)).astype(np.float64)
        s_info = snap_by[tid]
        prompt = s_info["prompt"][:60]
        print(f"[{i}/{len(MISSING_SAMPLES)}] {tid} (T={motion.shape[0]}): '{prompt}'")

        # vs B2-large (over-modification).
        b2_large = _apply_vs(motion, "large")
        _save_overlay(
            out_dir / f"{tid}_vs_b2_large.gif", motion, b2_large,
            f"{tid} - original (gray) vs B2-large (orange, over-mod)",
        )

        # vs B2-small (subtle, for Group B's val-best pair on motion_002).
        b2_small = _apply_vs(motion, "small")
        _save_overlay(
            out_dir / f"{tid}_vs_b2_small.gif", motion, b2_small,
            f"{tid} - original (gray) vs B2-small (orange)",
        )

        # vs oracle.
        seq = s_info["sequence_oracle"]["sequence"]
        if s_info["sequence_oracle"]["length"] == 0:
            oracle_motion = motion.copy()
            label = "oracle STOP (= original)"
        else:
            oracle_motion = _apply_sequence(motion, seq)
            label = f"oracle (len {s_info['sequence_oracle']['length']})"
        _save_overlay(
            out_dir / f"{tid}_vs_oracle.gif", motion, oracle_motion,
            f"{tid} - original (gray) vs {label} (orange)",
        )

    print(f"\n[OK] all 5 missing GIFs (15 files) saved to: {out_dir}")


if __name__ == "__main__":
    main()
