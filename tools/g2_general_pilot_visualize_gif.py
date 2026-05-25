"""Step 5: G2 general-prompt pilot 의 GIF/MP4 시각화 확장.

사용자 directive (2026-05-25 9-step plan Step 5):
> "각 category 에서 1개씩 총 5개 sample. 비교 GIF: Original vs B2-small /
>  Original vs B2-large / Original vs sequence oracle."

5 sample × 3 GIF = 15 GIF.

본 도구는 g2_general_pilot_v1 의 10 motion 중 5 category 대표 sample 선정 후
3-way comparison GIF 생성. external_assets/code/plot_3d_motion.py 의
plot_3d_motion 활용 (overlay = gray vs orange).

CLI:
    python -m tools.g2_general_pilot_visualize_gif \\
        --pilot-dir external_assets/g2_general_pilot_v1 \\
        --pilot-snapshot evals/snapshots/g2_general_pilot_v1.json \\
        --output-dir reports/figures/2026-05-25/g2_general_gif \\
        --frame-stride 2 --fps 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external_assets" / "code"))
from plot_3d_motion import plot_3d_motion  # type: ignore

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool

TOOL_BY_NAME = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}

# 5 category 대표 sample (n_frames 충분한 것 선정 — motion_005/006 제외).
SAMPLE_SELECTION = [
    ("motion_001", "locomotion (walk normal)"),
    ("motion_004", "upper-body (wave hand)"),
    ("motion_008", "turning/balance (turn left)"),
    ("motion_007", "transition (stand up)"),
    ("motion_010", "contact-heavy (kick forward)"),
]


def _apply_vs(motion: np.ndarray, strength: str) -> np.ndarray:
    T = motion.shape[0]
    out, _ = TOOL_BY_NAME["VelocitySmoothingTool"].apply(
        motion, target_part="full_body", target_joints=[],
        frame_range=(0, T - 1), strength=strength,
    )
    return out


def _apply_sequence(motion: np.ndarray, sequence: list[list[str]]) -> np.ndarray:
    T = motion.shape[0]
    frame_range = (0, T - 1)
    out = motion.copy()
    for step in sequence:
        if step[0] in ("STOP", "APPLY_FAIL", "SCORE_VIOLATION_ROLLBACK"):
            break
        tool = TOOL_BY_NAME[step[0]]
        out, _ = tool.apply(out, target_part=step[1], target_joints=[],
                             frame_range=frame_range, strength=step[2])
    return out


def _save_gif(
    *, save_path: Path, motion_a: np.ndarray, motion_b: np.ndarray,
    title: str, fps: int = 8, frame_stride: int = 2,
) -> None:
    a = motion_a[::frame_stride].astype(np.float32)
    b = motion_b[::frame_stride].astype(np.float32)
    m = min(a.shape[0], b.shape[0])
    a, b = a[:m], b[:m]
    obs_arr = np.zeros((0, 22, 3), dtype=np.float32)
    plot_3d_motion(
        save_path=str(save_path),
        obs_arr=obs_arr, pred_gt_arr=a, pred_model_arr=b,
        title=title, fps=fps, figsize=(6, 6), hold_last_frame_ms=600,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 general-prompt pilot GIF visualization (Step 5)")
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--pilot-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load snapshot for sequence oracle paths.
    with open(args.pilot_snapshot, encoding="utf-8") as f:
        snap = json.load(f)
    snap_by_trial = {s["trial_id"]: s for s in snap["per_sample"]}

    for idx, (trial_id, category) in enumerate(SAMPLE_SELECTION, 1):
        if trial_id not in snap_by_trial:
            print(f"[WARN] {trial_id} not in snapshot — skipping")
            continue
        npy_path = args.pilot_dir / f"{trial_id}.npy"
        if not npy_path.exists():
            print(f"[WARN] missing motion file: {npy_path}")
            continue
        motion = np.load(str(npy_path)).astype(np.float64)
        s_info = snap_by_trial[trial_id]
        prompt = s_info["prompt"][:60]
        print(f"\n[{idx}/{len(SAMPLE_SELECTION)}] {trial_id} ({category}, T={motion.shape[0]}): '{prompt}...'")

        # GIF 1: Original vs B2-small.
        b2_small = _apply_vs(motion, "small")
        _save_gif(
            save_path=args.output_dir / f"{trial_id}_vs_b2_small.gif",
            motion_a=motion, motion_b=b2_small,
            title=f"{trial_id} ({category}) - original (gray) vs B2-small (orange)",
            fps=args.fps, frame_stride=args.frame_stride,
        )

        # GIF 2: Original vs B2-large (over-modification visual).
        b2_large = _apply_vs(motion, "large")
        _save_gif(
            save_path=args.output_dir / f"{trial_id}_vs_b2_large.gif",
            motion_a=motion, motion_b=b2_large,
            title=f"{trial_id} ({category}) - original (gray) vs B2-large (orange, over-mod)",
            fps=args.fps, frame_stride=args.frame_stride,
        )

        # GIF 3: Original vs sequence oracle.
        seq = s_info["sequence_oracle"]["sequence"]
        oracle_len = s_info["sequence_oracle"]["length"]
        if oracle_len == 0:
            # STOP-best — oracle = original (do-nothing).
            oracle_motion = motion.copy()
            oracle_label = "oracle STOP (= original)"
        else:
            oracle_motion = _apply_sequence(motion, seq)
            oracle_label = f"oracle (len {oracle_len})"
        _save_gif(
            save_path=args.output_dir / f"{trial_id}_vs_oracle.gif",
            motion_a=motion, motion_b=oracle_motion,
            title=f"{trial_id} ({category}) - original (gray) vs {oracle_label} (orange)",
            fps=args.fps, frame_stride=args.frame_stride,
        )

    print(f"\n[OK] all GIFs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
