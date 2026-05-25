"""Skeleton GIF 시각화 — 사용자 directive (2026-05-25 Order 3.7).

5 question (B2-family 정정의 정성 검증):
  1. G2 에서 B2-small/medium/large 실제 차이?
  2. large over-modification 눈으로 보이는가?
  3. Synthetic 에서 medium 이 왜 best?
  4. RL-1/HGB STOP 이 G2 에서 자연스러운가?
  5. Sequence oracle 이 synthetic 에서 B2-medium 보다 눈으로도 좋은가?

본 도구는 external_assets/code/plot_3d_motion.py 의 plot_3d_motion 활용. 두
motion overlay GIF (gray=baseline, orange=variant) 로 비교.

GIF 6 종:
  - G2: original (single) — STOP-best baseline.
  - G2: original vs B2-small overlay.
  - G2: original vs B2-medium overlay.
  - G2: original vs B2-large overlay (over-modification visual).
  - Synthetic: corrupted vs B2-medium overlay.
  - Synthetic: corrupted vs sequence oracle overlay.

효율화:
  - Frame downsample (every Nth frame).
  - fps 조절.

CLI:
    python -m tools.rl1_visualize_skeleton_gif \\
        --g2-sample motion_004 \\
        --syn-sample 002806 \\
        --output-dir reports/figures/2026-05-25/skeleton_gif \\
        --frame-stride 2 --fps 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# external_assets/code 모듈 import.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external_assets" / "code"))
from plot_3d_motion import plot_3d_motion  # type: ignore

from correction_tools import BoneProjectionTool, FootLockTool, VelocitySmoothingTool
from tools.synthetic_injection import inject_foot_floating, inject_jitter

DEFAULT_HUMANML3D_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
DEFAULT_G2_DIR = REPO_ROOT / "external_assets" / "g2_generated_v1"

TOOL_BY_NAME = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}


def _multi_inject(clean: np.ndarray, seed: int) -> np.ndarray:
    m1 = inject_foot_floating(clean, lift_height=0.08, seed=seed)
    return inject_jitter(m1, noise_std=0.05, seed=seed + 1000)


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
        out, _ = tool.apply(out, target_part=step[1], target_joints=[], frame_range=frame_range, strength=step[2])
    return out


def _save_gif(
    *, save_path: Path, motion_a: np.ndarray, motion_b: np.ndarray | None,
    title: str, fps: int = 8, frame_stride: int = 2,
) -> None:
    """motion_a (pred_gt, gray) vs motion_b (pred_model, orange) overlay GIF.

    motion_b None 이면 motion_a 만 (green color).
    """
    # Downsample.
    a = motion_a[::frame_stride].astype(np.float32)
    if motion_b is not None:
        # Length 맞추기.
        b = motion_b[::frame_stride].astype(np.float32)
        m = min(a.shape[0], b.shape[0])
        a, b = a[:m], b[:m]
    else:
        b = None
    T = a.shape[0]
    obs_arr = np.zeros((0, 22, 3), dtype=np.float32)  # 빈 prefix.
    plot_3d_motion(
        save_path=str(save_path),
        obs_arr=obs_arr,
        pred_gt_arr=a,
        pred_model_arr=b,
        title=title,
        fps=fps,
        figsize=(6, 6),
        hold_last_frame_ms=600,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Skeleton GIF visualization (B2-family + oracle)")
    parser.add_argument("--g2-sample", type=str, default="motion_004")
    parser.add_argument("--syn-sample", type=str, default="002806")
    parser.add_argument("--humanml3d-dir", type=Path, default=DEFAULT_HUMANML3D_DIR)
    parser.add_argument("--g2-dir", type=Path, default=DEFAULT_G2_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oracle-snapshot", type=Path,
                        default=Path("evals/snapshots/oracle_sequence_multi_v2_n60.json"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # === G2 motion (motion_004 by default) ===
    g2_path = args.g2_dir / f"{args.g2_sample}.npy"
    g2_motion = np.load(str(g2_path)).astype(np.float64)
    print(f"[1/6] G2 {args.g2_sample}: single original (STOP-best)")
    _save_gif(
        save_path=args.output_dir / f"g2_{args.g2_sample}_original_single.gif",
        motion_a=g2_motion, motion_b=None,
        title=f"G2 {args.g2_sample} — original (RL-1 STOP)",
        fps=args.fps, frame_stride=args.frame_stride,
    )

    print(f"[2/6] G2 {args.g2_sample}: original (gray) vs B2-small (orange)")
    b2_small = _apply_vs(g2_motion, "small")
    _save_gif(
        save_path=args.output_dir / f"g2_{args.g2_sample}_vs_b2_small.gif",
        motion_a=g2_motion, motion_b=b2_small,
        title=f"G2 {args.g2_sample} — original (gray) vs B2-small (orange)",
        fps=args.fps, frame_stride=args.frame_stride,
    )

    print(f"[3/6] G2 {args.g2_sample}: original vs B2-medium")
    b2_medium = _apply_vs(g2_motion, "medium")
    _save_gif(
        save_path=args.output_dir / f"g2_{args.g2_sample}_vs_b2_medium.gif",
        motion_a=g2_motion, motion_b=b2_medium,
        title=f"G2 {args.g2_sample} — original (gray) vs B2-medium (orange)",
        fps=args.fps, frame_stride=args.frame_stride,
    )

    print(f"[4/6] G2 {args.g2_sample}: original vs B2-large (over-modification visual)")
    b2_large = _apply_vs(g2_motion, "large")
    _save_gif(
        save_path=args.output_dir / f"g2_{args.g2_sample}_vs_b2_large.gif",
        motion_a=g2_motion, motion_b=b2_large,
        title=f"G2 {args.g2_sample} — original (gray) vs B2-large (orange, over-mod)",
        fps=args.fps, frame_stride=args.frame_stride,
    )

    # === Synthetic 002806 (RL-1 SUCCESS, length=5) ===
    syn_path = args.humanml3d_dir / f"{args.syn_sample}.npy"
    clean = np.load(str(syn_path)).astype(np.float64)
    corrupted = _multi_inject(clean, seed=args.seed)

    print(f"[5/6] Synthetic {args.syn_sample}: corrupted vs B2-medium")
    b2_med_syn = _apply_vs(corrupted, "medium")
    _save_gif(
        save_path=args.output_dir / f"synthetic_{args.syn_sample}_vs_b2_medium.gif",
        motion_a=corrupted, motion_b=b2_med_syn,
        title=f"Synthetic {args.syn_sample} — corrupted (gray) vs B2-medium (orange)",
        fps=args.fps, frame_stride=args.frame_stride,
    )

    # Sequence oracle (best path).
    print(f"[6/6] Synthetic {args.syn_sample}: corrupted vs sequence oracle")
    with open(args.oracle_snapshot, encoding="utf-8") as f:
        oracle = json.load(f)
    oracle_by_trial = {r["trial_id"]: r for r in oracle["per_sample"] if r.get("best_A")}
    if args.syn_sample in oracle_by_trial:
        seq = oracle_by_trial[args.syn_sample]["best_A"]["sequence"]
        oracle_motion = _apply_sequence(corrupted, seq)
        _save_gif(
            save_path=args.output_dir / f"synthetic_{args.syn_sample}_vs_oracle.gif",
            motion_a=corrupted, motion_b=oracle_motion,
            title=f"Synthetic {args.syn_sample} — corrupted (gray) vs oracle (orange, len {len(seq)})",
            fps=args.fps, frame_stride=args.frame_stride,
        )
    else:
        print(f"  [WARN] {args.syn_sample} not in oracle snapshot — skipping GIF 6")

    print(f"\n[OK] all GIFs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
