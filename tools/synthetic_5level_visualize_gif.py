"""Step 8-B: Perceptual Pilot v2 Group C — synthetic 5-level high-Δ enriched GIF 생성.

사용자 directive (2026-05-26):
> "GIF 생성은 Group C synthetic부터 시작 — 새로 생성해야 하는 핵심 GIF 가 대부분 Group C,
>  5-level action space 결정의 perceptual 근거와 직접 연결."

본 도구는 perceptual pilot v2 Group C 의 25 GIF 를 생성한다:
- 10 sample × 2 comparison (3-level oracle vs 5-level oracle, B2-val-best vs 5-level oracle)
- 5 sample × 1 comparison (corrupted vs 5-level oracle, positive control)

Synthetic corruption protocol (oracle_sequence_multi_5level_v1 동일):
  clean (HumanML3D) → inject_foot_floating(lift=0.08, seed=42) → inject_jitter(noise=0.05, seed=1042)
  → corrupted

각 sample 에 대해:
  - corrupted (base)
  - oracle_3level = apply_sequence(corrupted, 3-level best_A sequence)
  - oracle_5level = apply_sequence(corrupted, 5-level best_A sequence)
  - b2_val_best = VelocitySmoothingTool(corrupted, full_body, medium)  # synthetic 의 best 90%
                                                                         + large 8% (014552 등) 의 분포 따라 per-sample 매핑

GIF overlay (Y-up convention, AGENTS.md §3-19):
  gray = method_A, orange = method_B (5-level)

CLI:
    python -m tools.synthetic_5level_visualize_gif \
        --top-n 10 --top-n-pc 5 \
        --output-dir reports/figures/2026-05-26/perceptual_v2_gif_yup_fix
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
from tools.synthetic_injection import inject_foot_floating, inject_jitter

TOOL_BY_NAME: dict[str, CorrectionTool] = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}


def _multi_inject(clean: np.ndarray, seed: int = 42) -> np.ndarray:
    """Multi-artifact: foot_floating + global_jitter chained (oracle protocol)."""
    m1 = inject_foot_floating(clean, lift_height=0.08, seed=seed)
    return inject_jitter(m1, noise_std=0.05, seed=seed + 1000)


def _apply_sequence(motion: np.ndarray, sequence: list[list[str]]) -> np.ndarray:
    """Sequence apply — terminal action (STOP/APPLY_FAIL/SCORE_VIOLATION_ROLLBACK) early-stop."""
    T = motion.shape[0]
    out = motion.copy()
    for step in sequence:
        if step[0] in ("STOP", "APPLY_FAIL", "SCORE_VIOLATION_ROLLBACK"):
            break
        tool = TOOL_BY_NAME[step[0]]
        out, _ = tool.apply(
            out, target_part=step[1], target_joints=[],
            frame_range=(0, T - 1), strength=step[2],
        )
    return out


def _apply_b2_val_best(motion: np.ndarray, strength: str = "medium") -> np.ndarray:
    """B2 family — VelocitySmoothing full_body (synthetic 의 best 90%=medium)."""
    T = motion.shape[0]
    out, _ = TOOL_BY_NAME["VelocitySmoothingTool"].apply(
        motion, target_part="full_body", target_joints=[],
        frame_range=(0, T - 1), strength=strength,
    )
    return out


def _save_overlay_gif(
    *, save_path: Path, motion_a: np.ndarray, motion_b: np.ndarray,
    title: str, fps: int = 8, frame_stride: int = 2,
) -> None:
    a = motion_a[::frame_stride].astype(np.float32)
    b = motion_b[::frame_stride].astype(np.float32)
    m = min(a.shape[0], b.shape[0])
    a, b = a[:m], b[:m]
    obs_arr = np.zeros((0, 22, 3), dtype=np.float32)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_3d_motion(
        save_path=str(save_path),
        obs_arr=obs_arr, pred_gt_arr=a, pred_model_arr=b,
        title=title, fps=fps, figsize=(6, 6), hold_last_frame_ms=600,
    )


# Top-10 high-Δ(5-3) synthetic samples + their B2 best strength (from family sweep).
# 90% medium, 8% large (014552 등), 2% small — 매핑 (per-sample data 없으면 medium default).
TOP_10_SAMPLES = [
    "014552", "M007995", "012798", "M000741", "012631",
    "M008358", "001885", "010382", "012943", "M007140",
]

# Per-sample B2 best strength override (sweep distribution 의 large/small case 매핑).
# 014552 의 Δ 가 가장 커서 large strength 의 가능성 — 단 정확한 per-sample 정보 없어 default=medium 사용.
B2_BEST_PER_SAMPLE: dict[str, str] = {
    # 모든 sample default = "medium" (90% prevalence). 향후 per-sample compute 시 갱신.
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Perceptual v2 Group C — synthetic 5-level GIF (Step 8-B)")
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--snapshot-3level", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "oracle_sequence_multi_v2_n60.json")
    parser.add_argument("--snapshot-5level", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "oracle_sequence_multi_5level_v1.json")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "reports" / "figures" / "2026-05-26" / "perceptual_v2_gif_yup_fix")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of high-Δ samples (for 3-vs-5 + B2-vs-5 pairs)")
    parser.add_argument("--top-n-pc", type=int, default=5,
                        help="Number of samples for positive control (corrupted vs 5-level)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    args = parser.parse_args()

    # Load oracle sequences.
    with open(args.snapshot_3level, encoding="utf-8") as f:
        snap3 = json.load(f)
    with open(args.snapshot_5level, encoding="utf-8") as f:
        snap5 = json.load(f)
    seq3_by = {s["trial_id"]: s["best_A"]["sequence"] for s in snap3["per_sample"]}
    seq5_by = {s["trial_id"]: s["best_A"]["sequence"] for s in snap5["per_sample"]}

    samples = TOP_10_SAMPLES[: args.top_n]
    pc_samples = TOP_10_SAMPLES[: args.top_n_pc]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(samples) * 2 + len(pc_samples)
    print(f"[INFO] generating {n_total} GIFs ({len(samples)} samples × 2 + {len(pc_samples)} PC)")
    print(f"[INFO] output: {args.output_dir}")

    for i, tid in enumerate(samples, 1):
        npy_path = args.data_dir / f"{tid}.npy"
        if not npy_path.exists():
            print(f"[WARN] [{i}/{len(samples)}] {tid}: missing {npy_path}")
            continue
        clean = np.load(str(npy_path)).astype(np.float64)
        if clean.ndim != 3 or clean.shape[1] != 22 or clean.shape[2] != 3:
            print(f"[WARN] [{i}/{len(samples)}] {tid}: bad shape {clean.shape}")
            continue

        corrupted = _multi_inject(clean, seed=args.seed)
        seq3 = seq3_by.get(tid)
        seq5 = seq5_by.get(tid)
        if seq3 is None or seq5 is None:
            print(f"[WARN] [{i}/{len(samples)}] {tid}: missing oracle sequence")
            continue

        oracle3 = _apply_sequence(corrupted, seq3)
        oracle5 = _apply_sequence(corrupted, seq5)
        b2_strength = B2_BEST_PER_SAMPLE.get(tid, "medium")
        b2_motion = _apply_b2_val_best(corrupted, strength=b2_strength)

        print(f"[{i}/{len(samples)}] {tid}: T={clean.shape[0]}, len_3={len(seq3)}, len_5={len(seq5)}, b2={b2_strength}")

        # GIF 1: 3-level oracle vs 5-level oracle (gray=3, orange=5).
        _save_overlay_gif(
            save_path=args.output_dir / f"synthetic_{tid}_3level_vs_5level.gif",
            motion_a=oracle3, motion_b=oracle5,
            title=f"synthetic {tid} - 3-level oracle (gray) vs 5-level oracle (orange)",
            fps=args.fps, frame_stride=args.frame_stride,
        )

        # GIF 2: B2-val-best vs 5-level oracle (gray=B2, orange=5).
        _save_overlay_gif(
            save_path=args.output_dir / f"synthetic_{tid}_b2valbest_vs_5level.gif",
            motion_a=b2_motion, motion_b=oracle5,
            title=f"synthetic {tid} - B2-val-best/{b2_strength} (gray) vs 5-level oracle (orange)",
            fps=args.fps, frame_stride=args.frame_stride,
        )

        # GIF 3 (positive control, top-5 only): corrupted vs 5-level oracle.
        if tid in pc_samples:
            _save_overlay_gif(
                save_path=args.output_dir / f"synthetic_{tid}_corrupted_vs_5level.gif",
                motion_a=corrupted, motion_b=oracle5,
                title=f"synthetic {tid} - corrupted (gray) vs 5-level oracle (orange) [PC]",
                fps=args.fps, frame_stride=args.frame_stride,
            )

    print(f"\n[OK] all Group C GIFs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
