"""Step B-2-1: G2 top-4 correction motion 의 SMPL fit 입력 12 npy 생성.

각 sample (motion_006, _007, _008, _028) 에 대해 3 method:
  - {tid}_original.npy
  - {tid}_b2small.npy
  - {tid}_oracle5level.npy

motion 형식: (T, 22, 3) HumanML3D / AMASS joint, Y-up. SMPLify3D 의 AMASS category.
저장 위치: external_assets/MotionGPT/staging/g2_top4/  (mgpt env 의 fit.py 가 처리).

CLI:
    python -m tools._prepare_g2_top_motions_for_smpl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool

TOOL_BY_NAME: dict[str, CorrectionTool] = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}
SAMPLES = ["motion_006", "motion_007", "motion_008", "motion_028"]


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


def _apply_b2(motion: np.ndarray, strength: str = "small") -> np.ndarray:
    T = motion.shape[0]
    out, _ = TOOL_BY_NAME["VelocitySmoothingTool"].apply(
        motion, target_part="full_body", target_joints=[],
        frame_range=(0, T - 1), strength=strength,
    )
    return out


def main() -> None:
    snap_path = REPO_ROOT / "evals" / "snapshots" / "oracle_sequence_g2_natural_5level_v1.json"
    data_dir = REPO_ROOT / "external_assets" / "g2_generated_v1"
    out_dir = REPO_ROOT / "external_assets" / "MotionGPT" / "staging" / "g2_top4"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(snap_path, encoding="utf-8") as f:
        snap = json.load(f)
    snap_by = {s["trial_id"]: s for s in snap["per_sample"]}

    manifest = {}
    for tid in SAMPLES:
        npy = data_dir / f"{tid}.npy"
        motion = np.load(str(npy)).astype(np.float64)
        s = snap_by[tid]
        seq = s["best"]["sequence"]
        b2 = _apply_b2(motion, "small")
        oracle = _apply_sequence(motion, seq)
        m = min(motion.shape[0], b2.shape[0], oracle.shape[0])
        # frame stride 2 to halve compute (SMPLify on CPU is slow).
        stride = 2
        original_sub = motion[:m:stride].astype(np.float32)
        b2_sub = b2[:m:stride].astype(np.float32)
        oracle_sub = oracle[:m:stride].astype(np.float32)
        # Save each variant as separate .npy (so fit.py processes each).
        for variant, arr in [
            ("original", original_sub), ("b2small", b2_sub), ("oracle5level", oracle_sub),
        ]:
            out_path = out_dir / f"{tid}_{variant}.npy"
            np.save(str(out_path), arr)
        manifest[tid] = {
            "T_original": motion.shape[0], "T_subsampled": original_sub.shape[0],
            "oracle_sequence": seq, "g2_prompt": s.get("g2_prompt", "")[:120],
        }
        print(f"[OK] {tid}: T={motion.shape[0]} -> {original_sub.shape[0]} (stride={stride})")
    # Save manifest.
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] 12 npy + manifest saved to: {out_dir}")


if __name__ == "__main__":
    main()
