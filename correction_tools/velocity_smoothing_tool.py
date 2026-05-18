"""VelocitySmoothingTool — Week 3 correction tool prototype.

[`VelocityJitterEvaluator`](../evaluators/velocity_jitter_evaluator.py) 와 짝
지어진 tool. frame 간 acceleration 의 크기를 줄이기 위해 1D gaussian smoothing
을 frame 축에 따라 각 joint 좌표에 적용.

구체:
  - sigma_frames = (small=0.5, medium=1.0, large=2.0) frame 단위.
  - target_joints 가 명시되면 그 joint 만 smoothing. 없으면 target_part 의 모든
    joint (legs / arms / spine_head / full_body) 적용.
  - frame_range 외 영역은 원본 그대로 (boundary 효과 최소화 위해 reflect padding).

명세 §6.3 CorrectionTool 인터페이스 + KDG affected joints 의무.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from correction_tools.base import CorrectionTool, CorrectionReport, Strength
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX, SMPL_22

STRENGTH_SIGMA: dict[str, float] = {"small": 0.5, "medium": 1.0, "large": 2.0}

#: target_part → joint name list (VelocityJitterEvaluator 의 PART_JOINTS 와 동기).
PART_TO_JOINTS: dict[str, list[str]] = {
    "legs": [
        "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT",
        "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT",
    ],
    "arms": [
        "LEFT_COLLAR", "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
        "RIGHT_COLLAR", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
    ],
    "spine_head": [
        "PELVIS", "SPINE1", "SPINE2", "SPINE3", "NECK", "HEAD",
    ],
    "full_body": SMPL_22,
}


class VelocitySmoothingTool(CorrectionTool):
    """Velocity smoothing — gaussian filter 로 acceleration 크기 감소."""

    name = "VelocitySmoothingTool"

    def apply(
        self,
        motion: np.ndarray,
        target_part: str,
        target_joints: list[str],
        frame_range: tuple[int, int],
        strength: Strength = "medium",
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, CorrectionReport]:
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            raise ValueError(
                f"motion shape must be [T, 22, 3], got {motion.shape}. "
                "AGENTS.md §3-1 Canonical Motion Format."
            )
        T = motion.shape[0]
        if T < 2:
            # smoothing 의 의미가 없으므로 그대로 반환
            return motion.copy(), CorrectionReport(
                tool=self.name,
                target_part=target_part,
                frame_range=frame_range,
                strength=strength,
                modified_joints=[],
                correction_magnitude=0.0,
                metadata={"reason": "T < 2 — smoothing skipped"},
            )

        start, end = frame_range
        start = max(0, start)
        end = min(T, end + 1) if end < T else T

        sigma: float = STRENGTH_SIGMA.get(strength, STRENGTH_SIGMA["medium"])

        # target joints 결정
        if target_joints:
            joints = [j for j in target_joints if j in NAME_TO_IDX]
        elif target_part in PART_TO_JOINTS:
            joints = PART_TO_JOINTS[target_part]
        else:
            joints = list(SMPL_22)  # fallback full body

        joint_indices = [NAME_TO_IDX[j] for j in joints]

        corrected = motion.copy()
        # frame_range 안의 좌표만 smoothing (scipy 의 reflect mode 로 경계 안정화).
        slice_ = corrected[start:end, joint_indices, :]  # [n_frames, K, 3]
        if slice_.shape[0] < 2:
            return corrected, CorrectionReport(
                tool=self.name,
                target_part=target_part,
                frame_range=(int(start), int(end - 1)),
                strength=strength,
                modified_joints=[],
                correction_magnitude=0.0,
                metadata={"reason": "frame_range too short for smoothing"},
            )
        smoothed = gaussian_filter1d(slice_, sigma=sigma, axis=0, mode="reflect")
        delta = smoothed - slice_
        corrected[start:end, joint_indices, :] = smoothed

        correction_magnitude = float(np.mean(np.linalg.norm(delta, axis=-1)))

        report = CorrectionReport(
            tool=self.name,
            target_part=target_part,
            frame_range=(int(start), int(end - 1)),
            strength=strength,
            modified_joints=joints,
            correction_magnitude=correction_magnitude,
            metadata={
                "sigma_frames": sigma,
                "n_joints": len(joints),
                "n_frames_smoothed": int(slice_.shape[0]),
            },
        )
        return corrected, report

    def kdg_affected_joints(self) -> list[str]:
        # default: 모든 joint 잠재적 영향 (target_part 에 따라 동적).
        return list(SMPL_22)

    def kdg_propagation_weights(self) -> dict[str, float]:
        # smoothing 은 forward kinematics propagation 없이 좌표 직접 변경.
        return {}
