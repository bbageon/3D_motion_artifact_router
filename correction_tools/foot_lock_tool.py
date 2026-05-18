"""FootLockTool — Week 3 correction tool prototype.

[`FootFloatingEvaluator`](../evaluators/foot_floating_evaluator.py) 와 짝지어진 tool.
contact 으로 추정되거나 명시된 frame 에서 foot joint 의 height (y 좌표) 를
ground 근처로 "lock" 한다. strength 에 따라 lock 강도 (interpolation factor)
조절. propagation 은 ankle / knee 까지 짧은 chain 으로만 보정 (forward
kinematics 의 단순 inverse displacement).

명세 §6.3 CorrectionTool 인터페이스 + KDG affected joints 의무.
AGENTS.md §3-3 KDG ordering 의무 — 본 tool 의 root 가까운 항목이라 ordering
의 후순위 (다른 root-modify tool 가 있다면 그 다음).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from correction_tools.base import CorrectionTool, CorrectionReport, Strength
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX, SMPL_22

#: 본 tool 이 직접 modify 하는 joint 집합 (좌·우 foot).
DIRECT_JOINTS: list[str] = ["LEFT_FOOT", "RIGHT_FOOT"]

#: 본 tool 이 forward kinematics 로 간접 영향 주는 joint (ankle / knee, weight < 1).
PROPAGATION_WEIGHTS: dict[str, float] = {
    "LEFT_ANKLE": 0.5,
    "RIGHT_ANKLE": 0.5,
    "LEFT_KNEE": 0.2,
    "RIGHT_KNEE": 0.2,
}

#: strength → lock interpolation factor (1.0 = ground 로 강제, 0.0 = 변화 없음).
STRENGTH_FACTOR: dict[str, float] = {"small": 0.3, "medium": 0.6, "large": 1.0}


class FootLockTool(CorrectionTool):
    """Foot lock — contact frame 의 foot height 를 ground 로 끌어내림.

    target_part 은 'left_foot' / 'right_foot' / 'both_feet' 중 하나.
    target_joints 가 명시되면 그것을 우선 (예: ['LEFT_FOOT']).
    """

    name = "FootLockTool"

    def __init__(self, default_ground_y: float = 0.0) -> None:
        self.default_ground_y = default_ground_y

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
        start, end = frame_range
        start = max(0, start)
        end = min(T, end + 1) if end < T else T  # inclusive [start, end] → slice [start, end+1)

        meta = metadata or {}
        ground_y: float = float(meta.get("ground_y", self.default_ground_y))
        factor: float = STRENGTH_FACTOR.get(strength, STRENGTH_FACTOR["medium"])

        # target joints 결정
        if target_joints:
            joints = [j for j in target_joints if j in NAME_TO_IDX]
        else:
            if target_part == "left_foot":
                joints = ["LEFT_FOOT"]
            elif target_part == "right_foot":
                joints = ["RIGHT_FOOT"]
            else:  # both_feet 또는 unknown → 양 foot
                joints = ["LEFT_FOOT", "RIGHT_FOOT"]

        corrected = motion.copy()
        deltas: list[float] = []
        for jname in joints:
            jidx = NAME_TO_IDX[jname]
            current_y = corrected[start:end, jidx, 1]
            target_y = np.full_like(current_y, ground_y)
            new_y = current_y + factor * (target_y - current_y)
            delta = new_y - current_y
            corrected[start:end, jidx, 1] = new_y
            deltas.append(float(np.mean(np.abs(delta))))

            # 짧은 chain propagation — ankle 도 부분 따라감.
            ankle_name = "LEFT_ANKLE" if jname == "LEFT_FOOT" else "RIGHT_ANKLE"
            ankle_idx = NAME_TO_IDX[ankle_name]
            ankle_weight = PROPAGATION_WEIGHTS[ankle_name]
            corrected[start:end, ankle_idx, 1] += ankle_weight * delta

        correction_magnitude = float(np.mean(deltas)) if deltas else 0.0
        report = CorrectionReport(
            tool=self.name,
            target_part=target_part,
            frame_range=(int(start), int(end - 1)),
            strength=strength,
            modified_joints=joints,
            correction_magnitude=correction_magnitude,
            metadata={
                "ground_y": ground_y,
                "strength_factor": factor,
                "propagation_weights": {k: PROPAGATION_WEIGHTS[k] for k in PROPAGATION_WEIGHTS},
            },
        )
        return corrected, report

    def kdg_affected_joints(self) -> list[str]:
        return list(DIRECT_JOINTS)

    def kdg_propagation_weights(self) -> dict[str, float]:
        return dict(PROPAGATION_WEIGHTS)
