"""FootFloatingEvaluator — Week 2 evaluator prototype.

명세 §9.3.1 의 FootFloating 정의:

    FootFloating = mean_t I(contact_foot(t)) * I(p_foot_y(t) - ground_y > tau_float)

contact 로 추정되는 frame 에서 foot 의 ground 대비 높이가 임계 (tau_float) 를
초과하는 비율을 측정. 의미: "딛고 있어야 할 발이 공중에 떠 있는 정도".

본 evaluator 는 좌·우 foot 을 별도 score 로 보고하고, frame range 는
floating 이 처음 등장한 frame ~ 마지막 frame 으로 설정한다.

명세 §6.2 Evaluator 출력 schema 준수.
AGENTS.md §3-2 Tool Registry 인터페이스 의무.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from evaluators.base import Evaluator, EvaluatorReport, Severity
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX

FOOT_JOINTS: dict[str, int] = {
    "left_foot": NAME_TO_IDX["LEFT_FOOT"],
    "right_foot": NAME_TO_IDX["RIGHT_FOOT"],
}

# Severity thresholds (FootFloating ratio, 0~1)
SEV_LOW = 0.05
SEV_MED = 0.15
SEV_HIGH = 0.30

DEFAULT_TAU_FLOAT = 0.05  # 5cm — foot 가 ground 대비 5cm 이상 떠 있으면 floating


class FootFloatingEvaluator(Evaluator):
    """Foot floating 평가기.

    Quality-tier-agnostic — G1/G2 output 또는 synthetic injection 결과 모두에 적용.
    """

    name = "FootFloatingEvaluator"

    def __init__(self, tau_float: float = DEFAULT_TAU_FLOAT) -> None:
        """
        Args:
            tau_float: foot height 가 ground 대비 본 값보다 클 때 floating 으로 판정 (m 단위).
        """
        self.tau_float = tau_float

    def evaluate(
        self,
        motion: np.ndarray,
        fps: int = 20,
        ground_y: Optional[float] = None,
        contact_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> list[EvaluatorReport]:
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            raise ValueError(
                f"motion shape must be [T, 22, 3], got {motion.shape}. "
                "AGENTS.md §3-1 Canonical Motion Format."
            )
        T = motion.shape[0]
        if T < 2:
            return []

        # Ground 추정: 명시 안 됨이면 motion 전체에서 최저 y 값 (heuristic).
        if ground_y is None:
            ground_y = float(np.min(motion[:, :, 1]))

        reports: list[EvaluatorReport] = []
        for part, joint_idx in FOOT_JOINTS.items():
            foot_y = motion[:, joint_idx, 1]  # [T]
            height_above_ground = foot_y - ground_y  # [T]

            # contact 추정: 명시 안 됨이면 height_above_ground 가 작은 frame 을 contact 로 추정 (heuristic).
            if contact_labels is None:
                contact = height_above_ground < (self.tau_float * 2.0)
            else:
                contact = contact_labels[:, joint_idx].astype(bool)

            floating = (height_above_ground > self.tau_float) & contact  # [T]
            score = float(np.mean(floating))

            if score == 0.0:
                continue  # 본 foot 에 floating 없음 — report 생략

            # frame range — floating 이 일어난 첫·마지막 frame
            float_idx = np.where(floating)[0]
            start_frame = int(float_idx[0])
            end_frame = int(float_idx[-1])

            severity: Severity = _classify_severity(score)
            reports.append(
                EvaluatorReport(
                    agent=self.name,
                    error_type=f"{part}_floating",
                    body_part=part,
                    frames=(start_frame, end_frame),
                    score=score,
                    severity=severity,
                    recommendation="foot_lock_tool",
                    metadata={
                        "tau_float_m": self.tau_float,
                        "ground_y": float(ground_y),
                        "mean_height_above_ground": float(np.mean(height_above_ground)),
                        "max_height_above_ground": float(np.max(height_above_ground)),
                        "contact_frame_ratio": float(np.mean(contact)),
                    },
                )
            )
        return reports


def _classify_severity(score: float) -> Severity:
    if score >= SEV_HIGH:
        return "high"
    if score >= SEV_MED:
        return "medium"
    if score >= SEV_LOW:
        return "low"
    return "low"
