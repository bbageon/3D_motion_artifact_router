"""Evaluator base interface.

모든 evaluator(contact, temporal, skeletal, root_torso, upper_limb, lower_limb, coordination)는
본 인터페이스를 구현해 motion에서 artifact를 평가하고 EvaluatorReport list를 반환한다.

명세 §6.2 Evaluator Tool Registry 참조.
AGENTS.md §3-2 Tool Registry 인터페이스 의무.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

Severity = Literal["low", "medium", "high"]

#: ground 추정 percentile — tools/coords_protocol.GROUND_PERCENTILE 과 동일 semantics.
GROUND_PERCENTILE = 10.0


def estimate_ground_y(motion: np.ndarray, percentile: float = GROUND_PERCENTILE) -> float:
    """ground_y = frame별 lower-foot-height 의 하위 percentile (NOT min-Y).

    AR-062 audit A-2 fix (AR-063): 기존 min-Y(전 joint) 추정은 "어떤 joint 도 전-joint
    최저점 아래에 있을 수 없다"는 정의상 이유로 PenetrateEvaluator 를 vacuous 하게
    만들었다. 본 함수는 tools/coords_protocol.estimate_ground 와 동일 정의 (발이 '쉬는'
    높이; occasional penetration/float outlier 에 robust) 를 evaluator 계층에 제공한다.

    한계 (self-referential estimator 의 내재 성질): 관통/부양이 **transient**
    (< percentile 비율) 일 때만 robust — 양발이 전 구간 균일하게 뜨거나 percentile
    이상 비율로 지속 관통하면 ground 가 발을 따라가 검출 불가. 그런 경우 외부
    `ground_y` (예: trajectory 기반 floor, GT ground) 를 전달하라.

    LEFT_FOOT=10, RIGHT_FOOT=11 (canonical SMPL-22).
    """
    m = np.asarray(motion, dtype=np.float64)
    lower_foot_y = np.minimum(m[:, 10, 1], m[:, 11, 1])  # [T]
    return float(np.percentile(lower_foot_y, percentile))


@dataclass
class EvaluatorReport:
    """Evaluator 1개의 single-artifact 평가 결과.

    명세 §6.2의 evaluator output 형식과 동일.

    Attributes:
        agent: evaluator 이름 (예: 'ContactEvaluator').
        error_type: artifact 종류 (예: 'right_foot_sliding').
        body_part: 영향 받은 body part (예: 'right_foot').
        frames: 영향 frame 범위 [start, end].
        score: 본 artifact의 정량 score (높을수록 심각).
        severity: 'low' / 'medium' / 'high'.
        recommendation: 권장 correction tool 이름 (있는 경우).
        metadata: 추가 정보.
    """

    agent: str
    error_type: str
    body_part: str
    frames: tuple[int, int]
    score: float
    severity: Severity
    recommendation: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Evaluator(ABC):
    """Evaluator abstract base class."""

    name: str = "BaseEvaluator"

    @abstractmethod
    def evaluate(
        self,
        motion: np.ndarray,
        fps: int = 20,
        ground_y: Optional[float] = None,
        contact_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> list[EvaluatorReport]:
        """Motion에서 artifact를 평가.

        Args:
            motion: [T, 22, 3] canonical SMPL 22-joint, root-relative.
            fps: frames per second.
            ground_y: Skeleton Normalizer가 추정한 ground plane Y 좌표 (있으면).
            contact_labels: [T, 22] bool array (contact 추정, 있으면).
            **kwargs: evaluator-specific arguments.

        Returns:
            EvaluatorReport list. 본 evaluator가 감지한 artifact가 없으면 빈 list.
        """

    def evaluator_class_hash(self) -> str:
        """본 evaluator 구현 코드의 sha256 hash.

        [`data-versioning SKILL §2-3`](../../.claude/skills/data-versioning/SKILL.md) Evaluator config metadata 용.
        """
        import hashlib
        import inspect

        source = inspect.getsource(type(self))
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
