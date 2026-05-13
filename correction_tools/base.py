"""CorrectionTool base interface.

명세 §6.3의 공통 인터페이스.
모든 correction tool은 본 인터페이스를 구현해 motion을 국소 보정하고 report를 반환한다.

AGENTS.md §3-2 Tool Registry 인터페이스 의무.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

Strength = Literal["small", "medium", "large"]


@dataclass
class CorrectionReport:
    """Tool 적용 결과 보고.

    명세 §6.3의 report 형식.

    Attributes:
        tool: tool 이름.
        target_part: 적용 body part.
        frame_range: 적용 frame 범위 [start, end].
        strength: 적용 강도.
        modified_joints: 실제 수정된 joint 이름 list.
        correction_magnitude: mean position delta (m).
        score_before: 적용 전 TotalArtifactScore (있으면).
        score_after: 적용 후 TotalArtifactScore (있으면).
        metadata: tool-specific 추가 정보.
    """

    tool: str
    target_part: str
    frame_range: tuple[int, int]
    strength: Strength
    modified_joints: list[str]
    correction_magnitude: float
    score_before: Optional[float] = None
    score_after: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CorrectionTool(ABC):
    """Correction tool abstract base class.

    명세 §6.3 공통 인터페이스.
    """

    name: str = "BaseTool"

    @abstractmethod
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
        """Motion을 국소 보정.

        Args:
            motion: [T, 22, 3] canonical SMPL 22-joint motion.
            target_part: 보정 대상 body part 이름 (예: 'right_foot').
            target_joints: 직접 수정할 joint 이름 list.
            frame_range: 보정 frame 범위 [start, end].
            strength: 'small'/'medium'/'large'.
            metadata: tool-specific 입력.

        Returns:
            (corrected_motion, report) tuple.
            corrected_motion shape == motion shape ([T, 22, 3]).
        """

    def kdg_affected_joints(self) -> list[str]:
        """본 tool이 직접 modify하는 joint set (KDG의 A(t)).

        명세 §6.4.0 Kinematic Dependency Graph 의 edge E_aff 정의.
        subclass에서 override.
        """
        raise NotImplementedError(
            "Subclass must override kdg_affected_joints() — 명세 §6.4.0 KDG affected joints 의무."
        )

    def kdg_propagation_weights(self) -> dict[str, float]:
        """본 tool이 forward kinematics를 통해 간접 영향을 주는 joint와 weight.

        명세 §6.4.0 edge E_prop (가중치 < 1) 정의.
        subclass에서 override (default 빈 dict — propagation 없음).
        """
        return {}

    def tool_class_hash(self) -> str:
        """본 tool 구현 코드의 sha256 hash."""
        import hashlib
        import inspect

        source = inspect.getsource(type(self))
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
