"""Orchestrator base interface.

명세 §6.4. evaluator reports + tool call history → tool decision.
rule_based / supervised_selector / contextual_bandit / multi_step_rl 알고리즘 분리.

AGENTS.md §3-3 KDG Ordering 의무.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from correction_tools.base import CorrectionReport, Strength
from evaluators.base import EvaluatorReport

Decision = Literal["STOP", "revise", "reject"]


@dataclass
class OrchestratorDecision:
    """Orchestrator의 결정.

    명세 §6.4 의 orchestrator 출력 형식.

    Attributes:
        decision: 'STOP' (종료) / 'revise' (tool 적용) / 'reject' (재시도).
        primary_error: 우선 처리할 artifact error_type (있으면).
        selected_tool: 선택한 tool 이름 (revise 시).
        target_part: 적용 body part.
        target_frames: 적용 frame 범위.
        strength: 'small'/'medium'/'large'.
        next_step: 'STOP' 또는 're_evaluate' 또는 'apply_then_evaluate'.
        score: 본 decision의 expected NetGain score (scoring.py 결과).
        metadata: 추가 정보 (예: KDG conflict_risk, history references).
    """

    decision: Decision
    primary_error: Optional[str] = None
    selected_tool: Optional[str] = None
    target_part: Optional[str] = None
    target_frames: Optional[tuple[int, int]] = None
    strength: Optional[Strength] = None
    next_step: str = "STOP"
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Orchestrator(ABC):
    """Orchestrator abstract base class.

    명세 §6.4 의 역할:
    1. evaluator reports 수집.
    2. artifact severity 정렬.
    3. primary error 결정.
    4. correction tool 선택.
    5. tool 적용 후 재평가 여부 결정.
    6. stop / revise / reject 판단.
    7. tool call trace 저장.
    8. KDG conflict detection.
    9. tool ordering.

    AGENTS.md §3-3 KDG Ordering 의무 — orchestrator는 KDG ordering rule을 위반하지 않아야 한다.
    """

    name: str = "BaseOrchestrator"

    @abstractmethod
    def decide(
        self,
        evaluator_reports: list[EvaluatorReport],
        tool_history: list[CorrectionReport],
        **kwargs: Any,
    ) -> OrchestratorDecision:
        """다음 action 결정.

        Args:
            evaluator_reports: 현재 motion에서 evaluator들이 산출한 reports.
            tool_history: 본 refinement loop의 이전 tool call reports.
            **kwargs: orchestrator-specific arguments (예: max_iterations, score thresholds).

        Returns:
            OrchestratorDecision — 다음 action.

        Raises:
            KDGOrderingViolation: KDG ordering rule 위반 시 (AGENTS.md §3-3).
        """

    def orchestrator_class_hash(self) -> str:
        """본 orchestrator 구현 코드의 sha256."""
        import hashlib
        import inspect

        source = inspect.getsource(type(self))
        return hashlib.sha256(source.encode("utf-8")).hexdigest()


class KDGOrderingViolation(Exception):
    """KDG ordering rule 위반 (AGENTS.md §3-3).

    Orchestrator decision이 다음 중 하나를 위반:
    1. ancestor joint 보정 전 descendant joint 보정.
    2. 동일 depth에서 soft tool이 hard-constraint tool보다 먼저.
    3. KDG ancestor-descendant 관계 joint를 동일 step에서 병렬 modify.
    """
