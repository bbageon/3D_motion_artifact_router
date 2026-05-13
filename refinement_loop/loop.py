"""Closed-loop refinement implementation — 명세 §6.5.

RefinementLoop의 흐름:

    motion = base_generator(prompt)
    tool_history = []
    for step in range(max_iterations):
        reports = evaluator_layer.evaluate(motion)
        decision = orchestrator.decide(reports, tool_history)
        if decision.action == "STOP":
            break
        motion, tool_report = correction_tools[decision.tool].apply(...)
        tool_history.append(tool_report)
    refined_motion = motion

AGENTS.md §3-4 Closed-loop Score 비감소 의무.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from correction_tools.base import CorrectionReport, CorrectionTool
from evaluators.base import Evaluator, EvaluatorReport
from orchestrator.base import Orchestrator, OrchestratorDecision


@dataclass
class RefinementResult:
    """Refinement loop의 최종 결과.

    Attributes:
        original_motion: generator output (정규화 후).
        refined_motion: refinement 후 motion.
        tool_history: 본 loop에서 적용한 모든 tool call의 reports.
        decision_history: orchestrator의 step별 decisions.
        score_trace: step별 TotalArtifactScore 변화.
        converged: max_iterations 안에 STOP 도달 여부.
        max_iterations_reached: True if loop가 max_iterations에 도달해 종료.
        metadata: 추가 정보.
    """

    original_motion: np.ndarray
    refined_motion: np.ndarray
    tool_history: list[CorrectionReport]
    decision_history: list[OrchestratorDecision]
    score_trace: list[float]
    converged: bool
    max_iterations_reached: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class RefinementLoop:
    """명세 §6.5 closed-loop refinement.

    AGENTS.md §3-4 Score 비감소 의무:
    - step별 종합 Score (expected_artifact_reduction - fidelity_risk - conflict_risk - tool_cost)는
      비감소(non-decreasing)여야 한다.
    - Score가 악화되는 tool call은 reject 또는 rollback.
    - same (tool, target) 재호출은 strength를 줄여서만 허용 (oscillation 방지).
    """

    def __init__(
        self,
        evaluators: list[Evaluator],
        correction_tools: dict[str, CorrectionTool],
        orchestrator: Orchestrator,
        max_iterations: int = 5,
        score_decrease_tolerance: float = 0.01,
    ) -> None:
        """
        Args:
            evaluators: 사용할 evaluator list (registry).
            correction_tools: tool name → tool instance dict (registry).
            orchestrator: tool decision 알고리즘.
            max_iterations: refinement loop 최대 step.
            score_decrease_tolerance: Score 비감소 위반 허용 epsilon.
        """
        self.evaluators = evaluators
        self.correction_tools = correction_tools
        self.orchestrator = orchestrator
        self.max_iterations = max_iterations
        self.score_decrease_tolerance = score_decrease_tolerance

    def run(
        self,
        motion: np.ndarray,
        fps: int = 20,
        ground_y: Optional[float] = None,
        contact_labels: Optional[np.ndarray] = None,
    ) -> RefinementResult:
        """Refinement loop 실행.

        Args:
            motion: [T, 22, 3] canonical SMPL 22-joint, root-relative.
            fps: frames per second.
            ground_y: ground plane Y (있으면).
            contact_labels: [T, 22] bool (있으면).

        Returns:
            RefinementResult.

        구현 stub — Phase 1 (rule-based orchestrator) MVP 에서 작성.
        """
        # TODO Phase 1 MVP — 명세 §6.5의 의사코드 구현.
        # 1. for step in range(max_iterations):
        # 2.   reports = sum(e.evaluate(motion, fps, ground_y, contact_labels) for e in self.evaluators)
        # 3.   decision = orchestrator.decide(reports, tool_history)
        # 4.   if decision.decision == 'STOP': break
        # 5.   if decision.decision == 'reject': continue (or rollback)
        # 6.   tool = correction_tools[decision.selected_tool]
        # 7.   motion, report = tool.apply(motion, ..., decision.strength)
        # 8.   score_after = sum(NormalizedArtifactScore from new reports)
        # 9.   if score_after - score_before > score_decrease_tolerance: rollback (Score 비감소 의무)
        # 10.  tool_history.append(report)
        raise NotImplementedError(
            "RefinementLoop.run — Phase 1 MVP에서 구현 예정. "
            "명세 §6.5 closed-loop refinement 의사코드 참조."
        )
