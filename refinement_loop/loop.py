"""Closed-loop refinement implementation — 명세 §6.5.

RefinementLoop 의 흐름 (prototype, 명세 §6.5 의사코드):

    motion0 = input
    tool_history = []
    score_trace = [TotalArtifactScore(motion0)]
    for step in range(max_iterations):
        reports = evaluator_layer.evaluate(motion)
        decision = orchestrator.decide(reports, tool_history)
        if decision == STOP or reject: break
        motion_candidate, report = correction_tools[decision.tool].apply(...)
        score_candidate = TotalArtifactScore(motion_candidate)
        if score_candidate > score_trace[-1] + tolerance:
            # AGENTS.md §3-4 — Score 비감소 의무. 본 candidate 는 score 악화 → rollback.
            break (with rollback metadata)
        motion = motion_candidate
        tool_history.append(report)
        score_trace.append(score_candidate)
    return RefinementResult(...)

AGENTS.md §3-4 Closed-loop Score 비감소 의무:
  본 prototype 에서는 `TotalArtifactScore = sum of all evaluator scores` 정의.
  값이 작을수록 좋음. step 마다 본 값이 직전보다 증가하면 (악화) tolerance 안
  이면 accept, 초과면 rollback (loop 종료, 직전 motion 반환).

AGENTS.md §6-13 integration smoke 의무:
  본 loop 의 single-sample 실행 결과는 integration test 결과로만 기록. 가설
  supports/contradicts 의 근거 인용 금지.
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
    """Refinement loop 의 최종 결과.

    Attributes:
        original_motion: input motion.
        refined_motion: refinement 후 motion (rollback 시 직전 step motion).
        tool_history: 본 loop 에서 successfully applied 된 tool call reports.
        decision_history: orchestrator 의 step 별 decisions.
        score_trace: step 별 TotalArtifactScore (=sum of evaluator scores).
        converged: STOP / reject / rollback 으로 자연 종료.
        max_iterations_reached: max_iterations 에 도달해 자연 종료.
        rolled_back: 마지막 step 의 candidate 가 score 악화로 rollback 되었음.
        metadata: 추가 정보 (rollback 사유, errors 등).
    """

    original_motion: np.ndarray
    refined_motion: np.ndarray
    tool_history: list[CorrectionReport]
    decision_history: list[OrchestratorDecision]
    score_trace: list[float]
    converged: bool
    max_iterations_reached: bool
    rolled_back: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class RefinementLoop:
    """Closed-loop refinement (prototype).

    AGENTS.md §3-4 Score 비감소 의무 — score (sum of evaluator scores) 가 step 별로
    증가하면 직전 motion 으로 rollback 후 loop 종료.
    AGENTS.md §6-13 — 본 loop 의 single-sample smoke 결과는 integration test 결과
    로만 기록. 가설 평가의 근거 아님.
    """

    def __init__(
        self,
        evaluators: list[Evaluator],
        correction_tools: dict[str, CorrectionTool],
        orchestrator: Orchestrator,
        max_iterations: int = 5,
        score_increase_tolerance: float = 0.01,
    ) -> None:
        """
        Args:
            evaluators: 사용할 evaluator list (registry).
            correction_tools: tool name (class name) → tool instance dict.
            orchestrator: tool decision 알고리즘.
            max_iterations: refinement loop 최대 step.
            score_increase_tolerance: score 가 직전보다 본 값 이하로 증가하면 accept.
                초과면 rollback.
        """
        self.evaluators = evaluators
        self.correction_tools = correction_tools
        self.orchestrator = orchestrator
        self.max_iterations = max_iterations
        self.score_increase_tolerance = score_increase_tolerance

    # --- helpers ---

    def _collect_reports(
        self,
        motion: np.ndarray,
        fps: int,
        ground_y: Optional[float],
        contact_labels: Optional[np.ndarray],
    ) -> list[EvaluatorReport]:
        reports: list[EvaluatorReport] = []
        for ev in self.evaluators:
            reports.extend(
                ev.evaluate(motion, fps=fps, ground_y=ground_y, contact_labels=contact_labels)
            )
        return reports

    @staticmethod
    def _total_artifact_score(reports: list[EvaluatorReport]) -> float:
        """명세 §9.3 의 TotalArtifactScore 의 prototype — sum of raw scores.

        본격 평가에서는 NormalizedArtifactScore + weighted sum 으로 대체 예정.
        본 prototype 에서는 raw sum (값 작을수록 좋음).
        """
        if not reports:
            return 0.0
        return float(sum(r.score for r in reports))

    # --- main loop ---

    def run(
        self,
        motion: np.ndarray,
        fps: int = 20,
        ground_y: Optional[float] = None,
        contact_labels: Optional[np.ndarray] = None,
    ) -> RefinementResult:
        """Refinement loop 실행.

        Args:
            motion: [T, 22, 3] canonical SMPL 22-joint.
            fps: frames per second.
            ground_y: 명시 ground plane Y (있으면 evaluator 에 전달).
            contact_labels: 명시 contact mask (있으면 evaluator 에 전달).

        Returns:
            RefinementResult.
        """
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            raise ValueError(
                f"motion shape must be [T, 22, 3], got {motion.shape}. "
                "AGENTS.md §3-1 Canonical Motion Format."
            )

        original = motion.copy()
        current = motion.copy()
        tool_history: list[CorrectionReport] = []
        decision_history: list[OrchestratorDecision] = []

        reports_current = self._collect_reports(current, fps, ground_y, contact_labels)
        score_current = self._total_artifact_score(reports_current)
        score_trace: list[float] = [score_current]

        converged = False
        max_iterations_reached = False
        rolled_back = False
        meta: dict[str, Any] = {
            "stop_reason": None,
            "errors": [],
            "rollback": None,
        }

        for step in range(self.max_iterations):
            decision = self.orchestrator.decide(reports_current, tool_history)
            decision_history.append(decision)

            if decision.decision == "STOP":
                meta["stop_reason"] = decision.metadata.get("stop_reason", "stop")
                converged = True
                break

            if decision.decision == "reject":
                meta["stop_reason"] = decision.metadata.get(
                    "reject_reason", "rejected"
                )
                converged = True
                break

            # revise: tool apply.
            tool_name = decision.selected_tool
            if tool_name is None or tool_name not in self.correction_tools:
                meta["stop_reason"] = f"unknown_tool_{tool_name!r}"
                converged = True
                break
            tool = self.correction_tools[tool_name]

            try:
                candidate, report = tool.apply(
                    current,
                    target_part=decision.target_part or "full_body",
                    target_joints=[],
                    frame_range=decision.target_frames or (0, current.shape[0] - 1),
                    strength=decision.strength or "medium",
                )
            except ValueError as e:
                meta["errors"].append({
                    "step": step,
                    "tool": tool_name,
                    "error": str(e),
                })
                meta["stop_reason"] = f"tool_apply_value_error_at_step_{step}"
                converged = True
                break

            # Score 비감소 의무 검증 (AGENTS.md §3-4).
            reports_candidate = self._collect_reports(
                candidate, fps, ground_y, contact_labels
            )
            score_candidate = self._total_artifact_score(reports_candidate)

            if score_candidate > score_current + self.score_increase_tolerance:
                # rollback — direct 비교: candidate 가 직전보다 score 증가 (악화).
                meta["rollback"] = {
                    "step": step,
                    "tool": tool_name,
                    "strength": decision.strength,
                    "score_before": score_current,
                    "score_candidate": score_candidate,
                    "delta": score_candidate - score_current,
                }
                meta["stop_reason"] = "rollback_score_increase"
                rolled_back = True
                converged = True
                break

            # accept.
            current = candidate
            tool_history.append(report)
            score_trace.append(score_candidate)
            score_current = score_candidate
            reports_current = reports_candidate
        else:
            # for/else: loop 자연 종료 (max_iterations 도달).
            max_iterations_reached = True

        return RefinementResult(
            original_motion=original,
            refined_motion=current,
            tool_history=tool_history,
            decision_history=decision_history,
            score_trace=score_trace,
            converged=converged,
            max_iterations_reached=max_iterations_reached,
            rolled_back=rolled_back,
            metadata={
                **meta,
                "max_iterations": self.max_iterations,
                "score_increase_tolerance": self.score_increase_tolerance,
            },
        )
