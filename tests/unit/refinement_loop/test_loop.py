"""Unit tests for RefinementLoop.

phase 03-test §2-2 property + reference oracle 패턴.

property:
  - clean motion → orchestrator STOP → loop 즉시 종료, tool_history 빈 list.
  - corrupted motion → tool 적용 → score 감소 → continue 또는 STOP.
  - max_iterations 도달 → max_iterations_reached=True.
  - tool 적용으로 score 가 악화되면 rollback + 종료.
  - unknown tool → break, converged=True.

AGENTS.md §3-4 Score 비감소 의무 + §6-13 integration smoke (가설 근거 인용 금지).
"""
from __future__ import annotations

import numpy as np
import pytest

from correction_tools import FootLockTool, VelocitySmoothingTool
from correction_tools.base import CorrectionReport
from evaluators import EvaluatorReport, FootFloatingEvaluator, VelocityJitterEvaluator
from evaluators.base import Evaluator
from orchestrator import RuleBasedOrchestrator
from orchestrator.base import Orchestrator, OrchestratorDecision
from refinement_loop import RefinementLoop, RefinementResult
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX
from tools.synthetic_injection import inject_foot_floating, inject_jitter


def _make_standing_motion(T: int = 30) -> np.ndarray:
    motion = np.zeros((T, 22, 3), dtype=np.float64)
    motion[:, :, 1] = 1.0
    motion[:, NAME_TO_IDX["LEFT_FOOT"], 1] = 0.0
    motion[:, NAME_TO_IDX["RIGHT_FOOT"], 1] = 0.0
    motion[:, NAME_TO_IDX["PELVIS"], 1] = 0.0  # ground anchor
    return motion


def _basic_loop(max_iterations: int = 5, stop_thresh: str = "low") -> RefinementLoop:
    """default loop fixture."""
    tools = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
    }
    return RefinementLoop(
        evaluators=[FootFloatingEvaluator(), VelocityJitterEvaluator()],
        correction_tools=tools,
        orchestrator=RuleBasedOrchestrator(
            tool_registry=list(tools.values()),
            stop_severity_threshold=stop_thresh,
        ),
        max_iterations=max_iterations,
    )


def test_clean_motion_immediate_stop() -> None:
    """clean (artifact 없음) motion → orchestrator STOP → 즉시 종료."""
    motion = _make_standing_motion(T=30)
    loop = _basic_loop()
    result = loop.run(motion)
    assert result.converged
    assert result.tool_history == []
    assert len(result.score_trace) == 1  # initial score 만
    assert result.metadata["stop_reason"] in ("no_evaluator_reports", "all_severities_below_threshold")
    np.testing.assert_array_equal(result.refined_motion, motion)


def test_foot_floating_corrupted_runs_and_reduces_score() -> None:
    """foot floating corrupted → tool 적용 → score 감소 → STOP 또는 max_iter."""
    motion = _make_standing_motion(T=30)
    corrupted = inject_foot_floating(motion, lift_height=0.08, frame_range=(0, 30))
    loop = _basic_loop(max_iterations=5)
    result = loop.run(corrupted)

    # score_trace 의 첫 score 가 마지막보다 작거나 같지 않아야 (악화 안 됨).
    assert result.score_trace[0] >= result.score_trace[-1] - loop.score_increase_tolerance, (
        f"score did not improve: {result.score_trace}"
    )
    # tool 적용이 1 번 이상 있어야 함 (corruption 이 있는 상태).
    assert len(result.tool_history) >= 1
    # rollback 시에도 converged=True. natural max_iter 일 경우는 False.
    if result.max_iterations_reached:
        assert not result.converged
    else:
        assert result.converged


def test_max_iterations_terminates() -> None:
    """max_iterations 도달 시 종료."""
    motion = _make_standing_motion(T=30)
    corrupted = inject_foot_floating(motion, lift_height=0.08, frame_range=(0, 30))
    loop = _basic_loop(max_iterations=2)
    result = loop.run(corrupted)
    # 5 step 안에 STOP 되거나, max_iter 도달.
    assert len(result.decision_history) <= 2


def test_unknown_tool_breaks() -> None:
    """orchestrator 가 selected_tool 를 알려줬는데 correction_tools dict 에 없으면 break."""

    class _FakeOrch(Orchestrator):
        name = "FakeOrch"

        def decide(self, reports, history, **kwargs):
            return OrchestratorDecision(
                decision="revise",
                primary_error="x",
                selected_tool="NonexistentTool",
                target_part="full_body",
                target_frames=(0, 5),
                strength="medium",
                next_step="apply_then_evaluate",
            )

    motion = _make_standing_motion(T=20)
    loop = RefinementLoop(
        evaluators=[FootFloatingEvaluator()],
        correction_tools={"FootLockTool": FootLockTool(default_ground_y=0.0)},
        orchestrator=_FakeOrch(),
        max_iterations=5,
    )
    result = loop.run(motion)
    assert result.converged
    assert "unknown_tool_'NonexistentTool'" in result.metadata["stop_reason"]
    assert result.tool_history == []


def test_rollback_on_score_increase() -> None:
    """tool 이 score 를 악화시키면 rollback + 종료."""

    class _IncreasingScoreEval(Evaluator):
        """매 호출마다 score 가 커지는 evaluator (rollback 유발용 mock)."""
        name = "IncreasingScoreEval"

        def __init__(self):
            self.call_count = 0

        def evaluate(self, motion, fps=20, ground_y=None, contact_labels=None, **kwargs):
            self.call_count += 1
            return [EvaluatorReport(
                agent=self.name,
                error_type="mock_high",
                body_part="full_body",
                frames=(0, motion.shape[0] - 1),
                score=0.1 * self.call_count,  # 매 evaluate 마다 증가
                severity="high",  # type: ignore[arg-type]
                recommendation="VelocitySmoothingTool",
            )]

    class _ForceReviseOrch(Orchestrator):
        name = "ForceReviseOrch"

        def decide(self, reports, history, **kwargs):
            if not reports:
                return OrchestratorDecision(decision="STOP")
            return OrchestratorDecision(
                decision="revise",
                primary_error=reports[0].error_type,
                selected_tool="VelocitySmoothingTool",
                target_part="full_body",
                target_frames=reports[0].frames,
                strength="small",
                next_step="apply_then_evaluate",
            )

    motion = _make_standing_motion(T=20)
    loop = RefinementLoop(
        evaluators=[_IncreasingScoreEval()],
        correction_tools={"VelocitySmoothingTool": VelocitySmoothingTool()},
        orchestrator=_ForceReviseOrch(),
        max_iterations=5,
        score_increase_tolerance=0.01,
    )
    result = loop.run(motion)
    assert result.rolled_back
    assert result.converged
    assert result.metadata["stop_reason"] == "rollback_score_increase"


def test_invalid_motion_shape_raises() -> None:
    loop = _basic_loop()
    with pytest.raises(ValueError, match=r"\[T, 22, 3\]"):
        loop.run(np.zeros((10, 21, 3)))


def test_score_trace_length_consistent_with_history() -> None:
    """score_trace length = 1 + len(tool_history) (accepted step 만 기록)."""
    motion = _make_standing_motion(T=30)
    corrupted = inject_jitter(motion, noise_std=0.05, seed=42)
    loop = _basic_loop(max_iterations=3)
    result = loop.run(corrupted)
    # accepted step 만 score 가 append 되므로 차이 = 1.
    expected_len = 1 + len(result.tool_history)
    assert len(result.score_trace) == expected_len, (
        f"score_trace={result.score_trace}, tool_history len={len(result.tool_history)}"
    )
