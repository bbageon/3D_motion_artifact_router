"""Rule-based orchestrator + tool effect matrix — Week 4 prototype.

명세 §6.4 의 rule-based orchestrator 구현. 본 알고리즘은 evaluator report 의
severity·score 순으로 정렬해 가장 심한 artifact 의 권장 tool 을 선택하는 단순
mapping. KDG ordering rule ([orchestrator/base.py](base.py) `KDGOrderingViolation`)
은 추후 `orchestrator/kdg.py` 도입 시 강화 예정. 본 prototype 에서는 evaluator 의
`recommendation` field 를 그대로 매핑한다.

본 모듈은 두 API 노출:

  1. `RuleBasedOrchestrator` — Orchestrator 인터페이스 구현. decide() 가
     OrchestratorDecision 을 반환.
  2. `compute_tool_effect_matrix` — 합성 데이터 + correction tool 등록을 받아
     (artifact 종류 × tool) 행렬을 구성. **target evaluator 의 score 변화뿐 아니라
     다른 evaluator (cross-evaluator side effects) 의 score 도 함께 기록** (AGENTS.md
     §6-12 cross-evaluator side effects 미기록 금지).

본 prototype 의 산출물은 [H-2026-204](../evals/hypotheses/H-2026-204.md) 의 Stage 1
(MVP Week 3) 의 tool effect matrix 의 underpinning 이며, 본 행렬은 향후 oracle
best-tool (single-step or sequence — AGENTS.md §3-16 oracle type 명시 의무) 의 후보
tool 선택 logic 의 입력으로 사용된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from correction_tools import DEFAULT_CORRECTION_TOOLS, CorrectionReport, CorrectionTool
from evaluators import DEFAULT_EVALUATORS, Evaluator, EvaluatorReport
from orchestrator.base import Orchestrator, OrchestratorDecision

#: severity 우선순위 — high > medium > low. STOP 결정의 임계 (severity 가 본 값 미만이면 STOP).
SEVERITY_RANK: dict[str, int] = {"low": 1, "medium": 2, "high": 3}

#: evaluator recommendation field 에서 받은 tool 이름 → CorrectionTool 인스턴스 매핑.
#: tool registry 를 본 dict 로 lookup. 명세 §6.4.1 artifact-tool compatibility matrix 의
#: prototype.
DEFAULT_TOOL_LOOKUP: dict[str, str] = {
    "foot_lock_tool": "FootLockTool",
    "bone_projection_tool": "BoneProjectionTool",
    "velocity_smoothing_tool": "VelocitySmoothingTool",
}

#: severity → tool strength mapping (간단한 default).
SEVERITY_TO_STRENGTH: dict[str, str] = {
    "low": "small",
    "medium": "medium",
    "high": "large",
}

#: artifact_kind → target evaluator (소속/책임 evaluator). synthetic injection 처럼
#: artifact_kind 가 사전 알려진 경우 decide() 에 hint 로 전달하면 본 evaluator 의
#: report 만 primary 후보로 사용. multi-evaluator 환경 (e.g. jitter motion 의 bone
#: length 가 임계 초과) 에서 잘못된 cross-evaluator primary 선택 차단.
#:
#: [W-2026-001 RESOLVED 이후 발견 — B5 v1 의 jitter→BoneProjectionTool mis-selection
#: bug (`evals/reports/2026-05-19_h_2026_204_rq1_threshold_2.md` §3-1)] 의 fix.
ARTIFACT_TO_TARGET_EVALUATOR: dict[str, str] = {
    "foot_floating": "FootFloatingEvaluator",
    "bone_stretch_right_arm": "BoneLengthEvaluator",
    "bone_stretch_left_arm": "BoneLengthEvaluator",
    "bone_stretch_right_leg": "BoneLengthEvaluator",
    "bone_stretch_left_leg": "BoneLengthEvaluator",
    "global_jitter": "VelocityJitterEvaluator",
}


class RuleBasedOrchestrator(Orchestrator):
    """Rule-based orchestrator (Week 4 prototype).

    Algorithm:
      1. evaluator_reports 중 severity ≥ STOP 임계 (default 'low') 가 1 개 이상 있으면 revise.
      2. severity 가 가장 높은 (tie 시 score 가 가장 큰) report 를 primary_error 로.
      3. primary_error 의 recommendation 을 tool 이름으로 받고, tool_lookup 에서 등록된
         tool 을 찾아 selected_tool 로.
      4. target_part, target_frames 는 evaluator report 의 body_part·frames 그대로.
      5. strength 는 severity 매핑.
      6. 결정 metadata 에 본 record 의 모든 evaluator score (cross-evaluator side
         effects 의 before snapshot) 를 박제.

    STOP 조건: 모든 report 의 severity 가 STOP 임계 미만이거나 reports 가 비어있음.
    """

    name = "RuleBasedOrchestrator"

    def __init__(
        self,
        tool_registry: Optional[list[CorrectionTool]] = None,
        stop_severity_threshold: str = "low",
    ) -> None:
        """
        Args:
            tool_registry: 사용 가능한 correction tool list. None 이면 DEFAULT_CORRECTION_TOOLS.
            stop_severity_threshold: 본 severity 미만인 report 만 있으면 STOP.
                'low' / 'medium' / 'high' 중 하나.
        """
        if stop_severity_threshold not in SEVERITY_RANK:
            raise ValueError(
                f"stop_severity_threshold {stop_severity_threshold!r} not in "
                f"{list(SEVERITY_RANK)}"
            )
        self.tool_registry: list[CorrectionTool] = (
            tool_registry if tool_registry is not None else list(DEFAULT_CORRECTION_TOOLS)
        )
        self._tool_by_class_name: dict[str, CorrectionTool] = {
            type(t).__name__: t for t in self.tool_registry
        }
        self.stop_severity_threshold = stop_severity_threshold

    def decide(
        self,
        evaluator_reports: list[EvaluatorReport],
        tool_history: list[CorrectionReport],
        artifact_kind_hint: Optional[str] = None,
        **kwargs: Any,
    ) -> OrchestratorDecision:
        """규칙 기반 결정.

        Args:
            evaluator_reports: 현재 motion 의 evaluator reports.
            tool_history: 이전 step 들의 CorrectionReport (현 prototype 에서는
                oscillation 방지의 보조 정보로만 사용).
            artifact_kind_hint: synthetic injection 처럼 artifact_kind 가 사전 알려진
                경우, [ARTIFACT_TO_TARGET_EVALUATOR](./rule_based.py) 의 매핑으로
                target evaluator 를 결정하고 본 evaluator 의 report 만 actionable
                후보로 사용. None 이면 모든 evaluator 의 report 가 후보 (real-world
                generator-output 시 default).

        Returns:
            OrchestratorDecision — STOP 또는 revise.
        """
        # STOP 조건 1: report 없음.
        if not evaluator_reports:
            return OrchestratorDecision(
                decision="STOP",
                next_step="STOP",
                metadata={
                    "orchestrator": self.name,
                    "stop_reason": "no_evaluator_reports",
                    "tool_history_len": len(tool_history),
                },
            )

        # artifact_kind_hint 적용: target evaluator 의 report 만 filter.
        if artifact_kind_hint is not None:
            target_eval = ARTIFACT_TO_TARGET_EVALUATOR.get(artifact_kind_hint)
            if target_eval is not None:
                evaluator_reports = [r for r in evaluator_reports if r.agent == target_eval]

        stop_rank = SEVERITY_RANK[self.stop_severity_threshold]
        actionable = [
            r for r in evaluator_reports if SEVERITY_RANK.get(r.severity, 0) >= stop_rank
        ]

        # STOP 조건 2: 모든 severity 가 임계 미만.
        if not actionable:
            return OrchestratorDecision(
                decision="STOP",
                next_step="STOP",
                metadata={
                    "orchestrator": self.name,
                    "stop_reason": "all_severities_below_threshold",
                    "n_reports": len(evaluator_reports),
                    "stop_severity_threshold": self.stop_severity_threshold,
                    "before_snapshot": _build_before_snapshot(evaluator_reports),
                    "tool_history_len": len(tool_history),
                },
            )

        # primary_error 선정: severity desc, score desc.
        actionable.sort(
            key=lambda r: (SEVERITY_RANK.get(r.severity, 0), r.score),
            reverse=True,
        )
        primary = actionable[0]

        # tool lookup.
        rec = primary.recommendation
        tool_class_name = DEFAULT_TOOL_LOOKUP.get(rec) if rec else None
        if tool_class_name is None or tool_class_name not in self._tool_by_class_name:
            # 매핑된 tool 이 registry 에 없음 — reject (재시도 신호) 또는 STOP.
            return OrchestratorDecision(
                decision="reject",
                primary_error=primary.error_type,
                next_step="STOP",
                metadata={
                    "orchestrator": self.name,
                    "reject_reason": f"no_tool_for_recommendation_{rec!r}",
                    "primary_error": primary.error_type,
                    "primary_severity": primary.severity,
                    "before_snapshot": _build_before_snapshot(evaluator_reports),
                    "tool_history_len": len(tool_history),
                },
            )

        selected_tool = self._tool_by_class_name[tool_class_name]
        strength = SEVERITY_TO_STRENGTH.get(primary.severity, "medium")

        return OrchestratorDecision(
            decision="revise",
            primary_error=primary.error_type,
            selected_tool=type(selected_tool).__name__,
            target_part=primary.body_part,
            target_frames=tuple(primary.frames),
            strength=strength,  # type: ignore[arg-type]
            next_step="apply_then_evaluate",
            score=primary.score,
            metadata={
                "orchestrator": self.name,
                "selected_tool_recommendation": rec,
                "primary_severity": primary.severity,
                "primary_agent": primary.agent,
                "before_snapshot": _build_before_snapshot(evaluator_reports),
                "tool_history_len": len(tool_history),
            },
        )


def _build_before_snapshot(reports: list[EvaluatorReport]) -> list[dict[str, Any]]:
    """현재 motion 의 evaluator report 들을 (cross-evaluator 비교용) 직렬화.

    AGENTS.md §6-12 cross-evaluator side effects 의 'before' 절반.
    """
    return [
        {
            "agent": r.agent,
            "error_type": r.error_type,
            "body_part": r.body_part,
            "severity": r.severity,
            "score": float(r.score),
        }
        for r in reports
    ]


# ---------------------------------------------------------------------------
# Tool effect matrix
# ---------------------------------------------------------------------------


@dataclass
class ToolEffectEntry:
    """Tool effect matrix 의 한 cell — (artifact 종류, tool) 쌍의 적용 결과.

    Cross-evaluator side effects (AGENTS.md §6-12) 를 위해 **target evaluator 뿐 아니라
    모든 evaluator 의 before/after score** 를 함께 저장. tool 이 target artifact 는
    줄이지만 다른 artifact 를 악화시키는 trade-off 를 그대로 노출.
    """

    artifact_kind: str
    tool_name: str
    strength: str
    target_evaluator: str
    target_score_before: float
    target_score_after: float
    target_delta: float
    correction_magnitude: float
    cross_evaluator_scores_before: dict[str, list[dict[str, Any]]]
    cross_evaluator_scores_after: dict[str, list[dict[str, Any]]]
    cross_evaluator_delta: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "tool_name": self.tool_name,
            "strength": self.strength,
            "target_evaluator": self.target_evaluator,
            "target_score_before": self.target_score_before,
            "target_score_after": self.target_score_after,
            "target_delta": self.target_delta,
            "correction_magnitude": self.correction_magnitude,
            "cross_evaluator_scores_before": self.cross_evaluator_scores_before,
            "cross_evaluator_scores_after": self.cross_evaluator_scores_after,
            "cross_evaluator_delta": self.cross_evaluator_delta,
            "metadata": self.metadata,
        }


def _evaluate_all(
    motion: np.ndarray,
    evaluators: list[Evaluator],
) -> dict[str, list[EvaluatorReport]]:
    return {ev.name: ev.evaluate(motion) for ev in evaluators}


def _aggregate_target_score(reports: list[EvaluatorReport]) -> float:
    """target evaluator 의 report 들을 단일 score 로 축약 (전 report 평균)."""
    if not reports:
        return 0.0
    return float(np.mean([r.score for r in reports]))


def _serialize_reports(d: dict[str, list[EvaluatorReport]]) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [
            {
                "error_type": r.error_type,
                "body_part": r.body_part,
                "severity": r.severity,
                "score": float(r.score),
            }
            for r in reports
        ]
        for name, reports in d.items()
    }


def compute_tool_effect_matrix(
    *,
    artifact_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    tools: Optional[list[CorrectionTool]] = None,
    evaluators: Optional[list[Evaluator]] = None,
    target_evaluator_by_artifact: dict[str, str],
    target_part_by_artifact: dict[str, str],
    strengths: tuple[str, ...] = ("medium",),
    frame_range_by_artifact: Optional[dict[str, tuple[int, int]]] = None,
) -> list[ToolEffectEntry]:
    """Tool effect matrix 계산.

    각 (artifact_kind × tool × strength) 조합에 대해:
      1. corrupted motion 으로 시작.
      2. tool 적용 → corrected motion.
      3. corrupted, corrected 양쪽에서 모든 evaluator 의 score 측정.
      4. target evaluator 의 score delta + 다른 evaluator 의 score delta 모두 기록
         (AGENTS.md §6-12).

    Args:
        artifact_pairs: (artifact_kind, clean_motion, corrupted_motion) tuple list.
        tools: 적용할 correction tool list. None 이면 DEFAULT_CORRECTION_TOOLS.
        evaluators: cross-effect 측정에 사용할 evaluator list. None 이면 DEFAULT_EVALUATORS.
        target_evaluator_by_artifact: artifact_kind → 본 artifact 의 target evaluator 이름.
            tool 효과의 'target metric' 식별용.
        target_part_by_artifact: artifact_kind → tool apply 시 사용할 target_part 인자.
        strengths: 시험할 strength tuple.
        frame_range_by_artifact: artifact_kind → frame_range. None 이면 (0, T-1).

    Returns:
        ToolEffectEntry list.
    """
    tools = tools if tools is not None else list(DEFAULT_CORRECTION_TOOLS)
    evaluators = evaluators if evaluators is not None else list(DEFAULT_EVALUATORS)
    fr_by_art = frame_range_by_artifact or {}

    entries: list[ToolEffectEntry] = []
    for artifact_kind, _clean, corrupted in artifact_pairs:
        T = corrupted.shape[0]
        frame_range = fr_by_art.get(artifact_kind, (0, T - 1))
        target_name = target_evaluator_by_artifact.get(artifact_kind)
        if target_name is None:
            raise ValueError(
                f"target_evaluator_by_artifact 에 {artifact_kind!r} 누락. "
                "tool effect 행렬 의 target evaluator 식별 필수."
            )
        target_part = target_part_by_artifact.get(artifact_kind, "full_body")

        # before — corrupted motion 의 모든 evaluator 측정.
        reports_before = _evaluate_all(corrupted, evaluators)
        before_score = _aggregate_target_score(reports_before.get(target_name, []))
        before_serialized = _serialize_reports(reports_before)

        for tool in tools:
            for strength in strengths:
                # tool apply.
                try:
                    corrected, report = tool.apply(
                        corrupted,
                        target_part=target_part,
                        target_joints=[],
                        frame_range=frame_range,
                        strength=strength,  # type: ignore[arg-type]
                    )
                except ValueError as e:
                    # 예: BoneProjectionTool 의 target_part 가 chain 아님 — skip 으로 기록.
                    entries.append(
                        ToolEffectEntry(
                            artifact_kind=artifact_kind,
                            tool_name=type(tool).__name__,
                            strength=strength,
                            target_evaluator=target_name,
                            target_score_before=before_score,
                            target_score_after=before_score,
                            target_delta=0.0,
                            correction_magnitude=0.0,
                            cross_evaluator_scores_before=before_serialized,
                            cross_evaluator_scores_after=before_serialized,
                            cross_evaluator_delta={},
                            metadata={"skipped": True, "reason": str(e)},
                        )
                    )
                    continue

                # after — corrected motion 의 모든 evaluator 측정.
                reports_after = _evaluate_all(corrected, evaluators)
                after_score = _aggregate_target_score(reports_after.get(target_name, []))
                after_serialized = _serialize_reports(reports_after)

                # cross-evaluator delta — target 이외 evaluator 별 score 평균 차.
                cross_delta: dict[str, float] = {}
                for ev_name in reports_before:
                    if ev_name == target_name:
                        continue
                    b = _aggregate_target_score(reports_before[ev_name])
                    a = _aggregate_target_score(reports_after.get(ev_name, []))
                    cross_delta[ev_name] = float(a - b)

                entries.append(
                    ToolEffectEntry(
                        artifact_kind=artifact_kind,
                        tool_name=type(tool).__name__,
                        strength=strength,
                        target_evaluator=target_name,
                        target_score_before=before_score,
                        target_score_after=after_score,
                        target_delta=float(after_score - before_score),
                        correction_magnitude=float(report.correction_magnitude),
                        cross_evaluator_scores_before=before_serialized,
                        cross_evaluator_scores_after=after_serialized,
                        cross_evaluator_delta=cross_delta,
                        metadata={
                            "tool_target_part": target_part,
                            "tool_frame_range": list(frame_range),
                            "tool_report_strength": report.strength,
                        },
                    )
                )
    return entries
