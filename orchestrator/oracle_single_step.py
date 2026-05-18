"""Single-step oracle best-tool baseline — Week 4 deliverable.

AGENTS.md §3-16 의 oracle type 명시 의무 — 본 모듈은 **`oracle_type = "single_step"`**.

알고리즘:
  1. corrupted motion + clean GT (synthetic injection 에서 가용) 가 주어짐.
  2. 후보 tool 각각을 (각 strength 별로) corrupted 에 **한 번씩** 적용 → corrected.
  3. 각 candidate 의 NetGain (provisional) 을 계산:
     NetGain = ArtifactReduction − α·FidelityLoss − β·CorrectionMagnitude − γ·ToolCallCost
     ArtifactReduction = − target_delta  (target_delta < 0 = 개선)
     FidelityLoss = MPJPE(corrected, clean) − MPJPE(corrupted, clean)   (Protocol A)
     CorrectionMagnitude = CorrectionReport.correction_magnitude
     ToolCallCost = 1                                                     (single-step)
  4. NetGain best 인 candidate 선택.

AGENTS.md §6-11 (provisional NetGain weight tagless 인용 금지) — 본 oracle 의 결과는
`netgain_weight_status="provisional"` 로 박제. α/β/γ 의 grid search 가 완료되어
`calibrated` 가 되기 전까지 본 oracle 결과를 외부 공개에 인용 시 "provisional"
태그 동반 의무.

AGENTS.md §6-12 (cross-evaluator side effects 미기록 금지) — 모든 candidate 의
target 외 evaluator 의 delta 도 함께 박제.

본 oracle 은 [`compute_tool_effect_matrix`](rule_based.py) 와 같은 측정 layer 의 동족
이지만, **선택 결정** 까지 한다는 점이 다르다 (행렬은 측정만, oracle 은 best 선택).

명세 §9.1 baseline B8 (Oracle best-tool, upper bound) 의 구현.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from correction_tools import (
    DEFAULT_CORRECTION_TOOLS,
    CorrectionReport,
    CorrectionTool,
)
from evaluators import DEFAULT_EVALUATORS, Evaluator, EvaluatorReport

#: Provisional NetGain weights — α/β/γ 가 perceptual rating grid search 전 임시값.
#: AGENTS.md §6-11 의무: 본 weight 로 계산된 NetGain 은 "provisional" 태그 동반.
DEFAULT_PROVISIONAL_NETGAIN_WEIGHTS: dict[str, float] = {
    "alpha": 1.0,  # FidelityLoss
    "beta": 1.0,   # CorrectionMagnitude
    "gamma": 0.1,  # ToolCallCost
}

ORACLE_TYPE = "single_step"  # AGENTS.md §3-16 의 명시 의무


@dataclass
class OracleCandidate:
    """한 (tool, strength) 후보의 평가 결과."""

    tool_name: str
    strength: str
    target_part: str
    target_score_before: float
    target_score_after: float
    target_delta: float
    fidelity_loss_protocol_a: float
    correction_magnitude: float
    cross_evaluator_delta: dict[str, float]
    netgain_provisional: float
    correction_report_metadata: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "strength": self.strength,
            "target_part": self.target_part,
            "target_score_before": self.target_score_before,
            "target_score_after": self.target_score_after,
            "target_delta": self.target_delta,
            "fidelity_loss_protocol_a": self.fidelity_loss_protocol_a,
            "correction_magnitude": self.correction_magnitude,
            "cross_evaluator_delta": self.cross_evaluator_delta,
            "netgain_provisional": self.netgain_provisional,
            "correction_report_metadata": self.correction_report_metadata,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass
class OracleSelection:
    """단일 sample 에 대한 single-step oracle 의 best 선택 + 모든 candidate.

    AGENTS.md §3-16 의 `oracle_type` field + §6-11 의 `netgain_weight_status` 박제.
    """

    artifact_kind: str
    target_evaluator: str
    oracle_type: str  # "single_step" — AGENTS.md §3-16
    netgain_weight_status: str  # "provisional" — AGENTS.md §6-11
    netgain_weights: dict[str, float]
    candidates: list[OracleCandidate]
    best_candidate: Optional[OracleCandidate]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "target_evaluator": self.target_evaluator,
            "oracle_type": self.oracle_type,
            "netgain_weight_status": self.netgain_weight_status,
            "netgain_weights": self.netgain_weights,
            "candidates": [c.to_dict() for c in self.candidates],
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "metadata": self.metadata,
        }


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Per-Joint Position Error — frame·joint 전체 L2 norm 평균.

    명세 §9.3 의 MPJPE 정의.
    """
    diff = a - b  # [T, 22, 3]
    return float(np.mean(np.linalg.norm(diff, axis=-1)))


def _aggregate_target_score(reports: list[EvaluatorReport]) -> float:
    """target evaluator 의 report 들을 단일 score 로 축약 (rule_based 와 동일 정의)."""
    if not reports:
        return 0.0
    return float(np.mean([r.score for r in reports]))


def select_best_tool_single_step(
    *,
    clean_motion: np.ndarray,
    corrupted_motion: np.ndarray,
    artifact_kind: str,
    target_evaluator_name: str,
    tools_with_target_parts: list[tuple[CorrectionTool, str]],
    evaluators: Optional[list[Evaluator]] = None,
    strengths: tuple[str, ...] = ("small", "medium", "large"),
    netgain_weights: Optional[dict[str, float]] = None,
    frame_range: Optional[tuple[int, int]] = None,
) -> OracleSelection:
    """Single-step oracle 의 best (tool, strength) 선택.

    Args:
        clean_motion: GT motion (Protocol A FidelityLoss 의 reference).
        corrupted_motion: synthetic injection 결과.
        artifact_kind: 측정 대상 artifact 명 (예: 'foot_floating').
        target_evaluator_name: artifact 의 target evaluator class name.
        tools_with_target_parts: 후보 tool 과 본 tool 의 natural target_part 의 list.
            예: [(FootLockTool(), "both_feet"), (BoneProjectionTool(), "right_arm"), ...].
        evaluators: cross-evaluator 측정에 사용할 evaluator list. None 이면 DEFAULT.
        strengths: 시험할 strength tuple.
        netgain_weights: provisional weights dict {"alpha": ..., "beta": ..., "gamma": ...}.
            None 이면 DEFAULT_PROVISIONAL_NETGAIN_WEIGHTS.
        frame_range: tool apply frame range. None 이면 (0, T-1).

    Returns:
        OracleSelection — best candidate + 모든 candidates + provisional 메타.
    """
    if evaluators is None:
        evaluators = list(DEFAULT_EVALUATORS)
    if netgain_weights is None:
        netgain_weights = dict(DEFAULT_PROVISIONAL_NETGAIN_WEIGHTS)

    T = corrupted_motion.shape[0]
    if frame_range is None:
        frame_range = (0, T - 1)

    target_eval = None
    for ev in evaluators:
        if ev.name == target_evaluator_name:
            target_eval = ev
            break
    if target_eval is None:
        raise ValueError(
            f"target_evaluator_name {target_evaluator_name!r} not found in evaluators "
            f"({[e.name for e in evaluators]})"
        )

    # before — corrupted 의 모든 evaluator + reference MPJPE.
    reports_before: dict[str, list[EvaluatorReport]] = {
        ev.name: ev.evaluate(corrupted_motion) for ev in evaluators
    }
    target_score_before = _aggregate_target_score(reports_before.get(target_evaluator_name, []))
    mpjpe_corrupted = _mpjpe(corrupted_motion, clean_motion)

    alpha = float(netgain_weights["alpha"])
    beta = float(netgain_weights["beta"])
    gamma = float(netgain_weights["gamma"])

    candidates: list[OracleCandidate] = []
    for tool, target_part in tools_with_target_parts:
        for strength in strengths:
            try:
                corrected, report = tool.apply(
                    corrupted_motion,
                    target_part=target_part,
                    target_joints=[],
                    frame_range=frame_range,
                    strength=strength,  # type: ignore[arg-type]
                )
            except ValueError as e:
                candidates.append(
                    OracleCandidate(
                        tool_name=type(tool).__name__,
                        strength=strength,
                        target_part=target_part,
                        target_score_before=target_score_before,
                        target_score_after=target_score_before,
                        target_delta=0.0,
                        fidelity_loss_protocol_a=0.0,
                        correction_magnitude=0.0,
                        cross_evaluator_delta={},
                        netgain_provisional=float("-inf"),  # skip 된 candidate 는 선택 안 되도록
                        skipped=True,
                        skip_reason=str(e),
                    )
                )
                continue

            reports_after = {ev.name: ev.evaluate(corrected) for ev in evaluators}
            target_score_after = _aggregate_target_score(reports_after.get(target_evaluator_name, []))
            target_delta = target_score_after - target_score_before

            mpjpe_corrected = _mpjpe(corrected, clean_motion)
            fidelity_loss = mpjpe_corrected - mpjpe_corrupted  # Protocol A
            artifact_reduction = -target_delta

            cross_delta: dict[str, float] = {}
            for ev_name in reports_before:
                if ev_name == target_evaluator_name:
                    continue
                b = _aggregate_target_score(reports_before[ev_name])
                a = _aggregate_target_score(reports_after.get(ev_name, []))
                cross_delta[ev_name] = float(a - b)

            netgain = artifact_reduction - alpha * fidelity_loss - beta * float(report.correction_magnitude) - gamma * 1.0

            candidates.append(
                OracleCandidate(
                    tool_name=type(tool).__name__,
                    strength=strength,
                    target_part=target_part,
                    target_score_before=target_score_before,
                    target_score_after=target_score_after,
                    target_delta=float(target_delta),
                    fidelity_loss_protocol_a=float(fidelity_loss),
                    correction_magnitude=float(report.correction_magnitude),
                    cross_evaluator_delta=cross_delta,
                    netgain_provisional=float(netgain),
                    correction_report_metadata=dict(report.metadata),
                )
            )

    # best 선정 (netgain_provisional max). 모두 skip 이면 None.
    valid = [c for c in candidates if not c.skipped]
    best = max(valid, key=lambda c: c.netgain_provisional) if valid else None

    return OracleSelection(
        artifact_kind=artifact_kind,
        target_evaluator=target_evaluator_name,
        oracle_type=ORACLE_TYPE,
        netgain_weight_status="provisional",
        netgain_weights=dict(netgain_weights),
        candidates=candidates,
        best_candidate=best,
        metadata={
            "n_candidates": len(candidates),
            "n_valid": len(valid),
            "n_skipped": len(candidates) - len(valid),
            "mpjpe_corrupted_vs_clean": mpjpe_corrupted,
            "frame_range": list(frame_range),
        },
    )
