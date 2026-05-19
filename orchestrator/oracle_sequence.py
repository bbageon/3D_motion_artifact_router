"""Sequence oracle best-tool baseline — Week 4 deliverable (Priority 2).

AGENTS.md §3-16 의 oracle type 명시 의무 — 본 모듈은 `oracle_type="sequence"`.

Sequence oracle (시퀀스 신탁) 의 정의:
  한 sample 에 tool 을 여러 번 순차 적용한 모든 가능한 적용 순서 중 NetGain
  (provisional/calibrated) 이 가장 높은 시퀀스 (sequence) 를 고르는 baseline.
  closed-loop refinement 의 fair upper bound (공정한 상한선).

알고리즘:
  - DFS (Depth-First Search, 깊이 우선 탐색) 로 max_depth 까지의 모든 시퀀스를
    완전 열거 (exhaustive).
  - 각 step 의 TotalArtifactScore (모든 evaluator 의 raw score 합계, 작을수록
    좋음) 가 직전보다 tolerance 초과로 증가하면 본 분기는 가지치기 (prune) —
    AGENTS.md §3-4 의 Score 비감소 의무 와 동등한 제약.
  - 빈 sequence (length 0, = "아무것도 안 하기") 도 candidate 로 포함.

NetGain (명세 §9.4):
    NetGain = ArtifactReduction − α·FidelityLoss − β·CorrectionMagnitude − γ·ToolCallCost
    ArtifactReduction = − target_delta_total (= initial - final, target evaluator 기준)
    FidelityLoss = MPJPE(final, clean) − MPJPE(corrupted, clean)    (Protocol A)
    CorrectionMagnitude = sum of step-wise correction_magnitude
    ToolCallCost = len(sequence)

calibrated_protocol_a_v1 (α=5.0, β=0.0, γ=0.0) — Priority 1 의 grid search 결과.

AGENTS.md §6-12 (cross-evaluator side effects) — 모든 candidate 에 final state
의 모든 evaluator 의 score delta 박제.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from correction_tools.base import CorrectionTool
from evaluators import DEFAULT_EVALUATORS
from evaluators.base import Evaluator, EvaluatorReport

ORACLE_TYPE = "sequence"  # AGENTS.md §3-16

#: TotalArtifactScore — 모든 evaluator 의 raw score 합계 (loop 의 정의와 동일).
#: 작을수록 motion 이 깨끗 (artifact 적음).
def _total_artifact_score(reports_by_eval: dict[str, list[EvaluatorReport]]) -> float:
    total = 0.0
    for reports in reports_by_eval.values():
        for r in reports:
            total += float(r.score)
    return total


def _aggregate_target_score(reports: list[EvaluatorReport]) -> float:
    """target evaluator 의 report list 를 단일 score 로 축약 (= 평균)."""
    if not reports:
        return 0.0
    return float(np.mean([r.score for r in reports]))


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    """MPJPE (Mean Per-Joint Position Error) — frame·joint 별 L2 distance 평균.

    명세 §9.3 의 정의 — 두 motion 간의 거리 측정.
    """
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


@dataclass
class SequenceCandidate:
    """한 sequence (시퀀스, tool 적용 순서) 의 평가 결과."""

    sequence: list[tuple[str, str]]  # [(tool_class_name, strength), ...]
    length: int
    target_score_initial: float
    target_score_final: float
    target_delta_total: float
    fidelity_loss_protocol_a: float
    cumulative_correction_magnitude: float
    cross_evaluator_delta_total: dict[str, float]
    netgain_provisional: float
    intermediate_total_scores: list[float]  # [initial, after step 1, ..., after step L]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": [list(s) for s in self.sequence],
            "length": self.length,
            "target_score_initial": self.target_score_initial,
            "target_score_final": self.target_score_final,
            "target_delta_total": self.target_delta_total,
            "fidelity_loss_protocol_a": self.fidelity_loss_protocol_a,
            "cumulative_correction_magnitude": self.cumulative_correction_magnitude,
            "cross_evaluator_delta_total": self.cross_evaluator_delta_total,
            "netgain_provisional": self.netgain_provisional,
            "intermediate_total_scores": self.intermediate_total_scores,
            "metadata": self.metadata,
        }


@dataclass
class SequenceOracleSelection:
    """한 sample 에 대한 sequence oracle 의 best 선택 + top-K candidate."""

    artifact_kind: str
    target_evaluator: str
    oracle_type: str  # "sequence"
    netgain_weight_status: str
    netgain_weights: dict[str, float]
    max_depth: int
    score_increase_tolerance: float
    n_candidates_explored: int  # exhaustive 의 결과 총 개수
    n_candidates_pruned: int    # Score 비감소 위반으로 잘린 분기 수
    best_candidate: Optional[SequenceCandidate]
    top_k_candidates: list[SequenceCandidate]
    candidate_netgain_stats: dict[str, float]  # mean / median / p95 / max
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "target_evaluator": self.target_evaluator,
            "oracle_type": self.oracle_type,
            "netgain_weight_status": self.netgain_weight_status,
            "netgain_weights": self.netgain_weights,
            "max_depth": self.max_depth,
            "score_increase_tolerance": self.score_increase_tolerance,
            "n_candidates_explored": self.n_candidates_explored,
            "n_candidates_pruned": self.n_candidates_pruned,
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "top_k_candidates": [c.to_dict() for c in self.top_k_candidates],
            "candidate_netgain_stats": self.candidate_netgain_stats,
            "metadata": self.metadata,
        }


def _evaluate_all(
    motion: np.ndarray, evaluators: list[Evaluator]
) -> dict[str, list[EvaluatorReport]]:
    return {ev.name: ev.evaluate(motion) for ev in evaluators}


def select_best_sequence_oracle(
    *,
    clean_motion: np.ndarray,
    corrupted_motion: np.ndarray,
    artifact_kind: str,
    target_evaluator_name: str,
    tools_with_target_parts: list[tuple[CorrectionTool, str]],
    evaluators: Optional[list[Evaluator]] = None,
    strengths: tuple[str, ...] = ("small", "medium", "large"),
    netgain_weights: Optional[dict[str, float]] = None,
    netgain_weight_status: str = "calibrated_protocol_a_v1",
    max_depth: int = 3,
    score_increase_tolerance: float = 0.01,
    top_k: int = 10,
) -> SequenceOracleSelection:
    """Sequence oracle 의 best (sequence of tool applies) 선택.

    Args:
        clean_motion: GT motion (Protocol A FidelityLoss 의 reference).
        corrupted_motion: synthetic injection 결과.
        artifact_kind: 측정 대상 artifact 명.
        target_evaluator_name: artifact 의 target evaluator class name.
        tools_with_target_parts: 후보 tool 과 본 tool 의 natural target_part 의 list.
        evaluators: cross-evaluator 측정에 사용할 evaluator list. None 이면 DEFAULT.
        strengths: 시험할 strength tuple.
        netgain_weights: weights dict. None 이면 calibrated_protocol_a_v1 default.
        netgain_weight_status: status tag (AGENTS.md §6-11).
        max_depth: 최대 시퀀스 길이.
        score_increase_tolerance: AGENTS.md §3-4 의 Score 비감소 tolerance.
        top_k: 저장할 top candidate 개수.

    Returns:
        SequenceOracleSelection.
    """
    if evaluators is None:
        evaluators = list(DEFAULT_EVALUATORS)
    if netgain_weights is None:
        # default: calibrated_protocol_a_v1
        from orchestrator.oracle_single_step import (
            CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1,
        )
        netgain_weights = dict(CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1)

    # target evaluator 찾기
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

    T = corrupted_motion.shape[0]
    frame_range = (0, T - 1)
    alpha = float(netgain_weights["alpha"])
    beta = float(netgain_weights["beta"])
    gamma = float(netgain_weights["gamma"])

    # initial state
    reports_initial = _evaluate_all(corrupted_motion, evaluators)
    target_initial = _aggregate_target_score(reports_initial.get(target_evaluator_name, []))
    cross_initial: dict[str, float] = {
        name: _aggregate_target_score(reports_initial[name])
        for name in reports_initial
        if name != target_evaluator_name
    }
    total_initial = _total_artifact_score(reports_initial)
    mpjpe_corrupted = _mpjpe(corrupted_motion, clean_motion)

    # 9 actions = (tool, target_part, strength) combos
    actions: list[tuple[CorrectionTool, str, str]] = []
    for tool, tp in tools_with_target_parts:
        for s in strengths:
            actions.append((tool, tp, s))

    all_candidates: list[SequenceCandidate] = []
    counters = {"explored": 0, "pruned": 0}

    def _record_candidate(
        motion: np.ndarray,
        sequence: list[tuple[str, str]],
        cum_corr_mag: float,
        intermediate_scores: list[float],
    ) -> None:
        """현재 motion 을 candidate 로 저장 (빈 sequence 포함)."""
        reports = _evaluate_all(motion, evaluators)
        target_final = _aggregate_target_score(reports.get(target_evaluator_name, []))
        target_delta = target_final - target_initial
        mpjpe_final = _mpjpe(motion, clean_motion)
        fidelity_loss = mpjpe_final - mpjpe_corrupted
        artifact_reduction = -target_delta
        netgain = (
            artifact_reduction
            - alpha * fidelity_loss
            - beta * cum_corr_mag
            - gamma * len(sequence)
        )
        cross_delta: dict[str, float] = {}
        for name in reports:
            if name == target_evaluator_name:
                continue
            final_score = _aggregate_target_score(reports[name])
            cross_delta[name] = final_score - cross_initial.get(name, 0.0)

        all_candidates.append(
            SequenceCandidate(
                sequence=list(sequence),
                length=len(sequence),
                target_score_initial=target_initial,
                target_score_final=target_final,
                target_delta_total=float(target_delta),
                fidelity_loss_protocol_a=float(fidelity_loss),
                cumulative_correction_magnitude=float(cum_corr_mag),
                cross_evaluator_delta_total=cross_delta,
                netgain_provisional=float(netgain),
                intermediate_total_scores=list(intermediate_scores),
            )
        )
        counters["explored"] += 1

    def _dfs(
        motion: np.ndarray,
        sequence: list[tuple[str, str]],
        cum_corr_mag: float,
        prev_total: float,
        intermediate_scores: list[float],
    ) -> None:
        # 현재 state 를 candidate 로 기록 (length 0 부터 max_depth 까지 모두 후보).
        _record_candidate(motion, sequence, cum_corr_mag, intermediate_scores)

        if len(sequence) >= max_depth:
            return

        for tool, target_part, strength in actions:
            try:
                new_motion, report = tool.apply(
                    motion,
                    target_part=target_part,
                    target_joints=[],
                    frame_range=frame_range,
                    strength=strength,  # type: ignore[arg-type]
                )
            except ValueError:
                continue

            # Score 비감소 가지치기 (AGENTS.md §3-4).
            new_reports = _evaluate_all(new_motion, evaluators)
            new_total = _total_artifact_score(new_reports)
            if new_total > prev_total + score_increase_tolerance:
                counters["pruned"] += 1
                continue

            _dfs(
                new_motion,
                sequence + [(type(tool).__name__, strength)],
                cum_corr_mag + float(report.correction_magnitude),
                new_total,
                intermediate_scores + [new_total],
            )

    _dfs(
        corrupted_motion,
        [],
        0.0,
        total_initial,
        [total_initial],
    )

    if not all_candidates:
        # 빈 시퀀스 (do-nothing) 만이라도 record 됨. 빈 상태 안전 처리.
        best = None
        top_k_list: list[SequenceCandidate] = []
        stats: dict[str, float] = {}
    else:
        # Tie-break: NetGain 동률 시 **짧은 sequence** 우선 (Occam 의 간결성 원칙).
        # 이유: 동일 NetGain 을 얻을 수 있다면 더 적은 tool 호출이 simpler·cheaper·
        # tool_call_cost 면에서도 우월. 또한 closed-loop refinement 의 "추가 가치"
        # 측정에서 sequence 가 single-step 보다 길어야만 우위라고 주장하려면 strict
        # 우위여야 함 — 동률에 길이 우위는 closed-loop 의 fair upper bound 측정의
        # artifact. tuple key (NetGain desc, length asc) 로 정렬.
        best = max(
            all_candidates,
            key=lambda c: (c.netgain_provisional, -c.length),
        )
        top_k_list = sorted(
            all_candidates,
            key=lambda c: (c.netgain_provisional, -c.length),
            reverse=True,
        )[:top_k]
        netgains = np.array([c.netgain_provisional for c in all_candidates])
        stats = {
            "mean": float(netgains.mean()),
            "median": float(np.median(netgains)),
            "p95": float(np.percentile(netgains, 95)),
            "min": float(netgains.min()),
            "max": float(netgains.max()),
        }

    return SequenceOracleSelection(
        artifact_kind=artifact_kind,
        target_evaluator=target_evaluator_name,
        oracle_type=ORACLE_TYPE,
        netgain_weight_status=netgain_weight_status,
        netgain_weights=dict(netgain_weights),
        max_depth=max_depth,
        score_increase_tolerance=score_increase_tolerance,
        n_candidates_explored=counters["explored"],
        n_candidates_pruned=counters["pruned"],
        best_candidate=best,
        top_k_candidates=top_k_list,
        candidate_netgain_stats=stats,
        metadata={
            "mpjpe_corrupted_vs_clean": mpjpe_corrupted,
            "target_score_initial": target_initial,
            "total_score_initial": total_initial,
        },
    )
