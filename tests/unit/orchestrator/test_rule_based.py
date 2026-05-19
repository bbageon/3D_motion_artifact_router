"""Unit tests for RuleBasedOrchestrator + compute_tool_effect_matrix.

phase 03-test §2-2 property + reference oracle 패턴.

orchestrator:
  - empty reports → STOP.
  - 모든 severity < threshold → STOP.
  - severity high 1 개 + medium 다수 → high 가 primary.
  - tie severity 시 score 큰 쪽이 primary.
  - recommendation 매핑 안 됨 → reject.
  - decision metadata 에 before_snapshot 포함 (Guard 5 의 'before' 절반).

tool effect matrix:
  - target evaluator 의 score 감소 (corrupted 가 의도된 artifact 일 때).
  - cross-evaluator delta 가 반환 (target 이외 evaluator 도 포함).
  - 잘못된 target_part (예: foot artifact 에 chain 이름) 는 skip 으로 기록.
"""
from __future__ import annotations

import numpy as np
import pytest

from correction_tools import FootLockTool, VelocitySmoothingTool
from evaluators import (
    FootFloatingEvaluator,
    VelocityJitterEvaluator,
    EvaluatorReport,
)
from orchestrator import RuleBasedOrchestrator, compute_tool_effect_matrix
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX
from tools.synthetic_injection import inject_foot_floating, inject_jitter


def _mk_report(
    agent: str,
    error_type: str,
    body_part: str,
    score: float,
    severity: str,
    recommendation: str | None = None,
) -> EvaluatorReport:
    return EvaluatorReport(
        agent=agent,
        error_type=error_type,
        body_part=body_part,
        frames=(0, 10),
        score=score,
        severity=severity,  # type: ignore[arg-type]
        recommendation=recommendation,
    )


# --- RuleBasedOrchestrator ---


def test_orchestrator_empty_reports_stops() -> None:
    orch = RuleBasedOrchestrator()
    decision = orch.decide([], [])
    assert decision.decision == "STOP"
    assert decision.metadata["stop_reason"] == "no_evaluator_reports"


def test_orchestrator_all_below_threshold_stops() -> None:
    orch = RuleBasedOrchestrator(stop_severity_threshold="medium")
    reports = [
        _mk_report("FootFloatingEvaluator", "left_foot_floating", "left_foot", 0.03, "low", "foot_lock_tool"),
    ]
    decision = orch.decide(reports, [])
    assert decision.decision == "STOP"
    assert decision.metadata["stop_reason"] == "all_severities_below_threshold"
    # before_snapshot 박제 (Guard 5 의 일부)
    assert "before_snapshot" in decision.metadata
    assert len(decision.metadata["before_snapshot"]) == 1


def test_orchestrator_high_severity_takes_precedence() -> None:
    orch = RuleBasedOrchestrator()
    reports = [
        _mk_report("VelocityJitterEvaluator", "legs_velocity_jitter", "legs", 0.5, "medium", "velocity_smoothing_tool"),
        _mk_report("FootFloatingEvaluator", "right_foot_floating", "right_foot", 0.4, "high", "foot_lock_tool"),
    ]
    decision = orch.decide(reports, [])
    assert decision.decision == "revise"
    assert decision.primary_error == "right_foot_floating"
    assert decision.selected_tool == "FootLockTool"
    assert decision.strength == "large"  # high → large
    assert decision.target_part == "right_foot"


def test_orchestrator_tie_severity_breaks_by_score() -> None:
    orch = RuleBasedOrchestrator()
    reports = [
        _mk_report("FootFloatingEvaluator", "left_foot_floating", "left_foot", 0.6, "medium", "foot_lock_tool"),
        _mk_report("VelocityJitterEvaluator", "legs_velocity_jitter", "legs", 0.8, "medium", "velocity_smoothing_tool"),
    ]
    decision = orch.decide(reports, [])
    assert decision.decision == "revise"
    assert decision.primary_error == "legs_velocity_jitter"  # score 0.8 > 0.6
    assert decision.selected_tool == "VelocitySmoothingTool"


def test_orchestrator_unknown_recommendation_rejects() -> None:
    orch = RuleBasedOrchestrator()
    reports = [
        _mk_report("UnknownEvaluator", "some_error", "head", 0.5, "high", "unknown_tool"),
    ]
    decision = orch.decide(reports, [])
    assert decision.decision == "reject"
    assert "no_tool_for_recommendation" in decision.metadata["reject_reason"]
    assert "before_snapshot" in decision.metadata


def test_orchestrator_hint_jitter_selects_velocity_smoothing() -> None:
    """artifact_kind_hint='global_jitter' 시 BoneLengthEvaluator 가 더 높은 severity
    인 환경에서도 VelocityJitterEvaluator 의 report 만 primary 후보가 되어
    VelocitySmoothingTool 이 선택되어야 한다.

    [W-2026-001 RESOLVED 이후 발견된 B5 v1 bug]
    (jitter motion 에 BoneLengthEvaluator 가 'high' severity report → BoneProjectionTool
    잘못 선택) 의 fix 검증.
    """
    orch = RuleBasedOrchestrator()
    reports = [
        # Jitter motion 에 BoneLengthEvaluator 가 더 큰 severity 를 보고 (실제 발생 case)
        _mk_report("BoneLengthEvaluator", "spine_head_bone_length_variation", "spine_head",
                   0.7, "high", "bone_projection_tool"),
        # VelocityJitter 는 medium 또는 score 가 더 작음
        _mk_report("VelocityJitterEvaluator", "global_velocity_jitter", "full_body",
                   0.3, "medium", "velocity_smoothing_tool"),
    ]
    # hint 없으면 (default) BoneProjectionTool 선택 (v1 bug 재현)
    d_no_hint = orch.decide(reports, [], artifact_kind_hint=None)
    assert d_no_hint.selected_tool == "BoneProjectionTool"
    # hint 있으면 VelocitySmoothingTool 선택 (v2 fix)
    d_hint = orch.decide(reports, [], artifact_kind_hint="global_jitter")
    assert d_hint.decision == "revise"
    assert d_hint.selected_tool == "VelocitySmoothingTool"
    assert d_hint.primary_error == "global_velocity_jitter"


def test_orchestrator_hint_bone_selects_bone_projection() -> None:
    """artifact_kind_hint='bone_stretch_right_arm' → BoneLengthEvaluator filter."""
    orch = RuleBasedOrchestrator()
    reports = [
        _mk_report("FootFloatingEvaluator", "right_foot_floating", "right_foot",
                   0.8, "high", "foot_lock_tool"),
        _mk_report("BoneLengthEvaluator", "right_arm_bone_length_variation", "right_arm",
                   0.4, "medium", "bone_projection_tool"),
    ]
    d_hint = orch.decide(reports, [], artifact_kind_hint="bone_stretch_right_arm")
    assert d_hint.decision == "revise"
    assert d_hint.selected_tool == "BoneProjectionTool"
    assert d_hint.primary_error == "right_arm_bone_length_variation"


def test_orchestrator_hint_foot_selects_foot_lock() -> None:
    """artifact_kind_hint='foot_floating' → FootFloatingEvaluator filter."""
    orch = RuleBasedOrchestrator()
    reports = [
        _mk_report("VelocityJitterEvaluator", "global_velocity_jitter", "full_body",
                   0.9, "high", "velocity_smoothing_tool"),
        _mk_report("FootFloatingEvaluator", "left_foot_floating", "left_foot",
                   0.3, "medium", "foot_lock_tool"),
    ]
    d_hint = orch.decide(reports, [], artifact_kind_hint="foot_floating")
    assert d_hint.decision == "revise"
    assert d_hint.selected_tool == "FootLockTool"
    assert d_hint.primary_error == "left_foot_floating"


def test_orchestrator_hint_unknown_artifact_falls_back() -> None:
    """artifact_kind_hint 가 ARTIFACT_TO_TARGET_EVALUATOR 에 없으면 filter 없이 동작."""
    orch = RuleBasedOrchestrator()
    reports = [
        _mk_report("FootFloatingEvaluator", "left_foot_floating", "left_foot",
                   0.5, "high", "foot_lock_tool"),
    ]
    d_hint = orch.decide(reports, [], artifact_kind_hint="unknown_kind")
    assert d_hint.decision == "revise"
    assert d_hint.selected_tool == "FootLockTool"


def test_orchestrator_jitter_hint_applies_override_full_body_medium() -> None:
    """artifact_kind_hint='global_jitter' 시 ARTIFACT_TOOL_CONFIG_OVERRIDE 가
    strength='medium', target_part='full_body' 로 강제 override 한다.

    [B5 v2 → v3 fix] severity-based small strength + evaluator 의 body_part='legs'
    가 fixed (B2) 보다 약한 효과를 주는 한계를 해소.
    """
    orch = RuleBasedOrchestrator()
    reports = [
        # severity=low (default → strength=small), body_part=legs
        _mk_report("VelocityJitterEvaluator", "legs_velocity_jitter", "legs",
                   0.025, "low", "velocity_smoothing_tool"),
    ]
    d = orch.decide(reports, [], artifact_kind_hint="global_jitter")
    assert d.decision == "revise"
    assert d.selected_tool == "VelocitySmoothingTool"
    # override 적용 확인.
    assert d.strength == "medium", f"expected medium override, got {d.strength}"
    assert d.target_part == "full_body", f"expected full_body override, got {d.target_part}"
    assert d.metadata["override_applied"] is True


def test_orchestrator_no_override_for_bone_artifact() -> None:
    """ARTIFACT_TOOL_CONFIG_OVERRIDE 에 없는 artifact 는 default 동작 (severity/body_part)."""
    orch = RuleBasedOrchestrator()
    reports = [
        _mk_report("BoneLengthEvaluator", "right_arm_bone_length_variation", "right_arm",
                   0.15, "high", "bone_projection_tool"),
    ]
    d = orch.decide(reports, [], artifact_kind_hint="bone_stretch_right_arm")
    assert d.decision == "revise"
    assert d.selected_tool == "BoneProjectionTool"
    # default: severity=high → strength=large, body_part=right_arm.
    assert d.strength == "large"
    assert d.target_part == "right_arm"
    assert d.metadata["override_applied"] is False


def test_orchestrator_hint_target_evaluator_no_reports_stops() -> None:
    """target evaluator 가 report 가 없으면 STOP."""
    orch = RuleBasedOrchestrator()
    reports = [
        _mk_report("BoneLengthEvaluator", "right_arm_bone_length_variation", "right_arm",
                   0.5, "high", "bone_projection_tool"),
    ]
    d_hint = orch.decide(reports, [], artifact_kind_hint="global_jitter")
    # VelocityJitterEvaluator 의 report 가 없으므로 STOP
    assert d_hint.decision == "STOP"


def test_orchestrator_decision_metadata_includes_before_snapshot() -> None:
    """Guard 5 의 'before' snapshot — decision 시점의 모든 evaluator score 가 박제됨."""
    orch = RuleBasedOrchestrator()
    reports = [
        _mk_report("FootFloatingEvaluator", "left_foot_floating", "left_foot", 0.6, "high", "foot_lock_tool"),
        _mk_report("VelocityJitterEvaluator", "global_velocity_jitter", "full_body", 0.4, "low", "velocity_smoothing_tool"),
    ]
    decision = orch.decide(reports, [])
    snap = decision.metadata["before_snapshot"]
    assert len(snap) == 2
    agents = {s["agent"] for s in snap}
    assert agents == {"FootFloatingEvaluator", "VelocityJitterEvaluator"}


# --- compute_tool_effect_matrix ---


def _make_standing_motion(T: int = 30) -> np.ndarray:
    """Test fixture — ground anchor 가 lift 후에도 유지되도록 구성.

    auto ground_y 추정 (= motion.min y) 이 발 lift 와 무관하게 0 으로 유지되려면
    발이 아닌 다른 joint 도 y=0 에 박혀있어야 한다. 본 test 에서는 PELVIS 를
    ground anchor 로 사용 (anatomically 비현실적이지만 unit test 범위에서는 무방).
    """
    motion = np.zeros((T, 22, 3), dtype=np.float64)
    motion[:, :, 1] = 1.0
    motion[:, NAME_TO_IDX["LEFT_FOOT"], 1] = 0.0
    motion[:, NAME_TO_IDX["RIGHT_FOOT"], 1] = 0.0
    motion[:, NAME_TO_IDX["PELVIS"], 1] = 0.0  # ground anchor
    return motion


def test_tool_effect_matrix_foot_lock_reduces_floating() -> None:
    """foot floating corrupted 에 FootLockTool 적용 → target_delta < 0.

    v1.2.0 FootFloating contact heuristic 은 'velocity 정지 AND height ≤ tau_contact_height'
    이므로 lift_height 를 (tau_float, tau_contact_height) 범위 안에 둬 contact + floating
    동시 만족하도록 한다. 0.08 m = tau_float 0.05 < 0.08 < tau_contact_height 0.10.
    """
    clean = _make_standing_motion(T=30)
    corrupted = inject_foot_floating(clean, lift_height=0.08, frame_range=(0, 30))
    entries = compute_tool_effect_matrix(
        artifact_pairs=[("foot_floating", clean, corrupted)],
        tools=[FootLockTool(default_ground_y=0.0)],
        evaluators=[FootFloatingEvaluator()],  # target only — single-evaluator 케이스
        target_evaluator_by_artifact={"foot_floating": "FootFloatingEvaluator"},
        target_part_by_artifact={"foot_floating": "both_feet"},
        strengths=("large",),
    )
    assert len(entries) == 1
    e = entries[0]
    assert e.tool_name == "FootLockTool"
    assert e.target_evaluator == "FootFloatingEvaluator"
    # target score 가 의미 있게 감소해야 함.
    assert e.target_delta < -0.1, f"expected significant decrease, got {e.target_delta}"
    assert e.correction_magnitude > 0.0


def test_tool_effect_matrix_records_cross_evaluator_delta() -> None:
    """Guard 5 — target evaluator 외에 다른 evaluator 의 delta 도 함께 기록."""
    clean = _make_standing_motion(T=30)
    # 동일 사유 — lift_height 0.08 (tau_float < lift < tau_contact_height).
    corrupted = inject_foot_floating(clean, lift_height=0.08, frame_range=(0, 30))
    entries = compute_tool_effect_matrix(
        artifact_pairs=[("foot_floating", clean, corrupted)],
        tools=[FootLockTool(default_ground_y=0.0)],
        # target = FootFloating, side = VelocityJitter
        evaluators=[FootFloatingEvaluator(), VelocityJitterEvaluator()],
        target_evaluator_by_artifact={"foot_floating": "FootFloatingEvaluator"},
        target_part_by_artifact={"foot_floating": "both_feet"},
        strengths=("medium",),
    )
    assert len(entries) == 1
    e = entries[0]
    # cross_evaluator_delta 는 target 외에 다른 evaluator 1 개 포함.
    assert "VelocityJitterEvaluator" in e.cross_evaluator_delta
    # foot 을 위로 끌어내리면 velocity 가 약간 발생 → delta 가 정의됨 (값 자체는 ≥ 0 이어야 의미)
    # 보장: dict 가 빈 게 아니라 키 존재.
    assert "FootFloatingEvaluator" not in e.cross_evaluator_delta  # target 은 cross 에서 제외


def test_tool_effect_matrix_jitter_x_smoothing_reduces_target() -> None:
    """jitter corrupted 에 VelocitySmoothingTool → VelocityJitter target_delta < 0."""
    clean = np.tile(
        np.random.default_rng(7).uniform(-0.3, 0.3, (22, 3))[None, :, :], (30, 1, 1)
    ).astype(np.float64)
    corrupted = inject_jitter(clean, noise_std=0.05, seed=42)
    entries = compute_tool_effect_matrix(
        artifact_pairs=[("global_jitter", clean, corrupted)],
        tools=[VelocitySmoothingTool()],
        evaluators=[VelocityJitterEvaluator(), FootFloatingEvaluator()],
        target_evaluator_by_artifact={"global_jitter": "VelocityJitterEvaluator"},
        target_part_by_artifact={"global_jitter": "full_body"},
        strengths=("large",),
    )
    assert len(entries) == 1
    e = entries[0]
    assert e.target_delta < 0.0, f"expected jitter decrease, got {e.target_delta}"
    assert "FootFloatingEvaluator" in e.cross_evaluator_delta


def test_tool_effect_matrix_invalid_target_part_skipped() -> None:
    """BoneProjectionTool 의 target_part='both_feet' 는 chain 이름이 아니라 ValueError → skip 처리."""
    from correction_tools import BoneProjectionTool

    clean = _make_standing_motion(T=20)
    corrupted = inject_foot_floating(clean, lift_height=0.10, frame_range=(0, 20))
    entries = compute_tool_effect_matrix(
        artifact_pairs=[("foot_floating", clean, corrupted)],
        tools=[BoneProjectionTool()],  # 잘못된 tool — chain 입력 기대
        evaluators=[FootFloatingEvaluator()],
        target_evaluator_by_artifact={"foot_floating": "FootFloatingEvaluator"},
        target_part_by_artifact={"foot_floating": "both_feet"},  # chain 아님
        strengths=("medium",),
    )
    assert len(entries) == 1
    assert entries[0].metadata.get("skipped") is True


def test_tool_effect_matrix_target_evaluator_missing_raises() -> None:
    clean = _make_standing_motion(T=10)
    corrupted = clean.copy()
    with pytest.raises(ValueError, match="target_evaluator_by_artifact"):
        compute_tool_effect_matrix(
            artifact_pairs=[("foo", clean, corrupted)],
            tools=[FootLockTool()],
            evaluators=[FootFloatingEvaluator()],
            target_evaluator_by_artifact={},  # 누락
            target_part_by_artifact={"foo": "both_feet"},
        )
