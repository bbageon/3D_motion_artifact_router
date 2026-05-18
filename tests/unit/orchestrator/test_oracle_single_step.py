"""Unit tests for single-step oracle.

phase 03-test §2-2 property + reference oracle 패턴.

property:
  - matching tool (예: foot_floating × FootLockTool) 이 best 로 선택돼야 함.
  - oracle_type == "single_step" (AGENTS.md §3-16 명시 의무).
  - netgain_weight_status == "provisional" (AGENTS.md §6-11 명시 의무).
  - best_candidate 의 NetGain 이 모든 다른 candidate 의 NetGain 보다 ≥.
  - skipped candidate 는 best 로 선택 안 됨.
  - target evaluator 누락 → ValueError.
"""
from __future__ import annotations

import numpy as np
import pytest

from correction_tools import BoneProjectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import FootFloatingEvaluator, VelocityJitterEvaluator
from orchestrator import OracleSelection, select_best_tool_single_step
from orchestrator.oracle_single_step import ORACLE_TYPE
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX
from tools.synthetic_injection import inject_foot_floating, inject_jitter


def _make_standing_motion(T: int = 30) -> np.ndarray:
    """PELVIS ground anchor (rule_based test 와 동일 fixture 패턴)."""
    motion = np.zeros((T, 22, 3), dtype=np.float64)
    motion[:, :, 1] = 1.0
    motion[:, NAME_TO_IDX["LEFT_FOOT"], 1] = 0.0
    motion[:, NAME_TO_IDX["RIGHT_FOOT"], 1] = 0.0
    motion[:, NAME_TO_IDX["PELVIS"], 1] = 0.0
    return motion


def test_oracle_type_and_weight_status_metadata() -> None:
    """AGENTS.md §3-16, §6-11 의 명시 의무 — field 가 박제돼 있어야 함."""
    clean = _make_standing_motion(T=20)
    corrupted = inject_foot_floating(clean, lift_height=0.08, frame_range=(0, 20))
    sel = select_best_tool_single_step(
        clean_motion=clean,
        corrupted_motion=corrupted,
        artifact_kind="foot_floating",
        target_evaluator_name="FootFloatingEvaluator",
        tools_with_target_parts=[
            (FootLockTool(default_ground_y=0.0), "both_feet"),
        ],
        evaluators=[FootFloatingEvaluator()],
        strengths=("medium",),
    )
    assert sel.oracle_type == ORACLE_TYPE == "single_step"
    assert sel.netgain_weight_status == "provisional"
    assert "alpha" in sel.netgain_weights
    assert "beta" in sel.netgain_weights
    assert "gamma" in sel.netgain_weights


def test_oracle_selects_matching_tool_for_foot_floating() -> None:
    """foot floating corrupted → FootLockTool 이 best 로 선택돼야 함."""
    clean = _make_standing_motion(T=30)
    corrupted = inject_foot_floating(clean, lift_height=0.08, frame_range=(0, 30))
    sel = select_best_tool_single_step(
        clean_motion=clean,
        corrupted_motion=corrupted,
        artifact_kind="foot_floating",
        target_evaluator_name="FootFloatingEvaluator",
        tools_with_target_parts=[
            (FootLockTool(default_ground_y=0.0), "both_feet"),
            (BoneProjectionTool(), "right_arm"),
            (VelocitySmoothingTool(), "full_body"),
        ],
        evaluators=[FootFloatingEvaluator(), VelocityJitterEvaluator()],
        strengths=("small", "medium", "large"),
    )
    assert sel.best_candidate is not None
    assert sel.best_candidate.tool_name == "FootLockTool"


def test_oracle_selects_matching_tool_for_global_jitter() -> None:
    """global jitter corrupted → VelocitySmoothingTool 이 best 로 선택돼야 함."""
    rng = np.random.default_rng(7)
    rest_pose = rng.uniform(-0.3, 0.3, (22, 3)).astype(np.float64)
    clean = np.tile(rest_pose[None, :, :], (30, 1, 1))
    corrupted = inject_jitter(clean, noise_std=0.05, seed=42)
    sel = select_best_tool_single_step(
        clean_motion=clean,
        corrupted_motion=corrupted,
        artifact_kind="global_jitter",
        target_evaluator_name="VelocityJitterEvaluator",
        tools_with_target_parts=[
            (FootLockTool(default_ground_y=0.0), "both_feet"),
            (VelocitySmoothingTool(), "full_body"),
        ],
        evaluators=[FootFloatingEvaluator(), VelocityJitterEvaluator()],
        strengths=("small", "medium", "large"),
    )
    assert sel.best_candidate is not None
    assert sel.best_candidate.tool_name == "VelocitySmoothingTool"


def test_oracle_best_has_max_netgain() -> None:
    """best 의 netgain_provisional 이 모든 valid candidate 중 최대."""
    clean = _make_standing_motion(T=30)
    corrupted = inject_foot_floating(clean, lift_height=0.08, frame_range=(0, 30))
    sel = select_best_tool_single_step(
        clean_motion=clean,
        corrupted_motion=corrupted,
        artifact_kind="foot_floating",
        target_evaluator_name="FootFloatingEvaluator",
        tools_with_target_parts=[
            (FootLockTool(default_ground_y=0.0), "both_feet"),
            (VelocitySmoothingTool(), "full_body"),
        ],
        evaluators=[FootFloatingEvaluator()],
        strengths=("small", "medium", "large"),
    )
    valid = [c for c in sel.candidates if not c.skipped]
    assert sel.best_candidate is not None
    assert sel.best_candidate.netgain_provisional == max(c.netgain_provisional for c in valid)


def test_oracle_skipped_tool_not_selected_as_best() -> None:
    """BoneProjectionTool 이 target_part='both_feet' 로 호출되면 ValueError → skipped."""
    clean = _make_standing_motion(T=20)
    corrupted = inject_foot_floating(clean, lift_height=0.08, frame_range=(0, 20))
    sel = select_best_tool_single_step(
        clean_motion=clean,
        corrupted_motion=corrupted,
        artifact_kind="foot_floating",
        target_evaluator_name="FootFloatingEvaluator",
        # BoneProjectionTool 에 잘못된 target_part — skip 처리되어야 함.
        tools_with_target_parts=[
            (FootLockTool(default_ground_y=0.0), "both_feet"),
            (BoneProjectionTool(), "both_feet"),  # ValueError 유발 (chain 아님)
        ],
        evaluators=[FootFloatingEvaluator()],
        strengths=("medium",),
    )
    bone_cands = [c for c in sel.candidates if c.tool_name == "BoneProjectionTool"]
    assert bone_cands and all(c.skipped for c in bone_cands)
    assert sel.best_candidate is not None
    assert sel.best_candidate.tool_name == "FootLockTool"


def test_oracle_missing_target_evaluator_raises() -> None:
    clean = _make_standing_motion(T=10)
    corrupted = clean.copy()
    with pytest.raises(ValueError, match="not found in evaluators"):
        select_best_tool_single_step(
            clean_motion=clean,
            corrupted_motion=corrupted,
            artifact_kind="foot_floating",
            target_evaluator_name="NonexistentEvaluator",
            tools_with_target_parts=[(FootLockTool(), "both_feet")],
            evaluators=[FootFloatingEvaluator()],
        )


def test_oracle_candidate_records_cross_evaluator_delta() -> None:
    """AGENTS.md §6-12 — best 외 candidate 도 cross_evaluator_delta 박제."""
    clean = _make_standing_motion(T=30)
    corrupted = inject_foot_floating(clean, lift_height=0.08, frame_range=(0, 30))
    sel = select_best_tool_single_step(
        clean_motion=clean,
        corrupted_motion=corrupted,
        artifact_kind="foot_floating",
        target_evaluator_name="FootFloatingEvaluator",
        tools_with_target_parts=[
            (FootLockTool(default_ground_y=0.0), "both_feet"),
            (VelocitySmoothingTool(), "full_body"),
        ],
        evaluators=[FootFloatingEvaluator(), VelocityJitterEvaluator()],
        strengths=("medium",),
    )
    # 모든 valid candidate 가 cross_evaluator_delta 안에 VelocityJitterEvaluator 키 가짐.
    for c in sel.candidates:
        if c.skipped:
            continue
        assert "VelocityJitterEvaluator" in c.cross_evaluator_delta
        # target 은 cross 에서 제외.
        assert "FootFloatingEvaluator" not in c.cross_evaluator_delta
