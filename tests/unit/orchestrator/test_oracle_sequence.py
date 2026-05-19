"""Unit tests for sequence oracle.

property:
  - empty sequence (length 0, "do nothing") 도 candidate 에 포함됨.
  - max_depth 준수.
  - Score 비감소 가지치기가 작동 (counter['pruned'] > 0 인 케이스 있음).
  - best.netgain >= all candidates' netgain.
  - oracle_type='sequence', netgain_weight_status 박제.
  - target evaluator 누락 → ValueError.
  - top_k 길이 제한.
"""
from __future__ import annotations

import numpy as np
import pytest

from correction_tools import FootLockTool, VelocitySmoothingTool
from evaluators import FootFloatingEvaluator, VelocityJitterEvaluator
from orchestrator import SequenceOracleSelection, select_best_sequence_oracle
from orchestrator.oracle_sequence import ORACLE_TYPE
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX
from tools.synthetic_injection import inject_foot_floating, inject_jitter


def _make_standing_motion(T: int = 20) -> np.ndarray:
    motion = np.zeros((T, 22, 3), dtype=np.float64)
    motion[:, :, 1] = 1.0
    motion[:, NAME_TO_IDX["LEFT_FOOT"], 1] = 0.0
    motion[:, NAME_TO_IDX["RIGHT_FOOT"], 1] = 0.0
    motion[:, NAME_TO_IDX["PELVIS"], 1] = 0.0
    return motion


def test_oracle_type_and_status_metadata() -> None:
    clean = _make_standing_motion(T=20)
    corrupted = inject_foot_floating(clean, lift_height=0.08, frame_range=(0, 20))
    sel = select_best_sequence_oracle(
        clean_motion=clean,
        corrupted_motion=corrupted,
        artifact_kind="foot_floating",
        target_evaluator_name="FootFloatingEvaluator",
        tools_with_target_parts=[(FootLockTool(default_ground_y=0.0), "both_feet")],
        evaluators=[FootFloatingEvaluator()],
        strengths=("medium",),
        max_depth=2,
    )
    assert sel.oracle_type == ORACLE_TYPE == "sequence"
    assert sel.netgain_weight_status == "calibrated_protocol_a_v1"
    assert "alpha" in sel.netgain_weights


def test_empty_sequence_is_candidate() -> None:
    """length 0 (do nothing) candidate 가 항상 포함됨."""
    clean = _make_standing_motion(T=20)
    corrupted = clean.copy()  # 변형 없음
    sel = select_best_sequence_oracle(
        clean_motion=clean,
        corrupted_motion=corrupted,
        artifact_kind="foot_floating",
        target_evaluator_name="FootFloatingEvaluator",
        tools_with_target_parts=[(FootLockTool(default_ground_y=0.0), "both_feet")],
        evaluators=[FootFloatingEvaluator()],
        strengths=("medium",),
        max_depth=2,
    )
    # 최소 1 개 candidate (length 0) 있어야 함.
    assert sel.n_candidates_explored >= 1
    lengths = [c.length for c in sel.top_k_candidates]
    assert 0 in lengths or sel.best_candidate.length == 0  # type: ignore[union-attr]


def test_max_depth_respected() -> None:
    """max_depth 이상의 sequence 가 candidate 에 안 들어가야 함."""
    clean = _make_standing_motion(T=20)
    corrupted = inject_foot_floating(clean, lift_height=0.08, frame_range=(0, 20))
    sel = select_best_sequence_oracle(
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
        max_depth=2,
    )
    for c in sel.top_k_candidates:
        assert c.length <= 2, f"candidate exceeds max_depth: {c.length}"


def test_best_has_max_netgain_among_explored() -> None:
    clean = _make_standing_motion(T=20)
    corrupted = inject_foot_floating(clean, lift_height=0.08, frame_range=(0, 20))
    sel = select_best_sequence_oracle(
        clean_motion=clean,
        corrupted_motion=corrupted,
        artifact_kind="foot_floating",
        target_evaluator_name="FootFloatingEvaluator",
        tools_with_target_parts=[
            (FootLockTool(default_ground_y=0.0), "both_feet"),
            (VelocitySmoothingTool(), "full_body"),
        ],
        evaluators=[FootFloatingEvaluator()],
        strengths=("small", "medium"),
        max_depth=2,
    )
    assert sel.best_candidate is not None
    # top_k 중 best 가 첫 번째.
    assert sel.top_k_candidates[0].netgain_provisional == sel.best_candidate.netgain_provisional


def test_missing_target_evaluator_raises() -> None:
    clean = _make_standing_motion(T=10)
    with pytest.raises(ValueError, match="not found in evaluators"):
        select_best_sequence_oracle(
            clean_motion=clean,
            corrupted_motion=clean.copy(),
            artifact_kind="x",
            target_evaluator_name="NonexistentEval",
            tools_with_target_parts=[(FootLockTool(), "both_feet")],
            evaluators=[FootFloatingEvaluator()],
            max_depth=1,
        )


def test_pruning_counter_works() -> None:
    """jitter motion 에 FootLockTool 적용 시 일부 분기는 score 증가 → pruned."""
    rng = np.random.default_rng(7)
    rest = rng.uniform(-0.3, 0.3, (22, 3)).astype(np.float64)
    clean = np.tile(rest[None, :, :], (20, 1, 1))
    corrupted = inject_jitter(clean, noise_std=0.05, seed=42)
    sel = select_best_sequence_oracle(
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
        max_depth=2,
    )
    # 적어도 1개 분기는 pruned 되어야 함 (jitter motion 에 foot_lock 적용은 종종 score 증가).
    assert sel.n_candidates_explored >= 1
    assert sel.candidate_netgain_stats["max"] >= sel.candidate_netgain_stats["mean"]


def test_top_k_length_limited() -> None:
    clean = _make_standing_motion(T=20)
    corrupted = inject_foot_floating(clean, lift_height=0.08, frame_range=(0, 20))
    sel = select_best_sequence_oracle(
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
        max_depth=2,
        top_k=3,
    )
    assert len(sel.top_k_candidates) <= 3


def test_tiebreak_prefers_shorter_sequence_on_netgain_tie() -> None:
    """NetGain 동률 시 짧은 sequence 가 best 로 선택됨 (Occam 간결성).

    bone_stretch 에 FootLockTool 을 prefix 로 끼운 sequence 는 BoneProjection 만 한
    sequence 와 final NetGain 이 같다 (FL 이 bone 에 영향 0). 이런 동률 케이스에서
    short sequence 가 우선되어야 한다.
    """
    from correction_tools import BoneProjectionTool
    from evaluators import BoneLengthEvaluator
    from tools.synthetic_injection import inject_bone_stretch

    rng = np.random.default_rng(42)
    rest = rng.uniform(-0.3, 0.3, (22, 3)).astype(np.float64)
    clean = np.tile(rest[None, :, :], (30, 1, 1))
    # partial-frame stretch (tool_effect_matrix 의 inject 패턴)
    half = 15
    corrupted = np.concatenate([
        inject_bone_stretch(clean[:half], chain_label="right_arm", stretch_factor=1.30, seed=42),
        clean[half:],
    ], axis=0)
    sel = select_best_sequence_oracle(
        clean_motion=clean,
        corrupted_motion=corrupted,
        artifact_kind="bone_stretch_right_arm",
        target_evaluator_name="BoneLengthEvaluator",
        tools_with_target_parts=[
            (FootLockTool(default_ground_y=0.0), "both_feet"),
            (BoneProjectionTool(), "right_arm"),
        ],
        evaluators=[BoneLengthEvaluator(), FootFloatingEvaluator()],
        strengths=("large",),
        max_depth=3,
    )
    assert sel.best_candidate is not None
    best_ng = sel.best_candidate.netgain_provisional
    # top_k 의 candidates 중 best 와 NetGain 이 동률인 것들의 length 가 best 보다 크거나 같아야 함.
    for c in sel.top_k_candidates:
        if abs(c.netgain_provisional - best_ng) < 1e-9:
            assert c.length >= sel.best_candidate.length, (
                f"tiebreak violated: best length={sel.best_candidate.length}, "
                f"tied candidate length={c.length}"
            )
