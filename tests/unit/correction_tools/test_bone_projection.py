"""Unit tests for BoneProjectionTool.

property: bone-stretched motion 에 tool 적용 → BoneLengthEvaluator score 감소.
property: 다른 chain 은 영향 없음 (target chain 만 modify).
property: KDG affected joints 는 PELVIS 외 모든 joint.
"""
from __future__ import annotations

import numpy as np
import pytest

from correction_tools import BoneProjectionTool
from evaluators import BoneLengthEvaluator
from skeleton_normalizer.canonical_smpl_22 import (
    CHAIN_LABELS,
    T2M_KINEMATIC_CHAIN,
)
from tools.synthetic_injection import inject_bone_stretch


def make_rigid_motion(T: int = 40, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rest_pose = rng.uniform(low=-0.5, high=0.5, size=(22, 3)).astype(np.float64)
    motion = np.tile(rest_pose[None, :, :], (T, 1, 1))
    for t in range(T):
        motion[t] += np.array([0.01 * t, 0.0, 0.0])
    return motion


def test_kdg_affected_joints_includes_chain() -> None:
    tool = BoneProjectionTool()
    affected = tool.kdg_affected_joints()
    # PELVIS (root) 제외, 모든 chain 의 child 들 포함.
    for chain in T2M_KINEMATIC_CHAIN:
        for j_idx in chain[1:]:
            from skeleton_normalizer.canonical_smpl_22 import SMPL_22
            assert SMPL_22[j_idx] in affected, f"missing joint {SMPL_22[j_idx]}"


def test_projection_reduces_partial_stretch_score() -> None:
    """절반 frame 만 stretch → tool 적용 후 BoneLengthEvaluator score 감소."""
    motion = make_rigid_motion(T=40)
    corrupted_half = inject_bone_stretch(motion[:20], chain_label="right_arm", stretch_factor=1.30)
    combined = np.concatenate([corrupted_half, motion[20:]], axis=0)

    evaluator = BoneLengthEvaluator()
    before = evaluator.evaluate(combined)
    right_arm_before = next((r for r in before if r.body_part == "right_arm"), None)
    assert right_arm_before is not None, "expected right_arm bone variation in corrupted motion"
    score_before = right_arm_before.score

    tool = BoneProjectionTool()
    corrected, report = tool.apply(
        combined,
        target_part="right_arm",
        target_joints=[],
        frame_range=(0, 39),
        strength="large",
    )
    after = evaluator.evaluate(corrected)
    right_arm_after = next((r for r in after if r.body_part == "right_arm"), None)
    score_after = right_arm_after.score if right_arm_after else 0.0
    assert score_after < score_before, f"score did not decrease: before={score_before}, after={score_after}"
    assert report.correction_magnitude > 0.0
    assert report.target_part == "right_arm"


def test_invalid_target_part_raises() -> None:
    motion = make_rigid_motion(T=20)
    tool = BoneProjectionTool()
    with pytest.raises(ValueError, match="not in"):
        tool.apply(motion, target_part="unknown_chain", target_joints=[], frame_range=(0, 19))


def test_strength_monotonic() -> None:
    motion = make_rigid_motion(T=40)
    corrupted_half = inject_bone_stretch(motion[:20], chain_label="left_leg", stretch_factor=1.30)
    combined = np.concatenate([corrupted_half, motion[20:]], axis=0)
    tool = BoneProjectionTool()
    mags: list[float] = []
    for strength in ("small", "medium", "large"):
        _, report = tool.apply(combined, target_part="left_leg", target_joints=[], frame_range=(0, 39), strength=strength)
        mags.append(report.correction_magnitude)
    assert mags[0] < mags[1] < mags[2], f"non-monotonic: {mags}"


def test_other_chain_untouched() -> None:
    """target chain 외 chain 은 변경 없음."""
    motion = make_rigid_motion(T=30)
    tool = BoneProjectionTool()
    corrected, _ = tool.apply(
        motion, target_part="right_leg", target_joints=[], frame_range=(0, 29), strength="large"
    )
    # right_arm chain 의 joint 좌표는 그대로
    right_arm_chain = T2M_KINEMATIC_CHAIN[CHAIN_LABELS.index("right_arm")]
    for j_idx in right_arm_chain:
        np.testing.assert_array_equal(corrected[:, j_idx, :], motion[:, j_idx, :])


def test_invalid_shape_raises() -> None:
    tool = BoneProjectionTool()
    with pytest.raises(ValueError, match=r"\[T, 22, 3\]"):
        tool.apply(np.zeros((10, 21, 3)), target_part="right_leg", target_joints=[], frame_range=(0, 9))
