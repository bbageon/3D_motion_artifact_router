"""Unit tests for VelocitySmoothingTool.

property: jitter-injected motion 에 tool 적용 → VelocityJitterEvaluator score 감소.
property: 더 큰 strength (sigma) → 더 큰 correction_magnitude.
property: target_part 별 smoothing 이 다른 part 의 좌표를 건드리지 않음.
"""
from __future__ import annotations

import numpy as np
import pytest

from correction_tools import VelocitySmoothingTool
from evaluators import VelocityJitterEvaluator
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX
from tools.synthetic_injection import inject_jitter


def make_constant_motion(T: int = 30, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rest_pose = rng.uniform(low=-0.5, high=0.5, size=(22, 3)).astype(np.float64)
    return np.tile(rest_pose[None, :, :], (T, 1, 1))


def test_smoothing_reduces_global_jitter_score() -> None:
    motion = make_constant_motion(T=30)
    corrupted = inject_jitter(motion, noise_std=0.05, seed=42)

    evaluator = VelocityJitterEvaluator()
    before = evaluator.evaluate(corrupted)
    score_before_global = next(r.score for r in before if r.error_type == "global_velocity_jitter")
    assert score_before_global > 0.05

    tool = VelocitySmoothingTool()
    corrected, report = tool.apply(
        corrupted,
        target_part="full_body",
        target_joints=[],
        frame_range=(0, 29),
        strength="large",
    )
    after = evaluator.evaluate(corrected)
    score_after_global = next(
        (r.score for r in after if r.error_type == "global_velocity_jitter"), 0.0
    )
    assert score_after_global < score_before_global, (
        f"score did not decrease: before={score_before_global}, after={score_after_global}"
    )
    assert report.correction_magnitude > 0.0


def test_smoothing_strength_monotonic_correction_magnitude() -> None:
    motion = make_constant_motion(T=30)
    corrupted = inject_jitter(motion, noise_std=0.05, seed=42)
    tool = VelocitySmoothingTool()
    mags: list[float] = []
    for strength in ("small", "medium", "large"):
        _, report = tool.apply(
            corrupted,
            target_part="full_body",
            target_joints=[],
            frame_range=(0, 29),
            strength=strength,
        )
        mags.append(report.correction_magnitude)
    assert mags[0] < mags[1] < mags[2], f"non-monotonic: {mags}"


def test_legs_only_does_not_touch_arms() -> None:
    motion = make_constant_motion(T=30)
    corrupted = inject_jitter(motion, noise_std=0.05, seed=42)
    tool = VelocitySmoothingTool()
    corrected, _ = tool.apply(
        corrupted,
        target_part="legs",
        target_joints=[],
        frame_range=(0, 29),
        strength="large",
    )
    arm_indices = [
        NAME_TO_IDX["LEFT_COLLAR"], NAME_TO_IDX["LEFT_SHOULDER"], NAME_TO_IDX["LEFT_ELBOW"],
        NAME_TO_IDX["LEFT_WRIST"], NAME_TO_IDX["RIGHT_COLLAR"], NAME_TO_IDX["RIGHT_SHOULDER"],
        NAME_TO_IDX["RIGHT_ELBOW"], NAME_TO_IDX["RIGHT_WRIST"],
    ]
    np.testing.assert_array_equal(
        corrected[:, arm_indices, :], corrupted[:, arm_indices, :]
    )


def test_invalid_shape_raises() -> None:
    tool = VelocitySmoothingTool()
    with pytest.raises(ValueError, match=r"\[T, 22, 3\]"):
        tool.apply(np.zeros((10, 21, 3)), target_part="full_body", target_joints=[], frame_range=(0, 9))


def test_too_short_motion_returns_no_op() -> None:
    """T < 2 → smoothing skip, correction_magnitude == 0."""
    motion = np.zeros((1, 22, 3))
    tool = VelocitySmoothingTool()
    corrected, report = tool.apply(motion, target_part="full_body", target_joints=[], frame_range=(0, 0))
    np.testing.assert_array_equal(corrected, motion)
    assert report.correction_magnitude == 0.0
