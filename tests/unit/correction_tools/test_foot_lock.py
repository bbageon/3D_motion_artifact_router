"""Unit tests for FootLockTool.

property: floating-injected motion 에 tool 적용 → FootFloatingEvaluator score 감소.
reference: strength='large' 면 floating score 가 거의 0 으로 떨어져야 함.
property: KDG affected joints 는 LEFT_FOOT / RIGHT_FOOT 포함.
"""
from __future__ import annotations

import numpy as np
import pytest

from correction_tools import FootLockTool
from evaluators import FootFloatingEvaluator
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX
from tools.synthetic_injection import inject_foot_floating

LEFT_FOOT = NAME_TO_IDX["LEFT_FOOT"]
RIGHT_FOOT = NAME_TO_IDX["RIGHT_FOOT"]


def make_standing_motion(T: int = 30, ground_y: float = 0.0) -> np.ndarray:
    motion = np.zeros((T, 22, 3), dtype=np.float64)
    motion[:, :, 1] = 1.0
    motion[:, LEFT_FOOT, 1] = ground_y
    motion[:, RIGHT_FOOT, 1] = ground_y
    return motion


def test_kdg_affected_joints() -> None:
    tool = FootLockTool()
    joints = tool.kdg_affected_joints()
    assert "LEFT_FOOT" in joints and "RIGHT_FOOT" in joints
    weights = tool.kdg_propagation_weights()
    assert "LEFT_ANKLE" in weights and "RIGHT_ANKLE" in weights


def test_lock_reduces_floating_score_large_strength() -> None:
    motion = make_standing_motion(T=30)
    contact = np.zeros((30, 22), dtype=bool)
    contact[:, LEFT_FOOT] = True
    contact[:, RIGHT_FOOT] = True
    corrupted = inject_foot_floating(motion, lift_height=0.20, frame_range=(0, 30))

    evaluator = FootFloatingEvaluator(tau_float=0.05)
    before = evaluator.evaluate(corrupted, ground_y=0.0, contact_labels=contact)
    score_before = sum(r.score for r in before)
    assert score_before > 0.0  # corruption 이 detection 되어야 함

    tool = FootLockTool(default_ground_y=0.0)
    corrected, report = tool.apply(
        corrupted,
        target_part="both_feet",
        target_joints=[],
        frame_range=(0, 29),
        strength="large",
    )
    after = evaluator.evaluate(corrected, ground_y=0.0, contact_labels=contact)
    score_after = sum(r.score for r in after)
    assert score_after < score_before, f"score did not decrease: before={score_before}, after={score_after}"
    # large strength + 일률 floating → 거의 0 으로 떨어져야 함
    assert score_after < 0.1, f"score_after={score_after} unexpectedly high"
    assert report.correction_magnitude > 0.0
    assert "LEFT_FOOT" in report.modified_joints or "RIGHT_FOOT" in report.modified_joints


def test_lock_strength_monotonic() -> None:
    """small < medium < large strength → correction_magnitude 단조 증가."""
    motion = make_standing_motion(T=20)
    contact = np.zeros((20, 22), dtype=bool)
    contact[:, LEFT_FOOT] = True
    contact[:, RIGHT_FOOT] = True
    corrupted = inject_foot_floating(motion, lift_height=0.20, frame_range=(0, 20))
    tool = FootLockTool(default_ground_y=0.0)
    mags: list[float] = []
    for strength in ("small", "medium", "large"):
        _, report = tool.apply(
            corrupted,
            target_part="both_feet",
            target_joints=[],
            frame_range=(0, 19),
            strength=strength,
        )
        mags.append(report.correction_magnitude)
    assert mags[0] < mags[1] < mags[2], f"non-monotonic correction_magnitude: {mags}"


def test_lock_outside_frame_range_untouched() -> None:
    """frame_range 밖의 foot 은 그대로 유지."""
    motion = make_standing_motion(T=40)
    contact = np.zeros((40, 22), dtype=bool)
    contact[:, LEFT_FOOT] = True
    contact[:, RIGHT_FOOT] = True
    corrupted = inject_foot_floating(motion, lift_height=0.20, frame_range=(0, 40))

    tool = FootLockTool(default_ground_y=0.0)
    corrected, _ = tool.apply(
        corrupted,
        target_part="both_feet",
        target_joints=[],
        frame_range=(0, 19),  # 절반만 lock
        strength="large",
    )
    # frame 0~19: corrected y 거의 0
    assert np.max(np.abs(corrected[:20, LEFT_FOOT, 1])) < 0.05
    # frame 20~39: 원본 corrupted 그대로 (y == 0.20)
    assert np.all(np.isclose(corrected[20:, LEFT_FOOT, 1], 0.20))


def test_invalid_shape_raises() -> None:
    tool = FootLockTool()
    with pytest.raises(ValueError, match=r"\[T, 22, 3\]"):
        tool.apply(
            np.zeros((10, 21, 3)),
            target_part="both_feet",
            target_joints=[],
            frame_range=(0, 9),
        )
