"""Unit tests for FootFloatingEvaluator.

phase 03-test §2-2 의 property + reference oracle 패턴 적용.

property: clean (foot on ground) motion → score 0, injected (foot lifted) → score > 0.
reference: 100% frame 에서 foot 을 tau 위로 들어올리면 score == 1.0.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluators import FootFloatingEvaluator
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX
from tools.synthetic_injection import inject_foot_floating

LEFT_FOOT = NAME_TO_IDX["LEFT_FOOT"]
RIGHT_FOOT = NAME_TO_IDX["RIGHT_FOOT"]


def make_standing_motion(T: int = 20, ground_y: float = 0.0) -> np.ndarray:
    """T-pose 비슷한 standing motion. foot 은 ground 에 닿아있음 (y == ground_y)."""
    motion = np.zeros((T, 22, 3), dtype=np.float64)
    # 모든 joint 를 y=1.0 (몸통 가정), foot 만 y=0 으로
    motion[:, :, 1] = 1.0
    motion[:, LEFT_FOOT, 1] = ground_y
    motion[:, RIGHT_FOOT, 1] = ground_y
    return motion


def test_clean_standing_motion_no_floating() -> None:
    """foot 이 ground 에 닿은 motion → floating report 없음 (또는 score 0)."""
    motion = make_standing_motion(T=30, ground_y=0.0)
    evaluator = FootFloatingEvaluator(tau_float=0.05)
    reports = evaluator.evaluate(motion, ground_y=0.0)
    # 모든 report 의 score 가 0 이거나 report 자체가 없어야 함
    assert all(r.score == 0.0 for r in reports)


def test_full_lift_full_score() -> None:
    """foot 을 tau 보다 훨씬 위로 들어올리면 floating score 가 정확히 1.0 (100%) 이 되어야 함."""
    motion = make_standing_motion(T=30, ground_y=0.0)
    # tau=0.05 보다 충분히 큰 0.20 (20cm) 만큼 들어올림. contact heuristic 이
    # contact 로 인식 못 할 수 있어 contact_labels 명시 (foot 은 항상 contact 가정).
    contact = np.zeros((30, 22), dtype=bool)
    contact[:, LEFT_FOOT] = True
    contact[:, RIGHT_FOOT] = True
    corrupted = inject_foot_floating(motion, lift_height=0.20, frame_range=(0, 30))
    evaluator = FootFloatingEvaluator(tau_float=0.05)
    reports = evaluator.evaluate(corrupted, ground_y=0.0, contact_labels=contact)
    # 좌·우 foot 둘 다 floating report 가 있어야 하고 score 1.0
    parts = {r.body_part: r.score for r in reports}
    assert "left_foot" in parts and parts["left_foot"] == 1.0
    assert "right_foot" in parts and parts["right_foot"] == 1.0


def test_partial_frame_range_partial_score() -> None:
    """절반 frame 만 floating → score 약 0.5."""
    motion = make_standing_motion(T=40, ground_y=0.0)
    contact = np.zeros((40, 22), dtype=bool)
    contact[:, LEFT_FOOT] = True
    contact[:, RIGHT_FOOT] = True
    corrupted = inject_foot_floating(motion, lift_height=0.20, frame_range=(0, 20))
    evaluator = FootFloatingEvaluator(tau_float=0.05)
    reports = evaluator.evaluate(corrupted, ground_y=0.0, contact_labels=contact)
    parts = {r.body_part: r.score for r in reports}
    assert "left_foot" in parts and parts["left_foot"] == pytest.approx(0.5, abs=0.01)
    assert "right_foot" in parts and parts["right_foot"] == pytest.approx(0.5, abs=0.01)


def test_invalid_shape_raises() -> None:
    motion = np.zeros((10, 21, 3))  # 21 joint
    evaluator = FootFloatingEvaluator()
    with pytest.raises(ValueError, match=r"\[T, 22, 3\]"):
        evaluator.evaluate(motion)


def test_severity_classification() -> None:
    """score 0.30 이상이면 high severity."""
    motion = make_standing_motion(T=10, ground_y=0.0)
    contact = np.ones((10, 22), dtype=bool)
    corrupted = inject_foot_floating(motion, lift_height=0.20, frame_range=(0, 10))
    evaluator = FootFloatingEvaluator(tau_float=0.05)
    reports = evaluator.evaluate(corrupted, ground_y=0.0, contact_labels=contact)
    for r in reports:
        assert r.severity == "high"
        assert r.recommendation == "foot_lock_tool"
