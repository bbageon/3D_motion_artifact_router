"""Unit tests for PhysicalGateV0 v0.2.0 수리 (AR-063 — AR-062 audit A-1/A-2/A-4 fix).

reference: AR-062 vacuity 재현 case 가 수리 후 **fire** 하는지 (내부 추정만으로).
property: clean motion 은 여전히 fire 안 함 (false-positive 회피 유지).
regression(A-4): 접지한 채 수평 이동+살짝 뜬 발 (보행 stance 근사) 을 Float 이 잡는지
  — 기존 수평속도 contact 은 이동 중 발을 contact 에서 제외해 구조적으로 놓쳤음.
"""
from __future__ import annotations

import numpy as np

from evaluators import (
    FloatEvaluator,
    FootFloatingEvaluator,
    PenetrateEvaluator,
    SkateEvaluator,
)

LEFT_FOOT, RIGHT_FOOT = 10, 11


def make_skate_motion(T: int = 30, slide: float = 0.06) -> np.ndarray:
    """AR-062 재현 case: 두 발 접지(y=0) + slide m/frame 수평 이동 (전 frame skate)."""
    m = np.zeros((T, 22, 3))
    m[:, :, 1] = 1.0
    for f in (LEFT_FOOT, RIGHT_FOOT):
        m[:, f, 1] = 0.0
        m[:, f, 0] = np.arange(T) * slide
    m[:, 7, 1] = 0.1
    m[:, 8, 1] = 0.1
    return m


def make_penetration_motion(T: int = 30, depth: float = 0.08) -> np.ndarray:
    """AR-062 재현 case (transient 화): RIGHT_FOOT 이 frame 10~11 에서 -depth 관통.

    feet-percentile ground(10th pct) 는 **transient** 관통(<10% frames)에 robust 하도록
    설계됨 — 관통 비율이 percentile 을 넘으면 ground 가 따라 내려가 검출 불가 (문서화된
    한계; 전 구간 offset 은 외부 ground_y 필요). 따라서 2/30 frames (6.7%) 로 구성.
    """
    m = np.zeros((T, 22, 3))
    m[:, :, 1] = 1.0
    m[:, LEFT_FOOT, 1] = 0.0
    m[:, RIGHT_FOOT, 1] = 0.0
    m[10:12, RIGHT_FOOT, 1] = -depth
    return m


def make_clean_standing(T: int = 30) -> np.ndarray:
    m = np.zeros((T, 22, 3))
    m[:, :, 1] = 1.0
    m[:, LEFT_FOOT, 1] = 0.0
    m[:, RIGHT_FOOT, 1] = 0.0
    return m


def test_skate_fires_with_internal_contact_after_fix() -> None:
    """A-1 fix: 내부 contact 추정만으로 명백한 skate 를 검출 (기존 0 report → 수리 후 fire)."""
    reports = SkateEvaluator().evaluate(make_skate_motion())
    assert len(reports) == 2, f"expected 2 foot reports, got {len(reports)}"
    for r in reports:
        assert r.score > 0.9  # 전 frame skate
        assert r.metadata["severity_version"].startswith("0.2.0")


def test_skate_zero_on_clean_standing() -> None:
    reports = SkateEvaluator().evaluate(make_clean_standing())
    assert reports == []


def test_penetrate_fires_with_internal_ground_after_fix() -> None:
    """A-2 fix: 내부 ground(feet 10th pct)만으로 관통 검출 (기존 min-Y 는 정의상 불가)."""
    reports = PenetrateEvaluator().evaluate(make_penetration_motion())
    parts = {r.body_part for r in reports}
    assert "right_foot" in parts
    rf = next(r for r in reports if r.body_part == "right_foot")
    assert 0.03 <= rf.score <= 0.15  # 2/30 frames ≈ 0.067


def test_penetrate_zero_on_clean_standing() -> None:
    reports = PenetrateEvaluator().evaluate(make_clean_standing())
    assert reports == []


def test_float_catches_moving_low_float_a4_regression() -> None:
    """A-4 fix: 수평 이동 중이지만 낮게(0.07m) 떠서 수직 정지한 발 → Float fire.

    기존 수평속도 contact(≤0.02) 은 이동(0.05/frame) 중 발을 contact 에서 제외해
    본 case 를 구조적으로 놓쳤다 (local 좌표 보행 stance 근사).
    """
    T = 30
    m = make_clean_standing(T)
    m[:, RIGHT_FOOT, 1] = 0.07                     # float band (0.05 < h ≤ 0.10)
    m[:, RIGHT_FOOT, 0] = np.arange(T) * 0.05      # 수평 이동 (pelvis-상대 stance 근사)
    reports = FloatEvaluator().evaluate(m, ground_y=0.0)
    parts = {r.body_part for r in reports}
    assert "right_foot" in parts, "moving low-float foot 를 Float 이 놓침 (A-4 regression)"


def test_foot_floating_v2_contact_moving_float() -> None:
    """FootFloating v2.0.0: 같은 A-4 case 를 Layer-A evaluator 도 검출."""
    T = 30
    m = make_clean_standing(T)
    m[:, RIGHT_FOOT, 1] = 0.07
    m[:, RIGHT_FOOT, 0] = np.arange(T) * 0.05
    reports = FootFloatingEvaluator().evaluate(m, ground_y=0.0)
    parts = {r.body_part for r in reports}
    assert "right_foot" in parts
    rf = next(r for r in reports if r.body_part == "right_foot")
    assert rf.metadata["severity_version"].startswith("2.0.0")
    assert rf.metadata["contact_heuristic"] == "vertical_velocity_based_v2.0.0"


def test_foot_floating_intentional_raised_foot_still_excluded() -> None:
    """v1.2.0 의 raised-foot false-positive 회피가 유지되는지 (height > 0.10 은 contact 아님)."""
    T = 30
    m = make_clean_standing(T)
    m[:, RIGHT_FOOT, 1] = 0.30  # 의도적으로 들어올린 발 (정지)
    reports = FootFloatingEvaluator().evaluate(m, ground_y=0.0)
    assert all(r.body_part != "right_foot" for r in reports)
