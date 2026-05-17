"""Unit tests for VelocityJitterEvaluator.

property: 정지 motion (constant pose) → acceleration 0 → score 0.
property: linear-velocity motion (uniform velocity) → acceleration 0 → score 0.
reference: gaussian jitter noise_std → expected accel magnitude 비례.
property: jitter 가 클수록 score 가 monotonic 증가.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluators import VelocityJitterEvaluator
from tools.synthetic_injection import inject_jitter


def make_constant_motion(T: int = 20, seed: int = 42) -> np.ndarray:
    """모든 frame 에서 동일 pose (velocity 0, acceleration 0)."""
    rng = np.random.default_rng(seed)
    rest_pose = rng.uniform(low=-0.5, high=0.5, size=(22, 3)).astype(np.float64)
    return np.tile(rest_pose[None, :, :], (T, 1, 1))


def make_linear_velocity_motion(T: int = 20, seed: int = 42) -> np.ndarray:
    """모든 joint 가 일정 velocity 로 직선 이동 (acceleration 0)."""
    rng = np.random.default_rng(seed)
    rest_pose = rng.uniform(low=-0.5, high=0.5, size=(22, 3)).astype(np.float64)
    velocity = rng.uniform(low=-0.05, high=0.05, size=(22, 3))
    motion = np.zeros((T, 22, 3), dtype=np.float64)
    for t in range(T):
        motion[t] = rest_pose + t * velocity
    return motion


def test_constant_motion_zero_jitter() -> None:
    """움직임 없음 → acceleration 0 → report 없음."""
    motion = make_constant_motion(T=20)
    evaluator = VelocityJitterEvaluator()
    reports = evaluator.evaluate(motion)
    # accel magnitude 가 0 이므로 SEV_LOW * 0.5 (= 0.0025) 미만 → 모두 생략
    assert reports == []


def test_linear_velocity_zero_jitter() -> None:
    """일정 velocity 이동 → acceleration 0 → report 없음."""
    motion = make_linear_velocity_motion(T=20)
    evaluator = VelocityJitterEvaluator()
    reports = evaluator.evaluate(motion)
    assert reports == []


def test_jitter_noise_detected() -> None:
    """gaussian jitter 추가 → global + part-wise jitter 모두 detection."""
    motion = make_constant_motion(T=30)
    corrupted = inject_jitter(motion, noise_std=0.05, seed=42)
    evaluator = VelocityJitterEvaluator()
    reports = evaluator.evaluate(corrupted)
    assert len(reports) >= 1
    global_reports = [r for r in reports if r.error_type == "global_velocity_jitter"]
    assert len(global_reports) == 1
    # noise_std=0.05 의 acceleration L2 norm 기대값 — 약 noise_std * sqrt(3) * 2
    # (frame 간 noise difference 의 분산이 두 배). 보수적으로 score > 0.05 검증.
    assert global_reports[0].score > 0.05, f"expected jitter score > 0.05, got {global_reports[0].score}"


def test_jitter_monotonic_in_noise_std() -> None:
    """noise_std 가 클수록 score 가 단조 증가해야 함."""
    motion = make_constant_motion(T=30)
    evaluator = VelocityJitterEvaluator()
    scores: list[float] = []
    for noise_std in (0.01, 0.05, 0.10):
        corrupted = inject_jitter(motion, noise_std=noise_std, seed=42)
        reports = evaluator.evaluate(corrupted)
        global_r = next(r for r in reports if r.error_type == "global_velocity_jitter")
        scores.append(global_r.score)
    assert scores[0] < scores[1] < scores[2], f"non-monotonic: {scores}"


def test_part_wise_reports() -> None:
    motion = make_constant_motion(T=30)
    corrupted = inject_jitter(motion, noise_std=0.05, seed=42)
    evaluator = VelocityJitterEvaluator()
    reports = evaluator.evaluate(corrupted)
    body_parts = {r.body_part for r in reports}
    assert "full_body" in body_parts  # global report
    assert "legs" in body_parts
    assert "arms" in body_parts
    assert "spine_head" in body_parts
    for r in reports:
        assert r.recommendation == "velocity_smoothing_tool"


def test_invalid_shape_raises() -> None:
    motion = np.zeros((10, 21, 3))
    evaluator = VelocityJitterEvaluator()
    with pytest.raises(ValueError, match=r"\[T, 22, 3\]"):
        evaluator.evaluate(motion)


def test_too_short_motion_returns_empty() -> None:
    """T < 3 이면 acceleration 계산 불가 → 빈 list."""
    motion = np.zeros((2, 22, 3))
    evaluator = VelocityJitterEvaluator()
    assert evaluator.evaluate(motion) == []
