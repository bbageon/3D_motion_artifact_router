"""Unit tests for CoordinateFootSkateCleanupTool (AR-061, frozen spec).

property: skate 주입 motion 에 tool 적용 → v2 skate frame 감소 + 발 XZ 분산 감소.
property: 깨끗한 접지(slide 없음) → no-op (n_correct=0, motion 불변).
property: ambiguous (0 < skate_frac < min_skate_frac) → no-op + ambiguous report.
property: continuous_u 단조 증가 → correction_magnitude 단조 증가.
property: skate 없는 발(반대쪽)은 미수정 (segment 별 자동 선택).
reference: coord_space != "trajectory" → ValueError / invalid shape → ValueError.
frame_range: window 밖 skate 는 미보정.
"""
from __future__ import annotations

import numpy as np
import pytest

from correction_tools import CoordinateFootSkateCleanupTool
from correction_tools.coordinate_footskate_cleanup_tool import _v2_flags
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX

LEFT_FOOT = NAME_TO_IDX["LEFT_FOOT"]
RIGHT_FOOT = NAME_TO_IDX["RIGHT_FOOT"]


def make_skating_motion(T: int = 40, slide_per_frame: float = 0.04,
                        slide_range: tuple[int, int] = (10, 25)) -> np.ndarray:
    """LEFT_FOOT 이 접지(y=0) 상태로 slide_range 동안 X 로 미끄러짐. RIGHT_FOOT 은 정지 접지."""
    m = np.zeros((T, 22, 3), dtype=np.float64)
    m[:, :, 1] = 1.0  # body
    for f in (LEFT_FOOT, RIGHT_FOOT):
        m[:, f, 1] = 0.0
    m[:, NAME_TO_IDX["LEFT_ANKLE"], 1] = 0.1
    m[:, NAME_TO_IDX["RIGHT_ANKLE"], 1] = 0.1
    s, e = slide_range
    x = np.zeros(T)
    x[s:e + 1] = (np.arange(s, e + 1) - s) * slide_per_frame
    x[e + 1:] = x[e]
    m[:, LEFT_FOOT, 0] = x
    return m


def make_clean_motion(T: int = 40) -> np.ndarray:
    """양 발 정지 접지 (skate 없음) — 건드리면 안 되는 정상 접지."""
    m = np.zeros((T, 22, 3), dtype=np.float64)
    m[:, :, 1] = 1.0
    for f in (LEFT_FOOT, RIGHT_FOOT):
        m[:, f, 1] = 0.0
    return m


def test_kdg_affected_joints() -> None:
    tool = CoordinateFootSkateCleanupTool()
    joints = tool.kdg_affected_joints()
    assert "LEFT_FOOT" in joints and "RIGHT_FOOT" in joints
    weights = tool.kdg_propagation_weights()
    for j in ("LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_HIP", "RIGHT_HIP"):
        assert j in weights


def test_reduces_v2_skate_and_xz_spread() -> None:
    """property: 보정 후 v2 skate frame 수와 skate 구간 XZ 표준편차가 감소."""
    motion = make_skating_motion()
    T = motion.shape[0]
    _, skate_before = _v2_flags(motion, LEFT_FOOT, 0.0, 0.05, 0.035, 0.025)
    assert skate_before.sum() >= 10  # 주입 확인

    tool = CoordinateFootSkateCleanupTool()
    corrected, report = tool.apply(
        motion, target_part="both_feet", target_joints=[], frame_range=(0, T - 1),
        strength="xlarge",  # u=1.0
    )
    _, skate_after = _v2_flags(corrected, LEFT_FOOT, 0.0, 0.05, 0.035, 0.025)
    assert skate_after.sum() < skate_before.sum(), (
        f"skate did not decrease: before={int(skate_before.sum())} after={int(skate_after.sum())}"
    )
    # 원래 slide 구간의 X 분산 감소 (anchor 로 수렴).
    std_before = float(np.std(motion[10:26, LEFT_FOOT, 0]))
    std_after = float(np.std(corrected[10:26, LEFT_FOOT, 0]))
    assert std_after < std_before
    assert report.correction_magnitude > 0.0
    assert report.metadata["n_correct"] >= 1
    assert "LEFT_FOOT" in report.modified_joints


def test_clean_contact_is_noop() -> None:
    """property: skate 신호 없는 정상 접지는 건드리지 않음 (frozen spec §3-1)."""
    motion = make_clean_motion()
    tool = CoordinateFootSkateCleanupTool()
    corrected, report = tool.apply(
        motion, target_part="both_feet", target_joints=[], frame_range=(0, 39),
        strength="xlarge",
    )
    assert np.allclose(corrected, motion)
    assert report.metadata["n_correct"] == 0
    assert report.correction_magnitude == 0.0


def test_ambiguous_segment_reported_not_corrected() -> None:
    """property: 0 < skate_frac < min_skate_frac → 보정 없이 ambiguous report."""
    motion = make_clean_motion(T=60)
    # 단일 skate frame (1/60 ≈ 0.017 < 0.2): frame 30 에서 0.03m 수평 점프.
    motion[31:, LEFT_FOOT, 0] = 0.03
    tool = CoordinateFootSkateCleanupTool()
    corrected, report = tool.apply(
        motion, target_part="both_feet", target_joints=[], frame_range=(0, 59),
        strength="xlarge",
    )
    assert np.allclose(corrected, motion)
    assert report.metadata["n_correct"] == 0
    assert report.metadata["n_ambiguous"] >= 1


def test_u_monotonic_correction_magnitude() -> None:
    """property: continuous_u 증가 → correction_magnitude 단조 증가 (u_grid)."""
    motion = make_skating_motion()
    tool = CoordinateFootSkateCleanupTool()
    mags = []
    for u in (0.25, 0.5, 0.75, 1.0):
        _, report = tool.apply(
            motion, target_part="both_feet", target_joints=[], frame_range=(0, 39),
            strength="medium", metadata={"continuous_u": u},
        )
        assert report.metadata["u"] == u and report.metadata["u_source"] == "continuous"
        mags.append(report.correction_magnitude)
    assert mags[0] < mags[1] < mags[2] < mags[3], f"non-monotonic: {mags}"


def test_non_skating_foot_untouched() -> None:
    """property: skate 없는 반대쪽 발은 미수정 (segment 별 자동 선택)."""
    motion = make_skating_motion()
    tool = CoordinateFootSkateCleanupTool()
    corrected, _ = tool.apply(
        motion, target_part="both_feet", target_joints=[], frame_range=(0, 39),
        strength="xlarge",
    )
    assert np.allclose(corrected[:, RIGHT_FOOT, :], motion[:, RIGHT_FOOT, :])
    # RIGHT 다리 chain 도 미수정 (propagation 은 보정된 발에만).
    assert np.allclose(corrected[:, NAME_TO_IDX["RIGHT_KNEE"], :], motion[:, NAME_TO_IDX["RIGHT_KNEE"], :])


def test_frame_range_window_respected() -> None:
    """frame_range 밖 skate segment (clip 후 skate_frac=0) → 미보정."""
    motion = make_skating_motion()  # slide 10..25
    tool = CoordinateFootSkateCleanupTool()
    corrected, report = tool.apply(
        motion, target_part="both_feet", target_joints=[], frame_range=(0, 5),
        strength="xlarge",
    )
    assert np.allclose(corrected, motion)
    assert report.metadata["n_correct"] == 0


def test_local_coord_space_raises() -> None:
    """reference: motion_local 입력 차단 (frozen spec §1-2)."""
    motion = make_skating_motion()
    tool = CoordinateFootSkateCleanupTool()
    with pytest.raises(ValueError, match="trajectory"):
        tool.apply(motion, target_part="both_feet", target_joints=[], frame_range=(0, 39),
                   metadata={"coord_space": "local"})


def test_invalid_shape_raises() -> None:
    tool = CoordinateFootSkateCleanupTool()
    with pytest.raises(ValueError, match=r"\[T, 22, 3\]"):
        tool.apply(np.zeros((10, 21, 3)), target_part="both_feet", target_joints=[],
                   frame_range=(0, 9))


def test_leg_chain_propagation_weights_applied() -> None:
    """property: 보정 시 ankle/knee/hip 이 1.0/0.5/0.2/0.1 비율로 따라감."""
    motion = make_skating_motion()
    tool = CoordinateFootSkateCleanupTool(blend_frames=1)
    corrected, _ = tool.apply(
        motion, target_part="left_foot", target_joints=[], frame_range=(0, 39),
        strength="xlarge",
    )
    d_foot = corrected[:, LEFT_FOOT, :] - motion[:, LEFT_FOOT, :]
    d_ankle = corrected[:, NAME_TO_IDX["LEFT_ANKLE"], :] - motion[:, NAME_TO_IDX["LEFT_ANKLE"], :]
    d_knee = corrected[:, NAME_TO_IDX["LEFT_KNEE"], :] - motion[:, NAME_TO_IDX["LEFT_KNEE"], :]
    d_hip = corrected[:, NAME_TO_IDX["LEFT_HIP"], :] - motion[:, NAME_TO_IDX["LEFT_HIP"], :]
    assert np.allclose(d_ankle, 0.5 * d_foot, atol=1e-9)
    assert np.allclose(d_knee, 0.2 * d_foot, atol=1e-9)
    assert np.allclose(d_hip, 0.1 * d_foot, atol=1e-9)
