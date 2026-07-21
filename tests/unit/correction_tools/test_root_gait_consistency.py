"""Unit tests for RootGaitConsistencyTool (AR-072, 사전등록 설계).

property: root-deficit 합성 모션(다리는 걷는데 root 전진 부족) → 보정 후
  (a) 접지발의 world 미끄러짐 감소, (b) root 경로 길이 증가 (부족 회복).
property: 불변성 — local(root-relative)·bone 길이·y 높이 완전 불변 (구조 보장).
property: 정상 모션(접지발 world 고정) → 보정 ≈ no-op.
reference: coord_space != trajectory → ValueError / invalid shape → ValueError.
"""
from __future__ import annotations

import numpy as np
import pytest

from correction_tools import RootGaitConsistencyTool
from correction_tools.coordinate_footskate_cleanup_tool import _v2_flags
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX

PELVIS = NAME_TO_IDX["PELVIS"]
LF, RF = NAME_TO_IDX["LEFT_FOOT"], NAME_TO_IDX["RIGHT_FOOT"]


def make_deficit_walk(T: int = 60, gait_v: float = 0.04, root_v: float = 0.015) -> np.ndarray:
    """다리는 gait_v 로 걷는 pose 인데 root 는 root_v 로만 전진하는 합성 (root deficit).

    좌/우 발이 15-frame 씩 번갈아 접지: 접지발은 골반 기준 뒤로 gait_v 로 흐르고
    (world 로는 root_v−gait_v < 0 → 미끄러짐), 반대발은 앞으로 스윙.
    """
    m = np.zeros((T, 22, 3))
    m[:, :, 1] = 1.0
    m[:, PELVIS, 1] = 0.9
    root_x = np.arange(T) * root_v
    m[:, :, 0] += root_x[:, None]          # 모든 joint 가 root 를 따라 전진
    stance_len = 15
    rel = {LF: 0.15, RF: -0.15}            # 골반 기준 발 x 초기 오프셋
    for t in range(T):
        phase, k = divmod(t, stance_len)
        stance = LF if phase % 2 == 0 else RF
        swing = RF if stance == LF else LF
        # 접지발: 골반 기준 뒤로 흐름 (world 고정이려면 root 가 gait_v 로 가야 하는데 root_v 만 감)
        rel[stance] -= (gait_v - 0.0)
        rel[swing] += gait_v * 1.0          # 스윙발: 앞으로 복귀
        m[t, stance, 0] = root_x[t] + rel[stance] + 0.15
        m[t, swing, 0] = root_x[t] + rel[swing] - 0.15
        m[t, stance, 1] = 0.0               # 접지 (y=0, 수직 정지)
        m[t, swing, 1] = 0.08               # 스윙 (살짝 듦)
    return m


def _stance_world_slide(motion: np.ndarray) -> float:
    """접지 frame 의 발 world 수평 이동량 평균 (미끄러짐 proxy)."""
    total, n = 0.0, 0
    for f in (LF, RF):
        contact, _ = _v2_flags(motion, f, 0.0, 0.05, 0.035, 1e9)
        fx = motion[:, f, 0]
        d = np.abs(np.diff(fx))
        m = contact[:-1] & contact[1:]
        total += float(d[m].sum()); n += int(m.sum())
    return total / max(n, 1)


def test_reduces_stance_slide_and_recovers_path() -> None:
    motion = make_deficit_walk()
    tool = RootGaitConsistencyTool()
    corrected, rep = tool.apply(motion, target_part="root", target_joints=[],
                                frame_range=(0, motion.shape[0] - 1), strength="large",
                                metadata={"ground_y": 0.0})
    before = _stance_world_slide(motion)
    after = _stance_world_slide(corrected)
    assert before > 0.01                        # deficit 이 실제로 미끄러짐을 만들었는지
    assert after < before * 0.5, f"slide not reduced: {before:.4f}->{after:.4f}"
    # root 경로 길이가 늘어야 함 (부족 회복 방향).
    assert rep.metadata["path_len_after"] > rep.metadata["path_len_before"]
    assert rep.metadata["constrained_frame_frac"] > 0.5


def test_local_pose_and_bone_invariant() -> None:
    """구조적 불변성: local(root-relative)·bone 길이·y 완전 보존."""
    motion = make_deficit_walk()
    tool = RootGaitConsistencyTool()
    corrected, _ = tool.apply(motion, target_part="root", target_joints=[],
                              frame_range=(0, motion.shape[0] - 1), strength="large",
                              metadata={"ground_y": 0.0})
    local_b = motion - motion[:, PELVIS:PELVIS + 1, :]
    local_a = corrected - corrected[:, PELVIS:PELVIS + 1, :]
    assert np.allclose(local_a, local_b, atol=1e-9)
    assert np.allclose(corrected[:, :, 1], motion[:, :, 1], atol=1e-12)  # y 불변


def test_consistent_walk_is_near_noop() -> None:
    """접지발이 world 에 이미 고정인 정상 보행 → 보정 ≈ no-op."""
    motion = make_deficit_walk(gait_v=0.04, root_v=0.04)  # root 가 gait 를 정확히 따라감
    tool = RootGaitConsistencyTool()
    corrected, rep = tool.apply(motion, target_part="root", target_joints=[],
                                frame_range=(0, motion.shape[0] - 1), strength="large",
                                metadata={"ground_y": 0.0})
    assert rep.metadata["max_offset"] < 0.05, f"unexpected large offset {rep.metadata['max_offset']}"


def test_u_linear_in_displacement_including_over_correction() -> None:
    """AR-085: offset(u) = u·offset(u=1) 가 u∈[0, U_MAX] 전 범위에서 성립 (u>1 포함).

    strength 정의의 근거 — u 는 접지-일관 root 변위 보정의 '적용 분율' (선형).
    """
    from correction_tools.root_gait_consistency_tool import U_MAX_CONTINUOUS
    motion = make_deficit_walk()
    tool = RootGaitConsistencyTool()
    T = motion.shape[0]
    base, _ = tool.apply(motion, "root", [], (0, T - 1),
                         metadata={"coord_space": "trajectory", "continuous_u": 1.0})
    off1 = (base[:, PELVIS, :] - motion[:, PELVIS, :])[:, [0, 2]]
    for u in (0.0, 0.5, 1.5, U_MAX_CONTINUOUS):
        corr, _ = tool.apply(motion, "root", [], (0, T - 1),
                             metadata={"coord_space": "trajectory", "continuous_u": u})
        off_u = (corr[:, PELVIS, :] - motion[:, PELVIS, :])[:, [0, 2]]
        assert np.allclose(off_u, u * off1, atol=1e-9), f"offset(u={u}) != u·offset(1)"
    # 상한 clip: U_MAX 초과 요청은 U_MAX 로 saturate.
    over, rep = tool.apply(motion, "root", [], (0, T - 1),
                           metadata={"coord_space": "trajectory", "continuous_u": U_MAX_CONTINUOUS + 5})
    assert rep.metadata["u"] == U_MAX_CONTINUOUS


def test_local_coord_space_raises() -> None:
    motion = make_deficit_walk()
    tool = RootGaitConsistencyTool()
    with pytest.raises(ValueError, match="trajectory"):
        tool.apply(motion, target_part="root", target_joints=[], frame_range=(0, 59),
                   metadata={"coord_space": "local"})


def test_invalid_shape_raises() -> None:
    tool = RootGaitConsistencyTool()
    with pytest.raises(ValueError, match=r"\[T, 22, 3\]"):
        tool.apply(np.zeros((10, 21, 3)), target_part="root", target_joints=[],
                   frame_range=(0, 9))


def test_kdg_root_node() -> None:
    tool = RootGaitConsistencyTool()
    assert tool.kdg_affected_joints() == ["PELVIS"]
    assert all(w == 1.0 for w in tool.kdg_propagation_weights().values())
