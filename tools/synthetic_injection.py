"""Synthetic artifact injection tool — 명세 §9.3 Protocol A.

본 모듈은 HumanML3D 등 GT motion 에 controlled artifact 를 주입한다. 본
주입 결과는 (a) evaluator 단위 테스트의 positive case 생성, (b) refinement
framework 의 ground-truth-grounded 평가 (corrupted 입력 → refined 와 clean
GT 의 MPJPE 비교) 의 두 용도로 사용된다.

지원 artifact 종류 (Week 2 prototype — 3 종):
  - foot_floating: 좌·우 foot 을 지정 frame 구간에서 위쪽으로 들어올림.
  - bone_stretch: 지정 chain 의 bone 들을 frame-wise scaling 으로 늘이거나 줄임.
  - jitter: 모든 joint 에 frame-wise gaussian noise 추가 (velocity 변화 유발).

모든 함수는 입력 motion 을 변형하지 않고 새 array 를 반환 (in-place 변경 금지).

AGENTS.md §3-1 Canonical Motion Format ([T, 22, 3]) 준수.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from skeleton_normalizer.canonical_smpl_22 import (
    NAME_TO_IDX,
    T2M_KINEMATIC_CHAIN,
    CHAIN_LABELS,
)

FOOT_JOINT_IDX = [NAME_TO_IDX["LEFT_FOOT"], NAME_TO_IDX["RIGHT_FOOT"]]


def inject_foot_floating(
    motion: np.ndarray,
    lift_height: float = 0.10,
    frame_range: Optional[tuple[int, int]] = None,
    seed: int = 42,
) -> np.ndarray:
    """좌·우 foot 을 지정 frame 구간에서 lift_height (m) 만큼 위로 들어올림.

    의미: contact 으로 가정되는 frame 에서 발이 ground 위로 떠 있는 artifact 재현.

    Args:
        motion: [T, 22, 3] canonical motion.
        lift_height: foot 을 들어올릴 높이 (m). 양수면 위로.
        frame_range: 주입할 frame 구간 [start, end). None 이면 motion 전체.
        seed: random seed (현 함수는 결정론적이지만 시그니처 일관성 위해 보존).

    Returns:
        주입된 새 motion array [T, 22, 3].
    """
    _ = np.random.default_rng(seed)
    if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
        raise ValueError(f"motion shape must be [T, 22, 3], got {motion.shape}")
    out = motion.copy()
    T = out.shape[0]
    start, end = frame_range if frame_range is not None else (0, T)
    start = max(0, start)
    end = min(T, end)
    for j in FOOT_JOINT_IDX:
        out[start:end, j, 1] += lift_height
    return out


def inject_bone_stretch(
    motion: np.ndarray,
    chain_label: str = "right_leg",
    stretch_factor: float = 1.25,
    seed: int = 42,
) -> np.ndarray:
    """지정 chain 의 모든 child joint 를 parent 로부터 scale 만큼 늘리거나 줄임.

    의미: bone length variation artifact 재현. stretch_factor=1.0 이면 변화 없음.

    Args:
        motion: [T, 22, 3] canonical motion.
        chain_label: T2M_KINEMATIC_CHAIN 의 chain 라벨 (right_leg / left_leg / spine_head / right_arm / left_arm).
        stretch_factor: bone vector 의 scaling factor. 1.25 = 25% 늘림, 0.8 = 20% 줄임.
        seed: random seed.

    Returns:
        주입된 새 motion array.
    """
    _ = np.random.default_rng(seed)
    if chain_label not in CHAIN_LABELS:
        raise ValueError(
            f"unknown chain_label {chain_label!r}, expected one of {CHAIN_LABELS}"
        )
    chain = T2M_KINEMATIC_CHAIN[CHAIN_LABELS.index(chain_label)]
    if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
        raise ValueError(f"motion shape must be [T, 22, 3], got {motion.shape}")
    out = motion.copy()
    # chain 을 따라 parent → child 순으로 bone vector 를 scale.
    # 자식 joint 의 child 들도 동일 변환을 받게 forward-kinematics 식으로 propagate 한다.
    for parent_idx, child_idx in zip(chain[:-1], chain[1:]):
        bone_vec = out[:, child_idx, :] - out[:, parent_idx, :]
        delta = bone_vec * (stretch_factor - 1.0)
        # child 와 그 이후 chain 의 모든 joint 에 동일 delta 더함 (단순 propagate).
        descendants = chain[chain.index(child_idx):]
        for d in descendants:
            out[:, d, :] += delta
    return out


def inject_jitter(
    motion: np.ndarray,
    noise_std: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """모든 joint 의 모든 frame 에 gaussian noise 추가.

    의미: velocity jitter / acceleration jerk artifact 재현. noise_std 가 클수록 떨림 강함.

    Args:
        motion: [T, 22, 3] canonical motion.
        noise_std: gaussian noise 의 표준편차 (m). 1cm 가 default.
        seed: random seed (결정론적 noise 재현).

    Returns:
        주입된 새 motion array.
    """
    rng = np.random.default_rng(seed)
    if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
        raise ValueError(f"motion shape must be [T, 22, 3], got {motion.shape}")
    noise = rng.normal(loc=0.0, scale=noise_std, size=motion.shape)
    return motion + noise
