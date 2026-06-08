"""AR-048: coordinate-representation protocol helpers (trajectory <-> local, ground).

표현 통일 원칙 (AR-048 spec B):
  - motion_trajectory: root(PELVIS) 이동을 유지한 표현 (generator 복원 결과, root 제거 전).
  - motion_local: motion_trajectory 에서 frame별 PELVIS 를 뺀 canonical root-relative.
  - foot skate/contact/root path/ground = trajectory 에서, local pose/bone = local 에서 측정.
  - ground 는 **min-Y 로 정의하지 않는다** (단일 최저점 = penetration outlier 에 민감).

canonical SMPL-22 joint index (skeleton_normalizer.canonical_smpl_22 일치).
"""
from __future__ import annotations

import numpy as np

PELVIS = 0
LEFT_FOOT, RIGHT_FOOT = 10, 11
LEFT_ANKLE, RIGHT_ANKLE = 7, 8
FOOT_JOINTS = (LEFT_FOOT, RIGHT_FOOT)

GROUND_PERCENTILE = 10.0  # lower-foot-height 의 하위 %; outlier(penetration) 무시한 robust floor.


def derive_local(motion_trajectory: np.ndarray) -> np.ndarray:
    """motion_local = trajectory - frame별 PELVIS. canonical root-relative."""
    m = np.asarray(motion_trajectory, dtype=np.float64)
    if m.ndim != 3 or m.shape[1] != 22 or m.shape[2] != 3:
        raise ValueError(f"expected [T,22,3], got {m.shape}")
    return m - m[:, PELVIS:PELVIS + 1, :]


def estimate_ground(motion_trajectory: np.ndarray, percentile: float = GROUND_PERCENTILE) -> float:
    """ground_y = lower-foot-height 의 하위 percentile (NOT min-Y).

    각 frame 의 두 발 중 낮은 높이를 모아 하위 percentile → 발이 '쉬는' 높이.
    min-Y(단일 최저점)와 달리 occasional penetration/float outlier 에 robust.
    motion_trajectory(world) 에서 계산해야 의미 있음(root 이동 포함).
    """
    m = np.asarray(motion_trajectory, dtype=np.float64)
    if m.ndim != 3 or m.shape[1] != 22 or m.shape[2] != 3:
        raise ValueError(f"expected [T,22,3], got {m.shape}")
    lower_foot_y = np.minimum(m[:, LEFT_FOOT, 1], m[:, RIGHT_FOOT, 1])  # [T]
    return float(np.percentile(lower_foot_y, percentile))


def local_matches_trajectory(motion_local: np.ndarray, motion_trajectory: np.ndarray,
                             atol: float = 1e-6) -> bool:
    """check #4 정합: motion_local == trajectory - pelvis (tolerance 내)."""
    return bool(np.allclose(np.asarray(motion_local, dtype=np.float64),
                            derive_local(motion_trajectory), atol=atol))
