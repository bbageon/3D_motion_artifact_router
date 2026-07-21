"""RootGaitConsistencyTool — AR-072 prototype (병인 직접 처방: root 를 발에 맞춘다).

사전등록 설계: `.claude/docs/dashboard-task-specs/AR-072-root-aware-correction-design.md` ①.

배경: anchoring 계열(발을 root 에 맞춤)은 blind A/B 2회 우연 — 병인은 **root 전진 부족**
(AR-071: MDM v_root = GT 의 42%, 165/165 prompt). 본 tool 은 반대 방향:
**접지발이 world 에 고정이려면 root 가 얼마나 움직였어야 하는가를 역산**해
root 수평 궤적만 재구성한다 (다리 pose 는 한 점도 안 건드림 — GT-free).

원리 (stance 제약): 접지발의 world 속도 = root 속도 + (발−골반) 상대 속도 = 0
  ⇒ root_vel_target = −(발−골반 상대 수평 속도)   … "러닝머신 벨트 속도"

구조적 보존: 보정은 frame 별 동일 offset 을 전 joint xz 에 더하는 world 평행이동
  ⇒ local(root-relative)·bone 길이·높이(y)·contact 판정 = **완전 불변**.

명세 §6.3 CorrectionTool 인터페이스 + KDG 의무 (root 노드 — KDG 최상위, §3-3 정합).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from correction_tools.base import CorrectionTool, CorrectionReport, Strength
from correction_tools.coordinate_footskate_cleanup_tool import (
    DEFAULT_CONTACT_H, DEFAULT_CONTACT_VY, DEFAULT_GROUND_PERCENTILE,
    _estimate_ground, _v2_flags,
)
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX, SMPL_22

PELVIS = NAME_TO_IDX["PELVIS"]
FEET = (NAME_TO_IDX["LEFT_FOOT"], NAME_TO_IDX["RIGHT_FOOT"])

#: strength token → blend u (사전등록: primary u=1.0).
STRENGTH_U: dict[str, float] = {
    "small": 0.25, "medium": 0.5, "large": 1.0,
    "xsmall": 0.2, "small5": 0.4, "medium5": 0.6, "large5": 0.8, "xlarge": 1.0,
}
#: 해 velocity smoothing (사전등록 상수 — 노이즈가 root jitter 로 전이되는 것 방지).
SOLVE_SMOOTH_SIGMA = 2.0

#: continuous_u 상한 (AR-085, 2026-07-22 사용자 directive "열어보자"). u>1 = 접지-일관
#: 제약을 과-적용 (extrapolation: v_new = v_target + (u-1)(v_target - v_orig)). u=1 은
#: "우리 접지 추정 기준 완전 강제"일 뿐 GT-최적이 아니므로, 추정이 낮게 잡힌 모션에서
#: u>1 이 GT 에 더 가까울 수 있음 (검정 대상). offset(u) = u·offset(u=1) 선형은 u 범위
#: 무관하게 성립. u≤1 기존 호출 불변 (backward-compat: AR-072/077/081/082 영향 없음).
U_MAX_CONTINUOUS = 2.0


class RootGaitConsistencyTool(CorrectionTool):
    """Contact-consistent root solve — root 수평 궤적 재구성 (pose 무수정).

    target_part: 'root' (관례). trajectory(world) 전제 — local 입력 시 ValueError.
    """

    name = "RootGaitConsistencyTool"

    def __init__(
        self,
        contact_h: float = DEFAULT_CONTACT_H,
        contact_vy: float = DEFAULT_CONTACT_VY,
        ground_percentile: float = DEFAULT_GROUND_PERCENTILE,
        smooth_sigma: float = SOLVE_SMOOTH_SIGMA,
    ) -> None:
        self.contact_h = contact_h
        self.contact_vy = contact_vy
        self.ground_percentile = ground_percentile
        self.smooth_sigma = smooth_sigma

    def apply(
        self,
        motion: np.ndarray,
        target_part: str,
        target_joints: list[str],
        frame_range: tuple[int, int],
        strength: Strength = "large",
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, CorrectionReport]:
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            raise ValueError(
                f"motion shape must be [T, 22, 3], got {motion.shape}. "
                "AGENTS.md §3-1 Canonical Motion Format."
            )
        meta = metadata or {}
        coord_space = str(meta.get("coord_space", "trajectory"))
        if coord_space != "trajectory":
            raise ValueError(
                f"coord_space={coord_space!r}: RootGaitConsistencyTool 은 trajectory(world) "
                "에서만 동작한다 (root 이동이 제거된 local 에선 무의미)."
            )
        T = motion.shape[0]
        if T < 3:
            return motion.copy(), CorrectionReport(
                tool=self.name, target_part=target_part, frame_range=(0, T - 1),
                strength=strength, modified_joints=[], correction_magnitude=0.0,
                metadata={"reason": "T < 3 — solve skipped"})

        if "continuous_u" in meta:
            u = float(np.clip(float(meta["continuous_u"]), 0.0, U_MAX_CONTINUOUS))
            u_source = "continuous"
        else:
            u = STRENGTH_U.get(strength, 1.0)
            u_source = "token"

        ground = float(meta.get("ground_y", _estimate_ground(motion, self.ground_percentile)))

        pel_xz = motion[:, PELVIS, :][:, [0, 2]]
        v_orig = np.diff(pel_xz, axis=0)                       # [T-1, 2] frame t→t+1
        # stance 제약에서 target root velocity 역산.
        num = np.zeros((T - 1, 2))
        cnt = np.zeros(T - 1)
        for f in FEET:
            contact, _ = _v2_flags(motion, f, ground, self.contact_h, self.contact_vy, 1e9)
            rel = motion[:, f, :][:, [0, 2]] - pel_xz          # 발−골반 상대 xz
            rel_v = np.diff(rel, axis=0)                       # [T-1, 2]
            # velocity t→t+1 에 대한 접지: 양 끝 frame 모두 접지일 때만 제약으로 사용.
            m = contact[:-1] & contact[1:]
            num[m] += -rel_v[m]
            cnt[m] += 1.0
        has = cnt > 0
        v_target = np.full((T - 1, 2), np.nan)
        v_target[has] = num[has] / cnt[has, None]
        # 무접지 구간: 이웃 정의값 선형 보간 (전 구간 무접지면 원본 유지).
        if has.any():
            idx = np.arange(T - 1)
            for d in range(2):
                v_target[:, d] = np.interp(idx, idx[has], v_target[has, d])
        else:
            v_target = v_orig.copy()
        # smoothing (사전등록 σ) — 해의 노이즈가 root jitter 로 전이되는 것 방지.
        if self.smooth_sigma > 0:
            v_target = gaussian_filter1d(v_target, sigma=self.smooth_sigma, axis=0, mode="nearest")
        # blend + 적분 (시작점 고정).
        v_new = (1.0 - u) * v_orig + u * v_target
        new_pel = np.vstack([pel_xz[0], pel_xz[0] + np.cumsum(v_new, axis=0)])  # [T,2]
        offset = new_pel - pel_xz                                               # [T,2]

        corrected = motion.copy()
        corrected[:, :, 0] += offset[:, 0:1]
        corrected[:, :, 2] += offset[:, 1:2]

        disp = np.linalg.norm(offset, axis=1)
        report = CorrectionReport(
            tool=self.name,
            target_part=target_part,
            frame_range=(0, T - 1),
            strength=strength,
            modified_joints=list(SMPL_22),  # world 평행이동 — 전 joint 이동 (local 불변)
            correction_magnitude=float(np.mean(disp)),
            metadata={
                "ground_y": ground, "coord_space": coord_space,
                "u": u, "u_source": u_source,
                "constrained_frame_frac": round(float(has.mean()), 4),
                "path_len_before": round(float(np.linalg.norm(v_orig, axis=1).sum()), 4),
                "path_len_after": round(float(np.linalg.norm(v_new, axis=1).sum()), 4),
                "max_offset": round(float(disp.max()), 4),
                "smooth_sigma": self.smooth_sigma,
                "invariance": "local pose/bone/height 불변 (world xz 평행이동만)",
            },
        )
        return corrected, report

    def kdg_affected_joints(self) -> list[str]:
        # root(전신 평행이동) — KDG 최상위 노드.
        return ["PELVIS"]

    def kdg_propagation_weights(self) -> dict[str, float]:
        # 평행이동은 전 joint 에 weight 1.0 로 전파 (강체 이동 — pose 불변).
        return {j: 1.0 for j in SMPL_22 if j != "PELVIS"}
