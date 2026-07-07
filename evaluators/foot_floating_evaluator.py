"""FootFloatingEvaluator — Week 2 evaluator prototype.

명세 §9.3.1 의 FootFloating 정의:

    FootFloating = mean_t I(contact_foot(t)) * I(p_foot_y(t) - ground_y > tau_float)

contact 로 추정되는 frame 에서 foot 의 ground 대비 높이가 임계 (tau_float) 를
초과하는 비율을 측정. 의미: "딛고 있어야 할 발이 공중에 떠 있는 정도".

본 evaluator 는 좌·우 foot 을 별도 score 로 보고하고, frame range 는
floating 이 처음 등장한 frame ~ 마지막 frame 으로 설정한다.

명세 §6.2 Evaluator 출력 schema 준수.
AGENTS.md §3-2 Tool Registry 인터페이스 의무.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from evaluators.base import Evaluator, EvaluatorReport, Severity, estimate_ground_y
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX

FOOT_JOINTS: dict[str, int] = {
    "left_foot": NAME_TO_IDX["LEFT_FOOT"],
    "right_foot": NAME_TO_IDX["RIGHT_FOOT"],
}

#: Severity 정의 버전. threshold 또는 contact heuristic 변경 시 bump.
#: AGENTS.md §4-2 evaluator 정의 변경 → aggregation_rule_version 의무.
#: 2.0.0 (2026-07-07, AR-063 — AR-062 audit A-4/A-6 fix):
#:   - contact heuristic: **수평(xz)속도 → 수직(y)속도** 기반 (v2, AR-058-3b 일치).
#:     수평속도 기반은 local(root-relative) 좌표에서 보행 중 지지발이 pelvis 기준
#:     수평 이동(≈보행속도)해 contact 판정 실패 → 보행 구간 floating blind (A-4).
#:   - ground 추정: min-Y(전 joint) → feet lower-height 10th percentile (A-2 동일 fix).
#:   호환성: 1.x 기록과 **비호환** (score 분포 이동) — 비교 시 severity_version 구분 의무.
#: 1.2.0 (2026-05-18 hold-out 결과 반영): contact heuristic 을 velocity + height 결합으로 sharpen.
SEVERITY_VERSION = "2.0.0-2026-07-07"

# Severity thresholds (FootFloating ratio, 0~1).
# Version 1.0.0 (Week 2 prototype): 0.05 / 0.15 / 0.30 — height 기반 contact heuristic.
# Version 1.1.0 (2026-05-18): contact heuristic 을 velocity 기반으로 교체 + threshold 유지.
#   기존 (1.0.0) 의 false-positive 가 contact heuristic 의 문제였지 threshold 의 문제가
#   아니었으므로 threshold 값 자체는 보수적 default 유지. hold-out 측정 후 추가 보정 검토.
SEV_LOW = 0.05
SEV_MED = 0.15
SEV_HIGH = 0.30

DEFAULT_TAU_FLOAT = 0.05  # 5cm — foot 가 ground 대비 5cm 이상 떠 있으면 floating
#: 수직속도 기반 contact heuristic 의 임계 (foot height 의 frame 간 변화 m/frame, v2).
#: v2.0.0 (AR-063): 수평속도(0.02) → 수직속도(0.035) 교체 — 수평속도는 local 좌표에서
#: 보행 중 지지발의 pelvis-상대 이동과 구분 불가 (AR-062 A-4). 수직 정지는 좌표계
#: 무관하게 "발이 내려앉아 머무는" 상태를 포착.
DEFAULT_VY_CONTACT_THRESH = 0.035
#: Contact 추정 시 foot 의 최대 허용 ground 대비 높이 (m). 본 값보다 위면 정지 상태라도
#: contact 가 아니라 "raised foot" 으로 분류 — 사용자 의도된 자세 (sitting/lying with feet
#: up, 정지된 발 들기 등) 의 false-positive floating 방지.
DEFAULT_TAU_CONTACT_HEIGHT = 0.10


class FootFloatingEvaluator(Evaluator):
    """Foot floating 평가기.

    Quality-tier-agnostic — G1/G2 output 또는 synthetic injection 결과 모두에 적용.

    Contact heuristic (v2.0.0, 2026-07-07 AR-063 — AR-062 audit A-4 fix):
      - 외부에서 `contact_labels` 가 명시되면 그대로 사용.
      - 명시 안 됨이면 **foot 의 vertical (y) velocity ≤ vy_contact_thresh
        AND foot height (above ground) ≤ tau_contact_height** 둘 다 만족하는 frame
        을 contact-intended 로 판정 (v2 계열 — AR-058-3b).
      - 의미: "발이 내려앉아 수직으로 머무는" 상태. 수직 정지는 local/trajectory
        좌표계 무관하게 유효 — 기존 수평속도 기반(v1.x)은 local 좌표에서 보행 중
        지지발이 pelvis 기준 수평 이동해 contact 실패 (보행 구간 floating blind).
      - 정지된 raised foot (> tau_contact_height) 은 여전히 contact 제외 →
        intentional pose 의 false-positive 회피 (v1.2.0 의 height 조건 유지).
      - ground 추정 (v2.0.0): min-Y(전 joint) → feet lower-height 10th percentile.

    이전 버전:
      - v1.2.0 (2026-05-18): horizontal velocity + height 결합. local 좌표 보행
        blind (AR-062 A-4) 로 v2.0.0 에서 대체.
      - v1.1.0 (2026-05-18 1st calibration): velocity-only contact. false-positive.
      - v1.0.0 (Week 2 prototype): height-only contact. gross false-positive.
    """

    name = "FootFloatingEvaluator"

    def __init__(
        self,
        tau_float: float = DEFAULT_TAU_FLOAT,
        vy_contact_thresh: float = DEFAULT_VY_CONTACT_THRESH,
        tau_contact_height: float = DEFAULT_TAU_CONTACT_HEIGHT,
    ) -> None:
        """
        Args:
            tau_float: foot height 가 ground 대비 본 값보다 클 때 floating 으로 판정 (m 단위).
            vy_contact_thresh: foot 의 vertical velocity 가 본 값 이하면 contact 후보 (m/frame, v2).
            tau_contact_height: foot 의 ground 대비 높이가 본 값 이하면 contact 후보 (m).
                두 조건 모두 만족해야 contact.
        """
        self.tau_float = tau_float
        self.vy_contact_thresh = vy_contact_thresh
        self.tau_contact_height = tau_contact_height

    def evaluate(
        self,
        motion: np.ndarray,
        fps: int = 20,
        ground_y: Optional[float] = None,
        contact_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> list[EvaluatorReport]:
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            raise ValueError(
                f"motion shape must be [T, 22, 3], got {motion.shape}. "
                "AGENTS.md §3-1 Canonical Motion Format."
            )
        T = motion.shape[0]
        if T < 2:
            return []

        # Ground 추정 (v2.0.0): 명시 안 됨이면 feet lower-height 10th percentile (NOT min-Y).
        if ground_y is None:
            ground_y = estimate_ground_y(motion)

        reports: list[EvaluatorReport] = []
        for part, joint_idx in FOOT_JOINTS.items():
            foot_y = motion[:, joint_idx, 1]  # [T]
            height_above_ground = foot_y - ground_y  # [T]

            # contact 추정 (v2.0.0): **수직속도** 정지 AND height 낮음 둘 다 (v2 계열).
            if contact_labels is None:
                if T >= 2:
                    foot_vy = np.abs(np.concatenate([np.diff(height_above_ground), [0.0]]))  # [T]
                else:
                    foot_vy = np.zeros(T)
                low_vertical_velocity = foot_vy <= self.vy_contact_thresh
                low_height = height_above_ground <= self.tau_contact_height
                contact = low_vertical_velocity & low_height
            else:
                contact = contact_labels[:, joint_idx].astype(bool)

            floating = (height_above_ground > self.tau_float) & contact  # [T]
            score = float(np.mean(floating))

            if score == 0.0:
                continue  # 본 foot 에 floating 없음 — report 생략

            # frame range — floating 이 일어난 첫·마지막 frame
            float_idx = np.where(floating)[0]
            start_frame = int(float_idx[0])
            end_frame = int(float_idx[-1])

            severity: Severity = _classify_severity(score)
            reports.append(
                EvaluatorReport(
                    agent=self.name,
                    error_type=f"{part}_floating",
                    body_part=part,
                    frames=(start_frame, end_frame),
                    score=score,
                    severity=severity,
                    recommendation="foot_lock_tool",
                    metadata={
                        "severity_version": SEVERITY_VERSION,
                        "tau_float_m": self.tau_float,
                        "vy_contact_thresh_m_per_frame": self.vy_contact_thresh,
                        "contact_heuristic": "vertical_velocity_based_v2.0.0"
                            if contact_labels is None
                            else "external",
                        "ground_y": float(ground_y),
                        "mean_height_above_ground": float(np.mean(height_above_ground)),
                        "max_height_above_ground": float(np.max(height_above_ground)),
                        "contact_frame_ratio": float(np.mean(contact)),
                    },
                )
            )
        return reports


def _classify_severity(score: float) -> Severity:
    if score >= SEV_HIGH:
        return "high"
    if score >= SEV_MED:
        return "medium"
    if score >= SEV_LOW:
        return "low"
    return "low"
