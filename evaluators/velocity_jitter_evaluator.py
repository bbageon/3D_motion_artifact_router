"""VelocityJitterEvaluator — Week 2 evaluator prototype.

명세 §9.3.1 의 Velocity jitter 정의:

    VelocityJitter = mean_{t,j} || v_j(t+1) - v_j(t) ||
                  = mean_{t,j} || a_j(t) ||

frame 간 joint velocity 의 변화량 (= acceleration) 의 L2 norm 의 평균.
의미: "motion 이 얼마나 떨리는가 — 짧은 시간 구간에서의 가속도 크기".
diffusion-based generator 가 일반적으로 부드러운 motion 을 만드는 반면
token-based generator (MotionGPT 등) 는 token boundary 에서 jitter 가
나타날 수 있다.

본 evaluator 는 (a) 전체 평균 score 1 개 + (b) body part 별 (legs / arms /
spine_head) 분해 score 를 보고. severity 는 reference 분포 (HumanML3D GT)
와 비교한 z-score 기반으로 산정 (현 prototype 단계에서는 hard-coded
threshold; 추후 reference 분포 학습 후 갱신 예정).

명세 §6.2 Evaluator 출력 schema 준수.
AGENTS.md §3-2 Tool Registry 인터페이스 의무.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from evaluators.base import Evaluator, EvaluatorReport, Severity
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX

# Body part 그룹 — joint index list
PART_JOINTS: dict[str, list[int]] = {
    "legs": [
        NAME_TO_IDX["LEFT_HIP"], NAME_TO_IDX["LEFT_KNEE"], NAME_TO_IDX["LEFT_ANKLE"], NAME_TO_IDX["LEFT_FOOT"],
        NAME_TO_IDX["RIGHT_HIP"], NAME_TO_IDX["RIGHT_KNEE"], NAME_TO_IDX["RIGHT_ANKLE"], NAME_TO_IDX["RIGHT_FOOT"],
    ],
    "arms": [
        NAME_TO_IDX["LEFT_COLLAR"], NAME_TO_IDX["LEFT_SHOULDER"], NAME_TO_IDX["LEFT_ELBOW"], NAME_TO_IDX["LEFT_WRIST"],
        NAME_TO_IDX["RIGHT_COLLAR"], NAME_TO_IDX["RIGHT_SHOULDER"], NAME_TO_IDX["RIGHT_ELBOW"], NAME_TO_IDX["RIGHT_WRIST"],
    ],
    "spine_head": [
        NAME_TO_IDX["PELVIS"], NAME_TO_IDX["SPINE1"], NAME_TO_IDX["SPINE2"], NAME_TO_IDX["SPINE3"],
        NAME_TO_IDX["NECK"], NAME_TO_IDX["HEAD"],
    ],
}

#: Severity 정의 버전. threshold 변경 시 bump + 일지·snapshot 에 versioning 기록.
#: AGENTS.md §4-2 evaluator 정의 변경 → aggregation_rule_version 상승 의무.
SEVERITY_VERSION = "1.1.0-2026-05-18"

# Severity thresholds — mean acceleration magnitude (m / frame^2). 20fps 기준.
#
# Version 1.1.0 (2026-05-18) — HumanML3D 150 sample (seed=42, task_id=
# baseline_calibration_v1) 분포 기반 calibration:
#   clean motion 의 global jitter score 분포:
#     median 0.0068, p75 0.011, p90 0.019, p95 0.026, p99 0.040, max 0.057
#   목표: clean motion 의 ~95% 가 sub-low (report 면제). corrupted motion
#   (synthetic inject_jitter noise_std≥0.05) 은 의미 있는 severity 분류.
#   raw record: evals/raw/<...>_baseline_calibration_v1_*.json (150 개).
#   snapshot: evals/snapshots/baseline_calibration_v1.json.
#
# Version 1.0.0 (Week 2 prototype) — 보수적 default (0.005 / 0.020 / 0.050).
SEV_LOW = 0.030
SEV_MED = 0.060
SEV_HIGH = 0.120


class VelocityJitterEvaluator(Evaluator):
    """Velocity jitter 평가기.

    전체 평균 + body-part 별 분해 보고. acceleration L2 norm 의 평균을 score 로.
    """

    name = "VelocityJitterEvaluator"

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
        if T < 3:
            return []  # 가속도 계산은 T >= 3 필요

        # velocity[t] = motion[t+1] - motion[t]                  → [T-1, 22, 3]
        # acceleration[t] = velocity[t+1] - velocity[t]          → [T-2, 22, 3]
        velocity = np.diff(motion, axis=0)
        acceleration = np.diff(velocity, axis=0)  # [T-2, 22, 3]
        accel_mag = np.linalg.norm(acceleration, axis=2)  # [T-2, 22]

        reports: list[EvaluatorReport] = []

        # (a) 전체 score
        global_score = float(np.mean(accel_mag))
        if global_score >= SEV_LOW * 0.5:
            reports.append(
                EvaluatorReport(
                    agent=self.name,
                    error_type="global_velocity_jitter",
                    body_part="full_body",
                    frames=(0, T - 1),
                    score=global_score,
                    severity=_classify_severity(global_score),
                    recommendation="velocity_smoothing_tool",
                    metadata={
                        "fps": fps,
                        "max_per_frame": float(np.max(accel_mag)),
                        "n_frames_accel": int(accel_mag.shape[0]),
                    },
                )
            )

        # (b) body part 별 분해
        for part, joints in PART_JOINTS.items():
            part_accel = accel_mag[:, joints]  # [T-2, K]
            part_score = float(np.mean(part_accel))
            if part_score < SEV_LOW * 0.5:
                continue
            severity = _classify_severity(part_score)
            reports.append(
                EvaluatorReport(
                    agent=self.name,
                    error_type=f"{part}_velocity_jitter",
                    body_part=part,
                    frames=(0, T - 1),
                    score=part_score,
                    severity=severity,
                    recommendation="velocity_smoothing_tool",
                    metadata={
                        "fps": fps,
                        "n_joints": len(joints),
                        "max_per_frame": float(np.max(part_accel)),
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
