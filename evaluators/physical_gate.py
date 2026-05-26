"""PhysicalGateV0 — 5 evaluators for Safe Orchestration Layer B (사용자 directive 2026-05-26).

본 module 은 [`docs/current_research_position.md §0-4`](../docs/current_research_position.md)
의 Safe Orchestration architecture 의 **Evaluator Layer B (Physical Constraint Gate)**
의 정식 구현. 5 evaluator:

  1. PenetrateEvaluator    — foot/ankle ground penetration ratio (PhysDiff motivation)
  2. FloatEvaluator        — contact frame floating ratio (PhysDiff + MDM)
  3. SkateEvaluator        — contact frame foot horizontal velocity ratio (PhysDiff + MDM)
  4. JerkSpikeEvaluator    — per-joint acceleration p95 (MDM + VIBE + TCMR)
  5. BoneLengthCVEvaluator — per-bone length coefficient of variation (HuMoR + MDM + ACTOR)

본 evaluator 들은 **gate decision (accept/repair/rollback/STOP)** 의 정량 입력. NetGain
의 weight 가 아닌 hard gate 의 boundary.

Threshold calibration: HumanML3D clean N=500~1000 의 p99 (보수적) 또는 p95 (loose).
calibration 결과는 `evals/snapshots/physical_gate_clean_calibration_v1.json` 에 저장.

Reference papers (AGENTS.md §3-22, 2020+ peer-reviewed top-tier):
- PhysDiff (Yuan et al. 2023, ICCV)
- MDM (Tevet et al. 2023, ICLR)
- HuMoR (Rempe et al. 2021, ICCV)
- VIBE (Kocabas et al. 2020, CVPR)
- TCMR (Choi et al. 2021, CVPR)

상세 — [`docs/metric_provenance.md §3-5`](../docs/metric_provenance.md).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from evaluators.base import Evaluator, EvaluatorReport, Severity
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX, T2M_KINEMATIC_CHAIN

# Severity version — calibration 또는 threshold formula 변경 시 bump.
SEVERITY_VERSION = "0.1.0-2026-05-26"

FOOT_JOINTS = {
    "left_foot": NAME_TO_IDX["LEFT_FOOT"],
    "right_foot": NAME_TO_IDX["RIGHT_FOOT"],
    "left_ankle": NAME_TO_IDX["LEFT_ANKLE"],
    "right_ankle": NAME_TO_IDX["RIGHT_ANKLE"],
}

# Default thresholds (provisional — HumanML3D clean calibration 전).
# calibration 후 replace with p99 values.
DEFAULT_PENETRATE_EPS = 0.02       # 발이 ground 아래로 본 값 이상 들어가면 penetrate (m).
DEFAULT_FLOAT_THRESHOLD = 0.05     # contact frame 의 foot height > 본 값이면 float (m).
DEFAULT_SKATE_THRESHOLD = 0.05     # contact frame 의 foot horizontal velocity > 본 값이면 skate (m/frame).
DEFAULT_V_CONTACT_THRESH = 0.02    # contact 추정 의 velocity 임계 (foot_floating_evaluator 일관).
DEFAULT_TAU_CONTACT_HEIGHT = 0.10  # contact 추정 의 height 임계 (foot_floating_evaluator 일관).
DEFAULT_JERKSPIKE_PERCENTILE = 95  # JerkSpike 의 percentile.

# Provisional severity thresholds — calibration 후 replace.
SEV_PENETRATE = (0.01, 0.05, 0.15)
SEV_FLOAT = (0.05, 0.15, 0.30)
SEV_SKATE = (0.05, 0.15, 0.30)
SEV_JERKSPIKE = (0.5, 1.0, 2.0)
SEV_BONELENGTH_CV = (0.02, 0.05, 0.10)


def _classify_severity(score: float, thresholds: tuple[float, float, float]) -> Severity:
    low, med, high = thresholds
    if score >= high:
        return "high"
    if score >= med:
        return "medium"
    if score >= low:
        return "low"
    return "low"


def _estimate_ground_y(motion: np.ndarray) -> float:
    """Heuristic ground_y — motion 전체 의 min Y (HumanML3D root-relative)."""
    return float(np.min(motion[:, :, 1]))


def _estimate_contact(
    motion: np.ndarray, ground_y: float, joint_idx: int,
    v_thresh: float = DEFAULT_V_CONTACT_THRESH, h_thresh: float = DEFAULT_TAU_CONTACT_HEIGHT,
) -> np.ndarray:
    """Foot contact 추정 (foot_floating_evaluator v1.2.0 의 velocity + height 결합)."""
    T = motion.shape[0]
    foot_y = motion[:, joint_idx, 1]
    height = foot_y - ground_y
    foot_xz = motion[:, joint_idx, :][:, [0, 2]]
    if T >= 2:
        xz_disp = np.linalg.norm(np.diff(foot_xz, axis=0), axis=1)
        foot_xz_vel = np.concatenate([xz_disp, [xz_disp[-1]]])
    else:
        foot_xz_vel = np.zeros(T)
    return (foot_xz_vel <= v_thresh) & (height <= h_thresh)


class PenetrateEvaluator(Evaluator):
    """Ground penetration ratio (per foot/ankle).

    Score = mean_t I(joint_y(t) < ground_y - penetrate_eps).
    PhysDiff (Yuan 2023, ICCV) motivation.
    """

    name = "PenetrateEvaluator"

    def __init__(self, penetrate_eps: float = DEFAULT_PENETRATE_EPS):
        self.penetrate_eps = penetrate_eps

    def evaluate(
        self, motion: np.ndarray, fps: int = 20,
        ground_y: Optional[float] = None, contact_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> list[EvaluatorReport]:
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            raise ValueError(f"motion shape must be [T, 22, 3], got {motion.shape}")
        T = motion.shape[0]
        if T < 1:
            return []
        if ground_y is None:
            ground_y = _estimate_ground_y(motion)
        reports = []
        for part, idx in FOOT_JOINTS.items():
            joint_y = motion[:, idx, 1]
            penetrate_mask = joint_y < (ground_y - self.penetrate_eps)
            score = float(np.mean(penetrate_mask))
            if score == 0.0:
                continue
            pen_idx = np.where(penetrate_mask)[0]
            reports.append(EvaluatorReport(
                agent=self.name, error_type=f"{part}_penetrate", body_part=part,
                frames=(int(pen_idx[0]), int(pen_idx[-1])),
                score=score, severity=_classify_severity(score, SEV_PENETRATE),
                recommendation="rollback_or_repair",
                metadata={
                    "severity_version": SEVERITY_VERSION,
                    "penetrate_eps_m": self.penetrate_eps,
                    "ground_y": float(ground_y),
                    "max_depth_below_ground": float(max(0.0, ground_y - float(np.min(joint_y)))),
                },
            ))
        return reports


class FloatEvaluator(Evaluator):
    """Contact-frame foot floating ratio.

    Score = mean_t I(contact(t)) * I(foot_y(t) - ground_y > float_threshold).
    PhysDiff + MDM. FootFloatingEvaluator 의 gate-form variant.
    """

    name = "FloatEvaluator"

    def __init__(
        self, float_threshold: float = DEFAULT_FLOAT_THRESHOLD,
        v_contact_thresh: float = DEFAULT_V_CONTACT_THRESH,
        tau_contact_height: float = DEFAULT_TAU_CONTACT_HEIGHT,
    ):
        self.float_threshold = float_threshold
        self.v_contact_thresh = v_contact_thresh
        self.tau_contact_height = tau_contact_height

    def evaluate(
        self, motion: np.ndarray, fps: int = 20,
        ground_y: Optional[float] = None, contact_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> list[EvaluatorReport]:
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            raise ValueError(f"motion shape must be [T, 22, 3], got {motion.shape}")
        T = motion.shape[0]
        if T < 2:
            return []
        if ground_y is None:
            ground_y = _estimate_ground_y(motion)
        reports = []
        for part, idx in {"left_foot": NAME_TO_IDX["LEFT_FOOT"], "right_foot": NAME_TO_IDX["RIGHT_FOOT"]}.items():
            foot_y = motion[:, idx, 1]
            height = foot_y - ground_y
            if contact_labels is not None:
                contact = contact_labels[:, idx].astype(bool)
            else:
                contact = _estimate_contact(motion, ground_y, idx,
                                             self.v_contact_thresh, self.tau_contact_height)
            floating = (height > self.float_threshold) & contact
            score = float(np.mean(floating))
            if score == 0.0:
                continue
            float_idx = np.where(floating)[0]
            reports.append(EvaluatorReport(
                agent=self.name, error_type=f"{part}_float", body_part=part,
                frames=(int(float_idx[0]), int(float_idx[-1])),
                score=score, severity=_classify_severity(score, SEV_FLOAT),
                recommendation="foot_lock_tool_or_rollback",
                metadata={
                    "severity_version": SEVERITY_VERSION,
                    "float_threshold_m": self.float_threshold,
                    "ground_y": float(ground_y),
                    "contact_frame_ratio": float(np.mean(contact)),
                },
            ))
        return reports


class SkateEvaluator(Evaluator):
    """Contact-frame foot sliding ratio.

    Score = mean_t I(contact(t)) * I(|foot_xz_velocity(t)| > skate_threshold).
    PhysDiff + MDM. foot floating 과 orthogonal artifact.
    """

    name = "SkateEvaluator"

    def __init__(
        self, skate_threshold: float = DEFAULT_SKATE_THRESHOLD,
        v_contact_thresh: float = DEFAULT_V_CONTACT_THRESH,
        tau_contact_height: float = DEFAULT_TAU_CONTACT_HEIGHT,
    ):
        self.skate_threshold = skate_threshold
        self.v_contact_thresh = v_contact_thresh
        self.tau_contact_height = tau_contact_height

    def evaluate(
        self, motion: np.ndarray, fps: int = 20,
        ground_y: Optional[float] = None, contact_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> list[EvaluatorReport]:
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            raise ValueError(f"motion shape must be [T, 22, 3], got {motion.shape}")
        T = motion.shape[0]
        if T < 2:
            return []
        if ground_y is None:
            ground_y = _estimate_ground_y(motion)
        reports = []
        for part, idx in {"left_foot": NAME_TO_IDX["LEFT_FOOT"], "right_foot": NAME_TO_IDX["RIGHT_FOOT"]}.items():
            foot_xz = motion[:, idx, :][:, [0, 2]]
            xz_disp = np.linalg.norm(np.diff(foot_xz, axis=0), axis=1)
            foot_xz_vel = np.concatenate([xz_disp, [xz_disp[-1]]])
            if contact_labels is not None:
                contact = contact_labels[:, idx].astype(bool)
            else:
                contact = _estimate_contact(motion, ground_y, idx,
                                             self.v_contact_thresh, self.tau_contact_height)
            # NB: skate measured against the same v_contact_thresh used to detect contact
            # would be vacuous (contact already requires vel <= v_thresh). Use skate_threshold
            # for sliding amount (which is independent of contact heuristic strictness).
            sliding = (foot_xz_vel > self.skate_threshold) & contact
            score = float(np.mean(sliding))
            if score == 0.0:
                continue
            sk_idx = np.where(sliding)[0]
            reports.append(EvaluatorReport(
                agent=self.name, error_type=f"{part}_skate", body_part=part,
                frames=(int(sk_idx[0]), int(sk_idx[-1])),
                score=score, severity=_classify_severity(score, SEV_SKATE),
                recommendation="foot_lock_tool_or_rollback",
                metadata={
                    "severity_version": SEVERITY_VERSION,
                    "skate_threshold_m_per_frame": self.skate_threshold,
                    "ground_y": float(ground_y),
                    "contact_frame_ratio": float(np.mean(contact)),
                    "max_xz_velocity": float(np.max(foot_xz_vel)),
                },
            ))
        return reports


class JerkSpikeEvaluator(Evaluator):
    """Acceleration spike (p95) — MDM/VIBE/TCMR motivation.

    Score = p95 of (per-joint, per-frame) acceleration norm, motion-wide.
    Single report (motion-level), not per-joint or per-frame.
    """

    name = "JerkSpikeEvaluator"

    def __init__(self, percentile: int = DEFAULT_JERKSPIKE_PERCENTILE):
        self.percentile = percentile

    def evaluate(
        self, motion: np.ndarray, fps: int = 20,
        ground_y: Optional[float] = None, contact_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> list[EvaluatorReport]:
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            raise ValueError(f"motion shape must be [T, 22, 3], got {motion.shape}")
        T = motion.shape[0]
        if T < 3:
            return []
        # acceleration = 2nd derivative of position.
        vel = np.diff(motion, axis=0)  # [T-1, 22, 3]
        accel = np.diff(vel, axis=0)   # [T-2, 22, 3]
        accel_norm = np.linalg.norm(accel, axis=-1)  # [T-2, 22]
        score = float(np.percentile(accel_norm.flatten(), self.percentile))
        # Single report — motion-level.
        # Locate worst-case frame for `frames` field.
        max_per_frame = accel_norm.max(axis=1)  # [T-2]
        worst_frame = int(np.argmax(max_per_frame)) + 1  # +1 because accel index = pos index - 2
        return [EvaluatorReport(
            agent=self.name, error_type="acceleration_spike", body_part="full_body",
            frames=(worst_frame, worst_frame),
            score=score, severity=_classify_severity(score, SEV_JERKSPIKE),
            recommendation="velocity_smoothing_tool_or_rollback",
            metadata={
                "severity_version": SEVERITY_VERSION,
                "percentile": self.percentile,
                "p50": float(np.percentile(accel_norm.flatten(), 50)),
                "p90": float(np.percentile(accel_norm.flatten(), 90)),
                "p99": float(np.percentile(accel_norm.flatten(), 99)),
                "mean": float(np.mean(accel_norm)),
            },
        )]


class BoneLengthCVEvaluator(Evaluator):
    """Per-bone length coefficient of variation (std/mean) — HuMoR/MDM/ACTOR motivation.

    Score = max over bones of (std(bone_length) / mean(bone_length)) across frames.
    BoneLengthEvaluator (mean variation) 의 strict gate variant.
    """

    name = "BoneLengthCVEvaluator"

    def __init__(self):
        # Build (parent, child) bone pairs from T2M_KINEMATIC_CHAIN.
        bones: list[tuple[int, int]] = []
        for chain in T2M_KINEMATIC_CHAIN:
            for a, b in zip(chain[:-1], chain[1:]):
                bones.append((a, b))
        # Dedup.
        self.bones = list(dict.fromkeys(bones))

    def evaluate(
        self, motion: np.ndarray, fps: int = 20,
        ground_y: Optional[float] = None, contact_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> list[EvaluatorReport]:
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            raise ValueError(f"motion shape must be [T, 22, 3], got {motion.shape}")
        T = motion.shape[0]
        if T < 2:
            return []
        cv_per_bone: list[float] = []
        for a, b in self.bones:
            lengths = np.linalg.norm(motion[:, a, :] - motion[:, b, :], axis=-1)  # [T]
            mean_l = float(np.mean(lengths))
            std_l = float(np.std(lengths))
            cv = std_l / mean_l if mean_l > 1e-9 else 0.0
            cv_per_bone.append(cv)
        worst_bone_idx = int(np.argmax(cv_per_bone))
        score = float(max(cv_per_bone))
        worst_a, worst_b = self.bones[worst_bone_idx]
        return [EvaluatorReport(
            agent=self.name, error_type="bone_length_cv", body_part="full_body",
            frames=(0, T - 1),
            score=score, severity=_classify_severity(score, SEV_BONELENGTH_CV),
            recommendation="bone_projection_tool_or_rollback",
            metadata={
                "severity_version": SEVERITY_VERSION,
                "worst_bone": [worst_a, worst_b],
                "all_bone_cvs": cv_per_bone,
                "mean_cv": float(np.mean(cv_per_bone)),
                "p95_cv": float(np.percentile(cv_per_bone, 95)),
            },
        )]


#: PhysicalGateV0 의 default evaluator registry.
DEFAULT_PHYSICAL_GATE_EVALUATORS: list[Evaluator] = [
    PenetrateEvaluator(),
    FloatEvaluator(),
    SkateEvaluator(),
    JerkSpikeEvaluator(),
    BoneLengthCVEvaluator(),
]
