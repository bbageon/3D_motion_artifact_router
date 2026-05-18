"""BoneLengthEvaluator — Week 2 evaluator prototype.

명세 §9.3.1 의 Bone length variation 정의:

    BoneVar = mean_{t,b} | length_b(t) - length_b_ref | / length_b_ref

canonical SMPL kinematic chain 의 각 bone 에 대해 frame 별 길이와 reference
길이의 상대 편차를 측정. 의미: "골격의 강체 가정 (frame 간 bone 길이 일정)
이 얼마나 깨졌는가". generator 가 생성한 motion 이 강체성 제약을 위반하면
이 값이 커진다.

본 evaluator 는 chain 별 (right_leg / left_leg / spine_head / right_arm / left_arm)
score 를 별도 report 로 보고. reference length 는 motion 전체의 frame-wise
median 으로 추정 (robust statistic).

명세 §6.2 Evaluator 출력 schema 준수.
AGENTS.md §3-2 Tool Registry 인터페이스 의무.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from evaluators.base import Evaluator, EvaluatorReport, Severity
from skeleton_normalizer.canonical_smpl_22 import (
    CHAIN_LABELS,
    T2M_KINEMATIC_CHAIN,
)

#: Severity 정의 버전. AGENTS.md §3-15 raw record metadata 의무.
#: BoneLength 는 baseline calibration (150 sample) 에서 0 report 였으므로 prototype
#: threshold 유지가 안전 — calibration 으로 변경 사유가 발생할 때 bump.
SEVERITY_VERSION = "1.0.0-2026-05-13"

# Severity thresholds (mean relative bone length deviation)
SEV_LOW = 0.02   # 2%
SEV_MED = 0.05   # 5%
SEV_HIGH = 0.10  # 10%


class BoneLengthEvaluator(Evaluator):
    """Bone length variation 평가기.

    각 kinematic chain 안의 bone 들에 대해 reference 길이 대비 frame-wise
    상대 편차의 평균을 score 로 보고.
    """

    name = "BoneLengthEvaluator"

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

        reports: list[EvaluatorReport] = []
        for chain, label in zip(T2M_KINEMATIC_CHAIN, CHAIN_LABELS):
            chain_scores: list[float] = []
            chain_bone_devs: dict[str, float] = {}
            for parent_idx, child_idx in zip(chain[:-1], chain[1:]):
                bone_vec = motion[:, child_idx, :] - motion[:, parent_idx, :]  # [T, 3]
                bone_len = np.linalg.norm(bone_vec, axis=1)  # [T]
                if bone_len.size == 0:
                    continue
                ref_len = float(np.median(bone_len))
                if ref_len < 1e-6:
                    continue
                rel_dev = np.abs(bone_len - ref_len) / ref_len  # [T]
                bone_score = float(np.mean(rel_dev))
                chain_scores.append(bone_score)
                chain_bone_devs[f"{parent_idx}->{child_idx}"] = bone_score

            if not chain_scores:
                continue

            score = float(np.mean(chain_scores))
            if score < SEV_LOW * 0.5:
                continue  # noise floor 이하 — report 생략

            # frame range 는 본 chain 의 max-deviation frame 위주로 (전 frame 통계라 [0, T-1] 보고)
            severity: Severity = _classify_severity(score)
            reports.append(
                EvaluatorReport(
                    agent=self.name,
                    error_type=f"{label}_bone_length_variation",
                    body_part=label,
                    frames=(0, T - 1),
                    score=score,
                    severity=severity,
                    recommendation="bone_projection_tool",
                    metadata={
                        "chain_bone_devs": chain_bone_devs,
                        "max_bone_dev": float(np.max(chain_scores)),
                        "min_bone_dev": float(np.min(chain_scores)),
                        "n_bones": len(chain_scores),
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
