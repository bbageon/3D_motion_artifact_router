"""BoneProjectionTool — Week 3 correction tool prototype.

[`BoneLengthEvaluator`](../evaluators/bone_length_evaluator.py) 와 짝지어진 tool.
지정 chain 의 각 bone (parent → child pair) 의 길이를 reference 길이로 강제 투영.
구체:
  - bone vector = motion[child] - motion[parent].
  - new_bone_vec = bone_vec * (ref_len / current_len).
  - child joint 와 그 chain 의 모든 descendant 에 동일 displacement 를 더해
    forward-kinematics 일관성 유지.
  - strength 에 따라 ref 와의 interpolation factor 조절 (small ~ 0.3 / large ~ 1.0).

reference 길이는 metadata['reference_lengths'] 에서 가져오거나, 없으면 frame-
wise median 으로 추정.

명세 §6.3 CorrectionTool 인터페이스 + KDG affected joints 의무.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from correction_tools.base import CorrectionTool, CorrectionReport, Strength
from skeleton_normalizer.canonical_smpl_22 import (
    CHAIN_LABELS,
    NAME_TO_IDX,
    SMPL_22,
    T2M_KINEMATIC_CHAIN,
)

STRENGTH_FACTOR: dict[str, float] = {"small": 0.3, "medium": 0.6, "large": 1.0}


class BoneProjectionTool(CorrectionTool):
    """Bone length projection — bone 들을 reference 길이로 강제.

    target_part 은 [`CHAIN_LABELS`](../skeleton_normalizer/canonical_smpl_22.py)
    (right_leg / left_leg / spine_head / right_arm / left_arm) 중 하나.
    target_joints 가 명시되면 chain 의 subset 으로 한정 가능.
    """

    name = "BoneProjectionTool"

    def apply(
        self,
        motion: np.ndarray,
        target_part: str,
        target_joints: list[str],
        frame_range: tuple[int, int],
        strength: Strength = "medium",
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, CorrectionReport]:
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            raise ValueError(
                f"motion shape must be [T, 22, 3], got {motion.shape}. "
                "AGENTS.md §3-1 Canonical Motion Format."
            )
        if target_part not in CHAIN_LABELS:
            raise ValueError(
                f"target_part {target_part!r} not in {CHAIN_LABELS}. "
                "BoneProjectionTool 는 kinematic chain 단위로만 적용."
            )
        T = motion.shape[0]
        start, end = frame_range
        start = max(0, start)
        end = min(T, end + 1) if end < T else T

        meta = metadata or {}
        factor: float = STRENGTH_FACTOR.get(strength, STRENGTH_FACTOR["medium"])
        explicit_refs: dict[str, float] = meta.get("reference_lengths", {})

        chain = T2M_KINEMATIC_CHAIN[CHAIN_LABELS.index(target_part)]

        corrected = motion.copy()
        deltas: list[float] = []
        modified_joints: list[str] = []

        # bone 별로 parent → child 순 처리. child 위치 변경 시 descendant 도 동일 delta 적용.
        for parent_idx, child_idx in zip(chain[:-1], chain[1:]):
            bone_key = f"{parent_idx}->{child_idx}"
            bone_vec = corrected[start:end, child_idx, :] - corrected[start:end, parent_idx, :]
            current_len = np.linalg.norm(bone_vec, axis=1)  # [n_frames]

            if bone_key in explicit_refs:
                ref_len = float(explicit_refs[bone_key])
            else:
                # 본 chain 전체 motion 의 median (보고된 reference 와 동일 정의).
                full_bone = motion[:, child_idx, :] - motion[:, parent_idx, :]
                ref_len = float(np.median(np.linalg.norm(full_bone, axis=1)))

            if ref_len < 1e-6:
                continue
            # 0 길이 frame 회피
            current_len_safe = np.maximum(current_len, 1e-6)[:, None]
            target_bone_vec = bone_vec * (ref_len / current_len_safe)
            new_bone_vec = bone_vec + factor * (target_bone_vec - bone_vec)
            displacement = new_bone_vec - bone_vec  # [n_frames, 3]

            # child 부터 descendant 까지 모두 동일 displacement 적용.
            descendants = chain[chain.index(child_idx):]
            for d in descendants:
                corrected[start:end, d, :] += displacement
                dname = SMPL_22[d]
                if dname not in modified_joints:
                    modified_joints.append(dname)

            deltas.append(float(np.mean(np.linalg.norm(displacement, axis=1))))

        correction_magnitude = float(np.mean(deltas)) if deltas else 0.0
        report = CorrectionReport(
            tool=self.name,
            target_part=target_part,
            frame_range=(int(start), int(end - 1)),
            strength=strength,
            modified_joints=modified_joints,
            correction_magnitude=correction_magnitude,
            metadata={
                "strength_factor": factor,
                "chain": [SMPL_22[i] for i in chain],
                "n_bones_corrected": len(deltas),
            },
        )
        return corrected, report

    def kdg_affected_joints(self) -> list[str]:
        # chain 의 모든 joint (PELVIS root 제외).
        affected = set()
        for chain in T2M_KINEMATIC_CHAIN:
            for j in chain[1:]:
                affected.add(SMPL_22[j])
        return sorted(affected)

    def kdg_propagation_weights(self) -> dict[str, float]:
        # bone 보정은 chain 내부에서 forward 로 직접 적용 → propagation weight 명시
        # (descendant 가 affected 에 포함되므로 빈 dict 로 둠).
        return {}
