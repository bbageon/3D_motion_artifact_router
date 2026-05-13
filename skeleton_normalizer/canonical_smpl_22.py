"""Canonical SMPL 22-joint format 정의 — 단일 출처.

AGENTS.md §3-1 Canonical Motion Format 의무에 따라
모든 컴포넌트(generator output, evaluator input, correction tool input)는
본 22-joint 순서를 따른다.

HumanML3D 표준 + 이전 저장소의 tools/plot_3d_motion.py와 동일.
"""
from __future__ import annotations

# SMPL 22-joint 인덱스 (HumanML3D 표준)
SMPL_22: list[str] = [
    "PELVIS",         # 0
    "LEFT_HIP",       # 1
    "RIGHT_HIP",      # 2
    "SPINE1",         # 3
    "LEFT_KNEE",      # 4
    "RIGHT_KNEE",     # 5
    "SPINE2",         # 6
    "LEFT_ANKLE",     # 7
    "RIGHT_ANKLE",    # 8
    "SPINE3",         # 9
    "LEFT_FOOT",      # 10
    "RIGHT_FOOT",     # 11
    "NECK",           # 12
    "LEFT_COLLAR",    # 13
    "RIGHT_COLLAR",   # 14
    "HEAD",           # 15
    "LEFT_SHOULDER",  # 16
    "RIGHT_SHOULDER", # 17
    "LEFT_ELBOW",     # 18
    "RIGHT_ELBOW",    # 19
    "LEFT_WRIST",     # 20
    "RIGHT_WRIST",    # 21
]

NAME_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(SMPL_22)}

# HumanML3D 표준 5 kinematic chain
# (right leg / left leg / spine+head / left arm / right arm)
T2M_KINEMATIC_CHAIN: list[list[int]] = [
    [0, 2, 5, 8, 11],         # right leg: PELVIS → RIGHT_HIP → RIGHT_KNEE → RIGHT_ANKLE → RIGHT_FOOT
    [0, 1, 4, 7, 10],         # left leg
    [0, 3, 6, 9, 12, 15],     # spine + head: PELVIS → SPINE1 → SPINE2 → SPINE3 → NECK → HEAD
    [9, 14, 17, 19, 21],      # right arm: SPINE3 → RIGHT_COLLAR → RIGHT_SHOULDER → RIGHT_ELBOW → RIGHT_WRIST
    [9, 13, 16, 18, 20],      # left arm
]

CHAIN_LABELS: list[str] = ["right_leg", "left_leg", "spine_head", "right_arm", "left_arm"]

# Body part 그룹 (target_part 명명에 사용)
BODY_PART_GROUPS: dict[str, list[str]] = {
    "full_body": SMPL_22,
    "root_torso": ["PELVIS", "SPINE1", "SPINE2", "SPINE3", "NECK", "HEAD"],
    "left_arm": ["LEFT_COLLAR", "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"],
    "right_arm": ["RIGHT_COLLAR", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
    "left_leg": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT"],
    "right_leg": ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT"],
    "left_foot": ["LEFT_ANKLE", "LEFT_FOOT"],
    "right_foot": ["RIGHT_ANKLE", "RIGHT_FOOT"],
}

# fps default (HumanML3D)
DEFAULT_FPS: int = 20
