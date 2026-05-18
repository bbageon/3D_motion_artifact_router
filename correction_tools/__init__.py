"""Correction tool registry — 명세 §6.3.

각 tool 은 correction_tools.base.CorrectionTool 인터페이스를 구현하고
(corrected_motion, CorrectionReport) tuple 을 반환한다. 본 모듈은 (a) base
interface 와 (b) Week 3 prototype tool 3 종 (foot_lock / bone_projection /
velocity_smoothing) 의 registry 를 노출한다.

명세 §6.3 Correction Tool Registry.
AGENTS.md §3-2 Tool Registry 인터페이스 의무.
"""
from correction_tools.base import CorrectionReport, CorrectionTool
from correction_tools.bone_projection_tool import BoneProjectionTool
from correction_tools.foot_lock_tool import FootLockTool
from correction_tools.velocity_smoothing_tool import VelocitySmoothingTool

#: 본 저장소의 default correction tool registry. orchestrator 가 본 list 를 iter 하며 apply().
DEFAULT_CORRECTION_TOOLS: list[CorrectionTool] = [
    FootLockTool(),
    BoneProjectionTool(),
    VelocitySmoothingTool(),
]

__all__ = [
    "CorrectionTool",
    "CorrectionReport",
    "FootLockTool",
    "BoneProjectionTool",
    "VelocitySmoothingTool",
    "DEFAULT_CORRECTION_TOOLS",
]
