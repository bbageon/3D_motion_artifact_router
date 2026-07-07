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
from correction_tools.coordinate_footskate_cleanup_tool import CoordinateFootSkateCleanupTool
from correction_tools.foot_lock_tool import FootLockTool
from correction_tools.velocity_smoothing_tool import VelocitySmoothingTool

#: 본 저장소의 default correction tool registry. orchestrator 가 본 list 를 iter 하며 apply().
#:
#: NB (AR-061): CoordinateFootSkateCleanupTool 은 **의도적으로 본 default list 미포함** —
#: 기존 snapshot/oracle 의 candidate set (3-tool) 과의 비교 가능성 보존 (§3-25
#: candidate_trace / policy_contribution_baseline). tool 효과가 pool 평가로 검증된 뒤
#: registry 포함 여부를 routing-stage 결정으로 별도 진행 (그 시점에 §3-11/§4 재적용).
#: 또한 본 tool 은 trajectory(world) 좌표 전제라 local 기반 default loop 와 좌표 전제가
#: 다름 (frozen spec §1-2).
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
    "CoordinateFootSkateCleanupTool",
    "DEFAULT_CORRECTION_TOOLS",
]
