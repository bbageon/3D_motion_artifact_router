"""CoordinateFootSkateCleanupTool — AR-061 prototype (frozen spec 구현).

Frozen spec: `.claude/docs/design/coordinate_footskate_cleanup_tool_frozen_spec.md` (AR-060).
Y-only FootLockTool 의 mixed effect (artifact↓ vs foot_skate↑, AR-058-4) 후속:
접지(contact) 중 발이 수평(X/Z)으로 미끄러지는 것이 문제이므로, contact segment 를
탐지해 그 구간에서만 발을 segment anchor (median X/Z, ground Y) 로 부분 고정한다.

Y-only 와의 결정적 차이 (spec §3-3):
  - 축: Y only → **X/Z/Y**
  - 대상 frame: frame_range 전체 → **탐지된 skate contact segment 만**
  - 대상 발: 항상 both → **segment 별 skating foot**
  - anchor 없음 → **segment median X/Z**
  - blend 없음 → **enter/exit smoothstep**

좌표계 전제 (spec §1-2): 본 tool 은 **motion_trajectory (world, root 포함)** 에서만
동작한다. motion_local (PELVIS=원점) 은 root 이동이 제거돼 X/Z anchor 가 무의미.
`metadata["coord_space"]` 가 "trajectory" 가 아니면 ValueError.

leg-chain propagation (foot/ankle/knee/hip = 1.0/0.5/0.2/0.1) 은 true IK 가 아닌
좌표 근사 — engineering heuristic (spec §3-4). BoneLengthCV guard 로 감시.

명세 §6.3 CorrectionTool 인터페이스 + KDG affected joints 의무 (AGENTS.md §3-2).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from correction_tools.base import CorrectionTool, CorrectionReport, Strength
from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX

#: v2 contact/skate 정의 상수 (AR-058-3b/3f/3g 와 동일 — spec §3-1).
DEFAULT_CONTACT_H = 0.05    # 접지 높이 임계 (m, ground 위)
DEFAULT_CONTACT_VY = 0.035  # 접지 수직속도 임계 (m/frame)
DEFAULT_SKATE_DXZ = 0.025   # skate 수평변위 임계 (m/frame)

#: ground = lower-foot-height 하위 percentile (tools/coords_protocol.estimate_ground 와
#: 동일 semantics; §7 디렉토리 책임상 tools/ 를 import 하지 않고 여기 재정의).
DEFAULT_GROUND_PERCENTILE = 10.0

#: strength token → u ∈ [0,1] (spec §4; AGENTS.md §3-21 bounded_continuous_u,
#: u-mapper version = coordinate_footskate_u_v1).
STRENGTH_U: dict[str, float] = {
    # 3-level.
    "small": 0.25, "medium": 0.5, "large": 0.75,
    # 5-level.
    "xsmall": 0.2, "small5": 0.4, "medium5": 0.6, "large5": 0.8, "xlarge": 1.0,
}
U_MAPPER_VERSION = "coordinate_footskate_u_v1"

#: leg chain (foot → ankle → knee → hip) propagation weight (spec §3-4, NOT IK).
PROPAGATION_CHAIN: dict[str, list[tuple[str, float]]] = {
    "LEFT_FOOT": [("LEFT_FOOT", 1.0), ("LEFT_ANKLE", 0.5), ("LEFT_KNEE", 0.2), ("LEFT_HIP", 0.1)],
    "RIGHT_FOOT": [("RIGHT_FOOT", 1.0), ("RIGHT_ANKLE", 0.5), ("RIGHT_KNEE", 0.2), ("RIGHT_HIP", 0.1)],
}

DIRECT_JOINTS: list[str] = ["LEFT_FOOT", "RIGHT_FOOT"]
PROPAGATION_WEIGHTS: dict[str, float] = {
    "LEFT_ANKLE": 0.5, "RIGHT_ANKLE": 0.5,
    "LEFT_KNEE": 0.2, "RIGHT_KNEE": 0.2,
    "LEFT_HIP": 0.1, "RIGHT_HIP": 0.1,
}


def _estimate_ground(motion: np.ndarray, percentile: float) -> float:
    """ground_y = frame별 lower-foot-height 의 하위 percentile (NOT min-Y).

    tools/coords_protocol.estimate_ground 와 동일 정의 (penetration outlier robust).
    """
    lf, rf = NAME_TO_IDX["LEFT_FOOT"], NAME_TO_IDX["RIGHT_FOOT"]
    lower = np.minimum(motion[:, lf, 1], motion[:, rf, 1])
    return float(np.percentile(lower, percentile))


def _v2_flags(motion: np.ndarray, foot_idx: int, ground_y: float,
              contact_h: float, contact_vy: float, skate_dxz: float,
              ) -> tuple[np.ndarray, np.ndarray]:
    """v2 contact/skate mask (padding convention = AR-058-3f 도구와 동일)."""
    fy = motion[:, foot_idx, 1] - ground_y
    fxz = motion[:, foot_idx, :][:, [0, 2]]
    disp = np.linalg.norm(np.diff(fxz, axis=0), axis=1)
    disp = np.concatenate([disp, [disp[-1] if len(disp) else 0.0]])
    vy = np.abs(np.concatenate([np.diff(fy), [0.0]]))
    contact = (fy <= contact_h) & (vy <= contact_vy)
    skate = contact & (disp >= skate_dxz)
    return contact, skate


def _contact_segments(contact: np.ndarray, max_gap: int = 1) -> list[tuple[int, int]]:
    """contact runs → gap ≤ max_gap frame 은 같은 segment 로 병합 (spec §3-1)."""
    runs: list[tuple[int, int]] = []
    s: Optional[int] = None
    for i, f in enumerate(contact.astype(bool)):
        if f and s is None:
            s = i
        elif not f and s is not None:
            runs.append((s, i - 1)); s = None
    if s is not None:
        runs.append((s, len(contact) - 1))
    if not runs:
        return []
    merged = [runs[0]]
    for (a, b) in runs[1:]:
        if a - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def _blend_weights(seg_len: int, blend_frames: int) -> np.ndarray:
    """enter/exit smoothstep (3s²−2s³) 곱 — interior=1, 짧은 segment 는 peak<1 (spec §3-5)."""
    idx = np.arange(seg_len, dtype=np.float64)
    b = max(int(blend_frames), 1)
    s_in = np.clip((idx + 1.0) / b, 0.0, 1.0)
    s_out = np.clip((seg_len - idx) / b, 0.0, 1.0)
    smooth = lambda x: 3.0 * x ** 2 - 2.0 * x ** 3  # noqa: E731
    return smooth(s_in) * smooth(s_out)


class CoordinateFootSkateCleanupTool(CorrectionTool):
    """Contact-aware coordinate foot-skate cleanup (frozen spec §2 파이프라인).

    target_part: 'left_foot' / 'right_foot' / 'both_feet' (후보 발 제한;
    'both_feet' 면 segment 별로 skating foot 자동 선택).
    frame_range: tool 이 동작 허용된 window — 이 안에서만 segment 탐지·보정.
    """

    name = "CoordinateFootSkateCleanupTool"

    def __init__(
        self,
        contact_h: float = DEFAULT_CONTACT_H,
        contact_vy: float = DEFAULT_CONTACT_VY,
        skate_dxz: float = DEFAULT_SKATE_DXZ,
        min_contact_len: int = 3,
        min_contact_ratio: float = 0.6,
        min_skate_frac: float = 0.2,
        blend_frames: int = 3,
        ground_percentile: float = DEFAULT_GROUND_PERCENTILE,
    ) -> None:
        self.contact_h = contact_h
        self.contact_vy = contact_vy
        self.skate_dxz = skate_dxz
        self.min_contact_len = min_contact_len
        self.min_contact_ratio = min_contact_ratio
        self.min_skate_frac = min_skate_frac
        self.blend_frames = blend_frames
        self.ground_percentile = ground_percentile

    # ------------------------------------------------------------------ apply
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
        meta = metadata or {}

        # spec §1-2: world trajectory 전제 (local 에서 X/Z anchor 는 무의미).
        coord_space = str(meta.get("coord_space", "trajectory"))
        if coord_space != "trajectory":
            raise ValueError(
                f"coord_space={coord_space!r}: CoordinateFootSkateCleanupTool 은 "
                "motion_trajectory (world) 에서만 동작한다 (frozen spec §1-2). "
                "motion_local 입력 금지."
            )

        T = motion.shape[0]
        start = max(0, int(frame_range[0]))
        end = min(T - 1, int(frame_range[1]))

        ground_y = float(meta.get("ground_y", _estimate_ground(motion, self.ground_percentile)))

        # u: continuous 우선, 없으면 token map (spec §4).
        if "continuous_u" in meta:
            u = float(np.clip(float(meta["continuous_u"]), 0.0, 1.0))
            u_source = "continuous"
        else:
            u = STRENGTH_U.get(strength, STRENGTH_U["medium"])
            u_source = "token"

        # 후보 발 (spec §1-1).
        if target_joints:
            feet = [j for j in target_joints if j in ("LEFT_FOOT", "RIGHT_FOOT")]
        elif target_part == "left_foot":
            feet = ["LEFT_FOOT"]
        elif target_part == "right_foot":
            feet = ["RIGHT_FOOT"]
        else:  # both_feet / unknown → 양 발 후보, segment 별 자동 선택
            feet = ["LEFT_FOOT", "RIGHT_FOOT"]

        corrected = motion.copy()
        seg_records: list[dict[str, Any]] = []
        ambiguous_records: list[dict[str, Any]] = []
        n_skip = 0
        foot_deltas: list[float] = []
        modified: set[str] = set()

        for fname in feet:
            fidx = NAME_TO_IDX[fname]
            contact, skate = _v2_flags(motion, fidx, ground_y,
                                       self.contact_h, self.contact_vy, self.skate_dxz)
            fy = motion[:, fidx, 1] - ground_y
            for (s0, e0) in _contact_segments(contact):
                # frame_range window 로 clip (spec §2).
                s, e = max(s0, start), min(e0, end)
                if e < s:
                    continue
                seg_len = e - s + 1
                seg_contact = contact[s:e + 1]
                seg_skate = skate[s:e + 1]
                contact_ratio = float(np.mean(seg_contact))
                skate_frac = float(np.mean(seg_skate))
                mean_fy = float(np.mean(fy[s:e + 1]))

                base_valid = (
                    seg_len >= self.min_contact_len
                    and contact_ratio >= self.min_contact_ratio
                    and mean_fy <= self.contact_h
                )
                if not base_valid:
                    n_skip += 1
                    continue
                if skate_frac < self.min_skate_frac:
                    if skate_frac > 0.0:
                        # 애매 — 보정하지 않고 report 만 (spec §3-1 ambiguous).
                        ambiguous_records.append({
                            "foot": fname, "start": int(s), "end": int(e),
                            "skate_frac": round(skate_frac, 4),
                        })
                    else:
                        n_skip += 1  # 깨끗한 정상 접지 — 건드리지 않음.
                    continue

                # ---- correct segment: anchor = contact frame 들의 median X/Z (spec §3-2).
                contact_frames = np.arange(s, e + 1)[seg_contact]
                anchor_x = float(np.median(motion[contact_frames, fidx, 0]))
                anchor_z = float(np.median(motion[contact_frames, fidx, 2]))
                target = np.array([anchor_x, ground_y, anchor_z], dtype=np.float64)

                w = _blend_weights(seg_len, self.blend_frames)  # [seg_len]
                cur = corrected[s:e + 1, fidx, :]               # [seg_len, 3]
                delta = (u * w)[:, None] * (target[None, :] - cur)
                # leg-chain propagation (spec §3-4, engineering heuristic).
                for (jname, jw) in PROPAGATION_CHAIN[fname]:
                    corrected[s:e + 1, NAME_TO_IDX[jname], :] += jw * delta
                    modified.add(jname)
                foot_deltas.extend(np.linalg.norm(delta, axis=1).tolist())

                seg_records.append({
                    "foot": fname, "start": int(s), "end": int(e),
                    "skate_frac": round(skate_frac, 4),
                    "anchor_xz": [round(anchor_x, 5), round(anchor_z, 5)],
                })

        correction_magnitude = float(np.mean(foot_deltas)) if foot_deltas else 0.0
        report = CorrectionReport(
            tool=self.name,
            target_part=target_part,
            frame_range=(int(start), int(end)),
            strength=strength,
            modified_joints=sorted(modified),
            correction_magnitude=correction_magnitude,
            metadata={
                "ground_y": ground_y,
                "coord_space": coord_space,
                "u": u,
                "u_source": u_source,
                "u_mapper_version": U_MAPPER_VERSION,
                "n_correct": len(seg_records),
                "n_ambiguous": len(ambiguous_records),
                "n_skip": n_skip,
                "segments": seg_records,
                "ambiguous_segments": ambiguous_records,
                "params": {
                    "contact_h": self.contact_h, "contact_vy": self.contact_vy,
                    "skate_dxz": self.skate_dxz, "min_contact_len": self.min_contact_len,
                    "min_contact_ratio": self.min_contact_ratio,
                    "min_skate_frac": self.min_skate_frac,
                    "blend_frames": self.blend_frames,
                    "ground_percentile": self.ground_percentile,
                },
            },
        )
        return corrected, report

    # ------------------------------------------------------------------- KDG
    def kdg_affected_joints(self) -> list[str]:
        return list(DIRECT_JOINTS)

    def kdg_propagation_weights(self) -> dict[str, float]:
        return dict(PROPAGATION_WEIGHTS)
