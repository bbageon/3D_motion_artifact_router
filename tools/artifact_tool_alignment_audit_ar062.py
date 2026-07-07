"""AR-062: artifact <-> tool definition alignment audit — controlled diagnostic.

두 가지 **구조적 vacuity** (정의상 fire 불가) 를 결정적으로 재현한다:

  (1) SkateEvaluator (gate): 내부 contact 추정 = xz_vel <= 0.02 (physical_gate.py
      DEFAULT_V_CONTACT_THRESH), sliding 판정 = xz_vel > 0.05 (DEFAULT_SKATE_THRESHOLD).
      sliding = contact & (xz_vel > 0.05) 인데 contact ⊆ (xz_vel <= 0.02) 이므로
      **교집합 = 공집합** — contact_labels 없이는 어떤 motion 도 fire 불가.
      -> HumanML3D clean p99 = 0.0 (physical_gate_clean_calibration_v1) 과
         representative pool Skate_gate_fire = 0.0 (3 generator 전부) 는
         "skate 없음" 의 증거가 아니라 vacuous-by-construction.

  (2) PenetrateEvaluator (gate): 내부 ground = min-Y (전 joint) — 정의상 어떤 joint 도
      min 아래에 있을 수 없어 penetrate = joint_y < ground - eps 는 **불가능**.
      ground_y 외부 전달 시에만 유의미.

본 스크립트는 성능 결론이 아니라 **정의(논리) 재현** — §3-17 controlled diagnostic.
동일 synthetic motion 에서 v2 정의 (수직속도 기반 contact, AR-058-3b) 는 skate 를
정상 검출함을 함께 보인다.

CLI (motion3d env):
    python tools/artifact_tool_alignment_audit_ar062.py \
        --output evals/snapshots/artifact_tool_alignment_audit_ar062_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from evaluators import PenetrateEvaluator, SkateEvaluator

# v2 contact/skate 정의 (AR-058-3b/3f 와 동일 — self-contained, 상수 재기재).
V2_CONTACT_H, V2_CONTACT_VY, V2_SKATE_DXZ = 0.05, 0.035, 0.025
LEFT_FOOT, RIGHT_FOOT = 10, 11


def v2_flags(motion: np.ndarray, foot: int, ground: float):
    fy = motion[:, foot, 1] - ground
    fxz = motion[:, foot, :][:, [0, 2]]
    disp = np.linalg.norm(np.diff(fxz, axis=0), axis=1)
    disp = np.concatenate([disp, [disp[-1] if len(disp) else 0.0]])
    vy = np.abs(np.concatenate([np.diff(fy), [0.0]]))
    contact = (fy <= V2_CONTACT_H) & (vy <= V2_CONTACT_VY)
    skate = contact & (disp >= V2_SKATE_DXZ)
    return contact, skate


def build_skate_motion(T: int = 30, slide: float = 0.06) -> np.ndarray:
    """발이 바닥(y=0)에 붙은 채 slide m/frame 수평 이동 — 명백한 foot-skate."""
    m = np.zeros((T, 22, 3))
    m[:, :, 1] = 1.0
    for f in (LEFT_FOOT, RIGHT_FOOT):
        m[:, f, 1] = 0.0
        m[:, f, 0] = np.arange(T) * slide
    m[:, 7, 1] = 0.1
    m[:, 8, 1] = 0.1
    return m


def build_penetration_motion(T: int = 30, depth: float = 0.08) -> np.ndarray:
    """RIGHT_FOOT 이 frame 10~19 에서 ground(0) 아래 -depth 로 관통."""
    m = np.zeros((T, 22, 3))
    m[:, :, 1] = 1.0
    m[:, LEFT_FOOT, 1] = 0.0
    m[:, RIGHT_FOOT, 1] = 0.0
    m[10:20, RIGHT_FOOT, 1] = -depth
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "artifact_tool_alignment_audit_ar062_v1.json")
    args = ap.parse_args()

    T = 30
    # ---- (1) SkateEvaluator vacuity.
    m_skate = build_skate_motion(T)
    sk = SkateEvaluator()
    r_internal = sk.evaluate(m_skate)
    labels = np.zeros((T, 22), dtype=bool)
    labels[:, LEFT_FOOT] = True
    labels[:, RIGHT_FOOT] = True
    r_labels = sk.evaluate(m_skate, contact_labels=labels)
    contact, skate = v2_flags(m_skate, LEFT_FOOT, 0.0)

    # ---- (2) PenetrateEvaluator vacuity.
    m_pen = build_penetration_motion(T)
    pen = PenetrateEvaluator()
    r_pen_internal = pen.evaluate(m_pen)
    r_pen_ground = pen.evaluate(m_pen, ground_y=0.0)

    out = {
        "schema_version": "1.0.0",
        "record_type": "artifact_tool_alignment_audit",
        "board_id": "AR-062",
        "evidence_tier": "controlled_diagnostic (synthetic; 정의/논리 재현 — 성능 결론 아님, §3-17)",
        "skate_evaluator_vacuity": {
            "claim": "내부 contact(xz_vel<=0.02) 와 sliding(xz_vel>0.05) 교집합 공집합 — contact_labels 없이 fire 불가",
            "synthetic": "T=30, 두 발 ground(y=0) 접지 + 0.06 m/frame 수평 slide (frame 전체가 skate)",
            "internal_contact_reports": len(r_internal),
            "external_labels_reports": len(r_labels),
            "external_labels_scores": [round(float(r.score), 4) for r in r_labels],
            "v2_flags_same_motion": {"contact": int(contact.sum()), "skate": int(skate.sum()), "T": T},
            "code_refs": [
                "evaluators/physical_gate.py:52 DEFAULT_V_CONTACT_THRESH=0.02",
                "evaluators/physical_gate.py:51 DEFAULT_SKATE_THRESHOLD=0.05",
                "evaluators/physical_gate.py:246 sliding = (vel > skate_threshold) & contact",
                "tools/physical_gate_clean_calibration.py:86 ev.evaluate(motion) — labels/ground 미전달",
            ],
            "affected_prior_records": [
                "evals/snapshots/physical_gate_clean_calibration_v1.json (Skate p99=0.0 = vacuous)",
                "evals/snapshots/representative_pool_measure_v1.json Skate_gate_fire=0.0 (3 gen)",
                "AR-043/AR-044 의 Skate_gate 컬럼 — 'skate 없음' 근거로 인용 금지",
            ],
        },
        "penetrate_evaluator_vacuity": {
            "claim": "내부 ground = min-Y(전 joint) — joint_y < ground - eps 는 정의상 불가능",
            "synthetic": "T=30, RIGHT_FOOT 이 frame 10~19 에서 -0.08m 관통 (ground=0)",
            "internal_ground_reports": len(r_pen_internal),
            "external_ground_reports": len(r_pen_ground),
            "external_ground_scores": [round(float(r.score), 4) for r in r_pen_ground],
            "code_refs": [
                "evaluators/physical_gate.py:75-77 _estimate_ground_y = min-Y 전 joint",
                "evaluators/physical_gate.py:124 penetrate_mask = joint_y < (ground_y - eps)",
            ],
            "affected_prior_records": [
                "evals/snapshots/physical_gate_clean_calibration_v1.json (Penetrate p99=0.0 = vacuous)",
                "AR-043/AR-044 의 Penetrate 유병률 0% — '관통 없음' 근거로 인용 금지",
            ],
        },
        "unaffected_conclusions": [
            "foot_skate_world (trajectory + estimate_ground 10th pct, Category B) 는 건전 — MDM 2x 등 AR-044 headline 유지",
            "float_mag / FID / R-Precision 기반 결론 유지",
        ],
        "full_matrix_doc": ".claude/docs/findings/artifact_tool_alignment_audit.md",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"(1) SkateEvaluator   internal={len(r_internal)} reports / external labels={len(r_labels)} "
          f"(scores {[round(float(r.score), 3) for r in r_labels]}) / v2 skate={int(skate.sum())}/{T}")
    print(f"(2) PenetrateEval    internal={len(r_pen_internal)} reports / ground_y=0.0 -> "
          f"{len(r_pen_ground)} (scores {[round(float(r.score), 3) for r in r_pen_ground]})")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
