"""AR-070: Bone 축 P1+P4 chain 측정 (+ cleanup→bone 조합 — A/B v2 설계 근거).

사용자 directive (2026-07-08): "BoneCV 까지 적용해서 다시 A/B 대조. Bone P1-P5 우선 진행."
+ 사용자 지각 관찰: 보정본에서 "발이 짧아져서 땅으로 들어간 듯한 느낌" — coord cleanup
propagation (발 100% / 무릎 20%) 의 다리 압축과 기전 일치.

Arms (per motion, representative pool 3 generators):
  1. baseline       : 무보정
  2. bone_large     : BoneProjectionTool (left_leg + right_leg 순차, strength=large)
  3. cleanup_u75    : CoordinateFootSkateCleanupTool u=0.75 (A/B v1 과 동일 설정)
  4. combo          : cleanup_u75 → bone_large  (기존 tool 만의 조합 — A/B v2 후보)

Metric (state 별): bone_cv_max / bone_cv_mean / **leg_cv_max** (다리 8개 bone 만 —
사용자 지각의 표적) / foot_skate_world (trajectory) / float_mag / artifact_total.
P1 = baseline 분포 (percentile + worst-bone 부위 빈도). P4 = fixed bone tool 의
효과·부작용 + 조합이 (a) cleanup 의 fs 이득을 보존하고 (b) 추가 bone 왜곡을
제거하는지 — A/B v2 를 기존 tool 조합으로 갈지 AR-064 신규 구현으로 갈지 결정.

좌표: 모든 tool 을 trajectory 에서 적용 (bone projection 은 평행이동 불변이라 동일;
PELVIS 는 두 tool 모두 무수정 → root 보존). 측정은 fs=trajectory, 나머지=local.

CLI (motion3d env):
    python tools/bone_chain_p1p4_ar070.py --limit 8   # preflight
    python tools/bone_chain_p1p4_ar070.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import BoneProjectionTool, CoordinateFootSkateCleanupTool
from evaluators import DEFAULT_EVALUATORS, BoneLengthCVEvaluator
from tools.coords_protocol import estimate_ground, PELVIS
from tools.representative_pool_measure import foot_skate_world
from tools.physical_metric_g2_stress import float_mag

ARTIFACT_NAMES = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
ARMS = ("bone_large", "cleanup_u75", "combo")
METRICS = ("bone_cv_max", "bone_cv_mean", "leg_cv_max", "foot_skate_world", "float_mag", "artifact_total")
U_CLEANUP = 0.75  # A/B v1 과 동일 (비교 가능성)
#: BoneLengthCVEvaluator.bones 의 앞 8개 = 다리 (right leg 4 + left leg 4) — T2M chain 순서.
N_LEG_BONES = 8
#: worst_bone (a,b) → 부위 분류.
LEG_JOINTS = {1, 2, 4, 5, 7, 8, 10, 11}
ARM_JOINTS = {13, 14, 16, 17, 18, 19, 20, 21}


def _bone_part(a: int, b: int) -> str:
    if a in LEG_JOINTS or b in LEG_JOINTS:
        return "leg"
    if a in ARM_JOINTS or b in ARM_JOINTS:
        return "arm"
    return "spine_head"


def _boot_ci(vals, rng, n=1000):
    nv = len(vals)
    bs = [float(np.mean(vals[rng.integers(0, nv, nv)])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path,
                    default=REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608")
    ap.add_argument("--generators", nargs="+", default=["motiongpt", "mdm", "momask"])
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "bone_chain_p1p4_ar070_v1.json")
    args = ap.parse_args()
    args.pool_root = args.pool_root.resolve()

    rng = np.random.default_rng(args.seed)
    bone_tool = BoneProjectionTool()
    cleanup = CoordinateFootSkateCleanupTool()
    bone_ev = BoneLengthCVEvaluator()
    art_evs = [ev for ev in DEFAULT_EVALUATORS if ev.name in ARTIFACT_NAMES]

    def bone_stats(local):
        r = bone_ev.evaluate(local)[0]
        cvs = np.asarray(r.metadata["all_bone_cvs"], dtype=float)
        return {"bone_cv_max": float(r.score), "bone_cv_mean": float(r.metadata["mean_cv"]),
                "leg_cv_max": float(cvs[:N_LEG_BONES].max()),
                "worst_bone": tuple(r.metadata["worst_bone"])}

    def measure(traj):
        local = traj - traj[:, PELVIS:PELVIS + 1, :]
        bs = bone_stats(local)
        return {**{k: bs[k] for k in ("bone_cv_max", "bone_cv_mean", "leg_cv_max")},
                "foot_skate_world": foot_skate_world(traj),
                "float_mag": float_mag(local),
                "artifact_total": float(np.mean([max((r.score for r in ev.evaluate(local)), default=0.0)
                                                 for ev in art_evs]))}, bs["worst_bone"]

    def apply_bone(traj):
        out = traj
        for part in ("left_leg", "right_leg"):
            T = out.shape[0]
            out, _ = bone_tool.apply(out, target_part=part, target_joints=[],
                                     frame_range=(0, T - 1), strength="large")
        return out

    def apply_cleanup(traj):
        T = traj.shape[0]
        out, _ = cleanup.apply(traj, target_part="both_feet", target_joints=[],
                               frame_range=(0, T - 1), strength="medium",
                               metadata={"continuous_u": U_CLEANUP,
                                         "ground_y": estimate_ground(traj),
                                         "coord_space": "trajectory"})
        return out

    results = {}
    for gen in args.generators:
        d = args.pool_root / gen
        metas = [json.load(open(p, encoding="utf-8")) for p in sorted(d.glob("*.json"))
                 if p.name != "_pool_summary.json"]
        if args.limit:
            metas = metas[:args.limit]
        base_vals = {mt: defaultdict(list) for mt in METRICS}
        worst_parts = Counter()
        deltas = {a: {mt: defaultdict(list) for mt in METRICS} for a in ARMS}
        for m in metas:
            traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
            before, wb = measure(traj)
            worst_parts[_bone_part(*wb)] += 1
            for mt in METRICS:
                base_vals[mt][m["sample_id"]].append(before[mt])
            states = {
                "bone_large": apply_bone(traj),
                "cleanup_u75": apply_cleanup(traj),
            }
            states["combo"] = apply_bone(states["cleanup_u75"])
            for a in ARMS:
                after, _ = measure(states[a])
                for mt in METRICS:
                    deltas[a][mt][m["sample_id"]].append(after[mt] - before[mt])

        gen_res = {"n_prompts": len(base_vals[METRICS[0]])}
        # ---- P1: baseline 분포.
        p1 = {}
        for mt in ("bone_cv_max", "bone_cv_mean", "leg_cv_max", "foot_skate_world"):
            pp = np.array([float(np.mean(v)) for v in base_vals[mt].values()])
            p1[mt] = {"mean": round(float(pp.mean()), 5), "ci95": _boot_ci(pp, rng),
                      "p50": round(float(np.percentile(pp, 50)), 5),
                      "p90": round(float(np.percentile(pp, 90)), 5),
                      "p99": round(float(np.percentile(pp, 99)), 5)}
        total_wb = sum(worst_parts.values())
        p1["worst_bone_part_frequency"] = {k: round(v / total_wb, 3) for k, v in worst_parts.items()}
        gen_res["P1_baseline"] = p1
        # ---- P4: arm 별 Δ.
        p4 = {}
        for a in ARMS:
            ar = {}
            for mt in METRICS:
                pp = np.array([float(np.mean(v)) for v in deltas[a][mt].values()])
                ar[mt] = {"mean_delta": round(float(pp.mean()), 5), "ci95": _boot_ci(pp, rng),
                          "frac_improved": round(float(np.mean(pp < 0)), 3)}
            p4[a] = ar
        gen_res["P4_deltas"] = p4
        results[gen] = gen_res
        print(f"[{gen}] P1 bone_cv_max mean={p1['bone_cv_max']['mean']:.4f} "
              f"leg_cv_max mean={p1['leg_cv_max']['mean']:.4f} worst-part={dict(worst_parts)}")
        for a in ARMS:
            r = p4[a]
            print(f"   {a:<12} boneCVmax {r['bone_cv_max']['mean_delta']:+.4f} | "
                  f"legCV {r['leg_cv_max']['mean_delta']:+.4f} | fs {r['foot_skate_world']['mean_delta']:+.5f} | "
                  f"float {r['float_mag']['mean_delta']:+.4f}")

    out = {
        "schema_version": "1.0.0", "record_type": "bone_chain_p1p4", "board_id": "AR-070",
        "split_id": "protocol_rep_pool_seed20260608 (representative-300 full, 3 seeds/prompt)",
        "arms": {"bone_large": "BoneProjectionTool left_leg+right_leg 순차, strength=large (ref=per-motion median)",
                 "cleanup_u75": "CoordinateFootSkateCleanupTool u=0.75 (A/B v1 동일)",
                 "combo": "cleanup_u75 -> bone_large (기존 tool 조합; KDG 참고 — 순차 2-step 진단, orchestrator 아님)"},
        "metric_provenance": {
            "bone_cv_*": "raw BoneLengthCVEvaluator (gate-fire 아님 — AR-062); ref=per-motion median (자기참조 정의 명시)",
            "leg_cv_max": "다리 8 bone 한정 max CV — 사용자 지각('발이 짧아져 땅으로') 표적 지표",
            "foot_skate_world": "B (trajectory+estimate_ground)", "float_mag": "B", "artifact_total": "C proxy"},
        "stat_unit": "prompt (seed 평균, n=300/gen). lower=better; Δ<0=개선.",
        "results": results,
        "perceptual_context": "사용자 b1 관찰 (2026-07-08): cleanup 보정본에서 '발이 짧아져 땅으로 들어간 느낌' — propagation 다리 압축 기전과 일치. 본 측정은 그 관찰의 정량 대응물(leg_cv)과 조합 처방의 효과를 확인.",
        "decision_rule_for_ab_v2": "combo 가 (a) cleanup 의 fs 이득 대부분 보존 AND (b) bone_cv/leg_cv 추가 왜곡 제거(Δ<=0) 면 -> A/B v2 는 기존 tool 조합으로. 아니면 AR-064 (foot-anchored re-projection) 구현 후.",
        "claim_boundary": "물리 지표 측정 — 지각 결론 아님 (A/B v2 로만). bone ref=자기 median 이라 'GT 복원' 아님.",
        "grounding": ["HuMoR ICCV2021 (bone consistency)", "PhysDiff ICCV2023 (post-proc side effect)", "GMD ICCV2023 (fs metric)"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
