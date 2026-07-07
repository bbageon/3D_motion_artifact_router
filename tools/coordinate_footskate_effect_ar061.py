"""AR-061: CoordinateFootSkateCleanupTool representative-pool 비교 (frozen spec §6).

비교 arm (per generator, per motion):
  1. none            : baseline (no correction)
  2. yonly_large     : Y-only FootLockTool large — AR-058-4(P4) 재현 조건
                       (local 에 적용, both_feet, full range, ground=estimate_ground(local))
  3. coord_u{25,50,75,100} : CoordinateFootSkateCleanupTool (trajectory 에 적용,
                       continuous_u ∈ {0.25,0.5,0.75,1.0}, ground=estimate_ground(traj))
  4. oracle_grid     : per-prompt best-u (u_grid 내 foot_skate Δ 최소) — single-step
                       diagnostic ceiling (§3-16 oracle_type 명시; learned 아님)

Metric (AR-062 audit 준수 — gate-fire 인용 금지):
  - foot_skate_world (Category B, trajectory+estimate_ground) : primary, lower=better
  - float_mag        (Category B, local)                       : guard
  - artifact_total   (Category C proxy, local, Layer-A 3종 max 평균) : guard(참고)
  - bone_cv_max      (raw BoneLengthCVEvaluator score = max bone CV) : guard
  - correction magnitude / apply wall-clock                     : 보고

통계 단위 = prompt (seed 평균) → n=300/generator. bootstrap CI + paired Wilcoxon
(per-prompt Δ vs 0). FID(Category A) 는 본 snapshot 범위 밖 (별도 층 — P4-FID 방식).

CLI (motion3d env):
    python tools/coordinate_footskate_effect_ar061.py \
        --pool-root external_assets/protocol_rep_pool_seed20260608 \
        --output evals/snapshots/coordinate_footskate_effect_ar061_v1.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import CoordinateFootSkateCleanupTool, FootLockTool
from evaluators import DEFAULT_EVALUATORS, BoneLengthCVEvaluator
from tools.coords_protocol import estimate_ground, PELVIS
from tools.representative_pool_measure import foot_skate_world
from tools.physical_metric_g2_stress import float_mag

ARTIFACT_NAMES = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
U_GRID = (0.25, 0.5, 0.75, 1.0)
ARMS = ("yonly_large",) + tuple(f"coord_u{int(u * 100)}" for u in U_GRID)
METRICS = ("foot_skate_world", "float_mag", "artifact_total", "bone_cv_max")


def _measure(local, traj, art_evs, bone_cv_ev):
    return {
        "foot_skate_world": foot_skate_world(traj),
        "float_mag": float_mag(local),
        "artifact_total": float(np.mean([max((r.score for r in ev.evaluate(local)), default=0.0)
                                         for ev in art_evs])),
        "bone_cv_max": float(max((r.score for r in bone_cv_ev.evaluate(local)), default=0.0)),
    }


def _boot_ci(vals, rng, n=1000):
    nv = len(vals)
    bs = [float(np.mean(vals[rng.integers(0, nv, nv)])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]


def _wilcoxon_p(deltas: np.ndarray) -> float:
    """per-prompt Δ vs 0 paired Wilcoxon signed-rank (zero Δ 제외)."""
    nz = deltas[deltas != 0.0]
    if len(nz) < 10:
        return float("nan")
    return float(wilcoxon(nz).pvalue)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path,
                    default=REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608")
    ap.add_argument("--generators", nargs="+", default=["motiongpt", "mdm", "momask"])
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--limit", type=int, default=None,
                    help="pre-flight 용 meta 수 제한 (본 측정에선 사용 금지)")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "coordinate_footskate_effect_ar061_v1.json")
    args = ap.parse_args()
    args.pool_root = args.pool_root.resolve()

    rng = np.random.default_rng(args.seed)
    yonly = FootLockTool()
    coord = CoordinateFootSkateCleanupTool()
    art_evs = [ev for ev in DEFAULT_EVALUATORS if ev.name in ARTIFACT_NAMES]
    bone_cv_ev = BoneLengthCVEvaluator()

    results = {}
    for gen in args.generators:
        d = args.pool_root / gen
        metas = [json.load(open(p, encoding="utf-8")) for p in sorted(d.glob("*.json"))
                 if p.name != "_pool_summary.json"]
        if args.limit:
            metas = metas[:args.limit]
        # arm -> metric -> sample_id -> [Δ per seed]
        deltas = {a: {mt: defaultdict(list) for mt in METRICS} for a in ARMS}
        cmag = {a: [] for a in ARMS}
        wall = {a: [] for a in ARMS}
        n_correct_total = 0
        for m in metas:
            local = np.load(REPO_ROOT / m["local_npy"]).astype(np.float64)
            traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
            root = traj[:, PELVIS:PELVIS + 1, :]
            T = local.shape[0]
            before = _measure(local, traj, art_evs, bone_cv_ev)

            # --- arm: Y-only FootLock large (P4 재현: local 적용).
            t0 = time.perf_counter()
            cl, rep = yonly.apply(local, target_part="both_feet",
                                  target_joints=["LEFT_FOOT", "RIGHT_FOOT"],
                                  frame_range=(0, T - 1), strength="large",
                                  metadata={"ground_y": estimate_ground(local)})
            wall["yonly_large"].append(time.perf_counter() - t0)
            after = _measure(cl, cl + root, art_evs, bone_cv_ev)
            cmag["yonly_large"].append(rep.correction_magnitude)
            for mt in METRICS:
                deltas["yonly_large"][mt][m["sample_id"]].append(after[mt] - before[mt])

            # --- arms: coordinate tool (trajectory 적용).
            gy_traj = estimate_ground(traj)
            for u in U_GRID:
                a = f"coord_u{int(u * 100)}"
                t0 = time.perf_counter()
                ct, rep = coord.apply(traj, target_part="both_feet", target_joints=[],
                                      frame_range=(0, T - 1), strength="medium",
                                      metadata={"continuous_u": u, "ground_y": gy_traj,
                                                "coord_space": "trajectory"})
                wall[a].append(time.perf_counter() - t0)
                cl2 = ct - ct[:, PELVIS:PELVIS + 1, :]  # root 무수정 → root 동일
                after = _measure(cl2, ct, art_evs, bone_cv_ev)
                cmag[a].append(rep.correction_magnitude)
                if u == 1.0:
                    n_correct_total += rep.metadata["n_correct"]
                for mt in METRICS:
                    deltas[a][mt][m["sample_id"]].append(after[mt] - before[mt])

        # per-prompt (seed 평균) 집계.
        gen_res = {}
        per_prompt_fs = {}  # arm -> {sid: Δ} (oracle 용)
        for a in ARMS:
            ar = {}
            for mt in METRICS:
                pp = {sid: float(np.mean(v)) for sid, v in deltas[a][mt].items()}
                vals = np.array(list(pp.values()))
                ar[mt] = {"mean_delta": round(float(vals.mean()), 5),
                          "ci95": _boot_ci(vals, rng),
                          "frac_improved": round(float(np.mean(vals < 0)), 3),
                          "wilcoxon_p": (None if np.isnan(_wilcoxon_p(vals)) else
                                         round(_wilcoxon_p(vals), 6))}
                if mt == "foot_skate_world":
                    per_prompt_fs[a] = pp
            ar["correction_magnitude_mean"] = round(float(np.mean(cmag[a])), 5)
            ar["apply_ms_mean"] = round(float(np.mean(wall[a])) * 1000, 2)
            gen_res[a] = ar

        # oracle_grid: per-prompt best-u (foot_skate Δ 최소; 0 = no-op 허용 안 함 —
        # grid 강제 선택. no-op 포함 ceiling 은 best_or_zero 로 별도).
        sids = list(per_prompt_fs[ARMS[1]].keys())
        best = np.array([min(per_prompt_fs[f"coord_u{int(u * 100)}"][sid] for u in U_GRID)
                         for sid in sids])
        best_or_zero = np.minimum(best, 0.0)
        gen_res["oracle_grid"] = {
            "foot_skate_world": {"mean_delta": round(float(best.mean()), 5),
                                 "ci95": _boot_ci(best, rng),
                                 "frac_improved": round(float(np.mean(best < 0)), 3)},
            "foot_skate_world_with_stop": {"mean_delta": round(float(best_or_zero.mean()), 5),
                                           "ci95": _boot_ci(best_or_zero, rng)},
            "oracle_type": "single-step (per-prompt best-u within u_grid; STOP 변형 별도 표기)",
        }
        results[gen] = {"n_prompts": len(sids), "arms": gen_res,
                        "coord_u100_n_correct_segments_total": int(n_correct_total)}
        g = gen_res
        print(f"[{gen}] fs Δ  yonly={g['yonly_large']['foot_skate_world']['mean_delta']:+.5f} "
              + " ".join(f"u{int(u*100)}={g[f'coord_u{int(u*100)}']['foot_skate_world']['mean_delta']:+.5f}"
                         for u in U_GRID)
              + f" oracle={g['oracle_grid']['foot_skate_world']['mean_delta']:+.5f}")

    out = {
        "schema_version": "1.0.0",
        "record_type": "coordinate_footskate_effect",
        "board_id": "AR-061",
        "split_id": "protocol_rep_pool_seed20260608 (representative-300 full, 3 seeds/prompt)",
        "severity_versions": {ev.name: getattr(sys.modules[type(ev).__module__], "SEVERITY_VERSION", "n/a")
                              for ev in art_evs},
        "evaluator_config_hashes": {
            "FootLockTool": yonly.tool_class_hash()[:16],
            "CoordinateFootSkateCleanupTool": coord.tool_class_hash()[:16],
        },
        "action_space": {"action_space_type": "bounded_continuous_u", "u_grid": list(U_GRID),
                         "u_mapper_version": "coordinate_footskate_u_v1",
                         "stage": "AR-061 fixed-u sweep (learned policy 아님)"},
        "metric_provenance": {
            "foot_skate_world": "B (GMD/EDGE, trajectory+estimate_ground 10th pct) — primary",
            "float_mag": "B (local) — guard",
            "artifact_total": "C proxy (local, Layer-A 3종) — guard 참고용",
            "bone_cv_max": "B-계 raw (BoneLengthCVEvaluator score; gate-fire 아님 — AR-062 audit)",
            "FID": "Category A — 본 snapshot 범위 밖 (별도 층)",
        },
        "coordinate_note": "yonly=local 적용(P4 재현), coord=trajectory 적용(frozen spec §1-2); "
                           "측정은 두 arm 모두 동일 함수 (fs=traj, 나머지=local).",
        "stat_unit": "prompt (seed 평균; n=300/gen). lower=better; Δ<0=개선. bootstrap CI + Wilcoxon.",
        "results": results,
        "claim_boundary": "first snapshot — snapshot>=2 전 가설 status 근거 금지 (§3-9). "
                          "learned optimum 주장 없음. FID(A) 검증 전 외부 최종성능 claim 금지 (§3-20).",
        "grounding": ["GMD ICCV2023 (foot-skate metric)", "PhysDiff ICCV2023 (post-proc side effect)"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
