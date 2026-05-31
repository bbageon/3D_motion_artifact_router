"""Step 1 (사용자 directive 2026-05-31): G2 real stress profiling.

사용자 directive 박제:
> "RL 일반화 실험의 중심을 synthetic severe 에서 실제 generator output stress data 로 옮긴다.
>  synthetic severe / near_boundary 는 보조 진단 데이터 로 낮춤. 절차: G2 generated pool 실행
>  → evaluator 전체 실행 → artifact score 계산 → 상위 p80/p90/p95 stress case 추출 → normal/
>  stress/clean 분리. 목적: 인위적으로 망가뜨린 모션이 아니라, 실제 생성기가 만든 나쁜 모션에서
>  policy 를 학습/평가."

본 도구는 기존 G2 generated pool 의 각 motion 에 evaluator 전체 (artifact 3 + physical 5)
적용 → per-motion stress profile (artifact + physical) → percentile-based 분류 (normal /
stress / g2_clean_like) 산출.

분류 기준 (사용자 directive):
  - stress_p80: artifact total_score >= p80 (G2 pool 내 worst 20%)
  - stress_p90: >= p90 (top 10%)
  - stress_p95: >= p95 (top 5%)
  - g2_clean_like: <= p20 (bottom 20%, G2 도 가끔 clean 한 결과 생성)
  - normal: 그 사이 (60%)

artifact_total_score = mean(FootFloating_max, BoneLength_max, VelocityJitter_max).
physical_load = mean over 5 physical evaluators of (max / clean_p99_threshold) — gate 대비
정규화 부하 (1.0 = clean p99 임계 평균).

CLI:
    python -m tools.g2_real_stress_profile \
        --g2-batch-dir external_assets/g2_generated_hml3d_test300_seed20260527 \
        --calibration evals/snapshots/physical_gate_clean_calibration_v1.json \
        --output evals/snapshots/g2_real_stress_profile_v1.json

근거 (AGENTS.md §3-22): real-distribution evidence (AGENTS.md §3-17) — synthetic 은 controlled
diagnostic only, real generator output 이 최종 성능 evidence. PhysDiff (Yuan ICCV 2023) 의
physics-aware evaluation framework + HumanML3D (Guo CVPR 2022) 의 generator output 평가 spirit.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS, EvaluatorReport
from tools.harness_metadata import REAL_DISTRIBUTION, common_snapshot_metadata
from tools.rl2_build_training_data import ARTIFACT_EVALUATORS, PHYSICAL_EVALUATORS


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--calibration", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    parser.add_argument("--p-stress", type=str, default="80,90,95",
                        help="upper-tail percentile breakpoints for stress tiers")
    parser.add_argument("--p-clean", type=float, default=20.0,
                        help="lower-tail percentile for g2_clean_like")
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "g2_real_stress_profile_v1.json")
    args = parser.parse_args()

    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_p99 = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                if calib["summary"][n].get("n", 0) > 0}
    print(f"[INFO] gate p99 thresholds: { {k: round(v,4) for k,v in gate_p99.items()} }")
    evaluators = list(DEFAULT_EVALUATORS)
    gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)

    files = sorted(args.g2_batch_dir.glob("motion_*.npy"))
    print(f"[INFO] G2 pool: {len(files)} motions")
    per_motion = []
    for i, p in enumerate(files, 1):
        m = np.load(str(p)).astype(np.float64)
        art = {ev.name: round(_max_score(ev.evaluate(m)), 6) for ev in evaluators if ev.name in ARTIFACT_EVALUATORS}
        phy = {ev.name: round(_max_score(ev.evaluate(m)), 6) for ev in gate_evaluators if ev.name in PHYSICAL_EVALUATORS}
        # artifact_total_score = mean of 3 artifact maxes.
        art_total = float(np.mean([art[n] for n in ARTIFACT_EVALUATORS]))
        # physical_load = mean of (score / clean_p99), normalized by gate threshold (1.0 == p99 평균).
        loads = []
        for n in PHYSICAL_EVALUATORS:
            thr = gate_p99.get(n, 1.0)
            if thr > 1e-9:
                loads.append(phy[n] / thr)
        phy_load = float(np.mean(loads)) if loads else 0.0
        # Dominant evaluator (artifact-side) — which evaluator drives the score.
        dom_art = max(ARTIFACT_EVALUATORS, key=lambda n: art[n])
        dom_phy = max(PHYSICAL_EVALUATORS, key=lambda n: (phy[n] / max(gate_p99.get(n, 1.0), 1e-9)))
        # Physical "already violating" — any physical score > clean p99.
        any_phy_violating = any(phy[n] > gate_p99.get(n, float("inf")) for n in PHYSICAL_EVALUATORS)
        per_motion.append({
            "trial_id": p.stem,
            "artifact_scores": art,
            "physical_scores": phy,
            "artifact_total_score": round(art_total, 6),
            "physical_load": round(phy_load, 6),
            "dominant_artifact_evaluator": dom_art,
            "dominant_physical_evaluator": dom_phy,
            "any_phy_above_clean_p99": bool(any_phy_violating),
        })
        if i % 100 == 0:
            print(f"   {i}/{len(files)}")

    # Percentile-based classification.
    arts = np.array([r["artifact_total_score"] for r in per_motion])
    p_stress_list = sorted([float(x) for x in args.p_stress.split(",")])  # 80, 90, 95
    p_breaks = {
        f"p{int(round(args.p_clean))}": float(np.percentile(arts, args.p_clean)),
        **{f"p{int(round(p))}": float(np.percentile(arts, p)) for p in p_stress_list},
        "p50": float(np.percentile(arts, 50)),
    }
    print(f"[INFO] percentile breakpoints (artifact_total_score): "
          f"{ {k: round(v,4) for k,v in p_breaks.items()} }")
    # Classify each motion.
    clean_thr = p_breaks[f"p{int(round(args.p_clean))}"]
    stress_thrs = {f"stress_p{int(round(p))}": p_breaks[f"p{int(round(p))}"] for p in p_stress_list}
    primary_stress_thr = max(stress_thrs.values()) if False else p_breaks[f"p{int(round(min(p_stress_list)))}"]  # default = p80
    # Primary band classification: g2_clean_like | normal | stress.
    for r in per_motion:
        s = r["artifact_total_score"]
        if s <= clean_thr:
            band = "g2_clean_like"
        elif s >= primary_stress_thr:
            band = "stress"
        else:
            band = "normal"
        r["band"] = band
        # Tier within stress.
        r["stress_tier"] = None
        for tier_name, thr in sorted(stress_thrs.items(), key=lambda x: -x[1]):
            if s >= thr:
                r["stress_tier"] = tier_name
                break

    # Aggregate counts.
    from collections import Counter
    band_counts = Counter(r["band"] for r in per_motion)
    stress_tier_counts = Counter(r["stress_tier"] for r in per_motion if r["stress_tier"])
    dom_art_in_stress = Counter(r["dominant_artifact_evaluator"] for r in per_motion if r["band"] == "stress")
    dom_phy_in_stress = Counter(r["dominant_physical_evaluator"] for r in per_motion if r["band"] == "stress")
    n_phy_violating = sum(1 for r in per_motion if r["any_phy_above_clean_p99"])
    n_phy_violating_stress = sum(1 for r in per_motion if r["band"] == "stress" and r["any_phy_above_clean_p99"])

    out = {
        "schema_version": "1.0.0", "record_type": "g2_real_stress_profile",
        "task_id": "g2_real_stress_profile_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "g2_real_stress_profile_v1",
            oracle_type="profile", action_grid="N/A (profile only)",
            stage="Stage-2G-step1-profile", evidence_tier=[REAL_DISTRIBUTION],
            evaluators=evaluators, gate_evaluators=gate_evaluators,
        ),
        "directive": "G2 real generator output 의 artifact/physical 분포 profile + percentile-based 분류 (normal/stress/g2_clean_like). RL 일반화 실험 의 중심 데이터.",
        "g2_pool_dir": str(args.g2_batch_dir.relative_to(REPO_ROOT)),
        "n_motions": len(per_motion),
        "percentile_breakpoints": p_breaks,
        "classification_rule": {
            "g2_clean_like": f"artifact_total_score <= p{int(round(args.p_clean))} ({clean_thr:.4f})",
            "stress": f"artifact_total_score >= p{int(round(min(p_stress_list)))} ({primary_stress_thr:.4f})",
            "normal": "그 사이",
        },
        "stress_tier_thresholds": stress_thrs,
        "gate_p99": gate_p99,
        "summary": {
            "band_counts": dict(band_counts),
            "stress_tier_counts": dict(stress_tier_counts),
            "n_any_phy_above_clean_p99": n_phy_violating,
            "n_stress_any_phy_above_clean_p99": n_phy_violating_stress,
            "dominant_artifact_in_stress": dict(dom_art_in_stress),
            "dominant_physical_in_stress": dict(dom_phy_in_stress),
            "artifact_total_score_stats": {
                "mean": float(np.mean(arts)), "std": float(np.std(arts)),
                "min": float(np.min(arts)), "max": float(np.max(arts)),
                "p50": p_breaks["p50"],
            },
        },
        "per_motion": per_motion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== G2 Real Stress Profile (n={len(per_motion)}) ===")
    print(f"  band counts: {dict(band_counts)}")
    print(f"  stress tiers: {dict(stress_tier_counts)} (단, 상위 tier 만 1개 label)")
    print(f"  any phy > clean_p99: {n_phy_violating}/{len(per_motion)} ({n_phy_violating/len(per_motion)*100:.1f}%)")
    print(f"  stress + phy violating: {n_phy_violating_stress}/{band_counts.get('stress', 0)}")
    print(f"  dominant artifact eval in stress: {dict(dom_art_in_stress)}")
    print(f"  dominant physical eval in stress: {dict(dom_phy_in_stress)}")
    s = out["summary"]["artifact_total_score_stats"]
    print(f"  artifact_total_score: mean={s['mean']:.4f} std={s['std']:.4f} min={s['min']:.4f} max={s['max']:.4f} p50={s['p50']:.4f}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
