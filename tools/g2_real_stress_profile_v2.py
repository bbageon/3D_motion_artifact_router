"""Step 1 v2 (사용자 directive 2026-05-31): group-aware G2 real stress profile.

사용자 directive 박제:
> "전체 pool 에서 stress top p 를 뽑지 말고 각 motion group 내부에서 stress 를 뽑기.
>  walking 중심 분포에 과적합 방지."

v1 (g2_real_stress_profile.py) 의 변경점:
  - TWO pool directories 합산 (existing 300 + balanced 300 = 600).
  - 각 motion 의 prompt 를 motion-group classifier 로 분류.
  - **percentile threshold 가 motion_group 별로 계산** (pool-wide 아님).
  - band 분류 = 각 group 내부의 p20 / p80 / p90 / p95 기준.

CLI:
    python -m tools.g2_real_stress_profile_v2 \
        --pool-dirs external_assets/g2_generated_hml3d_test300_seed20260527,external_assets/g2_generated_hml3d_balanced300_seed20260531 \
        --output evals/snapshots/g2_real_stress_profile_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS, EvaluatorReport
from tools.harness_metadata import REAL_DISTRIBUTION, common_snapshot_metadata
from tools.rl2_build_training_data import ARTIFACT_EVALUATORS, PHYSICAL_EVALUATORS
from tools.g2_balanced_prompt_selector import classify, GROUPS_ORDER


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _load_motion_with_meta(npy: Path):
    """Returns (motion, prompt) or (None, None)."""
    meta_path = npy.with_suffix(".json")
    if not meta_path.exists():
        return None, None
    try:
        m = np.load(str(npy)).astype(np.float64)
        meta = json.load(open(meta_path, encoding="utf-8"))
    except Exception:
        return None, None
    return m, meta.get("prompt", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dirs", type=str, required=True,
                        help="comma-separated G2 pool dirs (existing + balanced).")
    parser.add_argument("--calibration", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    parser.add_argument("--p-stress", type=str, default="80,90,95")
    parser.add_argument("--p-clean", type=float, default=20.0)
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "g2_real_stress_profile_v2.json")
    args = parser.parse_args()

    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_p99 = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                if calib["summary"][n].get("n", 0) > 0}
    evaluators = list(DEFAULT_EVALUATORS); gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)

    pool_dirs = [Path(p).resolve() for p in args.pool_dirs.split(",")]
    print(f"[INFO] pool dirs: {[str(p.relative_to(REPO_ROOT)) for p in pool_dirs]}")
    per_motion = []
    seen_ids = set()
    duplicate_warnings = []
    for pdir in pool_dirs:
        npys = sorted(pdir.glob("motion_*.npy"))
        print(f"[INFO] {pdir.name}: {len(npys)} motions")
        for npy in npys:
            tid = npy.stem
            if tid in seen_ids:
                # motion_001 같은 stem 이 두 pool 에 동일하게 존재 — pool tag 로 disambiguate.
                tid = f"{pdir.name}/{tid}"
            seen_ids.add(tid)
            m, prompt = _load_motion_with_meta(npy)
            if m is None or m.ndim != 3:
                continue
            group = classify(prompt)
            art = {ev.name: round(_max_score(ev.evaluate(m)), 6) for ev in evaluators if ev.name in ARTIFACT_EVALUATORS}
            phy = {ev.name: round(_max_score(ev.evaluate(m)), 6) for ev in gate_evaluators if ev.name in PHYSICAL_EVALUATORS}
            art_total = float(np.mean([art[n] for n in ARTIFACT_EVALUATORS]))
            loads = [phy[n] / max(gate_p99.get(n, 1e-9), 1e-9) for n in PHYSICAL_EVALUATORS]
            phy_load = float(np.mean(loads)) if loads else 0.0
            dom_art = max(ARTIFACT_EVALUATORS, key=lambda n: art[n])
            dom_phy = max(PHYSICAL_EVALUATORS, key=lambda n: (phy[n] / max(gate_p99.get(n, 1e-9), 1e-9)))
            any_phy_violating = any(phy[n] > gate_p99.get(n, float("inf")) for n in PHYSICAL_EVALUATORS)
            per_motion.append({
                "trial_id": tid, "pool_dir": pdir.name, "prompt": prompt, "motion_group": group,
                "artifact_scores": art, "physical_scores": phy,
                "artifact_total_score": round(art_total, 6), "physical_load": round(phy_load, 6),
                "dominant_artifact_evaluator": dom_art, "dominant_physical_evaluator": dom_phy,
                "any_phy_above_clean_p99": bool(any_phy_violating),
            })

    print(f"[INFO] total motions profiled: {len(per_motion)}")

    # Per-group percentile + band assignment.
    by_group = defaultdict(list)
    for r in per_motion:
        by_group[r["motion_group"]].append(r)
    p_stress_list = sorted([float(x) for x in args.p_stress.split(",")])
    per_group_summary = {}
    for g in list(by_group):
        rows = by_group[g]
        arts = np.array([r["artifact_total_score"] for r in rows])
        if len(arts) < 5:
            # 너무 작은 group → group-wise percentile 적용 안 함, normal 로 marking.
            for r in rows:
                r["band"] = "normal"; r["stress_tier"] = None
                r["band_source"] = "small_group_default_normal"
            per_group_summary[g] = {"n": len(rows), "small_group_caveat": True}
            continue
        clean_thr = float(np.percentile(arts, args.p_clean))
        stress_thrs = {f"stress_p{int(round(p))}": float(np.percentile(arts, p)) for p in p_stress_list}
        primary_stress_thr = stress_thrs[f"stress_p{int(round(min(p_stress_list)))}"]
        for r in rows:
            s = r["artifact_total_score"]
            if s <= clean_thr:
                band = "g2_clean_like"
            elif s >= primary_stress_thr:
                band = "stress"
            else:
                band = "normal"
            r["band"] = band; r["band_source"] = "per_group_percentile"
            r["stress_tier"] = None
            for tier_name, thr in sorted(stress_thrs.items(), key=lambda x: -x[1]):
                if s >= thr:
                    r["stress_tier"] = tier_name; break
        bc = Counter(r["band"] for r in rows)
        per_group_summary[g] = {
            "n": len(rows),
            "p_clean_thr": clean_thr,
            "stress_thrs": stress_thrs,
            "band_counts": dict(bc),
            "p50": float(np.percentile(arts, 50)),
            "mean_artifact_total": float(np.mean(arts)),
        }

    # Global summary.
    band_counts_global = Counter(r["band"] for r in per_motion)
    group_counts_global = Counter(r["motion_group"] for r in per_motion)
    group_band_xtab = defaultdict(lambda: Counter())
    for r in per_motion:
        group_band_xtab[r["motion_group"]][r["band"]] += 1

    out = {
        "schema_version": "1.0.0", "record_type": "g2_real_stress_profile_v2",
        "task_id": "g2_real_stress_profile_v2",
        **common_snapshot_metadata(
            split_id=args.split_id or "g2_real_stress_profile_v2",
            oracle_type="profile", action_grid="N/A (profile only)",
            stage="Stage-2G-step1-profile-v2", evidence_tier=[REAL_DISTRIBUTION],
            evaluators=evaluators, gate_evaluators=gate_evaluators,
        ),
        "directive": "group-aware percentile: stress = top-20% of each motion_group (not pool-wide).",
        "pool_dirs": [str(p.relative_to(REPO_ROOT)) for p in pool_dirs],
        "n_motions": len(per_motion),
        "global_group_counts": dict(group_counts_global),
        "global_band_counts": dict(band_counts_global),
        "group_band_crosstab": {g: dict(c) for g, c in group_band_xtab.items()},
        "per_group_summary": per_group_summary,
        "gate_p99": gate_p99,
        "per_motion": per_motion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== G2 Real Stress Profile v2 (group-aware) — n={len(per_motion)} ===")
    print(f"\n  group              n    stress  normal  clean_like  p50_artifact")
    for g, _ in GROUPS_ORDER + [("other", [])]:
        if g not in by_group: continue
        s = per_group_summary[g]; bc = s.get("band_counts", {})
        p50 = s.get("p50", float("nan"))
        print(f"   {g:<24} {s['n']:3d}   {bc.get('stress',0):3d}     {bc.get('normal',0):3d}     {bc.get('g2_clean_like',0):3d}        {p50:.4f}")
    print(f"\n  global band counts: {dict(band_counts_global)}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
