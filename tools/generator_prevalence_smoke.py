"""Cross-generator problem-prevalence smoke (AR-022/AR-045 preliminary, pre-AR-048 protocol).

동기: F7/F8 — MotionGPT(VQ)엔 정의된 물리 문제(foot skate/penetrate)가 거의 0%.
문헌(PhysDiff)은 diffusion(MDM)이 그 문제를 낸다고 예측. **MotionGPT 와 동일 threshold**로
새 generator pool 의 evaluator prevalence + 표준 foot_skate magnitude 를 측정해 직접 대조.

Threshold = AR-043 (dataset_issue_prevalence_audit_v1) 와 동일:
  - artifact: score >= evaluator SEV_LOW (FootFloating 0.05 / BoneLength 0.02 / VelocityJitter 0.03)
  - physical: score > clean calibration p99 (Penetrate 0.0 / Float 0.798 / Skate 0.0 / Jerk 0.106 / BoneCV 5.5e-6)
  (Penetrate·Skate p99=0 → nonzero 만 떠도 위반. MotionGPT 는 0%.)

foot_skate/accel/float = AR-029 (physical_metric_g2_stress) 표준 magnitude.

CLI (motion3d env):
    python tools/generator_prevalence_smoke.py \
        external_assets/mdm_generated_hml3d_test50_seed20260603 MDM
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from tools.physical_metric_g2_stress import foot_skate, accel_stats, float_mag

ARTIFACT_THRESH = {"FootFloatingEvaluator": 0.05, "BoneLengthEvaluator": 0.02, "VelocityJitterEvaluator": 0.03}
PHYS_P99 = {
    "PenetrateEvaluator": 0.0, "FloatEvaluator": 0.7983539458952797, "SkateEvaluator": 0.0,
    "JerkSpikeEvaluator": 0.10556407202395618, "BoneLengthCVEvaluator": 5.506674955074729e-06,
}

# MotionGPT 대조 baseline (AR-043 g2_natural prevalence + AR-029 g2_natural magnitude).
G2_NATURAL = {
    "artifact_prev": {"FootFloatingEvaluator": 0.10, "BoneLengthEvaluator": 0.84, "VelocityJitterEvaluator": 0.04},
    "phys_above_p99": {"PenetrateEvaluator": 0.0, "FloatEvaluator": 0.0, "SkateEvaluator": 0.0,
                       "JerkSpikeEvaluator": 0.0, "BoneLengthCVEvaluator": 1.0},
    "foot_skate_mean": 0.0051, "accel_mean": None, "float_mag_mean": 0.0168,
}


def _max_score(ev, m):
    return max((r.score for r in ev.evaluate(m)), default=0.0)


def main() -> None:
    pool_dir = (Path(sys.argv[1]) if len(sys.argv) > 1 else
                REPO_ROOT / "external_assets" / "mdm_generated_hml3d_test50_seed20260603").resolve()
    gen_id = sys.argv[2] if len(sys.argv) > 2 else "MDM"
    try:
        pool_rel = str(pool_dir.relative_to(REPO_ROOT))
    except ValueError:
        pool_rel = str(pool_dir)

    files = sorted(pool_dir.glob("motion_*.npy"))
    motions = [np.load(str(f)).astype(np.float64) for f in files]
    n = len(motions)
    if n == 0:
        print(f"[ERR] no motion_*.npy in {pool_dir}"); return

    art_evs = {ev.name: ev for ev in DEFAULT_EVALUATORS if ev.name in ARTIFACT_THRESH}
    phy_evs = {ev.name: ev for ev in DEFAULT_PHYSICAL_GATE_EVALUATORS}

    art_scores = {k: [_max_score(art_evs[k], m) for m in motions] for k in ARTIFACT_THRESH}
    phy_scores = {k: [_max_score(phy_evs[k], m) for m in motions] for k in PHYS_P99}
    fs = [foot_skate(m) for m in motions]
    acc = [accel_stats(m)[0] for m in motions]
    fm = [float_mag(m) for m in motions]

    art_prev = {k: float(np.mean([s >= ARTIFACT_THRESH[k] for s in art_scores[k]])) for k in ARTIFACT_THRESH}
    phy_above = {k: float(np.mean([s > PHYS_P99[k] for s in phy_scores[k]])) for k in PHYS_P99}
    phy_nonzero = {k: float(np.mean([s > 0 for s in phy_scores[k]])) for k in PHYS_P99}

    out = {
        "schema_version": "1.0.0", "record_type": "generator_prevalence_smoke",
        "board_id": "AR-022/AR-045 (preliminary smoke, pre-AR-048 protocol)",
        "generator_id": gen_id, "n": n, "pool": pool_rel,
        "evidence_tier": ["real-distribution evidence", "smoke (small-n)"],
        "thresholds": {"artifact_SEV_LOW": ARTIFACT_THRESH, "physical_clean_p99": PHYS_P99,
                       "source": "AR-043 dataset_issue_prevalence_audit_v1 (동일 threshold)"},
        "artifact_prevalence": art_prev,
        "physical_above_clean_p99": phy_above,
        "physical_nonzero_rate": phy_nonzero,
        "physical_mean": {k: float(np.mean(phy_scores[k])) for k in PHYS_P99},
        "magnitude": {"foot_skate_mean": float(np.mean(fs)), "accel_mean": float(np.mean(acc)),
                      "float_mag_mean": float(np.mean(fm))},
        "baseline_g2_natural": G2_NATURAL,
        "caveat": "smoke n=%d, n_frames=40, root-relative canonical. 동일 threshold 로 MotionGPT 대조. 본격은 stratified pool 후." % n,
    }
    out_path = REPO_ROOT / "evals" / "snapshots" / f"generator_prevalence_smoke_{gen_id.lower()}_v1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # console (ascii-safe).
    print(f"=== {gen_id} prevalence smoke (n={n}) vs MotionGPT g2_natural ===\n")
    print(f"{'PHYSICAL (above clean p99)':<26} {gen_id:>8} {'nonzero':>8}  | {'MotionGPT':>9}")
    for k in PHYS_P99:
        g2 = G2_NATURAL["phys_above_p99"][k]
        print(f"  {k:<24} {phy_above[k]*100:>6.0f}% {phy_nonzero[k]*100:>7.0f}%  | {g2*100:>7.0f}%")
    print(f"\n{'ARTIFACT (>= SEV_LOW)':<26} {gen_id:>8}  | {'MotionGPT':>9}")
    for k in ARTIFACT_THRESH:
        print(f"  {k:<24} {art_prev[k]*100:>6.0f}%  | {G2_NATURAL['artifact_prev'][k]*100:>7.0f}%")
    print(f"\n{'MAGNITUDE':<26} {gen_id:>8}  | {'MotionGPT':>9}")
    print(f"  {'foot_skate (GMD, m)':<24} {np.mean(fs):>7.4f}  | {G2_NATURAL['foot_skate_mean']:>8.4f}")
    print(f"  {'float_mag (m)':<24} {np.mean(fm):>7.4f}  | {G2_NATURAL['float_mag_mean']:>8.4f}")
    print(f"  {'accel_mean (m/fr^2)':<24} {np.mean(acc):>7.4f}  |   (n/a)")
    print(f"\n[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
