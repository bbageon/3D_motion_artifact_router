"""AR-029: standard physical-plausibility metric on closed-loop output (real g2_stress).

동기 (2026-06-03 사용자 directive):
> "흠집 전용 자(physical metric)로 real g2_stress 를 다시 재서, FID 가 가렸던 도구 효과가
>  실제로 있는지 확인" — FID(Category A, distributional)는 per-frame local foot artifact 에
>  둔감. PhysDiff(ICCV2023)/GMD(ICCV2023)/EDGE(CVPR2023) 의 표준 physical metric 으로 재측정.

배경: standard_metric_closed_loop_v1 에서 real g2_stress 의 ΔFID 는 neutral~degrade.
하지만 FID 는 local foot skating/jitter 를 거의 못 잡음 → 표준 physical metric 으로 재평가하면
도구 효과(foot lock/smoothing)가 보일 수 있음. 본 도구는 closed-loop 보정 모션(이미 cache 됨,
evals/snapshots/closed_loop_final_motions_v1/)을 로드해 original/noop 대비 Δ 를 paired 로 측정.

METRIC (Category B — standard physical plausibility, lower = better):
  - foot_skate   : EDGE(Tseng CVPR2023)/GMD(Karunratanakul ICCV2023) 표준 foot-skating
                   magnitude. s = Σ_feet ||Δp_xz|| · (2 − 2^(h/H)), h<H. 단위 m/frame.
  - accel_p95    : per-joint acceleration norm p95 (MDM Tevet ICLR2023 smoothness).
  - accel_mean   : per-joint acceleration norm mean.
  - float_mag    : contact-candidate frame 의 평균 foot height (planted foot 가 뜬 정도, m).
  (cross-ref Category C: gate 가 최적화 target → 아래 GATE_RELATIONSHIP caveat 참조.)

§3-25 GATE_RELATIONSHIP: foot_skate 는 FootLockTool 의 target 과 상관(독립 free-lunch 아님).
본 측정의 claim = "표준 physical metric 으로도 개선이 보이는가 + FID 보존되는가(이미 측정)",
NOT "최적화 목표와 무관한 독립 개선". policy_contribution baseline = original/noop.

CLI (motion3d env):
    python tools/physical_metric_g2_stress.py \
        --split evals/splits/g2_real_stress_split_v2.json \
        --final-motion-dir evals/snapshots/closed_loop_final_motions_v1 \
        --output evals/snapshots/physical_metric_g2_stress_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Canonical SMPL-22 joint indices (verified: skeleton_normalizer.canonical_smpl_22).
LEFT_FOOT, RIGHT_FOOT = 10, 11
LEFT_ANKLE, RIGHT_ANKLE = 7, 8
FOOT_JOINTS = (LEFT_FOOT, RIGHT_FOOT)

HOLDOUTS = ("g2_stress_holdout", "g2_natural_holdout")
POLICIES = ("M0", "M1", "M2", "M3", "random_gate", "heuristic_gate", "dense_oracle_step1")
KEY_POLICIES = ("M0", "heuristic_gate", "dense_oracle_step1")

DEFAULT_EXISTING_POOL = REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527"
DEFAULT_BALANCED_POOL = REPO_ROOT / "external_assets" / "g2_generated_hml3d_balanced300_seed20260531"

FOOT_SKATE_H = 0.05  # contact-height threshold (m) for GMD/EDGE foot-skate weighting.
V_CONTACT = 0.02     # horizontal velocity (m/frame) for contact-candidate (float_mag).
H_CONTACT = 0.10     # height (m) for contact-candidate (float_mag).


def _resolve_g2(trial_id: str, existing_pool: Path, balanced_pool: Path) -> tuple[Path, str]:
    """trial_id → (pool_dir, stem). 'pool_name/motion_NNN' → balanced; else existing."""
    if "/" in trial_id:
        _prefix, stem = trial_id.split("/", 1)
        return balanced_pool, stem
    return existing_pool, trial_id


# ---------------- standard physical metrics (pure numpy, lower = better) ----------------

def foot_skate(motion: np.ndarray, H: float = FOOT_SKATE_H) -> float:
    """EDGE(CVPR2023)/GMD(ICCV2023) foot-skating magnitude (m/frame).

    s_t = ||Δp_xz(t)|| · max(0, 2 − 2^(h_t/H)) for each foot; averaged over feet & frames.
    h_t = foot height above ground at the moving step (ground = per-motion min Y).
    """
    T = motion.shape[0]
    if T < 2:
        return 0.0
    ground_y = float(np.min(motion[:, :, 1]))
    per_foot = []
    for j in FOOT_JOINTS:
        fxz = motion[:, j, :][:, [0, 2]]
        disp = np.linalg.norm(np.diff(fxz, axis=0), axis=1)   # [T-1]
        h = motion[1:, j, 1] - ground_y                       # height at arrival frame [T-1]
        weight = np.clip(2.0 - np.power(2.0, h / H), 0.0, None)
        per_foot.append(disp * weight)
    return float(np.mean(np.concatenate(per_foot)))


def accel_stats(motion: np.ndarray) -> tuple[float, float]:
    """per-joint acceleration norm (m/frame^2): (mean, p95). MDM smoothness proxy."""
    T = motion.shape[0]
    if T < 3:
        return 0.0, 0.0
    accel = np.diff(motion, axis=0, n=2)            # [T-2, 22, 3]
    an = np.linalg.norm(accel, axis=-1)             # [T-2, 22]
    return float(np.mean(an)), float(np.percentile(an, 95))


def float_mag(motion: np.ndarray, v_thresh: float = V_CONTACT, h_thresh: float = H_CONTACT) -> float:
    """Contact-candidate 평균 foot height (m): planted foot 가 뜬 정도. lower = better."""
    T = motion.shape[0]
    if T < 2:
        return 0.0
    ground_y = float(np.min(motion[:, :, 1]))
    heights = []
    for j in FOOT_JOINTS:
        fxz = motion[:, j, :][:, [0, 2]]
        disp = np.linalg.norm(np.diff(fxz, axis=0), axis=1)
        vel = np.concatenate([disp, [disp[-1]]])
        height = motion[:, j, 1] - ground_y
        contact = (vel <= v_thresh) & (height <= h_thresh)
        if contact.any():
            heights.append(height[contact])
    if not heights:
        return 0.0
    return float(np.mean(np.concatenate(heights)))


METRICS = ("foot_skate", "accel_p95", "accel_mean", "float_mag")


def compute_metrics(motion: np.ndarray) -> dict[str, float]:
    am, ap = accel_stats(motion)
    return {
        "foot_skate": foot_skate(motion),
        "accel_p95": ap,
        "accel_mean": am,
        "float_mag": float_mag(motion),
    }


# ---------------- paired statistics ----------------

def paired_stats(orig: np.ndarray, pol: np.ndarray, rng: np.random.Generator) -> dict:
    """orig, pol: per-sample metric arrays (lower=better). delta = pol - orig (음수=개선)."""
    from scipy.stats import wilcoxon
    delta = pol - orig
    n = len(delta)
    out = {
        "n": n,
        "orig_mean": float(np.mean(orig)), "orig_median": float(np.median(orig)),
        "pol_mean": float(np.mean(pol)), "pol_median": float(np.median(pol)),
        "delta_mean": float(np.mean(delta)), "delta_median": float(np.median(delta)),
        "frac_improved": float(np.mean(delta < 0)),
        "rel_change_median": (float(np.median(delta) / np.median(orig)) if np.median(orig) > 1e-12 else None),
    }
    # Wilcoxon signed-rank (paired). zero-difference handling.
    if np.allclose(delta, 0):
        out["wilcoxon_p"] = 1.0
    else:
        try:
            _stat, p = wilcoxon(orig, pol)
            out["wilcoxon_p"] = float(p)
        except Exception as e:
            out["wilcoxon_p"] = None
            out["wilcoxon_err"] = str(e)
    # Cohen's d (paired) = mean(delta)/std(delta).
    sd = float(np.std(delta, ddof=1)) if n > 1 else 0.0
    out["cohens_d_paired"] = (float(np.mean(delta) / sd) if sd > 1e-12 else 0.0)
    # bootstrap 95% CI of median delta.
    boots = [float(np.median(delta[rng.integers(0, n, n)])) for _ in range(1000)]
    out["delta_median_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "evals" / "splits" / "g2_real_stress_split_v2.json")
    ap.add_argument("--g2-real", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_g2_real_stage2_v1.json")
    ap.add_argument("--final-motion-dir", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "closed_loop_final_motions_v1")
    ap.add_argument("--existing-pool", type=Path, default=DEFAULT_EXISTING_POOL)
    ap.add_argument("--balanced-pool", type=Path, default=DEFAULT_BALANCED_POOL)
    ap.add_argument("--seed", type=int, default=20260603)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "physical_metric_g2_stress_v1.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    g2 = json.load(open(args.g2_real, encoding="utf-8"))
    state_map = g2["states"]

    results = {}
    for ho in HOLDOUTS:
        sids = [s for s in state_map if state_map[s]["split"] == ho]
        # collect paired metric arrays for states where original + all policy motions exist.
        per_metric = {"original": {m: [] for m in METRICS}}
        for pol in POLICIES:
            per_metric[pol] = {m: [] for m in METRICS}
        groups = []
        n_skipped = 0
        for sid in sids:
            pool_dir, stem = _resolve_g2(sid, args.existing_pool, args.balanced_pool)
            opath = pool_dir / f"{stem}.npy"
            sid_safe = sid.replace("/", "_")
            ppaths = {pol: args.final_motion_dir / f"{ho}__{pol}__{sid_safe}.npy" for pol in POLICIES}
            if not opath.exists() or not all(p.exists() for p in ppaths.values()):
                n_skipped += 1
                continue
            orig = np.load(str(opath)).astype(np.float64)
            mots = {pol: np.load(str(ppaths[pol])).astype(np.float64) for pol in POLICIES}
            om = compute_metrics(orig)
            for m in METRICS:
                per_metric["original"][m].append(om[m])
            for pol in POLICIES:
                pm = compute_metrics(mots[pol])
                for m in METRICS:
                    per_metric[pol][m].append(pm[m])
            groups.append(state_map[sid].get("motion_group", "?"))
        n = len(per_metric["original"][METRICS[0]])
        print(f"[INFO] {ho}: n={n} paired (skipped {n_skipped})")
        if n < 5:
            results[ho] = {"n": n, "skipped": True}
            continue
        orig_arr = {m: np.array(per_metric["original"][m]) for m in METRICS}
        ho_res = {"n": n, "original_mean": {m: float(np.mean(orig_arr[m])) for m in METRICS}, "policies": {}}
        for pol in POLICIES:
            pol_res = {}
            for m in METRICS:
                pol_arr = np.array(per_metric[pol][m])
                pol_res[m] = paired_stats(orig_arr[m], pol_arr, rng)
            ho_res["policies"][pol] = pol_res
        results[ho] = ho_res

    out = {
        "schema_version": "1.0.0",
        "record_type": "physical_metric_standard",
        "task_id": "physical_metric_g2_stress_v1",
        "board_id": "AR-029",
        "evaluator": "standard physical plausibility (foot_skate EDGE/GMD, accel MDM) — pure-numpy",
        "metric_category": "B (standard physical plausibility; foot_skate=EDGE/GMD, accel=MDM smoothness)",
        "evidence_tier": ["real-distribution evidence"],
        "directive": "FID 가 가린 도구 효과를 표준 physical metric 으로 재측정. Δ = policy - original (음수=개선).",
        "metric_definitions": {
            "foot_skate": "EDGE(Tseng CVPR2023)/GMD(Karunratanakul ICCV2023): sum_feet ||dxz||*(2-2^(h/H)), H=%.3fm. m/frame. lower better." % FOOT_SKATE_H,
            "accel_p95": "per-joint accel norm p95 (MDM Tevet ICLR2023). m/frame^2. lower=smoother.",
            "accel_mean": "per-joint accel norm mean. m/frame^2.",
            "float_mag": "contact-candidate frame 평균 foot height (m). v<=%.2f & h<=%.2f. lower better." % (V_CONTACT, H_CONTACT),
        },
        "gate_relationship": (
            "§3-25: foot_skate/float_mag 는 FootLockTool 의 target 과 상관 → 독립 free-lunch 아님. "
            "claim = '표준 physical metric(Category B)으로도 개선이 보이며 FID(Category A)는 보존(별도 측정)', "
            "NOT '최적화 목표와 무관한 독립 개선'. policy_contribution baseline = original/noop. "
            "gate_recheck=true (closed-loop 이 gate-ON 으로 생성)."
        ),
        "references": [
            "EDGE (Tseng et al., CVPR 2023) — foot skating / PFC",
            "GMD (Karunratanakul et al., ICCV 2023) — foot skating ratio",
            "PhysDiff (Yuan et al., ICCV 2023) — penetrate/float/skate motivation",
            "MDM (Tevet et al., ICLR 2023) — acceleration smoothness",
        ],
        "params": {"foot_skate_H_m": FOOT_SKATE_H, "v_contact": V_CONTACT, "h_contact": H_CONTACT, "seed": args.seed},
        "key_policies": list(KEY_POLICIES),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # console summary.
    print("\n=== AR-029 Standard Physical Metric on Closed-Loop Output (Δ = policy - original, 음수=개선) ===")
    for ho in HOLDOUTS:
        r = results.get(ho, {})
        if r.get("skipped"):
            print(f"\n[{ho}] skipped (n={r.get('n')})"); continue
        print(f"\n[{ho}] n={r['n']}  original means: " +
              "  ".join(f"{m}={r['original_mean'][m]:.4f}" for m in METRICS))
        for pol in KEY_POLICIES:
            print(f"  -- {pol} --")
            for m in METRICS:
                s = r["policies"][pol][m]
                star = "*" if (s.get("wilcoxon_p") is not None and s["wilcoxon_p"] < 0.05) else " "
                rc = s.get("rel_change_median")
                rc_s = f"{rc:+.1%}" if rc is not None else "n/a"
                print(f"     {m:<11} Δmed={s['delta_median']:+.4f} ({rc_s})  p={s.get('wilcoxon_p')}  "
                      f"d={s['cohens_d_paired']:+.2f}  improved={s['frac_improved']:.0%} {star}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
