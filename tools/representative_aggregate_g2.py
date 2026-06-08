"""AR-046b: prevalence-weighted REPRESENTATIVE aggregation of closed-loop effect on G2.

동기 (사용자 directive 2026-06-03): "stress 중심 수치를 전체 비율(full pool)로 다시 집계해
대표 결과 + no-harm 을 헤드라인으로, stress/natural 은 진단 보조표로."

방법론 (§3-22 grounded): stress 는 artifact proxy 로 고른 enriched subset → 단독 보고는
selection bias (낙관적). 대표 성능·no-harm 주장은 generator 의 실제 출력 분포(전체 비율)로.
HumanML3D 평가 프로토콜(Guo CVPR2022)·MDM(ICLR2023)·MoMask(CVPR2024) 모두 full test set 보고.

Band prevalence (full real-G2 pool n=600, rl2_transition_g2_real_stage2):
  stress=124 (20.67%) / normal=353 / g2_clean_like=123  →  low-artifact(normal+clean_like)=476 (79.33%)
holdout cache 는 stress(band=stress, n=65) + natural(band=normal, n=99) 만 보유.
→ clean_like 는 normal 보다 더 깨끗 → natural holdout 의 (이미 ~0) Δ 로 represent (conservative,
  clean_like 의 활동을 오히려 약간 과대평가 → 결과를 부풀리지 않음).

대표 가중: Δ_representative = w_stress·Δ_stress + w_low·Δ_natural,
  w_stress=124/600=0.2067, w_low=476/600=0.7933.

METRIC: physical(Category B; foot_skate/accel/float, per-sample) + artifact_total(Category C).
  FID/R-Prec(Category A, set-level) 는 standard_metric_closed_loop_v1 의 per-band Δ 를
  prevalence-weighted 로 결합(근사 — true 대표 FID 는 clean_like motion 필요, caveat).

no-harm: clean_noharm_holdout(HumanML3D GT, n=100) Δ — 이상적으로 ~0.

CLI (motion3d env):
    python tools/representative_aggregate_g2.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.physical_metric_g2_stress import (
    compute_metrics, METRICS, POLICIES, KEY_POLICIES, _resolve_g2,
    DEFAULT_EXISTING_POOL, DEFAULT_BALANCED_POOL,
)

# Band prevalence (full real-G2 pool n=600).
BAND_COUNTS = {"stress": 124, "normal": 353, "g2_clean_like": 123}
N_POOL = sum(BAND_COUNTS.values())
W_STRESS = BAND_COUNTS["stress"] / N_POOL                       # 0.2067
W_LOW = (BAND_COUNTS["normal"] + BAND_COUNTS["g2_clean_like"]) / N_POOL  # 0.7933

ARTIFACT_NAMES = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
# all metrics reported: physical (Category B) + artifact_total (Category C). lower=better.
ALL_METRICS = METRICS + ("artifact_total",)


def _artifact_total_fn():
    from evaluators import DEFAULT_EVALUATORS
    art = [ev for ev in DEFAULT_EVALUATORS if ev.name in ARTIFACT_NAMES]

    def _fn(m):
        return float(np.mean([max((r.score for r in ev.evaluate(m)), default=0.0) for ev in art]))
    return _fn


def _all_metrics(motion, artifact_fn):
    d = compute_metrics(motion)
    d["artifact_total"] = artifact_fn(motion)
    return d


def _collect(sids, ho, orig_loader, final_dir, artifact_fn):
    """per-sample metric arrays for original + each policy (paired, only fully-valid states)."""
    per = {"original": {m: [] for m in ALL_METRICS}}
    for pol in POLICIES:
        per[pol] = {m: [] for m in ALL_METRICS}
    n_skip = 0
    for sid in sids:
        opath = orig_loader(sid)
        sid_safe = sid.replace("/", "_")
        ppaths = {pol: final_dir / f"{ho}__{pol}__{sid_safe}.npy" for pol in POLICIES}
        if opath is None or not opath.exists() or not all(p.exists() for p in ppaths.values()):
            n_skip += 1
            continue
        om = _all_metrics(np.load(str(opath)).astype(np.float64), artifact_fn)
        pol_ms = {pol: _all_metrics(np.load(str(ppaths[pol])).astype(np.float64), artifact_fn) for pol in POLICIES}
        for m in ALL_METRICS:
            per["original"][m].append(om[m])
            for pol in POLICIES:
                per[pol][m].append(pol_ms[pol][m])
    n = len(per["original"][ALL_METRICS[0]])
    return per, n, n_skip


def _band_delta(per, pol, m):
    """per-sample delta array (pol - original) for one band/metric."""
    return np.array(per[pol][m]) - np.array(per["original"][m])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2-real", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_g2_real_stage2_v1.json")
    ap.add_argument("--final-motion-dir", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "closed_loop_final_motions_v1")
    ap.add_argument("--hml3d-dir", type=Path,
                    default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    ap.add_argument("--standard-snapshot", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "standard_metric_closed_loop_v1.json")
    ap.add_argument("--seed", type=int, default=20260603)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "representative_aggregate_g2_v1.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    g2 = json.load(open(args.g2_real, encoding="utf-8"))
    state_map = g2["states"]
    artifact_fn = _artifact_total_fn()

    def stress_natural_loader(sid):
        pool_dir, stem = _resolve_g2(sid, DEFAULT_EXISTING_POOL, DEFAULT_BALANCED_POOL)
        return pool_dir / f"{stem}.npy"

    def clean_loader(sid):
        return args.hml3d_dir / f"{sid}.npy"

    bands = {}
    # stress band (g2_stress_holdout) + low band (g2_natural_holdout) + clean (clean_noharm_holdout)
    for ho, loader in [("g2_stress_holdout", stress_natural_loader),
                       ("g2_natural_holdout", stress_natural_loader),
                       ("clean_noharm_holdout", clean_loader)]:
        sids = [s for s in state_map if state_map[s]["split"] == ho]
        per, n, n_skip = _collect(sids, ho, loader, args.final_motion_dir, artifact_fn)
        bands[ho] = {"per": per, "n": n, "n_skip": n_skip}
        print(f"[INFO] {ho}: n={n} (skip {n_skip})")

    s_per, n_s = bands["g2_stress_holdout"]["per"], bands["g2_stress_holdout"]["n"]
    l_per, n_l = bands["g2_natural_holdout"]["per"], bands["g2_natural_holdout"]["n"]

    # ---- representative aggregation (physical + artifact) ----
    representative = {}
    diagnostic = {"stress_band(20.7%)": {}, "low_band(79.3%)": {}, "clean_noharm(no-harm ref)": {}}
    for pol in POLICIES:
        rep_pol = {}
        for m in ALL_METRICS:
            ds = _band_delta(s_per, pol, m)   # stress band deltas
            dl = _band_delta(l_per, pol, m)   # low band deltas
            rep_mean = W_STRESS * float(np.mean(ds)) + W_LOW * float(np.mean(dl))
            # prevalence-weighted bootstrap CI of mean Δ.
            boots = []
            for _ in range(args.n_boot):
                bs = ds[rng.integers(0, n_s, n_s)]
                bl = dl[rng.integers(0, n_l, n_l)]
                boots.append(W_STRESS * float(np.mean(bs)) + W_LOW * float(np.mean(bl)))
            ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            rep_frac_improved = W_STRESS * float(np.mean(ds < 0)) + W_LOW * float(np.mean(dl < 0))
            orig_rep = W_STRESS * float(np.mean(s_per["original"][m])) + W_LOW * float(np.mean(l_per["original"][m]))
            rep_pol[m] = {
                "representative_delta_mean": rep_mean,
                "delta_mean_ci95": ci,
                "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
                "rel_change": (rep_mean / orig_rep if abs(orig_rep) > 1e-12 else None),
                "representative_frac_improved": rep_frac_improved,
                "original_representative_mean": orig_rep,
            }
        representative[pol] = rep_pol

    # diagnostic per-band means (Δ) for key metrics.
    for label, per, n in [("stress_band(20.7%)", s_per, n_s),
                          ("low_band(79.3%)", l_per, n_l),
                          ("clean_noharm(no-harm ref)", bands["clean_noharm_holdout"]["per"],
                           bands["clean_noharm_holdout"]["n"])]:
        block = {"n": n}
        for pol in POLICIES:
            block[pol] = {m: float(np.mean(_band_delta(per, pol, m))) for m in ALL_METRICS}
        diagnostic[label] = block

    # ---- FID / R-Prec (Category A, set-level) prevalence-weighted from existing snapshot ----
    fid_rep = {}
    try:
        std = json.load(open(args.standard_snapshot, encoding="utf-8"))["results"]
        st = std["g2_stress_holdout"]["methods"]
        na = std["g2_natural_holdout"]["methods"]
        for pol in POLICIES:
            fid_rep[pol] = {
                "delta_FID_weighted": W_STRESS * st[pol]["delta_FID"] + W_LOW * na[pol]["delta_FID"],
                "delta_R1_weighted": W_STRESS * st[pol]["delta_R1"] + W_LOW * na[pol]["delta_R1"],
                "stress_delta_FID": st[pol]["delta_FID"], "low_delta_FID": na[pol]["delta_FID"],
            }
        fid_rep["_original_FID"] = {
            "stress": std["g2_stress_holdout"]["methods"]["original"]["FID_vs_GT"],
            "low(natural)": std["g2_natural_holdout"]["methods"]["original"]["FID_vs_GT"],
            "clean_ref": std["clean_noharm_holdout"]["methods"]["original"]["FID_vs_GT"],
        }
    except Exception as e:
        fid_rep["error"] = str(e)

    out = {
        "schema_version": "1.0.0", "record_type": "representative_aggregate_g2",
        "task_id": "representative_aggregate_g2_v1", "board_id": "AR-046",
        "evidence_tier": ["real-distribution evidence"],
        "directive": "stress(enriched) 단독 → 전체 비율(prevalence-weighted) 대표 결과 + no-harm 헤드라인. stress/natural=진단.",
        "method": {
            "band_prevalence_full_pool_n600": BAND_COUNTS,
            "weights": {"w_stress": W_STRESS, "w_low_artifact": W_LOW},
            "naive_concat_stress_weight": n_s / (n_s + n_l),
            "note": f"stress band={W_STRESS:.1%} (실제); naive {n_s}:{n_l} concat 은 stress={n_s / (n_s + n_l):.1%} 로 과대표집. low band=normal+clean_like; clean_like 는 natural holdout Δ 로 represent(conservative).",
            "selection_bias_rationale": "stress 는 artifact proxy(=router 단서)로 선택 → 단독 보고는 optimistic. 대표 주장은 full 분포 (HumanML3D protocol Guo CVPR2022).",
        },
        "metric_categories": {
            "physical (foot_skate/accel/float)": "B (standard; EDGE CVPR2023 / GMD ICCV2023 / MDM ICLR2023)",
            "artifact_total": "C (internal proxy)",
            "FID/R-Prec": "A (HumanML3D tm2t) — set-level, prevalence-weighted=근사",
        },
        "lower_is_better": True,
        "key_policies": list(KEY_POLICIES),
        "representative": representative,
        "fid_representative_weighted": fid_rep,
        "diagnostic_per_band": diagnostic,
        "caveats": [
            "FID/R-Prec 의 prevalence-weighted Δ 는 per-band Δ 의 가중합(근사). true 대표 set-level FID 는 clean_like band motion 필요.",
            "clean_like(20.5%) 는 normal holdout Δ 로 represent (clean_like 가 더 깨끗 → conservative).",
            "physical foot_skate 는 ground=minY 추정 민감 (§3-1-1).",
            "single snapshot — 가설 status 입력 아님 (diagnostic).",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- console headline (ascii-safe for cp949 stdout) ----
    print("\n=== REPRESENTATIVE (prevalence-weighted, full G2 dist) -- delta vs original, neg=improve ===")
    print(f"weights: stress={W_STRESS:.3f}  low-artifact={W_LOW:.3f}   (naive concat would be stress={n_s/(n_s+n_l):.3f})")
    hdr = ["foot_skate", "accel_mean", "float_mag", "artifact_total"]
    for pol in KEY_POLICIES:
        print(f"\n[{pol}]")
        for m in hdr:
            r = representative[pol][m]
            sig = "SIG" if r["ci_excludes_zero"] else "n.s."
            rc = r["rel_change"]
            rc_s = f"{rc:+.1%}" if rc is not None else "n/a"
            print(f"  {m:<14} repD={r['representative_delta_mean']:+.5f} ({rc_s})  CI95=[{r['delta_mean_ci95'][0]:+.5f},{r['delta_mean_ci95'][1]:+.5f}] {sig}  imp={r['representative_frac_improved']:.0%}")
        if pol in fid_rep:
            print(f"  {'FID(weighted)':<14} repD={fid_rep[pol]['delta_FID_weighted']:+.4f}   R@1 repD={fid_rep[pol]['delta_R1_weighted']:+.4f}")
    print("\n--- no-harm (clean GT) delta (expect ~0) ---")
    cb = diagnostic["clean_noharm(no-harm ref)"]
    for pol in KEY_POLICIES:
        print(f"  {pol:<18} " + "  ".join(f"{m}={cb[pol][m]:+.5f}" for m in hdr))
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
