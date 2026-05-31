"""G2 evidence 강화 (사용자 directive 2026-06-01): per-state paired statistical tests.

사용자 directive 박제:
> "G1 보류, G2 evidence 강화 → 그 다음 MDM 신규 구축. 먼저 G2에서 통계 검정, 결과표,
>  limitation, claim 범위를 정리하면 논문 뼈대가 잡힌다."

Step 6 ablation 은 aggregate 만 저장 → 본 도구는 G2 holdout state 별 per-state NetGain 을
재계산 + **paired statistical test** (AGENTS.md §3-9: 단일 trial 금지, 분포 + paired test 의무):
  - Wilcoxon signed-rank (M0 vs each baseline, per-state NetGain).
  - matched-pairs effect size (Cohen's d on paired diffs + rank-biserial).
  - bootstrap 95% CI of median paired difference (N=1000).
  - physical violation rate (paired McNemar-style count).

비교: M0 (primary, broad-support) vs {random+gate, heuristic+gate, STOP, M1, M2, dense_oracle}.
대상 holdout: g2_stress_holdout, g2_natural_holdout, clean_noharm_holdout (real-distribution).

NOTE: NetGain = Category C internal routing reward (metric_provenance §4-1-1). 본 paired test 는
"learned policy 가 rule-based/random 보다 나은 routing decision 을 하는가" (RQ3-adjacent) 의
근거. final quality 는 Step 6 part 2 의 standard metric (distributional, Category A) 가 authoritative.

CLI:
    python -m tools.rl2_g2_paired_stats --seeds 0 \
        --output evals/snapshots/rl2_g2_paired_stats_v1.json
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

from correction_tools import BoneProjectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from tools.synthetic_injection import inject_foot_floating, inject_jitter
from tools.safe_sequence_oracle_run import _gate_scores, _gate_violation
from tools.harness_metadata import REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC, common_snapshot_metadata
from tools.rl2_transition_g2_real_build import _resolve_g2
from tools.rl2_closed_loop_ablation import (
    _stop_only, _random_gate, _heuristic_gate, _q_policy_gate, _dense_oracle_step1,
    _netgain, _gate_decision, PROTOCOL_A_DISTS,
)

HOLDOUTS_REAL = ("g2_stress_holdout", "g2_natural_holdout", "clean_noharm_holdout")
HOLDOUTS_DIAG = ("synthetic_diag_holdout",)
BASELINES = ("random_gate", "heuristic_gate", "STOP_only", "M1", "M2", "dense_oracle_step1")


def _load_motion(sid, dist, existing_pool, balanced_pool, data_dir, synthetic_seed):
    if dist == "g2_natural":
        pool_dir, stem = _resolve_g2(sid, existing_pool, balanced_pool)
        return np.load(str(pool_dir / f"{stem}.npy")).astype(np.float64), None
    if dist == "clean":
        m = np.load(str(data_dir / f"{sid}.npy")).astype(np.float64)
        return m, m
    if dist == "synthetic_severe":
        clean = np.load(str(data_dir / f"{sid}.npy")).astype(np.float64)
        m1 = inject_foot_floating(clean, lift_height=0.08, seed=synthetic_seed)
        return inject_jitter(m1, noise_std=0.05, seed=synthetic_seed + 1000), clean
    return None, None


def _wilcoxon(diffs):
    """Wilcoxon signed-rank p-value (two-sided). diffs = M0 - baseline per-state."""
    from scipy.stats import wilcoxon
    nz = [d for d in diffs if abs(d) > 1e-12]
    if len(nz) < 5:
        return {"p_value": None, "n_nonzero": len(nz), "note": "too few nonzero pairs"}
    try:
        stat, p = wilcoxon(nz)
        return {"statistic": float(stat), "p_value": float(p), "n_nonzero": len(nz)}
    except Exception as e:
        return {"p_value": None, "error": str(e), "n_nonzero": len(nz)}


def _effect_size(diffs):
    """paired Cohen's d + rank-biserial correlation."""
    d = np.array(diffs)
    cohen_d = float(np.mean(d) / (np.std(d, ddof=1) + 1e-12)) if len(d) > 1 else 0.0
    # rank-biserial = (n_pos - n_neg) / n_nonzero.
    nz = d[np.abs(d) > 1e-12]
    rbc = float((np.sum(nz > 0) - np.sum(nz < 0)) / max(len(nz), 1)) if len(nz) else 0.0
    return {"cohens_d_paired": cohen_d, "rank_biserial": rbc,
            "n_M0_better": int(np.sum(nz > 0)), "n_baseline_better": int(np.sum(nz < 0)),
            "n_tie": int(len(d) - len(nz))}


def _bootstrap_ci(diffs, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.array(diffs)
    if len(d) < 3:
        return {"median_diff": float(np.median(d)) if len(d) else 0.0, "ci95": [None, None]}
    meds = []
    for _ in range(n_boot):
        samp = d[rng.integers(0, len(d), len(d))]
        meds.append(np.median(samp))
    return {"median_diff": float(np.median(d)), "mean_diff": float(np.mean(d)),
            "ci95": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path,
                        default=REPO_ROOT / "evals" / "models" / "rl2_q_policies_stage2_v1.joblib")
    parser.add_argument("--g2-real", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_g2_real_stage2_v1.json")
    parser.add_argument("--existing-pool", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--balanced-pool", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_balanced300_seed20260531")
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--calibration", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=20260601)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--u-grid", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--include-diag", action="store_true", help="synthetic_diag_holdout 도 포함 (appendix).")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_g2_paired_stats_v1.json")
    args = parser.parse_args()

    u_grid = [round(float(x), 4) for x in args.u_grid.split(",")]
    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_thresholds = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                       if calib["summary"][n].get("n", 0) > 0}
    alpha = float(CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1["alpha"])
    evaluators = list(DEFAULT_EVALUATORS); gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)

    import joblib
    models = joblib.load(str(args.models))
    g2_data = json.load(open(args.g2_real, encoding="utf-8"))
    state_map = g2_data["states"]
    transitions_by_state = defaultdict(list)
    for t in g2_data["transitions"]:
        transitions_by_state[t["state_id"]].append(t)

    holdouts = list(HOLDOUTS_REAL) + (list(HOLDOUTS_DIAG) if args.include_diag else [])
    results = {}
    for ho in holdouts:
        sids = [s for s in state_map if state_map[s]["split"] == ho]
        if not sids:
            continue
        print(f"\n[INFO] {ho}: {len(sids)} states")
        # Per-state NetGain for each policy.
        ng = defaultdict(list); viol = defaultdict(list); sid_order = []
        for i, sid in enumerate(sids, 1):
            dist = state_map[sid]["distribution"]
            motion0, ref = _load_motion(sid, dist, args.existing_pool, args.balanced_pool,
                                        args.data_dir, args.synthetic_seed)
            if motion0 is None:
                continue
            common_ref = ref if dist in PROTOCOL_A_DISTS else motion0
            rng = np.random.default_rng(args.random_seed + (hash(sid) % (2**31)))
            outs = {
                "M0": _q_policy_gate(motion0, *models["M0"][:2], models["M0"][2], evaluators,
                                     gate_evaluators, gate_thresholds, u_grid, args.max_depth),
                "M1": _q_policy_gate(motion0, *models["M1"][:2], models["M1"][2], evaluators,
                                     gate_evaluators, gate_thresholds, u_grid, args.max_depth),
                "M2": _q_policy_gate(motion0, *models["M2"][:2], models["M2"][2], evaluators,
                                     gate_evaluators, gate_thresholds, u_grid, args.max_depth),
                "random_gate": _random_gate(motion0, evaluators, gate_evaluators, gate_thresholds,
                                            u_grid, rng, args.max_depth, 3),
                "heuristic_gate": _heuristic_gate(motion0, evaluators, gate_evaluators, gate_thresholds,
                                                  None, None, args.max_depth),
                "STOP_only": _stop_only(motion0),
                "dense_oracle_step1": _dense_oracle_step1(motion0, transitions_by_state[sid], dist,
                                                          common_ref, evaluators, alpha),
            }
            init_gate = _gate_scores(motion0, gate_evaluators)
            for pname, out in outs.items():
                ng[pname].append(_netgain(motion0, out["final_motion"], common_ref, dist, evaluators, alpha))
                _, _, v = _gate_decision(out["final_motion"], init_gate, gate_evaluators, gate_thresholds)
                viol[pname].append(1.0 if v else 0.0)
            sid_order.append(sid)
            if i % 30 == 0:
                print(f"   {ho} {i}/{len(sids)}")

        # Paired comparisons: M0 vs each baseline.
        m0 = np.array(ng["M0"])
        comp = {}
        for base in BASELINES:
            b = np.array(ng[base])
            diffs = (m0 - b).tolist()
            comp[base] = {
                "M0_mean_netgain": float(np.mean(m0)), "baseline_mean_netgain": float(np.mean(b)),
                "mean_paired_diff": float(np.mean(m0 - b)),
                "wilcoxon": _wilcoxon(diffs),
                "effect_size": _effect_size(diffs),
                "bootstrap": _bootstrap_ci(diffs, seed=args.random_seed),
                "M0_violation_rate": float(np.mean(viol["M0"])),
                "baseline_violation_rate": float(np.mean(viol[base])),
            }
        results[ho] = {
            "n_states": len(sid_order),
            "policy_mean_netgain": {p: float(np.mean(v)) for p, v in ng.items()},
            "policy_violation_rate": {p: float(np.mean(v)) for p, v in viol.items()},
            "M0_vs_baseline": comp,
        }

    out = {
        "schema_version": "1.0.0", "record_type": "rl2_g2_paired_stats",
        "task_id": "rl2_g2_paired_stats_v1",
        **common_snapshot_metadata(
            split_id="rl2_g2_paired_stats_v1", oracle_type="closed_loop_single_step_dense",
            action_grid="continuous-dense-u11", stage="G2-evidence-paired-stats",
            evidence_tier=[REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
            evaluators=evaluators, gate_evaluators=gate_evaluators,
        ),
        "directive": "M0 (primary) vs baselines per-state NetGain paired test (Wilcoxon + effect size + bootstrap CI).",
        "netgain_status": "Category C internal routing reward (metric_provenance §4-1-1). final quality = Step 6 part 2 standard metric.",
        "netgain_protocol": {"g2_natural": "B (vs original)", "clean/synthetic": "A (vs clean)"},
        "max_depth": args.max_depth, "u_grid": u_grid,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== G2 Paired Statistical Tests (M0 vs baselines, per-state NetGain) ===")
    for ho in holdouts:
        r = results.get(ho)
        if not r: continue
        print(f"\n[{ho}] n={r['n_states']}")
        print(f"  M0 mean NetGain = {r['policy_mean_netgain']['M0']:+.4f}, viol = {r['policy_violation_rate']['M0']*100:.1f}%")
        print(f"  {'vs baseline':<22} {'M0-base Δ':<12} {'Wilcoxon p':<14} {'Cohen d':<10} {'M0>base':<9} {'boot CI95'}")
        for base in BASELINES:
            c = r["M0_vs_baseline"][base]
            p = c["wilcoxon"].get("p_value")
            p_s = f"{p:.2e}" if p is not None else "—"
            d = c["effect_size"]["cohens_d_paired"]
            nb = c["effect_size"]["n_M0_better"]; nw = c["effect_size"]["n_baseline_better"]
            ci = c["bootstrap"]["ci95"]
            ci_s = f"[{ci[0]:+.4f},{ci[1]:+.4f}]" if ci[0] is not None else "—"
            print(f"  {base:<22} {c['mean_paired_diff']:<+12.4f} {p_s:<14} {d:<+10.3f} {nb}/{nw:<7} {ci_s}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
