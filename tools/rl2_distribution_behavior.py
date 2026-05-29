"""Step F-4 (사용자 directive 2026-05-28): distribution별 정책 행동 상세 분석.

사용자 directive:
> "RL-2가 각 분포에서 어떻게 행동하는지 설명해야. G2: STOP correctness, unnecessary
>  correction rate, weak correction rate. Synthetic: multi-step accuracy, tool sequence
>  accuracy, strength error, safe oracle gap closure."

추가 appendix:
> "B2-small 94% violation 의 bone별 CV distribution + smoothing 후 bone length mean/std."

CLI:
    python -m tools.rl2_distribution_behavior --seeds 0,1,2 \
        --output evals/snapshots/rl2_distribution_behavior_v1.json
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

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from skeleton_normalizer.canonical_smpl_22 import T2M_KINEMATIC_CHAIN
from tools.synthetic_injection import inject_foot_floating, inject_jitter
from tools.safe_sequence_oracle_run import _gate_scores, _gate_violation
from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, REAL_DISTRIBUTION, common_snapshot_metadata
from tools.rl2_build_training_data import ACTION_LIST, STRENGTHS_5LEVEL, STRENGTHS_3LEVEL, TOOLS_ORDER
from tools.rl2_train_imitation import _flatten_state, _sample_level_split, _build_models
from tools.rl2_closed_loop_eval import _run_policy_closed_loop, _idx_to_action

# STRENGTH_RANK is grid-set in main() (default 5-level for import-time compatibility).
STRENGTH_RANK = {s: i for i, s in enumerate(STRENGTHS_5LEVEL)}
TOOL_BY_NAME = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}
# Bone pairs.
BONES = []
for chain in T2M_KINEMATIC_CHAIN:
    for a, b in zip(chain[:-1], chain[1:]):
        BONES.append((a, b))
BONES = list(dict.fromkeys(BONES))


def _seq_to_actions(seq):
    """Oracle sequence → list of action names (skip terminals)."""
    out = []
    for s in seq:
        if s[0] in ("STOP", "APPLY_FAIL", "SCORE_VIOLATION_ROLLBACK"):
            break
        out.append(f"{s[0]}|{s[2]}")
    return out


def _policy_actions(trace_actions):
    """policy trace 의 action list (gate accept 만)."""
    return [a["action"] for a in trace_actions if a.get("gate") == "accept"]


def _bone_cv(motion: np.ndarray) -> np.ndarray:
    """Per-bone CV (std/mean) across frames."""
    cvs = []
    for a, b in BONES:
        lens = np.linalg.norm(motion[:, a, :] - motion[:, b, :], axis=-1)
        mean_l = float(np.mean(lens))
        cvs.append(float(np.std(lens) / mean_l) if mean_l > 1e-9 else 0.0)
    return np.array(cvs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_imitation_dataset_v1.json")
    parser.add_argument("--g2-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_g2_hml3d_test300_v1.json")
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--synthetic-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_synthetic_severe_v1.json")
    parser.add_argument("--synthetic-data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--calibration", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_distribution_behavior_v1.json")
    args = parser.parse_args()

    data = json.load(open(args.dataset, encoding="utf-8"))
    rows = data["rows"]
    # Grid-aware: derive action_list / strengths / rank from dataset.
    global STRENGTH_RANK
    ds_action_list = data.get("action_list", ACTION_LIST)
    ds_strengths = STRENGTHS_3LEVEL if len(ds_action_list) == 10 else STRENGTHS_5LEVEL
    STRENGTH_RANK = {s: i for i, s in enumerate(ds_strengths)}
    print(f"[INFO] grid: {len(ds_action_list)} actions, strengths={ds_strengths}")
    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_thresholds = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                       if calib["summary"][n].get("n", 0) > 0}
    seeds = [int(s) for s in args.seeds.split(",")]
    evaluators = list(DEFAULT_EVALUATORS)
    gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)
    g2_oracle = {p["trial_id"]: p for p in json.load(open(args.g2_oracle, encoding="utf-8"))["per_sample"]}
    syn_oracle = {p["trial_id"]: p for p in json.load(open(args.synthetic_oracle, encoding="utf-8"))["per_sample"]}

    models = _build_models()
    X_all = np.array([_flatten_state(r["state"], False) for r in rows])
    y_all = np.array([r["action_idx"] for r in rows])

    # Per-distribution, per-model: aggregate behavior stats.
    rows_g2 = defaultdict(list)   # model -> list of per-sample dict
    rows_syn = defaultdict(list)

    for seed in seeds:
        train_idx, eval_idx = _sample_level_split(rows, seed)
        eval_g2 = sorted({rows[i]["sample_id"] for i in eval_idx if rows[i]["distribution"] == "g2"})
        eval_syn = sorted({rows[i]["sample_id"] for i in eval_idx if rows[i]["distribution"] == "synthetic"})
        trained = {mn: models[mn]().fit(X_all[train_idx], y_all[train_idx]) for mn in ("RF", "HGB")}

        for tid in eval_g2:
            npy = args.g2_batch_dir / f"{tid}.npy"
            if not npy.exists():
                continue
            ref = np.load(str(npy)).astype(np.float64)
            sb = g2_oracle[tid].get("safe_best")
            oracle_actions = _seq_to_actions(sb["sequence"]) if sb and sb["length"] > 0 else []
            for mn in ("RF", "HGB"):
                _, trace = _run_policy_closed_loop(ref, trained[mn], evaluators, gate_evaluators,
                                                    gate_thresholds, dist_tag=0, max_depth=args.max_depth,
                                                    strengths=ds_strengths, action_list=ds_action_list)
                policy_acts = _policy_actions(trace["actions"])
                rows_g2[mn].append({
                    "trial_id": tid,
                    "oracle_is_stop": len(oracle_actions) == 0,
                    "policy_is_stop": len(policy_acts) == 0,
                    "oracle_actions": oracle_actions, "policy_actions": policy_acts,
                    "stopped_reason": trace["stopped_reason"],
                })

        for tid in eval_syn:
            npy = args.synthetic_data_dir / f"{tid}.npy"
            if not npy.exists():
                continue
            clean = np.load(str(npy)).astype(np.float64)
            m1 = inject_foot_floating(clean, lift_height=0.08, seed=args.synthetic_seed)
            corrupted = inject_jitter(m1, noise_std=0.05, seed=args.synthetic_seed + 1000)
            sb = syn_oracle[tid].get("safe_best")
            oracle_actions = _seq_to_actions(sb["sequence"]) if sb and sb["length"] > 0 else []
            for mn in ("RF", "HGB"):
                _, trace = _run_policy_closed_loop(corrupted, trained[mn], evaluators, gate_evaluators,
                                                    gate_thresholds, dist_tag=1, max_depth=args.max_depth,
                                                    strengths=ds_strengths, action_list=ds_action_list)
                policy_acts = _policy_actions(trace["actions"])
                rows_syn[mn].append({
                    "trial_id": tid,
                    "oracle_is_stop": len(oracle_actions) == 0,
                    "policy_is_stop": len(policy_acts) == 0,
                    "oracle_actions": oracle_actions, "policy_actions": policy_acts,
                    "stopped_reason": trace["stopped_reason"],
                })
        print(f"[seed {seed}] done")

    # === G2 aggregation ===
    def _agg_g2(model_rows):
        n = len(model_rows)
        if n == 0:
            return {}
        # STOP correctness (oracle STOP & policy STOP).
        oracle_stop = sum(1 for r in model_rows if r["oracle_is_stop"])
        oracle_corr = n - oracle_stop
        # Confusion matrix.
        tp_stop = sum(1 for r in model_rows if r["oracle_is_stop"] and r["policy_is_stop"])
        fn_stop = sum(1 for r in model_rows if r["oracle_is_stop"] and not r["policy_is_stop"])  # unnecessary correction
        fp_stop = sum(1 for r in model_rows if not r["oracle_is_stop"] and r["policy_is_stop"])  # missed correction
        tp_corr = sum(1 for r in model_rows if not r["oracle_is_stop"] and not r["policy_is_stop"])
        # Weak correction rate (policy non-STOP samples with xsmall/small5 first action).
        weak_count = 0
        non_stop_pol = 0
        for r in model_rows:
            if not r["policy_is_stop"] and r["policy_actions"]:
                non_stop_pol += 1
                first_strength = r["policy_actions"][0].split("|")[1]
                if STRENGTH_RANK[first_strength] <= 1:  # xsmall(0) or small5(1)
                    weak_count += 1
        return {
            "n": n,
            "oracle_stop_rate": oracle_stop / n,
            "policy_stop_rate": sum(1 for r in model_rows if r["policy_is_stop"]) / n,
            "stop_correctness_recall": tp_stop / max(oracle_stop, 1),  # of samples where oracle says STOP
            "stop_correctness_precision": tp_stop / max(tp_stop + fp_stop, 1),
            "unnecessary_correction_rate": fn_stop / max(oracle_stop, 1),  # oracle STOP but policy corrected
            "missed_correction_rate": fp_stop / max(oracle_corr, 1),  # oracle says correct but policy STOP
            "weak_correction_rate_of_corrections": weak_count / max(non_stop_pol, 1),
        }

    # === Synthetic aggregation ===
    def _agg_syn(model_rows):
        n = len(model_rows)
        if n == 0:
            return {}
        # multi-step accuracy: policy length >= 2 when oracle length >= 2.
        oracle_multi = sum(1 for r in model_rows if len(r["oracle_actions"]) >= 2)
        policy_multi = sum(1 for r in model_rows if len(r["policy_actions"]) >= 2)
        # tool sequence match (ordered, first-N).
        tool_match_step0 = 0
        tool_match_step1 = 0
        strength_match_step0 = 0
        strength_dist_sum = 0
        strength_n = 0
        for r in model_rows:
            oa, pa = r["oracle_actions"], r["policy_actions"]
            for t in range(min(len(oa), len(pa))):
                ot, ost = oa[t].split("|")
                pt, pst = pa[t].split("|")
                if t == 0:
                    if ot == pt:
                        tool_match_step0 += 1
                        if ost == pst:
                            strength_match_step0 += 1
                if t == 1:
                    if ot == pt:
                        tool_match_step1 += 1
                if ot == pt:
                    strength_dist_sum += abs(STRENGTH_RANK[ost] - STRENGTH_RANK[pst])
                    strength_n += 1
        return {
            "n": n,
            "oracle_multistep_rate": oracle_multi / n,
            "policy_multistep_rate": policy_multi / n,
            "tool_match_step0": tool_match_step0 / n,
            "tool_match_step1_of_multistep": tool_match_step1 / max(oracle_multi, 1),
            "strength_match_step0_given_tool": strength_match_step0 / max(tool_match_step0, 1),
            "mean_strength_distance_given_tool_match": (strength_dist_sum / strength_n) if strength_n > 0 else 0.0,
        }

    g2_agg = {mn: _agg_g2(rows_g2[mn]) for mn in ("RF", "HGB")}
    syn_agg = {mn: _agg_syn(rows_syn[mn]) for mn in ("RF", "HGB")}

    # === B2 strength bone CV appendix ===
    print("\n[INFO] computing B2 strength bone CV appendix (synthetic n=10 sample)...")
    rng = np.random.default_rng(0)
    syn_files = sorted(args.synthetic_data_dir.glob("*.npy"))
    sample_paths = rng.choice(syn_files, size=10, replace=False)
    b2_bone_appendix = {"corrupted": [], "B2-small": [], "B2-medium": [], "B2-large": []}
    for p in sample_paths:
        clean = np.load(str(p)).astype(np.float64)
        if clean.ndim != 3 or clean.shape[1] != 22:
            continue
        m1 = inject_foot_floating(clean, lift_height=0.08, seed=args.synthetic_seed)
        corrupted = inject_jitter(m1, noise_std=0.05, seed=args.synthetic_seed + 1000)
        cv_corr = _bone_cv(corrupted)
        b2_bone_appendix["corrupted"].append({"mean": float(cv_corr.mean()),
                                              "max": float(cv_corr.max()), "p95": float(np.percentile(cv_corr, 95))})
        for st in ("small", "medium", "large"):
            out, _ = TOOL_BY_NAME["VelocitySmoothingTool"].apply(
                corrupted, target_part="full_body", target_joints=[], frame_range=(0, corrupted.shape[0]-1), strength=st)
            cv = _bone_cv(out)
            b2_bone_appendix[f"B2-{st}"].append({"mean": float(cv.mean()),
                                                  "max": float(cv.max()), "p95": float(np.percentile(cv, 95))})

    def _avg(lst, key):
        return float(np.mean([d[key] for d in lst])) if lst else 0.0

    appendix_summary = {}
    for k, lst in b2_bone_appendix.items():
        appendix_summary[k] = {
            "mean_bone_cv": _avg(lst, "mean"),
            "max_bone_cv": _avg(lst, "max"),
            "p95_bone_cv": _avg(lst, "p95"),
            "n_samples": len(lst),
        }

    out = {
        "schema_version": "1.0.0", "record_type": "rl2_distribution_behavior",
        "task_id": "rl2_distribution_behavior_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "rl2_distribution_behavior_v1",
            oracle_type="sequence",
            action_grid="5-level",
            stage="RL-2-analysis",
            evidence_tier=[REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
            evaluators=evaluators,
            gate_evaluators=gate_evaluators,
        ),
        "seeds": seeds,
        "g2_behavior": g2_agg,
        "synthetic_behavior": syn_agg,
        "b2_bone_cv_appendix": appendix_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Step F-4: Distribution Behavior ===")
    print("\n[G2 behavior]")
    for mn, a in g2_agg.items():
        print(f"  {mn}: oracle_STOP={a['oracle_stop_rate']*100:.0f}%  policy_STOP={a['policy_stop_rate']*100:.0f}%  "
              f"STOP_recall={a['stop_correctness_recall']*100:.0f}%  STOP_prec={a['stop_correctness_precision']*100:.0f}%")
        print(f"        unnec_correction={a['unnecessary_correction_rate']*100:.0f}%  "
              f"missed_correction={a['missed_correction_rate']*100:.0f}%  "
              f"weak_strength={a['weak_correction_rate_of_corrections']*100:.0f}% (of policy corrections)")
    print("\n[Synthetic behavior]")
    for mn, a in syn_agg.items():
        print(f"  {mn}: oracle_multi={a['oracle_multistep_rate']*100:.0f}%  policy_multi={a['policy_multistep_rate']*100:.0f}%")
        print(f"        tool_match_step0={a['tool_match_step0']*100:.0f}%  "
              f"strength_match|tool={a['strength_match_step0_given_tool']*100:.0f}%  "
              f"mean_strength_dist={a['mean_strength_distance_given_tool_match']:.2f}")
    print("\n[B2 bone CV appendix (synthetic n=10)]")
    for k, v in appendix_summary.items():
        print(f"  {k:<12} mean_CV={v['mean_bone_cv']:.5f}  max_CV={v['max_bone_cv']:.5f}  p95_CV={v['p95_bone_cv']:.5f}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
