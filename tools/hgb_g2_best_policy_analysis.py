"""HGB G2 best policy 박제 — 사용자 directive (2026-05-25 Order 3.8).

부록 T 의 4 model 비교에서 HGB 가 G2 oracle closure 98% best.
본 도구는 HGB G2 의 정식 박제 5 항목:

  1. Mechanism: HGB 의 G2 best path distribution + sample-level decision.
  2. Paired test: HGB G2 NetGain vs B2-family-best (sample-level), B2-medium,
     B5 (있으면), sequence oracle.
  3. Action distribution: STOP / FL / BP / VS 의 ratio + strength.
  4. STOP precision/recall: HGB 가 oracle 의 STOP-best 를 잘 식별?
     - eval pair (training 의 G2 portion 의 held-out) 에서 STOP true vs predicted.
     - 또는 closed-loop 에서 oracle STOP-best 와 HGB selector_stop 의 매칭.
  5. B2-family-best 대비 비교: HGB G2 mean/median vs B2-val-best per sample.

CLI:
    python -m tools.hgb_g2_best_policy_analysis \\
        --training-data evals/snapshots/rl1_imitation_training_data_v2.json \\
        --alt-models evals/snapshots/rl1_alt_models_v1.json \\
        --b2-family-sweep evals/snapshots/baseline_b2_family_sweep_v1.json \\
        --oracle-g2 evals/snapshots/oracle_sequence_g2_v1.json \\
        --output evals/snapshots/hgb_g2_best_policy_v1.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from orchestrator.sequence_imitation import (
    ACTIONS,
    SequenceImitationSelector,
    pairs_to_arrays,
    split_train_eval_by_trial,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTION_LABELS = [f"{a[0]}/{a[1]}" for a in ACTIONS]


def _paired(arr_a: np.ndarray, arr_b: np.ndarray) -> dict:
    diff = arr_a - arr_b
    nonzero = diff[np.abs(diff) > 1e-12]
    if nonzero.size > 0:
        w_g = stats.wilcoxon(arr_a, arr_b, alternative="greater", zero_method="wilcox")
        w_two = stats.wilcoxon(arr_a, arr_b, alternative="two-sided", zero_method="wilcox")
        p_g = float(w_g.pvalue); p_two = float(w_two.pvalue)
    else:
        p_g = 1.0; p_two = 1.0
    d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 1e-15 else 0.0
    rng = np.random.default_rng(42)
    boot_med = np.empty(1000); boot_mean = np.empty(1000)
    for i in range(1000):
        idx = rng.integers(0, diff.size, size=diff.size)
        boot_med[i] = np.median(diff[idx]); boot_mean[i] = diff[idx].mean()
    return {
        "n": len(diff),
        "a_median": float(np.median(arr_a)), "a_mean": float(arr_a.mean()),
        "b_median": float(np.median(arr_b)), "b_mean": float(arr_b.mean()),
        "diff_median": float(np.median(diff)), "diff_mean": float(diff.mean()),
        "n_strict_a": int((diff > 1e-12).sum()), "n_tie": int((np.abs(diff) <= 1e-12).sum()),
        "n_loss_a": int((diff < -1e-12).sum()),
        "wilcoxon_p_greater": p_g, "wilcoxon_p_two_sided": p_two,
        "cohen_d_paired": d,
        "boot_ci_median": [float(np.percentile(boot_med, 2.5)), float(np.percentile(boot_med, 97.5))],
        "boot_ci_mean": [float(np.percentile(boot_mean, 2.5)), float(np.percentile(boot_mean, 97.5))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HGB G2 best policy analysis (Order 3.8)")
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--alt-models", type=Path, required=True,
                        help="rl1_alt_models_v1.json — HGB 의 G2 closed-loop 결과 포함.")
    parser.add_argument("--b2-family-sweep", type=Path, required=True)
    parser.add_argument("--oracle-g2", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--model-random-state", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # === Load ===
    with open(args.training_data, encoding="utf-8") as f:
        td = json.load(f)
    pairs = td["pairs"]
    train_pairs, eval_pairs = split_train_eval_by_trial(pairs, train_ratio=args.train_ratio, seed=args.seed)
    X_train, y_train = pairs_to_arrays(train_pairs)
    X_eval, y_eval = pairs_to_arrays(eval_pairs)

    # Train HGB.
    selector = SequenceImitationSelector(model_type="hist_gradient_boosting",
                                         random_state=args.model_random_state)
    train_metrics = selector.train(X_train, y_train)
    eval_acc = float(selector.model.score(X_eval, y_eval))
    print(f"[INFO] HGB trained: train_acc={train_metrics['train_accuracy']:.4f}, eval_acc={eval_acc:.4f}")

    # G2-only eval pairs.
    g2_eval_pairs = [p for p in eval_pairs if p["distribution"] == "g2_natural"]
    X_g2, y_g2 = pairs_to_arrays(g2_eval_pairs)
    g2_eval_acc = float(selector.model.score(X_g2, y_g2)) if len(y_g2) else 0.0
    print(f"[INFO] HGB G2-only eval (held-out pairs n={len(y_g2)}): action_acc={g2_eval_acc:.4f}")

    # G2 confusion matrix on held-out pairs.
    y_g2_pred = selector.model.predict(X_g2)
    cm_g2 = confusion_matrix(y_g2, y_g2_pred, labels=list(range(len(ACTIONS))))
    # STOP precision/recall.
    stop_id = 0
    tp_stop = int(cm_g2[stop_id, stop_id])
    fn_stop = int(cm_g2[stop_id, :].sum() - tp_stop)
    fp_stop = int(cm_g2[:, stop_id].sum() - tp_stop)
    support_stop = int(cm_g2[stop_id, :].sum())
    stop_precision = tp_stop / (tp_stop + fp_stop) if (tp_stop + fp_stop) > 0 else 0.0
    stop_recall = tp_stop / (tp_stop + fn_stop) if (tp_stop + fn_stop) > 0 else 0.0
    print(f"[INFO] HGB G2 STOP P/R (held-out pairs): P={stop_precision:.3f} R={stop_recall:.3f} support={support_stop}")

    # === Load alt-models snapshot for HGB closed-loop G2 results ===
    with open(args.alt_models, encoding="utf-8") as f:
        alt = json.load(f)
    hgb_result = next(r for r in alt["per_model_results"] if r["model_type"] == "hist_gradient_boosting")
    print(f"[INFO] HGB G2 closed-loop result (from alt-models): n={hgb_result['g2']['n_eval']}, "
          f"NetGain mean={hgb_result['g2']['netgain']['mean']:.5f}, median={hgb_result['g2']['netgain']['median']:.5f}")
    print(f"  oracle gap closure: {hgb_result['g2']['oracle_gap_closure']*100:.1f}%")
    print(f"  vs B2-medium d: {hgb_result['g2']['vs_b2_paired']['cohen_d_paired']:+.3f}")

    # HGB G2 per-sample NetGain — from alt-models snapshot.
    # alt 의 per_model_results 에 per-sample 이 없으므로, eval_trials_g2 와 NetGain 만 추출.
    # 보강: alt 의 eval_trials_g2 list + HGB run 다시 (per_sample 결과 저장 안 되어 있음).
    # → 별도 closed-loop rollout 재실행.

    # === HGB closed-loop G2 rollout 재실행 (per-sample 박제) ===
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from tools.baseline_rl1_run import _rollout_g2
    from correction_tools import BoneProjectionTool, FootLockTool, VelocitySmoothingTool
    from evaluators import DEFAULT_EVALUATORS
    from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1

    tools_by_name = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "BoneProjectionTool": BoneProjectionTool(),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
    }
    evaluators = list(DEFAULT_EVALUATORS)
    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
    G2_DIR = REPO_ROOT / "external_assets" / "g2_generated_v1"

    eval_trials_g2 = sorted({p["trial_id"] for p in g2_eval_pairs})
    print(f"\n[INFO] re-running HGB closed-loop G2 ({len(eval_trials_g2)} trials)...")
    hgb_g2_per_sample = []
    for tid in eval_trials_g2:
        path = G2_DIR / f"{tid}.npy"
        if not path.exists():
            continue
        motion = np.load(str(path)).astype(np.float64)
        res = _rollout_g2(
            trial_id=tid, motion=motion, selector=selector,
            tools_by_name=tools_by_name, evaluators=evaluators,
            netgain_weights=netgain_weights, k_max=5,
        )
        hgb_g2_per_sample.append(res)

    # === Per-sample paired test vs B2-family-best (val-best) ===
    with open(args.b2_family_sweep, encoding="utf-8") as f:
        b2_sweep = json.load(f)
    b2_val_best_by_trial = {s["trial_id"]: s["per_variant"]["val_best"]["netgain"]
                            for s in b2_sweep["per_sample_g2"]}
    b2_medium_by_trial = {s["trial_id"]: s["per_variant"]["medium"]["netgain"]
                          for s in b2_sweep["per_sample_g2"]}

    common = sorted({r["trial_id"] for r in hgb_g2_per_sample} & b2_val_best_by_trial.keys())
    hgb_arr = np.array([next(r["netgain"] for r in hgb_g2_per_sample if r["trial_id"] == t) for t in common])
    b2_val_arr = np.array([b2_val_best_by_trial[t] for t in common])
    b2_med_arr = np.array([b2_medium_by_trial[t] for t in common])

    paired_vs_val_best = _paired(hgb_arr, b2_val_arr)
    paired_vs_medium = _paired(hgb_arr, b2_med_arr)
    print(f"\n[Paired vs B2-val-best n={len(common)}]")
    print(f"  HGB median={paired_vs_val_best['a_median']:+.5f}, B2-val-best median={paired_vs_val_best['b_median']:+.5f}")
    print(f"  Δ median={paired_vs_val_best['diff_median']:+.5f}, mean={paired_vs_val_best['diff_mean']:+.5f}")
    print(f"  n_strict_hgb/tie/loss: {paired_vs_val_best['n_strict_a']}/{paired_vs_val_best['n_tie']}/{paired_vs_val_best['n_loss_a']}")
    print(f"  Wilcoxon p (greater)={paired_vs_val_best['wilcoxon_p_greater']:.5g}, d={paired_vs_val_best['cohen_d_paired']:+.3f}")
    print(f"\n[Paired vs B2-medium n={len(common)}]")
    print(f"  Δ median={paired_vs_medium['diff_median']:+.5f}, n_strict={paired_vs_medium['n_strict_a']}/{paired_vs_medium['n']}, d={paired_vs_medium['cohen_d_paired']:+.3f}")

    # === Paired vs oracle G2 ===
    with open(args.oracle_g2, encoding="utf-8") as f:
        oracle_g2 = json.load(f)
    oracle_g2_by_trial = {r["trial_id"]: r["best"]["netgain"] for r in oracle_g2["per_sample"] if r.get("best")}
    common_oracle = sorted({r["trial_id"] for r in hgb_g2_per_sample} & oracle_g2_by_trial.keys())
    hgb_arr_o = np.array([next(r["netgain"] for r in hgb_g2_per_sample if r["trial_id"] == t) for t in common_oracle])
    oracle_arr = np.array([oracle_g2_by_trial[t] for t in common_oracle])
    paired_vs_oracle = _paired(hgb_arr_o, oracle_arr)
    print(f"\n[Paired vs oracle G2 n={len(common_oracle)}]")
    print(f"  HGB median={paired_vs_oracle['a_median']:+.5f}, oracle median={paired_vs_oracle['b_median']:+.5f}")
    print(f"  Δ median={paired_vs_oracle['diff_median']:+.5f}, mean={paired_vs_oracle['diff_mean']:+.5f}")
    print(f"  n_strict_hgb: {paired_vs_oracle['n_strict_a']}, n_tie: {paired_vs_oracle['n_tie']}, n_loss: {paired_vs_oracle['n_loss_a']}")

    # === Action distribution + STOP-best matching with oracle ===
    hgb_first_action_dist = Counter(
        (r["sequence"][0][0] if r["sequence"] else "n/a") for r in hgb_g2_per_sample
    )
    hgb_length_dist = Counter(r["length"] for r in hgb_g2_per_sample)
    hgb_stop_reason_dist = Counter(r["stop_reason"] for r in hgb_g2_per_sample)

    # Oracle 의 STOP-best trial 과 HGB 의 STOP 결정 매칭.
    oracle_stop_trials = set()
    for r in oracle_g2["per_sample"]:
        if r.get("best") and r["best"]["length"] == 0:
            oracle_stop_trials.add(r["trial_id"])
    hgb_stop_trials = {r["trial_id"] for r in hgb_g2_per_sample
                       if r["sequence"] and r["sequence"][0][0] == "STOP"}
    intersection = oracle_stop_trials & hgb_stop_trials
    only_oracle_stop = oracle_stop_trials - hgb_stop_trials
    only_hgb_stop = hgb_stop_trials - oracle_stop_trials
    eval_set = {r["trial_id"] for r in hgb_g2_per_sample}
    oracle_stop_in_eval = oracle_stop_trials & eval_set

    # STOP-best matching metrics on closed-loop level.
    cl_stop_precision = len(intersection) / len(hgb_stop_trials) if hgb_stop_trials else 0.0
    cl_stop_recall = len(intersection) / len(oracle_stop_in_eval) if oracle_stop_in_eval else 0.0
    print(f"\n[Closed-loop STOP matching (eval set n={len(eval_set)})]")
    print(f"  HGB STOP: {len(hgb_stop_trials)}, Oracle STOP (in eval): {len(oracle_stop_in_eval)}")
    print(f"  intersection: {len(intersection)}")
    print(f"  STOP precision (HGB correct STOP): {cl_stop_precision:.3f}")
    print(f"  STOP recall (Oracle STOP matched): {cl_stop_recall:.3f}")

    # === Summary ===
    summary = {
        "schema_version": "1.0.0",
        "record_type": "hgb_g2_best_policy_analysis",
        "training_data": str(args.training_data),
        "model_type": "hist_gradient_boosting",
        "model_random_state": args.model_random_state,
        "train_ratio": args.train_ratio, "split_seed": args.seed,
        "n_train_pairs": len(y_train), "n_eval_pairs": len(y_eval),
        "n_g2_eval_pairs": len(y_g2),
        "n_g2_eval_trials": len(eval_trials_g2),
        "train_metrics": train_metrics,
        "eval_action_accuracy_all": eval_acc,
        "eval_action_accuracy_g2_only": g2_eval_acc,
        "g2_confusion_matrix": cm_g2.tolist(),
        "g2_stop_held_out_pair_precision_recall": {
            "support": support_stop,
            "tp": tp_stop, "fp": fp_stop, "fn": fn_stop,
            "precision": float(stop_precision),
            "recall": float(stop_recall),
        },
        "closed_loop_g2": {
            "n": len(hgb_g2_per_sample),
            "per_sample": hgb_g2_per_sample,
            "first_action_distribution": dict(hgb_first_action_dist),
            "length_distribution": dict(hgb_length_dist),
            "stop_reason_distribution": dict(hgb_stop_reason_dist),
        },
        "stop_matching_closed_loop": {
            "n_eval": len(eval_set),
            "n_hgb_stop": len(hgb_stop_trials),
            "n_oracle_stop_in_eval": len(oracle_stop_in_eval),
            "n_intersection": len(intersection),
            "stop_precision": float(cl_stop_precision),
            "stop_recall": float(cl_stop_recall),
            "intersection_trials": sorted(intersection),
            "only_hgb_stop_trials": sorted(only_hgb_stop),
            "only_oracle_stop_in_eval": sorted(only_oracle_stop & eval_set),
        },
        "paired_vs_b2_val_best": paired_vs_val_best,
        "paired_vs_b2_medium": paired_vs_medium,
        "paired_vs_oracle_g2": paired_vs_oracle,
        "alt_models_hgb_g2_summary": hgb_result["g2"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
