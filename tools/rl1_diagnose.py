"""RL-1 진단 도구 — 사용자 directive (2026-05-25 roadmap step 1).

목표: RL-1 실패 원인 분리 (data / state / model / imitation 자체).

본 도구 4 진단:
  (B) multi-seed (model_random_state ∈ {1..5}) — variance 측정.
  (C) state feature ablation — 4 variants:
       full16 / no_delta (drop score_delta 3-5) / no_prev (drop prev_tool/strength,
       only scores+delta+step+budget = 8-dim) / scores_only (3-dim, ablation
       extreme).
  (D) action confusion matrix on eval (state, action) pairs.
  (E) failure trace — score_violation_rollback sample 의 step-by-step
       (state → predicted action + proba top-3 → applied → score delta).

NOTE: closed-loop NetGain 측정은 baseline_rl1_run.py 이 담당. 본 도구는 정적
분석 + per-pair eval (closed-loop 미실행). action 학습 capacity 진단 우선.

CLI:
    python -m tools.rl1_diagnose \\
        --training-data evals/snapshots/rl1_imitation_training_data_v1.json \\
        --rl1-snapshot evals/snapshots/baseline_rl1_imitation_v1.json \\
        --output-dir evals/snapshots/rl1_diagnose_v1 \\
        --seed 42 --train-ratio 0.7
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from orchestrator.sequence_imitation import (
    ACTIONS,
    SequenceImitationSelector,
    pairs_to_arrays,
    split_train_eval_by_trial,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: State feature ablation variants — index list (16-dim full).
ABLATIONS = {
    "full16": list(range(16)),
    "no_delta": [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # 13-dim (drop 3-5 delta).
    "no_prev": [0, 1, 2, 3, 4, 5, 6, 15],  # 8-dim (drop 7-14 prev tool+strength one-hot).
    "scores_only": [0, 1, 2],  # 3-dim — current scores only.
}

ACTION_LABELS = [f"{a[0][:2]}/{a[1][:3]}" if a[0] != "STOP" else "STOP" for a in ACTIONS]


def _train_eval(X_train, y_train, X_eval, y_eval, random_state: int) -> dict[str, Any]:
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    train_acc = float(clf.score(X_train, y_train))
    eval_acc = float(clf.score(X_eval, y_eval))
    return {
        "train_accuracy": train_acc, "eval_accuracy": eval_acc,
        "model_random_state": random_state,
        "n_train": len(y_train), "n_eval": len(y_eval),
        "classes_seen": [int(c) for c in clf.classes_],
        "_clf": clf,
    }


def _confusion_per_class(clf, X_eval, y_eval) -> dict[str, Any]:
    y_pred = clf.predict(X_eval)
    labels = list(range(len(ACTIONS)))
    cm = sk_confusion_matrix(y_eval, y_pred, labels=labels)
    # Per-class metrics.
    per_class = []
    for i in labels:
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        support = int(cm[i, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class.append({
            "action_id": i, "action": f"{ACTIONS[i][0]}/{ACTIONS[i][1]}",
            "support": support, "tp": tp, "fp": fp, "fn": fn,
            "precision": float(precision), "recall": float(recall),
        })
    return {"matrix": cm.tolist(), "per_class": per_class}


def _proba_top_k(clf, x: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    if not hasattr(clf, "predict_proba"):
        return []
    p = clf.predict_proba(x.reshape(1, -1))[0]
    full = [0.0] * len(ACTIONS)
    for cls_idx, prob in zip(clf.classes_, p):
        full[int(cls_idx)] = float(prob)
    top = sorted(enumerate(full), key=lambda kv: kv[1], reverse=True)[:k]
    return top


def _failure_trace(clf, pairs_by_trial: dict[str, list[dict[str, Any]]],
                   fail_trials: list[str]) -> dict[str, Any]:
    """Failure sample 의 step-by-step trace.

    Note: 본 trace 는 training data 의 (state, action) pair 위에서 predict (즉
    oracle 의 simulated state 에서 RL-1 predicts what). closed-loop 의 motion
    actually applied state 와는 약간 다름 — RL-1 의 (state, action) supervised
    학습 자체 의 진단.
    """
    out = {}
    for trial in fail_trials:
        pairs = pairs_by_trial.get(trial, [])
        steps = []
        for p in pairs:
            x = np.array(p["state"], dtype=np.float64)
            true_id = int(p["action_id"])
            pred_id = int(clf.predict(x.reshape(1, -1))[0])
            top3 = _proba_top_k(clf, x, k=3)
            steps.append({
                "step_index": int(p["step_index"]),
                "true_action": f"{ACTIONS[true_id][0]}/{ACTIONS[true_id][1]}",
                "true_action_id": true_id,
                "predicted_action": f"{ACTIONS[pred_id][0]}/{ACTIONS[pred_id][1]}",
                "predicted_action_id": pred_id,
                "match": pred_id == true_id,
                "top3_proba": [(f"{ACTIONS[i][0]}/{ACTIONS[i][1]}", float(prob)) for i, prob in top3],
                "current_scores": {k: float(v) for k, v in p["current_scores"].items()},
            })
        out[trial] = steps
    return out


def _plot_confusion(cm: np.ndarray, save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(ACTIONS)))
    ax.set_yticks(range(len(ACTIONS)))
    ax.set_xticklabels(ACTION_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ACTION_LABELS, fontsize=8)
    ax.set_xlabel("Predicted action")
    ax.set_ylabel("True action")
    ax.set_title(title, fontsize=10)
    # Annotate cells.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                        color=color, fontsize=7)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-1 diagnosis (multi-seed + state ablation + confusion + failure trace)")
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--rl1-snapshot", type=Path, required=True,
                        help="for failure sample identification.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--multi-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    with open(args.training_data, encoding="utf-8") as f:
        td = json.load(f)
    pairs = td["pairs"]
    print(f"[INFO] training data: {len(pairs)} pairs")

    train_pairs, eval_pairs = split_train_eval_by_trial(
        pairs, train_ratio=args.train_ratio, seed=args.seed,
    )
    X_train, y_train = pairs_to_arrays(train_pairs)
    X_eval, y_eval = pairs_to_arrays(eval_pairs)
    print(f"[INFO] split: train {len(y_train)} pairs, eval {len(y_eval)} pairs")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # === (B) Multi-seed (full16) ===
    print(f"\n[B] Multi-seed (model_random_state ∈ {args.multi_seeds}, state=full16) ===")
    multi_seed_results = []
    for rs in args.multi_seeds:
        r = _train_eval(X_train, y_train, X_eval, y_eval, random_state=rs)
        del r["_clf"]
        multi_seed_results.append(r)
        print(f"  seed={rs}: train_acc={r['train_accuracy']:.4f}, eval_acc={r['eval_accuracy']:.4f}")
    eval_accs = np.array([r["eval_accuracy"] for r in multi_seed_results])
    multi_seed_agg = {
        "n_seeds": len(args.multi_seeds), "seeds": args.multi_seeds,
        "eval_accuracy_mean": float(eval_accs.mean()),
        "eval_accuracy_std": float(eval_accs.std(ddof=1)) if len(eval_accs) > 1 else 0.0,
        "eval_accuracy_min": float(eval_accs.min()),
        "eval_accuracy_max": float(eval_accs.max()),
        "per_seed": multi_seed_results,
    }
    print(f"  → eval_accuracy mean ± std: {multi_seed_agg['eval_accuracy_mean']:.4f} ± {multi_seed_agg['eval_accuracy_std']:.4f}")
    print(f"  → range: [{multi_seed_agg['eval_accuracy_min']:.4f}, {multi_seed_agg['eval_accuracy_max']:.4f}]")

    # === (C) State feature ablation ===
    print(f"\n[C] State feature ablation (4 variants, model_random_state=1) ===")
    ablation_results = []
    for ab_name, ab_indices in ABLATIONS.items():
        X_tr_ab = X_train[:, ab_indices]
        X_ev_ab = X_eval[:, ab_indices]
        r = _train_eval(X_tr_ab, y_train, X_ev_ab, y_eval, random_state=1)
        ab_result = {
            "ablation": ab_name, "state_dim": len(ab_indices),
            "feature_indices": ab_indices,
            "train_accuracy": r["train_accuracy"], "eval_accuracy": r["eval_accuracy"],
        }
        ablation_results.append(ab_result)
        print(f"  {ab_name} ({len(ab_indices)}d): train_acc={r['train_accuracy']:.4f}, eval_acc={r['eval_accuracy']:.4f}")

    # === (D) Confusion matrix (full16, seed=1) ===
    print(f"\n[D] Confusion matrix (full16, model_random_state=1) ===")
    clf = RandomForestClassifier(n_estimators=100, random_state=1)
    clf.fit(X_train, y_train)
    cm_info = _confusion_per_class(clf, X_eval, y_eval)
    cm_arr = np.array(cm_info["matrix"])
    print(f"  per-class precision/recall:")
    for c in cm_info["per_class"]:
        if c["support"] > 0:
            print(f"    {c['action']:18s} (support={c['support']:2d}): P={c['precision']:.3f}, R={c['recall']:.3f}, TP={c['tp']}, FN={c['fn']}, FP={c['fp']}")
    _plot_confusion(cm_arr, args.output_dir / "confusion_matrix_eval.png",
                    title=f"RL-1 confusion matrix (eval pairs n={len(y_eval)}, full16, seed=1)")

    # === (E) Failure trace ===
    print(f"\n[E] Failure trace (synthetic rollback samples) ===")
    with open(args.rl1_snapshot, encoding="utf-8") as f:
        rl1 = json.load(f)
    fail_trials = [
        r["trial_id"] for r in rl1["per_sample_synthetic"]
        if r["stop_reason"] == "score_violation_rollback"
    ]
    print(f"  rollback samples (synthetic eval): {fail_trials}")

    # Build pairs_by_trial (all training data — for eval trials, the path is in training pairs).
    # Note: 본 rollback sample 은 eval set 에 속한 trial. eval pairs 중 같은 trial 의 oracle path 를 찾는다.
    pairs_by_trial = {}
    for p in pairs:
        pairs_by_trial.setdefault(p["trial_id"], []).append(p)
    # Sort by step_index per trial.
    for t in pairs_by_trial:
        pairs_by_trial[t].sort(key=lambda p: p["step_index"])

    failure_trace = _failure_trace(clf, pairs_by_trial, fail_trials)
    print(f"\n  trace 발췌 (first 2 fail samples):")
    for trial in fail_trials[:2]:
        print(f"\n  --- {trial} ---")
        for step in failure_trace[trial]:
            match_mark = "✓" if step["match"] else "✗"
            scores_str = f"FF={step['current_scores'].get('FootFloatingEvaluator', 0):.3f}, BL={step['current_scores'].get('BoneLengthEvaluator', 0):.3f}, VJ={step['current_scores'].get('VelocityJitterEvaluator', 0):.3f}"
            print(f"    step {step['step_index']}: {match_mark} true={step['true_action']:18s} pred={step['predicted_action']:18s} | {scores_str}")
            print(f"        top3 proba: {step['top3_proba']}")

    # === Summary ===
    summary = {
        "schema_version": "1.0.0",
        "record_type": "rl1_diagnose",
        "split_seed": args.seed,
        "train_ratio": args.train_ratio,
        "n_train_pairs": len(y_train),
        "n_eval_pairs": len(y_eval),
        "multi_seed": multi_seed_agg,
        "state_ablation": ablation_results,
        "confusion_matrix": cm_info,
        "failure_trace_synthetic_rollback": failure_trace,
    }
    out_json = args.output_dir / "rl1_diagnose_v1.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
