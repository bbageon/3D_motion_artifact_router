"""Step F-2 (사용자 directive 2026-05-27): RL-2 16-action imitation 학습 (offline classification).

사용자 directive:
> "바로 Q-learning보다 먼저 imitation이 맞습니다. 모델: HGB / MLP / RF / LR baseline.
>  RL-1 경험상 HGB/MLP 우선. Train/eval split: sample-level, G2 240/60, synthetic 42/18,
>  seed ≥ 3. distribution tag 유/무 두 버전 비교."

본 도구는 rl2_imitation_dataset_v1.json 의 state-action pair 로 16-action classifier 학습 +
offline accuracy 측정 (sample-level split, multi-seed, distribution tag ablation).

NOTE: 본 단계 는 **offline imitation accuracy** (teacher action 재현). closed-loop 평가
(실제 policy 실행) 는 Step F-3 (별도 도구). oracle ceiling vs learned gap 의 분리 보고 의무
([feedback-oracle-vs-learned] memory).

CLI:
    python -m tools.rl2_train_imitation \
        --dataset evals/snapshots/rl2_imitation_dataset_v1.json \
        --seeds 0,1,2 \
        --output evals/snapshots/rl2_imitation_accuracy_v1.json
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

from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, REAL_DISTRIBUTION, common_snapshot_metadata


def _flatten_state(state: dict, include_dist_tag: bool) -> list[float]:
    feats = []
    feats.extend(state["artifact_scores"])       # 3
    feats.extend(state["physical_scores"])        # 5
    feats.extend(state["score_delta"])            # 8
    feats.extend(state["prev_action_onehot"])     # 16
    feats.append(float(state["remaining_budget"]))  # 1
    feats.append(float(state["step_index"]))         # 1
    if include_dist_tag:
        feats.append(float(state["distribution_tag"]))  # 1
    return feats


def _sample_level_split(rows: list[dict], seed: int):
    """Group rows by (distribution, sample_id), split samples G2 240/60, synthetic 42/18."""
    rng = np.random.default_rng(seed)
    by_dist_sample = defaultdict(list)
    for i, r in enumerate(rows):
        by_dist_sample[(r["distribution"], r["sample_id"])].append(i)
    g2_samples = sorted([k for k in by_dist_sample if k[0] == "g2"])
    syn_samples = sorted([k for k in by_dist_sample if k[0] == "synthetic"])
    rng.shuffle(g2_samples)
    rng.shuffle(syn_samples)
    # G2: 80% train (240/300), synthetic: 70% train (42/60).
    g2_train_n = int(round(len(g2_samples) * 0.8))
    syn_train_n = int(round(len(syn_samples) * 0.7))
    train_keys = set(g2_samples[:g2_train_n]) | set(syn_samples[:syn_train_n])
    train_idx, eval_idx = [], []
    for k, idxs in by_dist_sample.items():
        (train_idx if k in train_keys else eval_idx).extend(idxs)
    return train_idx, eval_idx


def _build_models():
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    return {
        "HGB": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                       max_depth=6, random_state=0),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                     early_stopping=True, random_state=0),
        "RF": lambda: RandomForestClassifier(n_estimators=200, max_depth=12,
                                             class_weight="balanced", random_state=0),
        "LR": lambda: LogisticRegression(max_iter=1000, class_weight="balanced",
                                         multi_class="multinomial", random_state=0),
    }


def _evaluate(y_true, y_pred, distributions, stop_idx=0):
    """Action accuracy + STOP precision/recall + per-distribution accuracy."""
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    distributions = np.array(distributions)
    acc = float(np.mean(y_true == y_pred))
    # STOP precision/recall.
    is_stop_true = y_true == stop_idx
    is_stop_pred = y_pred == stop_idx
    tp = int(np.sum(is_stop_true & is_stop_pred))
    fp = int(np.sum(~is_stop_true & is_stop_pred))
    fn = int(np.sum(is_stop_true & ~is_stop_pred))
    stop_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    stop_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Tool-level accuracy (ignore strength): map action to tool group.
    def _tool_group(a):
        if a == 0: return 0  # STOP
        return (a - 1) // 5 + 1  # 1=FootLock, 2=BoneProj, 3=VelSmooth
    tool_true = np.array([_tool_group(a) for a in y_true])
    tool_pred = np.array([_tool_group(a) for a in y_pred])
    tool_acc = float(np.mean(tool_true == tool_pred))
    # Per-distribution.
    per_dist = {}
    for d in ("g2", "synthetic"):
        mask = distributions == d
        if mask.sum() > 0:
            per_dist[d] = {
                "n": int(mask.sum()),
                "action_acc": float(np.mean(y_true[mask] == y_pred[mask])),
                "tool_acc": float(np.mean(tool_true[mask] == tool_pred[mask])),
            }
    return {
        "action_accuracy": acc, "tool_accuracy": tool_acc,
        "stop_precision": stop_prec, "stop_recall": stop_rec,
        "per_distribution": per_dist,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_imitation_dataset_v1.json")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_imitation_accuracy_v1.json")
    args = parser.parse_args()

    data = json.load(open(args.dataset, encoding="utf-8"))
    rows = data["rows"]
    seeds = [int(s) for s in args.seeds.split(",")]
    models = _build_models()
    print(f"[INFO] dataset: {len(rows)} rows, seeds={seeds}, models={list(models.keys())}")

    results = {}  # results[dist_tag_mode][model] = list of per-seed metric dicts
    for dist_tag_mode in (False, True):
        tag_key = "with_dist_tag" if dist_tag_mode else "no_dist_tag"
        results[tag_key] = {}
        X_all = np.array([_flatten_state(r["state"], dist_tag_mode) for r in rows])
        y_all = np.array([r["action_idx"] for r in rows])
        dist_all = [r["distribution"] for r in rows]
        for model_name, model_factory in models.items():
            seed_metrics = []
            for seed in seeds:
                train_idx, eval_idx = _sample_level_split(rows, seed)
                Xtr, ytr = X_all[train_idx], y_all[train_idx]
                Xev, yev = X_all[eval_idx], y_all[eval_idx]
                dev = [dist_all[i] for i in eval_idx]
                try:
                    clf = model_factory()
                    clf.fit(Xtr, ytr)
                    ypred = clf.predict(Xev)
                except Exception as e:
                    print(f"[WARN] {model_name} seed={seed} ({tag_key}): {e}")
                    continue
                m = _evaluate(yev, ypred, dev)
                seed_metrics.append(m)
            if not seed_metrics:
                continue
            # Aggregate over seeds.
            def _agg(key):
                vals = [m[key] for m in seed_metrics]
                return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            g2_acc = [m["per_distribution"].get("g2", {}).get("action_acc", 0) for m in seed_metrics]
            syn_acc = [m["per_distribution"].get("synthetic", {}).get("action_acc", 0) for m in seed_metrics]
            results[tag_key][model_name] = {
                "n_seeds": len(seed_metrics),
                "action_accuracy": _agg("action_accuracy"),
                "tool_accuracy": _agg("tool_accuracy"),
                "stop_precision": _agg("stop_precision"),
                "stop_recall": _agg("stop_recall"),
                "g2_action_acc": {"mean": float(np.mean(g2_acc)), "std": float(np.std(g2_acc))},
                "synthetic_action_acc": {"mean": float(np.mean(syn_acc)), "std": float(np.std(syn_acc))},
            }

    evaluators = list(DEFAULT_EVALUATORS)
    gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)
    out = {
        "schema_version": "1.0.0", "record_type": "rl2_imitation_accuracy",
        "task_id": "rl2_imitation_accuracy_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "rl2_imitation_accuracy_v1",
            oracle_type="sequence",
            action_grid="5-level",
            stage="RL-2-imitation",
            evidence_tier=[REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
            evaluators=evaluators,
            gate_evaluators=gate_evaluators,
        ),
        "dataset_source": str(args.dataset),
        "seeds": seeds, "n_feature_dims": {"no_dist_tag": 34, "with_dist_tag": 35},
        "note": "OFFLINE imitation accuracy (teacher action 재현). closed-loop = Step F-3 별도.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RL-2 Imitation Accuracy (Step F-2, offline) ===")
    for tag_key in results:
        print(f"\n[{tag_key}]")
        print(f"  {'model':<6} {'action_acc':<16} {'tool_acc':<14} {'STOP_rec':<12} {'g2_acc':<12} {'syn_acc'}")
        for model_name, m in results[tag_key].items():
            print(f"  {model_name:<6} "
                  f"{m['action_accuracy']['mean']:.3f}±{m['action_accuracy']['std']:.3f}    "
                  f"{m['tool_accuracy']['mean']:.3f}±{m['tool_accuracy']['std']:.3f}   "
                  f"{m['stop_recall']['mean']:.3f}±{m['stop_recall']['std']:.3f}  "
                  f"{m['g2_action_acc']['mean']:.3f}      "
                  f"{m['synthetic_action_acc']['mean']:.3f}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
