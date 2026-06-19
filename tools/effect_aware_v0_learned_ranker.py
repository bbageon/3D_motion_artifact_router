"""AR-052: train/evaluate v0 learned ranker against AR-040 M1 oracle.

The model predicts ``q_proxy_v0.1`` for candidate (state, action) pairs, then
ranks the same dense candidate set used by AR-040. Evaluation is prompt-level:
all seeds for a prompt stay in the same split to avoid leakage.

CLI:
    python tools/effect_aware_v0_learned_ranker.py \
        --output evals/snapshots/effect_aware_v0_learned_ranker_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from tools.effect_aware_state import build_v0, encode_action, feature_names
from tools.effect_aware_v0_oracle import (
    GENERATORS,
    NEUTRAL_SEMANTIC,
    _action_for_variant,
    _effects,
    _finite_round,
    _score_q_proxy,
    _variants,
    GROUP_ID,
)
from tools.g2_balanced_prompt_selector import classify
from tools.harness_metadata import REAL_DISTRIBUTION, common_snapshot_metadata
from tools.representative_refinement_effect import _apply_variant, _measure


ACTION_NAMES = [
    "action_tool_encoding",
    "action_tool_family",
    "action_tool_cost",
    "action_target_type_encoding",
    "action_target_side",
    "action_target_chain_id",
    "action_u",
    "action_u_sq",
    "action_is_continuous_u",
    "action_is_stop_action",
]
THRESHOLDS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)


def _load_json(path: Path) -> dict[str, Any]:
    return json.load(open(path, encoding="utf-8"))


def _stable_float_id(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) / float(0xFFFFFFFF)


def _make_split(sample_ids: list[str], seed: int, train_frac: float) -> dict[str, Any]:
    ids = np.array(sorted(sample_ids), dtype=object)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    n_train = int(round(len(ids) * train_frac))
    train = sorted(ids[order[:n_train]].tolist())
    test = sorted(ids[order[n_train:]].tolist())
    return {
        "seed": seed,
        "train_frac": train_frac,
        "unit": "prompt/sample_id",
        "train_ids": train,
        "test_ids": test,
        "n_train": len(train),
        "n_test": len(test),
        "overlap": sorted(set(train) & set(test)),
    }


def _state_vector(state: dict[str, float], names: list[str]) -> list[float]:
    return [_finite_round(state.get(name, 0.0)) for name in names]


def _build_rows(args: argparse.Namespace, scales: dict[str, float]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    from correction_tools import BoneProjectionTool, FootLockTool, VelocitySmoothingTool

    v0_names = feature_names("v0")
    feature_names_all = v0_names + ACTION_NAMES
    calib = _load_json(args.calibration)
    gate_thr = {n: calib["summary"][n]["p99"] for n in calib["summary"] if calib["summary"][n].get("n", 0) > 0}
    eval_by_name = {ev.name: ev for ev in list(DEFAULT_EVALUATORS) + list(DEFAULT_PHYSICAL_GATE_EVALUATORS)}
    tools = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
        "BoneProjectionTool": BoneProjectionTool(),
    }
    bank = _load_json(args.bank)
    bank_ids = [r["sample_id"] for r in bank["rows"]]
    if args.limit_prompts is not None:
        bank_ids = bank_ids[: args.limit_prompts]
    keep_ids = set(bank_ids)

    rows: list[dict[str, Any]] = []
    variants = _variants()
    for gen in args.generators:
        metas = [
            _load_json(p)
            for p in sorted((args.pool_root / gen).glob("*.json"))
            if not p.name.startswith("_")
        ]
        metas = [m for m in metas if m["sample_id"] in keep_ids]
        for idx, meta in enumerate(metas, 1):
            traj = np.load(REPO_ROOT / meta["trajectory_npy"]).astype(np.float64)
            local = np.load(REPO_ROOT / meta["local_npy"]).astype(np.float64)
            group = classify(meta.get("prompt", ""))
            state = build_v0(
                local,
                motion_group_id=GROUP_ID.get(group, GROUP_ID["other"]),
                evaluators_by_name=eval_by_name,
                gate_thresholds=gate_thr,
                semantic_scalars=NEUTRAL_SEMANTIC,
                history=None,
            )
            state_vec = _state_vector(state, v0_names)
            before = _measure(traj, local)
            before["fidelity_mpjpe"] = 0.0
            before["correction_magnitude"] = 0.0
            for variant in variants:
                out_traj, out_local, prov = _apply_variant(variant, traj, local, meta, tools)
                after = _measure(out_traj, out_local)
                after["fidelity_mpjpe"] = float(np.mean(np.linalg.norm(out_local - local, axis=-1)))
                after["correction_magnitude"] = float(prov.get("correction_magnitude", 0.0))
                effect = _effects(before, after)
                q, _ = _score_q_proxy(effect, scales)
                action = _action_for_variant(variant)
                action_vec = encode_action(action["tool"], action["target_side"], action["u"], action["is_stop"])
                x = state_vec + [_finite_round(action_vec[n]) for n in ACTION_NAMES]
                rows.append({
                    "generator": gen,
                    "sample_id": meta["sample_id"],
                    "seed": int(meta["seed"]),
                    "variant": variant,
                    "x": x,
                    "q_proxy": float(q),
                })
            if idx % 150 == 0:
                print(f"[build:{gen}] {idx}/{len(metas)} seed states")
    return rows, v0_names, feature_names_all


def _fit_model(name: str, seed: int):
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=180,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.055,
            l2_regularization=0.01,
            random_state=seed,
        )
    raise ValueError(name)


def _rows_to_xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([r["x"] for r in rows], dtype=np.float32)
    y = np.array([r["q_proxy"] for r in rows], dtype=np.float32)
    return x, y


def _mean_by_variant(rows: list[dict[str, Any]], pred: np.ndarray | None = None) -> dict[str, float]:
    by_variant: dict[str, list[float]] = defaultdict(list)
    for i, row in enumerate(rows):
        val = float(pred[i]) if pred is not None else float(row["q_proxy"])
        by_variant[row["variant"]].append(val)
    return {k: float(np.mean(v)) for k, v in by_variant.items()}


def _deterministic_random_variant(generator: str, sample_id: str, variants: list[str]) -> str:
    idx = int(_stable_float_id(f"{generator}/{sample_id}") * len(variants))
    return variants[min(idx, len(variants) - 1)]


def _ranking_eval(rows: list[dict[str, Any]], pred: np.ndarray) -> dict[str, Any]:
    variants = _variants()
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, p in zip(rows, pred):
        grouped[(row["generator"], row["sample_id"])].append((row, float(p)))

    records = []
    threshold_stats = {str(t): [] for t in THRESHOLDS}
    for (gen, sid), pairs in grouped.items():
        cand_rows = [r for r, _ in pairs]
        cand_pred = np.array([p for _, p in pairs], dtype=np.float64)
        true_by = _mean_by_variant(cand_rows)
        pred_by = _mean_by_variant(cand_rows, cand_pred)
        oracle = max(true_by, key=true_by.get)
        learned = max(pred_by, key=pred_by.get)
        pred_top3 = [v for v, _ in sorted(pred_by.items(), key=lambda kv: -kv[1])[:3]]
        random_v = _deterministic_random_variant(gen, sid, variants)
        oracle_q = true_by[oracle]
        selected_q = true_by[learned]
        noop_q = true_by.get("noop", 0.0)
        rec = {
            "generator": gen,
            "sample_id": sid,
            "oracle": oracle,
            "learned": learned,
            "oracle_q": oracle_q,
            "learned_q": selected_q,
            "noop_q": noop_q,
            "regret": oracle_q - selected_q,
            "random_regret": oracle_q - true_by[random_v],
            "top1_match": float(learned == oracle),
            "top3_contains_oracle": float(oracle in pred_top3),
            "false_improve_zero": float(learned != "noop" and selected_q <= 0.0),
            "false_improve_noop": float(learned != "noop" and selected_q <= noop_q),
            "act": float(learned != "noop"),
        }
        records.append(rec)
        pred_noop = pred_by.get("noop", 0.0)
        pred_best = pred_by[learned]
        for t in THRESHOLDS:
            chosen = learned if (pred_best - pred_noop) >= t else "noop"
            q = true_by[chosen]
            threshold_stats[str(t)].append({
                "act": float(chosen != "noop"),
                "regret": oracle_q - q,
                "false_improve_zero": float(chosen != "noop" and q <= 0.0),
                "false_improve_noop": float(chosen != "noop" and q <= noop_q),
            })

    def mean(key: str) -> float:
        return _finite_round(float(np.mean([r[key] for r in records]))) if records else 0.0

    out = {
        "n_prompts": len(records),
        "oracle_mean_q": _finite_round(float(np.mean([r["oracle_q"] for r in records]))),
        "learned_selected_mean_q": _finite_round(float(np.mean([r["learned_q"] for r in records]))),
        "mean_regret": mean("regret"),
        "median_regret": _finite_round(float(np.median([r["regret"] for r in records]))) if records else 0.0,
        "random_mean_regret": mean("random_regret"),
        "regret_reduction_vs_random": _finite_round(mean("random_regret") - mean("regret")),
        "top1_match": mean("top1_match"),
        "top3_contains_oracle": mean("top3_contains_oracle"),
        "false_improve_zero_rate": mean("false_improve_zero"),
        "false_improve_noop_rate": mean("false_improve_noop"),
        "action_rate_non_stop": mean("act"),
        "threshold_sweep": {},
    }
    for t, vals in threshold_stats.items():
        out["threshold_sweep"][t] = {
            "act_rate": _finite_round(float(np.mean([v["act"] for v in vals]))) if vals else 0.0,
            "mean_regret": _finite_round(float(np.mean([v["regret"] for v in vals]))) if vals else 0.0,
            "false_improve_zero_rate": _finite_round(float(np.mean([v["false_improve_zero"] for v in vals]))) if vals else 0.0,
            "false_improve_noop_rate": _finite_round(float(np.mean([v["false_improve_noop"] for v in vals]))) if vals else 0.0,
        }
    return out


def _evaluate_scope(
    *,
    rows: list[dict[str, Any]],
    train_ids: set[str],
    test_ids: set[str],
    scope_name: str,
    model_name: str,
    seed: int,
    generator: str | None = None,
    feature_names_all: list[str],
) -> dict[str, Any]:
    scoped = [r for r in rows if generator is None or r["generator"] == generator]
    train_rows = [r for r in scoped if r["sample_id"] in train_ids]
    test_rows = [r for r in scoped if r["sample_id"] in test_ids]
    x_train, y_train = _rows_to_xy(train_rows)
    x_test, y_test = _rows_to_xy(test_rows)
    model = _fit_model(model_name, seed)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    out: dict[str, Any] = {
        "scope": scope_name,
        "model": model_name,
        "generator": generator or "pooled",
        "n_train_rows": len(train_rows),
        "n_test_rows": len(test_rows),
        "q_proxy_r2": _finite_round(float(r2_score(y_test, pred))) if len(set(y_test.tolist())) > 1 else 0.0,
        "q_proxy_mae": _finite_round(float(mean_absolute_error(y_test, pred))),
        "ranking": _ranking_eval(test_rows, pred),
    }
    if hasattr(model, "feature_importances_"):
        imps = getattr(model, "feature_importances_")
        top = sorted(zip(feature_names_all, imps), key=lambda kv: -kv[1])[:12]
        out["top_feature_importances"] = [
            {"feature": name, "importance": _finite_round(float(val))}
            for name, val in top
        ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path, default=REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608")
    ap.add_argument("--bank", type=Path, default=REPO_ROOT / "evals" / "prompts" / "protocol_rep_test_300_seed20260608.json")
    ap.add_argument("--calibration", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    ap.add_argument("--oracle-snapshot", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "effect_aware_v0_oracle_v1.json")
    ap.add_argument("--generators", nargs="+", default=list(GENERATORS))
    ap.add_argument("--models", nargs="+", default=["rf", "hgb"])
    ap.add_argument("--limit-prompts", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=20260619)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "effect_aware_v0_learned_ranker_v1.json")
    args = ap.parse_args()

    oracle = _load_json(args.oracle_snapshot)
    scales = oracle["q_proxy"]["robust_scales_p90_abs_or_p90"]
    rows, v0_names, feature_names_all = _build_rows(args, scales)
    sample_ids = sorted({r["sample_id"] for r in rows})
    split = _make_split(sample_ids, args.seed, args.train_frac)
    train_ids = set(split["train_ids"])
    test_ids = set(split["test_ids"])

    evaluations = []
    for model_name in args.models:
        evaluations.append(_evaluate_scope(
            rows=rows,
            train_ids=train_ids,
            test_ids=test_ids,
            scope_name="pooled_prompt_split",
            model_name=model_name,
            seed=args.seed,
            generator=None,
            feature_names_all=feature_names_all,
        ))
        for gen in args.generators:
            evaluations.append(_evaluate_scope(
                rows=rows,
                train_ids=train_ids,
                test_ids=test_ids,
                scope_name=f"{gen}_prompt_split",
                model_name=model_name,
                seed=args.seed,
                generator=gen,
                feature_names_all=feature_names_all,
            ))

    out = {
        "schema_version": "1.0.0",
        "record_type": "effect_aware_v0_learned_ranker_eval",
        "board_id": "AR-052",
        **common_snapshot_metadata(
            split_id="protocol_rep_300_seed20260608_prompt_split_20260619",
            oracle_type="learned_ranker_vs_single_step_dense_effect_oracle",
            action_grid="dense_grid_proxy",
            stage="AR-052-v0-learned-ranker-train-eval",
            evidence_tier=[REAL_DISTRIBUTION],
            evaluators=list(DEFAULT_EVALUATORS),
            gate_evaluators=list(DEFAULT_PHYSICAL_GATE_EVALUATORS),
        ),
        "claim_boundary": "Offline q_proxy ranking only. No target-aware, final Category-A, or closed-loop quality claim.",
        "q_proxy_source": str(args.oracle_snapshot.relative_to(REPO_ROOT)),
        "state": {"version": "v0", "dim": len(v0_names), "feature_names": v0_names},
        "action": {"dim": len(ACTION_NAMES), "feature_names": ACTION_NAMES, "variants": _variants()},
        "dataset": {
            "n_rows": len(rows),
            "n_prompts": len(sample_ids),
            "n_generators": len(args.generators),
            "stat_unit": "prompt; all seeds for a prompt stay in the same split",
        },
        "split": split,
        "leakage_audit": {
            "train_test_prompt_overlap": split["overlap"],
            "passed": len(split["overlap"]) == 0,
        },
        "evaluations": evaluations,
        "success_condition_check": {
            "prompt_level_split_recorded": True,
            "random_regret_baseline_reported": True,
            "false_improve_reported": True,
            "limitations_reported": True,
        },
        "limitations": [
            "q_proxy_v0.1 is an internal proxy assumption.",
            "v0 state has no target/local/relation block.",
            "semantic block is neutral-imputed; semantic-aware ranking is not claimed.",
            "Final Category-A/FID validation is a later task.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== AR-052 v0 learned ranker eval ===")
    print(f"rows={len(rows)} prompts={len(sample_ids)} split={split['n_train']}/{split['n_test']} overlap={len(split['overlap'])}")
    for ev in evaluations:
        rank = ev["ranking"]
        print(
            f"[{ev['model']}:{ev['generator']}] "
            f"R2={ev['q_proxy_r2']:+.3f} MAE={ev['q_proxy_mae']:.3f} "
            f"top1={rank['top1_match']:.2f} top3={rank['top3_contains_oracle']:.2f} "
            f"regret={rank['mean_regret']:.3f} random={rank['random_mean_regret']:.3f} "
            f"false={rank['false_improve_zero_rate']:.2f} act={rank['action_rate_non_stop']:.2f}"
        )
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
