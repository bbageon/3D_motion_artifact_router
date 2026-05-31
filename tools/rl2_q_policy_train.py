"""Step 4 (사용자 directive 2026-05-31): RL-2 Q policy training (M0/M1/M2/M3).

사용자 directive 박제 (M0~M3 design):
  - M0: Stage A only (legacy baseline — synthetic 중심)
  - M1: G2 real transitions only (train_g2_real + calib_g2_real, real-distribution 단독)
  - M2: G2 real + synthetic auxiliary (M1 + synthetic_aux_train)
  - M3: M2 + post-hoc calibration (isotonic on calib_g2_real)
  → 논문 중심 = M1 / M2. synthetic-only (M0) 는 appendix.

각 모델 = two-head:
  Q_utility(s, tool, u) — HGB regressor, target safe_utility.
  P_safe(s, tool, u)    — HGB classifier, target is_safe.

평가 (sanity): calib + train holdout 의 R²/AUC, 모델별 비교.

CLI:
    python -m tools.rl2_q_policy_train \
        --g2-real evals/snapshots/rl2_transition_g2_real_stage2_v1.json \
        --stage-a evals/snapshots/rl2_transition_dataset_stageA_v1.json \
        --output-models evals/models/rl2_q_policies_stage2_v1.joblib \
        --output evals/snapshots/rl2_q_policy_train_stage2_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, REAL_DISTRIBUTION, common_snapshot_metadata
from tools.rl2_build_training_data import ARTIFACT_EVALUATORS, PHYSICAL_EVALUATORS
from tools.rl2_transition_build import TOOLS_ORDER

TOOL_IDX = {t: i for i, t in enumerate(TOOLS_ORDER)}


def _cand_features_from_state(state_dict, tool, u):
    """state dict (artifact_scores, physical_scores) + tool one-hot + u → 12-dim feature."""
    a = [state_dict["artifact_scores"][n] for n in ARTIFACT_EVALUATORS]
    p = [state_dict["physical_scores"][n] for n in PHYSICAL_EVALUATORS]
    oh = [0.0] * len(TOOLS_ORDER); oh[TOOL_IDX[tool]] = 1.0
    return a + p + oh + [float(u)]


def _stage_a_rows_to_xy(stage_a_data):
    """Stage A rows (per-(state,tool,u) transitions) → (X, y_util, y_safe, sample_keys).

    Stage A schema (rl2_transition_build.py): rows = flat transitions with inline state/after_state.
    """
    X, yu, ys, keys = [], [], [], []
    for r in stage_a_data["rows"]:
        state = r["state"]
        X.append(_cand_features_from_state(state, r["tool"], r["u"]))
        yu.append(r["safe_utility"])
        ys.append(1 if r["gate_result"] == "pass" else 0)
        keys.append((r["distribution"], r["sample_id"]))
    return np.array(X), np.array(yu), np.array(ys), keys


def _g2_real_rows_to_xy(g2_data, states_map, include_splits=None, include_distributions=None):
    """G2 real transitions → (X, y_util, y_safe, meta) filter by splits/distributions."""
    X, yu, ys, meta = [], [], [], []
    for t in g2_data["transitions"]:
        if include_splits and t["split"] not in include_splits: continue
        if include_distributions and t["distribution"] not in include_distributions: continue
        state_dict = states_map[t["state_id"]]
        X.append(_cand_features_from_state(state_dict, t["tool"], t["u"]))
        yu.append(t["safe_utility"])
        ys.append(1 if t["gate_result"] == "pass" else 0)
        meta.append({"state_id": t["state_id"], "split": t["split"],
                     "distribution": t["distribution"], "tool": t["tool"], "u": t["u"]})
    return np.array(X), np.array(yu), np.array(ys), meta


def _build_heads(seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    util = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_depth=6, random_state=seed)
    safe = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_depth=6, random_state=seed)
    return util, safe


def _train_model(X, yu, ys, seed=0):
    util, safe = _build_heads(seed)
    util.fit(X, yu)
    if len(np.unique(ys)) > 1:
        safe.fit(X, ys)
    else:
        # Degenerate case: all same class — use dummy.
        from sklearn.dummy import DummyClassifier
        safe = DummyClassifier(strategy="constant", constant=int(ys[0])); safe.fit(X, ys)
    return util, safe


def _eval_heads(util, safe, X, yu, ys, name=""):
    from sklearn.metrics import r2_score, mean_absolute_error
    out = {"n": len(X)}
    if len(X) > 0:
        u_pred = util.predict(X)
        out["util_r2"] = float(r2_score(yu, u_pred))
        out["util_mae"] = float(mean_absolute_error(yu, u_pred))
        if hasattr(safe, "predict_proba") and len(np.unique(ys)) > 1:
            from sklearn.metrics import roc_auc_score, average_precision_score
            p = safe.predict_proba(X)[:, 1]
            out["safe_roc_auc"] = float(roc_auc_score(ys, p))
            out["unsafe_pr_auc"] = float(average_precision_score(1 - ys, 1 - p))
        out["safe_acc_at_0p5"] = float(np.mean((safe.predict(X) == ys)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-real", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_g2_real_stage2_v1.json")
    parser.add_argument("--stage-a", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_dataset_stageA_v1.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-models", type=Path,
                        default=REPO_ROOT / "evals" / "models" / "rl2_q_policies_stage2_v1.joblib")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_q_policy_train_stage2_v1.json")
    parser.add_argument("--split-id", type=str, default=None)
    args = parser.parse_args()

    print("[INFO] loading transitions ...")
    g2_data = json.load(open(args.g2_real, encoding="utf-8"))
    states_map = g2_data["states"]

    # === M0: Stage A only (legacy baseline) ===
    print("\n[INFO] M0: Stage A only (legacy synthetic-centric)")
    sa_data = json.load(open(args.stage_a, encoding="utf-8"))
    X_sa, yu_sa, ys_sa, keys_sa = _stage_a_rows_to_xy(sa_data)
    util0, safe0 = _train_model(X_sa, yu_sa, ys_sa, args.seed)
    print(f"   trained on n={len(X_sa)}")

    # === M1: G2 real only (train_g2_real + calib_g2_real, distribution=g2_natural) ===
    print("\n[INFO] M1: G2 real only (train + calib)")
    X1, yu1, ys1, _ = _g2_real_rows_to_xy(g2_data, states_map,
                                           include_splits={"train_g2_real", "calib_g2_real"},
                                           include_distributions={"g2_natural"})
    util1, safe1 = _train_model(X1, yu1, ys1, args.seed)
    print(f"   trained on n={len(X1)}")

    # === M2: G2 real + synthetic auxiliary ===
    print("\n[INFO] M2: G2 real + synthetic auxiliary")
    X2, yu2, ys2, _ = _g2_real_rows_to_xy(g2_data, states_map,
                                           include_splits={"train_g2_real", "calib_g2_real", "synthetic_aux_train"})
    util2, safe2 = _train_model(X2, yu2, ys2, args.seed)
    print(f"   trained on n={len(X2)}")

    # === M3: M2 + isotonic post-hoc calibration on calib_g2_real ===
    print("\n[INFO] M3: M2 + isotonic calibration (calib_g2_real)")
    Xc, yuc, ysc, _ = _g2_real_rows_to_xy(g2_data, states_map,
                                           include_splits={"calib_g2_real"})
    # Wrap safe head with isotonic calibration (sklearn CalibratedClassifierCV-like via IsotonicRegression on probs).
    from sklearn.isotonic import IsotonicRegression
    if len(Xc) > 10 and len(np.unique(ysc)) > 1 and hasattr(safe2, "predict_proba"):
        raw_p = safe2.predict_proba(Xc)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_p, ysc.astype(float))
    else:
        iso = None
    # M3 = (util2, safe2, iso) — at inference, P_safe(x) = iso(safe2.predict_proba(x)).
    util3, safe3 = util2, safe2  # 같은 모델 reuse.

    # === Eval each model on calib + holdouts (in-sample sanity + OOD) ===
    print("\n[INFO] sanity eval per model on calib + each holdout")
    holdouts = {
        "calib_g2_real": {"include_splits": {"calib_g2_real"}, "include_distributions": {"g2_natural"}},
        "g2_stress_holdout": {"include_splits": {"g2_stress_holdout"}, "include_distributions": {"g2_natural"}},
        "g2_natural_holdout": {"include_splits": {"g2_natural_holdout"}, "include_distributions": {"g2_natural"}},
        "clean_noharm_holdout": {"include_splits": {"clean_noharm_holdout"}, "include_distributions": {"clean"}},
        "synthetic_diag_holdout": {"include_splits": {"synthetic_diag_holdout"}, "include_distributions": {"synthetic_severe"}},
    }
    models = {"M0": (util0, safe0, None), "M1": (util1, safe1, None),
              "M2": (util2, safe2, None), "M3": (util3, safe3, iso)}
    model_metrics = {}
    for mname, (util, safe, iso_or_none) in models.items():
        model_metrics[mname] = {}
        for ho_name, ho_filter in holdouts.items():
            Xh, yuh, ysh, _ = _g2_real_rows_to_xy(g2_data, states_map, **ho_filter)
            if len(Xh) == 0:
                continue
            m = _eval_heads(util, safe, Xh, yuh, ysh, name=ho_name)
            # For M3, apply isotonic to safe pred for safe_acc_at_0p5 recomputation.
            if iso_or_none is not None and hasattr(safe, "predict_proba"):
                p_raw = safe.predict_proba(Xh)[:, 1]
                p_cal = iso_or_none.predict(p_raw)
                # Recompute calibrated metrics.
                from sklearn.metrics import roc_auc_score, average_precision_score
                if len(np.unique(ysh)) > 1:
                    m["safe_roc_auc_calibrated"] = float(roc_auc_score(ysh, p_cal))
                    m["unsafe_pr_auc_calibrated"] = float(average_precision_score(1 - ysh, 1 - p_cal))
                m["safe_acc_at_0p5_calibrated"] = float(np.mean(((p_cal >= 0.5).astype(int) == ysh)))
            model_metrics[mname][ho_name] = m

    # Save models via joblib.
    args.output_models.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump({"M0": (util0, safe0, None),
                     "M1": (util1, safe1, None),
                     "M2": (util2, safe2, None),
                     "M3": (util3, safe3, iso)}, str(args.output_models))
        models_saved = True
    except Exception as e:
        print(f"[WARN] joblib save failed: {e}")
        models_saved = False

    out = {
        "schema_version": "1.0.0", "record_type": "rl2_q_policy_train",
        "task_id": "rl2_q_policy_train_stage2_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "rl2_q_policy_train_stage2_v1",
            oracle_type="action_effect_transition",
            action_grid="continuous-dense-u11", stage="Stage-2G-step4-Q-train",
            evidence_tier=[REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
            evaluators=list(DEFAULT_EVALUATORS), gate_evaluators=list(DEFAULT_PHYSICAL_GATE_EVALUATORS),
        ),
        "directive": "M0 (Stage A baseline) / M1 (G2 real only) / M2 (G2+synthetic) / M3 (M2+isotonic calibrated).",
        "training_sizes": {"M0": int(len(X_sa)), "M1": int(len(X1)), "M2": int(len(X2)),
                           "M3_calib_n": int(len(Xc))},
        "models_saved_path": str(args.output_models.resolve()) if models_saved else None,
        "model_eval_metrics": model_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Step 4 — Q policy training summary ===")
    print(f"  trained: M0 n={len(X_sa)} / M1 n={len(X1)} / M2 n={len(X2)} (M3 = M2 + isotonic calib n={len(Xc)})")
    print(f"\n  per-model holdout metrics:")
    print(f"  {'holdout':<24} {'model':<6} {'n':<6} {'util_R2':<10} {'unsafe_PR_AUC':<14} {'safe_acc'}")
    for ho_name in holdouts:
        for mname in models:
            m = model_metrics.get(mname, {}).get(ho_name)
            if m is None: continue
            r2 = m.get("util_r2", float("nan"))
            pra = m.get("unsafe_pr_auc", float("nan"))
            acc = m.get("safe_acc_at_0p5", float("nan"))
            if mname == "M3":
                pra_c = m.get("unsafe_pr_auc_calibrated", float("nan"))
                acc_c = m.get("safe_acc_at_0p5_calibrated", float("nan"))
                pra_s = f"{pra:.3f}→{pra_c:.3f}" if not np.isnan(pra) else "—"
                acc_s = f"{acc:.3f}→{acc_c:.3f}" if not np.isnan(acc) else "—"
            else:
                pra_s = f"{pra:.3f}" if not np.isnan(pra) else "—"
                acc_s = f"{acc:.3f}" if not np.isnan(acc) else "—"
            print(f"  {ho_name:<24} {mname:<6} {m['n']:<6} {r2:<10.3f} {pra_s:<14} {acc_s}")
    print(f"\n[OK] models -> {args.output_models}")
    print(f"[OK] metrics -> {args.output}")


if __name__ == "__main__":
    main()
