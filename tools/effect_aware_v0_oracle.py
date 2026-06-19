"""AR-040: v0 quality-aware q_proxy + dense effect oracle.

This tool turns AR-051's fixed tool-effect protocol into a v0 action-effect
dataset:

  state:  before-action v0 global state from ``effect_aware_state.py``
  action: (tool, u), no target-aware claim
  label:  q_proxy_v0.1, an internal quality-gain proxy
  oracle: M1 dense effect oracle = best candidate by prompt-level seed-mean q_proxy

The representative motion pool is read-only. Corrected motions are never saved.

CLI:
    python tools/effect_aware_v0_oracle.py \
        --output evals/snapshots/effect_aware_v0_oracle_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from tools.effect_aware_state import build_v0, encode_action, feature_names
from tools.g2_balanced_prompt_selector import GROUPS_ORDER, classify
from tools.harness_metadata import REAL_DISTRIBUTION, common_snapshot_metadata
from tools.representative_refinement_effect import _apply_variant, _measure


GENERATORS = ("motiongpt", "mdm", "momask")
STRENGTH_TO_U = {"small": 0.3, "medium": 0.6, "large": 1.0}
NEUTRAL_SEMANTIC = {
    "text_motion_similarity": 0.0,
    "text_motion_distance": 0.0,
    "semantic_confidence": 0.0,
    "prompt_motion_group_match": 0.0,
}
GROUP_ID = {g: i for i, (g, _) in enumerate(GROUPS_ORDER)}
GROUP_ID["other"] = len(GROUP_ID)

Q_WEIGHTS = {
    "artifact_gain": 0.80,
    "foot_skate_gain": 1.00,
    "accel_gain": 0.35,
    "foot_floating_fire_gain": 0.60,
    "fidelity_loss": -0.80,
    "correction_magnitude": -0.25,
    "new_gate_fire": -2.00,
    "foot_skate_worsening": -0.80,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.load(open(path, encoding="utf-8"))


def _variants() -> list[str]:
    return [
        "noop",
        "foot_lock_small",
        "foot_lock_medium",
        "foot_lock_large",
        "velocity_smoothing_small",
        "velocity_smoothing_medium",
        "velocity_smoothing_large",
        "bone_projection_all_small",
        "bone_projection_all_medium",
        "bone_projection_all_large",
    ]


def _action_for_variant(variant: str) -> dict[str, Any]:
    if variant == "noop":
        return {
            "tool": "STOP",
            "target_side": "none",
            "u": 0.0,
            "is_stop": True,
            "action_space_type": "dense_grid_proxy",
        }
    prefix, strength = variant.rsplit("_", 1)
    if prefix == "foot_lock":
        tool, target = "FootLockTool", "both"
    elif prefix == "velocity_smoothing":
        tool, target = "VelocitySmoothingTool", "full"
    elif prefix == "bone_projection_all":
        tool, target = "BoneProjectionTool", "full"
    else:
        raise ValueError(f"unknown variant: {variant}")
    return {
        "tool": tool,
        "target_side": target,
        "u": STRENGTH_TO_U[strength],
        "is_stop": False,
        "action_space_type": "dense_grid_proxy",
        "strength_label": strength,
    }


def _finite_round(v: float, ndigits: int = 6) -> float:
    v = float(v)
    if not np.isfinite(v):
        return 0.0
    return round(v, ndigits)


def _compact_metrics(m: dict[str, float]) -> dict[str, float]:
    keep = (
        "foot_skate_world",
        "accel",
        "artifact_total",
        "FootFloating_fire",
        "Skate_gate_fire",
        "Float_gate_fire",
        "BoneCV_fire",
        "fidelity_mpjpe",
        "correction_magnitude",
    )
    return {k: _finite_round(m.get(k, 0.0)) for k in keep}


def _effects(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    before_gate = max(before["Skate_gate_fire"], before["Float_gate_fire"], before["BoneCV_fire"])
    after_gate = max(after["Skate_gate_fire"], after["Float_gate_fire"], after["BoneCV_fire"])
    foot_skate_gain = before["foot_skate_world"] - after["foot_skate_world"]
    return {
        "artifact_gain": before["artifact_total"] - after["artifact_total"],
        "foot_skate_gain": foot_skate_gain,
        "accel_gain": before["accel"] - after["accel"],
        "foot_floating_fire_gain": before["FootFloating_fire"] - after["FootFloating_fire"],
        "fidelity_loss": after["fidelity_mpjpe"],
        "correction_magnitude": after["correction_magnitude"],
        "new_gate_fire": max(0.0, after_gate - before_gate),
        "foot_skate_worsening": max(0.0, -foot_skate_gain),
    }


def _robust_scales(effect_rows: list[dict[str, Any]]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for key in ("artifact_gain", "foot_skate_gain", "accel_gain"):
        vals = np.array([abs(r["effect"][key]) for r in effect_rows], dtype=np.float64)
        scales[key] = max(float(np.percentile(vals, 90)), 1e-6)
    for key in ("fidelity_loss", "correction_magnitude", "foot_skate_worsening"):
        vals = np.array([r["effect"][key] for r in effect_rows], dtype=np.float64)
        scales[key] = max(float(np.percentile(vals, 90)), 1e-6)
    return scales


def _score_q_proxy(effect: dict[str, float], scales: dict[str, float]) -> tuple[float, dict[str, float]]:
    comps = {
        "artifact_gain": np.clip(effect["artifact_gain"] / scales["artifact_gain"], -3.0, 3.0),
        "foot_skate_gain": np.clip(effect["foot_skate_gain"] / scales["foot_skate_gain"], -3.0, 3.0),
        "accel_gain": np.clip(effect["accel_gain"] / scales["accel_gain"], -3.0, 3.0),
        "foot_floating_fire_gain": effect["foot_floating_fire_gain"],
        "fidelity_loss": np.clip(effect["fidelity_loss"] / scales["fidelity_loss"], 0.0, 3.0),
        "correction_magnitude": np.clip(effect["correction_magnitude"] / scales["correction_magnitude"], 0.0, 3.0),
        "new_gate_fire": effect["new_gate_fire"],
        "foot_skate_worsening": np.clip(effect["foot_skate_worsening"] / scales["foot_skate_worsening"], 0.0, 3.0),
    }
    weighted = {k: float(Q_WEIGHTS[k] * comps[k]) for k in comps}
    q = float(sum(weighted.values()))
    return q, {k: _finite_round(v) for k, v in weighted.items()}


def _state_vector(state: dict[str, float], names: list[str]) -> list[float]:
    return [_finite_round(state.get(name, 0.0)) for name in names]


def _summarize_oracle(oracle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_gen: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in oracle_rows:
        by_gen[row["generator"]].append(row)
    out: dict[str, Any] = {}
    for gen, rows in by_gen.items():
        counts = Counter(r["best_variant"] for r in rows)
        best_q = np.array([r["best_q_proxy"] for r in rows], dtype=np.float64)
        noop_q = np.array([r["variant_q_proxy"]["noop"] for r in rows], dtype=np.float64)
        out[gen] = {
            "n_prompts": len(rows),
            "best_variant_counts": dict(counts),
            "action_rate_non_stop": _finite_round(float(np.mean([r["best_variant"] != "noop" for r in rows]))),
            "mean_best_q_proxy": _finite_round(best_q.mean()),
            "mean_gain_over_noop": _finite_round((best_q - noop_q).mean()),
            "foot_lock_large_mean_q": _finite_round(float(np.mean([r["variant_q_proxy"]["foot_lock_large"] for r in rows]))),
            "velocity_smoothing_medium_mean_q": _finite_round(float(np.mean([r["variant_q_proxy"]["velocity_smoothing_medium"] for r in rows]))),
            "bone_projection_all_large_mean_q": _finite_round(float(np.mean([r["variant_q_proxy"]["bone_projection_all_large"] for r in rows]))),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path, default=REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608")
    ap.add_argument("--bank", type=Path, default=REPO_ROOT / "evals" / "prompts" / "protocol_rep_test_300_seed20260608.json")
    ap.add_argument("--calibration", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    ap.add_argument("--generators", nargs="+", default=list(GENERATORS))
    ap.add_argument("--limit-prompts", type=int, default=None)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "effect_aware_v0_oracle_v1.json",
                    help="Compact, commit-friendly snapshot.")
    ap.add_argument("--raw-output", type=Path, default=None,
                    help="Optional full transition dataset. Large; keep in evals/raw or regenerate.")
    args = ap.parse_args()

    bank = _load_json(args.bank)
    bank_ids = [r["sample_id"] for r in bank["rows"]]
    if args.limit_prompts is not None:
        bank_ids = bank_ids[: args.limit_prompts]
    keep_ids = set(bank_ids)

    calib = _load_json(args.calibration)
    gate_thr = {n: calib["summary"][n]["p99"] for n in calib["summary"] if calib["summary"][n].get("n", 0) > 0}
    eval_by_name = {ev.name: ev for ev in list(DEFAULT_EVALUATORS) + list(DEFAULT_PHYSICAL_GATE_EVALUATORS)}
    v0_names = feature_names("v0")
    action_names = [
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

    states: dict[str, Any] = {}
    transitions: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    variants = _variants()

    from correction_tools import BoneProjectionTool, FootLockTool, VelocitySmoothingTool

    tools = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
        "BoneProjectionTool": BoneProjectionTool(),
    }

    for gen in args.generators:
        metas = [
            _load_json(p)
            for p in sorted((args.pool_root / gen).glob("*.json"))
            if not p.name.startswith("_")
        ]
        metas = [m for m in metas if m["sample_id"] in keep_ids]
        for idx, meta in enumerate(metas, 1):
            state_id = f"{gen}/{meta['sample_id']}/seed{meta['seed']}"
            traj = np.load(REPO_ROOT / meta["trajectory_npy"]).astype(np.float64)
            local = np.load(REPO_ROOT / meta["local_npy"]).astype(np.float64)
            group = classify(meta.get("prompt", ""))
            gid = GROUP_ID.get(group, GROUP_ID["other"])
            state = build_v0(
                local,
                motion_group_id=gid,
                evaluators_by_name=eval_by_name,
                gate_thresholds=gate_thr,
                semantic_scalars=NEUTRAL_SEMANTIC,
                history=None,
            )
            states[state_id] = {
                "generator": gen,
                "sample_id": meta["sample_id"],
                "seed": meta["seed"],
                "prompt": meta["prompt"],
                "motion_group": group,
                "target_length": meta["target_length"],
                "actual_generated": meta["actual_generated"],
                "length_ratio": _finite_round(meta["actual_generated"] / max(meta["target_length"], 1)),
                "state_v0": _state_vector(state, v0_names),
            }

            before = _measure(traj, local)
            before["fidelity_mpjpe"] = 0.0
            before["correction_magnitude"] = 0.0
            for variant in variants:
                out_traj, out_local, prov = _apply_variant(variant, traj, local, meta, tools)
                after = _measure(out_traj, out_local)
                after["fidelity_mpjpe"] = float(np.mean(np.linalg.norm(out_local - local, axis=-1)))
                after["correction_magnitude"] = float(prov.get("correction_magnitude", 0.0))
                effect = _effects(before, after)
                action = _action_for_variant(variant)
                action_vec = encode_action(action["tool"], action["target_side"], action["u"], action["is_stop"])
                raw_rows.append({
                    "state_id": state_id,
                    "generator": gen,
                    "sample_id": meta["sample_id"],
                    "seed": meta["seed"],
                    "variant": variant,
                    "action": action,
                    "action_v0": [_finite_round(action_vec[n]) for n in action_names],
                    "before": _compact_metrics(before),
                    "after": _compact_metrics(after),
                    "effect": {k: _finite_round(v) for k, v in effect.items()},
                    "selection_mode": prov.get("selection_mode", "fixed_tool_strength"),
                })
            if idx % 100 == 0:
                print(f"[{gen}] {idx}/{len(metas)} states")

    scales = _robust_scales(raw_rows)
    for row in raw_rows:
        q, comps = _score_q_proxy(row["effect"], scales)
        transitions.append({
            **row,
            "q_proxy": _finite_round(q),
            "q_proxy_components": comps,
        })

    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    prompt_text: dict[tuple[str, str], str] = {}
    for row in transitions:
        key = (row["generator"], row["sample_id"], row["variant"])
        grouped[key][row["variant"]].append(row["q_proxy"])
        prompt_text[(row["generator"], row["sample_id"])] = states[row["state_id"]]["prompt"]

    by_prompt_variant: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in transitions:
        by_prompt_variant[(row["generator"], row["sample_id"])][row["variant"]].append(row["q_proxy"])

    oracle_rows = []
    for (gen, sample_id), var_map in sorted(by_prompt_variant.items()):
        var_q = {variant: _finite_round(float(np.mean(vals))) for variant, vals in var_map.items()}
        best_variant = max(var_q, key=var_q.get)
        oracle_rows.append({
            "generator": gen,
            "sample_id": sample_id,
            "prompt": prompt_text[(gen, sample_id)],
            "best_variant": best_variant,
            "best_q_proxy": var_q[best_variant],
            "variant_q_proxy": var_q,
            "oracle_type": "single-step_dense_effect_oracle",
        })

    summary = _summarize_oracle(oracle_rows)
    failure_check = {}
    for gen in args.generators:
        rows = [r for r in transitions if r["generator"] == gen and r["variant"] == "foot_lock_large"]
        failure_check[gen] = {
            "mean_artifact_gain": _finite_round(float(np.mean([r["effect"]["artifact_gain"] for r in rows]))),
            "mean_foot_skate_gain": _finite_round(float(np.mean([r["effect"]["foot_skate_gain"] for r in rows]))),
            "mean_q_proxy": _finite_round(float(np.mean([r["q_proxy"] for r in rows]))),
            "expected": "artifact_gain may be positive, but foot_skate_gain negative should lower q_proxy",
        }

    full_out = {
        "schema_version": "1.0.0",
        "record_type": "effect_aware_v0_oracle",
        "board_id": "AR-040",
        **common_snapshot_metadata(
            split_id="protocol_rep_300_seed20260608",
            oracle_type="single-step_dense_effect_oracle",
            action_grid="dense_grid_proxy",
            stage="AR-040-v0-quality-aware-ranker",
            evidence_tier=[REAL_DISTRIBUTION],
            evaluators=list(DEFAULT_EVALUATORS),
            gate_evaluators=list(DEFAULT_PHYSICAL_GATE_EVALUATORS),
        ),
        "claim_boundary": "Internal q_proxy + M1 dense effect oracle. Learned policy performance not claimed. Target-aware claim not made.",
        "q_proxy": {
            "name": "q_proxy_v0.1",
            "status": "internal proxy assumption",
            "formula": "weighted clipped robust-normalized gains minus fidelity/correction/new-gate/skate-worsening penalties",
            "weights": Q_WEIGHTS,
            "robust_scales_p90_abs_or_p90": {k: _finite_round(v) for k, v in scales.items()},
            "standard_metric_policy": "FID/R-Precision/MM-Dist are final validation metrics, not direct q_proxy labels.",
            "semantic_policy": "semantic block is neutral-imputed with confidence=0; AR-040 does not claim semantic-aware ranking.",
        },
        "state": {
            "version": "v0",
            "source": "tools/effect_aware_state.py",
            "motion_representation": "motion_local canonical SMPL-22 root-relative",
            "feature_names": v0_names,
            "dim": len(v0_names),
            "semantic_imputation": NEUTRAL_SEMANTIC,
        },
        "action": {
            "feature_names": action_names,
            "dim": len(action_names),
            "variants": variants,
            "u_grid": STRENGTH_TO_U,
            "stop_included": True,
        },
        "n_states": len(states),
        "n_transitions": len(transitions),
        "states": states,
        "transitions": transitions,
        "oracle_by_prompt": oracle_rows,
        "summary": summary,
        "failure_check": failure_check,
    }
    compact_out = {
        **{k: v for k, v in full_out.items() if k not in ("states", "transitions")},
        "record_type": "effect_aware_v0_oracle_compact",
        "state_examples": dict(list(states.items())[:5]),
        "transition_examples": transitions[:20],
        "raw_transition_policy": (
            "Full states/transitions are deterministic and regenerable with "
            "`python tools/effect_aware_v0_oracle.py --raw-output evals/raw/effect_aware_v0_oracle_v1_full.json`. "
            "Compact snapshot is the committed evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact_out, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.raw_output is not None:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(json.dumps(full_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== AR-040 v0 q_proxy + M1 dense effect oracle ===")
    print(f"states={len(states)} transitions={len(transitions)} oracle_rows={len(oracle_rows)}")
    for gen, s in summary.items():
        print(
            f"[{gen}] act={s['action_rate_non_stop']:.2f} "
            f"mean_best_q={s['mean_best_q_proxy']:+.3f} "
            f"gain_vs_noop={s['mean_gain_over_noop']:+.3f} "
            f"top={s['best_variant_counts']}"
        )
    print("\nFootLock failure check:")
    for gen, fc in failure_check.items():
        print(
            f"  {gen}: artifact_gain={fc['mean_artifact_gain']:+.4f}, "
            f"skate_gain={fc['mean_foot_skate_gain']:+.4f}, q={fc['mean_q_proxy']:+.3f}"
        )
    print(f"\n[OK] wrote compact snapshot {args.output}")
    if args.raw_output is not None:
        print(f"[OK] wrote raw full transitions {args.raw_output}")


if __name__ == "__main__":
    main()
