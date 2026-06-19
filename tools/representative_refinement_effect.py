"""AR-051: cross-generator refinement effect validation (physical/proxy pass).

Applies existing refinement tools to the representative-300 pool without changing
the original pool, then reports paired before/after deltas by generator.

This is the first AR-051 pass:
  - Category B/C physical/proxy deltas, fidelity, side effects.
  - Category A standard metrics are a follow-up pass on selected outputs.

Stat unit: prompt. Seed-level values are averaged into prompt-level values before
aggregate bootstrap CIs. For MotionGPT, an additional sensitivity view excludes
seed samples whose native generated length ratio is below 0.6.

CLI:
    python tools/representative_refinement_effect.py \
        --pool-root external_assets/protocol_rep_pool_seed20260608 \
        --output evals/snapshots/representative_refinement_effect_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import BoneProjectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from skeleton_normalizer.canonical_smpl_22 import CHAIN_LABELS
from tools.coords_protocol import derive_local, estimate_ground
from tools.physical_metric_g2_stress import accel_stats
from tools.representative_pool_measure import ARTIFACT_THRESH, PHYS_P99, foot_skate_world


STRENGTHS = ("small", "medium", "large")
GENERATORS = ("motiongpt", "mdm", "momask")
METRICS = (
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


def _boot_ci(vals: np.ndarray, rng: np.random.Generator, n: int = 1000) -> list[float]:
    if len(vals) == 0:
        return [0.0, 0.0]
    bs = [float(np.mean(vals[rng.integers(0, len(vals), len(vals))])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 6), round(float(np.percentile(bs, 97.5)), 6)]


def _load_json(path: Path) -> dict[str, Any]:
    return json.load(open(path, encoding="utf-8"))


def _measure(traj: np.ndarray, local: np.ndarray) -> dict[str, float]:
    art_evs = {ev.name: ev for ev in DEFAULT_EVALUATORS if ev.name in ARTIFACT_THRESH}
    phy_evs = {ev.name: ev for ev in DEFAULT_PHYSICAL_GATE_EVALUATORS if ev.name in PHYS_P99}
    art = {k: max((r.score for r in art_evs[k].evaluate(local)), default=0.0) for k in ARTIFACT_THRESH}
    phy = {k: max((r.score for r in phy_evs[k].evaluate(local)), default=0.0) for k in PHYS_P99}
    return {
        "foot_skate_world": foot_skate_world(traj),
        "accel": accel_stats(local)[0],
        "artifact_total": float(np.mean(list(art.values()))),
        "FootFloating_fire": float(art["FootFloatingEvaluator"] >= ARTIFACT_THRESH["FootFloatingEvaluator"]),
        "Skate_gate_fire": float(phy["SkateEvaluator"] > PHYS_P99["SkateEvaluator"]),
        "Float_gate_fire": float(phy["FloatEvaluator"] > PHYS_P99["FloatEvaluator"]),
        "BoneCV_fire": float(phy["BoneLengthCVEvaluator"] > PHYS_P99["BoneLengthCVEvaluator"]),
    }


def _apply_variant(
    variant: str,
    traj: np.ndarray,
    local: np.ndarray,
    meta: dict[str, Any],
    tools: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return corrected (trajectory, local, provenance)."""
    if variant == "noop":
        return traj.copy(), local.copy(), {
            "selection_mode": "noop",
            "tool": "STOP",
            "applied_representation": "none",
            "correction_magnitude": 0.0,
        }

    tool_name, strength = variant.rsplit("_", 1)
    T = traj.shape[0]
    if tool_name == "foot_lock":
        # Foot skate/contact is world-space; apply to trajectory and derive local.
        out_traj, report = tools["FootLockTool"].apply(
            traj,
            target_part="both_feet",
            target_joints=[],
            frame_range=(0, T - 1),
            strength=strength,
            metadata={"ground_y": float(meta.get("ground_y", estimate_ground(traj)))},
        )
        return out_traj, derive_local(out_traj), {
            "selection_mode": "fixed_tool_strength",
            "tool": "FootLockTool",
            "target_part": "both_feet",
            "strength": strength,
            "applied_representation": "motion_trajectory",
            "correction_magnitude": report.correction_magnitude,
            "report_metadata": report.metadata,
        }

    if tool_name == "velocity_smoothing":
        out_local, report = tools["VelocitySmoothingTool"].apply(
            local,
            target_part="full_body",
            target_joints=[],
            frame_range=(0, T - 1),
            strength=strength,
        )
        pelvis = traj[:, 0:1, :]
        return out_local + pelvis, out_local, {
            "selection_mode": "fixed_tool_strength",
            "tool": "VelocitySmoothingTool",
            "target_part": "full_body",
            "strength": strength,
            "applied_representation": "motion_local",
            "correction_magnitude": report.correction_magnitude,
            "report_metadata": report.metadata,
        }

    if tool_name == "bone_projection_all":
        out_local = local.copy()
        reports = []
        for chain in CHAIN_LABELS:
            out_local, report = tools["BoneProjectionTool"].apply(
                out_local,
                target_part=chain,
                target_joints=[],
                frame_range=(0, T - 1),
                strength=strength,
            )
            reports.append(report)
        pelvis = traj[:, 0:1, :]
        return out_local + pelvis, out_local, {
            "selection_mode": "fixed_tool_strength_sequence",
            "tool": "BoneProjectionTool",
            "target_part": "all_chains",
            "strength": strength,
            "applied_representation": "motion_local",
            "correction_magnitude": float(np.mean([r.correction_magnitude for r in reports])),
            "n_tool_calls": len(reports),
        }

    raise ValueError(f"unknown variant: {variant}")


def _summarize_variant(seed_rows: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    by_prompt: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in seed_rows:
        by_prompt[row["sample_id"]].append(row["delta"])

    prompt_delta = {}
    for sid, rows in by_prompt.items():
        prompt_delta[sid] = {
            m: float(np.mean([r[m] for r in rows]))
            for m in METRICS
        }

    out = {"n_prompts": len(prompt_delta), "n_seed_rows": len(seed_rows), "metrics": {}}
    for m in METRICS:
        vals = np.array([d[m] for d in prompt_delta.values()], dtype=np.float64)
        before_vals = np.array([r["before"][m] for r in seed_rows if m in r["before"]], dtype=np.float64)
        after_vals = np.array([r["after"][m] for r in seed_rows if m in r["after"]], dtype=np.float64)
        out["metrics"][m] = {
            "delta_mean_after_minus_before": round(float(vals.mean()), 6) if len(vals) else 0.0,
            "delta_ci95": _boot_ci(vals, rng),
            "frac_improved_lower_is_better": round(float(np.mean(vals < 0.0)), 4) if len(vals) else 0.0,
            "before_seed_mean": round(float(before_vals.mean()), 6) if len(before_vals) else None,
            "after_seed_mean": round(float(after_vals.mean()), 6) if len(after_vals) else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path, default=REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608")
    ap.add_argument("--bank", type=Path, default=REPO_ROOT / "evals" / "prompts" / "protocol_rep_test_300_seed20260608.json")
    ap.add_argument("--generators", nargs="+", default=list(GENERATORS))
    ap.add_argument("--limit-prompts", type=int, default=None, help="debug only; first N prompts per generator")
    ap.add_argument("--seed", type=int, default=20260619)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "representative_refinement_effect_v1.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    bank = _load_json(args.bank)
    bank_ids = [r["sample_id"] for r in bank["rows"]]
    if args.limit_prompts is not None:
        bank_ids = bank_ids[: args.limit_prompts]
    keep_ids = set(bank_ids)

    tools = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
        "BoneProjectionTool": BoneProjectionTool(),
    }
    variants = ["noop"]
    variants += [f"foot_lock_{s}" for s in STRENGTHS]
    variants += [f"velocity_smoothing_{s}" for s in STRENGTHS]
    variants += [f"bone_projection_all_{s}" for s in STRENGTHS]

    seed_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        gen: {v: [] for v in variants} for gen in args.generators
    }
    examples: dict[str, dict[str, list[dict[str, Any]]]] = {
        gen: {"artifact_improved_foot_skate_worse": [], "best_foot_skate_improvement": []}
        for gen in args.generators
    }

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
            before = _measure(traj, local)
            before["fidelity_mpjpe"] = 0.0
            before["correction_magnitude"] = 0.0
            ratio = float(meta["actual_generated"] / max(meta["target_length"], 1))

            per_variant_for_sample = []
            for variant in variants:
                out_traj, out_local, prov = _apply_variant(variant, traj, local, meta, tools)
                after = _measure(out_traj, out_local)
                after["fidelity_mpjpe"] = float(np.mean(np.linalg.norm(out_local - local, axis=-1)))
                after["correction_magnitude"] = float(prov.get("correction_magnitude", 0.0))
                delta = {m: float(after[m] - before[m]) for m in METRICS}
                row = {
                    "sample_id": meta["sample_id"],
                    "seed": meta["seed"],
                    "prompt": meta["prompt"],
                    "length_ratio": ratio,
                    "variant": variant,
                    "before": before,
                    "after": after,
                    "delta": delta,
                    "provenance": prov,
                }
                seed_rows[gen][variant].append(row)
                per_variant_for_sample.append(row)

            best_fs = min(per_variant_for_sample, key=lambda r: r["delta"]["foot_skate_world"])
            if best_fs["variant"] != "noop":
                examples[gen]["best_foot_skate_improvement"].append({
                    "sample_id": meta["sample_id"], "seed": meta["seed"], "variant": best_fs["variant"],
                    "delta_foot_skate_world": round(best_fs["delta"]["foot_skate_world"], 6),
                    "delta_artifact_total": round(best_fs["delta"]["artifact_total"], 6),
                    "fidelity_mpjpe": round(best_fs["after"]["fidelity_mpjpe"], 6),
                    "prompt": meta["prompt"][:100],
                })
            for r in per_variant_for_sample:
                if r["delta"]["artifact_total"] < 0 and r["delta"]["foot_skate_world"] > 0.001:
                    examples[gen]["artifact_improved_foot_skate_worse"].append({
                        "sample_id": meta["sample_id"], "seed": meta["seed"], "variant": r["variant"],
                        "delta_foot_skate_world": round(r["delta"]["foot_skate_world"], 6),
                        "delta_artifact_total": round(r["delta"]["artifact_total"], 6),
                        "fidelity_mpjpe": round(r["after"]["fidelity_mpjpe"], 6),
                        "prompt": meta["prompt"][:100],
                    })

            if idx % 100 == 0:
                print(f"[{gen}] {idx}/{len(metas)} seed rows")

    summaries: dict[str, Any] = {}
    for gen in args.generators:
        summaries[gen] = {"native_all": {}, "sensitivity_ratio_ge_0.6": {}}
        for variant in variants:
            rows = seed_rows[gen][variant]
            summaries[gen]["native_all"][variant] = _summarize_variant(rows, rng)
            sens_rows = [r for r in rows if (gen != "motiongpt" or r["length_ratio"] >= 0.6)]
            summaries[gen]["sensitivity_ratio_ge_0.6"][variant] = _summarize_variant(sens_rows, rng)

    # Compact top examples only.
    for gen in examples:
        examples[gen]["best_foot_skate_improvement"] = sorted(
            examples[gen]["best_foot_skate_improvement"],
            key=lambda x: x["delta_foot_skate_world"],
        )[:8]
        examples[gen]["artifact_improved_foot_skate_worse"] = sorted(
            examples[gen]["artifact_improved_foot_skate_worse"],
            key=lambda x: -x["delta_foot_skate_world"],
        )[:8]

    out = {
        "schema_version": "1.0.0",
        "record_type": "representative_refinement_effect",
        "board_id": "AR-051",
        "pool": str(args.pool_root.name),
        "bank": str(args.bank.name),
        "generators": args.generators,
        "variants": variants,
        "stat_unit": "prompt (seed rows averaged before aggregate); sensitivity excludes MotionGPT seed rows with actual/target < 0.6",
        "metric_provenance": {
            "foot_skate_world": "Category B variant: GMD/EDGE formula on motion_trajectory, ground=10th percentile",
            "accel": "Category B variant: local-pose per-joint acceleration mean",
            "artifact_total/*_fire": "Category C proxy: local evaluators + AR-043 thresholds",
            "fidelity_mpjpe": "Category B variant: MPJPE vs original generator output",
        },
        "application_protocol": {
            "FootLockTool": "applied on motion_trajectory with per-sample ground_y, then local=trajectory-pelvis",
            "VelocitySmoothingTool": "applied on motion_local, then trajectory=local+original pelvis path",
            "BoneProjectionTool": "applied on motion_local to all CHAIN_LABELS sequentially, then trajectory=local+original pelvis path",
        },
        "summaries": summaries,
        "examples": examples,
        "claim_boundary": "Tool effect / side-effect diagnostic. Category-A quality metrics and visual pack are separate AR-051 passes; learned policy superiority not claimed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== AR-051 representative refinement effect (physical/proxy) ===")
    for gen in args.generators:
        print(f"\n[{gen}]")
        for variant in variants:
            met = summaries[gen]["native_all"][variant]["metrics"]
            print(
                f"  {variant:<28} "
                f"d_skate={met['foot_skate_world']['delta_mean_after_minus_before']:+.6f} "
                f"d_art={met['artifact_total']['delta_mean_after_minus_before']:+.6f} "
                f"d_accel={met['accel']['delta_mean_after_minus_before']:+.6f} "
                f"fidMPJPE={met['fidelity_mpjpe']['after_seed_mean']:.6f}"
            )
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
