"""RL-2 Stage 1 (사용자 directive 2026-05-29): bounded continuous action-effect surface 데이터.

사용자 directive 박제:
> "policy(s) → action class 가 아니라 Q_safe(s, tool, u) 를 학습한다. u ∈ [0,1] 은 bounded
>  continuous intervention intensity. continuous 를 목표로 하되, 데이터는 grid-sampled action
>  effects 로 만든다. Stage 1: u_grid = {0.3, 0.6, 1.0}."

본 도구는 discrete (state, oracle_action) imitation dataset 과 달리, **각 state 에서 모든
(tool, u) 후보의 실제 action effect 를 측정** 한다. 즉 label 이 action 하나가 아니라 action
effect measurement (`(state, tool, u, effect)`) 다.

State 는 imitation 과 동일 featurization (rl2_build_training_data._build_state, 10-action 3-level).
State set 은 oracle safe_best sequence 의 prefix state (imitation 이 본 것과 동일 분포).
각 state 에서 9 후보 (3 tool × 3 u) 의 single-step effect 를 측정.

Effect fields (사용자 directive):
  artifact_reduction       — target artifact 감소량 (positive = 개선)
  fidelity_loss            — G2: MPJPE(after, before) [Protocol B local] / synthetic: ΔMPJPE-to-clean [Protocol A]
  correction_magnitude     — report.correction_magnitude
  physical_violation       — regression-based gate hard violation (any)
  safe_utility             — artifact_reduction - α·fidelity_loss (α=5.0; violation 시 selection -inf)
  is_safe                  — not physical_violation
STOP (u=0) 은 utility 0 / safe 로 inference 에서 고정 baseline (dataset 에는 미포함).

u → strength (Stage 1, 3-level): {0.3: small, 0.6: medium, 1.0: large}
(action_space_provenance §5-2-2 normalized intensity mapping).

CLI:
    python -m tools.rl2_q_surface_build \
        --g2-oracle evals/snapshots/safe_sequence_oracle_g2_hml3d_test300_3level_v1.json \
        --synthetic-oracle evals/snapshots/safe_sequence_oracle_synthetic_severe_3level_v1.json \
        --u-grid 0.3,0.6,1.0 \
        --output evals/snapshots/rl2_q_surface_dataset_stage1_v1.json

근거 (AGENTS.md §3-22): grid-sampled action effects (Masson et al. AAAI 2016 parameterized
action). safe_utility = Category C internal routing reward (metric_provenance §4-1-1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS, EvaluatorReport
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from tools.synthetic_injection import inject_foot_floating, inject_jitter
from tools.safe_sequence_oracle_run import _gate_scores, _gate_violation
from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, REAL_DISTRIBUTION, common_snapshot_metadata
from tools.rl2_build_training_data import (
    ARTIFACT_EVALUATORS, PHYSICAL_EVALUATORS, TOOLS_ORDER, TOOL_TARGET,
    _artifact_scores, _physical_scores, _build_state, build_action_list,
)

TOOL_BY_NAME: dict[str, CorrectionTool] = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}
# u → 3-level strength token (Stage 1). action_space_provenance §5-2-2.
U_TO_STRENGTH_3LEVEL = {0.3: "small", 0.6: "medium", 1.0: "large"}
TARGET_EVALUATORS_A = ("FootFloatingEvaluator", "VelocityJitterEvaluator")
DEFAULT_CALIBRATION = REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json"
# 3-level action list (STOP + 3 tool × 3 strength) for prev_action one-hot dim consistency.
ACTION_LIST_3 = build_action_list(("small", "medium", "large"))
N_ACTIONS_3 = len(ACTION_LIST_3)  # 10


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _mpjpe(a, b) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _target_full(motion, evaluators) -> float:
    by = {ev.name: ev.evaluate(motion) for ev in evaluators}
    return float(np.mean([_max_score(by.get(n, [])) for n in ARTIFACT_EVALUATORS]))


def _target_A(motion, evaluators) -> float:
    by = {ev.name: ev.evaluate(motion) for ev in evaluators}
    return float(np.mean([_max_score(by.get(n, [])) for n in TARGET_EVALUATORS_A]))


def _u_strength(u: float) -> str:
    return U_TO_STRENGTH_3LEVEL[round(u, 3)]


def _measure_effects(motion_s, dist_tag, clean, evaluators, gate_evaluators,
                     gate_thresholds, u_grid, alpha):
    """state s 의 motion 에서 9 (tool, u) 후보의 single-step effect 측정."""
    T = motion_s.shape[0]
    gate_parent = _gate_scores(motion_s, gate_evaluators)
    if dist_tag == 0:  # G2: Protocol B (vs current state motion).
        target_before = _target_full(motion_s, evaluators)
    else:              # synthetic: Protocol A (vs clean).
        target_before = _target_A(motion_s, evaluators)
        mpjpe_before_clean = _mpjpe(motion_s, clean)
    effects = []
    for tool_name in TOOLS_ORDER:
        tp = TOOL_TARGET[tool_name]
        tool = TOOL_BY_NAME[tool_name]
        for u in u_grid:
            st = _u_strength(u)
            try:
                new_motion, report = tool.apply(motion_s, target_part=tp, target_joints=[],
                                                frame_range=(0, T - 1), strength=st)
            except ValueError:
                continue
            if dist_tag == 0:
                target_after = _target_full(new_motion, evaluators)
                artifact_reduction = target_before - target_after
                fidelity_loss = _mpjpe(new_motion, motion_s)
            else:
                target_after = _target_A(new_motion, evaluators)
                artifact_reduction = target_before - target_after
                fidelity_loss = _mpjpe(new_motion, clean) - mpjpe_before_clean
            gate_after = _gate_scores(new_motion, gate_evaluators)
            decisions = _gate_violation(gate_after, gate_parent, gate_thresholds)
            violation = any(d == "hard_violation" for d in decisions.values())
            safe_utility = artifact_reduction - alpha * fidelity_loss
            effects.append({
                "tool": tool_name, "u": float(u), "strength": st,
                "artifact_reduction": float(artifact_reduction),
                "fidelity_loss": float(fidelity_loss),
                "correction_magnitude": float(report.correction_magnitude),
                "physical_violation": bool(violation),
                "is_safe": (not violation),
                "safe_utility": float(safe_utility),
            })
    return effects


def _build_sample_rows(motion0, safe_seq, dist_tag, clean, evaluators, gate_evaluators,
                       gate_thresholds, u_grid, alpha, max_depth):
    """oracle safe_best prefix state 들을 replay 하며 각 state 에서 9 후보 effect 측정."""
    rows = []
    motion = motion0.copy()
    T = motion.shape[0]
    artifact = _artifact_scores(motion, evaluators)
    physical = _physical_scores(motion, gate_evaluators)
    prev_artifact, prev_physical = list(artifact), list(physical)
    prev_action_idx = 0
    n_steps = len(safe_seq)
    for t in range(n_steps + 1):
        delta = [a - pa for a, pa in zip(artifact, prev_artifact)] + \
                [p - pp for p, pp in zip(physical, prev_physical)]
        state = _build_state(artifact, physical, delta, prev_action_idx,
                             max_depth - t, t, dist_tag, n_actions=N_ACTIONS_3)
        effects = _measure_effects(motion, dist_tag, clean, evaluators, gate_evaluators,
                                   gate_thresholds, u_grid, alpha)
        rows.append({"step": t, "state": state, "effects": effects})
        if t >= n_steps:
            break
        step = safe_seq[t]
        tool = TOOL_BY_NAME[step[0]]
        new_motion, _ = tool.apply(motion, target_part=step[1], target_joints=[],
                                    frame_range=(0, T - 1), strength=step[2])
        motion = new_motion
        prev_artifact, prev_physical = list(artifact), list(physical)
        artifact = _artifact_scores(motion, evaluators)
        physical = _physical_scores(motion, gate_evaluators)
        # advance prev_action_idx (3-level action index).
        key = f"{step[0]}|{step[2]}"
        prev_action_idx = ACTION_LIST_3.index(key) if key in ACTION_LIST_3 else 0
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_g2_hml3d_test300_3level_v1.json")
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--synthetic-oracle", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_synthetic_severe_3level_v1.json")
    parser.add_argument("--synthetic-data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--u-grid", type=str, default="0.3,0.6,1.0")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_q_surface_dataset_stage1_v1.json")
    args = parser.parse_args()

    u_grid = [float(x) for x in args.u_grid.split(",")]
    print(f"[INFO] u_grid: {u_grid} -> strengths {[_u_strength(u) for u in u_grid]}")
    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_thresholds = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                       if calib["summary"][n].get("n", 0) > 0}
    alpha = float(CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1["alpha"])
    evaluators = list(DEFAULT_EVALUATORS)
    gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)

    all_rows = []  # each: {sample_id, distribution, step, state, effects}

    # === G2 ===
    g2 = json.load(open(args.g2_oracle, encoding="utf-8"))
    print(f"[INFO] G2 oracle: {len(g2['per_sample'])} samples")
    for i, ps in enumerate(g2["per_sample"], 1):
        tid = ps["trial_id"]; sb = ps.get("safe_best")
        if sb is None:
            continue
        npy = args.g2_batch_dir / f"{tid}.npy"
        if not npy.exists():
            continue
        motion0 = np.load(str(npy)).astype(np.float64)
        safe_seq = sb["sequence"] if sb["length"] > 0 else []
        rows = _build_sample_rows(motion0, safe_seq, 0, motion0, evaluators, gate_evaluators,
                                  gate_thresholds, u_grid, alpha, args.max_depth)
        for r in rows:
            all_rows.append({"sample_id": tid, "distribution": "g2", **r})
        if i % 50 == 0:
            print(f"   G2 {i}/{len(g2['per_sample'])}")

    # === Synthetic severe ===
    syn = json.load(open(args.synthetic_oracle, encoding="utf-8"))
    print(f"[INFO] synthetic oracle: {len(syn['per_sample'])} samples")
    for i, ps in enumerate(syn["per_sample"], 1):
        tid = ps["trial_id"]; sb = ps.get("safe_best")
        if sb is None:
            continue
        npy = args.synthetic_data_dir / f"{tid}.npy"
        if not npy.exists():
            continue
        clean = np.load(str(npy)).astype(np.float64)
        m1 = inject_foot_floating(clean, lift_height=0.08, seed=args.synthetic_seed)
        corrupted = inject_jitter(m1, noise_std=0.05, seed=args.synthetic_seed + 1000)
        safe_seq = sb["sequence"] if sb["length"] > 0 else []
        rows = _build_sample_rows(corrupted, safe_seq, 1, clean, evaluators, gate_evaluators,
                                  gate_thresholds, u_grid, alpha, args.max_depth)
        for r in rows:
            all_rows.append({"sample_id": tid, "distribution": "synthetic", **r})
        if i % 20 == 0:
            print(f"   synthetic {i}/{len(syn['per_sample'])}")

    # Effect summary (per distribution, per (tool, u)): mean utility + violation rate.
    def _surface_summary(dist):
        agg = {}
        for r in all_rows:
            if r["distribution"] != dist:
                continue
            for e in r["effects"]:
                k = f"{e['tool']}|u={e['u']}"
                agg.setdefault(k, {"util": [], "viol": [], "art": [], "fid": []})
                agg[k]["util"].append(e["safe_utility"])
                agg[k]["viol"].append(1.0 if e["physical_violation"] else 0.0)
                agg[k]["art"].append(e["artifact_reduction"])
                agg[k]["fid"].append(e["fidelity_loss"])
        return {k: {"mean_utility": float(np.mean(v["util"])),
                    "violation_rate": float(np.mean(v["viol"])),
                    "mean_artifact_reduction": float(np.mean(v["art"])),
                    "mean_fidelity_loss": float(np.mean(v["fid"])),
                    "n": len(v["util"])} for k, v in sorted(agg.items())}

    n_g2 = sum(1 for r in all_rows if r["distribution"] == "g2")
    n_syn = sum(1 for r in all_rows if r["distribution"] == "synthetic")
    out = {
        "schema_version": "1.0.0", "record_type": "rl2_q_surface_dataset",
        "task_id": "rl2_q_surface_dataset_stage1_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "rl2_q_surface_dataset_stage1_v1",
            oracle_type="sequence", action_grid="3-level", stage="RL-2-Q-surface-stage1",
            evidence_tier=[REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
            evaluators=evaluators, gate_evaluators=gate_evaluators,
        ),
        "formulation": "Q_safe(s, tool, u) bounded continuous action-effect surface (Stage 1 grid)",
        "u_grid": u_grid, "u_to_strength": {str(u): _u_strength(u) for u in u_grid},
        "alpha": alpha, "max_depth": args.max_depth,
        "safe_utility_metric": "Category C internal routing reward (metric_provenance §4-1-1)",
        "netgain_protocol": {"g2": "B (vs state motion)", "synthetic": "A (vs clean)"},
        "n_state_rows": len(all_rows), "n_g2_state_rows": n_g2, "n_synthetic_state_rows": n_syn,
        "n_candidates_per_state": len(TOOLS_ORDER) * len(u_grid),
        "surface_summary_g2": _surface_summary("g2"),
        "surface_summary_synthetic": _surface_summary("synthetic"),
        "rows": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== RL-2 Q-surface Dataset (Stage 1) ===")
    print(f"  state rows: {len(all_rows)} (G2 {n_g2}, synthetic {n_syn})")
    print(f"  candidates/state: {out['n_candidates_per_state']} (3 tool × {len(u_grid)} u)")
    for dist in ("g2", "synthetic"):
        print(f"\n  [{dist}] surface (mean_utility / violation_rate):")
        for k, v in out[f"surface_summary_{dist}"].items():
            print(f"    {k:<32} util={v['mean_utility']:+.4f}  viol={v['violation_rate']*100:5.1f}%  "
                  f"art={v['mean_artifact_reduction']:+.4f}  fid={v['mean_fidelity_loss']:+.4f}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
