"""Step 3 (사용자 directive 2026-05-31): G2 real action-effect transition dataset (split v2).

Stage 2-G split v2 의 G2/clean/synthetic state 들에 대해 (3 tool × u-grid 11 점) action-effect
transition 산출. Stage A (Stage 2-G v1 의 broad transition) 와 달리 본 데이터셋은:

  - Split v2 와 1:1 매핑 (train/calib/holdouts/reserve/clean/synthetic 모두 split tag).
  - G2 transitions: Protocol B (vs state motion, no clean GT).
  - clean_noharm transitions: Protocol A (vs self — 모든 correction 이 STOP 보다 손해).
  - synthetic_* transitions: Protocol A (vs clean — Stage B-1 corruption 재현).
  - u-grid {0.0, 0.1, …, 1.0} 11 점 (continuous override).

저장: compact schema (states map + transitions list, state featurization once per state).

CLI:
    python -m tools.rl2_transition_g2_real_build \
        --split evals/splits/g2_real_stress_split_v2.json \
        --u-grid 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
        --output evals/snapshots/rl2_transition_g2_real_stage2_v1.json
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

from correction_tools import BoneProjectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from tools.synthetic_injection import inject_foot_floating, inject_jitter
from tools.safe_sequence_oracle_run import _gate_scores, _gate_violation
from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, REAL_DISTRIBUTION, common_snapshot_metadata
from tools.rl2_build_training_data import ARTIFACT_EVALUATORS, PHYSICAL_EVALUATORS
from tools.rl2_transition_build import _sweep_state, PROTOCOL_A_DISTS

DEFAULT_CALIBRATION = REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json"
DEFAULT_EXISTING_POOL = REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527"
DEFAULT_BALANCED_POOL = REPO_ROOT / "external_assets" / "g2_generated_hml3d_balanced300_seed20260531"
DEFAULT_HML3D_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"


def _resolve_g2(trial_id: str, existing_pool: Path, balanced_pool: Path) -> tuple[Path, str]:
    """trial_id → (pool_dir, stem). 'pool_name/motion_NNN' 형식 또는 prefix 없으면 existing."""
    if "/" in trial_id:
        prefix, stem = trial_id.split("/", 1)
        return balanced_pool, stem  # prefix 가 있으면 balanced.
    return existing_pool, trial_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path,
                        default=REPO_ROOT / "evals" / "splits" / "g2_real_stress_split_v2.json")
    parser.add_argument("--existing-pool", type=Path, default=DEFAULT_EXISTING_POOL)
    parser.add_argument("--balanced-pool", type=Path, default=DEFAULT_BALANCED_POOL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_HML3D_DIR)
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--u-grid", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_g2_real_stage2_v1.json")
    args = parser.parse_args()

    u_grid = [round(float(x), 4) for x in args.u_grid.split(",")]
    print(f"[INFO] u_grid ({len(u_grid)}): {u_grid}")
    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_thresholds = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                       if calib["summary"][n].get("n", 0) > 0}
    alpha = float(CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1["alpha"])
    evaluators = list(DEFAULT_EVALUATORS); gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)

    split_data = json.load(open(args.split, encoding="utf-8"))
    splits = split_data["splits"]
    # G2 splits + clean + synthetic.
    G2_SPLITS = ("train_g2_real", "calib_g2_real", "g2_stress_holdout",
                 "g2_natural_holdout", "reserve")
    CLEAN_SPLITS = ("clean_noharm_holdout",)
    SYN_SPLITS = ("synthetic_aux_train", "synthetic_diag_holdout")

    # Load profile v2 for state metadata (motion_group, band).
    profile_v2 = json.load(open(REPO_ROOT / "evals" / "snapshots" / "g2_real_stress_profile_v2.json", encoding="utf-8"))
    profile_idx = {r["trial_id"]: r for r in profile_v2["per_motion"]}

    states = {}        # state_id → {distribution, split, motion_group, band, state_feats}
    transitions = []   # rows
    state_counter = Counter()

    def _add_state(state_id, dist, split_name, motion_group, band, motion_state, clean_motion):
        if state_id in states:
            return  # 이미 처리.
        sweep = _sweep_state(motion_state, dist, clean_motion, evaluators, gate_evaluators,
                             gate_thresholds, u_grid, alpha, state_id)
        if not sweep:
            return
        # State feats = first row's state (동일).
        s0 = sweep[0]["state"]
        states[state_id] = {
            "distribution": dist, "split": split_name,
            "motion_group": motion_group, "band": band,
            "artifact_scores": s0["artifact_scores"], "physical_scores": s0["physical_scores"],
        }
        state_counter[split_name] += 1
        # Transitions (compact): omit redundant fields.
        for r in sweep:
            transitions.append({
                "state_id": state_id, "split": split_name, "distribution": dist,
                "motion_group": motion_group, "band": band,
                "tool": r["tool"], "u": r["u"],
                "gate_result": r["gate_result"],
                "safe_utility": r["safe_utility"],
                "artifact_reduction": r["artifact_reduction"],
                "fidelity_loss": r["fidelity_loss"],
                "correction_magnitude": r["correction_magnitude"],
            })

    # G2 splits.
    for split_name in G2_SPLITS:
        ids = splits[split_name]["ids"]
        print(f"[INFO] {split_name}: {len(ids)} G2 states")
        for i, tid in enumerate(ids, 1):
            pool_dir, stem = _resolve_g2(tid, args.existing_pool, args.balanced_pool)
            npy = pool_dir / f"{stem}.npy"
            if not npy.exists():
                print(f"   [WARN] missing: {npy}"); continue
            m = np.load(str(npy)).astype(np.float64)
            if m.ndim != 3 or m.shape[1] != 22:
                continue
            prof = profile_idx.get(tid, {})
            mg = prof.get("motion_group", "unknown"); band = prof.get("band", "unknown")
            _add_state(tid, "g2_natural", split_name, mg, band, m, m)  # Protocol B.
            if i % 50 == 0:
                print(f"   {split_name} {i}/{len(ids)} ({len(transitions)} transitions)")

    # clean_noharm.
    for split_name in CLEAN_SPLITS:
        ids = splits[split_name]["ids"]
        print(f"[INFO] {split_name}: {len(ids)} clean states")
        for i, tid in enumerate(ids, 1):
            npy = args.data_dir / f"{tid}.npy"
            if not npy.exists():
                continue
            clean = np.load(str(npy)).astype(np.float64)
            if clean.ndim != 3 or clean.shape[1] != 22:
                continue
            _add_state(tid, "clean", split_name, "clean", "clean", clean, clean)  # Protocol A vs self.
            if i % 50 == 0:
                print(f"   {split_name} {i}/{len(ids)}")

    # synthetic_*.
    for split_name in SYN_SPLITS:
        ids = splits[split_name]["ids"]
        print(f"[INFO] {split_name}: {len(ids)} synthetic states")
        for i, tid in enumerate(ids, 1):
            npy = args.data_dir / f"{tid}.npy"
            if not npy.exists():
                continue
            clean = np.load(str(npy)).astype(np.float64)
            if clean.ndim != 3 or clean.shape[1] != 22:
                continue
            m1 = inject_foot_floating(clean, lift_height=0.08, seed=args.synthetic_seed)
            corrupted = inject_jitter(m1, noise_std=0.05, seed=args.synthetic_seed + 1000)
            _add_state(tid, "synthetic_severe", split_name, "synthetic", "synthetic", corrupted, clean)
            if i % 30 == 0:
                print(f"   {split_name} {i}/{len(ids)}")

    # Surface summary per (split, tool, u): mean utility + violation rate.
    def _summary_by(group_key):
        agg = {}
        for t in transitions:
            k = (group_key(t), t["tool"], t["u"])
            a = agg.setdefault(k, {"util": [], "viol": []})
            a["util"].append(t["safe_utility"])
            a["viol"].append(1.0 if t["gate_result"] == "hard_violation" else 0.0)
        out = {}
        for (gk, tool, u), v in sorted(agg.items()):
            key = f"{gk}|{tool}|u={u}"
            out[key] = {"mean_utility": round(float(np.mean(v["util"])), 6),
                        "violation_rate": round(float(np.mean(v["viol"])), 4),
                        "n": len(v["util"])}
        return out

    per_split_split_summary = _summary_by(lambda t: t["split"])

    n_g2_states = sum(1 for sid, m in states.items() if m["distribution"] == "g2_natural")
    n_clean_states = sum(1 for sid, m in states.items() if m["distribution"] == "clean")
    n_syn_states = sum(1 for sid, m in states.items() if m["distribution"] == "synthetic_severe")
    out = {
        "schema_version": "1.0.0", "record_type": "rl2_transition_g2_real",
        "task_id": "rl2_transition_g2_real_stage2_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "rl2_transition_g2_real_stage2_v1",
            oracle_type="action_effect_transition",
            action_grid="continuous-dense-u11", stage="Stage-2G-step3-transitions",
            evidence_tier=[REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
            evaluators=evaluators, gate_evaluators=gate_evaluators,
        ),
        "directive": "Stage 2-G v2 split 의 모든 state 에 (3 tool × u-grid 11) action-effect transition. G2=Protocol B, clean=A vs self, synthetic=A vs clean.",
        "u_grid": u_grid, "alpha": alpha,
        "n_states_total": len(states),
        "state_counts_per_split": dict(state_counter),
        "n_transitions": len(transitions),
        "n_g2_states": n_g2_states, "n_clean_states": n_clean_states, "n_synthetic_states": n_syn_states,
        "split_summary_compact": per_split_split_summary,
        "states": states,
        "transitions": transitions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== RL-2 Transition Dataset Stage 2 (G2 real, split v2) ===")
    print(f"  total states: {len(states)} (G2 {n_g2_states} / clean {n_clean_states} / synthetic {n_syn_states})")
    print(f"  total transitions: {len(transitions)}")
    print(f"  per-split state counts: {dict(state_counter)}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
