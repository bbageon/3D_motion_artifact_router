"""Step E-3 (사용자 directive 2026-05-27): synthetic severe Safe Sequence Oracle.

사용자 directive:
> "G2-HML3DText-300은 low/moderate severity 중심, synthetic severe는 high severity
>  multi-step correction 학습용. RL-2 teacher가 균형 잡히려면: G2 natural → STOP/weak
>  correction 학습, synthetic severe → multi-step correction 학습. 둘 다 필요."

본 도구는 synthetic 의 high-severity corruption (foot_floating 0.08 + jitter 0.05) 위에서
gate-aware Safe Sequence Oracle 을 적용 — multi-step correction 의 safe path 가 gate 를
통과하는지 + unsafe path 가 pruning 되는지 측정.

NetGain = Protocol A (clean GT reference, oracle_sequence_multi_5level_run 와 동일).
Gate = PhysicalGateV0 regression-based (before+eps), safe_sequence_oracle_run 와 동일.
Comparison: unsafe_best (gate 무관 NetGain max) vs safe_best (gate-aware).

CLI:
    python -m tools.safe_sequence_oracle_synthetic_run \
        --n-samples 60 --seed 42 --max-depth 3 \
        --output evals/snapshots/safe_sequence_oracle_synthetic_severe_v1.json

근거 (AGENTS.md §3-22): synthetic = controlled diagnostic (AGENTS.md §3-17), high-severity
multi-step correction 학습 영역. PhysDiff (Yuan 2023 ICCV) gate framework.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS, EvaluatorReport
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from tools.synthetic_injection import inject_foot_floating, inject_jitter
# Reuse gate functions from the G2 safe oracle.
from tools.safe_sequence_oracle_run import _gate_scores, _gate_violation
from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, common_snapshot_metadata

ALL_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
TARGET_EVALUATORS_A = ("FootFloatingEvaluator", "VelocityJitterEvaluator")
TOOLS_WITH_TARGET_PARTS = [
    ("FootLockTool", "both_feet"),
    ("BoneProjectionTool", "right_arm"),
    ("VelocitySmoothingTool", "full_body"),
]
STRENGTHS_5LEVEL = ("xsmall", "small5", "medium5", "large5", "xlarge")
STRENGTH_RANK_5 = {"xsmall": 0, "small5": 1, "medium5": 2, "large5": 3, "xlarge": 4}
STRENGTHS_3LEVEL = ("small", "medium", "large")
STRENGTH_RANK_3 = {"small": 0, "medium": 1, "large": 2}
_GRID = {"5level": (STRENGTHS_5LEVEL, STRENGTH_RANK_5),
         "3level": (STRENGTHS_3LEVEL, STRENGTH_RANK_3)}
DEFAULT_CALIBRATION = REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _multi_inject(clean: np.ndarray, seed: int) -> np.ndarray:
    m1 = inject_foot_floating(clean, lift_height=0.08, seed=seed)
    return inject_jitter(m1, noise_std=0.05, seed=seed + 1000)


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _evaluate_all(motion, evaluators) -> dict:
    return {ev.name: ev.evaluate(motion) for ev in evaluators}


def _target_score_A(reports) -> float:
    return float(np.mean([_max_score(reports.get(n, [])) for n in TARGET_EVALUATORS_A]))


def _strength_allowed(used, tn, tp, st, strength_rank=STRENGTH_RANK_5) -> bool:
    prev = used.get((tn, tp), [])
    if not prev:
        return True
    return strength_rank[st] < min(strength_rank[s] for s in prev)


def select_safe_sequence_synthetic(
    *, clean_motion, corrupted_motion, tools_by_name, evaluators, gate_evaluators,
    gate_thresholds, netgain_weights, max_depth=3, top_k=10, score_tol=0.01,
    strengths=STRENGTHS_5LEVEL, strength_rank=STRENGTH_RANK_5,
):
    T = corrupted_motion.shape[0]
    frame_range = (0, T - 1)
    reports_init = _evaluate_all(corrupted_motion, evaluators)
    target_init_A = _target_score_A(reports_init)
    total_init = sum(_max_score(r) for r in reports_init.values())
    mpjpe_corr_clean = _mpjpe(corrupted_motion, clean_motion)
    gate_init = _gate_scores(corrupted_motion, gate_evaluators)
    actions = [(tn, tp, st) for tn, tp in TOOLS_WITH_TARGET_PARTS for st in strengths]

    alpha = float(netgain_weights["alpha"])
    beta = float(netgain_weights["beta"])
    gamma = float(netgain_weights["gamma"])

    all_candidates = []
    counters = {"explored": 0, "pruned_strength": 0, "pruned_score": 0,
                "pruned_apply_error": 0, "pruned_gate_violation": 0}

    def _record(motion, sequence, cum_corr, gate_after, decisions):
        reports = _evaluate_all(motion, evaluators)
        target_final_A = _target_score_A(reports)
        target_delta_A = target_final_A - target_init_A
        mpjpe_final = _mpjpe(motion, clean_motion)
        fid_loss = mpjpe_final - mpjpe_corr_clean
        netgain = -target_delta_A - alpha * fid_loss - beta * cum_corr - gamma * len(sequence)
        n_hard = sum(1 for d in decisions.values() if d == "hard_violation")
        all_candidates.append({
            "sequence": [list(s) for s in sequence], "length": len(sequence),
            "target_delta_A": float(target_delta_A),
            "fidelity_loss_protocol_a": float(fid_loss),
            "cumulative_correction_magnitude": float(cum_corr),
            "netgain": float(netgain),
            "gate_scores": gate_after, "gate_decisions": decisions,
            "n_hard_violations": n_hard, "any_hard_violation": n_hard > 0,
        })
        counters["explored"] += 1

    def _dfs(motion, sequence, cum_corr, prev_total, used, parent_gate):
        if not sequence:
            _record(motion, sequence, cum_corr, gate_init, {n: "pass" for n in gate_init})
        if len(sequence) >= max_depth:
            return
        for tn, tp, st in actions:
            if not _strength_allowed(used, tn, tp, st, strength_rank):
                counters["pruned_strength"] += 1
                continue
            tool = tools_by_name[tn]
            try:
                new_motion, report = tool.apply(motion, target_part=tp, target_joints=[],
                                                 frame_range=frame_range, strength=st)
            except ValueError:
                counters["pruned_apply_error"] += 1
                continue
            new_reports = _evaluate_all(new_motion, evaluators)
            new_total = sum(_max_score(r) for r in new_reports.values())
            if new_total > prev_total + score_tol:
                counters["pruned_score"] += 1
                continue
            gate_after = _gate_scores(new_motion, gate_evaluators)
            decisions = _gate_violation(gate_after, parent_gate, gate_thresholds)
            _record(new_motion, sequence + [(tn, tp, st)],
                    cum_corr + float(report.correction_magnitude), gate_after, decisions)
            if any(d == "hard_violation" for d in decisions.values()):
                counters["pruned_gate_violation"] += 1
                continue
            new_used = dict(used)
            new_used[(tn, tp)] = list(used.get((tn, tp), [])) + [st]
            _dfs(new_motion, sequence + [(tn, tp, st)],
                 cum_corr + float(report.correction_magnitude), new_total, new_used, gate_after)

    _dfs(corrupted_motion, [], 0.0, total_init, {}, gate_init)
    if not all_candidates:
        return {"safe_best": None, "unsafe_best": None, "counters": counters,
                "metadata": {"gate_scores_initial": gate_init}}
    safe_cands = [c for c in all_candidates if not c["any_hard_violation"]]
    safe_best = max(safe_cands, key=lambda c: (c["netgain"], -c["length"])) if safe_cands else None
    unsafe_best = max(all_candidates, key=lambda c: (c["netgain"], -c["length"]))
    return {
        "safe_best": safe_best, "unsafe_best": unsafe_best,
        "safe_top_k": sorted(safe_cands, key=lambda c: (c["netgain"], -c["length"]), reverse=True)[:top_k],
        "unsafe_top_k": sorted(all_candidates, key=lambda c: (c["netgain"], -c["length"]), reverse=True)[:top_k],
        "counters": counters,
        "metadata": {"gate_scores_initial": gate_init, "mpjpe_corrupted_clean": mpjpe_corr_clean,
                     "n_candidates_total": len(all_candidates), "n_safe_candidates": len(safe_cands)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--n-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--strength-grid", type=str, default="5level", choices=["5level", "3level"])
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_synthetic_severe_v1.json")
    args = parser.parse_args()
    strengths, strength_rank = _GRID[args.strength_grid]
    print(f"[INFO] strength grid: {args.strength_grid} -> {strengths}")

    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_thresholds = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                       if calib["summary"][n].get("n", 0) > 0}
    print(f"[INFO] gate thresholds (p99): {gate_thresholds}")

    tools_by_name = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "BoneProjectionTool": BoneProjectionTool(),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
    }
    evaluators = list(DEFAULT_EVALUATORS)
    gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)
    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1

    rng = np.random.default_rng(args.seed)
    npy_files = sorted(args.data_dir.glob("*.npy"))
    chosen_idx = rng.choice(len(npy_files), size=args.n_samples, replace=False)
    chosen = [npy_files[i] for i in chosen_idx]
    print(f"[INFO] {len(chosen)} synthetic samples (seed={args.seed})")

    per_sample = []
    table = []
    for i, path in enumerate(chosen, 1):
        tid = path.stem
        try:
            clean = np.load(str(path)).astype(np.float64)
        except Exception as e:
            print(f"[WARN] {tid}: {e}"); continue
        if clean.ndim != 3 or clean.shape[1] != 22 or clean.shape[2] != 3:
            continue
        corrupted = _multi_inject(clean, seed=args.seed)
        result = select_safe_sequence_synthetic(
            clean_motion=clean, corrupted_motion=corrupted, tools_by_name=tools_by_name,
            evaluators=evaluators, gate_evaluators=gate_evaluators,
            gate_thresholds=gate_thresholds, netgain_weights=netgain_weights,
            max_depth=args.max_depth, top_k=args.top_k,
            strengths=strengths, strength_rank=strength_rank,
        )
        sb, ub = result["safe_best"], result["unsafe_best"]
        def _seqsum(c):
            if c is None: return "NONE"
            return "STOP" if c["length"] == 0 else " → ".join(f"{a[0].replace('Tool','')}/{a[2]}" for a in c["sequence"])
        if i % 10 == 0 or i <= 3:
            print(f"[{i}/{len(chosen)}] {tid}: safe={_seqsum(sb)} (ng={sb['netgain']:+.4f})" if sb else f"[{i}] {tid}: safe=NONE",
                  f"| unsafe={_seqsum(ub)} (ng={ub['netgain']:+.4f}, hard={ub['n_hard_violations']})" if ub else "")
        decision_shift = ("unchanged" if (sb and ub and sb["sequence"] == ub["sequence"])
                          else ("none_to_safe" if sb is None else "shifted"))
        table.append({
            "trial_id": tid,
            "unsafe_netgain": ub["netgain"] if ub else None,
            "unsafe_hard_viol": ub["n_hard_violations"] if ub else None,
            "unsafe_sequence": _seqsum(ub), "unsafe_length": ub["length"] if ub else None,
            "safe_netgain": sb["netgain"] if sb else None,
            "safe_sequence": _seqsum(sb), "safe_length": sb["length"] if sb else None,
            "decision_shift": decision_shift,
            "netgain_drop": (ub["netgain"] - sb["netgain"]) if (sb and ub) else None,
        })
        per_sample.append({"trial_id": tid, "sample_path": str(path), **result})

    n_shifted = sum(1 for t in table if t["decision_shift"] == "shifted")
    n_unchanged = sum(1 for t in table if t["decision_shift"] == "unchanged")
    # Multi-step analysis.
    safe_lengths = [t["safe_length"] for t in table if t["safe_length"] is not None]
    unsafe_lengths = [t["unsafe_length"] for t in table if t["unsafe_length"] is not None]
    safe_multistep = sum(1 for l in safe_lengths if l >= 2)
    unsafe_multistep = sum(1 for l in unsafe_lengths if l >= 2)

    def _to_jsonable(o):
        if isinstance(o, dict): return {k: _to_jsonable(v) for k, v in o.items()}
        if isinstance(o, list): return [_to_jsonable(v) for v in o]
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    out = {
        "schema_version": "1.0.0", "record_type": "safe_sequence_oracle_synthetic_summary",
        "task_id": "safe_sequence_oracle_synthetic_severe_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "safe_sequence_oracle_synthetic_severe_v1",
            oracle_type="sequence",
            action_grid="5-level",
            stage="RL-0-safe-oracle",
            evidence_tier=CONTROLLED_DIAGNOSTIC,
            evaluators=evaluators,
            gate_evaluators=gate_evaluators,
        ),
        "timestamp": _utcnow(),
        "corruption": "foot_floating(0.08) + jitter(0.05)", "netgain_protocol": "A (clean GT reference)",
        "gate_thresholds_p99": gate_thresholds, "max_depth": args.max_depth,
        "n_samples": len(per_sample),
        "comparison_summary": {
            "n_shifted": n_shifted, "n_unchanged": n_unchanged,
            "safe_mean_length": float(np.mean(safe_lengths)) if safe_lengths else 0.0,
            "unsafe_mean_length": float(np.mean(unsafe_lengths)) if unsafe_lengths else 0.0,
            "safe_multistep_count": safe_multistep, "unsafe_multistep_count": unsafe_multistep,
        },
        "comparison_table": table, "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_to_jsonable(out), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== Synthetic Severe Safe Oracle Summary (n={len(per_sample)}) ===")
    print(f"  shifted: {n_shifted}, unchanged: {n_unchanged}")
    print(f"  safe mean length: {out['comparison_summary']['safe_mean_length']:.2f}, "
          f"unsafe mean length: {out['comparison_summary']['unsafe_mean_length']:.2f}")
    print(f"  safe multi-step (>=2): {safe_multistep}/{len(per_sample)}, "
          f"unsafe multi-step: {unsafe_multistep}/{len(per_sample)}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
