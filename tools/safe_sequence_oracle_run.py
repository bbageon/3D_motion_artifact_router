"""Step E (사용자 directive 2026-05-26): Safe Sequence Oracle — gate-aware DFS pruning.

기존 sequence oracle ([oracle_sequence_g2_5level_run.py]) 은 NetGain argmax 만 — 사용자
directive: "physical hard violation path prune, safe path 중 NetGain 최대 선택".

본 도구는 G2 natural motion 의 5-level (16 actions: STOP + 3 tool × 5 strength) DFS 의
each step 에서 PhysicalGateV0 (5 evaluator) 검사 — `hard_violation` path 는 prune.

Output snapshot 의 각 sample 의 두 결과:
  - **safe_best**: safe path 중 best NetGain (gate-aware).
  - **unsafe_best**: original DFS 의 best (gate 무관, 비교용 reference).

비교 표 의무 (사용자 directive 박제):

| Sample | Unsafe NetGain | Unsafe Violation | Safe NetGain | Safe Violation | Decision Shift |

CLI (top-4 G2 diagnostic):
    python -m tools.safe_sequence_oracle_run \
        --g2-batch-dir external_assets/g2_generated_v1 \
        --sample-filter motion_006,motion_007,motion_008,motion_028 \
        --task-id safe_sequence_oracle_g2_top4_v1 \
        --output evals/snapshots/safe_sequence_oracle_g2_top4_v1.json

CLI (G2 natural n=50 expansion, Step E-2):
    python -m tools.safe_sequence_oracle_run \
        --g2-batch-dir external_assets/g2_generated_v1 \
        --task-id safe_sequence_oracle_g2_natural_v1 \
        --output evals/snapshots/safe_sequence_oracle_g2_natural_v1.json

근거 (AGENTS.md §3-22): PhysDiff ICCV 2023 + MDM ICLR 2023 + HuMoR ICCV 2021 — physical
safety 를 별도 hard constraint (NOT reward weight) 으로 처리하는 정식 framework.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import (
    DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS, EvaluatorReport,
)
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
from tools.harness_metadata import REAL_DISTRIBUTION, common_snapshot_metadata

SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "safe_sequence_oracle_sample"
SUMMARY_TYPE = "safe_sequence_oracle_summary"

ALL_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
TOOLS_WITH_TARGET_PARTS: list[tuple[str, str]] = [
    ("FootLockTool", "both_feet"),
    ("BoneProjectionTool", "right_arm"),
    ("VelocitySmoothingTool", "full_body"),
]
STRENGTHS_5LEVEL = ("xsmall", "small5", "medium5", "large5", "xlarge")
STRENGTH_RANK_5 = {"xsmall": 0, "small5": 1, "medium5": 2, "large5": 3, "xlarge": 4}

DEFAULT_CALIBRATION_PATH = REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json"


def _utcnow_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _evaluate_all(motion: np.ndarray, evaluators: list) -> dict[str, list[EvaluatorReport]]:
    return {ev.name: ev.evaluate(motion) for ev in evaluators}


def _target_score_full(reports: dict[str, list[EvaluatorReport]]) -> float:
    return float(np.mean([_max_score(reports.get(n, [])) for n in ALL_EVALUATORS]))


def _gate_scores(motion: np.ndarray, gate_evaluators: list) -> dict[str, float]:
    """5 PhysicalGate evaluator 의 max score."""
    out = {}
    for ev in gate_evaluators:
        try:
            reports = ev.evaluate(motion)
            out[ev.name] = float(max((r.score for r in reports), default=0.0))
        except Exception:
            out[ev.name] = float("nan")
    return out


def _gate_violation(
    after: dict[str, float], before: dict[str, float], thresholds: dict[str, float],
    eps: float = 1e-5, jerk_relax: float = 1.05,
    skip_evaluators: tuple[str, ...] = ("FloatEvaluator",),
) -> dict[str, str]:
    """Per-evaluator decision: pass / soft_violation / hard_violation.

    Float evaluator skip option 으로 — clean p99=0.80 의 FP issue (부록 Z) — Step E 의
    primary gate 에서 Float 제외 권장. Item 6 contact estimator 후 재포함.
    """
    decisions = {}
    for name, after_score in after.items():
        if name in skip_evaluators:
            decisions[name] = "skip"
            continue
        before_score = before.get(name, 0.0)
        clean_p99 = thresholds.get(name, 0.0)
        relax_thresh = (before_score * jerk_relax if name == "JerkSpikeEvaluator"
                        else before_score + eps)
        unsafe_thresh = max(relax_thresh, clean_p99)
        if after_score > unsafe_thresh:
            decisions[name] = "hard_violation" if after_score > clean_p99 + eps else "soft_violation"
        else:
            decisions[name] = "pass"
    return decisions


def _strength_allowed(used: dict, tn: str, tp: str, st: str) -> bool:
    prev = used.get((tn, tp), [])
    if not prev:
        return True
    return STRENGTH_RANK_5[st] < min(STRENGTH_RANK_5[s] for s in prev)


def _eval_candidate(
    *, motion: np.ndarray, original_motion: np.ndarray, sequence: list,
    cum_corr_mag: float, target_initial_full: float,
    evaluators: list, netgain_weights: dict[str, float],
    gate_scores_initial: dict[str, float], gate_scores_after: dict[str, float],
    gate_decisions: dict[str, str],
) -> dict[str, Any]:
    reports = _evaluate_all(motion, evaluators)
    target_final_full = _target_score_full(reports)
    target_delta_full = target_final_full - target_initial_full
    fidelity_loss = _mpjpe(motion, original_motion)
    alpha = float(netgain_weights["alpha"])
    beta = float(netgain_weights["beta"])
    gamma = float(netgain_weights["gamma"])
    netgain = (
        -target_delta_full - alpha * fidelity_loss - beta * cum_corr_mag - gamma * len(sequence)
    )
    n_hard = sum(1 for d in gate_decisions.values() if d == "hard_violation")
    n_soft = sum(1 for d in gate_decisions.values() if d == "soft_violation")
    return {
        "sequence": [list(s) for s in sequence], "length": len(sequence),
        "target_delta_full": float(target_delta_full),
        "fidelity_loss_protocol_b": float(fidelity_loss),
        "cumulative_correction_magnitude": float(cum_corr_mag),
        "netgain": float(netgain),
        "gate_scores": gate_scores_after,
        "gate_decisions": gate_decisions,
        "n_hard_violations": n_hard, "n_soft_violations": n_soft,
        "any_hard_violation": n_hard > 0,
    }


def select_safe_sequence(
    *, original_motion: np.ndarray, tools_by_name: dict[str, CorrectionTool],
    evaluators: list, gate_evaluators: list, gate_thresholds: dict[str, float],
    netgain_weights: dict[str, float], max_depth: int = 3, top_k: int = 10,
    score_tol: float = 0.01,
) -> dict[str, Any]:
    """DFS — record all candidates (safe + unsafe) for comparison."""
    T = original_motion.shape[0]
    frame_range = (0, T - 1)
    reports_initial = _evaluate_all(original_motion, evaluators)
    target_initial_full = _target_score_full(reports_initial)
    gate_initial = _gate_scores(original_motion, gate_evaluators)
    total_initial = sum(_max_score(r) for r in reports_initial.values())
    actions = [(tn, tp, st) for tn, tp in TOOLS_WITH_TARGET_PARTS for st in STRENGTHS_5LEVEL]

    all_candidates: list[dict[str, Any]] = []
    counters = {"explored": 0, "pruned_strength": 0, "pruned_score": 0,
                "pruned_apply_error": 0, "pruned_gate_violation": 0}

    def _record(motion: np.ndarray, sequence: list, cum_corr_mag: float,
                gate_after: dict, decisions: dict) -> None:
        cand = _eval_candidate(
            motion=motion, original_motion=original_motion, sequence=sequence,
            cum_corr_mag=cum_corr_mag, target_initial_full=target_initial_full,
            evaluators=evaluators, netgain_weights=netgain_weights,
            gate_scores_initial=gate_initial, gate_scores_after=gate_after,
            gate_decisions=decisions,
        )
        all_candidates.append(cand)
        counters["explored"] += 1

    def _dfs(motion: np.ndarray, sequence: list, cum_corr_mag: float,
             prev_total: float, used: dict, parent_gate: dict) -> None:
        # current depth — record candidate (STOP equivalent).
        if not sequence:
            # depth 0 = original (no action) — record.
            initial_decisions = {n: "pass" for n in gate_initial}
            _record(motion, sequence, cum_corr_mag, gate_initial, initial_decisions)
        # 종료 조건 — max depth 도달.
        if len(sequence) >= max_depth:
            return
        for tn, tp, st in actions:
            if not _strength_allowed(used, tn, tp, st):
                counters["pruned_strength"] += 1
                continue
            tool = tools_by_name[tn]
            try:
                new_motion, report = tool.apply(
                    motion, target_part=tp, target_joints=[],
                    frame_range=frame_range, strength=st,
                )
            except ValueError:
                counters["pruned_apply_error"] += 1
                continue
            # Artifact score 비감소 prune (기존 oracle 의 동일 rule).
            new_reports = _evaluate_all(new_motion, evaluators)
            new_total = sum(_max_score(r) for r in new_reports.values())
            if new_total > prev_total + score_tol:
                counters["pruned_score"] += 1
                continue
            # Physical Gate 검사 (parent state 대비).
            gate_after = _gate_scores(new_motion, gate_evaluators)
            decisions = _gate_violation(gate_after, parent_gate, gate_thresholds)
            # record candidate (gate 결과 동봉).
            _record(new_motion, sequence + [(tn, tp, st)],
                    cum_corr_mag + float(report.correction_magnitude),
                    gate_after, decisions)
            # SAFE pruning: hard_violation 발생 시 본 branch 추가 explore 안 함.
            if any(d == "hard_violation" for d in decisions.values()):
                counters["pruned_gate_violation"] += 1
                continue
            new_used = dict(used)
            new_used[(tn, tp)] = list(used.get((tn, tp), [])) + [st]
            _dfs(new_motion, sequence + [(tn, tp, st)],
                 cum_corr_mag + float(report.correction_magnitude),
                 new_total, new_used, gate_after)

    _dfs(original_motion, [], 0.0, total_initial, {}, gate_initial)

    if not all_candidates:
        return {"safe_best": None, "unsafe_best": None, "counters": counters,
                "metadata": {"target_initial_full": target_initial_full,
                             "gate_scores_initial": gate_initial}}

    # Safe best — all candidates with no hard_violation.
    safe_candidates = [c for c in all_candidates if not c["any_hard_violation"]]
    safe_best = (max(safe_candidates, key=lambda c: (c["netgain"], -c["length"]))
                 if safe_candidates else None)
    # Unsafe best — all candidates ignoring gate (original oracle behavior).
    unsafe_best = max(all_candidates, key=lambda c: (c["netgain"], -c["length"]))
    # Top-k for both.
    safe_top_k = sorted(safe_candidates, key=lambda c: (c["netgain"], -c["length"]), reverse=True)[:top_k]
    unsafe_top_k = sorted(all_candidates, key=lambda c: (c["netgain"], -c["length"]), reverse=True)[:top_k]

    return {
        "safe_best": safe_best, "unsafe_best": unsafe_best,
        "safe_top_k": safe_top_k, "unsafe_top_k": unsafe_top_k,
        "counters": counters,
        "metadata": {
            "target_initial_full": target_initial_full,
            "gate_scores_initial": gate_initial,
            "n_candidates_total": len(all_candidates),
            "n_safe_candidates": len(safe_candidates),
        },
    }


def _load_g2_motions(batch_dir: Path, sample_filter: Optional[list[str]] = None
                     ) -> list[tuple[str, Path, dict, np.ndarray]]:
    out = []
    for npy in sorted(batch_dir.glob("motion_*.npy")):
        tid = npy.stem
        if sample_filter and tid not in sample_filter:
            continue
        meta_path = npy.with_suffix(".json")
        meta = json.load(open(meta_path, encoding="utf-8")) if meta_path.exists() else {}
        motion = np.load(str(npy)).astype(np.float64)
        out.append((tid, npy, meta, motion))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_v1")
    parser.add_argument("--sample-filter", type=str, default="",
                        help="Comma-separated trial_ids (e.g., motion_006,motion_007).")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load gate thresholds (p99 of clean calibration).
    with open(args.calibration, encoding="utf-8") as f:
        calib = json.load(f)
    gate_thresholds = {name: calib["summary"][name]["p99"]
                       for name in calib["summary"]
                       if calib["summary"][name].get("n", 0) > 0}
    print(f"[INFO] gate thresholds (p99): {gate_thresholds}")

    tools_by_name: dict[str, CorrectionTool] = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "BoneProjectionTool": BoneProjectionTool(),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
    }
    evaluators = list(DEFAULT_EVALUATORS)
    gate_evaluators = list(DEFAULT_PHYSICAL_GATE_EVALUATORS)
    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1

    sample_filter = [s.strip() for s in args.sample_filter.split(",") if s.strip()] if args.sample_filter else None
    samples = _load_g2_motions(args.g2_batch_dir, sample_filter)
    print(f"[INFO] {len(samples)} samples to process (filter={sample_filter})")

    per_sample = []
    for i, (tid, npy, meta, motion) in enumerate(samples, 1):
        prompt = meta.get("prompt", "")[:80]
        print(f"\n[{i}/{len(samples)}] {tid} (T={motion.shape[0]}): '{prompt}'")
        result = select_safe_sequence(
            original_motion=motion, tools_by_name=tools_by_name,
            evaluators=evaluators, gate_evaluators=gate_evaluators,
            gate_thresholds=gate_thresholds, netgain_weights=netgain_weights,
            max_depth=args.max_depth, top_k=args.top_k,
        )
        # Compact summary print.
        sb = result["safe_best"]; ub = result["unsafe_best"]
        if sb is None:
            print(f"   safe_best: NONE (no safe path found)")
        else:
            seq_sum = "STOP" if sb["length"] == 0 else " → ".join(
                f"{a[0].replace('Tool','')}/{a[2]}" for a in sb["sequence"])
            print(f"   safe_best: ng={sb['netgain']:+.4f}, len={sb['length']}, seq={seq_sum}")
        if ub is not None:
            seq_sum = "STOP" if ub["length"] == 0 else " → ".join(
                f"{a[0].replace('Tool','')}/{a[2]}" for a in ub["sequence"])
            n_viol = ub.get("n_hard_violations", 0)
            print(f"   unsafe_best: ng={ub['netgain']:+.4f}, len={ub['length']}, seq={seq_sum}, hard_viol={n_viol}")
        print(f"   counters: {result['counters']}")

        per_sample.append({
            "trial_id": tid, "sample_path": str(npy), "g2_prompt": meta.get("prompt", ""),
            **result,
        })

    # Aggregate comparison table (Decision Shift).
    table = []
    for s in per_sample:
        sb = s["safe_best"]; ub = s["unsafe_best"]
        if ub is None:
            continue
        unsafe_ng = ub["netgain"]
        unsafe_hard = ub["n_hard_violations"]
        safe_ng = sb["netgain"] if sb else None
        safe_hard = sb["n_hard_violations"] if sb else None
        seq_unsafe = "STOP" if ub["length"] == 0 else " → ".join(
            f"{a[0].replace('Tool','')}/{a[2]}" for a in ub["sequence"])
        seq_safe = ("STOP" if (sb and sb["length"] == 0) else
                    (" → ".join(f"{a[0].replace('Tool','')}/{a[2]}" for a in sb["sequence"]) if sb else "NONE"))
        decision_shift = (
            "unchanged" if (sb and ub and sb["sequence"] == ub["sequence"])
            else ("none_to_safe" if sb is None else "shifted")
        )
        ng_drop = (unsafe_ng - safe_ng) if safe_ng is not None else None
        table.append({
            "trial_id": s["trial_id"],
            "unsafe_netgain": unsafe_ng, "unsafe_hard_viol": unsafe_hard,
            "unsafe_sequence": seq_unsafe,
            "safe_netgain": safe_ng, "safe_hard_viol": safe_hard,
            "safe_sequence": seq_safe,
            "decision_shift": decision_shift, "netgain_drop": ng_drop,
        })

    n_total = len(table)
    n_shifted = sum(1 for t in table if t["decision_shift"] == "shifted")
    n_unchanged = sum(1 for t in table if t["decision_shift"] == "unchanged")

    # Convert numpy scalar types via deep recursion (for JSON serialization).
    def _to_jsonable(obj):
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_jsonable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out = {
        "schema_version": SCHEMA_VERSION,
        "record_type": SUMMARY_TYPE,
        "task_id": args.task_id,
        **common_snapshot_metadata(
            split_id=args.split_id or args.task_id,
            oracle_type="sequence",
            action_grid="5-level",
            stage="RL-0-safe-oracle",
            evidence_tier=REAL_DISTRIBUTION,
            evaluators=evaluators,
            gate_evaluators=gate_evaluators,
        ),
        "timestamp": _utcnow_stamp(),
        "calibration_source": str(args.calibration),
        "gate_thresholds_p99": gate_thresholds,
        "max_depth": args.max_depth,
        "netgain_weights": netgain_weights,
        "n_samples": len(per_sample),
        "comparison_table": table,
        "comparison_summary": {
            "n_total": n_total,
            "n_decision_shifted": n_shifted,
            "n_decision_unchanged": n_unchanged,
        },
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_to_jsonable(out), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== Safe Sequence Oracle Summary ===")
    print(f"  n_samples: {len(per_sample)}")
    print(f"  decision_shifted: {n_shifted}/{n_total}")
    print(f"  decision_unchanged: {n_unchanged}/{n_total}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
