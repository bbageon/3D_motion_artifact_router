"""Per-sample gap (sequence - single_step) 의 α-sensitivity 분석.

각 (trial, artifact) 에서 sequence / single_step 의 모든 candidate 을 load 하여
주어진 α 에서 NetGain 을 재계산, 각 oracle 의 best 를 다시 고른 후 per-sample gap
을 측정한다. H-2026-204 RQ2 (closed-loop ≥ single-step) 가 calibration α 변화에
얼마나 robust 한지 검증.

CLI 예:
    python -m tools.per_sample_gap_alpha_sensitivity \\
        --single-step-prefix oracle_single_step_v2 \\
        --sequence-prefix oracle_sequence_v1_1_tiebreak \\
        --alphas 1 5 10 50 \\
        --output evals/snapshots/per_sample_gap_alpha_sensitivity_v1.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "evals" / "raw"


def _netgain_from_cand(c: dict[str, Any], alpha: float, beta: float = 0.0, gamma: float = 0.0) -> float:
    """명세 §9.4 NetGain. tool_call_cost = sequence length (sequence) 또는 1 (single_step)."""
    artifact_reduction = -float(c["target_delta"])
    fidelity_loss = float(c.get("fidelity_loss_protocol_a", 0.0))
    correction_mag = float(c.get("correction_magnitude", 0.0))
    tool_call_cost = float(c.get("length", 1))  # sequence 는 length, single_step 은 1.
    return artifact_reduction - alpha * fidelity_loss - beta * correction_mag - gamma * tool_call_cost


def _load_candidates_single_step(raw_dir: Path, prefix: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for p in sorted(raw_dir.glob(f"*{prefix}*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("record_type") != "oracle_single_step_sample":
            continue
        trial = r["trial_id"]
        for art, sel in r.get("selections", {}).items():
            for c in sel.get("candidates", []):
                if c.get("skipped"):
                    continue
                out[(trial, art)].append({
                    **c,
                    "length": 1,  # single_step 은 length=1.
                })
    return out


def _load_candidates_sequence(raw_dir: Path, prefix: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for p in sorted(raw_dir.glob(f"*{prefix}*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("record_type") != "oracle_sequence_sample":
            continue
        trial = r["trial_id"]
        for art, sel in r.get("selections", {}).items():
            # sequence raw 는 top_k_candidates 만 가지고 있을 수 있음 — 본 분석은
            # top_k 안에 final argmax 가 포함된다는 가정. (DFS 가 score-비감소 pruning 후
            # top_k 를 NetGain desc 로 keep 하므로 v1 weight 기준 top_k = highest-NetGain
            # 후보들. 다른 α 에서 best 가 top_k 밖일 가능성은 작지만 0 아님.)
            for c in sel.get("top_k_candidates", []):
                out[(trial, art)].append({
                    "target_delta": c["target_delta_total"],
                    "fidelity_loss_protocol_a": c.get("fidelity_loss_protocol_a", 0.0),
                    "correction_magnitude": c.get("cumulative_correction_magnitude", 0.0),
                    "length": c.get("length", 1),
                })
    return out


def compute_at_alpha(
    ss_cands: dict[tuple[str, str], list[dict[str, Any]]],
    seq_cands: dict[tuple[str, str], list[dict[str, Any]]],
    alpha: float,
) -> dict[str, Any]:
    common_keys = sorted(set(ss_cands) & set(seq_cands))
    gaps_by_art: dict[str, list[float]] = defaultdict(list)
    for k in common_keys:
        ss_best = max(_netgain_from_cand(c, alpha) for c in ss_cands[k])
        seq_best = max(_netgain_from_cand(c, alpha) for c in seq_cands[k])
        gaps_by_art[k[1]].append(seq_best - ss_best)

    per_artifact = []
    for art, gaps in sorted(gaps_by_art.items()):
        arr = np.array(gaps, dtype=np.float64)
        per_artifact.append({
            "artifact_kind": art,
            "n_samples": int(arr.size),
            "n_strict_advantage": int((arr > 1e-12).sum()),
            "n_tie": int((arr <= 1e-12).sum() & (arr >= -1e-12).sum()),
            "n_loss": int((arr < -1e-12).sum()),
            "gap": {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
            },
        })
    return {"alpha": alpha, "n_keys": len(common_keys), "per_artifact": per_artifact}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-sample gap (sequence - single_step) at multiple α values"
    )
    parser.add_argument("--raw-records-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--single-step-prefix", type=str, default="oracle_single_step_v2")
    parser.add_argument("--sequence-prefix", type=str, default="oracle_sequence_v1_1_tiebreak")
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 5.0, 10.0, 20.0, 50.0])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ss = _load_candidates_single_step(args.raw_records_dir, args.single_step_prefix)
    seq = _load_candidates_sequence(args.raw_records_dir, args.sequence_prefix)
    n_ss_keys = len(ss)
    n_seq_keys = len(seq)
    print(f"[INFO] single_step (trial, artifact) keys: {n_ss_keys}")
    print(f"[INFO] sequence (trial, artifact) keys: {n_seq_keys}")

    per_alpha = [compute_at_alpha(ss, seq, a) for a in args.alphas]
    summary = {
        "schema_version": "1.0.0",
        "record_type": "per_sample_gap_alpha_sensitivity",
        "single_step_prefix": args.single_step_prefix,
        "sequence_prefix": args.sequence_prefix,
        "n_alphas": len(args.alphas),
        "alphas": args.alphas,
        "n_single_step_keys": n_ss_keys,
        "n_sequence_keys": n_seq_keys,
        "per_alpha": per_alpha,
        "interpretation": {
            "purpose": (
                "H-2026-204 RQ2 (closed-loop ≥ single-step) 의 strict advantage 비율이 "
                "calibration α 변화 (5.0 v1 → 50.0 v2 grid) 에도 robust 한지 검증."
            ),
            "mathematical_invariant": (
                "Subset inclusion (single_step candidates ⊆ sequence candidates) 으로 gap ≥ 0 은 "
                "모든 α 에서 성립. 단 argmax candidate identity 는 α 에 따라 변할 수 있어 "
                "n_strict_advantage / n_tie 분포는 α-dependent."
            ),
            "caveat": (
                "sequence raw 는 top_k_candidates (v1 weight 기준) 만 저장 → 매우 큰 α 에서는 "
                "true best 가 top_k 밖일 가능성 (특히 fidelity-dominant 영역). 정확한 sensitivity 는 "
                "full re-run 필요. 본 분석은 first-order 근사."
            ),
        },
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OK] wrote to {args.output}")
        for pa in per_alpha:
            print(f"\nα = {pa['alpha']}")
            for ar in pa["per_artifact"]:
                print(
                    f"  {ar['artifact_kind']:30s} | "
                    f"strict={ar['n_strict_advantage']}, tie≈{ar['n_samples'] - ar['n_strict_advantage'] - ar['n_loss']}, loss={ar['n_loss']} | "
                    f"median={ar['gap']['median']:+.5f}, max={ar['gap']['max']:+.5f}"
                )
    else:
        print(text)


if __name__ == "__main__":
    main()
