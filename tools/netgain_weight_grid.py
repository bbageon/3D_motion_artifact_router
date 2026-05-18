"""NetGain weight (α/β/γ) grid search — Protocol A 기반 calibration.

명세 §9.4 의 NetGain:
    NetGain = ArtifactReduction − α·FidelityLoss − β·CorrectionMagnitude − γ·ToolCallCost

AGENTS.md §6-11: 본 weight 의 grid search 기준은 명세상 "perceptual rating
상관 최대화". 본 단계에서는 perceptual rating 이 없으므로 **Protocol A
FidelityLoss 의 음수 (= MPJPE refined 대비 corrupted 의 개선량)** 를 quality
proxy 로 사용. 결과 status tag = `"calibrated_protocol_a_v1"` (provisional 보다
한 단계 위, perceptual 보다는 아래).

본 도구는 [`oracle_single_step_v1`](../evals/raw) 의 raw record 의 모든
candidate (30 × 3 × 9 = 810) 의 (target_delta, fidelity_loss_protocol_a,
correction_magnitude) 를 입력으로 받아 (α, β, γ) grid 별로 NetGain 을
재계산하고 quality proxy Q = -fidelity_loss 와의 **Spearman rank
correlation** 을 측정한다.

산출:
  - 전체 candidate (810) 의 best (α, β, γ).
  - artifact_kind 별 best (3 group).
  - 본 grid 의 모든 (α, β, γ) × correlation 결과 (snapshot).

CLI 예:
    python -m tools.netgain_weight_grid \\
        --raw-records-dir evals/raw \\
        --task-id-prefix oracle_single_step_v1 \\
        --output evals/snapshots/netgain_weight_grid_v1.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "evals" / "raw"

#: Quality proxy 정의. AGENTS.md §6-11 의 calibration source 명시.
#: Q = -fidelity_loss_protocol_a (작은 fidelity_loss = 더 큰 Q = better quality).
QUALITY_PROXY = "neg_fidelity_loss_protocol_a"

#: Grid space. 6 × 5 × 4 = 120 조합.
DEFAULT_GRID = {
    "alpha": [0.0, 0.25, 0.5, 1.0, 2.0, 5.0],
    "beta": [0.0, 0.25, 0.5, 1.0, 2.0],
    "gamma": [0.0, 0.1, 0.5, 1.0],
}


def _load_candidates(
    raw_dir: Path, task_id_prefix: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """oracle_single_step raw record 들에서 모든 candidate 추출.

    Returns:
        (candidates, count_by_artifact).
    """
    candidates: list[dict[str, Any]] = []
    count_by_artifact: dict[str, int] = defaultdict(int)
    for path in sorted(raw_dir.glob(f"*{task_id_prefix}*.json")):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if rec.get("record_type") != "oracle_single_step_sample":
            continue
        trial_id = rec["trial_id"]
        for artifact_kind, sel in rec["selections"].items():
            for cand in sel.get("candidates", []):
                if cand.get("skipped", False):
                    continue
                candidates.append({
                    "trial_id": trial_id,
                    "artifact_kind": artifact_kind,
                    "tool_name": cand["tool_name"],
                    "strength": cand["strength"],
                    "target_delta": float(cand["target_delta"]),
                    "fidelity_loss_protocol_a": float(cand["fidelity_loss_protocol_a"]),
                    "correction_magnitude": float(cand["correction_magnitude"]),
                })
                count_by_artifact[artifact_kind] += 1
    return candidates, dict(count_by_artifact)


def _netgain(c: dict[str, Any], alpha: float, beta: float, gamma: float) -> float:
    """명세 §9.4 NetGain. ToolCallCost = 1 (single-step)."""
    artifact_reduction = -c["target_delta"]
    fidelity_loss = c["fidelity_loss_protocol_a"]
    correction_mag = c["correction_magnitude"]
    return artifact_reduction - alpha * fidelity_loss - beta * correction_mag - gamma * 1.0


def _quality_proxy(c: dict[str, Any]) -> float:
    """Q = -fidelity_loss_protocol_a (작을수록 fidelity 좋음 → 더 큰 quality)."""
    return -c["fidelity_loss_protocol_a"]


def grid_search(
    candidates: list[dict[str, Any]],
    grid: dict[str, list[float]],
) -> dict[str, Any]:
    """Grid 별 Spearman correlation 계산.

    전체 candidate 와 artifact_kind 별 group 모두에 대해.
    """
    quality_all = np.array([_quality_proxy(c) for c in candidates], dtype=np.float64)
    by_art: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(candidates):
        by_art[c["artifact_kind"]].append(i)

    results: list[dict[str, Any]] = []
    best_global: dict[str, Any] = {"correlation": -float("inf")}
    best_by_artifact: dict[str, dict[str, Any]] = {}

    for alpha, beta, gamma in itertools.product(grid["alpha"], grid["beta"], grid["gamma"]):
        netgains = np.array([_netgain(c, alpha, beta, gamma) for c in candidates], dtype=np.float64)
        # Spearman over all candidates.
        rho_all, _ = spearmanr(netgains, quality_all)
        if rho_all is None or np.isnan(rho_all):
            rho_all = 0.0

        # Per artifact.
        rho_by_art: dict[str, float] = {}
        for art, idxs in by_art.items():
            if len(idxs) < 2:
                rho_by_art[art] = 0.0
                continue
            rho, _ = spearmanr(netgains[idxs], quality_all[idxs])
            rho_by_art[art] = float(rho) if rho is not None and not np.isnan(rho) else 0.0

        result = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "spearman_correlation_all": float(rho_all),
            "spearman_correlation_by_artifact": rho_by_art,
        }
        results.append(result)

        if rho_all > best_global["correlation"]:
            best_global = {
                "correlation": float(rho_all),
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            }
        for art, rho in rho_by_art.items():
            cur = best_by_artifact.get(art, {"correlation": -float("inf")})
            if rho > cur["correlation"]:
                best_by_artifact[art] = {
                    "correlation": float(rho),
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                }

    return {
        "n_candidates_total": len(candidates),
        "n_candidates_by_artifact": {a: len(v) for a, v in by_art.items()},
        "quality_proxy": QUALITY_PROXY,
        "grid": grid,
        "n_grid_combinations": len(results),
        "best_global": best_global,
        "best_by_artifact": best_by_artifact,
        "all_results": results,
        "netgain_weight_status_tag": "calibrated_protocol_a_v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NetGain weight grid search (Protocol A)")
    parser.add_argument("--raw-records-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--task-id-prefix", type=str, default="oracle_single_step_v1")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    candidates, count_by_art = _load_candidates(args.raw_records_dir, args.task_id_prefix)
    if not candidates:
        raise FileNotFoundError(
            f"no oracle_single_step candidates found in {args.raw_records_dir} "
            f"with task_id prefix {args.task_id_prefix!r}"
        )

    summary = grid_search(candidates, DEFAULT_GRID)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OK] wrote grid search summary to {args.output}")
        print(f"  candidates: {summary['n_candidates_total']} ({summary['n_candidates_by_artifact']})")
        print(f"  best global: alpha={summary['best_global']['alpha']}, beta={summary['best_global']['beta']}, gamma={summary['best_global']['gamma']}, rho={summary['best_global']['correlation']:.4f}")
        print(f"  status tag: {summary['netgain_weight_status_tag']}")
    else:
        print(text)


if __name__ == "__main__":
    main()
