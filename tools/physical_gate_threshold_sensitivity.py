"""Step E-2.6 (사용자 directive 2026-05-27): BoneLengthCV threshold sensitivity ablation.

사용자 directive:
> "지금 가장 큰 불확실성이 BoneLengthCV p99=5.5e-6 가 너무 tight한가? 이기 때문.
>  threshold variants: clean p99 / clean p99 + eps / relative 1% / 2% / 5%.
>  봐야 할 것: violation rate 가 threshold 에 따라 얼마나 변하는가? FootLock-related
>  pattern 이 threshold 완화 후에도 유지되는가? top violation sample 의 absolute bone
>  length 변화가 실제로 큰가?"

본 도구는 n=300 audit snapshot 의 unsafe_best 의 BoneLengthCV (after) vs
gate_scores_initial (before) 를 다양한 threshold 정의 하에서 재분류 — violation rate 의
threshold sensitivity 정량 + FootLock pattern 유지 확인 + 절대 bone length 변화 측정.

Threshold variants (BoneLengthCV 의 hard_violation 정의):
  - clean_p99       : after > clean_p99 (현재, 5.5e-6)
  - before_plus_eps : after > before + 1e-5
  - relative_1pct   : after > before * 1.01
  - relative_2pct   : after > before * 1.02
  - relative_5pct   : after > before * 1.05
  - relative_10pct  : after > before * 1.10
  - abs_delta_0.005 : after - before > 0.005
  - abs_delta_0.01  : after - before > 0.01
  - abs_delta_0.02  : after - before > 0.02

CLI:
    python -m tools.physical_gate_threshold_sensitivity \
        --snapshot evals/snapshots/safe_sequence_oracle_g2_hml3d_test300_v1.json \
        --output evals/snapshots/physical_gate_threshold_sensitivity_v1.json

근거 (AGENTS.md §3-22): HuMoR (Rempe et al. 2021, ICCV) bone length consistency.
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

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from skeleton_normalizer.canonical_smpl_22 import T2M_KINEMATIC_CHAIN

EVALUATOR_KEY = "BoneLengthCVEvaluator"
CLEAN_P99 = 5.506674955074729e-06  # from physical_gate_clean_calibration_v1.json

TOOL_BY_NAME: dict[str, CorrectionTool] = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}

# Bone pairs from kinematic chain.
BONES: list[tuple[int, int]] = []
for chain in T2M_KINEMATIC_CHAIN:
    for a, b in zip(chain[:-1], chain[1:]):
        BONES.append((a, b))
BONES = list(dict.fromkeys(BONES))


def _threshold_variants(before: float) -> dict[str, float]:
    """각 threshold mode 의 unsafe boundary (after > boundary → violation)."""
    return {
        "clean_p99": CLEAN_P99,
        "before_plus_eps": before + 1e-5,
        "relative_1pct": before * 1.01,
        "relative_2pct": before * 1.02,
        "relative_5pct": before * 1.05,
        "relative_10pct": before * 1.10,
        "abs_delta_0.005": before + 0.005,
        "abs_delta_0.01": before + 0.01,
        "abs_delta_0.02": before + 0.02,
    }


def _apply_sequence(motion: np.ndarray, sequence: list) -> np.ndarray:
    T = motion.shape[0]
    out = motion.copy()
    for step in sequence:
        if step[0] in ("STOP", "APPLY_FAIL", "SCORE_VIOLATION_ROLLBACK"):
            break
        tool = TOOL_BY_NAME[step[0]]
        out, _ = tool.apply(out, target_part=step[1], target_joints=[],
                             frame_range=(0, T - 1), strength=step[2])
    return out


def _abs_bone_lengths(motion: np.ndarray) -> np.ndarray:
    """Per-bone mean length (meters) across frames. shape (n_bones,)."""
    lengths = []
    for a, b in BONES:
        d = np.linalg.norm(motion[:, a, :] - motion[:, b, :], axis=-1)  # [T]
        lengths.append(float(np.mean(d)))
    return np.array(lengths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_g2_hml3d_test300_v1.json")
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_threshold_sensitivity_v1.json")
    parser.add_argument("--abs-top-n", type=int, default=10,
                        help="Top-N unsafe-NetGain violators for absolute bone length analysis.")
    args = parser.parse_args()

    with open(args.snapshot, encoding="utf-8") as f:
        snap = json.load(f)
    per_sample = snap["per_sample"]
    n_total = len(per_sample)
    print(f"[INFO] n_samples = {n_total}")

    threshold_modes = list(_threshold_variants(0.0).keys())

    # === Part 1: violation rate sensitivity (unsafe_best re-classification) ===
    # For each sample's unsafe_best, re-classify BoneLengthCV violation under each threshold.
    viol_count_by_mode: dict[str, int] = {m: 0 for m in threshold_modes}
    footlock_count_by_mode: dict[str, int] = {m: 0 for m in threshold_modes}
    per_sample_classifications = []

    for p in per_sample:
        ub = p.get("unsafe_best")
        if ub is None or ub["length"] == 0:
            continue  # STOP unsafe_best — no correction applied, no bone violation.
        after = ub["gate_scores"].get(EVALUATOR_KEY, 0.0)
        before = p["metadata"]["gate_scores_initial"].get(EVALUATOR_KEY, 0.0)
        seq = ub["sequence"]
        first_tool = seq[0][0] if seq else None
        is_footlock = first_tool == "FootLockTool"
        variants = _threshold_variants(before)
        sample_class = {"trial_id": p["trial_id"], "before": before, "after": after,
                        "delta": after - before, "first_tool": first_tool,
                        "unsafe_netgain": ub["netgain"], "flagged": {}}
        for mode, boundary in variants.items():
            flagged = after > boundary
            sample_class["flagged"][mode] = bool(flagged)
            if flagged:
                viol_count_by_mode[mode] += 1
                if is_footlock:
                    footlock_count_by_mode[mode] += 1
        per_sample_classifications.append(sample_class)

    print("\n=== Part 1: BoneLengthCV violation rate sensitivity (unsafe_best) ===")
    print(f"{'threshold_mode':<20} {'viol_count':<12} {'rate':<10} {'footlock':<12} {'footlock_pct'}")
    part1_summary = {}
    for mode in threshold_modes:
        vc = viol_count_by_mode[mode]
        fc = footlock_count_by_mode[mode]
        rate = vc / n_total
        fpct = fc / vc if vc > 0 else 0.0
        print(f"{mode:<20} {vc:<12} {rate*100:<9.1f}% {fc:<12} {fpct*100:.0f}%")
        part1_summary[mode] = {
            "violation_count": vc, "violation_rate": rate,
            "footlock_count": fc, "footlock_pct": fpct,
        }

    # === Part 2: absolute bone length change for top-N unsafe violators ===
    print(f"\n=== Part 2: Absolute bone length change (top-{args.abs_top_n} unsafe NetGain violators) ===")
    # Top-N by unsafe NetGain among samples flagged under clean_p99 (the strictest, = current).
    flagged_clean = [c for c in per_sample_classifications if c["flagged"]["clean_p99"]]
    top_violators = sorted(flagged_clean, key=lambda c: -c["unsafe_netgain"])[: args.abs_top_n]

    part2_results = []
    print(f"{'trial_id':<14} {'CV_before':<10} {'CV_after':<10} {'max_bone_change_m':<18} {'max_bone_pct'}")
    for c in top_violators:
        tid = c["trial_id"]
        npy = args.g2_batch_dir / f"{tid}.npy"
        if not npy.exists():
            print(f"{tid}: missing npy")
            continue
        motion = np.load(str(npy)).astype(np.float64)
        # Find the sample's unsafe sequence.
        ps = next(p for p in per_sample if p["trial_id"] == tid)
        seq = ps["unsafe_best"]["sequence"]
        corrected = _apply_sequence(motion, seq)
        bl_before = _abs_bone_lengths(motion)
        bl_after = _abs_bone_lengths(corrected)
        abs_change = np.abs(bl_after - bl_before)  # meters
        pct_change = abs_change / (bl_before + 1e-9)
        max_change_idx = int(np.argmax(abs_change))
        max_change_m = float(abs_change[max_change_idx])
        max_pct = float(pct_change[max_change_idx])
        bone = BONES[max_change_idx]
        print(f"{tid:<14} {c['before']:<10.5f} {c['after']:<10.5f} {max_change_m:<18.5f} {max_pct*100:.1f}%")
        part2_results.append({
            "trial_id": tid, "cv_before": c["before"], "cv_after": c["after"],
            "unsafe_netgain": c["unsafe_netgain"], "unsafe_sequence": seq,
            "max_bone_change_m": max_change_m, "max_bone_change_pct": max_pct,
            "max_change_bone_joints": list(bone),
            "mean_bone_change_m": float(np.mean(abs_change)),
            "mean_bone_change_pct": float(np.mean(pct_change)),
        })

    # === Output ===
    out = {
        "schema_version": "1.0.0",
        "record_type": "physical_gate_threshold_sensitivity",
        "task_id": "physical_gate_threshold_sensitivity_v1",
        "source_snapshot": str(args.snapshot),
        "n_samples": n_total,
        "evaluator": EVALUATOR_KEY,
        "clean_p99": CLEAN_P99,
        "threshold_modes": threshold_modes,
        "part1_violation_rate_sensitivity": part1_summary,
        "part2_absolute_bone_length_change": part2_results,
        "per_sample_classifications": per_sample_classifications,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
