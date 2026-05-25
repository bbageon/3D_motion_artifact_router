"""Step 4: G2 general-prompt pilot 의 evaluator + 보조 metric 측정 + 방법별 비교.

사용자 9-step plan Step 4:
> "evaluator + 보조 metric 측정 — current evaluator scores / jerk / acceleration
>  / foot sliding / correction magnitude / MPJPE-to-original. 방법 별 비교:
>  Original G2 / B2-small/medium/large/val-best / B5 / RL-1 HGB / sequence oracle."

목적: 일반 동작 prompt 의 NetGain rank 가 시각/perceptual rank 와 일치 하는지
확인 (NetGain validity 검증).

본 도구는 g2_general_pilot_v1/ 의 10 motion 각각에:
  - 3 evaluator (FootFloating, BoneLength, VelocityJitter) max score.
  - 보조 metric: per-joint mean acceleration, mean jerk, foot displacement.
  - 방법별 적용 결과 (target_delta, fidelity_loss, NetGain).

방법 목록:
  - original (do-nothing, baseline reference).
  - B2-small / B2-medium / B2-large / B2-val-best (per-sample best strength).
  - sequence oracle (DFS K=5).
  - (B5/HGB 는 별도 도구 의 closed-loop rollout 필요 — 본 도구는 fixed action
    + oracle 측정 만).

CLI:
    python -m tools.g2_general_pilot_measure \\
        --pilot-dir external_assets/g2_general_pilot_v1 \\
        --output evals/snapshots/g2_general_pilot_v1.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from correction_tools import BoneProjectionTool, CorrectionTool, FootLockTool, VelocitySmoothingTool
from evaluators import DEFAULT_EVALUATORS, EvaluatorReport
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
STRENGTHS = ("small", "medium", "large")
STRENGTH_RANK = {"small": 0, "medium": 1, "large": 2}

LEFT_FOOT = 10
RIGHT_FOOT = 11


def _max_score(reports: list[EvaluatorReport]) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _eval_scores(motion: np.ndarray, evaluators: list[Any]) -> dict[str, float]:
    return {n: _max_score(ev.evaluate(motion)) for n, ev in zip(ALL_EVALUATORS, evaluators)}


def _target_score_full(scores: dict[str, float]) -> float:
    return float(np.mean([scores[n] for n in ALL_EVALUATORS]))


def _mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _accel_norm_mean(motion: np.ndarray) -> float:
    if motion.shape[0] < 3:
        return 0.0
    vel = np.diff(motion, axis=0)
    acc = np.diff(vel, axis=0)
    return float(np.linalg.norm(acc, axis=-1).mean())


def _jerk_norm_mean(motion: np.ndarray) -> float:
    if motion.shape[0] < 4:
        return 0.0
    vel = np.diff(motion, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    return float(np.linalg.norm(jerk, axis=-1).mean())


def _foot_sliding_proxy(motion: np.ndarray, contact_y_thresh: float = 0.05) -> float:
    """Foot sliding proxy: foot Y < threshold 인 frame 의 horizontal velocity 평균."""
    if motion.shape[0] < 2:
        return 0.0
    feet = motion[:, [LEFT_FOOT, RIGHT_FOOT], :]  # [T, 2, 3]
    contact = feet[:, :, 1] < contact_y_thresh  # [T, 2]
    horiz = np.diff(feet[:, :, [0, 2]], axis=0)  # [T-1, 2, 2]
    horiz_norm = np.linalg.norm(horiz, axis=-1)  # [T-1, 2]
    contact_t = contact[:-1] & contact[1:]  # [T-1, 2]
    sliding = horiz_norm[contact_t]
    return float(sliding.mean()) if sliding.size > 0 else 0.0


def _apply_b2(motion: np.ndarray, strength: str, vs: VelocitySmoothingTool) -> np.ndarray:
    T = motion.shape[0]
    out, _ = vs.apply(motion, target_part="full_body", target_joints=[],
                       frame_range=(0, T - 1), strength=strength)
    return out


def _compute_netgain_g2(
    *, original: np.ndarray, refined: np.ndarray, evaluators: list[Any], alpha: float,
) -> dict[str, float]:
    """Protocol B simplified: target=mean(all 3), reference=original."""
    scores_init = _eval_scores(original, evaluators)
    scores_final = _eval_scores(refined, evaluators)
    target_init = _target_score_full(scores_init)
    target_final = _target_score_full(scores_final)
    target_delta = target_final - target_init
    fidelity_loss = _mpjpe(refined, original)
    return {
        "scores_final": scores_final,
        "target_delta": target_delta,
        "fidelity_loss": fidelity_loss,
        "netgain": -target_delta - alpha * fidelity_loss,
    }


def _select_b2_val_best(per_strength: dict[str, dict[str, float]]) -> dict[str, Any]:
    best_strength = max(STRENGTHS, key=lambda s: per_strength[s]["netgain"])
    return {**per_strength[best_strength], "best_strength": best_strength}


def _dfs_sequence_oracle_g2(
    *, motion: np.ndarray, tools_by_name: dict[str, CorrectionTool],
    evaluators: list[Any], alpha: float, max_depth: int = 5,
    score_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Lightweight sequence oracle DFS for G2 (Protocol B, target=mean(all 3))."""
    T = motion.shape[0]
    frame_range = (0, T - 1)
    actions = [
        ("FootLockTool", "both_feet", st) for st in STRENGTHS
    ] + [("BoneProjectionTool", "right_arm", st) for st in STRENGTHS] + [
        ("VelocitySmoothingTool", "full_body", st) for st in STRENGTHS
    ]
    scores_init = _eval_scores(motion, evaluators)
    target_init = _target_score_full(scores_init)
    total_init = sum(scores_init.values())

    all_candidates = []

    def _eval_cand(m: np.ndarray, seq: list[tuple[str, str, str]]) -> None:
        s_final = _eval_scores(m, evaluators)
        t_final = _target_score_full(s_final)
        td = t_final - target_init
        fl = _mpjpe(m, motion)
        ng = -td - alpha * fl
        all_candidates.append({"sequence": [list(x) for x in seq], "length": len(seq),
                                "target_delta": td, "fidelity_loss": fl, "netgain": ng,
                                "scores_final": s_final})

    def _strength_allowed(used: dict, tn: str, tp: str, st: str) -> bool:
        prev = used.get((tn, tp), [])
        if not prev:
            return True
        return STRENGTH_RANK[st] < min(STRENGTH_RANK[s] for s in prev)

    def _dfs(m: np.ndarray, seq: list, prev_total: float, used: dict) -> None:
        _eval_cand(m, seq)
        if len(seq) >= max_depth:
            return
        for tn, tp, st in actions:
            if not _strength_allowed(used, tn, tp, st):
                continue
            tool = tools_by_name[tn]
            try:
                new_m, _ = tool.apply(m, target_part=tp, target_joints=[],
                                       frame_range=frame_range, strength=st)
            except ValueError:
                continue
            new_scores = _eval_scores(new_m, evaluators)
            new_total = sum(new_scores.values())
            if new_total > prev_total + score_tolerance:
                continue
            new_used = dict(used); new_used[(tn, tp)] = list(used.get((tn, tp), [])) + [st]
            _dfs(new_m, seq + [(tn, tp, st)], new_total, new_used)

    _dfs(motion, [], total_init, {})
    if not all_candidates:
        return {"sequence": [], "length": 0, "netgain": 0.0, "scores_final": scores_init}
    best = max(all_candidates, key=lambda c: (c["netgain"], -c["length"]))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 general-prompt pilot measurement (Step 4)")
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()

    netgain_weights = CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1
    alpha = float(netgain_weights["alpha"])
    vs = VelocitySmoothingTool()
    tools_by_name: dict[str, CorrectionTool] = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "BoneProjectionTool": BoneProjectionTool(),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
    }
    evaluators = list(DEFAULT_EVALUATORS)

    motion_files = sorted(args.pilot_dir.glob("motion_*.npy"))
    print(f"[INFO] G2 pilot: {len(motion_files)} motions")

    per_sample = []
    for i, npy_path in enumerate(motion_files, 1):
        meta_path = npy_path.with_suffix(".json")
        if not meta_path.exists():
            print(f"  [WARN] no meta: {npy_path.name}")
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        motion = np.load(str(npy_path)).astype(np.float64)
        if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
            print(f"  [WARN] shape mismatch: {motion.shape}")
            continue
        trial_id = meta.get("trial_id", npy_path.stem)
        prompt = meta.get("prompt", "")[:80]

        # Original metrics.
        scores_orig = _eval_scores(motion, evaluators)
        original_metrics = {
            "scores": scores_orig,
            "mean_acceleration_norm": _accel_norm_mean(motion),
            "mean_jerk_norm": _jerk_norm_mean(motion),
            "foot_sliding_proxy": _foot_sliding_proxy(motion),
            "motion_shape": list(motion.shape),
        }

        # B2-family.
        b2_per_strength = {}
        for st in STRENGTHS:
            refined = _apply_b2(motion, st, vs)
            ng = _compute_netgain_g2(original=motion, refined=refined, evaluators=evaluators, alpha=alpha)
            ng["mean_acc"] = _accel_norm_mean(refined)
            ng["mean_jerk"] = _jerk_norm_mean(refined)
            ng["foot_sliding"] = _foot_sliding_proxy(refined)
            b2_per_strength[st] = ng
        b2_val_best = _select_b2_val_best(b2_per_strength)

        # Sequence oracle.
        seq_oracle = _dfs_sequence_oracle_g2(
            motion=motion, tools_by_name=tools_by_name, evaluators=evaluators,
            alpha=alpha, max_depth=args.max_depth,
        )

        per_sample.append({
            "trial_id": trial_id,
            "prompt": prompt,
            "original": original_metrics,
            "B2-small": b2_per_strength["small"],
            "B2-medium": b2_per_strength["medium"],
            "B2-large": b2_per_strength["large"],
            "B2-val-best": b2_val_best,
            "sequence_oracle": seq_oracle,
        })
        ob = original_metrics
        print(f"[{i:2d}/{len(motion_files)}] {trial_id} '{prompt[:40]}...': orig scores FF={ob['scores']['FootFloatingEvaluator']:.4f} BL={ob['scores']['BoneLengthEvaluator']:.4f} VJ={ob['scores']['VelocityJitterEvaluator']:.4f}; B2-val-best={b2_val_best['best_strength']} NG={b2_val_best['netgain']:+.4f}; oracle len={seq_oracle['length']} NG={seq_oracle['netgain']:+.4f}")

    # Aggregate.
    def _agg(field: str, sub: str = "netgain") -> dict[str, float]:
        vals = [s[field][sub] for s in per_sample]
        arr = np.array(vals)
        return {"mean": float(arr.mean()), "median": float(np.median(arr)),
                "min": float(arr.min()), "max": float(arr.max())}

    summary = {
        "schema_version": "1.0.0",
        "record_type": "g2_general_pilot_v1",
        "n_samples": len(per_sample),
        "netgain_weight_status": "calibrated_protocol_a_v1",
        "alpha": alpha,
        "aggregate": {
            "B2-small": _agg("B2-small"),
            "B2-medium": _agg("B2-medium"),
            "B2-large": _agg("B2-large"),
            "B2-val-best": _agg("B2-val-best"),
            "sequence_oracle": _agg("sequence_oracle"),
        },
        "best_strength_distribution_val_best": dict(Counter(s["B2-val-best"]["best_strength"] for s in per_sample)),
        "best_length_distribution_oracle": dict(Counter(s["sequence_oracle"]["length"] for s in per_sample)),
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {args.output}")

    # Summary table.
    print(f"\n=== G2 general pilot v1 NetGain aggregate (n={len(per_sample)}) ===")
    for label in ["B2-small", "B2-medium", "B2-large", "B2-val-best", "sequence_oracle"]:
        a = summary["aggregate"][label]
        print(f"  {label:18s}: median={a['median']:+.5f}, mean={a['mean']:+.5f}, min={a['min']:+.5f}, max={a['max']:+.5f}")
    print(f"\n  B2-val-best strength dist: {summary['best_strength_distribution_val_best']}")
    print(f"  Sequence oracle length dist: {summary['best_length_distribution_oracle']}")


if __name__ == "__main__":
    main()
