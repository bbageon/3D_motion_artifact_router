"""AR-058-4 (P4): fixed tool insufficiency — representative pool 에 fixed FootLock 적용 전후 Δ.

Claim: fixed tool/strength post-processing 은 한 지표(artifact)를 개선하면서 다른 지표
(foot_skate 등)를 악화시킨다(mixed effect) → routing 필요.

방법: 대표 pool 각 motion 에 **fixed FootLock**(small/medium/large, 조건부 아님)을 적용하고
before-after Δ 를 generator별로 집계. artifact_total(Category C, tool target) vs foot_skate_world
(Category B) 의 mixed effect 확인. per-prompt(seed 평균) 단위, bootstrap CI.

FootLock 은 local-pose 에 적용(§3-1 canonical). foot_skate_world 는 corrected_local + 원본 root
로 trajectory 재구성 후 측정. ground_y = estimate_ground(local).

CLI (motion3d env):
    python tools/fixed_tool_effect_p4.py \
        --pool-root external_assets/protocol_rep_pool_seed20260608 \
        --output evals/snapshots/p4_fixed_tool_effect_v1.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import FootLockTool
from evaluators import DEFAULT_EVALUATORS
from tools.coords_protocol import estimate_ground, PELVIS
from tools.representative_pool_measure import foot_skate_world
from tools.physical_metric_g2_stress import float_mag

ARTIFACT_NAMES = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
STRENGTHS = ("small", "medium", "large")
METRICS = ("artifact_total", "foot_skate_world", "float_mag")


def _artifact_total(evs, m):
    return float(np.mean([max((r.score for r in ev.evaluate(m)), default=0.0) for ev in evs]))


def _measure(local, traj, art_evs):
    return {
        "artifact_total": _artifact_total(art_evs, local),
        "foot_skate_world": foot_skate_world(traj),
        "float_mag": float_mag(local),
    }


def _boot_ci(vals, rng, n=1000):
    nv = len(vals)
    bs = [float(np.mean(vals[rng.integers(0, nv, nv)])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path, default=REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608")
    ap.add_argument("--generators", nargs="+", default=["motiongpt", "mdm", "momask"])
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "p4_fixed_tool_effect_v1.json")
    args = ap.parse_args()
    args.pool_root = args.pool_root.resolve()

    rng = np.random.default_rng(args.seed)
    tool = FootLockTool()
    art_evs = [ev for ev in DEFAULT_EVALUATORS if ev.name in ARTIFACT_NAMES]

    results = {}
    for gen in args.generators:
        d = args.pool_root / gen
        metas = [json.load(open(p, encoding="utf-8")) for p in sorted(d.glob("*.json")) if p.name != "_pool_summary.json"]
        # per-prompt (seed 평균) Δ per strength per metric.
        by_sid_delta = {s: {mt: defaultdict(list) for mt in METRICS} for s in STRENGTHS}
        for m in metas:
            local = np.load(REPO_ROOT / m["local_npy"]).astype(np.float64)
            traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
            root = traj[:, PELVIS:PELVIS + 1, :]
            gy = estimate_ground(local)
            before = _measure(local, traj, art_evs)
            T = local.shape[0]
            for s in STRENGTHS:
                corr_local, _ = tool.apply(local, target_part="both_feet",
                                           target_joints=["LEFT_FOOT", "RIGHT_FOOT"],
                                           frame_range=(0, T - 1), strength=s,
                                           metadata={"ground_y": gy})
                corr_traj = corr_local + root  # trajectory 재구성 (root 불변)
                after = _measure(corr_local, corr_traj, art_evs)
                for mt in METRICS:
                    by_sid_delta[s][mt][m["sample_id"]].append(after[mt] - before[mt])
        # aggregate: seed 평균 → per-prompt Δ, then over prompts (mean + CI + frac worse/better).
        gen_res = {}
        for s in STRENGTHS:
            sr = {}
            for mt in METRICS:
                per_prompt = np.array([float(np.mean(v)) for v in by_sid_delta[s][mt].values()])
                sr[mt] = {"mean_delta": round(float(per_prompt.mean()), 5),
                          "ci95": _boot_ci(per_prompt, rng),
                          "frac_improved": round(float(np.mean(per_prompt < 0)), 3)}  # lower=better
            gen_res[s] = sr
        results[gen] = {"n_prompts": len({m["sample_id"] for m in metas}), "by_strength": gen_res}
        gl = results[gen]["by_strength"]["large"]
        print(f"[{gen}] FootLock large: artifact_total Δ={gl['artifact_total']['mean_delta']:+.4f} "
              f"foot_skate Δ={gl['foot_skate_world']['mean_delta']:+.4f} float Δ={gl['float_mag']['mean_delta']:+.4f}")

    # mixed-effect 판정: artifact 개선(음수) + foot_skate 악화(양수).
    mixed = {}
    for gen in args.generators:
        gl = results[gen]["by_strength"]["large"]
        mixed[gen] = {
            "artifact_improved": gl["artifact_total"]["mean_delta"] < 0 and gl["artifact_total"]["ci95"][1] < 0,
            "foot_skate_degraded": gl["foot_skate_world"]["mean_delta"] > 0 and gl["foot_skate_world"]["ci95"][0] > 0,
        }
        mixed[gen]["is_mixed"] = mixed[gen]["artifact_improved"] and mixed[gen]["foot_skate_degraded"]

    out = {
        "schema_version": "1.0.0", "record_type": "p4_fixed_tool_effect", "board_id": "AR-058-4",
        "tool": "FootLockTool (fixed, both_feet, full frame_range, ground_y=estimate_ground(local))",
        "strengths": list(STRENGTHS),
        "metric_provenance": {"artifact_total": "C proxy (local, tool target)",
                              "foot_skate_world": "B (trajectory=corrected_local+root, estimate_ground)",
                              "float_mag": "B (local)"},
        "stat_unit": "prompt (seed 평균; bootstrap CI). lower=better; Δ<0=개선, Δ>0=악화.",
        "results": results,
        "mixed_effect": mixed,
        "claim": "fixed post-processing insufficient — artifact 개선하면서 foot_skate 악화(mixed) → correction 은 motion state/generator/target/strength 조건부여야.",
        "claim_boundary": "현재 learned policy 가 optimal 이라 주장하지 않음. Category-A(FID) 악화는 별도 층(P4-FID).",
        "grounding": ["PhysDiff ICCV2023 (physical correction dynamics)", "HumanML3D CVPR2022 (standard quality)"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n=== P4 mixed effect (FootLock large) ===")
    for gen in args.generators:
        gl = results[gen]["by_strength"]["large"]
        print(f"  {gen:<10} artifact {gl['artifact_total']['mean_delta']:+.4f}{gl['artifact_total']['ci95']} | "
              f"foot_skate {gl['foot_skate_world']['mean_delta']:+.4f}{gl['foot_skate_world']['ci95']} | "
              f"MIXED={mixed[gen]['is_mixed']}")
    print("\n  strength 의존 (artifact / foot_skate Δ):")
    for gen in args.generators:
        row = "  %-10s " % gen
        for s in STRENGTHS:
            r = results[gen]["by_strength"][s]
            row += f"{s}:({r['artifact_total']['mean_delta']:+.4f},{r['foot_skate_world']['mean_delta']:+.4f}) "
        print(row)
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
