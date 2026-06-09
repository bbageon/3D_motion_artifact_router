"""AR-044: representative-300 3-generator problem taxonomy + tool headroom measurement.

per-generator 로 어떤 physical/artifact 문제가 얼마나 발생하는지(=tool headroom) 측정한다.
- foot_skate = trajectory(world-space) + estimate_ground(10th pct, ≠minY) → 표준 Category-B.
- gate/artifact prevalence = local(canonical) + AR-043 threshold → Category-C proxy (G2 와 동일 기준).
- 통계 단위 = prompt (seed 평균 후 300 prompt aggregate, bootstrap CI).
- complexity subgroup(overall top/bottom 25%) 보조 분석.

CLI (motion3d env):
    python tools/representative_pool_measure.py \
        --pool-root external_assets/protocol_rep_pool_seed20260608 \
        --bank evals/prompts/protocol_rep_test_300_seed20260608.json \
        --output evals/snapshots/representative_pool_measure_v1.json
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

from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from tools.coords_protocol import estimate_ground, LEFT_FOOT, RIGHT_FOOT
from tools.physical_metric_g2_stress import accel_stats

ARTIFACT_THRESH = {"FootFloatingEvaluator": 0.05, "BoneLengthEvaluator": 0.02, "VelocityJitterEvaluator": 0.03}
PHYS_P99 = {"PenetrateEvaluator": 0.0, "FloatEvaluator": 0.7983539458952797, "SkateEvaluator": 0.0,
            "JerkSpikeEvaluator": 0.10556407202395618, "BoneLengthCVEvaluator": 5.506674955074729e-06}


def foot_skate_world(traj: np.ndarray, H: float = 0.05) -> float:
    """GMD/EDGE foot-skate on world-space trajectory, ground = estimate_ground (≠minY)."""
    if traj.shape[0] < 2:
        return 0.0
    g = estimate_ground(traj)
    per = []
    for j in (LEFT_FOOT, RIGHT_FOOT):
        fxz = traj[:, j, :][:, [0, 2]]
        disp = np.linalg.norm(np.diff(fxz, axis=0), axis=1)
        h = traj[1:, j, 1] - g
        w = np.clip(2.0 - np.power(2.0, h / H), 0.0, None)
        per.append(disp * w)
    return float(np.mean(np.concatenate(per)))


def _boot_ci(vals: np.ndarray, rng, n=1000):
    n_v = len(vals)
    bs = [float(np.mean(vals[rng.integers(0, n_v, n_v)])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path, default=REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608")
    ap.add_argument("--bank", type=Path, default=REPO_ROOT / "evals" / "prompts" / "protocol_rep_test_300_seed20260608.json")
    ap.add_argument("--generators", nargs="+", default=["motiongpt", "mdm", "momask"])
    ap.add_argument("--seed", type=int, default=20260609)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "representative_pool_measure_v1.json")
    args = ap.parse_args()
    args.pool_root = args.pool_root.resolve()

    rng = np.random.default_rng(args.seed)
    bank = json.load(open(args.bank, encoding="utf-8"))
    membership = {r["sample_id"]: r["complexity"]["membership"] for r in bank["rows"]}

    art_evs = {ev.name: ev for ev in DEFAULT_EVALUATORS if ev.name in ARTIFACT_THRESH}
    phy_evs = {ev.name: ev for ev in DEFAULT_PHYSICAL_GATE_EVALUATORS}

    def measure(traj, local):
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

    METRICS = ["foot_skate_world", "accel", "artifact_total", "FootFloating_fire",
               "Skate_gate_fire", "Float_gate_fire", "BoneCV_fire"]
    results = {}
    per_prompt_store = {}  # gen -> metric -> {sid: value}
    for gen in args.generators:
        d = args.pool_root / gen
        metas = [json.load(open(p, encoding="utf-8")) for p in sorted(d.glob("*.json")) if p.name != "_pool_summary.json"]
        by_sid = defaultdict(list)
        for m in metas:
            by_sid[m["sample_id"]].append(m)
        prompt_vals = {mt: {} for mt in METRICS}
        for sid, ms in by_sid.items():
            seed_vals = defaultdict(list)
            for m in ms:
                traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
                local = np.load(REPO_ROOT / m["local_npy"]).astype(np.float64)
                mv = measure(traj, local)
                for mt in METRICS:
                    seed_vals[mt].append(mv[mt])
            for mt in METRICS:
                prompt_vals[mt][sid] = float(np.mean(seed_vals[mt]))  # seed 평균 → per-prompt
        per_prompt_store[gen] = prompt_vals
        # aggregate over 300 prompts.
        agg = {}
        for mt in METRICS:
            v = np.array(list(prompt_vals[mt].values()))
            agg[mt] = {"mean": round(float(v.mean()), 5), "ci95": _boot_ci(v, rng)}
        results[gen] = {"n_prompts": len(by_sid), "overall": agg}
        print(f"[{gen}] n_prompt={len(by_sid)}  foot_skate_world={agg['foot_skate_world']['mean']:.4f}  "
              f"accel={agg['accel']['mean']:.4f}  artifact={agg['artifact_total']['mean']:.4f}  "
              f"FootFloating={agg['FootFloating_fire']['mean']:.0%}  Skate_gate={agg['Skate_gate_fire']['mean']:.0%}")

    # complexity subgroup (overall top/bottom 25%): per-generator key metric.
    subgroup = {}
    for sg in ("overall_top25", "overall_bottom25"):
        sids = [sid for sid, mem in membership.items() if mem.get(sg)]
        subgroup[sg] = {"n": len(sids)}
        for gen in args.generators:
            pv = per_prompt_store[gen]
            subgroup[sg][gen] = {
                mt: round(float(np.mean([pv[mt][s] for s in sids if s in pv[mt]])), 5)
                for mt in ("foot_skate_world", "accel", "artifact_total", "FootFloating_fire")
            }

    out = {
        "schema_version": "1.0.0", "record_type": "representative_pool_measure", "board_id": "AR-044",
        "pool": str(args.pool_root.name), "bank": str(args.bank.name),
        "generators": args.generators, "n_prompts": len(membership),
        "metric_provenance": {
            "foot_skate_world": "GMD/EDGE on trajectory(world) + estimate_ground 10th pct (Category B)",
            "accel": "per-joint accel norm mean on local-pose (smoothness, MDM)",
            "artifact_total/*_fire": "local-pose, AR-043 threshold (Category C proxy)",
        },
        "evidence_tier": ["real-distribution evidence (representative-300)"],
        "stat_unit": "prompt (seed 평균 후 300 prompt; bootstrap CI)",
        "overall": results,
        "complexity_subgroup": subgroup,
        "claim_boundary": "representative-300 taxonomy/headroom. generator 우열·tool 효과는 보정 적용 후 별도. n=300 절대 FID 단독 금지.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
