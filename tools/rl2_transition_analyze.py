"""RL-2 Stage A (사용자 directive 2026-05-29): action-effect transition 분석.

사용자 directive (Stage A 목표):
> "policy 학습이 아니라: utility curve 가 smooth 한가? gate boundary 가 보이는가?
>  tool 별 u sensitivity 가 다른가? B2 failure 가 u curve 에서 재현되는가? 를 보는 것."

본 도구는 rl2_transition_dataset_stageA 를 읽어:
  (1) tool 별 utility / violation curve (u 축, 4 distribution) figure.
  (2) curve smoothness 지표 (utility 의 2차 차분 max).
  (3) gate boundary (violation_rate 가 0.5 교차하는 u).
  (4) B2 failure 재현 — VelocitySmoothing 의 per-evaluator violation 기여 (BoneLengthCV 지배 여부).

CLI:
    python -m tools.rl2_transition_analyze \
        --dataset evals/snapshots/rl2_transition_dataset_stageA_v1.json \
        --output-fig reports/figures/2026-05-29/transition_stageA_curves.png \
        --output evals/snapshots/rl2_transition_analysis_stageA_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from tools.safe_sequence_oracle_run import _gate_violation

TOOLS = ("FootLockTool", "BoneProjectionTool", "VelocitySmoothingTool")
PHYSICAL = ("PenetrateEvaluator", "FloatEvaluator", "SkateEvaluator", "JerkSpikeEvaluator", "BoneLengthCVEvaluator")
DISTS = ("clean", "near_boundary", "g2_natural", "synthetic_severe")
COLORS = {"FootLockTool": "#e6550d", "BoneProjectionTool": "#31a354", "VelocitySmoothingTool": "#3182bd"}


def _curve(summary, tool, key):
    pts = []
    for k, v in summary.items():
        if k.startswith(tool + "|u="):
            u = float(k.split("u=")[1])
            pts.append((u, v[key]))
    return sorted(pts)


def _smoothness(ys):
    """max |2차 차분| — 작을수록 smooth."""
    if len(ys) < 3:
        return 0.0
    d2 = np.diff(ys, 2)
    return float(np.max(np.abs(d2)))


def _boundary_u(summary, tool, thr=0.5):
    """violation_rate 가 thr 을 처음 넘는 u (없으면 None)."""
    pts = _curve(summary, tool, "violation_rate")
    for u, v in pts:
        if v >= thr:
            return u
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_dataset_stageA_v1.json")
    parser.add_argument("--calibration", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    parser.add_argument("--output-fig", type=Path,
                        default=REPO_ROOT / "reports" / "figures" / "2026-05-29" / "transition_stageA_curves.png")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_analysis_stageA_v1.json")
    args = parser.parse_args()

    d = json.load(open(args.dataset, encoding="utf-8"))
    summaries = {dist: d[f"surface_summary_{dist}"] for dist in DISTS}
    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_thresholds = {n: calib["summary"][n]["p99"] for n in calib["summary"]
                       if calib["summary"][n].get("n", 0) > 0}

    # B2 failure: VelocitySmoothing per-evaluator violation contribution over u (recompute from rows).
    vel_eval_viol = defaultdict(lambda: defaultdict(lambda: {"hit": 0, "n": 0}))  # [dist][u][evaluator]
    for r in d["rows"]:
        if r["tool"] != "VelocitySmoothingTool":
            continue
        before = r["state"]["physical_scores"]; after = r["after_state"]["physical_scores"]
        dec = _gate_violation(after, before, gate_thresholds)
        u = r["u"]; dist = r["distribution"]
        for n in PHYSICAL:
            cell = vel_eval_viol[dist][u]
            if n not in cell:
                cell[n] = {"hit": 0, "n": 0}
            cell[n]["n"] += 1
            if dec.get(n) == "hard_violation":
                cell[n]["hit"] += 1

    # Analysis JSON.
    analysis = {"smoothness": {}, "gate_boundary_u": {}, "u_sensitivity": {}, "b2_failure_velsmooth": {}}
    for dist in DISTS:
        s = summaries[dist]
        analysis["smoothness"][dist] = {}
        analysis["gate_boundary_u"][dist] = {}
        analysis["u_sensitivity"][dist] = {}
        for tool in TOOLS:
            util = [y for _, y in _curve(s, tool, "mean_utility")]
            viol = [y for _, y in _curve(s, tool, "violation_rate")]
            analysis["smoothness"][dist][tool] = round(_smoothness(util), 6)
            analysis["gate_boundary_u"][dist][tool] = _boundary_u(s, tool)
            # u sensitivity = utility range (max-min) over u.
            analysis["u_sensitivity"][dist][tool] = round((max(util) - min(util)) if util else 0.0, 6)
    # B2 failure summary: at u=1.0, which evaluator dominates VelSmooth violations per dist.
    for dist in DISTS:
        u1 = max(vel_eval_viol[dist].keys()) if vel_eval_viol[dist] else None
        if u1 is None:
            continue
        cell = vel_eval_viol[dist][u1]
        rates = {n: round(cell[n]["hit"] / cell[n]["n"], 4) if cell[n]["n"] else 0.0 for n in PHYSICAL}
        dominant = max(rates.items(), key=lambda kv: kv[1])
        analysis["b2_failure_velsmooth"][dist] = {"u": u1, "per_evaluator_violation_rate": rates,
                                                  "dominant": dominant[0], "dominant_rate": dominant[1]}

    # === Figure: 4 dist rows × 2 cols (utility, violation) ===
    fig, axes = plt.subplots(len(DISTS), 2, figsize=(13, 4 * len(DISTS)))
    for ri, dist in enumerate(DISTS):
        s = summaries[dist]
        for ci, (key, ylabel) in enumerate([("mean_utility", "safe utility (Q)"), ("violation_rate", "physical violation rate")]):
            ax = axes[ri, ci]
            for tool in TOOLS:
                pts = _curve(s, tool, key)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                            color=COLORS[tool], label=tool.replace("Tool", ""), linewidth=2, markersize=5)
            ax.set_xlabel("intervention intensity u")
            ax.set_ylabel(ylabel)
            ax.set_title(f"[{dist}] {ylabel}", fontsize=11)
            ax.grid(alpha=0.3)
            if key == "mean_utility":
                ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=9)
    fig.suptitle("RL-2 Stage A — action-effect transition surface (dense u-grid, single-step)\n"
                 "clean / near_boundary / g2_natural (Protocol B) / synthetic_severe — utility + gate boundary", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    args.output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output_fig), dpi=120)
    plt.close(fig)

    out = {
        "schema_version": "1.0.0", "record_type": "rl2_transition_analysis",
        "task_id": "rl2_transition_analysis_stageA_v1",
        "dataset_source": str(args.dataset),
        "n_transitions": d["n_transitions"], "n_states": d["n_states"],
        "u_grid": d["u_grid"], "figure": str(args.output_fig.relative_to(REPO_ROOT)),
        "note": "Stage A 진단 (사용자): utility smoothness / gate boundary / u sensitivity / B2 failure 재현.",
        **analysis,
    }
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RL-2 Stage A Transition Analysis ===")
    print(f"  transitions: {d['n_transitions']}, states: {d['n_states']}")
    for dist in DISTS:
        print(f"\n  [{dist}]")
        for tool in TOOLS:
            sm = analysis["smoothness"][dist][tool]
            bd = analysis["gate_boundary_u"][dist][tool]
            se = analysis["u_sensitivity"][dist][tool]
            print(f"    {tool.replace('Tool',''):<18} smoothness(|d2|max)={sm:.4f}  gate_boundary_u={bd}  u_sensitivity={se:.4f}")
        b2 = analysis["b2_failure_velsmooth"].get(dist)
        if b2:
            print(f"    VelSmooth@u={b2['u']} dominant violation = {b2['dominant']} ({b2['dominant_rate']*100:.0f}%)")
    print(f"\n[OK] fig -> {args.output_fig}")
    print(f"[OK] analysis -> {args.output}")


if __name__ == "__main__":
    main()
