"""RL-2 Stage 1 (사용자 directive 2026-05-29): Q-surface utility / violation curve figure.

사용자 directive:
> "u increases: artifact improves, fidelity loss increases, BoneCV violation jumps after
>  u=0.55. 이런 곡선은 thesis 에 아주 좋습니다."

본 도구는 rl2_q_surface_dataset 의 surface_summary 를 읽어 tool 별 utility / violation /
artifact_reduction / fidelity_loss curve 를 u 축으로 plot (G2 vs synthetic 2-row).
Stage 1 은 3-point grid (0.3/0.6/1.0) — Stage 2 dense grid 에서 더 매끄러운 곡선 기대.

CLI:
    python -m tools.rl2_q_surface_figure \
        --dataset evals/snapshots/rl2_q_surface_dataset_stage1_v1.json \
        --output reports/figures/2026-05-29/q_surface_stage1.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = ("FootLockTool", "BoneProjectionTool", "VelocitySmoothingTool")
COLORS = {"FootLockTool": "#e6550d", "BoneProjectionTool": "#31a354", "VelocitySmoothingTool": "#3182bd"}


def _parse(summary):
    """surface_summary dict → {tool: {u: metrics}}."""
    out = {t: {} for t in TOOLS}
    for k, v in summary.items():
        m = re.match(r"(\w+)\|u=([\d.]+)", k)
        if not m:
            continue
        tool, u = m.group(1), float(m.group(2))
        if tool in out:
            out[tool][u] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_q_surface_dataset_stage1_v1.json")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "reports" / "figures" / "2026-05-29" / "q_surface_stage1.png")
    args = parser.parse_args()

    d = json.load(open(args.dataset, encoding="utf-8"))
    dists = [("g2", d["surface_summary_g2"]), ("synthetic", d["surface_summary_synthetic"])]
    metrics = [("mean_utility", "safe utility (Q)"), ("violation_rate", "physical violation rate"),
               ("mean_artifact_reduction", "artifact reduction"), ("mean_fidelity_loss", "fidelity loss")]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ri, (dname, summ) in enumerate(dists):
        parsed = _parse(summ)
        for ci, (mkey, mlabel) in enumerate(metrics):
            ax = axes[ri, ci]
            for tool in TOOLS:
                pts = sorted(parsed[tool].items())
                if not pts:
                    continue
                us = [p[0] for p in pts]
                ys = [p[1][mkey] for p in pts]
                ax.plot(us, ys, "-o", color=COLORS[tool], label=tool.replace("Tool", ""), linewidth=2, markersize=7)
            ax.set_xlabel("intervention intensity u")
            ax.set_ylabel(mlabel)
            ax.set_title(f"[{dname}] {mlabel}", fontsize=11)
            ax.grid(alpha=0.3)
            if mkey == "mean_utility":
                ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, label="STOP (u=0)")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=9)
    fig.suptitle("RL-2 Stage 1 — Q_safe(s, tool, u) action-effect surface (3-point grid, mean over states)\n"
                 "G2 (low severity, abstain-dominant) vs synthetic severe (multi-step correction regime)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output), dpi=120)
    plt.close(fig)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
