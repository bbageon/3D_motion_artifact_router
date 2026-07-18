"""AR-058-5 P5 최종화: 결정 계층 파트 보조 figure (2-panel) — frozen snapshot 로드만.

Panel E: 같은 처방의 generator-조건부 효과 (Cat-A Δ per generator — AR-077 최종 수치:
         R@1 = rprec fix, MM/FID = v3). MDM 개선 vs VQ 악화.
Panel F: routing gate 비교 (AR-078) — benefit-AUC (CI) + harm@30% 나란히.
         "순위는 개선, 안전 기준 미달" 을 한눈에.

CLI (motion3d env):
    python tools/p5_decision_figure_ar058_5.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAP = REPO_ROOT / "evals" / "snapshots"
OUT = REPO_ROOT / "reports" / "figures" / "2026-07-13" / "poc_necessity_p1_p5"
GENS = ("mdm", "motiongpt", "momask")
GLAB = {"mdm": "MDM\n(diffusion)", "motiongpt": "MotionGPT\n(VQ)", "momask": "MoMask\n(VQ)"}
C_MDM, C_VQ = "#d62728", "#7f7f7f"


def _load(name):
    return json.load(open(SNAP / name, encoding="utf-8"))


def main() -> None:
    rp = {g: _load(f"catA_rprec_fix_ar077_{g}_v1.json") for g in GENS}
    v3 = {g: _load(f"vq_root_catA_v3_ar077_{g}_v1.json") for g in GENS}
    gate = _load("routing_gate_compare_ar078_v1.json")

    fig, (axE, axF) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.suptitle("Correction must be conditional — and per-motion gating remains open",
                 fontsize=12, y=0.98)

    # ---- Panel E: Cat-A deltas per generator (R@1 / MM / FID — 부호 통일: >0 = 개선).
    x = np.arange(3); w = 0.26
    r1 = [rp[g]["R@1_diff_prompt_bootstrap"]["mean"] for g in GENS]
    mm = [-v3[g]["paired_diff_prompt_bootstrap"]["MM_Dist"]["mean"] for g in GENS]       # 부호 반전
    fid = [-v3[g]["paired_diff_prompt_bootstrap"]["FID"]["mean"] / 10.0 for g in GENS]   # /10 스케일
    axE.bar(x - w, r1, w, label="R@1 Δ", color="#1f77b4")
    axE.bar(x, mm, w, label="−MM-Dist Δ", color="#2ca02c")
    axE.bar(x + w, fid, w, label="−FID Δ (÷10)", color="#9467bd")
    axE.axhline(0, color="k", lw=0.8)
    axE.set_xticks(x); axE.set_xticklabels([GLAB[g] for g in GENS], fontsize=9)
    axE.set_ylabel("improvement (>0 = better)")
    axE.legend(fontsize=8)
    axE.set_title("E. Same correction: helps MDM, harms VQ\n(Cat-A, locomotion pool — pool-scoped)", fontsize=10)

    # ---- Panel F: gate AUC (CI) + harm@30.
    order = ["G1_generator_only", "G2_skate_only", "G3_mismatch_only", "G4_richer_nogen", "G5_richer_gen"]
    lab = ["gen-only", "skate", "mismatch", "richer\n(no-gen)", "richer\n(+gen)"]
    auc = [gate["holdout_benefit_auc"][g]["auc"] for g in order]
    lo = [auc[i] - gate["holdout_benefit_auc"][g]["ci95"][0] for i, g in enumerate(order)]
    hi = [gate["holdout_benefit_auc"][g]["ci95"][1] - auc[i] for i, g in enumerate(order)]
    harm30 = [gate["harmful_apply_at_rate"][g]["30%"]["harmful_apply"] for g in order]
    xs = np.arange(5)
    axF.bar(xs - 0.2, auc, 0.38, yerr=[lo, hi], capsize=3, label="benefit-AUC (holdout)", color="#1f77b4")
    axF.bar(xs + 0.2, harm30, 0.38, label="harmful-apply @30%", color="#ff7f0e")
    axF.axhline(0.5, color="#1f77b4", ls=":", lw=1)
    axF.axhline(gate["base_rates_holdout"]["harm"], color="#ff7f0e", ls=":", lw=1)
    axF.text(4.55, 0.505, "chance", fontsize=7, color="#1f77b4")
    axF.text(4.55, gate["base_rates_holdout"]["harm"] + 0.005, "base harm", fontsize=7, color="#ff7f0e")
    axF.set_xticks(xs); axF.set_xticklabels(lab, fontsize=8)
    axF.set_ylim(0, 0.85)
    axF.legend(fontsize=8)
    axF.set_title("F. Gates: richer state ranks better (AUC ↑, CI-clean)\nbut no-harm bar not met (harm@30% not sig. lower)", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    out_png = OUT / "p5_decision_layer_2panel.png"
    fig.savefig(out_png, dpi=150)
    print(f"[OK] {out_png}")


if __name__ == "__main__":
    main()
