"""AR-058-5 (P5): routing 필요성 종합 — 4-panel figure + evidence-chain snapshot.

새 frame (2026-07-12 확정): "증상(artifact score) 최적화 ≠ 지각 품질 — refinement 는
motion 기전을 고려해야 한다". 모든 수치는 **frozen snapshot 에서 로드** (재계산 없음).

Panels:
  A. 문제 실재 + generator 프로파일 (P1/P2): foot_skate_world per generator + v2 gate fire
  B. 고정 처방의 실패 (P4): Y-only FootLock 의 mixed effect (artifact↓ vs fs↑)
  C. 물리 ≠ 지각 (AR-061 + A/B v1/v2): 조건부 물리 개선 vs 선호 우연
  D. 기전 (AR-071): root 전진 = GT 의 42%, 부족↔fs 상관

CLI (motion3d env):
    python tools/p5_synthesis_ar058_5.py
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
OUT_DIR = REPO_ROOT / "reports" / "figures" / "2026-07-12" / "poc_necessity_p1_p5"
OUT_SNAP = SNAP / "poc_necessity_p1_p5_v1.json"

GENS = ("mdm", "motiongpt", "momask")
GEN_LABEL = {"mdm": "MDM\n(diffusion)", "motiongpt": "MotionGPT\n(VQ)", "momask": "MoMask\n(VQ)"}
C_MDM, C_VQ = "#d62728", "#7f7f7f"


def _load(name):
    return json.load(open(SNAP / name, encoding="utf-8"))


def main() -> None:
    rep = _load("representative_pool_measure_v1.json")           # P1/P2 fs
    gate = _load("gate_prevalence_remeasure_ar063_v1.json")      # v2 skate fire
    p4 = _load("p4_fixed_tool_effect_v1.json")                   # fixed mixed effect
    ar061 = _load("coordinate_footskate_effect_ar061_v1.json")   # conditional effect
    ab1 = _load("ab_preference_result_ar058_3h_v1.json")         # A/B v1
    ab2 = _load("ab_preference_result_v2_ar058_3i_v1.json")      # A/B v2
    ab2p = _load("ab_preference_pack_v2_ar058_3i_v1.json")       # v2 pack physical
    ar071 = _load("root_gait_mismatch_ar071_v1.json")            # mechanism

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    fig.suptitle("Why artifact-score optimization is not enough — evidence chain (P1–P5)",
                 fontsize=13, y=0.985)

    # ---- Panel A: prevalence & profile.
    ax = axes[0, 0]
    fs = [rep["overall"][g]["overall"]["foot_skate_world"] for g in GENS]
    means = [x["mean"] for x in fs]
    errs = [[m - x["ci95"][0] for m, x in zip(means, fs)], [x["ci95"][1] - m for m, x in zip(means, fs)]]
    colors = [C_MDM, C_VQ, C_VQ]
    ax.bar(range(3), means, yerr=errs, color=colors, capsize=4)
    fire = [gate["results"][g]["by_space"]["trajectory"]["SkateEvaluator"]["fire"]["mean"] for g in GENS]
    for i, (m, f) in enumerate(zip(means, fire)):
        ax.text(i, m * 1.05, f"gate fire {f*100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels([GEN_LABEL[g] for g in GENS], fontsize=9)
    ax.set_ylabel("foot_skate_world (m/frame)")
    ax.set_title("A. Foot skating is real & generator-specific\n(representative-300, Cat-B; v2 gate)", fontsize=10)

    # ---- Panel B: fixed tool mixed effect (P4, FootLock large).
    ax = axes[0, 1]
    art = [p4["results"][g]["by_strength"]["large"]["artifact_total"]["mean_delta"] for g in GENS]
    fsk = [p4["results"][g]["by_strength"]["large"]["foot_skate_world"]["mean_delta"] for g in GENS]
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, art, w, label="artifact_total Δ (proxy target)", color="#1f77b4")
    ax.bar(x + w / 2, fsk, w, label="foot_skate Δ (side effect)", color="#ff7f0e")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([GEN_LABEL[g] for g in GENS], fontsize=9)
    ax.set_ylabel("Δ after fixed FootLock (large)")
    ax.legend(fontsize=8)
    ax.set_title("B. Fixed correction: target improves, side effect worsens\n(P4 — mixed effect on all 3 generators)", fontsize=10)

    # ---- Panel C: physics != perception.
    ax = axes[1, 0]
    d61 = [ar061["results"][g]["arms"]["coord_u100"]["foot_skate_world"]["mean_delta"] for g in GENS]
    ax.bar(np.arange(3) - 0.55, d61, 0.5, color=[C_MDM, C_VQ, C_VQ])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("coord tool fs Δ (u=1.0)")
    ax2 = ax.twinx()
    v1c = ab1["result"]["corrected_preferred"]; v2c = ab2["result"]["corrected_preferred"]
    ax2.bar([3.2, 4.2], [v1c / 20, v2c / 20], 0.6, color="#9467bd")
    ax2.axhline(0.5, color="#9467bd", ls="--", lw=1)
    ax2.set_ylim(0, 1); ax2.set_ylabel("A/B corrected preference", color="#9467bd")
    # twinx 는 x축 공유 — tick 은 한 번만, 좌측 bar + 우측 A/B 위치 모두 지정.
    ax.set_xticks([-0.55, 0.45, 1.45, 3.2, 4.2])
    ax.set_xticklabels(["MDM", "MGPT", "MoMask", f"A/B v1\n{v1c}/20", f"A/B v2\n{v2c}/20"], fontsize=8)
    fs_impr = ab2p["honesty_disclosure"]["fs_improved_pairs"]
    ax.set_title(f"C. Conditional correction improves physics (CI-clean)\nbut preference = chance (physical improved {fs_impr}/20 in v2)", fontsize=10)

    # ---- Panel D: mechanism.
    ax = axes[1, 1]
    t2 = ar071["T2_root_deficit_vs_GT"]; t3 = ar071["T3_deficit_fs_correlation"]
    ax.bar([0], [t2["speed_ratio_mean"]],
           yerr=[[t2["speed_ratio_mean"] - t2["speed_ratio_ci95"][0]],
                 [t2["speed_ratio_ci95"][1] - t2["speed_ratio_mean"]]],
           color=C_MDM, capsize=5, width=0.5)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.text(0.35, 1.01, "GT-consistent (=1.0)", fontsize=8)
    ax.text(0, t2["speed_ratio_mean"] / 2,
            f"{t2['speed_ratio_mean']:.2f}\n({int(t2['prompts_ratio_below_1']*t2['n_locomotion_prompts'])}/{t2['n_locomotion_prompts']}\nprompts < 1)",
            ha="center", fontsize=9, color="w")
    ax.text(0.62, 0.45,
            f"deficit vs foot_skate:\nSpearman rho = +{t3['spearman_rho_deficit_vs_fs']:.2f}\n(p = {t3['p_value']:.1e}, n = {t3['n']})\n\ndirection signature:\ninconclusive (T1)",
            fontsize=9, va="center")
    ax.set_xlim(-0.5, 1.6); ax.set_ylim(0, 1.15)
    ax.set_xticks([0]); ax.set_xticklabels(["MDM root speed / GT root speed"], fontsize=9)
    ax.set_title("D. Mechanism: root advances at ~42% of GT speed\n(AR-071 — deficit universal & correlated with skating)", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "p5_evidence_chain_4panel.png", dpi=150)
    plt.close(fig)

    chain = [
        {"step": "P1/P2 problem & profile", "claim": "foot skating 은 실재하고 generator-특이적 (MDM = VQ 의 ~2x; v2 gate fire 23% vs 1~2%)",
         "numbers": {"fs": {g: rep["overall"][g]["overall"]["foot_skate_world"]["mean"] for g in GENS},
                     "gate_fire_v2": {g: fire_v for g, fire_v in zip(GENS, fire)}},
         "evidence": ["representative_pool_measure_v1.json", "gate_prevalence_remeasure_ar063_v1.json"],
         "tier": "real-distribution", "category": "B (+v2 gate)"},
        {"step": "P3 quality relation", "claim": "generator 수준에서 MDM foot-skate ↔ 나쁜 FID/R@1; bone 축은 matched_dist 방향 일치 (상관)",
         "evidence": ["poc_artifact_quality_link_ar058_3_v1.json"], "tier": "real-distribution", "category": "A proxy 연결"},
        {"step": "P4 fixed insufficiency", "claim": "고정 FootLock: artifact_total↓ 이면서 foot_skate↑ — 3/3 generator mixed",
         "numbers": {"fs_delta_large": {g: p4["results"][g]["by_strength"]["large"]["foot_skate_world"]["mean_delta"] for g in GENS}},
         "evidence": ["p4_fixed_tool_effect_v1.json"], "tier": "real-distribution", "category": "B/C"},
        {"step": "E5 conditional physics", "claim": "coord cleanup: MDM CI-clean 개선 vs VQ 는 STOP 이 정답 — 조건부 선택 실증",
         "numbers": {"fs_delta_u100": {g: ar061["results"][g]["arms"]["coord_u100"]["foot_skate_world"]["mean_delta"] for g in GENS}},
         "evidence": ["coordinate_footskate_effect_ar061_v1.json"], "tier": "real-distribution", "category": "B"},
        {"step": "E6 physics != perception", "claim": "물리 개선(fs -48%, legCV 0)에도 blind 선호 = 우연 (v1 11/20 슬로모, v2 10/20 실속도) — 사전등록 2회",
         "numbers": {"ab_v1": f"{v1c}/20", "ab_v2": f"{v2c}/20", "v2_fs_improved": f"{fs_impr}/20"},
         "evidence": ["ab_preference_result_ar058_3h_v1.json", "ab_preference_result_v2_ar058_3i_v1.json"],
         "tier": "quality-validated 시도 (b1 한계)", "category": "perceptual b1"},
        {"step": "E7 mechanism", "claim": "MDM root 전진 = GT 의 42% (165/165 prompt) + 부족↔fs rho=+0.34 — 증상이 아니라 기전 (방향 서명은 불확정)",
         "numbers": {"speed_ratio": t2["speed_ratio_mean"], "rho": t3["spearman_rho_deficit_vs_fs"], "p": t3["p_value"]},
         "evidence": ["root_gait_mismatch_ar071_v1.json"], "tier": "real-distribution (진단)", "category": "B-계 (GT 참조)"},
    ]
    out = {
        "schema_version": "1.0.0", "record_type": "poc_necessity_p1_p5", "board_id": "AR-058-5",
        "frame": "증상(artifact score) 최적화 ≠ 지각 품질 — post-generation refinement 는 motion 기전을 고려해야 한다",
        "evidence_chain": chain,
        "figure": "reports/figures/2026-07-12/poc_necessity_p1_p5/p5_evidence_chain_4panel.png",
        "constraints_honored": ["v0.1.0 gate-fire 불인용 (v2 만)", "pair 구조 §0-0", "b1 한계 명시", "단일 벤치마크(HumanML3D) 스코프"],
        "claim_boundary": {
            "allowed": "관측된 발생·공통좌표 측정가능성·품질 관계·고정 처방의 부작용·조건부 물리 개선·지각 불충분·기전 진단이 'artifact-aware, mechanism-aware refinement' 의 필요성을 동기화한다",
            "forbidden": "ArtifactRouter 가 artifact 를 해결했다 / 지각 품질 향상을 달성했다 / 모든 generator 에 일반화된다 (b1, 단일 벤치마크)"},
    }
    OUT_SNAP.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_SNAP, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[OK] figure -> {OUT_DIR / 'p5_evidence_chain_4panel.png'}")
    print(f"[OK] snapshot -> {OUT_SNAP}")


if __name__ == "__main__":
    main()
