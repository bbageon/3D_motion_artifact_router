"""AR-043 — dataset predefined issue prevalence audit.

AR-040(q_proxy / v0 ranker) 전에 확인할 전제:
  "우리가 고치려는 artifact / physical issue 가 데이터셋 안에 실제로 존재하는가?"

본 도구는 기존 snapshot 을 재사용한다.
  - G2 full pool: evals/snapshots/g2_real_stress_profile_v2.json
  - Stage-2G states: evals/snapshots/rl2_transition_g2_real_stage2_v1.json
  - physical thresholds: evals/snapshots/physical_gate_clean_calibration_v1.json

출력:
  evals/snapshots/dataset_issue_prevalence_audit_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluators.bone_length_evaluator import SEV_LOW as BONE_LOW
from evaluators.foot_floating_evaluator import SEV_LOW as FOOT_FLOAT_LOW
from evaluators.velocity_jitter_evaluator import SEV_LOW as VEL_JITTER_LOW
from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, REAL_DISTRIBUTION

ARTIFACT_THRESHOLDS = {
    "FootFloatingEvaluator": float(FOOT_FLOAT_LOW),
    "BoneLengthEvaluator": float(BONE_LOW),
    "VelocityJitterEvaluator": float(VEL_JITTER_LOW),
}
PHYSICAL_NAMES = (
    "PenetrateEvaluator",
    "FloatEvaluator",
    "SkateEvaluator",
    "JerkSpikeEvaluator",
    "BoneLengthCVEvaluator",
)
ARTIFACT_NAMES = tuple(ARTIFACT_THRESHOLDS)


def _summary(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": round(float(np.mean(arr)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def _artifact_total(row: dict[str, Any]) -> float:
    art = row.get("artifact_scores", {})
    vals = [float(art.get(n, 0.0)) for n in ARTIFACT_NAMES]
    return float(np.mean(vals)) if vals else 0.0


def _dominant_issue(row: dict[str, Any], physical_thresholds: dict[str, float]) -> str:
    art = row.get("artifact_scores", {})
    phy = row.get("physical_scores", {})
    candidates: dict[str, float] = {}
    for n, thr in ARTIFACT_THRESHOLDS.items():
        candidates[n] = float(art.get(n, 0.0)) / max(thr, 1e-9)
    for n, thr in physical_thresholds.items():
        candidates[n] = float(phy.get(n, 0.0)) / max(thr, 1e-9)
    return max(candidates, key=candidates.get) if candidates else "none"


def _summarize_group(rows: list[dict[str, Any]], physical_thresholds: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    out["artifact_total"] = _summary([_artifact_total(r) for r in rows])

    art_out = {}
    for n, thr in ARTIFACT_THRESHOLDS.items():
        scores = [float(r.get("artifact_scores", {}).get(n, 0.0)) for r in rows]
        art_out[n] = {
            **_summary(scores),
            "threshold_low": thr,
            "nonzero_rate": round(float(np.mean(np.asarray(scores) > 0.0)), 6) if scores else 0.0,
            "low_or_above_rate": round(float(np.mean(np.asarray(scores) >= thr)), 6) if scores else 0.0,
        }
    out["artifact_prevalence"] = art_out

    phy_out = {}
    for n in PHYSICAL_NAMES:
        thr = float(physical_thresholds.get(n, 0.0))
        scores = [float(r.get("physical_scores", {}).get(n, 0.0)) for r in rows]
        arr = np.asarray(scores, dtype=np.float64)
        phy_out[n] = {
            **_summary(scores),
            "clean_p99_threshold": thr,
            "nonzero_rate": round(float(np.mean(arr > 0.0)), 6) if scores else 0.0,
            "above_clean_p99_rate": round(float(np.mean(arr > thr)), 6) if scores else 0.0,
        }
    out["physical_prevalence"] = phy_out

    dominant = Counter(_dominant_issue(r, physical_thresholds) for r in rows)
    out["dominant_issue_counts"] = dict(dominant)
    out["dominant_issue_top3"] = dominant.most_common(3)
    return out


def _evaluator_reliability(split_summary: dict[str, Any]) -> dict[str, Any]:
    """명세 성공조건 #4 / Audit 질문 #5 — evaluator 별 신뢰도 등급.

    신뢰 가능한 issue detector = stress 에 잘 fire (sensitivity) + clean 에 안 fire (specificity)
    + stress vs natural 구분 (separation). 약한 evaluator = clean 에서 과다 fire(weak_specificity)
    또는 stress 에서도 거의 안 fire(weak_signal) 또는 전 split 100% fire(no_discrimination).
    """
    stress = split_summary.get("g2_stress_holdout", {})
    natural = split_summary.get("g2_natural_holdout", {})
    clean = split_summary.get("clean_noharm_holdout", {})

    def _grade(stress_f, natural_f, clean_f, p99_zero=False):
        sep = stress_f - natural_f
        if stress_f >= 0.95 and natural_f >= 0.95 and clean_f >= 0.95:
            g = "no_discrimination"  # 전 split fire → issue 존재는 맞으나 discriminator 로 약함
        elif clean_f > 0.30:
            g = "weak_specificity"   # clean 에서도 과다 fire → false-positive 경향
        elif stress_f < 0.20:
            g = "weak_signal"        # stress 에서도 거의 안 fire
        elif sep >= 0.20 and clean_f <= 0.30:
            g = "reliable"           # stress fire + clean 보존 + 구분
        else:
            g = "moderate"
        return g

    out = {"artifact": {}, "physical": {}}
    for n in ARTIFACT_THRESHOLDS:
        sf = float(stress.get("artifact_prevalence", {}).get(n, {}).get("low_or_above_rate", 0.0))
        nf = float(natural.get("artifact_prevalence", {}).get(n, {}).get("low_or_above_rate", 0.0))
        cf = float(clean.get("artifact_prevalence", {}).get(n, {}).get("low_or_above_rate", 0.0))
        out["artifact"][n] = {"stress_fire": round(sf, 4), "natural_fire": round(nf, 4),
                              "clean_fire": round(cf, 4), "stress_minus_natural": round(sf - nf, 4),
                              "grade": _grade(sf, nf, cf)}
    for n in PHYSICAL_NAMES:
        sf = float(stress.get("physical_prevalence", {}).get(n, {}).get("above_clean_p99_rate", 0.0))
        nf = float(natural.get("physical_prevalence", {}).get(n, {}).get("above_clean_p99_rate", 0.0))
        cf = float(clean.get("physical_prevalence", {}).get(n, {}).get("above_clean_p99_rate", 0.0))
        thr = float(stress.get("physical_prevalence", {}).get(n, {}).get("clean_p99_threshold", 0.0))
        out["physical"][n] = {"stress_fire": round(sf, 4), "natural_fire": round(nf, 4),
                              "clean_fire": round(cf, 4), "stress_minus_natural": round(sf - nf, 4),
                              "clean_p99_threshold": thr, "p99_near_zero": thr <= 1e-9,
                              "grade": _grade(sf, nf, cf, p99_zero=thr <= 1e-9)}
    # 등급 집계.
    all_grades = {**{f"ART/{k}": v["grade"] for k, v in out["artifact"].items()},
                  **{f"PHY/{k}": v["grade"] for k, v in out["physical"].items()}}
    out["grade_counts"] = dict(Counter(all_grades.values()))
    out["reliable_evaluators"] = [k for k, g in all_grades.items() if g == "reliable"]
    out["weak_evaluators"] = [k for k, g in all_grades.items() if g in ("weak_specificity", "weak_signal", "no_discrimination")]
    return out


def _ar040_decision(key_findings: dict[str, Any], reliability: dict[str, Any]) -> dict[str, Any]:
    """명세 성공조건 #5 — AR-040 진행 여부 formal 결정."""
    ratio = float(key_findings.get("g2_stress_vs_natural_artifact_total_mean_ratio", 0.0))
    stress_has = bool(key_findings.get("stress_split_has_predefined_issue"))
    syn_has = bool(key_findings.get("synthetic_diag_artifact_total_mean", 0.0) > 0.05)
    premise = stress_has and ratio >= 2.0  # stress 가 real issue 보유 + natural 대비 분리.
    return {
        "decision": "GO" if premise else "HOLD",
        "premise_holds": premise,
        "rationale": (
            f"g2_stress 가 predefined issue 보유(={stress_has}), stress/natural artifact_total ratio="
            f"{ratio:.2f}x (>=2.0), synthetic_diag 강한 issue(={syn_has}). → AR-040 의 q_proxy/oracle 학습이 "
            "헛돌지 않을 데이터 전제 성립."
        ),
        "conditions_for_ar040": [
            "q_proxy 가 no_discrimination evaluator (BoneLengthCV 등 전 split 100% fire) 에 지배되지 않게 weight 설계 — 아무 smoothing 보상 함정(이전 B2 100% BoneLengthCV violation) 회피.",
            "weak_specificity evaluator (clean 에서 과다 fire) 의 reward 기여 축소 — clean false-positive 를 '고쳐서' 보상 금지.",
            "clean_noharm 은 artifact-free 가 아니라 no-harm reference split — q_proxy 는 clean 에서 개입을 보상하지 않음.",
            "q_proxy ↔ standard metric(FID/R-Prec/MM-Dist) 정렬은 AR-040 이후 검증 (internal proxy validity, AR-037 §10-1).",
        ],
        "primary_headroom_target": "g2_stress_holdout (real-distribution, 4.1x artifact-rich)",
    }


def _states_from_transition(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.load(open(path, encoding="utf-8"))
    rows = []
    for sid, s in data["states"].items():
        rows.append({
            "state_id": sid,
            "split": s.get("split", "unknown"),
            "distribution": s.get("distribution", "unknown"),
            "motion_group": s.get("motion_group", "unknown"),
            "band": s.get("band", "unknown"),
            "artifact_scores": s.get("artifact_scores", {}),
            "physical_scores": s.get("physical_scores", {}),
        })
    return data, rows


def _rows_from_g2_profile(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.load(open(path, encoding="utf-8"))
    rows = []
    for r in data["per_motion"]:
        rows.append({
            "state_id": r.get("trial_id"),
            "split": "g2_full_pool",
            "distribution": "g2_natural",
            "motion_group": r.get("motion_group", "unknown"),
            "band": r.get("band", "unknown"),
            "artifact_scores": r.get("artifact_scores", {}),
            "physical_scores": r.get("physical_scores", {}),
        })
    return data, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_g2_real_stage2_v1.json")
    parser.add_argument("--g2-profile", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "g2_real_stress_profile_v2.json")
    parser.add_argument("--physical-calibration", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "dataset_issue_prevalence_audit_v1.json")
    args = parser.parse_args()

    transition_meta, transition_rows = _states_from_transition(args.transition)
    g2_meta, g2_rows = _rows_from_g2_profile(args.g2_profile)
    calibration = json.load(open(args.physical_calibration, encoding="utf-8"))
    physical_thresholds = {n: float(calibration["summary"][n]["p99"]) for n in PHYSICAL_NAMES}

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in transition_rows:
        by_split[r["split"]].append(r)
    by_split["g2_full_pool"] = g2_rows

    split_summary = {k: _summarize_group(v, physical_thresholds) for k, v in sorted(by_split.items())}

    # Group-level summary only for G2 stress/natural/full-pool because motion-group balance matters there.
    group_summary: dict[str, dict[str, Any]] = {}
    for split in ("g2_full_pool", "g2_stress_holdout", "g2_natural_holdout"):
        rows = by_split.get(split, [])
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_group[r.get("motion_group", "unknown")].append(r)
        group_summary[split] = {g: _summarize_group(rs, physical_thresholds) for g, rs in sorted(by_group.items())}

    stress = split_summary.get("g2_stress_holdout", {})
    natural = split_summary.get("g2_natural_holdout", {})
    clean = split_summary.get("clean_noharm_holdout", {})
    synthetic = split_summary.get("synthetic_diag_holdout", {})

    def _mean_art(split_summary_obj: dict[str, Any]) -> float:
        return float(split_summary_obj.get("artifact_total", {}).get("mean", 0.0))

    key_findings = {
        "g2_stress_vs_natural_artifact_total_mean_ratio": round(
            _mean_art(stress) / max(_mean_art(natural), 1e-9), 6
        ),
        "g2_stress_artifact_total_mean": _mean_art(stress),
        "g2_natural_artifact_total_mean": _mean_art(natural),
        "clean_artifact_total_mean": _mean_art(clean),
        "synthetic_diag_artifact_total_mean": _mean_art(synthetic),
        "stress_split_has_predefined_issue": bool(
            stress and (
                any(v.get("low_or_above_rate", 0.0) > 0.05 for v in stress.get("artifact_prevalence", {}).values())
                or any(v.get("above_clean_p99_rate", 0.0) > 0.05 for v in stress.get("physical_prevalence", {}).values())
            )
        ),
        "natural_split_has_predefined_issue": bool(
            natural and (
                any(v.get("low_or_above_rate", 0.0) > 0.05 for v in natural.get("artifact_prevalence", {}).values())
                or any(v.get("above_clean_p99_rate", 0.0) > 0.05 for v in natural.get("physical_prevalence", {}).values())
            )
        ),
        "clean_noharm_has_predefined_issue": bool(
            clean and (
                any(v.get("low_or_above_rate", 0.0) > 0.05 for v in clean.get("artifact_prevalence", {}).values())
                or any(v.get("above_clean_p99_rate", 0.0) > 0.05 for v in clean.get("physical_prevalence", {}).values())
            )
        ),
    }

    out = {
        "schema_version": "1.0.0",
        "record_type": "dataset_issue_prevalence_audit",
        "task_id": "AR-043-dataset-issue-prevalence-audit",
        "split_id": "dataset_issue_prevalence_audit_v1",
        "oracle_type": "N/A (dataset audit)",
        "action_space_grid": {"grid": "N/A", "stage": "AR-043-dataset-audit", "include_stop": True,
                              "action_type": "not_applicable"},
        "evidence_tier": [REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
        "source_snapshots": {
            "transition": str(args.transition.relative_to(REPO_ROOT)),
            "g2_profile": str(args.g2_profile.relative_to(REPO_ROOT)),
            "physical_calibration": str(args.physical_calibration.relative_to(REPO_ROOT)),
        },
        "evaluator_config_hashes": transition_meta.get("evaluator_config_hashes", {}),
        "evaluator_severity_versions": transition_meta.get("evaluator_severity_versions", {}),
        "artifact_thresholds": ARTIFACT_THRESHOLDS,
        "physical_thresholds_clean_p99": physical_thresholds,
        "threshold_policy": {
            "artifact": "score >= evaluator SEV_LOW",
            "physical": "score > clean calibration p99",
            "nonzero_rates": "reported as diagnostic, not sole issue criterion",
        },
        "state_counts_per_split": {k: len(v) for k, v in sorted(by_split.items())},
        "split_summary": split_summary,
        "group_summary": group_summary,
        "g2_profile_band_counts": g2_meta.get("global_band_counts", {}),
        "key_findings": key_findings,
        "evaluator_reliability": _evaluator_reliability(split_summary),  # 명세 #4
        "ar040_decision": _ar040_decision(key_findings, _evaluator_reliability(split_summary)),  # 명세 #5
        "claim_boundary": "Dataset issue presence audit only. Does not prove policy performance or final motion quality.",
        "limitations": [
            "Artifact evaluators are Category C proxies unless metric_provenance marks otherwise.",
            "Physical threshold p99 can be zero for Penetrate/Skate, so above_clean_p99 means any positive score.",
            "G2 stress/natural split is evaluator-defined; perceptual stress alignment is not proven here.",
            "evaluator_reliability grade 는 stress/natural/clean fire rate 기반 진단 (sensitivity/specificity proxy); perceptual ground-truth 아님.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== AR-043 Dataset Issue Prevalence Audit ===")
    for split in ("g2_full_pool", "g2_stress_holdout", "g2_natural_holdout", "clean_noharm_holdout", "synthetic_diag_holdout"):
        s = split_summary.get(split, {})
        if not s:
            continue
        print(f"\n{split}: n={s['n']} artifact_total_mean={s['artifact_total']['mean']:.6f} p95={s['artifact_total']['p95']:.6f}")
        for n, v in s["artifact_prevalence"].items():
            print(f"  ART {n:<24} mean={v['mean']:.6f} low+={v['low_or_above_rate']*100:5.1f}%")
        for n, v in s["physical_prevalence"].items():
            print(f"  PHY {n:<24} mean={v['mean']:.6f} >p99={v['above_clean_p99_rate']*100:5.1f}%")
        print(f"  top issues: {s['dominant_issue_top3']}")
    rel = out["evaluator_reliability"]; dec = out["ar040_decision"]
    print("\n=== evaluator reliability (명세 #4) ===")
    for fam in ("artifact", "physical"):
        for n, v in rel[fam].items():
            print(f"  {fam.upper()[:3]} {n:<24} stress={v['stress_fire']*100:5.1f}% natural={v['natural_fire']*100:5.1f}% "
                  f"clean={v['clean_fire']*100:5.1f}% → {v['grade']}")
    print(f"  reliable: {rel['reliable_evaluators']}")
    print(f"  weak: {rel['weak_evaluators']}")
    print(f"\n=== AR-040 decision (명세 #5) ===")
    print(f"  decision = {dec['decision']} (premise_holds={dec['premise_holds']})")
    print(f"  rationale: {dec['rationale']}")
    print(f"\nkey_findings: {key_findings}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
