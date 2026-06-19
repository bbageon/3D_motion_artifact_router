"""AR-054: select 100 artifact examples per motion generation framework.

This builds a qualitative artifact library from the frozen representative pool:

  - MotionGPT: FootFloating + length-control failures
  - MDM: world-space foot-skate
  - MoMask: FootFloating/contact instability

The selected cases are qualitative examples, not a prevalence estimate or final
generator ranking.

CLI:
    python tools/generator_artifact_100case_select.py
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

from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from tools.g2_balanced_prompt_selector import classify
from tools.physical_metric_g2_stress import accel_stats
from tools.representative_pool_measure import ARTIFACT_THRESH, PHYS_P99, foot_skate_world


GENERATOR_BUCKET_QUOTAS: dict[str, dict[str, int]] = {
    "motiongpt": {
        "footfloating": 45,
        "length_control_failure": 20,
        "high_artifact_total": 15,
        "footskate": 10,
        "mixed_visible": 10,
    },
    "mdm": {
        "world_footskate": 55,
        "trajectory_contact_drift": 15,
        "footfloating": 10,
        "jitter_smoothness": 10,
        "mixed_visible": 10,
    },
    "momask": {
        "footfloating": 45,
        "complex_pose_contact": 20,
        "rare_footskate": 10,
        "temporal_smoothness": 10,
        "mixed_visible": 15,
    },
}

GROUP_QUOTAS = {
    "walk_locomotion": 16,
    "run_jog": 10,
    "turn_path": 12,
    "jump_hop": 12,
    "kick_squat_lunge": 12,
    "dance_rhythmic": 8,
    "sit_stand_transition": 10,
    "upper_body_low_loco": 10,
    "other": 10,
}

GENERATOR_TITLES = {
    "motiongpt": "MotionGPT",
    "mdm": "MDM",
    "momask": "MoMask",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.load(open(path, encoding="utf-8"))


def _max_score(reports) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _shortage_score(case: dict[str, Any], group_counts: Counter[str], bucket_counts: Counter[str]) -> float:
    group = case["motion_group"]
    bucket = case["artifact_bucket"]
    group_need = max(GROUP_QUOTAS.get(group, 0) - group_counts[group], 0)
    bucket_need = max(case["bucket_quota"] - bucket_counts[bucket], 0)
    return float(group_need * 10 + bucket_need)


def _candidate_score(case: dict[str, Any]) -> float:
    bucket = case["artifact_bucket"]
    if bucket in {"footfloating", "complex_pose_contact"}:
        return float(case["metrics"]["footfloating_score"])
    if bucket in {"world_footskate", "footskate", "rare_footskate", "trajectory_contact_drift"}:
        return float(case["metrics"]["foot_skate_world"])
    if bucket == "length_control_failure":
        return float(abs(1.0 - case["metrics"]["length_ratio"]))
    if bucket in {"jitter_smoothness", "temporal_smoothness"}:
        return float(case["metrics"]["accel"])
    return float(case["metrics"]["artifact_total"])


def _artifact_bucket(gen: str, meta: dict[str, Any], metrics: dict[str, float], membership: dict[str, bool]) -> str:
    length_bad = (
        not bool(meta.get("length_matches_target", True))
        or metrics["length_ratio"] < 0.8
        or metrics["length_ratio"] > 1.25
    )
    if gen == "motiongpt":
        if length_bad:
            return "length_control_failure"
        if metrics["footfloating_score"] >= 0.05:
            return "footfloating"
        if metrics["foot_skate_world"] >= 0.01:
            return "footskate"
        if metrics["artifact_total"] >= 0.04 or metrics["accel"] >= 0.008:
            return "high_artifact_total"
        return "mixed_visible"

    if gen == "mdm":
        if metrics["foot_skate_world"] >= 0.012:
            return "world_footskate"
        if metrics["foot_skate_world"] >= 0.006:
            return "trajectory_contact_drift"
        if metrics["footfloating_score"] >= 0.05:
            return "footfloating"
        if metrics["accel"] >= 0.0065:
            return "jitter_smoothness"
        return "mixed_visible"

    if gen == "momask":
        if metrics["footfloating_score"] >= 0.05:
            return "footfloating"
        if membership.get("fine_top25") or membership.get("overall_top25"):
            return "complex_pose_contact"
        if metrics["foot_skate_world"] >= 0.006:
            return "rare_footskate"
        if metrics["accel"] >= 0.0065:
            return "temporal_smoothness"
        return "mixed_visible"

    return "mixed_visible"


def _iter_generator_cases(pool_root: Path, bank_rows: dict[str, dict[str, Any]], generator: str):
    art_evs = {ev.name: ev for ev in DEFAULT_EVALUATORS if ev.name in ARTIFACT_THRESH}
    phy_evs = {ev.name: ev for ev in DEFAULT_PHYSICAL_GATE_EVALUATORS if ev.name in PHYS_P99}
    for meta_path in sorted((pool_root / generator).glob("*.json")):
        if meta_path.name.startswith("_"):
            continue
        meta = _load_json(meta_path)
        sid = meta["sample_id"]
        bank = bank_rows.get(sid, {})
        traj = np.load(REPO_ROOT / meta["trajectory_npy"]).astype(np.float64)
        local = np.load(REPO_ROOT / meta["local_npy"]).astype(np.float64)

        art = {name: _max_score(ev.evaluate(local)) for name, ev in art_evs.items()}
        phy = {name: _max_score(ev.evaluate(local)) for name, ev in phy_evs.items()}
        target_len = max(int(meta.get("target_length", local.shape[0])), 1)
        actual_len = int(meta.get("actual_generated", local.shape[0]))
        metrics = {
            "footfloating_score": float(art.get("FootFloatingEvaluator", 0.0)),
            "bone_length_score": float(art.get("BoneLengthEvaluator", 0.0)),
            "velocity_jitter_score": float(art.get("VelocityJitterEvaluator", 0.0)),
            "artifact_total": float(np.mean(list(art.values()))) if art else 0.0,
            "foot_skate_world": float(foot_skate_world(traj)),
            "accel": float(accel_stats(local)[0]),
            "length_ratio": float(actual_len / target_len),
            "final_length": int(meta.get("final_length", local.shape[0])),
            "target_length": target_len,
            "actual_generated": actual_len,
            "float_gate_fire": float(phy.get("FloatEvaluator", 0.0) > PHYS_P99["FloatEvaluator"]),
            "skate_gate_fire": float(phy.get("SkateEvaluator", 0.0) > PHYS_P99["SkateEvaluator"]),
            "bonecv_gate_fire": float(phy.get("BoneLengthCVEvaluator", 0.0) > PHYS_P99["BoneLengthCVEvaluator"]),
        }
        membership = bank.get("complexity", {}).get("membership", {})
        bucket = _artifact_bucket(generator, meta, metrics, membership)
        yield {
            "generator": generator,
            "sample_id": sid,
            "seed": int(meta["seed"]),
            "seed_idx": int(meta.get("seed_idx", -1)),
            "prompt": meta["prompt"],
            "motion_group": classify(meta["prompt"]),
            "artifact_bucket": bucket,
            "bucket_quota": GENERATOR_BUCKET_QUOTAS[generator].get(bucket, 0),
            "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
            "complexity": bank.get("complexity", {}),
            "trajectory_npy": meta["trajectory_npy"],
            "local_npy": meta["local_npy"],
            "ground_y": round(float(meta.get("ground_y", 0.0)), 6),
            "length_matches_target": bool(meta.get("length_matches_target", True)),
        }


def _select_for_generator(cases: list[dict[str, Any]], generator: str, n: int) -> list[dict[str, Any]]:
    bucket_quotas = GENERATOR_BUCKET_QUOTAS[generator]
    selected: list[dict[str, Any]] = []
    used_samples: set[str] = set()
    bucket_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()

    def _eligible(case: dict[str, Any], enforce_group: bool = True) -> bool:
        if case["sample_id"] in used_samples:
            return False
        if enforce_group and group_counts[case["motion_group"]] >= GROUP_QUOTAS.get(case["motion_group"], 0):
            return False
        return True

    while len(selected) < n:
        remaining = [c for c in cases if _eligible(c, enforce_group=True)]
        if not remaining:
            remaining = [c for c in cases if _eligible(c, enforce_group=False)]
        if not remaining:
            break
        remaining.sort(
            key=lambda c: (
                _shortage_score(c, group_counts, bucket_counts),
                max(bucket_quotas.get(c["artifact_bucket"], 0) - bucket_counts[c["artifact_bucket"]], 0),
                _candidate_score(c),
            ),
            reverse=True,
        )
        case = dict(remaining[0])
        case["selection_rank_within_bucket"] = bucket_counts[case["artifact_bucket"]] + 1
        case["selection_score"] = round(_candidate_score(case), 6)
        if group_counts[case["motion_group"]] >= GROUP_QUOTAS.get(case["motion_group"], 0):
            case["selection_note"] = "group_quota_overflow_fill"
        selected.append(case)
        used_samples.add(case["sample_id"])
        bucket_counts[case["artifact_bucket"]] += 1
        group_counts[case["motion_group"]] += 1

    selected = selected[:n]
    for i, case in enumerate(selected, start=1):
        case["selection_index"] = i
    return selected


def _summary_for(selected: list[dict[str, Any]], all_cases: list[dict[str, Any]]) -> dict[str, Any]:
    def _mean(key: str, rows: list[dict[str, Any]]) -> float:
        return round(float(np.mean([r["metrics"][key] for r in rows])), 6) if rows else 0.0

    available_unique_by_group = {
        group: len({c["sample_id"] for c in all_cases if c["motion_group"] == group})
        for group in GROUP_QUOTAS
    }
    selected_group_counts = Counter(c["motion_group"] for c in selected)
    group_shortage = {
        group: max(GROUP_QUOTAS[group] - selected_group_counts.get(group, 0), 0)
        for group in GROUP_QUOTAS
    }
    return {
        "n_selected": len(selected),
        "n_candidates": len(all_cases),
        "bucket_counts": dict(Counter(c["artifact_bucket"] for c in selected)),
        "motion_group_counts": dict(selected_group_counts),
        "available_unique_prompts_by_group": available_unique_by_group,
        "group_target_quotas": GROUP_QUOTAS,
        "group_shortage_vs_target": {k: v for k, v in group_shortage.items() if v > 0},
        "unique_prompts": len({c["sample_id"] for c in selected}),
        "metric_means_selected": {
            "footfloating_score": _mean("footfloating_score", selected),
            "foot_skate_world": _mean("foot_skate_world", selected),
            "artifact_total": _mean("artifact_total", selected),
            "accel": _mean("accel", selected),
            "length_ratio": _mean("length_ratio", selected),
        },
        "top_preview_candidates": [
            {
                "generator": c["generator"],
                "sample_id": c["sample_id"],
                "seed": c["seed"],
                "prompt": c["prompt"],
                "artifact_bucket": c["artifact_bucket"],
                "motion_group": c["motion_group"],
                "selection_score": c["selection_score"],
            }
            for c in selected[:24]
        ],
    }


def _write_summary_md(out_dir: Path, manifests: dict[str, list[dict[str, Any]]], summaries: dict[str, Any]) -> None:
    lines = [
        "# AR-054 Generator Artifact 100-Case Library",
        "",
        "작성일: 2026-06-19",
        "",
        "이 문서는 representative pool에서 수집한 generator별 artifact 예시 100개를 요약한다.",
        "이 세트는 정성적 오류 라이브러리이며, generator 성능 순위나 refinement 성능 claim이 아니다.",
        "",
        "## Generator별 요약",
        "",
        "| MGF | n | 주요 bucket | 주요 동작군 | 비고 |",
        "|---|---:|---|---|---|",
    ]
    for gen in ("motiongpt", "mdm", "momask"):
        s = summaries[gen]
        top_bucket = ", ".join(f"{k}:{v}" for k, v in Counter(s["bucket_counts"]).most_common(3))
        top_group = ", ".join(f"{k}:{v}" for k, v in Counter(s["motion_group_counts"]).most_common(3))
        note = {
            "motiongpt": "FootFloating + length-control failure 중심",
            "mdm": "world-space foot-skate 중심",
            "momask": "FootFloating/contact instability 중심",
        }[gen]
        lines.append(f"| {GENERATOR_TITLES[gen]} | {s['n_selected']} | {top_bucket} | {top_group} | {note} |")

    lines += [
        "",
        "## 산출물",
        "",
        "- `artifact_100_manifest_motiongpt.json`",
        "- `artifact_100_manifest_mdm.json`",
        "- `artifact_100_manifest_momask.json`",
        "- `manifest.json`",
        "",
        "## Preview 후보",
        "",
        "각 generator별 상위 24개는 GIF preview 후보로 사용할 수 있다. 전체 100개 GIF는 파일 수와 용량이 크므로, 먼저 preview 검증 후 생성한다.",
        "",
        "## 동작군 shortage",
        "",
        "대표 pool에는 dance prompt가 unique 기준 5개뿐이다. seed 중복을 피했기 때문에 `dance_rhythmic`은 목표 8개를 채우지 않고 5개로 보고한다.",
    ]
    for gen in ("motiongpt", "mdm", "momask"):
        lines += ["", f"### {GENERATOR_TITLES[gen]}"]
        for c in manifests[gen][:10]:
            lines.append(
                f"- #{c['selection_index']:03d} `{c['artifact_bucket']}` / `{c['motion_group']}` "
                f"/ {c['sample_id']} seed {c['seed']} — {c['prompt']}"
            )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path, default=REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608")
    ap.add_argument("--bank", type=Path, default=REPO_ROOT / "evals" / "prompts" / "protocol_rep_test_300_seed20260608.json")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "figures" / "2026-06-19" / "ar054_generator_artifact_100cases")
    ap.add_argument("--snapshot", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "generator_artifact_100case_library_v1.json")
    ap.add_argument("--n-per-generator", type=int, default=100)
    args = ap.parse_args()

    bank = _load_json(args.bank)
    bank_rows = {r["sample_id"]: r for r in bank["rows"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for gen in ("motiongpt", "mdm", "momask"):
        print(f"[INFO] measuring/selecting {gen}")
        cases = list(_iter_generator_cases(args.pool_root, bank_rows, gen))
        selected = _select_for_generator(cases, gen, args.n_per_generator)
        manifests[gen] = selected
        summaries[gen] = _summary_for(selected, cases)
        out_path = args.output_dir / f"artifact_100_manifest_{gen}.json"
        out_path.write_text(json.dumps({
            "schema_version": "1.0.0",
            "record_type": "generator_artifact_100case_manifest",
            "board_id": "AR-054",
            "generator": gen,
            "pool": args.pool_root.name,
            "bank": args.bank.name,
            "claim_boundary": "Qualitative artifact example library only. Not generator ranking or refinement-performance evidence.",
            "bucket_quotas": GENERATOR_BUCKET_QUOTAS[gen],
            "group_target_quotas": GROUP_QUOTAS,
            "summary": summaries[gen],
            "cases": selected,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {gen}: selected={len(selected)} -> {out_path}")

    manifest = {
        "schema_version": "1.0.0",
        "record_type": "generator_artifact_100case_library",
        "board_id": "AR-054",
        "pool": args.pool_root.name,
        "bank": args.bank.name,
        "n_per_generator": args.n_per_generator,
        "generators": ["motiongpt", "mdm", "momask"],
        "claim_boundary": "Qualitative artifact example library only. Not generator ranking, prevalence estimate, or refinement-performance evidence.",
        "selection_rules": {
            "unit": "one generated motion instance: (generator, sample_id, seed)",
            "source": "frozen representative pool only",
            "prompt_duplicate_policy": "avoid duplicate prompt within each bucket when possible",
            "preview_policy": "top 24 per generator can be used for GIF preview before full 100 GIF export",
        },
        "bucket_quotas": GENERATOR_BUCKET_QUOTAS,
        "group_target_quotas": GROUP_QUOTAS,
        "summaries": summaries,
        "manifest_files": {
            gen: str((args.output_dir / f"artifact_100_manifest_{gen}.json").resolve().relative_to(REPO_ROOT))
            for gen in manifests
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary_md(args.output_dir, manifests, summaries)
    print(f"[OK] wrote {args.output_dir / 'manifest.json'}")
    print(f"[OK] wrote {args.output_dir / 'summary.md'}")
    print(f"[OK] snapshot {args.snapshot}")


if __name__ == "__main__":
    main()
