"""AR-063: 수리된 gate evaluator (v0.2.0) 로 representative pool 유병률 재측정.

AR-062 audit 에서 Skate/Penetrate gate 가 구조적 vacuity (내부 contact/ground 로는
fire 불가) 로 판명 → AR-043/044 의 해당 컬럼은 "없음"의 증거가 아니었다. 본 도구는
수리된 evaluator 로 **처음으로 유효한** foot-계 gate 유병률을 산출한다.

좌표 (AR-048 protocol + AR-062 A-4): foot skate/contact/ground 는 **trajectory
(world)** 에서 측정 — local(root-relative) 은 root 이동이 수평변위에 혼입.
비교 참고용으로 local 측정도 병기 (좌표 선택의 효과 크기 확인용).

Metric: score>0 비율(발생률) + score 평균 + clean-p99(v2) 초과 비율(gate-fire).
통계 단위 = prompt (seed 평균, n=300/generator), bootstrap CI.

CLI (motion3d env):
    python tools/gate_prevalence_remeasure_ar063.py \
        --calibration evals/snapshots/physical_gate_clean_calibration_v2.json \
        --output evals/snapshots/gate_prevalence_remeasure_ar063_v1.json
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

from evaluators import FloatEvaluator, PenetrateEvaluator, SkateEvaluator
from evaluators.physical_gate import SEVERITY_VERSION

EVALS = {"PenetrateEvaluator": PenetrateEvaluator(),
         "FloatEvaluator": FloatEvaluator(),
         "SkateEvaluator": SkateEvaluator()}
SPACES = ("trajectory", "local")


def _boot_ci(vals, rng, n=1000):
    nv = len(vals)
    bs = [float(np.mean(vals[rng.integers(0, nv, nv)])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path,
                    default=REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608")
    ap.add_argument("--generators", nargs="+", default=["motiongpt", "mdm", "momask"])
    ap.add_argument("--calibration", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v2.json")
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "gate_prevalence_remeasure_ar063_v1.json")
    args = ap.parse_args()
    args.pool_root = args.pool_root.resolve()

    rng = np.random.default_rng(args.seed)
    calib = json.load(open(args.calibration, encoding="utf-8"))
    # calibration snapshot 구조: {"summary": {name: {"p99": ...}}}.
    p99 = {name: float(calib["summary"][name]["p99"]) for name in EVALS}

    results = {}
    for gen in args.generators:
        d = args.pool_root / gen
        metas = [json.load(open(p, encoding="utf-8")) for p in sorted(d.glob("*.json"))
                 if p.name != "_pool_summary.json"]
        if args.limit:
            metas = metas[:args.limit]
        # space -> evaluator -> stat -> sample_id -> [values per seed]
        acc = {sp: {name: {"occur": defaultdict(list), "score": defaultdict(list),
                           "fire": defaultdict(list)} for name in EVALS} for sp in SPACES}
        for m in metas:
            arrs = {"trajectory": np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64),
                    "local": np.load(REPO_ROOT / m["local_npy"]).astype(np.float64)}
            for sp in SPACES:
                for name, ev in EVALS.items():
                    score = max((r.score for r in ev.evaluate(arrs[sp])), default=0.0)
                    a = acc[sp][name]
                    a["occur"][m["sample_id"]].append(float(score > 0.0))
                    a["score"][m["sample_id"]].append(float(score))
                    thr = p99[name]
                    a["fire"][m["sample_id"]].append(float(score > thr) if np.isfinite(thr) else float("nan"))
        gen_res = {}
        for sp in SPACES:
            sp_res = {}
            for name in EVALS:
                a = acc[sp][name]
                row = {}
                for stat in ("occur", "score", "fire"):
                    pp = np.array([float(np.mean(v)) for v in a[stat].values()])
                    if np.isnan(pp).any():
                        row[stat] = None
                        continue
                    row[stat] = {"mean": round(float(pp.mean()), 5), "ci95": _boot_ci(pp, rng)}
                sp_res[name] = row
            gen_res[sp] = sp_res
        results[gen] = {"n_prompts": len({m["sample_id"] for m in metas}), "by_space": gen_res}
        tr = gen_res["trajectory"]
        print(f"[{gen}] traj occur%: " + " ".join(
            f"{n.replace('Evaluator','')}={tr[n]['occur']['mean']:.3f}" for n in EVALS))

    out = {
        "schema_version": "1.0.0",
        "record_type": "gate_prevalence_remeasure",
        "board_id": "AR-063",
        "split_id": "protocol_rep_pool_seed20260608 (representative-300 full, 3 seeds/prompt)",
        "severity_versions": {"physical_gate": SEVERITY_VERSION},
        "calibration_ref": str(args.calibration.name),
        "clean_p99_v2": {k: (None if not np.isfinite(v) else round(v, 6)) for k, v in p99.items()},
        "coordinate_note": "primary=trajectory (AR-048/AR-062 A-4: 수평변위·ground 는 world 에서만 의미). "
                           "local 병기는 좌표 선택의 효과 확인용 참고.",
        "stat_unit": "prompt (seed 평균; n=300/gen). occur=score>0 비율, fire=clean-p99(v2) 초과 비율.",
        "results": results,
        "retroactive_caveat": "AR-043/044 의 Skate_gate/Penetrate 컬럼(v0.1.0)은 vacuous — 본 v2 측정으로 대체. "
                              "구 컬럼은 '없음'의 근거로 인용 금지 (AR-062 audit A-1/A-2).",
        "claim_boundary": "유병률 기술 통계 (도구 효과 아님). Category-C gate proxy — 외부 최종 성능 근거 금지 (§3-20).",
        "grounding": ["PhysDiff ICCV2023", "GMD ICCV2023", "HuMoR ICCV2021"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
