"""AR-050 Part A: MotionGPT length-control failure 정량 audit (representative pool, read-only).

기존 representative pool 의 MotionGPT 900 sample metadata 에서 요청 길이(target_length) 대비
실제 생성 길이(actual_generated)의 불일치 분포를 집계한다. pool 원본은 변경하지 않으며, 짧은
출력은 native generation failure 로 유지(삭제·교체 금지, §연구 무결성).

CLI (motion3d env):
    python tools/motiongpt_length_audit.py \
        --pool-root external_assets/protocol_rep_pool_seed20260608 \
        --bank evals/prompts/protocol_rep_test_300_seed20260608.json \
        --output evals/snapshots/motiongpt_length_audit_v1.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path, default=REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608")
    ap.add_argument("--bank", type=Path, default=REPO_ROOT / "evals" / "prompts" / "protocol_rep_test_300_seed20260608.json")
    ap.add_argument("--generator", default="motiongpt")
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "motiongpt_length_audit_v1.json")
    args = ap.parse_args()

    bank = json.load(open(args.bank, encoding="utf-8"))
    membership = {r["sample_id"]: r["complexity"]["membership"] for r in bank["rows"]}

    d = args.pool_root / args.generator
    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(d.glob("*.json")) if p.name != "_pool_summary.json"]
    n = len(metas)

    # per-sample ratio (actual_generated = 모델이 실제 만든 raw 길이; truncation 전).
    for m in metas:
        m["ratio"] = m["actual_generated"] / m["target_length"] if m["target_length"] else 0.0

    def rate(pred):
        return round(sum(1 for m in metas if pred(m)) / n, 3)

    dist = {
        "n": n,
        "exact_match": rate(lambda m: m["actual_generated"] == m["target_length"]),
        "shorter": rate(lambda m: m["actual_generated"] < m["target_length"]),
        "longer": rate(lambda m: m["actual_generated"] > m["target_length"]),
        "ratio_below_0.8": rate(lambda m: m["ratio"] < 0.8),
        "ratio_below_0.6": rate(lambda m: m["ratio"] < 0.6),
        "ratio_below_0.25": rate(lambda m: m["ratio"] < 0.25),
        "actual_len_min": min(m["actual_generated"] for m in metas),
        "actual_len_median": int(np.median([m["actual_generated"] for m in metas])),
        "actual_len_max": max(m["actual_generated"] for m in metas),
        "ratio_median": round(float(np.median([m["ratio"] for m in metas])), 3),
    }

    # 극단 조기 종료 (<=8 frame = motion token <=2): prompt·seed·target 기록.
    extreme = sorted(
        [{"sample_id": m["sample_id"], "seed": m["seed"], "prompt": m["prompt"][:70],
          "target_length": m["target_length"], "actual_generated": m["actual_generated"],
          "trajectory_npy": m["trajectory_npy"]}
         for m in metas if m["actual_generated"] <= 8],
        key=lambda x: x["actual_generated"],
    )

    # per-prompt seed variance (동일 prompt 의 길이 변동 — 확률적 길이 결정의 직접 증거).
    by_sid = defaultdict(list)
    for m in metas:
        by_sid[m["sample_id"]].append(m["actual_generated"])
    high_var = sorted(
        [{"sample_id": s, "lengths": sorted(v), "spread": max(v) - min(v)}
         for s, v in by_sid.items() if (max(v) - min(v)) >= 50],
        key=lambda x: -x["spread"],
    )[:15]

    # complexity 관계 (탐색 분석만): ratio vs overall_top25.
    top = [m["ratio"] for m in metas if membership.get(m["sample_id"], {}).get("overall_top25")]
    bot = [m["ratio"] for m in metas if membership.get(m["sample_id"], {}).get("overall_bottom25")]
    complexity_explore = {
        "ratio_mean_overall_top25": round(float(np.mean(top)), 3) if top else None,
        "ratio_mean_overall_bottom25": round(float(np.mean(bot)), 3) if bot else None,
        "note": "탐색 분석 only — complexity 가 길이 실패의 원인이라 주장하지 않음",
    }

    out = {
        "schema_version": "1.0.0", "record_type": "motiongpt_length_audit", "board_id": "AR-050",
        "pool": str(args.pool_root.name), "generator": args.generator,
        "policy": "read-only audit. pool 원본 미변경. 짧은 출력은 native generation failure 로 유지.",
        "length_distribution": dist,
        "extreme_early_termination_count": len(extreme),
        "extreme_early_termination": extreme,
        "per_prompt_high_length_variance": high_var,
        "complexity_explore": complexity_explore,
        "root_cause_hypothesis": "Docker service 는 n_frames 를 batch 에 넣으나 MotionGPT forward() 가 batch['length'] 를 generate_direct(do_sample=True) 에 전달하지 않음 → 모델이 token 수·EOS 를 확률적으로 결정. VAE 시간 downsampling=4 → motion token 1개 조기 종료 시 4-frame 복원. (코드 확인은 Part B/C.)",
        "claim_boundary": "MotionGPT 전체 품질 낮음 증명 아님. 길이 제어 실패 발생률·원인 진단. cross-gen 측정 시 native vs sensitivity 분리.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"=== AR-050 MotionGPT length audit (n={n}, read-only) ===")
    for k, v in dist.items():
        print(f"  {k:<20} {v}")
    print(f"\n  극단 조기종료(<=8 frame): {len(extreme)} 개")
    for e in extreme[:8]:
        print(f"    {e['sample_id']} seed{e['seed']} target={e['target_length']} actual={e['actual_generated']}  {e['prompt'][:48]!r}")
    print(f"\n  동일 prompt 길이 변동 큰 것(>=50): {len(high_var)} prompt")
    for h in high_var[:5]:
        print(f"    {h['sample_id']} lengths={h['lengths']} spread={h['spread']}")
    print(f"  complexity(탐색): top25 ratio={complexity_explore['ratio_mean_overall_top25']} bottom25={complexity_explore['ratio_mean_overall_bottom25']}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
