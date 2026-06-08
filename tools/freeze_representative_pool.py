"""AR-022/045: representative-300 shared pool freeze (keep ALL 300, no floor reselection).

사용자 directive (board): 대표 300 끝까지 유지 — semantic floor 로 재선별·보충 금지.
생성 실패·길이 불일치도 **결과로 보고**. generator별 success/length 분리 기록. 전체 300 primary,
complexity threshold subgroup 은 보조 분석. AR-048 의 floor-selection freeze 와 달리 prompt 를
하나도 떨어뜨리지 않는다.

paired 비교용 per-prompt common_length = 3 generator × 3 seed 의 min(final_length) (보고만, drop 안함).

CLI (motion3d env):
    python tools/freeze_representative_pool.py \
        --pool-root external_assets/protocol_rep_pool_seed20260608 \
        --bank evals/prompts/protocol_rep_test_300_seed20260608.json \
        --output evals/snapshots/representative_pool_freeze_v1.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_metas(pool_root: Path, gens: list[str]) -> dict[str, list[dict]]:
    out = {}
    for g in gens:
        d = pool_root / g
        out[g] = [json.load(open(p, encoding="utf-8")) for p in sorted(d.glob("*.json"))
                  if p.name != "_pool_summary.json"] if d.exists() else []
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path,
                    default=REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608")
    ap.add_argument("--bank", type=Path,
                    default=REPO_ROOT / "evals" / "prompts" / "protocol_rep_test_300_seed20260608.json")
    ap.add_argument("--generators", nargs="+", default=["mdm", "momask", "motiongpt"])
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "representative_pool_freeze_v1.json")
    args = ap.parse_args()
    args.pool_root = args.pool_root.resolve()
    args.bank = args.bank.resolve()

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    bank = json.load(open(args.bank, encoding="utf-8"))
    bank_rows = bank["rows"]
    bank_by_id = {r["sample_id"]: r for r in bank_rows}
    prompt_ids = [r["sample_id"] for r in bank_rows]  # ALL 300, kept
    metas = _load_metas(args.pool_root, args.generators)

    by_prompt = defaultdict(list)
    for g, ms in metas.items():
        for m in ms:
            by_prompt[m["sample_id"]].append(m)

    checks = {}

    # 1. prompt/GT alignment (전체 300).
    mis = [f"{g}/{m['sample_id']}" for g, ms in metas.items() for m in ms
           if bank_by_id.get(m["sample_id"]) is None
           or m["prompt"] != bank_by_id[m["sample_id"]]["prompt"]
           or m["gt_length"] != bank_by_id[m["sample_id"]]["gt_length"]]
    checks["1_prompt_gt_alignment"] = {"pass": len(mis) == 0, "n_mismatch": len(mis)}

    # 2. length REPORT (NOT a selection gate): per-generator match rate + common_length 분포.
    per_gen_len = {}
    for g, ms in metas.items():
        n = len(ms)
        match = sum(1 for m in ms if m.get("length_matches_target"))
        shorter = sum(1 for m in ms if m["actual_generated"] < m["target_length"])
        longer = sum(1 for m in ms if m["actual_generated"] > m["target_length"])
        per_gen_len[g] = {"n": n, "target_match": match, "match_rate": round(match / n, 3) if n else 0,
                          "shorter_than_target": shorter, "longer_than_target": longer}
    common_lengths = {sid: int(min(m["final_length"] for m in by_prompt[sid]))
                      for sid in prompt_ids if by_prompt.get(sid)}
    cl_vals = np.array(list(common_lengths.values()))
    ratios = np.array([common_lengths[sid] / bank_by_id[sid]["target_length"] for sid in common_lengths])
    checks["2_length_report"] = {
        "pass": True,  # 보고용 — drop/select 안함. 항상 기록 통과.
        "policy": "대표 300 전부 유지. MDM/MoMask=target 준수, MotionGPT=native(강제 불가). 길이 불일치는 결과로 보고.",
        "per_generator": per_gen_len,
        "common_length": {"min": int(cl_vals.min()), "median": int(np.median(cl_vals)), "max": int(cl_vals.max()),
                          "ratio_to_target_min": round(float(ratios.min()), 2),
                          "ratio_to_target_median": round(float(np.median(ratios)), 2),
                          "n_ratio_below_0.6": int((ratios < 0.6).sum()),
                          "n_ratio_below_0.8": int((ratios < 0.8).sum())},
        "note": "paired Category-A 비교는 per-prompt common_length 로 truncate. complexity subgroup·전체 보고 시 함께 명시.",
    }

    # 3. seeds per prompt.
    bad3 = []
    for g, ms in metas.items():
        by_sid = defaultdict(set)
        for m in ms:
            by_sid[m["sample_id"]].add(m["seed"])
        for sid in prompt_ids:
            if len(by_sid.get(sid, set())) != args.n_seeds:
                bad3.append(f"{g}/{sid}={len(by_sid.get(sid, set()))}")
    checks["3_seeds_per_prompt"] = {"pass": len(bad3) == 0, "expected": args.n_seeds, "n_bad": len(bad3)}

    # 4. dual representation + 정합.
    miss = [f"{g}/{m['sample_id']}__{m['seed']}" for g, ms in metas.items() for m in ms
            if not (REPO_ROOT / m["trajectory_npy"]).exists()
            or not (REPO_ROOT / m["local_npy"]).exists()
            or not m.get("local_matches_trajectory")]
    checks["4_dual_representation"] = {"pass": len(miss) == 0, "n_bad": len(miss)}

    # 5. generator 분리.
    contam = [f"{g}:{m.get('generator')}" for g, ms in metas.items() for m in ms if m.get("generator") != g]
    gen_ids = {g: sorted({m.get("generator_id") for m in ms if m.get("generator_id")}) for g, ms in metas.items()}
    allids = [i for v in gen_ids.values() for i in v]
    checks["5_generator_separation"] = {"pass": len(contam) == 0 and len(allids) == len(set(allids)),
                                        "generator_ids": gen_ids}

    # 6. metadata/provenance + per-generator success rate (분리 기록).
    need = ["sample_id", "prompt", "seed", "generator_id", "target_length", "ground_y"]
    bad6 = [1 for g, ms in metas.items() for m in ms if any(m.get(k) in (None, "") for k in need)]
    success = {g: {"ok": len(ms), "expected": len(prompt_ids) * args.n_seeds} for g, ms in metas.items()}
    checks["6_metadata_provenance"] = {"pass": len(bad6) == 0, "per_generator_success": success}

    # 7. paired 완전성 (전체 300 × 3 gen × n_seeds).
    incomplete = [f"{sid}/{g}={sum(1 for m in metas[g] if m['sample_id'] == sid)}"
                  for sid in prompt_ids for g in args.generators
                  if sum(1 for m in metas[g] if m["sample_id"] == sid) != args.n_seeds]
    checks["7_paired_complete"] = {"pass": len(incomplete) == 0, "n_incomplete": len(incomplete)}

    # complexity subgroup (bank annotation 의 사전 고정 quartile membership) — 보조 분석 setup.
    subgroups = defaultdict(int)
    for r in bank_rows:
        for k, v in r.get("complexity", {}).get("membership", {}).items():
            if v:
                subgroups[k] += 1
    checks["8_complexity_subgroups_present"] = {
        "pass": all(r.get("complexity", {}).get("membership") for r in bank_rows),
        "subgroup_sizes": dict(subgroups),
        "role": "보조(조건부) 분석 — 전체 300 primary 대체 금지",
    }

    all_pass = all(c["pass"] for c in checks.values())
    out = {
        "schema_version": "1.0.0", "record_type": "representative_pool_freeze", "board_id": "AR-022/AR-045",
        "pool_root": _rel(args.pool_root), "bank": _rel(args.bank),
        "n_prompts": len(prompt_ids), "n_seeds": args.n_seeds, "generators": args.generators,
        "total_motions": sum(len(ms) for ms in metas.values()),
        "policy": "대표 300 전부 유지 (floor 재선별·보충 없음). 길이 불일치·실패는 결과로 보고.",
        "all_pass": all_pass,
        "checks": checks,
        "per_prompt_common_length": common_lengths,
        "ground_criterion": "lower-foot 10th pct (≠minY); tools/coords_protocol.estimate_ground",
        "evaluation_protocol": "Category-A(FID/R-Prec) = 전체 300 동일 N + per-prompt common_length truncate + paired/bootstrap CI. Category-B/C = per-sample. complexity subgroup = 보조.",
        "claim_boundary": "AR-048 적격 HumanML3D test representative-300. 절대 FID 단독 금지. generator 우열은 measurement 후.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"=== representative-300 pool freeze ({out['total_motions']} motions, 전부 유지) ===")
    for k, c in checks.items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {k}")
    print(f"\n  ALL_PASS = {all_pass}")
    print("\n  -- length 결과 (drop 없음) --")
    for g, v in per_gen_len.items():
        print(f"    {g:<10} target-match {v['target_match']}/{v['n']} ({v['match_rate']:.0%})  "
              f"shorter={v['shorter_than_target']} longer={v['longer_than_target']}")
    cl = checks["2_length_report"]["common_length"]
    print(f"    common_length: median={cl['median']} min={cl['min']}  ratio<0.8: {cl['n_ratio_below_0.8']}/300  ratio<0.6: {cl['n_ratio_below_0.6']}/300")
    print(f"\n  complexity subgroup sizes: {dict(subgroups)}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
