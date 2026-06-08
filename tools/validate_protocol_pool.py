"""AR-048: protocol freeze gate — semantic-floor selection + per-prompt common length + 7 checks.

common-min length 로 paired 비교 시 의미 손실(긴 generator 가 과도하게 잘림)을 막기 위해
ratio = common_min/target ≥ floor 인 prompt 만 채택(candidate bank 순서로 n_select 개).
채택 prompt 의 per-prompt common_length 를 기록 → 분석 시 3 generator 를 이 길이로 truncate
하면 동일 길이 paired (§6-10) 이면서 의미 보존(≥ floor·target).

CLI (motion3d env):
    python tools/validate_protocol_pool.py \
        --pool-root external_assets/protocol_v1_pool_seed20260607 \
        --candidate-bank evals/prompts/protocol_v1_test_50_seed20260607.json \
        --n-select 30 --ratio-floor 0.8 --n-seeds 3 \
        --output evals/snapshots/protocol_freeze_v1.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_metas(pool_root: Path, generators: list[str]) -> dict[str, list[dict]]:
    out = {}
    for gen in generators:
        d = pool_root / gen
        out[gen] = [json.load(open(p, encoding="utf-8")) for p in sorted(d.glob("*.json"))
                    if p.name != "_pool_summary.json"] if d.exists() else []
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path,
                    default=REPO_ROOT / "external_assets" / "protocol_v1_pool_seed20260607")
    ap.add_argument("--candidate-bank", type=Path,
                    default=REPO_ROOT / "evals" / "prompts" / "protocol_v1_test_50_seed20260607.json")
    ap.add_argument("--n-select", type=int, default=30)
    ap.add_argument("--ratio-floor", type=float, default=0.8)
    ap.add_argument("--generators", nargs="+", default=["mdm", "momask", "motiongpt"])
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "protocol_freeze_v1.json")
    args = ap.parse_args()
    args.pool_root = args.pool_root.resolve()
    args.candidate_bank = args.candidate_bank.resolve()

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    bank = json.load(open(args.candidate_bank, encoding="utf-8"))
    bank_rows = bank["rows"]
    bank_by_id = {r["sample_id"]: r for r in bank_rows}
    metas = _load_metas(args.pool_root, args.generators)

    # group all motions by prompt; per-prompt completeness + common_min + ratio.
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for gen, ms in metas.items():
        for m in ms:
            by_prompt[m["sample_id"]].append(m)

    per_prompt = {}
    for sid, ms in by_prompt.items():
        complete = all(sum(1 for m in ms if m["generator"] == g) == args.n_seeds for g in args.generators)
        target = ms[0]["target_length"]
        cmin = min(m["final_length"] for m in ms)
        per_prompt[sid] = {"complete": complete, "target": target,
                           "common_min": cmin, "ratio": cmin / target if target else 0.0}

    # selection: candidate-bank order, complete + ratio >= floor, until n_select.
    selected: list[str] = []
    for r in bank_rows:
        sid = r["sample_id"]
        pp = per_prompt.get(sid)
        if pp and pp["complete"] and pp["ratio"] >= args.ratio_floor:
            selected.append(sid)
        if len(selected) >= args.n_select:
            break
    enough = len(selected) >= args.n_select
    selected = selected[: args.n_select]
    sel_set = set(selected)
    common_lengths = {sid: per_prompt[sid]["common_min"] for sid in selected}

    # restrict metas to selected prompts.
    sel_metas = {g: [m for m in metas[g] if m["sample_id"] in sel_set] for g in args.generators}
    checks = {}

    # 1. prompt/GT 정렬.
    mis = [f"{g}/{m['sample_id']}" for g, ms in sel_metas.items() for m in ms
           if bank_by_id.get(m["sample_id"]) is None
           or m["prompt"] != bank_by_id[m["sample_id"]]["prompt"]
           or m["gt_length"] != bank_by_id[m["sample_id"]]["gt_length"]]
    checks["1_prompt_gt_alignment"] = {"pass": len(mis) == 0, "n_mismatch": len(mis), "examples": mis[:5]}

    # 2. 길이 규약: n_select 충족 + 채택 prompt 전부 ratio>=floor + MDM/MoMask target 준수.
    sel_ratios = {sid: round(per_prompt[sid]["ratio"], 3) for sid in selected}
    mdmmomask_target = {}
    for g in args.generators:
        if g == "motiongpt":
            continue
        ok = sum(1 for m in sel_metas[g] if m.get("length_matches_target"))
        mdmmomask_target[g] = {"n": len(sel_metas[g]), "match": ok}
    checks["2_length_protocol"] = {
        "pass": bool(enough and all(per_prompt[s]["ratio"] >= args.ratio_floor for s in selected)
                     and all(v["match"] == v["n"] for v in mdmmomask_target.values())),
        "ratio_floor": args.ratio_floor, "n_selected": len(selected), "n_requested": args.n_select,
        "enough": enough, "min_selected_ratio": min(sel_ratios.values()) if sel_ratios else None,
        "mdm_momask_target_match": mdmmomask_target,
        "note": "per-prompt common_length 로 truncate 시 3 generator 동일 길이 + 의미 보존(ratio>=floor). MotionGPT 는 native length(강제 불가) → common_length 가 MotionGPT length.",
    }

    # 3. seed n개/prompt.
    bad3 = []
    for g, ms in sel_metas.items():
        by_sid = defaultdict(set)
        for m in ms:
            by_sid[m["sample_id"]].add(m["seed"])
        for sid in selected:
            if len(by_sid.get(sid, set())) != args.n_seeds:
                bad3.append(f"{g}/{sid}={len(by_sid.get(sid, set()))}")
    checks["3_seeds_per_prompt"] = {"pass": len(bad3) == 0, "expected": args.n_seeds, "bad": bad3[:5]}

    # 4. trajectory/local 모두 존재·정합.
    miss = [f"{g}/{m['sample_id']}__{m['seed']}" for g, ms in sel_metas.items() for m in ms
            if not (REPO_ROOT / m["trajectory_npy"]).exists()
            or not (REPO_ROOT / m["local_npy"]).exists()
            or not m.get("local_matches_trajectory")]
    checks["4_dual_representation"] = {"pass": len(miss) == 0, "n_bad": len(miss), "examples": miss[:5]}

    # 5. generator 분리.
    contam = [f"{g}dir:{m.get('generator')}" for g, ms in sel_metas.items() for m in ms
              if m.get("generator") != g]
    gen_ids = {g: sorted({m.get("generator_id") for m in ms if m.get("generator_id")})
               for g, ms in sel_metas.items()}
    allids = [i for v in gen_ids.values() for i in v]
    checks["5_generator_separation"] = {"pass": len(contam) == 0 and len(allids) == len(set(allids)),
                                        "generator_ids": gen_ids, "contamination": contam[:5]}

    # 6. metadata + provenance(generator_id).
    need = ["sample_id", "prompt", "seed", "generator_id", "target_length", "ground_y"]
    bad6 = [f"{g}/{m['sample_id']}__{m['seed']}" for g, ms in sel_metas.items() for m in ms
            if any(m.get(k) in (None, "") for k in need)]
    checks["6_metadata_provenance"] = {"pass": len(bad6) == 0, "required": need, "n_bad": len(bad6)}

    # 7. paired 완전성 (선택 prompt 전부 3 gen × n_seeds).
    incomplete = [f"{sid}/{g}={sum(1 for m in sel_metas[g] if m['sample_id'] == sid)}"
                  for sid in selected for g in args.generators
                  if sum(1 for m in sel_metas[g] if m["sample_id"] == sid) != args.n_seeds]
    checks["7_paired_complete"] = {"pass": len(incomplete) == 0, "incomplete": incomplete[:5]}

    all_pass = enough and all(c["pass"] for c in checks.values())
    out = {
        "schema_version": "1.0.0", "record_type": "protocol_freeze_validation", "board_id": "AR-048",
        "pool_root": _rel(args.pool_root), "candidate_bank": _rel(args.candidate_bank),
        "n_candidates": len(bank_rows), "n_select": args.n_select, "ratio_floor": args.ratio_floor,
        "n_seeds": args.n_seeds, "generators": args.generators,
        "selected_prompts": selected,
        "per_prompt_common_length": common_lengths,
        "selected_ratios": sel_ratios,
        "ground_criterion": "lower-foot-height 10th percentile (NOT min-Y); tools/coords_protocol.estimate_ground",
        "length_protocol": "MDM/MoMask honor target; MotionGPT native length; paired compare at per-prompt common_length (>= floor*target).",
        "all_pass": all_pass,
        "checks": checks,
        "claim_boundary": "표현/provenance/length adapter freeze. generator 성능 우열 결론 아님.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"=== AR-048 protocol freeze (selected {len(selected)}/{args.n_select}, floor={args.ratio_floor}) ===")
    print(f"  candidates examined: {len(bank_rows)}, complete+pass-floor selected: {len(selected)}")
    print(f"  min selected ratio: {min(sel_ratios.values()) if sel_ratios else 'n/a'}")
    for k, c in checks.items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {k}")
    print(f"\n  ALL_PASS = {all_pass}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
