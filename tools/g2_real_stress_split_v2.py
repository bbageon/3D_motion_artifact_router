"""Step 2 v2 (사용자 directive 2026-05-31): Stage 2-G split freeze v2 — group-balanced.

사용자 spec 박제:
  - G2 pool 600 (existing 300 + balanced 300).
  - train 300 / calib 75 / stress_holdout 100 / natural_holdout 100 / reserve 25 = 600.
  - 각 split 안에서 motion_group 분포를 user target ratio 에 맞춰 stratified.
  - clean_noharm_holdout 100 (HumanML3D clean, Stage A).
  - synthetic_aux_train + synthetic_diag_holdout (Stage B-1, auxiliary).

v1 (g2_real_stress_split.py) 의 변경점:
  - profile_v2 (group-aware bands) 입력.
  - 각 motion_group 안에서 split allocation (proportional to split size).
  - stress_holdout 은 그룹별 stress band 에서, natural_holdout 은 normal band 에서.
  - 충분한 stress 가 없는 group 에 대해 fallback + 경고 (e.g. jump 의 stress n<10).

CLI:
    python -m tools.g2_real_stress_split_v2 \
        --profile evals/snapshots/g2_real_stress_profile_v2.json \
        --output evals/splits/g2_real_stress_split_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, REAL_DISTRIBUTION, common_snapshot_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "g2_real_stress_profile_v2.json")
    parser.add_argument("--stage-a-dataset", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_dataset_stageA_v1.json")
    parser.add_argument("--stage-b1", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_hard_mining_stageB1_v1.json")
    parser.add_argument("--seed", type=int, default=20260531)
    # 사용자 spec sizes.
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-calib", type=int, default=75)
    parser.add_argument("--n-stress-holdout", type=int, default=100)
    parser.add_argument("--n-natural-holdout", type=int, default=100)
    parser.add_argument("--n-reserve", type=int, default=25)
    parser.add_argument("--synthetic-diag-frac", type=float, default=0.30)
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "splits" / "g2_real_stress_split_v2.json")
    args = parser.parse_args()

    profile = json.load(open(args.profile, encoding="utf-8"))
    per_motion = profile["per_motion"]
    print(f"[INFO] profile v2: {len(per_motion)} motions, groups={profile['global_group_counts']}")
    # group by motion_group + band.
    by_grp_band = defaultdict(lambda: defaultdict(list))
    for r in per_motion:
        by_grp_band[r["motion_group"]][r["band"]].append(r["trial_id"])
    motion_groups = sorted(by_grp_band.keys())

    # Total G2 split size (must match pool size after summing target buckets).
    n_total = args.n_train + args.n_calib + args.n_stress_holdout + args.n_natural_holdout + args.n_reserve
    pool_size = len(per_motion)
    if n_total != pool_size:
        print(f"[WARN] sum of split sizes ({n_total}) != pool size ({pool_size}). split 은 user spec 사이즈 유지, 남은 motion = reserve_overflow.")

    rng = np.random.default_rng(args.seed)

    # Per-group quotas: each split's size × (group's share of pool).
    group_share = {g: sum(len(by_grp_band[g][b]) for b in by_grp_band[g]) / max(pool_size, 1)
                   for g in motion_groups}
    quotas = {}
    for split_name, n in (("train_g2_real", args.n_train), ("calib_g2_real", args.n_calib),
                          ("g2_stress_holdout", args.n_stress_holdout),
                          ("g2_natural_holdout", args.n_natural_holdout),
                          ("reserve", args.n_reserve)):
        quotas[split_name] = {g: int(round(n * group_share[g])) for g in motion_groups}

    # Allocation v2: **train 에도 stress 가 들어가도록** band 별 비율 분배 (사용자 directive
    # "G2 real stress 가 RL 학습/평가의 핵심"). user spec 사이즈는 근사 (band 분포에 맞춰 조정).
    #   stress band: 50% stress_holdout, 20% calib, 30% train.
    #   normal band: 28% natural_holdout, 14% calib, 50% train, 8% reserve.
    #   clean_like band: 100% train.
    BAND_ALLOC = {
        "stress": [("g2_stress_holdout", 0.50), ("calib_g2_real", 0.20), ("train_g2_real", 0.30)],
        "normal": [("g2_natural_holdout", 0.28), ("calib_g2_real", 0.14), ("train_g2_real", 0.50), ("reserve", 0.08)],
        "g2_clean_like": [("train_g2_real", 1.0)],
    }
    splits = {s: [] for s in ("train_g2_real", "calib_g2_real", "g2_stress_holdout",
                              "g2_natural_holdout", "reserve")}
    shortage = defaultdict(dict)

    def _portions(pool: list, fracs: list[tuple[str, float]]) -> list[tuple[str, list[str]]]:
        """결정론적 shuffle → cumulative integer counts 로 분배 (sum to len(pool))."""
        rng.shuffle(pool)
        N = len(pool); pieces = []
        # cumulative round to preserve all motions.
        cum = 0.0; cum_int = 0
        for i, (name, frac) in enumerate(fracs):
            cum += N * frac
            new_cum_int = N if i == len(fracs) - 1 else int(round(cum))
            n_this = new_cum_int - cum_int
            pieces.append((name, pool[cum_int:cum_int + n_this]))
            cum_int = new_cum_int
        return pieces

    for g in motion_groups:
        for band, fracs in BAND_ALLOC.items():
            pool = list(by_grp_band[g].get(band, []))
            if not pool:
                continue
            for split_name, ids in _portions(pool, fracs):
                splits[split_name].extend(ids)

    # Sort + de-dup defensively.
    for k in splits:
        splits[k] = sorted(set(splits[k]))

    # Disjointness assertion.
    all_g2 = set()
    for k in splits:
        ids = set(splits[k])
        dup = all_g2 & ids
        assert not dup, f"G2 split overlap in {k}: {list(dup)[:5]}"
        all_g2 |= ids

    # Group-band crosstab of final splits.
    motion_meta = {r["trial_id"]: r for r in per_motion}
    split_group_band = {k: defaultdict(lambda: defaultdict(int)) for k in splits}
    for k, ids in splits.items():
        for tid in ids:
            r = motion_meta[tid]
            split_group_band[k][r["motion_group"]][r["band"]] += 1

    # Clean no-harm holdout (Stage A).
    clean_noharm_holdout = []
    if args.stage_a_dataset.exists():
        sa = json.load(open(args.stage_a_dataset, encoding="utf-8"))
        clean_noharm_holdout = sorted({r["sample_id"] for r in sa["rows"] if r["distribution"] == "clean"})

    # Synthetic aux/diag (Stage B-1).
    syn_aux_train, syn_diag_holdout = [], []
    if args.stage_b1.exists():
        b1 = json.load(open(args.stage_b1, encoding="utf-8"))
        syn_ids = sorted(b1["states"].keys())
        rng2 = np.random.default_rng(args.seed + 1)
        rng2.shuffle(syn_ids)
        n_diag = int(round(len(syn_ids) * args.synthetic_diag_frac))
        syn_diag_holdout = sorted(syn_ids[:n_diag])
        syn_aux_train = sorted(syn_ids[n_diag:])

    splits_meta = {
        "train_g2_real": {"source": "G2 600 pool", "n": len(splits["train_g2_real"]),
                          "ids": splits["train_g2_real"], "evidence_tier": REAL_DISTRIBUTION,
                          "use": "주 학습", "group_band_breakdown": {g: dict(c) for g, c in split_group_band["train_g2_real"].items()}},
        "calib_g2_real": {"source": "G2 600 pool", "n": len(splits["calib_g2_real"]),
                          "ids": splits["calib_g2_real"], "evidence_tier": REAL_DISTRIBUTION,
                          "use": "P_safe calibration", "group_band_breakdown": {g: dict(c) for g, c in split_group_band["calib_g2_real"].items()}},
        "g2_stress_holdout": {"source": "G2 600 pool (group-wise stress band)",
                              "n": len(splits["g2_stress_holdout"]),
                              "ids": splits["g2_stress_holdout"], "evidence_tier": REAL_DISTRIBUTION,
                              "use": "일반화 테스트 (각 motion_group 의 stress)",
                              "group_band_breakdown": {g: dict(c) for g, c in split_group_band["g2_stress_holdout"].items()}},
        "g2_natural_holdout": {"source": "G2 600 pool (group-wise normal band)",
                               "n": len(splits["g2_natural_holdout"]),
                               "ids": splits["g2_natural_holdout"], "evidence_tier": REAL_DISTRIBUTION,
                               "use": "over-correction 방지 (각 group 의 normal)",
                               "group_band_breakdown": {g: dict(c) for g, c in split_group_band["g2_natural_holdout"].items()}},
        "reserve": {"source": "G2 600 pool", "n": len(splits["reserve"]),
                    "ids": splits["reserve"], "evidence_tier": REAL_DISTRIBUTION,
                    "use": "reserve (재시도 / 향후 확장)",
                    "group_band_breakdown": {g: dict(c) for g, c in split_group_band["reserve"].items()}},
        "clean_noharm_holdout": {"source": "HumanML3D clean (Stage A)",
                                 "n": len(clean_noharm_holdout), "ids": clean_noharm_holdout,
                                 "evidence_tier": REAL_DISTRIBUTION, "use": "no-harm 테스트"},
        "synthetic_aux_train": {"source": "Stage B-1 hard-mined synthetic_severe",
                                "n": len(syn_aux_train), "ids": syn_aux_train,
                                "evidence_tier": CONTROLLED_DIAGNOSTIC,
                                "use": "auxiliary boundary augmentation"},
        "synthetic_diag_holdout": {"source": "Stage B-1 hard-mined synthetic_severe",
                                   "n": len(syn_diag_holdout), "ids": syn_diag_holdout,
                                   "evidence_tier": CONTROLLED_DIAGNOSTIC,
                                   "use": "appendix diagnostic"},
    }

    out = {
        "schema_version": "1.0.0", "record_type": "g2_real_stress_split_v2",
        "task_id": "g2_real_stress_split_v2",
        **common_snapshot_metadata(
            split_id=args.split_id or "g2_real_stress_split_v2",
            oracle_type="N/A (split definition)", action_grid="N/A",
            stage="Stage-2G-step2-split-freeze-v2",
            evidence_tier=[REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
            evaluators=list(DEFAULT_EVALUATORS), gate_evaluators=list(DEFAULT_PHYSICAL_GATE_EVALUATORS),
        ),
        "directive": "Stage 2-G split v2 — group-balanced stratified, user spec sizing.",
        "g2_profile_source": str(args.profile.relative_to(REPO_ROOT)),
        "seed": args.seed,
        "spec_sizes": {"train": args.n_train, "calib": args.n_calib,
                       "stress_holdout": args.n_stress_holdout,
                       "natural_holdout": args.n_natural_holdout,
                       "reserve": args.n_reserve, "n_total": n_total},
        "group_quotas": quotas,
        "shortage_per_group_per_split": {g: dict(d) for g, d in shortage.items()},
        "splits": splits_meta,
        "totals": {"n_g2_total": sum(len(splits[k]) for k in splits),
                   "n_clean_total": len(clean_noharm_holdout),
                   "n_synthetic_total": len(syn_aux_train) + len(syn_diag_holdout)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== Stage 2-G Split Freeze v2 (group-balanced) ===")
    print(f"  G2 pool: {pool_size} motions")
    for k, meta in splits_meta.items():
        n_g2 = meta["n"]
        suffix = ""
        if "group_band_breakdown" in meta:
            gb = meta["group_band_breakdown"]
            tot = sum(sum(b.values()) for b in gb.values())
            stress_n = sum(b.get("stress", 0) for b in gb.values())
            normal_n = sum(b.get("normal", 0) for b in gb.values())
            clean_n = sum(b.get("g2_clean_like", 0) for b in gb.values())
            suffix = f"  bands(stress/normal/clean_like) = {stress_n}/{normal_n}/{clean_n}"
        print(f"  {k:<26} n={n_g2:<4}  ({meta['evidence_tier']}){suffix}")
    if shortage:
        print(f"\n  [WARN] shortages (per_group/per_split): {len([s for s in shortage.values() if s])} groups affected")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
