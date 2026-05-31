"""Step 2 (사용자 directive 2026-05-31): Stage 2-G Split Freeze.

사용자 directive 박제 (data 역할):
> "데이터를 motion/state 단위로 고정한다. 핵심: synthetic 은 auxiliary, G2 real stress 가 중심."

권장 split (사용자 표):
  - train_g2_real      : G2 normal + stress + clean_like (주 학습)
  - calib_g2_real      : G2 stress 일부 (P_safe calibration)
  - g2_stress_holdout  : G2 high-artifact (일반화 테스트)
  - g2_natural_holdout : G2 normal 일부 (over-correction 방지)
  - clean_noharm_holdout : HumanML3D clean (no-harm 테스트)
  - synthetic_aux_train : Stage B-1 일부 (보조 boundary augmentation)
  - synthetic_diag_holdout : Stage B-1 일부 (appendix diagnostic)

본 도구는 Step 1 의 G2 stress profile + Stage A clean ids + Stage B-1 synthetic ids 를 통합 →
deterministic split freeze JSON 산출. 한 motion id 는 정확히 하나의 split 에 속함 (단 G2 와
synthetic 의 source 가 다르므로 HumanML3D 의 같은 trial id 가 clean/synthetic 두 종류 데이터
의 source 일 수는 있음 — 본 도구가 별도 보고).

CLI:
    python -m tools.g2_real_stress_split \
        --profile evals/snapshots/g2_real_stress_profile_v1.json \
        --stage-a-dataset evals/snapshots/rl2_transition_dataset_stageA_v1.json \
        --stage-b1 evals/snapshots/rl2_hard_mining_stageB1_v1.json \
        --seed 20260531 \
        --output evals/splits/g2_real_stress_split_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.harness_metadata import CONTROLLED_DIAGNOSTIC, REAL_DISTRIBUTION, common_snapshot_metadata
from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS


def _deterministic_split(ids: list[str], frac: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    ids = sorted(ids)
    rng.shuffle(ids)
    n_hold = int(round(len(ids) * frac))
    return ids[:n_hold], ids[n_hold:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "g2_real_stress_profile_v1.json")
    parser.add_argument("--stage-a-dataset", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_dataset_stageA_v1.json")
    parser.add_argument("--stage-b1", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_hard_mining_stageB1_v1.json")
    parser.add_argument("--seed", type=int, default=20260531)
    # G2 split fractions.
    parser.add_argument("--calib-frac-of-stress", type=float, default=0.15,
                        help="fraction of G2 stress → calib_g2_real")
    parser.add_argument("--g2-stress-holdout-frac", type=float, default=0.15,
                        help="fraction of G2 stress → g2_stress_holdout")
    parser.add_argument("--g2-natural-holdout-frac", type=float, default=0.20,
                        help="fraction of G2 normal → g2_natural_holdout")
    parser.add_argument("--synthetic-diag-frac", type=float, default=0.30,
                        help="fraction of Stage B-1 ids → synthetic_diag_holdout")
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "splits" / "g2_real_stress_split_v1.json")
    args = parser.parse_args()

    profile = json.load(open(args.profile, encoding="utf-8"))
    g2_pool_dir = profile["g2_pool_dir"]
    g2_motions = {r["trial_id"]: r for r in profile["per_motion"]}
    print(f"[INFO] G2 profile: {len(g2_motions)} motions, bands={profile['summary']['band_counts']}")

    # G2 by band.
    g2_stress = sorted([t for t, r in g2_motions.items() if r["band"] == "stress"])
    g2_normal = sorted([t for t, r in g2_motions.items() if r["band"] == "normal"])
    g2_cleanlike = sorted([t for t, r in g2_motions.items() if r["band"] == "g2_clean_like"])

    # Split G2 stress: calib + holdout + rest → train.
    rng = np.random.default_rng(args.seed)
    stress_shuffled = sorted(g2_stress); rng.shuffle(stress_shuffled)
    n_calib = max(1, int(round(len(stress_shuffled) * args.calib_frac_of_stress)))
    n_str_ho = max(1, int(round(len(stress_shuffled) * args.g2_stress_holdout_frac)))
    calib_g2_real = sorted(stress_shuffled[:n_calib])
    g2_stress_holdout = sorted(stress_shuffled[n_calib:n_calib + n_str_ho])
    g2_stress_train = sorted(stress_shuffled[n_calib + n_str_ho:])

    # Split G2 normal: holdout + train.
    normal_shuffled = sorted(g2_normal); rng.shuffle(normal_shuffled)
    n_nat_ho = max(1, int(round(len(normal_shuffled) * args.g2_natural_holdout_frac)))
    g2_natural_holdout = sorted(normal_shuffled[:n_nat_ho])
    g2_normal_train = sorted(normal_shuffled[n_nat_ho:])

    train_g2_real = sorted(g2_stress_train + g2_normal_train + g2_cleanlike)

    # Clean no-harm holdout: Stage A 의 clean state_ids 100.
    clean_noharm_holdout = []
    if args.stage_a_dataset.exists():
        sa = json.load(open(args.stage_a_dataset, encoding="utf-8"))
        clean_noharm_holdout = sorted({r["sample_id"] for r in sa["rows"] if r["distribution"] == "clean"})
        print(f"[INFO] Stage A clean ids → clean_noharm_holdout: {len(clean_noharm_holdout)}")

    # Synthetic aux/diag from Stage B-1 (160 ids).
    synthetic_aux_train, synthetic_diag_holdout = [], []
    syn_ids = []
    if args.stage_b1.exists():
        b1 = json.load(open(args.stage_b1, encoding="utf-8"))
        syn_ids = sorted(b1["states"].keys())
        diag, aux = _deterministic_split(syn_ids, args.synthetic_diag_frac, args.seed + 1)
        synthetic_diag_holdout = sorted(diag)
        synthetic_aux_train = sorted(aux)
        print(f"[INFO] Stage B-1 synthetic ids: {len(syn_ids)} → aux_train {len(synthetic_aux_train)} + diag_holdout {len(synthetic_diag_holdout)}")

    # Source-id overlap audit (HumanML3D 의 같은 id 가 clean (Stage A) AND synthetic (Stage B-1) 의 source 일 수 있음).
    overlap_clean_syn = sorted(set(clean_noharm_holdout) & set(syn_ids))

    splits = {
        "train_g2_real": {
            "source": "G2 generated", "n": len(train_g2_real),
            "ids": train_g2_real,
            "composition": {"stress": len(g2_stress_train), "normal": len(g2_normal_train), "g2_clean_like": len(g2_cleanlike)},
            "evidence_tier": REAL_DISTRIBUTION,
            "use": "주 학습 데이터 (RL/Q policy training)",
        },
        "calib_g2_real": {
            "source": "G2 generated (stress subset)", "n": len(calib_g2_real),
            "ids": calib_g2_real, "evidence_tier": REAL_DISTRIBUTION,
            "use": "P_safe calibration (isotonic/Platt). train 에 미포함.",
        },
        "g2_stress_holdout": {
            "source": "G2 generated (stress subset)", "n": len(g2_stress_holdout),
            "ids": g2_stress_holdout, "evidence_tier": REAL_DISTRIBUTION,
            "use": "일반화 테스트 — train/calib 에 미포함된 high-artifact G2.",
        },
        "g2_natural_holdout": {
            "source": "G2 generated (normal subset)", "n": len(g2_natural_holdout),
            "ids": g2_natural_holdout, "evidence_tier": REAL_DISTRIBUTION,
            "use": "over-correction 방지 테스트 — train 에 미포함된 일반 G2.",
        },
        "clean_noharm_holdout": {
            "source": "HumanML3D clean (Stage A 의 clean ids)", "n": len(clean_noharm_holdout),
            "ids": clean_noharm_holdout, "evidence_tier": REAL_DISTRIBUTION,
            "use": "no-harm 테스트 — clean motion 에 correction 이 해로운지.",
        },
        "synthetic_aux_train": {
            "source": "Stage B-1 hard-mined synthetic_severe (subset)", "n": len(synthetic_aux_train),
            "ids": synthetic_aux_train, "evidence_tier": CONTROLLED_DIAGNOSTIC,
            "use": "보조 boundary augmentation (auxiliary, NOT 중심).",
        },
        "synthetic_diag_holdout": {
            "source": "Stage B-1 hard-mined synthetic_severe (subset)", "n": len(synthetic_diag_holdout),
            "ids": synthetic_diag_holdout, "evidence_tier": CONTROLLED_DIAGNOSTIC,
            "use": "appendix diagnostic — synthetic regime 의 holdout 진단.",
        },
    }

    # Disjointness check within G2.
    all_g2 = set()
    for k in ("train_g2_real", "calib_g2_real", "g2_stress_holdout", "g2_natural_holdout"):
        ids = set(splits[k]["ids"])
        dup = all_g2 & ids
        assert not dup, f"G2 split overlap: {k} has dup ids {dup}"
        all_g2 |= ids
    # Disjointness check within synthetic.
    syn_a = set(synthetic_aux_train); syn_d = set(synthetic_diag_holdout)
    assert not (syn_a & syn_d), "synthetic_aux_train ∩ synthetic_diag_holdout not empty"

    out = {
        "schema_version": "1.0.0", "record_type": "g2_real_stress_split",
        "task_id": "g2_real_stress_split_v1",
        **common_snapshot_metadata(
            split_id=args.split_id or "g2_real_stress_split_v1",
            oracle_type="N/A (split definition)",
            action_grid="N/A",
            stage="Stage-2G-step2-split-freeze",
            evidence_tier=[REAL_DISTRIBUTION, CONTROLLED_DIAGNOSTIC],
            evaluators=list(DEFAULT_EVALUATORS), gate_evaluators=list(DEFAULT_PHYSICAL_GATE_EVALUATORS),
        ),
        "directive": "G2 real stress 중심 RL 일반화 split freeze. synthetic 은 auxiliary.",
        "g2_pool_dir": g2_pool_dir,
        "g2_profile_source": str(args.profile.relative_to(REPO_ROOT)),
        "seed": args.seed,
        "splits": splits,
        "totals": {
            "n_g2_total": sum(splits[k]["n"] for k in ("train_g2_real", "calib_g2_real", "g2_stress_holdout", "g2_natural_holdout")),
            "n_clean_total": len(clean_noharm_holdout),
            "n_synthetic_total": len(synthetic_aux_train) + len(synthetic_diag_holdout),
        },
        "source_id_overlap_audit": {
            "description": "HumanML3D 의 같은 trial_id 가 clean (Stage A) AND synthetic_severe (Stage B-1) 의 source 일 수 있음. 두 종류 의 state 는 DIFFERENT (clean = 무손상; synthetic = 손상 적용). split disjointness 는 보장되나 source motion id overlap 은 정보적 보고.",
            "n_overlap_clean_vs_synthetic_source_ids": len(overlap_clean_syn),
            "overlap_ids_sample": overlap_clean_syn[:10],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== Stage 2-G Split Freeze ===")
    for name, s in splits.items():
        comp = f"  composition={s.get('composition')}" if "composition" in s else ""
        print(f"  {name:<26} n={s['n']:<4} ({s['evidence_tier']}){comp}")
    print(f"\n  source_id_overlap clean (Stage A) ∩ synthetic (B-1 source ids) = {len(overlap_clean_syn)}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
