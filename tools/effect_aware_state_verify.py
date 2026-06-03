"""AR-037 verify — Effect-Aware state schema 를 실제 G2 motion 에 적용해 검증.

성공 조건 (명세):
1. v0/v1/v2 feature schema 고정 (effect_aware_state.SCHEMA 단일 출처).
2. 각 feature 가 before-action observable 인지 확인 (금지 feature 배제 + history/uncertainty 사유).
3. state dimension 보고.
4. v0/v1/v2 ablation 이 같은 split/candidate/q_proxy 로 비교 가능 — 본 검증은 builder 가
   동일 motion 에서 결정적 vector 를 만든다는 것 확인 (q_proxy 는 다음 작업).
5. target-aware claim 은 v1 이후 — v0 는 target 없이, v1 은 sample target 으로 build 확인.

semantic block (text-motion similarity 등) 은 mgpt env tm2t 필요 → 본 motion-router 검증에선
NaN placeholder (schema 위치 고정). 실제 값은 다음 작업의 별도 mgpt pass.

CLI:
    python -m tools.effect_aware_state_verify --n 30 \
        --output evals/snapshots/effect_aware_state_schema_v0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluators import DEFAULT_EVALUATORS, DEFAULT_PHYSICAL_GATE_EVALUATORS
from tools.harness_metadata import REAL_DISTRIBUTION, common_snapshot_metadata
from tools.g2_balanced_prompt_selector import classify, GROUPS_ORDER
from tools.effect_aware_state import (
    SCHEMA, ACTION_SCHEMA, FORBIDDEN_FEATURES, feature_names, state_dim,
    build_v0, build_v1_add, encode_action, verify_schema,
)

GROUP_ID = {g: i for i, (g, _) in enumerate(GROUPS_ORDER)}
GROUP_ID["other"] = len(GROUPS_ORDER)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--calibration", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_clean_calibration_v1.json")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "effect_aware_state_schema_v0.json")
    args = parser.parse_args()

    calib = json.load(open(args.calibration, encoding="utf-8"))
    gate_thr = {n: calib["summary"][n]["p99"] for n in calib["summary"] if calib["summary"][n].get("n", 0) > 0}
    eval_by_name = {ev.name: ev for ev in list(DEFAULT_EVALUATORS) + list(DEFAULT_PHYSICAL_GATE_EVALUATORS)}

    vcheck = verify_schema()
    print("=== AR-037 schema self-verify ===")
    print(f"  dims v0/v1/v2 = {vcheck['n_v0']}/{vcheck['n_v1']}/{vcheck['n_v2']} (state), "
          f"{vcheck['n_v0_action']}/{vcheck['n_v1_action']}/{vcheck['n_v2_action']} (state+action)")
    print(f"  forbidden leaked: {vcheck['forbidden_leaked']} | suspicious: {vcheck['after_action_suspicious']} | passed: {vcheck['passed']}")

    files = sorted(args.pool_dir.glob("motion_*.npy"))[:args.n]
    print(f"\n[INFO] building v0/v1 on {len(files)} G2 motions ...")
    v0_names = feature_names("v0"); v1_names = feature_names("v1")
    nan_counts_v0 = {f: 0 for f in v0_names}
    v1_nan_counts: dict[str, int] = {}
    n_built = 0
    sample_v0 = None
    for p in files:
        m = np.load(str(p)).astype(np.float64)
        if m.ndim != 3 or m.shape[1] != 22:
            continue
        meta = json.load(open(p.with_suffix(".json"), encoding="utf-8")) if p.with_suffix(".json").exists() else {}
        grp = classify(meta.get("prompt", ""))
        gid = GROUP_ID.get(grp, GROUP_ID["other"])
        v0 = build_v0(m, motion_group_id=gid, evaluators_by_name=eval_by_name, gate_thresholds=gate_thr)
        # v1: sample target = both feet, full frame range.
        v1add = build_v1_add(m, target_part="both_feet", target_joints=["LEFT_FOOT", "RIGHT_FOOT"],
                             frame_range=(0, m.shape[0]-1),
                             evaluators_by_name=eval_by_name, gate_thresholds=gate_thr)
        for f, val in v1add.items():
            if isinstance(val, float) and np.isnan(val):
                v1_nan_counts[f] = v1_nan_counts.get(f, 0) + 1
        # dim check.
        assert len(v0) == state_dim("v0"), f"v0 dim {len(v0)} != {state_dim('v0')}"
        assert len(v0) + len(v1add) == state_dim("v1"), "v1 dim mismatch"
        for f in v0_names:
            if np.isnan(v0[f]):
                nan_counts_v0[f] += 1
        if sample_v0 is None:
            sample_v0 = {k: (None if np.isnan(v) else round(float(v), 5)) for k, v in v0.items()}
        n_built += 1

    # NaN audit: semantic block 은 NaN 예상 (mgpt precompute 전), 나머지는 NaN 0 이어야.
    semantic_feats = set(SCHEMA["v0"]["semantic_motion_state"])
    nonsem_nan = {f: c for f, c in nan_counts_v0.items() if c > 0 and f not in semantic_feats}
    sem_nan = {f: c for f, c in nan_counts_v0.items() if c > 0 and f in semantic_feats}

    # Block dims.
    block_dims = {}
    for blk, feats in SCHEMA["v0"].items():
        block_dims[f"v0/{blk}"] = len(feats)
    for blk, feats in SCHEMA["v1_add"].items():
        block_dims[f"v1/{blk}"] = len(feats)
    for blk, feats in SCHEMA["v2_add"].items():
        block_dims[f"v2/{blk}"] = len(feats)
    block_dims["action"] = sum(len(v) for v in ACTION_SCHEMA.values())

    out = {
        "schema_version": "1.0.0", "record_type": "effect_aware_state_schema",
        "task_id": "AR-037-effect-aware-state-design",
        **common_snapshot_metadata(
            split_id="effect_aware_state_schema_v0", oracle_type="N/A (state schema)",
            action_grid="bounded_continuous_u", stage="AR-037-state-design",
            evidence_tier=[REAL_DISTRIBUTION],
            evaluators=list(DEFAULT_EVALUATORS), gate_evaluators=list(DEFAULT_PHYSICAL_GATE_EVALUATORS),
        ),
        "spec": ".claude/docs/dashboard-task-specs/AR-037-effect-aware-state-design.md",
        "claim_boundary": "state 설계 고정만 — 정책 성능 증명 아님. q_proxy/effect-label/oracle 은 다음 작업.",
        "dimensions": {
            "v0_state": state_dim("v0"), "v0_state_action": state_dim("v0", True),
            "v1_state": state_dim("v1"), "v1_state_action": state_dim("v1", True),
            "v2_state": state_dim("v2"), "v2_state_action": state_dim("v2", True),
            "action": block_dims["action"],
        },
        "spec_target_ranges": {  # 명세 stated ranges (참고 — explicit feature list 가 authority)
            "v0_state": "40-55", "v0_state_action": "55-70",
            "v1_state": "70-95", "v1_state_action": "85-110",
            "v2_state": "80-110", "v2_state_action": "95-125",
            "note": "actual v1/v2 가 stated 하한 약간 미만 — explicit feature list (명세 §v1/v2) 합이 authority. dim 목표가 아니라 schema 고정이 성공 조건.",
        },
        "block_dims": block_dims,
        "schema": {  # 고정 schema 단일 출처 박제.
            "v0": SCHEMA["v0"], "v1_add": SCHEMA["v1_add"], "v2_add": SCHEMA["v2_add"],
            "action": ACTION_SCHEMA,
        },
        "observability_audit": {
            "forbidden_features_excluded": sorted(FORBIDDEN_FEATURES),
            "forbidden_leaked": vcheck["forbidden_leaked"],
            "after_action_suspicious": vcheck["after_action_suspicious"],
            "before_action_ambiguous_justified": vcheck["before_action_ambiguous_justified"],
            "state_action_name_collisions": vcheck["state_action_name_collisions"],
            "namespace_note": "action feature = action_ prefix (state block 이름과 분리, merge/vectorize 충돌 방지).",
            "passed": vcheck["passed"],
        },
        "build_verification": {
            "n_motions_built": n_built,
            "v0_dim_assert": "passed" if n_built else "no data",
            "nonsemantic_nan_counts": nonsem_nan,  # should be empty (all computable)
            "semantic_nan_counts": sem_nan,         # expected nonzero (mgpt precompute 전)
            "v1_add_nan_counts": v1_nan_counts,     # v1 placeholder NaN (이제 0 이어야 — local eval/relation 계산)
            "semantic_note": "semantic block = mgpt env tm2t precompute 필요 → 본 검증 NaN placeholder (schema 위치 고정). 다음 작업의 별도 pass.",
            "target_encoding_note": "target_type_encoding = TARGET_TYPE_ID 고정 dict (hash 금지, 재현성).",
        },
        "sample_v0_state": sample_v0,
        "ablation_plan": {
            "v0": "global quality-aware ranker (action=(tool,u))",
            "v1": "target-aware quality-gain ranker (action=(tool,target,u)) — target-aware claim 은 여기부터",
            "v2": "support-aware constrained ranker (action=(tool,target,u)) — OOD 과대평가 억제",
            "comparable": "같은 split / 같은 candidate set / 같은 q_proxy 로 v0<v1<v2 비교 (q_proxy=다음 작업)",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== build verification (n={n_built}) ===")
    print(f"  v0 dim assert: passed | block dims: {block_dims}")
    print(f"  non-semantic NaN (should be empty): {nonsem_nan}")
    print(f"  semantic NaN (expected, mgpt 전): { {k:v for k,v in sem_nan.items()} }")
    print(f"\n[OK] schema snapshot -> {args.output}")


if __name__ == "__main__":
    main()
