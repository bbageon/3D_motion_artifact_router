"""AR-037 — Effect-Aware Ranker State Ablation (v0/v1/v2 schema 고정 + builder).

명세: .claude/docs/dashboard-task-specs/AR-037-effect-aware-state-design.md

Claim Boundary (명세): 본 작업은 정책 성능을 증명하지 않는다. state 설계를 **고정** 해
이후 q_proxy / effect-vector label / M1 oracle / M2 learned ranker 실험이 비교 가능하게
만드는 준비 단계다.

핵심 원칙 (명세):
1. state(=action 적용 **전** 관측) 와 label(=적용 **후** 결과) 분리. 본 모듈은 state 만.
2. v0 = global quality-aware ranker (target-aware 아님). v1+ 부터 target-aware.
3. raw 512/768/1024 embedding 직접 투입 금지 → similarity/distance/confidence scalar 만.
4. FID/R-Prec/MM-Dist 는 최종 validation. 단 text-motion similarity 같은 per-sample scalar 는 state 후보.

State ablation (명세):
  v0 = global_motion + global_artifact + global_physical + semantic + history   (action (tool,u))
  v1 = v0 + local_target + relation                                              (action (tool,target,u))
  v2 = v1 + data_support / uncertainty                                           (action (tool,target,u))

본 모듈은 **schema 단일 출처** + before-action observable builder. 금지 feature (적용 후 score,
gate_result, quality_gain, dense oracle action, split name, id 등) 는 SCHEMA 에 절대 미포함.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from skeleton_normalizer.canonical_smpl_22 import NAME_TO_IDX

PELVIS = NAME_TO_IDX["PELVIS"]; HEAD = NAME_TO_IDX["HEAD"]
LFOOT = NAME_TO_IDX["LEFT_FOOT"]; RFOOT = NAME_TO_IDX["RIGHT_FOOT"]
FPS = 20.0
CONTACT_Y_THRESH = 0.05  # foot 'contact' 판정 height (root-relative, Y-up).

# === SCHEMA (단일 출처, 명세 §v0/v1/v2 그대로) ===
SCHEMA: dict[str, dict[str, list[str]]] = {
    "v0": {
        "global_motion_state": [
            "T_norm", "duration_sec", "root_path_length", "root_displacement",
            "root_speed_mean", "root_speed_std", "root_speed_p95",
            "joint_velocity_mean", "joint_velocity_p95", "joint_acceleration_p95",
            "joint_jerk_p95", "left_contact_ratio", "right_contact_ratio",
            "motion_group_id",
        ],
        "global_artifact_state": [
            "foot_floating_score", "foot_sliding_score", "velocity_jitter_score",
            "jerk_spike_score", "artifact_total_score", "dominant_artifact_type",
            "artifact_severity_band",
        ],
        "global_physical_state": [
            "bone_length_cv_mean", "bone_length_cv_max", "penetration_score",
            "physical_load_mean", "physical_load_max", "gate_margin_min",
            "dominant_physical_risk",
        ],
        "semantic_motion_state": [
            "text_motion_similarity", "text_motion_distance", "semantic_confidence",
            "prompt_motion_group_match",
        ],
        "correction_history_state": [
            "step_index", "last_tool_encoding", "last_u", "cumulative_joint_delta",
            "cumulative_root_delta", "cumulative_fidelity_loss", "previous_accepted_gain",
            "rejected_count",
        ],
    },
    "v1_add": {
        "local_target_state": [
            "target_type_encoding", "target_joint_count", "target_frame_start_norm",
            "target_frame_end_norm", "target_frame_length_norm", "local_artifact_score",
            "local_physical_score", "local_velocity_p95", "local_acceleration_p95",
            "local_jerk_p95", "local_foot_height_mean", "local_foot_height_min",
            "local_foot_velocity_p95", "local_contact_ratio", "local_penetration_score",
            "local_bone_cv_max",
        ],
        "relation_state": [
            "affected_joint_ratio", "affected_frame_ratio", "root_target_distance_mean",
            "root_target_velocity_mismatch", "parent_chain_bone_margin", "balance_proxy",
            "contact_phase_consistency", "expected_fidelity_risk_proxy",
        ],
    },
    "v2_add": {
        "data_support_state": [
            "knn_state_distance", "knn_state_action_distance", "motion_group_support_count",
            "tool_target_u_support_count", "ensemble_gain_variance", "ensemble_safe_variance",
            "ood_flag",
        ],
    },
}

ACTION_SCHEMA: dict[str, list[str]] = {
    "tool": ["tool_encoding", "tool_family", "tool_cost"],
    "target": ["target_type_encoding", "target_side", "target_chain_id"],
    "strength": ["u", "u_sq", "is_continuous_u"],
    "stop": ["is_stop_action"],
}

# 명세 §금지 Feature — SCHEMA 에 절대 포함 금지 (verify 가 강제).
FORBIDDEN_FEATURES = {
    "sample_id", "trial_id", "pool_dir", "split", "split_name", "split_id",
    "gate_result", "quality_after", "quality_gain", "after_state",
    "dense_oracle_action", "final_standard_metric", "fid", "r_precision", "mm_dist",
}

ARTIFACT_EVALS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
PHYSICAL_EVALS = ("PenetrateEvaluator", "FloatEvaluator", "SkateEvaluator",
                  "JerkSpikeEvaluator", "BoneLengthCVEvaluator")


def feature_names(version: str, include_action: bool = False) -> list[str]:
    """version ∈ {v0, v1, v2}. 명세 ablation 순서대로 ordered feature 이름."""
    blocks = list(SCHEMA["v0"].values())
    if version in ("v1", "v2"):
        blocks += list(SCHEMA["v1_add"].values())
    if version == "v2":
        blocks += list(SCHEMA["v2_add"].values())
    names = [f for blk in blocks for f in blk]
    if include_action:
        names += [f for blk in ACTION_SCHEMA.values() for f in blk]
    return names


def state_dim(version: str, include_action: bool = False) -> int:
    return len(feature_names(version, include_action))


# === before-action observable 계산 helper ===
def _max_score(reports) -> float:
    return float(max((r.score for r in reports), default=0.0))


def _percentile(a, p): return float(np.percentile(a, p)) if len(a) else 0.0


def _global_motion(motion: np.ndarray, motion_group_id: int) -> dict:
    T = motion.shape[0]
    root = motion[:, PELVIS, :]  # (T,3)
    root_xz = root[:, [0, 2]]
    root_step = np.linalg.norm(np.diff(root_xz, axis=0), axis=1) if T > 1 else np.array([0.0])
    root_speed = root_step * FPS
    vel = np.linalg.norm(np.diff(motion, axis=0), axis=-1) * FPS if T > 1 else np.zeros((1, 22))  # (T-1,22)
    acc = np.linalg.norm(np.diff(motion, n=2, axis=0), axis=-1) * FPS**2 if T > 2 else np.zeros((1, 22))
    jerk = np.linalg.norm(np.diff(motion, n=3, axis=0), axis=-1) * FPS**3 if T > 3 else np.zeros((1, 22))
    lfoot_y = motion[:, LFOOT, 1]; rfoot_y = motion[:, RFOOT, 1]
    return {
        "T_norm": T / 60.0, "duration_sec": T / FPS,
        "root_path_length": float(root_step.sum()),
        "root_displacement": float(np.linalg.norm(root_xz[-1] - root_xz[0])) if T > 1 else 0.0,
        "root_speed_mean": float(root_speed.mean()), "root_speed_std": float(root_speed.std()),
        "root_speed_p95": _percentile(root_speed, 95),
        "joint_velocity_mean": float(vel.mean()), "joint_velocity_p95": _percentile(vel, 95),
        "joint_acceleration_p95": _percentile(acc, 95), "joint_jerk_p95": _percentile(jerk, 95),
        "left_contact_ratio": float(np.mean(lfoot_y < CONTACT_Y_THRESH)),
        "right_contact_ratio": float(np.mean(rfoot_y < CONTACT_Y_THRESH)),
        "motion_group_id": float(motion_group_id),
    }


def _global_artifact(motion, evaluators_by_name, gate_thresholds) -> dict:
    s = {n: _max_score(evaluators_by_name[n].evaluate(motion)) for n in
         set(ARTIFACT_EVALS) | {"SkateEvaluator", "JerkSpikeEvaluator"} if n in evaluators_by_name}
    foot_float = s.get("FootFloatingEvaluator", 0.0)
    foot_slide = s.get("SkateEvaluator", 0.0)
    vel_jit = s.get("VelocityJitterEvaluator", 0.0)
    jerk_sp = s.get("JerkSpikeEvaluator", 0.0)
    art_total = float(np.mean([s.get(n, 0.0) for n in ARTIFACT_EVALS]))
    art_vals = {"foot_floating": foot_float, "bone": s.get("BoneLengthEvaluator", 0.0), "velocity_jitter": vel_jit}
    dom = max(art_vals, key=art_vals.get)
    dom_id = {"foot_floating": 0, "bone": 1, "velocity_jitter": 2}[dom]
    # severity band (group-aware 미적용 standalone → pool-wide proxy threshold).
    band = 2 if art_total >= 0.031 else (1 if art_total >= 0.009 else 0)  # stress/normal/clean_like p80/p20
    return {
        "foot_floating_score": foot_float, "foot_sliding_score": foot_slide,
        "velocity_jitter_score": vel_jit, "jerk_spike_score": jerk_sp,
        "artifact_total_score": art_total, "dominant_artifact_type": float(dom_id),
        "artifact_severity_band": float(band),
    }


def _global_physical(motion, evaluators_by_name, gate_thresholds) -> dict:
    s = {n: _max_score(evaluators_by_name[n].evaluate(motion)) for n in PHYSICAL_EVALS if n in evaluators_by_name}
    loads = {n: s[n] / max(gate_thresholds.get(n, 1e-9), 1e-9) for n in s}
    margins = {n: gate_thresholds.get(n, 0.0) - s[n] for n in s}
    bone_cv = s.get("BoneLengthCVEvaluator", 0.0)
    dom_risk = max(loads, key=loads.get) if loads else "none"
    dom_id = {n: i for i, n in enumerate(PHYSICAL_EVALS)}.get(dom_risk, -1)
    return {
        "bone_length_cv_mean": bone_cv, "bone_length_cv_max": bone_cv,  # evaluator=max; mean proxy=max (standalone)
        "penetration_score": s.get("PenetrateEvaluator", 0.0),
        "physical_load_mean": float(np.mean(list(loads.values()))) if loads else 0.0,
        "physical_load_max": float(np.max(list(loads.values()))) if loads else 0.0,
        "gate_margin_min": float(np.min(list(margins.values()))) if margins else 0.0,
        "dominant_physical_risk": float(dom_id),
    }


def _semantic(semantic_scalars: dict | None) -> dict:
    """text-motion similarity 등 — mgpt env tm2t co-embedding 으로 precompute (optional).
    명세 §원칙3: raw embedding 금지, scalar summary 만. 없으면 NaN (schema 위치 고정)."""
    d = semantic_scalars or {}
    return {
        "text_motion_similarity": float(d.get("text_motion_similarity", np.nan)),
        "text_motion_distance": float(d.get("text_motion_distance", np.nan)),
        "semantic_confidence": float(d.get("semantic_confidence", np.nan)),
        "prompt_motion_group_match": float(d.get("prompt_motion_group_match", np.nan)),
    }


def _history(history: dict | None) -> dict:
    """closed-loop step 의 누적 history (step 0 = 모두 0). 적용 '전' 관측."""
    h = history or {}
    return {
        "step_index": float(h.get("step_index", 0)),
        "last_tool_encoding": float(h.get("last_tool_encoding", -1)),
        "last_u": float(h.get("last_u", 0.0)),
        "cumulative_joint_delta": float(h.get("cumulative_joint_delta", 0.0)),
        "cumulative_root_delta": float(h.get("cumulative_root_delta", 0.0)),
        "cumulative_fidelity_loss": float(h.get("cumulative_fidelity_loss", 0.0)),
        "previous_accepted_gain": float(h.get("previous_accepted_gain", 0.0)),
        "rejected_count": float(h.get("rejected_count", 0)),
    }


def build_v0(motion, *, motion_group_id, evaluators_by_name, gate_thresholds,
             semantic_scalars=None, history=None) -> dict:
    """v0 global quality-aware state — 모두 action 적용 **전** 관측."""
    out = {}
    out.update(_global_motion(motion, motion_group_id))
    out.update(_global_artifact(motion, evaluators_by_name, gate_thresholds))
    out.update(_global_physical(motion, evaluators_by_name, gate_thresholds))
    out.update(_semantic(semantic_scalars))
    out.update(_history(history))
    return out


def build_v1_add(motion, *, target_joints, frame_range, evaluators_by_name, gate_thresholds) -> dict:
    """v1 추가 — local target + relation (target-aware). action 적용 전 관측."""
    T = motion.shape[0]
    s, e = frame_range
    s = max(0, s); e = min(T, e + 1) if e < T else T
    seg = motion[s:e]  # (L,22,3)
    tj = [NAME_TO_IDX[j] for j in target_joints if j in NAME_TO_IDX] or [LFOOT, RFOOT]
    local = seg[:, tj, :]  # (L,k,3)
    lvel = np.linalg.norm(np.diff(local, axis=0), axis=-1) * FPS if local.shape[0] > 1 else np.zeros((1, len(tj)))
    lacc = np.linalg.norm(np.diff(local, n=2, axis=0), axis=-1) * FPS**2 if local.shape[0] > 2 else np.zeros((1, len(tj)))
    ljerk = np.linalg.norm(np.diff(local, n=3, axis=0), axis=-1) * FPS**3 if local.shape[0] > 3 else np.zeros((1, len(tj)))
    foot_in_target = [j for j in tj if j in (LFOOT, RFOOT)]
    foot_y = seg[:, foot_in_target, 1] if foot_in_target else np.zeros((seg.shape[0], 1))
    root_seg = seg[:, PELVIS, :]
    local_target = {
        "target_type_encoding": float(hash(tuple(sorted(target_joints))) % 7),
        "target_joint_count": float(len(tj)),
        "target_frame_start_norm": s / max(T, 1), "target_frame_end_norm": e / max(T, 1),
        "target_frame_length_norm": (e - s) / max(T, 1),
        "local_artifact_score": float(np.nan),   # local evaluator pass (다음 작업; schema 고정)
        "local_physical_score": float(np.nan),
        "local_velocity_p95": _percentile(lvel, 95), "local_acceleration_p95": _percentile(lacc, 95),
        "local_jerk_p95": _percentile(ljerk, 95),
        "local_foot_height_mean": float(foot_y.mean()), "local_foot_height_min": float(foot_y.min()),
        "local_foot_velocity_p95": _percentile(lvel, 95),
        "local_contact_ratio": float(np.mean(foot_y < CONTACT_Y_THRESH)) if foot_in_target else 0.0,
        "local_penetration_score": float(max(0.0, -foot_y.min())) if foot_in_target else 0.0,
        "local_bone_cv_max": float(np.nan),
    }
    relation = {
        "affected_joint_ratio": len(tj) / 22.0, "affected_frame_ratio": (e - s) / max(T, 1),
        "root_target_distance_mean": float(np.mean(np.linalg.norm(local.mean(axis=1) - root_seg, axis=-1))),
        "root_target_velocity_mismatch": float(np.nan), "parent_chain_bone_margin": float(np.nan),
        "balance_proxy": float(np.std(local.reshape(-1, 3)[:, 0])),  # lateral spread proxy
        "contact_phase_consistency": float(np.nan), "expected_fidelity_risk_proxy": float(len(tj) * (e - s) / (22.0 * max(T, 1))),
    }
    return {**local_target, **relation}


def encode_action(tool: str, target_side: str, u: float, is_stop: bool) -> dict:
    tools = ("FootLockTool", "BoneProjectionTool", "VelocitySmoothingTool")
    fam = {"FootLockTool": 0, "BoneProjectionTool": 1, "VelocitySmoothingTool": 0}  # 0=geometric,1=skeleton
    cost = {"FootLockTool": 1.0, "BoneProjectionTool": 1.0, "VelocitySmoothingTool": 1.2}
    side = {"left": 0, "right": 1, "both": 2, "full": 3, "none": -1}
    return {
        "tool_encoding": float(tools.index(tool)) if tool in tools else -1.0,
        "tool_family": float(fam.get(tool, -1)), "tool_cost": float(cost.get(tool, 1.0)),
        "target_type_encoding": float(side.get(target_side, -1)), "target_side": float(side.get(target_side, -1)),
        "target_chain_id": float(side.get(target_side, -1)),
        "u": float(u), "u_sq": float(u * u), "is_continuous_u": 1.0,
        "is_stop_action": 1.0 if is_stop else 0.0,
    }


def verify_schema() -> dict:
    """금지 feature 배제 + before-action observable 자가 검증."""
    all_feats = set(feature_names("v2", include_action=True))
    leaked = all_feats & FORBIDDEN_FEATURES
    # after-action 키워드 휴리스틱 audit. NOTE: `_gain` 단독은 제외 — history 의
    # previous_accepted_gain (직전 step 결과 = 현재 action 전 관측 가능) 과 uncertainty 의
    # ensemble_gain_variance (예측 분산, 적용 후 결과 아님) 는 명세상 before-action observable.
    after_kw = ("after", "gate_result", "quality_after", "_label", "oracle_action", "dense_oracle")
    suspicious = [f for f in all_feats if any(k in f for k in after_kw)]
    # 명세상 before-action 이 검증된 ambiguous feature (history/uncertainty) 의 사유 박제.
    before_action_ambiguous = {
        "previous_accepted_gain": "history — 직전 step accepted action 의 gain, 현재 action 결정 전 관측",
        "ensemble_gain_variance": "uncertainty — ensemble 예측의 분산, 적용 후 실제 gain 아님 (v2 data_support)",
    }
    return {
        "n_v0": state_dim("v0"), "n_v1": state_dim("v1"), "n_v2": state_dim("v2"),
        "n_v0_action": state_dim("v0", True), "n_v1_action": state_dim("v1", True),
        "n_v2_action": state_dim("v2", True),
        "n_action": len([f for blk in ACTION_SCHEMA.values() for f in blk]),
        "forbidden_leaked": sorted(leaked), "after_action_suspicious": suspicious,
        "before_action_ambiguous_justified": before_action_ambiguous,
        "passed": (not leaked) and (not suspicious),
    }


if __name__ == "__main__":
    v = verify_schema()
    print("=== AR-037 Effect-Aware State Schema ===")
    print(f"  v0 dim: {v['n_v0']} (state) / {v['n_v0_action']} (state+action)")
    print(f"  v1 dim: {v['n_v1']} / {v['n_v1_action']}")
    print(f"  v2 dim: {v['n_v2']} / {v['n_v2_action']}  (action={v['n_action']})")
    print(f"  forbidden leaked: {v['forbidden_leaked']}  after-action suspicious: {v['after_action_suspicious']}")
    print(f"  verify passed: {v['passed']}")
