# AR-037 — Effect-Aware Ranker State Ablation Spec

## 한 줄 목표

`q_proxy` 를 정의하기 전에, Effect-Aware Quality-Gain Ranker 가 관측할 state 를 v0/v1/v2 단계로 고정하고 ablation 가능한 입력 구조를 만든다.

## 배경

현재 M0 는 `artifact-centric` policy 에 가깝다. 즉 artifact score 와 tool 을 직접 매칭하는 방향으로 작동하며, g2_stress 에서 안전하고 보수적이지만 dense oracle 대비 headroom 이 남아 있다.

새 정책은 artifact 를 무조건 줄이는 것이 아니라, 특정 correction action 이 전체 motion quality 를 올릴 가능성이 있는지 예측해야 한다. 따라서 reward 인 `q_proxy` 보다 먼저 state 를 정리해야 한다.

## 핵심 원칙

1. State 와 label 을 분리한다.
   - state: action 적용 전 관측값.
   - label: action 적용 후 결과값 (`quality_gain`, `gate_result`, `fidelity_loss`, `semantic_drop`, `after_state`).
2. v0 는 target-aware 라고 부르지 않는다.
   - v0 는 global quality-aware ranker.
   - v1 부터 target-aware ranker.
3. raw embedding 은 v0/v1 에 직접 넣지 않는다.
   - 512/768/1024 차원 text/motion embedding 직접 투입은 보류.
   - 대신 similarity, distance, confidence 같은 scalar summary 만 사용한다.
4. FID/R-Precision/MM-Dist/Diversity/perceptual 은 최종 validation 지표다.
   - 단, text-motion embedding similarity 처럼 per-sample scalar 로 변환 가능한 항목은 state 후보가 될 수 있다.

## 정책 입력 정의

```text
motion x
prompt c
target r
history h
candidate action a = (tool, target, u)
```

STOP 은 별도 action 으로 둔다. `u=0` 과 STOP 은 같은 의미로 취급하지 않는다.

## State Ablation

| version | 이름 | state block | action | claim |
|---|---|---|---|---|
| v0 | global quality-aware ranker | global motion + artifact + physical + semantic + history | `(tool, u)` | 전체 상태를 보고 어떤 tool/strength 가 유리한지 판단 |
| v1 | target-aware quality-gain ranker | v0 + local target + relation | `(tool, target, u)` | 어느 부위/구간에 적용해야 품질이 좋아지는지 판단 |
| v2 | support-aware constrained ranker | v1 + data support / uncertainty | `(tool, target, u)` | offline data support 밖의 과대평가를 줄임 |

## 현재 고정된 State 설계

AR-037 완료 결과, state 는 다음 3단계 ablation 구조로 고정한다.

```text
v0 = global quality-aware state
v1 = v0 + local target state + relation state
v2 = v1 + data support / uncertainty state
```

가장 중요한 구분은 다음이다.

```text
v0: 전체 모션을 보고 tool/u 선택
v1: target 부위/구간까지 보고 tool/target/u 선택
v2: 그 판단을 학습 데이터 관점에서 믿을 수 있는지까지 확인
```

따라서 v0 는 target-aware policy 라고 부르지 않는다. v1 부터 target-aware claim 을 사용할 수 있다.

| version | 의미 | state dim | state+action dim |
|---|---|---:|---:|
| v0 | global quality-aware | 40 | 50 |
| v1 | target-aware | 64 | 74 |
| v2 | support-aware | 71 | 81 |

현재 action encoding 은 10차원이다.

| action block | feature 수 | feature |
|---|---:|---|
| tool | 3 | `tool_encoding`, `tool_family`, `tool_cost` |
| target | 3 | `target_type_encoding`, `target_side`, `target_chain_id` |
| strength | 3 | `u`, `u^2`, `is_continuous_u` |
| stop | 1 | `is_stop_action` |

### v0 의 의미

v0 는 전체 motion 상태만 본다.

```text
v0 = [
  global_motion_state,
  global_artifact_state,
  global_physical_state,
  semantic_motion_state,
  correction_history_state
]
```

v0 의 질문은 다음이다.

```text
이 모션은 얼마나 빠른가?
발 접촉 비율은 어떤가?
artifact 와 physical risk 는 어느 정도인가?
text-motion 의미 정합성은 어떤가?
이미 이전에 보정을 많이 했는가?
```

v0 의 action 은 `(tool, u)` 이다. 예: `FootLock 0.4`, `VelocitySmoothing 0.2`, `BoneProjection 0.7`, `STOP`.

### v1 의 의미

v1 은 v0 에 target 정보를 추가한다.

```text
v1 = v0 + [
  local_target_state,
  relation_state
]
```

v1 의 질문은 다음이다.

```text
왼발 contact 구간에 문제가 있는가?
이 target 이 전체 frame 중 얼마나 큰가?
local velocity / jerk 가 큰가?
발이 떠 있거나 땅을 파고드는가?
root 와 target 움직임이 충돌하는가?
이 부위를 고치면 fidelity risk 가 클 것 같은가?
```

v1 의 action 은 `(tool, target, u)` 이다.

### v2 의 의미

v2 는 v1 에 offline data support / uncertainty 를 추가한다.

```text
v2 = v1 + [
  data_support_state
]
```

v2 의 질문은 다음이다.

```text
이 상태와 비슷한 데이터가 학습셋에 있었는가?
이 tool-target-u 조합을 충분히 봤는가?
모델 ensemble 이 이 action effect 를 확신하는가?
OOD action 인가?
```

v2 는 offline policy 의 OOD value overestimation 을 줄이기 위한 단계다.

### 현재 미완 부분

| 부분 | 현재 상태 |
|---|---|
| v0 motion/artifact/physical/history | 실제 계산 가능 |
| v0 semantic 4개 | schema 고정, mgpt text-motion embedding pass 필요 |
| v1 local/relation 일부 | schema 고정, 일부 placeholder |
| v2 support/uncertainty | schema 고정, q_proxy/M1 이후 계산 가능 |
| target encoding | `hash()` 기반 encoding 은 재현성 위험. 고정 mapping 으로 바꿀 필요 있음 |

### 현재 설계의 한 줄 정리

현재 state 설계는 `artifact 를 보고 tool 을 고르는 정책` 에서 `어떤 보정이 전체 motion quality 를 올릴 가능성이 있는지 예측하는 정책` 으로 넘어가기 위한 입력 구조다.

## v0 State

목표 차원: state only 약 40-55, state+action 약 55-70.

### global_motion_state

- `T_norm`
- `duration_sec`
- `root_path_length`
- `root_displacement`
- `root_speed_mean`
- `root_speed_std`
- `root_speed_p95`
- `joint_velocity_mean`
- `joint_velocity_p95`
- `joint_acceleration_p95`
- `joint_jerk_p95`
- `left_contact_ratio`
- `right_contact_ratio`
- `motion_group_encoding` 또는 `motion_group_id`

### global_artifact_state

- `foot_floating_score`
- `foot_sliding_score`
- `velocity_jitter_score`
- `jerk_spike_score`
- `artifact_total_score`
- `dominant_artifact_type`
- `artifact_severity_band`

### global_physical_state

- `bone_length_cv_mean`
- `bone_length_cv_max`
- `penetration_score`
- `physical_load_mean`
- `physical_load_max`
- `gate_margin_min`
- `dominant_physical_risk`

### semantic_motion_state

- `text_motion_similarity`
- `text_motion_distance`
- `semantic_confidence`
- `prompt_motion_group_match`

### correction_history_state

- `step_index`
- `last_tool_encoding`
- `last_u`
- `cumulative_joint_delta`
- `cumulative_root_delta`
- `cumulative_fidelity_loss`
- `previous_accepted_gain`
- `rejected_count`

## v1 추가 State

목표 차원: state only 약 70-95, state+action 약 85-110.

### local_target_state(r)

- `target_type_encoding`
- `target_joint_count`
- `target_frame_start_norm`
- `target_frame_end_norm`
- `target_frame_length_norm`
- `local_artifact_score`
- `local_physical_score`
- `local_velocity_p95`
- `local_acceleration_p95`
- `local_jerk_p95`
- `local_foot_height_mean`
- `local_foot_height_min`
- `local_foot_velocity_p95`
- `local_contact_ratio`
- `local_penetration_score`
- `local_bone_cv_max`

### relation_state(r)

- `affected_joint_ratio`
- `affected_frame_ratio`
- `root_target_distance_mean`
- `root_target_velocity_mismatch`
- `parent_chain_bone_margin`
- `balance_proxy`
- `contact_phase_consistency`
- `expected_fidelity_risk_proxy`

## v2 추가 State

목표 차원: state only 약 80-110, state+action 약 95-125.

### data_support_state

- `knn_state_distance`
- `knn_state_action_distance`
- `motion_group_support_count`
- `tool_target_u_support_count`
- `ensemble_gain_variance`
- `ensemble_safe_variance`
- `ood_flag`

## Action Encoding

| action part | feature |
|---|---|
| tool | `tool_encoding`, `tool_family`, `tool_cost` |
| target | `target_type_encoding`, `target_side`, `target_chain_id` |
| strength | `u`, `u^2`, `is_continuous_u` |
| stop | `is_stop_action` |

## 금지 Feature

다음은 state 에 넣지 않는다.

- `sample_id`
- `trial_id`
- `pool_dir`
- split name
- action 적용 후 score
- `gate_result`
- `quality_after`
- `quality_gain`
- dense oracle action
- final standard metric aggregate

## 성공 조건

1. v0/v1/v2 feature schema 가 고정된다.
2. 각 feature 가 before-action observable 인지 확인된다.
3. state dimension 이 보고된다.
4. v0, v1, v2 ablation 이 같은 split / 같은 candidate set / 같은 q_proxy 로 비교 가능해진다.
5. target-aware claim 은 v1 이후에만 사용한다.

## 다음 작업

1. `q_proxy` 정의.
2. 기존 transition rows 에 v0 state vector 생성.
3. v0 M1 dense effect oracle 실행.
4. v1 target proposal 과 local/relation feature 구현.
5. v0/v1/v2 ablation 으로 M0 대비 g2_stress headroom 을 검증.

## Claim Boundary

이 작업은 정책 성능을 증명하지 않는다. state 설계를 고정해 이후 `q_proxy`, effect-vector label, M1 oracle, M2 learned ranker 실험이 비교 가능하도록 만드는 준비 단계다.
