# Human Motion Generation 연구 방향 정리

## 1. 초기 연구 방향과 한계

초기 아이디어는 GIST/LMTraj 계열의 **좌표 numerical tokenization**을 3D human skeleton motion으로 확장하는 것이었다.

기존 방향은 다음과 같다.

```text
2D trajectory numerical tokenization
→ 3D human joint coordinate / delta tokenization
→ LLM이 다음 motion sequence 예측
```

예시 표현은 다음과 같다.

```text
[ELBOW_X][NUM][INT]000[SEP][DEC]42361[ENDNUM]
```

이 방식의 장점은 다음과 같다.

- 관절별 좌표 정밀도 보존
- delta / velocity 기반 세부 motion 변화 표현 가능
- LLM이 numerical motion을 직접 처리할 수 있는지 검증 가능
- 기존 MotionGPT류와 달리 좌표 수준 해석 가능성 확보

하지만 한계가 매우 크다.

- human skeleton은 `22 joints × 3 channels × frames`(또는 263d 표현) 구조라 token explosion 발생
- LLM은 정밀 수치 회귀에 강한 모델이 아님
- MotionGPT/T2M-GPT류는 이미 LLM 특성에 맞게 motion을 discrete token으로 바꿔 처리함
- SCI 논문으로 주장하려면 “왜 굳이 좌표를 LLM에 직접 넣는가?”를 강하게 방어해야 함

따라서 이 방향은 **메인 SCI 연구로 밀기 어렵고**, preliminary study 또는 negative finding으로 활용하는 것이 적절하다고 판단했다.

---

## 2. MotionGPT와 기존 연구 대비 인식

MotionGPT 계열은 다음 방식으로 motion을 다룬다.

```text
raw motion
→ VQ-VAE / motion tokenizer
→ discrete motion token sequence
→ LLM / GPT-style model
→ decoder로 3D skeleton 복원
```

즉, MotionGPT는 좌표를 직접 LLM에 전달하지 않는다. 대신 LLM이 잘하는 discrete token prediction 문제로 변환한다.

비교하면 다음과 같다.

| 구분 | Numerical token 방식 | MotionGPT 방식 |
|---|---|---|
| 표현 | 좌표/Δ좌표를 직접 언어화 | motion을 VQ code token으로 변환 |
| 장점 | 좌표 정밀도, 해석 가능성 | token 효율, 긴 sequence 생성 |
| 단점 | token explosion | quantization loss, 좌표 세부 제어 약함 |
| 연구 포지션 | numerical motion language | token-based motion generation |

MotionGPT3는 기존 VQ 기반 discrete tokenization의 한계, 즉 quantization error와 cross-modal interference를 지적하고 continuous latent + dual-stream 방식으로 전환했다. 이는 “모션을 무조건 텍스트 숫자 token으로 변환하는 방식”의 한계를 뒷받침하는 근거로 볼 수 있다.

---

## 3. 연구 방향 전환

기존 방향을 다음과 같이 전환하는 것이 가장 적절하다고 판단했다.

### 기존 방향

```text
LLM에게 관절 좌표를 직접 입력하고 다음 motion을 생성하게 함
```

### 전환 방향

```text
기존 motion generator가 생성한 motion을 skeleton 수준에서 평가하고,
artifact 상태에 따라 적절한 evaluator/correction tool을 선택적으로 호출하여 refined motion을 생성하는 범용 orchestration harness를 제안
```

즉, 본 연구는 더 이상 “새로운 motion generator”가 아니다.

본 연구의 핵심은 다음이다.

```text
Base Motion Generator
(MotionGPT / T2M-GPT / MDM / MLD / future model)
        ↓
Generated Skeleton Motion [T, J, 3]
        ↓
Evaluator Tool Registry
        ↓
Orchestrator
        ↓
Correction Tool Registry
        ↓
Re-evaluation Loop
        ↓
Refined Motion + Quality Report + Tool Call Trace
```

---

## 4. 최종 연구 포지션

최종 연구 방향은 다음과 같다.

> 본 연구는 특정 motion generation model 또는 특정 correction algorithm에 종속되지 않는 **generator-agnostic, tool-extensible orchestration harness**를 제안한다. 제안 프레임워크는 다양한 모션 생성 모델의 decoded skeleton output을 입력으로 받아 evaluator tools와 correction tools를 모듈식으로 등록하고, motion artifact 상태에 따라 적절한 tool을 선택적으로 호출한다.

핵심 키워드는 다음 세 가지다.

```text
1. Generator-agnostic
   어떤 motion generator 출력이든 skeleton format만 맞으면 적용 가능

2. Tool-extensible
   새로운 evaluator/correction 기술을 tool registry에 추가 가능

3. Orchestrated closed-loop refinement
   평가 → tool 선택 → 보정 → 재평가 → 종료/반복 구조
```

### 4.1 Core Hypothesis and Scope

본 연구의 핵심 가설은 H1-H2이며, H3는 실용성을 확인하기 위한 보조 운영 가설로 둔다.

```text
H1. Artifact-conditional necessity
    Generated motion artifact는 단일 global post-processing으로
    균일하게 처리되기보다, artifact type, body part, temporal region에
    따라 서로 다른 correction operator를 필요로 한다.

H2. Learnable routing
    artifact 상태(state)에서 correction tool(action)로의 매핑
    π(a | s)가 데이터 기반으로 학습 가능한지 검증한다.
    특히 learned selector가 rule-based baseline 대비 net gain을 개선하고,
    새로운 generator output에서도 일정 수준의 일반화 성능을 보이는지 확인한다.

H3. No-harm operating regime (secondary)
    학습된 라우팅은 artifact-rich motion을 의미 있게 개선하면서도,
    이미 품질이 높은 motion에 대해서는 (적절히 STOP하거나
    small-strength로 수렴함으로써) 분포·의미·fidelity를 훼손하지 않는다.
```

따라서 artifact-conditioned tool selection과 re-evaluation은 fixed post-processing 또는 단일 learned refinement보다 artifact reduction과 motion fidelity 사이의 trade-off를 더 잘 관리할 수 있다.

본 연구의 scope는 다음처럼 제한한다.

```text
In scope:
- decoded skeleton motion [T, J, 3] 수준의 artifact 평가
- artifact-conditioned evaluator/correction tool registry
- rule-based 또는 lightweight learned tool selector
- tool call trace와 before/after metric 기반 explainable refinement

Out of scope:
- 새로운 text-to-motion generator 개발
- 새로운 범용 motion tokenizer 개발
- full-body physics simulator 자체 개발
- multi-step RL을 1차 논문의 필수 contribution으로 주장
```

초기 논문에서는 framework의 정당성을 다음 최소 조건으로 검증한다.

```text
1. artifact별로 효과적인 correction tool이 다르게 나타나는가?
2. tool selection이 fixed smoothing/fixed pipeline보다 net gain이 있는가?
3. re-evaluation loop가 artifact 감소와 fidelity 보존의 균형을 개선하는가?
4. 같은 interface가 최소 2개 generator output에서 작동하는가?
```

---

## 5. 기존 연구와의 관계

가까운 연구 흐름은 다음과 같다.

| 연구 흐름 | 관련성 | 차이점 |
|---|---|---|
| MotionGPT / T2M-GPT | token-based motion generation | 본 연구는 generator가 아니라 refinement layer |
| PhysDiff | physics-guided motion refinement | 본 연구는 generator 외부 tool orchestration |
| ReinDiffuse | RL로 physical plausibility 개선 | 본 연구는 생성 모델 학습이 아니라 외부 tool-calling policy로 작동 |
| Post-hoc calibrator (예: DMC) | generator-agnostic motion refinement | 본 연구는 단일 calibrator가 아니라 artifact-conditioned tool orchestration |
| Footskate correction | 특정 artifact correction | 본 연구는 다양한 artifact/tool을 통합 |
| CoMA / Motion-Agent | agentic motion generation/review | 본 연구는 생성보다 decoded skeleton refinement 중심 |

DMC와 같은 model-agnostic post-hoc calibrator는 참고해야 할 중요한 관련 흐름 중 하나이다. 다만 본 연구의 정당성은 특정 calibrator를 이기는 데 있지 않다. 핵심은 생성된 motion의 artifact 상태를 명시적으로 평가하고, artifact type / body part / frame range에 따라 서로 다른 correction operator를 선택·조합·재평가하는 **orchestration framework**를 제안하는 데 있다.

따라서 DMC류 방법은 related work에서 언급하되, 논문의 중심 경쟁 구도로 과도하게 전면화하지 않는다. 실험 조건이 맞는 경우에는 강한 optional baseline 또는 correction tool registry에 들어갈 수 있는 learned calibrator 예시로 다룬다.

```text
Post-hoc calibrator 단독:
generated motion → learned calibrator → refined motion

본 연구:
generated motion → evaluator tools → orchestrator
→ learned calibrator / foot-lock / smoothing / projection 중 선택
→ re-evaluation → refined motion
```

### 5.1 ArtifactRouter의 독립적 정당성

Post-hoc calibrator 계열은 다양한 artifact를 하나의 refinement model로 처리할 수 있다는 장점이 있다. 그러나 본 연구는 이 계열을 반박하거나 대체하는 논문이 아니라, artifact-conditioned orchestration이라는 별도의 문제 설정을 제안한다.

정당성은 다음 가설에 둔다.

1. **Artifact-specific inductive bias 명시적 활용 가능성**
   - foot contact는 contact constraint 기반 hard projection, jitter는 frequency-domain filter, bone length는 skeletal projection처럼, artifact 유형에 자연스러운 operator를 직접 등록할 수 있다.
   - 단일 learned calibrator는 이런 domain knowledge를 implicit하게 학습하지만, tool-specialized 구조는 이를 명시적으로 분리한다.

2. **선택적·해석 가능한 보정 (selective and interpretable correction)**
   - 현재 artifact 상태에 따라 어떤 tool이 어디에 얼마나 적용되었는지 trace로 남는다.
   - 이는 단일 end-to-end refinement보다 결과의 귀인 분석과 디버깅을 쉽게 한다.

3. **Artifact coverage 확장 용이성**
   - 새로운 artifact가 식별되었을 때, 전체 calibrator를 재학습하지 않고 registry에 specialized tool을 추가할 수 있다.

따라서 본 연구의 contribution은 "특정 calibrator가 못 하는 것을 한다"가 아니라, **artifact-conditioned tool selection이 명시적 inductive bias, 해석 가능성, 모듈식 확장성을 제공하며 generated motion refinement의 독립적인 framework가 될 수 있다**는 가설을 검증하는 것이다.

---

## 6. 제안 아키텍처

### 6.1 Base Motion Generator

본 연구는 generator를 새로 만들지 않는다.

사용 가능한 generator 예시는 다음과 같다.

```text
MotionGPT
T2M-GPT
MDM
MLD
MotionGPT3 계열
future human motion generator
```

공통 출력은 다음으로 정규화한다.

```text
motion: [T, J, 3]
skeleton_tree
fps
joint_names
ground_plane
optional: text prompt / action label
```

Generator마다 skeleton topology가 다르므로(HumanML3D 263d, SMPL 24 joints 등) 명시적 **Skeleton Normalizer** layer가 필요하다.

```text
Skeleton Normalizer
- input: generator-specific output
- output: canonical SMPL-X 24 joints, [T, J, 3]
- SMPL retargeting / forward kinematics 통일
- contact label 추정 모듈 (없는 경우 heuristic)
- ground plane 추정 (없는 경우 floor 추출)
```

이 layer 자체는 본 연구의 contribution이 아니며, 기존 retargeting tool(SMPL-X, mocap_utils 등)을 그대로 사용한다.

---

### 6.2 Evaluator Tool Registry

생성된 motion을 artifact별로 평가하는 tool pool이다.

가능한 evaluator는 다음과 같다.

```text
ContactEvaluator
- foot sliding
- foot floating
- ground penetration
- contact instability

TemporalEvaluator
- velocity jitter
- acceleration spike
- jerk

SkeletalEvaluator
- bone length variation
- joint angle violation

RootTorsoEvaluator
- root drift
- torso orientation jump

UpperLimbEvaluator
- wrist jitter
- elbow angle violation
- arm-torso mismatch

LowerLimbEvaluator
- knee over-extension
- ankle instability
- leg phase mismatch

CoordinationEvaluator
- left/right imbalance
- root-limb inconsistency
- arm-leg phase mismatch
```

Evaluator 출력 예시는 다음과 같다.

```json
{
  "agent": "ContactEvaluator",
  "error_type": "right_foot_sliding",
  "body_part": "right_foot",
  "frames": [18, 31],
  "score": 0.82,
  "severity": "high",
  "recommendation": "foot_lock"
}
```

---

### 6.3 Correction Tool Registry

실제로 motion을 국소적으로 보정하는 함수 또는 모델이다.

기본 tool 후보는 다음과 같다.

```text
foot_lock_tool
- foot sliding 감소

velocity_smoothing_tool
- jitter / velocity spike 감소

bone_projection_tool
- bone length inconsistency 감소

joint_angle_clamp_tool
- knee/elbow over-extension 감소

ground_projection_tool
- penetration / floating 감소

local_interpolation_tool
- 특정 frame block discontinuity 완화

learned_calibrator_tool
- DMC류를 포함한 learned post-hoc calibrator를 plug-in tool로 등록

physics_projection_tool
- PhysDiff류 physics-guided projection을 plug-in tool로 등록
```

각 tool은 공통 인터페이스를 갖는다.

```python
class CorrectionTool:
    name: str

    def apply(
        self,
        motion,
        target_part,
        target_joints,
        frame_range,
        strength,
        metadata=None
    ):
        corrected_motion = ...
        report = {
            "tool": self.name,
            "target_part": target_part,
            "frame_range": frame_range,
            "strength": strength,
            "modified_joints": target_joints,
            "correction_magnitude": ...
        }
        return corrected_motion, report
```

---

### 6.4 Orchestrator

Orchestrator는 evaluator 결과를 종합하고 어떤 tool을 호출할지 결정한다.

본 연구는 이 결정 과정을 단순한 규칙 분기가 아니라 **학습 가능한 라우팅 문제(Learnable Routing Problem)**로 정식화한다. 즉, generated motion의 artifact state와 correction action 사이의 mapping을 명시적으로 정의하고, 이 mapping이 rule-based baseline보다 더 나은 net gain을 낼 수 있는지 실험적으로 검증한다.

```text
Learnable Routing Problem 정의

state  s = artifact state (evaluator reports + history features)
action a = (tool_id, target_part, frame_range, strength)
goal     = π*(a | s) such that
           E[NetGain(a; s)] is maximized over the artifact distribution
           produced by base motion generators.
```

본 연구는 이 mapping `π*(a | s)`에 대해 다음을 검증한다.

1. artifact state가 tool 선택에 필요한 유의미한 signal을 포함하는가?
2. supervised tool selector / contextual-bandit이 rule-based baseline 대비 net gain을 개선하는가?
3. 학습된 selector가 새로운 generator output에서도 일정 수준의 일반화 성능을 보이는가?

따라서 본 연구의 핵심 학문적 contribution 중 하나는 motion refinement를 "post-processing pipeline 설계"에서 "artifact 상태 → tool 선택의 학습 가능한 라우팅 문제(Learnable Routing Problem)"로 재정의하고, 이 문제 설정의 정당성을 실험적으로 검증하는 것이다.

역할은 다음과 같다.

```text
1. evaluator reports 수집
2. artifact severity 정렬
3. primary error 결정
4. correction tool 선택
5. tool 적용 후 재평가 여부 결정
6. stop / revise / reject 판단
7. tool call trace 저장
8. tool conflict detection (kinematic prior 기반)
   - skeleton의 운동학적 위계와 tool effect propagation을 반영하여
     candidate tool이 기존 보정을 무효화/악화시킬 가능성을 평가
9. tool ordering
   - 동일 step 내 multiple correction이 필요한 경우 ordering rule
     (예: foot_lock → bone_projection → smoothing 순서로 bone violation 방지)
```

#### 6.4.0 Kinematic Dependency Graph Prior (KDG)

Tool 간 충돌과 순서 의존성을 일관되게 다루기 위해, 본 연구는 skeleton의 운동학적 위계와 각 tool의 영향 범위를 결합한 **Kinematic Dependency Graph (KDG)**를 structural prior로 사용한다. KDG 자체를 독립적인 graph model contribution으로 주장하지 않고, routing decision의 `conflict_risk`와 ordering penalty를 안정화하는 보조 장치로 둔다.

```text
Nodes:
  N_J = joint set (root, spine, chest, neck, shoulders, elbows, wrists,
                   hips, knees, ankles, feet, ...)
  N_T = tool set  (foot_lock, velocity_smoothing, bone_projection,
                   ground_projection, joint_angle_clamp, learned_calibrator, ...)

Edges:
  E_kin ⊂ N_J × N_J   skeleton parent → child kinematic edges
                       (root → hip → knee → ankle → foot 등)
  E_aff ⊂ N_T × N_J   tool t가 직접 modify 하는 joint(primary affect)
  E_prop⊂ N_T × N_J   tool t가 forward kinematics를 통해 간접 영향을 주는 joint
                       (propagation, 가중치 < 1)
```

각 tool `t`는 KDG 위에서 affected joint set `A(t)`와 영향 범위 가중치 `w_kin(t, j)`를 갖는다. 두 candidate tool `t_a, t_b`의 충돌 가능성은 다음과 같이 정의한다.

```text
ConflictScore(t_a, t_b) =
  sum_{j ∈ A(t_a) ∩ A(t_b)}
      w_kin(t_a, j) * w_kin(t_b, j) * directional_penalty(t_a, t_b, j)

directional_penalty:
  - t_a가 j에 hard constraint를 가했고 t_b가 같은 j를 smoothing-style로
    되돌릴 가능성이 있으면 +α
  - t_a, t_b가 서로 호환되는 operator면 0
```

KDG prior는 ordering 규칙을 운동학적으로 보조한다.

```text
Ordering rule (KDG-derived):
  1. Root 또는 상위 kinematic node를 수정하는 tool이 먼저 호출된다.
     - Root 수정은 forward kinematics를 통해 하위 joint position에
       전파되므로, 하위 joint의 보정량이 root 보정 이후에 결정되어야 한다.
  2. 동일 depth에서는 hard-constraint tool (contact projection,
     joint angle clamp) → soft tool (smoothing, learned calibrator) 순.
     - hard constraint가 만들어 둔 boundary를 soft tool이 점진적으로
       압축하는 형태가 안정적이기 때문이다.
  3. KDG 상 ancestor-descendant 관계가 있는 joint를 modify하는 두 tool은
     동일 step에서 병렬 적용하지 않고 위계 순서로 적용한다.
```

이 KDG prior는 conflict detection과 ordering을 모두 같은 구조적 표현에서 계산하게 해, "직전 tool 확인" 수준의 ad hoc check를 줄인다. 학습 기반 selector는 KDG에서 유도된 feature(예: candidate tool이 modify할 joint의 KDG-depth, ancestor 보정 여부, 영향 범위 overlap)를 optional state feature로 활용할 수 있다. 단, 1차 논문의 중심은 KDG 자체가 아니라 artifact-conditioned routing이므로, KDG는 ablation을 통해 유효성을 확인하는 보조 mechanism으로 둔다.

Orchestrator 출력 예시는 다음과 같다.

```json
{
  "decision": "revise",
  "primary_error": "right_foot_sliding",
  "selected_tool": "foot_lock_tool",
  "target_part": "right_foot",
  "target_frames": [18, 31],
  "strength": "medium",
  "next_step": "re_evaluate"
}
```

### 6.4.1 Orchestrator Scoring and Decision Rule

Orchestrator는 단순히 evaluator의 recommendation을 그대로 따르지 않고, 각 candidate action을 점수화한다.

```text
candidate action a = (tool_id, target_part, frame_range, strength)

Score(a | s, h) =
    expected_artifact_reduction(a, s)
  - fidelity_risk(a, s)
  - conflict_risk(a, h; KDG)
  - tool_cost(a)
```

여기서 `s`는 현재 evaluator report로 구성된 artifact state이고, `h`는 이전 tool call history이며, KDG는 §6.4.0의 Kinematic Dependency Graph이다.

```text
expected_artifact_reduction:
  해당 tool이 target artifact metric을 얼마나 줄일 것으로 예상되는가

fidelity_risk:
  correction magnitude, modified frame ratio, semantic consistency 손상 가능성

conflict_risk (KDG-based):
  sum_{t_prev ∈ h} ConflictScore(t_prev, t_cand) + KDG ordering 위반 penalty
  - 이전 tool과의 직접적 재악화뿐 아니라
    skeleton ancestor 보정 전에 descendant 보정을 시도하는 등의
    운동학적 ordering 위반도 함께 penalize

tool_cost:
  runtime, call count, correction strength, 불필요한 과수정 penalty
```

초기 MVP에서는 학습 모델 없이 rule-based scoring으로 시작한다.

```text
1. severity가 threshold 이상인 artifact만 candidate로 유지
2. artifact-tool compatibility table로 가능한 correction tool 후보 생성
3. target body part와 frame_range를 evaluator report에서 가져옴
4. strength는 severity에 따라 small/medium/large로 초기화
5. tool history를 보고 conflict risk가 높은 후보는 skip 또는 strength 감소
6. Score가 가장 높은 action을 적용
7. 적용 후 종합 평가가 악화되면 rollback 또는 lower-strength retry
   - 판단 기준은 단순 artifact score가 아니라
     Score(a | s, h) 정의에 사용된 종합 점수
     (artifact reduction − fidelity risk − conflict risk − tool cost)
8. max_iteration 또는 score 개선폭 threshold 이하에서 STOP
```

학습 기반 selector는 위 rule-based decision을 대체하는 것이 아니라, `expected_artifact_reduction`과 `fidelity_risk` 추정을 더 잘 하도록 확장하는 단계로 둔다.

---

### 6.5 Closed-loop Refinement

전체 loop는 다음과 같다.

```python
motion = base_generator(prompt)

tool_history = []

for step in range(max_iterations):
    reports = evaluator_layer.evaluate(motion)
    decision = orchestrator.decide(reports, tool_history)

    if decision.action == "STOP":
        break

    motion, tool_report = correction_tools[decision.tool].apply(
        motion=motion,
        target_part=decision.target_part,
        frame_range=decision.target_frames,
        strength=decision.strength
    )

    tool_history.append(tool_report)

refined_motion = motion
```

#### 6.5.1 Convergence consideration

Tool 호출이 oscillation을 일으킬 수 있다(tool A가 만든 artifact를 tool B가 고치고 다시 tool A로). 다음을 보장한다.

1. Composite objective(artifact reduction − fidelity loss − tool cost)는 step별 비감소(non-decreasing)여야 함
   - 단순 artifact score만 보면 fidelity가 무너지는 correction을 잡지 못하므로,
     §6.4.1의 Score 함수를 그대로 monitor metric으로 사용한다
   - 종합 점수가 악화되는 tool 호출은 reject 또는 rollback
2. Same `(tool, target)` 재호출은 strength를 줄여서만 허용
3. 실험에서 step 수에 따른 metric trajectory를 보고 convergence를 empirically 입증

---

## 7. Tool-calling Policy Learning (확장)

본 연구의 main contribution은 RL 자체가 아니다. SCI 1차 논문에서는 RL을 전면에 두지 않고, 다음과 같이 학습 강도가 점진적으로 증가하는 orchestrator 계층을 main으로 다룬다.

```text
Main contribution (논문 본문):
  Rule-based orchestrator
  → Oracle-guided supervised tool selector
  → Contextual-bandit orchestrator (single-step learned policy)

Extension (후속 확장 또는 논문 부록):
  Multi-step RL orchestrator (PPO/DQN 등 full RL)
```

즉 본 연구에서 "tool-calling policy learning"이라 하면 우선 contextual-bandit 수준의 학습된 selector를 의미하며, multi-step credit assignment가 필요한 full RL은 framework의 자연스러운 확장으로 둔다. 이는 SCI 1차 논문의 구현 리스크를 통제하고, framework 자체의 일반성을 먼저 입증하기 위한 전략적 선택이다.

아래 MDP 정의는 확장 단계(multi-step RL orchestrator)를 위한 설계이며, 1차 논문에서는 같은 state/reward 정의를 contextual-bandit 형태로 단순화(single-step decision)하여 활용한다.

### MDP 정의 (확장용)

#### State

```text
s_t = [
  foot_sliding_left,
  foot_sliding_right,
  ground_penetration,
  foot_floating,
  velocity_jitter_upper,
  velocity_jitter_lower,
  bone_length_error,
  joint_angle_violation,
  correction_magnitude,
  tool_call_count,
  iteration
]
```

#### Action

```text
a_t = (tool_id, target_part, strength)
```

예시:

```text
Tool:
0 STOP
1 FOOT_LOCK
2 VELOCITY_SMOOTHING
3 BONE_PROJECTION
4 GROUND_PROJECTION
5 JOINT_ANGLE_CLAMP
6 LEARNED_CALIBRATOR

Target:
full_body, root_torso, left_arm, right_arm, left_leg, right_leg, left_foot, right_foot

Strength:
small, medium, large
```

#### Reward

```text
R = artifact_reduction
    - fidelity_loss
    - correction_magnitude_penalty
    - tool_call_cost
```

구체식 예시:

```text
R =
  w1 * Δfoot_sliding
+ w2 * Δground_penetration
+ w3 * Δjitter
+ w4 * Δbone_consistency
+ w5 * Δjoint_validity
- λ1 * motion_distortion
- λ2 * modified_frame_ratio
- λ3 * number_of_tool_calls
```

### 학습 단계 (Learning Progression)

바로 PPO 같은 multi-step RL로 시작하지 않는다. 1~3단계가 본 연구의 main scope이며, 4단계는 후속 확장이다.

```text
[Main scope - SCI 1차 논문]
1. Rule-based orchestrator baseline
2. Candidate tool 전부 적용 후 oracle best-tool label 생성
3. Supervised tool selector 또는 contextual-bandit (single-step learned policy)

[Extension - 후속 확장]
4. Multi-step RL orchestrator (PPO/DQN, credit assignment 포함)
```

---

## 8. 핵심 연구 질문

최종 연구 질문은 본문에서 직접 주장할 **Main RQ**와 보조 분석용 **Analysis Question**으로 나눈다.

```text
Main RQ1. artifact별 evaluator signal을 기반으로 correction tool을 선택하는 것이
         fixed post-processing보다 효과적인가?

Main RQ2. closed-loop tool-orchestrated refinement는 single-step correction 또는
         fixed pipeline보다 artifact reduction과 motion fidelity preservation 사이의
         trade-off를 더 잘 관리하는가?

Main RQ3. artifact state → correction action 매핑은 학습 가능한가?
         즉, supervised / contextual-bandit selector가 rule-based baseline 대비
         net gain을 의미 있게 개선하는가?
         (Learnable Routing Problem 검증)

Main RQ4. 제안 framework는 서로 다른 generator output에서도 적용 가능한
         generator-agnostic 특성을 보이는가?
         (zero-shot transfer와 small-calibration transfer를 함께 확인)

Analysis Q1. 다양한 motion generator output에서 skeleton-level artifact는
             어떤 형태로 나타나는가?

Analysis Q2. 새로운 evaluator/correction tool을 registry에 추가했을 때
             framework 성능 또는 artifact coverage가 확장되는가?

Analysis Q3. kinematic prior 기반 ordering/composition은 결과에
             의미 있는 보조 이득을 주는가?

Analysis Q4. ArtifactRouter는 이미 품질이 높은 motion(SOTA generator output)을
             통계적으로 훼손하지 않는 No-harm 운영 특성을 보이는가?
```

---

## 9. 실험 설계

### 9.1 Baseline

비교군은 다음과 같다.

```text
B1. Base generator only
B2. Base + global smoothing
B3. Base + fixed correction pipeline
B4. Base + learned post-hoc calibrator (optional strong baseline)
B5. Base + rule-based orchestrator
B6. Base + learned/contextual-bandit orchestrator
B7. Base + proposed orchestrator + learned calibrator as optional tool
B8. Oracle best tool
```

**Critical baseline.** B8(Oracle best tool)은 본 연구의 핵심 baseline이다.

- Oracle vs rule-based 차이가 작다 → orchestrator의 학습 가치 없음(stop signal)
- Oracle vs rule-based 차이가 크고 learned orchestrator가 oracle에 근접 → 강한 contribution

따라서 feasibility study에서 가장 먼저 측정해야 한다.

B4와 B7은 특정 방법을 중심 경쟁자로 세우기 위한 항목이 아니라, 구현과 데이터 조건이 맞는 경우 포함하는 strong reference baseline이다. 핵심 비교는 fixed post-processing, rule-based selection, learned selection, oracle selection 사이의 차이를 보는 것이다.

### 9.2 Generator

generator-agnostic 주장을 강하게 입증하기 위해, 단순히 generator 두 개를 쓰는 것을 넘어 **품질 수준과 paradigm을 의도적으로 다르게** 섞어 실험한다.

```text
A. 구조적 다양성 (paradigm coverage)
   - Token-based generator 1개   : MotionGPT / T2M-GPT
   - Diffusion/latent-based generator 1개 : MDM / MLD

B. 품질 수준 다양성 (quality-tier coverage)
   - High-quality SOTA generator        : 최신 강력한 text-to-motion 모델
                                          (artifact가 상대적으로 적음)
   - Low-quality / legacy / lightweight : 구형 또는 distill / 경량화 모델
                                          (artifact가 풍부하게 발생)
```

이 두 축을 함께 충족하도록 최소 3개 generator를 권장한다. 예시 구성:

```text
G1. SOTA diffusion-based generator   (high-quality reference)
G2. Token-based generator (e.g., T2M-GPT)
G3. Legacy/lightweight generator     (artifact-rich)
```

이렇게 구성하면 다음 두 주장을 동시에 검증할 수 있다.

1. **Improvement on artifact-rich motion**
   - G3와 같이 artifact가 많은 motion에서 ArtifactRouter가 큰 net gain을 보이는가?
2. **No-harm on high-quality motion**
   - G1과 같이 이미 품질이 높은 motion에서 ArtifactRouter가 motion을 훼손하지 않는가?
     (즉 selector가 적절히 STOP 또는 small-strength로 수렴하는가)

No-harm 정량 기준은 다음으로 둔다.

```text
High-quality generator(G1) 입력에 대해:
- ArtifactRouter 적용 후 FID_motion 변화 ≤ +ε (분포 손상 없음)
- correction magnitude 평균 ≤ low threshold
- user study에서 refined가 original보다 통계적으로 유의미하게 낮게 평가되지 않음
  (non-inferiority; preference/MOS 차이가 사전 정의한 δ 이내)
```

이 No-harm 결과는 단순 quality improvement 결과보다 **framework의 실용성**을 입증하는 데 결정적이며, "보정해야 할 때만 보정한다"는 ArtifactRouter의 핵심 운영 특성을 보여준다.

### 9.3 Metrics

#### Artifact Metrics

```text
foot sliding distance
ground penetration ratio
foot floating ratio
bone length variation
joint angle violation rate
velocity jitter
acceleration jerk
```

#### 9.3.1 Metric Definitions

Metric은 skeleton 좌표계와 fps가 정규화된 뒤 계산한다. 아래 식에서 `p_j(t)`는 frame `t`의 joint `j` 위치이고, `v_j(t) = p_j(t+1) - p_j(t)`, `a_j(t) = v_j(t+1) - v_j(t)`이다.

```text
Foot sliding distance:
contact 상태인 foot joint가 ground 근처에 있으면서 horizontal velocity가 큰 경우를 측정한다.

FootSliding =
  mean_t I(contact_foot(t)) * || p_foot_xy(t+1) - p_foot_xy(t) ||
```

```text
Ground penetration ratio:
ground plane 아래로 내려간 joint 또는 foot point의 깊이를 측정한다.

GroundPenetration =
  mean_t max(0, ground_y - min_j p_j_y(t))
```

```text
Foot floating ratio:
contact로 추정되는 구간에서 foot height가 ground보다 과도하게 높은 비율을 측정한다.

FootFloating =
  mean_t I(contact_foot(t)) * I(p_foot_y(t) - ground_y > tau_float)
```

```text
Bone length variation:
canonical bone length 대비 frame별 bone length 변화율을 측정한다.

BoneVar =
  mean_{t,b} | length_b(t) - length_b_ref | / length_b_ref
```

```text
Joint angle violation rate:
무릎, 팔꿈치 등 hinge-like joint가 anatomical range를 벗어난 비율을 측정한다.

JointViolation =
  mean_{t,k} I(angle_k(t) < lower_k or angle_k(t) > upper_k)
```

```text
Velocity jitter / acceleration jerk:
짧은 시간 구간에서 가속도 및 가속도 변화율(jerk)의 크기를 측정한다.
v_j(t+1) - v_j(t) = a_j(t), a_j(t+1) - a_j(t) = jerk_j(t) 이므로
아래 두 식은 각각 mean acceleration norm과 mean jerk norm에 해당한다.

VelocityJitter (mean acceleration magnitude) =
  mean_{t,j} || v_j(t+1) - v_j(t) || = mean_{t,j} || a_j(t) ||

AccelerationJerk (mean jerk magnitude) =
  mean_{t,j} || a_j(t+1) - a_j(t) || = mean_{t,j} || jerk_j(t) ||
```

최종 보고에서는 raw metric뿐 아니라 normalized score도 함께 사용한다.

```text
NormalizedArtifactScore_m =
  (metric_m - reference_mean_m) / (reference_std_m + epsilon)

TotalArtifactScore =
  sum_m w_m * NormalizedArtifactScore_m
```

이 score는 tool 선택, rollback 판단, iteration 종료 조건에 공통으로 사용한다.

#### Fidelity Metrics

Generated motion에는 GT가 없으므로 두 가지 protocol을 병행한다.

**Protocol A: Synthetic artifact injection**

```text
- HumanML3D / AMASS test set의 GT motion에 controlled artifact 주입
  (foot sliding, jitter, ground penetration 등)
- 본 framework로 refine → 원본 GT와 MPJPE 비교
- 이를 통해 "refinement quality"를 정량 측정 가능
```

**Protocol B: Generator output 기반**

```text
- correction magnitude (모션이 얼마나 바뀌었나)
- modified frame ratio
- semantic consistency (text prompt가 있는 경우 CLIP-based)
- user study (small-scale MOS, 50~100 sample) → perceptual grounding
```

**Protocol C: Distributional fidelity (분포 단위 평가)**

User study와 sample-level metric을 보완하기 위해, refined motion 분포가 실제 사람의 motion 분포에 얼마나 가까운지 분포 단위로 측정한다.

```text
- FID_motion (motion-specific Fréchet Inception Distance)
    HumanML3D 등에서 학습된 motion feature extractor의 latent에서
    GT 분포와 refined 분포 간 Fréchet distance 계산
    → 낮을수록 GT 분포에 가까움

- FGD (Fréchet Gesture Distance)
    motion auto-encoder latent feature에서의 Fréchet distance
    (특히 upper-body / gesture 평가에 표준적으로 사용됨)

- Diversity
    refined motion 간 latent feature pairwise distance 평균
    → 보정이 motion mode를 collapse시키지 않는지 확인

- MM-Dist (text-to-motion의 경우)
    text feature와 motion feature 간 multimodal distance
    → semantic alignment가 보정 후에도 유지되는지 확인
```

분포 단위 평가는 다음 두 가지 주장을 지지하는 핵심 근거가 된다.
1. 보정 후 motion이 단순히 metric만 좋아진 것이 아니라 실제 사람 motion 분포에 더 가까워졌다.
2. orchestrated correction이 motion diversity와 semantic alignment를 훼손하지 않는다.

#### Efficiency Metrics

단순 wall-clock runtime뿐 아니라, learned calibrator 등 단일 monolithic refinement와 동일 조건에서의 연산량 / 처리속도 비교를 함께 보고한다.

```text
- number of tool calls
- iteration count
- wall-clock runtime per sample (ms)
- FLOPs per sample
    base generator inference 제외, refinement stage의 누적 FLOPs
    (evaluator + selected correction tools + re-evaluation)
- FPS (effective throughput)
    refined motion produced per second on a fixed hardware (예: single GPU)
- Relative cost vs single-pass calibrator
    cost_ratio = cost(ArtifactRouter pipeline) / cost(single learned calibrator)
    이 값이 1을 넘으면 그만큼의 net gain 향상으로 정당화되어야 함
```

논문에서는 다음 효율-품질 trade-off plot을 핵심 figure 중 하나로 제시한다.

```text
x축: FLOPs per sample (또는 runtime)
y축: NetGain (artifact reduction − fidelity loss)

점:
  - global smoothing
  - fixed pipeline
  - single learned calibrator
  - rule-based ArtifactRouter
  - learned (contextual-bandit) ArtifactRouter
  - oracle best-tool (upper bound)
```

이를 통해 "ArtifactRouter는 단일 calibrator 대비 약간의 연산 비용을 추가하지만, 추가 비용을 상쇄하고 남는 net gain을 제공한다"는 점을 정량적으로 입증한다.

#### Explainability

```text
tool call trace
error → selected tool → before/after metric
```

### 9.4 Ablation Study Design

실험은 framework 전체 성능뿐 아니라, 어떤 구성요소가 실제 기여하는지 분리해서 확인한다. 단, 모든 ablation을 동등한 main contribution으로 주장하지 않고 **Main / Secondary / Optional**로 나눈다.

#### Main Ablations

논문 본문에서 핵심 contribution을 직접 뒷받침하는 실험이다.

| Experiment | 목적 | 핵심 비교 |
|---|---|---|
| E2 Tool effect matrix | artifact별 best tool이 다르게 나타나는지 확인 | 모든 sample에 모든 tool 적용 후 before/after 비교 |
| E3 Fixed vs selected correction | tool selection의 가치 검증 | global smoothing / fixed pipeline / rule-based selection |
| E4 Re-evaluation loop | single-step correction보다 loop가 유리한지 확인 | one-shot correction vs closed-loop refinement |
| E6 Learned selector (routing) | rule-based 대비 learned selector의 net gain | rule-based vs supervised/contextual-bandit selector |
| E7 Generator transfer | generator-agnostic claim 검증 | generator A에서 학습한 selector를 generator B에 zero-shot 및 small-calibration 적용 |

#### Secondary Ablations

논문의 설득력을 높이지만, main contribution의 필수 조건으로 과도하게 주장하지 않는 실험이다.

| Experiment | 목적 | 핵심 비교 |
|---|---|---|
| E1 Artifact prevalence | generator output에 artifact가 충분히 존재하는지 확인 | generator별 artifact metric 분포 |
| E9 Quality-tier sensitivity | 다양한 품질 수준의 generator에 대한 일반성 검증 | SOTA G1 / token-based G2 / legacy G3 각각에서의 net gain |
| E10 No-harm on SOTA motion | 이미 좋은 motion을 훼손하지 않음을 검증 | G1 입력의 FID_motion, correction magnitude, non-inferiority user study |
| E11 Efficiency–quality trade-off | 연산 비용 대비 품질 이득 정량화 | FLOPs/FPS vs NetGain 산점도(§9.3 plot) |
| E12 Distributional fidelity | refined 분포가 GT 분포에 가까워지는지 검증 | FID_motion, FGD, Diversity, MM-Dist의 before/after |

#### Optional Ablations

구현 여건이 맞거나 reviewer 대응이 필요할 때 추가하는 실험이다.

| Experiment | 목적 | 핵심 비교 |
|---|---|---|
| E5 Ordering/composition (KDG prior) | kinematic prior 기반 ordering의 효과 검증 | random ordering vs KDG-derived ordering vs no-ordering |
| E8 Optional calibrator reference | learned post-hoc calibrator와의 참고 비교 | calibrator only vs ArtifactRouter with/without calibrator tool |

각 ablation의 primary metric은 다음과 같이 둔다.

```text
NetGain =
  ArtifactReduction
  - alpha * FidelityLoss
  - beta * CorrectionMagnitude
  - gamma * ToolCallCost
```

```text
ArtifactReduction =
  TotalArtifactScore_before - TotalArtifactScore_after

FidelityLoss =
  synthetic protocol: MPJPE(refined, clean GT) - MPJPE(corrupted, clean GT)
  generator protocol: correction magnitude + semantic consistency loss
```

가중치 `alpha, beta, gamma`는 다음과 같이 설정한다.

- 1차로 synthetic injection protocol(Protocol A)에서 perceptual rating과의 상관(§U2)이 최대가 되도록 grid search로 결정한다.
- 모든 baseline과 ablation은 동일 weight set으로 비교한다.
- weight 민감도는 ±50% 변동 시 ranking이 바뀌지 않는지 sensitivity table로 부록에 보고한다.

Main ablations(E2, E3, E4, E6, E7)이 성립하면 `ArtifactRouter`가 단순 metric/tool 모음이 아니라 **artifact 상태에서 correction tool로의 학습 가능한 라우팅(Learnable Routing)을 제공하는 generator-agnostic orchestration framework**라는 핵심 주장이 가능하다.

Research question ↔ Ablation 대응은 다음과 같다.

```text
Main RQ1       → E2, E3
Main RQ2       → E4
Main RQ3       → E6
Main RQ4       → E7, E9

Analysis Q1    → E1
Analysis Q2    → tool registry 확장 시 E2/E3 재실행으로 검증
Analysis Q3    → optional E5
Analysis Q4    → E10, E12  (No-harm: 분포 + sample + perceptual)
```

Secondary ablations(E9, E10, E11, E12)는 실용성·안정성·분포 단위 품질을 보강하며, No-harm은 main contribution이 아니라 deployment-oriented safety analysis로 둔다. Optional ablations(E5, E8)는 KDG prior 효과와 learned calibrator reference처럼 reviewer가 요구할 수 있는 세부 비교에 대응하기 위한 확장 실험으로 둔다.

---

## 10. 불확실성 리스트

현재 연구에서 검토해야 할 핵심 불확실성은 다음과 같다.

### U1. 생성 모션에 artifact가 충분히 존재하는가?

기존 연구상 artifact 존재는 충분히 근거가 있으나, 실험 대상 generator에서 실제 발생 양상을 확인해야 한다.

### U2. 평가 지표가 정성 품질과 대응되는가?

metric high sample이 시각적으로도 부자연스러운지 확인해야 한다. Feasibility 단계에서 최소 30~50 sample에 대해 author 본인 + 동료 2~3명의 간이 perceptual rating을 metric score와 상관분석한다. 본 연구의 SCI manuscript 단계에서는 user study(N≥20)로 확장한다.

### U3. correction tool이 target artifact를 줄이는가?

foot_lock, smoothing, bone_projection 등 tool별 before/after 효과를 확인한다.

### U4. correction이 sample 단위 fidelity를 망가뜨리지 않는가?

artifact reduction과 fidelity loss 사이의 trade-off를 sample 단위에서 측정한다(MPJPE, correction magnitude, semantic consistency 등). 이는 H1(artifact-conditional necessity) 검증에 필수적이다.

### U5. artifact별 최적 tool이 다르게 나타나는가?

이것이 orchestrator 필요성의 핵심이다.

### U6. 단일 tool 호출보다 multi-step loop가 필요한가?

`evaluate → correct → re-evaluate` 구조가 실제로 유리한지 확인해야 한다.

### U7. rule-based보다 learned orchestrator가 필요한가?

1차 논문 main scope에서는 oracle-guided supervised selector 또는 contextual-bandit까지 학습된 orchestrator로 다루며, rule-based baseline과의 net gain 차이를 측정한다. Multi-step RL(PPO/DQN)은 §7에서 정의한 후속 확장으로 분리한다.

### U8. 단일 learned calibrator와 비교했을 때 독립적 가치가 있는가?

특정 post-hoc calibrator는 가능한 참고 baseline 또는 plug-in tool 예시로만 다룬다. 핵심은 특정 방법을 이기는 것이 아니라, artifact-conditioned selection과 re-evaluation이 fixed 또는 monolithic refinement보다 독립적인 가치를 갖는지 확인하는 것이다.

### U9. tool-extensible 구조가 실제로 작동하는가?

새로운 tool을 추가했을 때 orchestrator가 필요한 상황에서 해당 tool을 선택하고 성능이 확장되는지 확인한다.

### U10. generator-agnostic이 실제로 성립하는가?

최소 2개 generator에서 같은 interface로 동작해야 한다.

### U11. correction이 motion semantics를 훼손하지 않는가?

text-to-motion의 경우 prompt 의미를 보존하는지 확인해야 한다.

### U12. 단순 metric/tool 모음처럼 보이지 않는가?

공통 interface, registry, orchestrator, closed-loop, trace를 명확히 제시해야 한다.

### U13. Tool ordering이 결과에 의미 있는 영향을 주는가?

같은 tool set을 다른 순서로 적용했을 때 최종 motion이 유의미하게 달라지는지 확인해야 한다. ordering이 무의미하면 KDG 기반 ordering rule의 가치가 약해지므로, framework에서 ordering 관련 주장의 강도를 조정해야 한다(KDG prior를 보조 mechanism으로 명시).

### U14. ArtifactRouter가 high-quality motion에 대해 No-harm을 만족하는가?

H3(No-harm operating regime)의 정량 검증. SOTA generator output에 대해 ArtifactRouter 적용 후 다음을 확인해야 한다.

- 분포 단위: FID_motion / FGD가 통계적으로 유의미하게 악화되지 않음
- sample 단위: correction magnitude가 low threshold 이하
- perceptual: refined가 original 대비 non-inferior (사전 정의한 δ 이내)

만약 No-harm이 깨지면 selector의 STOP/abstain 정책이 강화되어야 하며, contribution 3에서 No-harm 주장 강도를 조정해야 한다.

---

## 11. Go / Stop 판단

최종 판단은 다음과 같다.

```text
Go.
단, SCI 본 연구로 바로 6~9개월 투입하지 말고,
4주 feasibility study를 통해 핵심 가정을 확인한 뒤 확장한다.
```

### Go 기준

다음 조건 중 대부분이 확인되면 본 연구로 확장한다. 수치는 초기 decision boundary로, 실험 결과에 따라 조정 가능하다.

```text
1. 생성 모션에서 artifact metric이 의미 있게 측정됨
   - 최소 2개 generator에서 적어도 한 artifact category가
     "no-artifact reference 대비 통계적으로 유의미한 수준"으로 검출됨

2. correction tool 적용 후 target artifact가 감소함
   - target artifact metric이 correction 후 평균 15% 이상 감소
     (artifact 종류별 최소 1개 tool이 이 기준 충족)

3. tool 적용 후 motion fidelity가 크게 망가지지 않음
   - correction magnitude(modified joints의 평균 displacement)가
     전체 motion 평균 displacement의 10% 이내
   - synthetic injection protocol에서 MPJPE 증가 < 5%

4. artifact별 효과적인 tool이 다르게 나타남
   - 최소 2개 artifact type에서 서로 다른 best tool이 관찰됨
     (= tool selection의 의미가 존재)

5. fixed smoothing보다 tool selection 방식이 유리함
   - Oracle best-tool이 fixed smoothing 대비 net gain
     (artifact reduction − fidelity loss) 기준 최소 10% 이상 우세

6. Orchestrator 학습 가치 존재
   - Oracle best-tool과 rule-based orchestrator 사이에 충분한 gap 존재
     (= learned orchestrator로 채울 여지가 있음)
```

### Stop 또는 축소 기준

다음이면 연구 범위를 축소해야 한다. 수치는 위 Go 기준의 역방향이다.

```text
1. correction tool이 artifact를 거의 줄이지 못함
   - 모든 tool에서 target artifact 감소율 < 5%

2. smoothing 하나로 대부분 해결됨
   - global smoothing이 net gain 기준 oracle best-tool의 90% 이상 도달

3. 단일 calibrator 또는 fixed pipeline이 proposed보다 압도적으로 강함
   - optional learned calibrator 또는 fixed pipeline이 oracle best-tool과 비슷하거나 우세

4. generator 하나에서만 작동하고 일반화가 어려움
   - generator를 바꿨을 때 동일 tool 효과 순위가 완전히 깨짐

5. tool orchestration이 단순 metric/tool 모음처럼 보임
   - 모든 artifact에 단일 best tool이 우세
     (= orchestrator의 conditional selection 가치 부재)
```

---

## 12. 예상 일정

### MVP / Feasibility Study: 4주

```text
Week 1:
- generator output 확보
- [T, J, 3] skeleton format 통일
- skeleton visualization 구축

Week 2:
- evaluator metrics 구현 (foot, jitter, bone 중심 최소 3종)
- artifact 분석
- correction tool 2~3개 prototype 구현

Week 3:
- Oracle best-tool 실험 (각 sample에 모든 tool 적용 후 best 선택)
- Tool-effect matrix 작성
- Rule-based orchestrator baseline 구현
- Oracle vs rule-based gap 측정 → 이게 작으면 즉시 stop signal

Week 4:
- Fixed smoothing baseline 비교
- Synthetic artifact injection 실험 1차
- 30~50 sample perceptual rating
- Go/Stop 결정
```

4주 feasibility의 범위는 E2/E3/E4에 집중한다.

```text
Phase 1 (Feasibility, 4주):
- E2 Tool effect matrix
- E3 Fixed vs selected correction
- E4 One-shot vs closed-loop

Phase 2 (Main paper expansion):
- E6 Learned selector
- E7 Generator transfer

Phase 3 (Secondary evidence):
- E9 Quality-tier sensitivity
- E10 No-harm
- E11 Efficiency
- E12 Distributional fidelity
```

### SCI manuscript 가능 일정

```text
MVP prototype: 2~3개월
논문용 실험 결과 (rule-based + contextual-bandit까지): 4~5개월
SCI manuscript 초안: 6개월
Multi-step RL 확장 포함 완성형: 7~9개월 이상
```

---

## 13. 최종 결론

이 연구는 시작해도 된다.

단, 연구 주제를 다음처럼 정확히 잡아야 한다.

```text
새로운 motion generator 개발 X
새로운 correction algorithm 하나 개발 X

기존/미래 motion generator의 출력 결과를 skeleton 수준에서 평가하고,
기존/신규 evaluator와 correction module을 tool로 등록하여,
현재 artifact 상태에 따라 적절한 tool을 선택·조합·재평가하는
Generator-Agnostic Tool-Orchestrated Motion Refinement Harness 개발 O
```

### 13.1 최종 논문 제목 (확정)

```text
ArtifactRouter:
  Artifact-Conditioned Tool Selection for
  Generator-Agnostic Human Motion Refinement
```

선정 이유:

```text
- ArtifactRouter        → 무엇을 하는지 직관적
- Artifact-Conditioned  → 핵심 novelty (artifact 상태 기반 선택)
- Tool Selection        → 단순 refinement가 아닌 orchestration임을 명시
- Generator-Agnostic    → 프레임워크의 일반성 강조
```

대안 후보(참고용, 본 연구에서는 채택하지 않음):

```text
- Compose-and-Correct:
  A Tool-Composition Framework for
  Multi-Artifact Human Motion Refinement

- Kinematic-Orchestrator:
  A Generator-Agnostic and Tool-Extensible Harness
  for Refining Generated Human Motion
  (보다 generic하여 심사자 입장에서 선명도가 떨어짐)
```

### 13.2 최종 Contribution (3개로 압축)

논문 본문에서 강조할 contribution은 다음 세 가지로 압축한다.

```text
1. Formulation of motion refinement as a learnable routing problem.
   We define generated-motion refinement as selecting a correction
   action a from artifact state s, and empirically test whether
   supervised / contextual-bandit selectors improve net gain over
   rule-based and fixed post-processing baselines.

2. Artifact-conditioned tool orchestration framework with a
   unified evaluator/correction tool registry, kinematic-prior-
   assisted conflict handling, and closed-loop re-evaluation with
   full tool-call trace.

3. Empirical validation that artifact-specific tool selection
   improves the trade-off between artifact reduction and
   motion fidelity — across multiple generators of differing
   quality — compared with fixed post-processing and monolithic
   calibration. We additionally analyze no-harm behavior on
   high-quality generator outputs as secondary deployment evidence.
```

### 13.3 핵심 한 줄

> 본 연구는 기존 모션 생성 모델을 대체하지 않고, 생성된 skeleton motion의 artifact 상태에서 적절한 correction tool을 선택하는 과정을 **학습 가능한 라우팅 문제(Learnable Routing Problem)**로 정식화하고 실험적으로 검증한다. 이 라우팅은 lightweight kinematic prior로 충돌과 순서를 보조하며, 다양한 generator output에서 artifact reduction과 motion fidelity의 trade-off를 개선하는 generator-agnostic refinement harness(**ArtifactRouter**)를 제안한다. No-harm 특성은 high-quality generator output에 대한 보조 안전성 분석으로 검증한다.
