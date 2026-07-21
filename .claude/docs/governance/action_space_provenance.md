# Action Space Provenance — ArtifactRouter (2026-05-25; updated 2026-05-30)

> 본 문서는 사용자 directive (2026-05-25): "5-level strength 의 연구적 위치 문서화 — strength 는 motion quality score 가 아니라 correction intensity parameter. tool 선택 = discrete action, strength = ordinal/continuous action parameter. 근거: parameterized action / hybrid action / continuous action discretization 연구." 정식 박제.
>
> [AGENTS.md §3-21 (Action Space Provenance Gate)](../../../AGENTS.md) 의 정식 출처. [`.claude/docs/governance/metric_provenance.md`](metric_provenance.md) 와 동등 reference (metric 의 provenance vs action 의 provenance).

---

## 1. 본 문서 의 목적

ArtifactRouter 의 action space 의 정식 분류 + 근거 + grid 박제. RL-1 / RL-2 / oracle 의 action space 비교 의 single source.

---

## 2. Action Space 의 구조 (사용자 directive 박제)

### 2-1. Action 의 두 차원

| 차원 | 의미 | RL 영역 |
|---|---|---|
| **Tool selection** | discrete action (FootLock / BoneProjection / VelocitySmoothing / STOP) | discrete action policy |
| **Strength** | ordinal / continuous action parameter (correction intensity) | ordinal action / continuous action discretization |

본 분리는 **Parameterized Action 또는 Hybrid Action** 의 framework 와 일치 (Masson et al. 2016 "Reinforcement Learning with Parameterized Actions", AAAI 2016).

### 2-2. Strength 의 위치 — Correction Intensity Parameter (NOT quality score)

**중요 박제**:
- **Strength 는 motion quality score 가 아님**.
- Strength 는 tool 의 correction intensity 의 parameter — interpolation factor / smoothing sigma 등 의 scalar.
- 본 프로젝트 의 STRENGTH_FACTOR / STRENGTH_SIGMA 의 mapping 의 의미.

### 2-3. Discretization Grid 의 분류

| Grid | 정의 | 의미 | 출처 |
|---|---|---|---|
| **3-level (prototype)** | small / medium / large (factor 0.3 / 0.6 / 1.0; sigma 0.5 / 1.0 / 2.0) | coarse discretization. Step 1-5 의 prototype. 사용자 framing: "3-level 은 prototype coarse discretization". | 본 프로젝트 자체 (initial design, 2026 Q1) |
| **5-level (RL-2 candidate)** | xsmall / small / medium / large / xlarge (factor 0.2 / 0.4 / 0.6 / 0.8 / 1.0; sigma 0.4 / 0.8 / 1.2 / 1.6 / 2.0) | finer discretization. RL-2 candidate action space. | 본 프로젝트 자체 (2026-05-25 사용자 directive) + Hybrid Action / Continuous Discretization 연구 spirit |
| **Bounded continuous u** | normalized float `u ∈ `0.0, 1.0]` | grid label 대신 correction intensity 를 연속 parameter 로 표현. 실제 학습/평가는 sampled grid 또는 continuous optimizer 의 coverage 를 별도 기록 | parameterized / continuous action 연구 (DDPG, SAC, PA-DDPG 등) |
| **Dense-grid proxy** | e.g. `{0.0, 0.1, ..., 1.0}` 또는 finer boundary grid | continuous surface 를 직접 검증하기 위한 sampled action-effect grid. continuous policy 와 동일하게 인용 금지 | 본 프로젝트 Stage A/B diagnostic |

---

## 3. Strength Grid 의 정확한 Mapping

본 프로젝트 의 각 tool 별 strength → internal factor 의 mapping. RL agent 의 action space 의 정확한 정의.

### 3-1. FootLockTool

| Level | Token | Factor (interpolation) |
|---|---|---|
| 3-level small | `small` | 0.3 |
| 3-level medium | `medium` | 0.6 |
| 3-level large | `large` | 1.0 |
| 5-level xsmall | `xsmall` | **0.2** |
| 5-level small (5) | `small5` | **0.4** |
| 5-level medium (5) | `medium5` | **0.6** |
| 5-level large (5) | `large5` | **0.8** |
| 5-level xlarge | `xlarge` | **1.0** |

**Implementation note**: 3-level 의 `large` (1.0) 와 5-level 의 `xlarge` (1.0) 가 동일 factor. 3-level 의 `medium` (0.6) 와 5-level 의 `medium5` (0.6) 동일.

### 3-2. BoneProjectionTool

| Level | Token | Factor (projection) |
|---|---|---|
| 3-level | small/medium/large | 0.3 / 0.6 / 1.0 |
| 5-level | xsmall/small5/medium5/large5/xlarge | 0.2 / 0.4 / 0.6 / 0.8 / 1.0 |

(FootLockTool 와 같은 grid.)

### 3-3. VelocitySmoothingTool

| Level | Token | Sigma (Gaussian smoothing) |
|---|---|---|
| 3-level small | `small` | 0.5 |
| 3-level medium | `medium` | 1.0 |
| 3-level large | `large` | 2.0 |
| 5-level xsmall | `xsmall` | **0.4** |
| 5-level small (5) | `small5` | **0.8** |
| 5-level medium (5) | `medium5` | **1.2** |
| 5-level large (5) | `large5` | **1.6** |
| 5-level xlarge | `xlarge` | **2.0** |

**Implementation note**: 3-level 의 `large` (2.0) = 5-level 의 `xlarge` (2.0). 3-level 의 `small` (0.5) 와 5-level 의 `xsmall` (0.4) 약간 다름 — 5-level grid 의 균등 spacing.

---

## 4. Backward Compatibility 의 의무

기존 3-level token (small/medium/large) 의 backward compatibility 유지 — 부록 A-HH 의 모든 결과 의 reproducibility 보존.

| Implementation | 의무 |
|---|---|
| `correction_tools/base.py` 의 `Strength` Literal type | 3-level + 5-level token 의 union: `Literal`"small", "medium", "large", "xsmall", "small5", "medium5", "large5", "xlarge"]` |
| `STRENGTH_FACTOR` / `STRENGTH_SIGMA` dict | 8 token 의 mapping (3-level 3개 + 5-level 5개) |
| 기존 3-level 호출 (`strength="small"`) | factor 0.3 (변경 없음) |
| 새 5-level 호출 (`strength="xsmall"`) | factor 0.2 (신규) |

본 의무 충족 시 기존 부록 (M, N, O, R, S, T, V, X, CC 등) 의 모든 결과 의 reproducibility 보존. 5-level oracle 결과 는 별도 snapshot (e.g., `oracle_sequence_multi_5level_v1.json`) 로 분리 보고.

---

## 5. Action Space 의 RL Stage 별 사용

| Stage | Action space | Dimension |
|---|---|---|
| RL-0 (single-step oracle, 부록 1단계) | tool ∈ {FootLock, BoneProjection, VelocitySmoothing} × strength ∈ {small, medium, large} = **9 actions** | 9 |
| RL-0 (sequence oracle, 부록 M, N) | 9 + STOP = **10 actions** | 10 |
| **RL-0 (5-level oracle, 본 directive)** | tool × 5-level + STOP = **3 × 5 + 1 = 16 actions** | 16 |
| RL-1 (imitation policy, 부록 O) | 10 actions (3-level) | 10 |
| **RL-2 learned (safe imitation) — PRIMARY** | **3-level (10 actions, STOP + 3 tool × 3 strength)** | **10** |
| RL-2 oracle ceiling / future constrained RL | 5-level (16 actions) | 16 |
| RL-2 continuous action-effect surface (Stage A+) | tool × normalized intensity `u ∈ `0, 1]` sampled on recorded `u_grid` | discrete tool + bounded continuous param (hybrid). 학습/평가 grid 를 반드시 기록 |

### 5-1. RL-2 의 action space 결정 — oracle ceiling vs learned policy 의 분리 (사용자 승인 2026-05-28)

**정식 결정** (사용자 승인 게이트 통과, 2026-05-28):

| 측면 | grid | 근거 |
|---|---|---|
| **RL-2 learned policy (PRIMARY)** | **3-level (10 actions)** | learned closed-loop NetGain +0.171 > 5-level +0.134, oracle gap closure 83% > 54%, strength_match 62% > 27% (``reports/2026-05-28.md`](../../../reports/2026-05-28.md)) |
| **RL-2 oracle ceiling / future RL** | 5-level (16 actions) | oracle ceiling +0.246 > 3-level +0.207 (synthetic Case A, ``reports/2026-05-26.md`](../../../reports/2026-05-26.md)). 단 imitation 으로 회수 못 함 (bottleneck) |

#### 5-1-1. 2026-05-26 의 "RL-2 = 5-level (Case A)" 결정 의 reframe

- **2026-05-26 의 Case A 결정** = **oracle 기준** (synthetic 5-level oracle > 3-level oracle, p=3.28e-10). 본 결정 은 **oracle ceiling 의 우월** 을 의미 — 유효.
- **2026-05-28 의 learned policy 비교** = 5-level 의 oracle advantage 가 learned policy 로 transfer 안 됨 (strength fine-grained matching 이 imitation bottleneck, strength_match 27% vs 62%).
- **정정**: RL-2 의 **현재 learned policy primary = 3-level**. 5-level 은 **oracle upper-bound + future constrained RL (Q-learning/value iteration) 의 ceiling 회수 target** 으로 유지.

#### 5-1-2. 핵심 message (외부 공개)

> "Finer action grids increase oracle headroom, but may reduce learned policy performance under limited imitation data." / "Although the 5-level grid provides a higher oracle ceiling, the 3-level grid yields better learned closed-loop performance by reducing strength-selection errors."

근거: behavioral cloning 의 action space 가 클수록 label coverage 더 필요 + fine-grained discretization 의 optimization 난이도 (Masson et al. AAAI 2016 ; Tang & Agrawal AAAI 2020 ; PhysDiff Yuan et al. ICCV 2023 의 parameterized/physical action separation spirit).

### 5-2. RL-2 reframe — Bounded Continuous Action-Effect Surface `Q(s, tool, u)` (사용자 directive 2026-05-29, 2026-05-30 refined)

**정식 reframe** (사용자 directive 2026-05-29 박제, 2026-05-30 하네스 통제 표현 보정): RL-2 의 main formulation 을 discrete action **classification** (`policy(s) → action class`) 에서 **bounded continuous action-effect surface learning** 으로 전환. 단, 하네스는 policy 를 ranker / direct selector / risk estimator / oracle teacher 중 하나로 고정하지 않는다. 실험마다 `selection_mode` 와 validation 조건을 기록해 claim 을 제한한다.

> "바로 continuous policy 를 학습하는 게 아니라, bounded continuous action-effect surface 를 학습하는 방향이 맞다. 목표는 `policy(s) → action class` 가 아니라 `Q_safe(s, tool, u)` 학습."

#### 5-2-1. Formulation

- **u ∈ `0, 1]**: bounded continuous **intervention intensity** (normalized correction strength). tool 별 factor 와 별개의 정규화 축 — tool mapper 가 u → tool-specific factor 로 변환.
- **`Q(s, tool, u)` / `Q_safe(s, tool, u)`**: state s 에서 tool 을 intensity u 로 적용했을 때의 predicted utility / validation-aware utility surface. `Q_safe` 는 physical gate 를 통과한 action-effect 를 우선하도록 만든 shorthand 이며, learned score 자체가 physical safety certificate 라는 뜻은 아니다.
- **추론/평가 mode**: direct argmax, top-k candidate ranking, risk-filtered selection, physical-gate-validated execution, oracle diagnostic 등 여러 mode 를 허용한다. 단 모든 report 는 `selection_mode`, `gate_recheck`, `candidate_trace` 를 기록해야 한다.

논문 문장 (단일 출처):

> "We learn a validation-aware action-effect surface `Q(s, tool, u)` over bounded continuous intervention intensity u ∈ `0,1]. At inference time, the orchestration module selects or ranks tool-intensity candidates under the recorded validation mode, and final quality is evaluated with physical and standard motion metrics."

#### 5-2-2. u → strength factor mapping (normalized intensity 의무, 사용자 risk 대응)

사용자 risk 박제: "u 의 의미가 tool 별로 다름 → normalized u + tool mapper 명시". u 는 정규화 축이고, 각 tool 의 physical factor 는 §3 의 tool-specific mapping 으로 결정.

| u | 3-level token (Stage 1) | 의미 |
|---|---|---|
| 0.0 | (STOP / no-op) | 개입 없음, utility 0, 항상 safe |
| 0.3 | small | 약한 개입 |
| 0.6 | medium | 중간 개입 |
| 1.0 | large | 강한 개입 |

각 tool 의 small/medium/large 의 실제 factor 는 tool-specific (§3-1 FootLock blend, §3-3 VelocitySmoothing sigma {0.5, 1.0, 2.0} 등) — u 는 그 위의 normalized index.

**연속 u 매핑 (Stage A+, dense/continuous grid)**: discrete token 대신 continuous u 를 직접 tool factor 로 변환 (correction tool 의 `metadata={"continuous_factor": ...}` / `{"continuous_sigma": ...}` override, backward compatible):

| tool | u → tool intensity | u=0 | u=1 |
|---|---|---|---|
| FootLockTool | `continuous_factor = u` | identity (no-op) | factor 1.0 (= xlarge) |
| BoneProjectionTool | `continuous_factor = u` | identity | factor 1.0 (= xlarge) |
| VelocitySmoothingTool | `continuous_sigma = 2.0·u` | identity (sigma≤0 short-circuit) | sigma 2.0 (= xlarge) |

연속 매핑은 u=0 → 모든 tool identity (STOP anchor), u=1 → xlarge factor 와 일치. Stage 1 의 3-point token 매핑 (u=0.3→small 등) 과 u=1.0 에서 동일, 중간값은 linear (token sigma 와 약간 차이, 예 u=0.3 continuous sigma 0.6 vs token small 0.5). 연속 매핑이 Stage A+ 의 canonical formulation.

#### 5-2-3. Staged 학습 계획 (grid-sampled action effects → continuous)

| Stage | u_grid | 방법 | 목적 |
|---|---|---|---|
| **Stage 1** | {0.3, 0.6, 1.0} | grid Q surface + reranking | 기존 3-level 결과를 Q/reranking formulation 으로 재현, imitation 대비 설명력, STOP threshold calibration |
| **Stage A/A+** | coarse train grid + fine holdout grid | action-effect transition + continuous argmax confirmation | unseen u 보간 가능성, gate boundary, false-safe risk 진단 |
| **Stage B** | boundary-focused dense grid | hard-example mining | high-utility unsafe / pass-fail boundary / STOP-vs-weak 애매 case 보강 |
| Stage 2 | {0.0, 0.1, …, 1.0} 또는 mined dense grid | dense grid Q 재학습 | tool 별 utility curve / safe-unsafe boundary / surface smoothness |
| Stage 3 | continuous | line search / Bayesian opt / golden-section | dense surface 가 smooth 확인 후 continuous argmax |
| Stage 4 | continuous | constrained offline RL (CQL/IQL) | 최종 policy optimization (OOD value overestimation 대응 conservative) |

핵심 원리: **continuous 를 목표로 하되, 데이터는 grid-sampled action effects 로 만든다** — continuous formulation 의 연구 가치를 살리면서 데이터 부족 + unsafe exploration 회피.

#### 5-2-4. 기존 3/5-level 결과의 재해석 (중심 → motivation)

| 기존 결과 | 새 해석 |
|---|---|
| 5-level oracle > 3-level oracle | fine intensity headroom 존재 |
| 5-level learned < 3-level learned | discrete fine-class imitation 은 어려움 (strength matching bottleneck) |
| 3-level learned 안정적 | coarse grid baseline |
| `Q_safe(s, tool, u)` | discrete class 대신 action-effect surface 학습 |

#### 5-2-5. 근거 (AGENTS.md §3-22)

- Parameterized action RL (discrete tool + continuous parameter): Masson et al. **AAAI 2016**, Hausknecht & Stone **ICLR 2016** (§7-1).
- Offline RL 의 OOD action value overestimation → conservative/reranking 우선 (CQL/IQL 는 Stage 4): Kumar et al. CQL **NeurIPS 2020**, Kostrikov et al. IQL **ICLR 2022**.
- **남는 불확실성** (`internal proxy assumption`): Q surface 가 실제로 smooth 한지, dense grid 가 standard metric preservation 과 일치하는지, continuous argmax 가 gate boundary 근처에서 안정적인지는 실험으로 확인. safe_utility 는 **Category C** (internal routing reward, [metric_provenance.md](metric_provenance.md)) — 외부 공개 최종 성능 근거 금지.

#### 5-2-6. Validation mode 및 physical gate 재검증 기록 의무 (사용자 risk 박제)

> "Q 가 높다고 바로 믿으면 안 된다. 추론 시에도 physical gate 를 실제로 한 번 더 돌려야 한다."

본 문장은 특정 아키텍처 역할을 영구 고정하려는 것이 아니라, **learned score 만으로 safety/final quality claim 을 하지 않기 위한 하네스 통제 규칙** 이다.

허용 mode:

| Mode | 의미 | 인용 가능 claim |
|---|---|---|
| `diagnostic_no_gate` | learned Q/P_safe 만으로 선택하거나 평가 | utility surface / risk head diagnostic only |
| `ranked_candidates` | policy 가 top-k 후보를 제안 | candidate efficiency / top-k recall |
| `risk_filtered` | learned risk head 로 후보를 낮추거나 제외 | risk forecasting 보조 evidence |
| `gated_execution` | 실제 tool 적용 후 physical gate 재검증, pass 시 실행 / fail 시 rollback 또는 다음 후보 | physical safety + closed-loop quality evidence |
| `oracle` | dense/grid search 로 upper bound 계산 | ceiling / regret reference |

`gated_execution` 을 사용한 경우 candidate_trace 에 rejected candidate 와 executed action 을 분리 기록한다. `diagnostic_no_gate` 결과는 physical safety 또는 final quality 의 sole evidence 로 인용하지 않는다.

---

### 5-3. Strength `u` 의 정식 정의 (RootGaitConsistencyTool) — AR-081/085

**정의 (단일 출처)**: strength `u` = **접지-일관 root 변위 보정의 적용 분율**.

- **선형 성질 (증명·unit 검증)**: 위치 offset(u) = `cumsum(v_new − v_orig)` = `u · cumsum(v_target − v_orig)` = **`u · offset(u=1)`** — u 에 정확히 선형 (u 범위 무관, [test_u_linear_in_displacement_including_over_correction](../../../tests/unit/correction_tools/test_root_gait_consistency.py)).
- **눈금 의미**: u=0 = STOP (무변경). u=1 = **우리 접지 추정 기준으로 stance foot world 속도 = 0 을 완전 강제**. **⚠️ u=1 은 GT-최적이 아니라 "제약 포화점"** — 접지 추정이 낮게 잡히면 u=1 도 잔여 deficit 이 남고(under), 반대로 GT 초과(over, AR-077 ratio>1.5 18.2%)도 가능. 따라서 **최적 strength 는 u=1 이 아닐 수 있고, 이것이 Q(s,u) 정책이 필요한 근본 이유.**
- **단위/부호**: dimensionless. 물리 변위(m)로 환산 = u · |offset(u=1)| (correction_magnitude 로 기록).

**Action space 이력:**

| stage | u 범위 | u_grid | 등록 |
|---|---|---|---|
| **strength-Q-v1** (AR-081) | `[0, 1]` | {0, .25, .5, .75, 1.0} | 2026-07-13 |
| **strength-Q-v2** (AR-085) | **`[0, 2]`** (over-correction 개방) | **{0, .25, .5, .75, 1.0, 1.25, 1.5, 2.0}** | 2026-07-22 |

| 항목 | 값 |
|---|---|
| `action_space_type` | `bounded_continuous_u` (단일 tool: RootGaitConsistencyTool) |
| stage | one-shot contextual Q (closed-loop MDP 아님) |
| u-mapper | `continuous_u` blend (v_new = (1−u)·v_orig + u·v_target; u>1 = extrapolation). tool 버전 = `U_MAX_CONTINUOUS=2.0` (AR-085). u≤1 backward-compat (AR-072/077/081/082 불변) |
| state | 13차원 의미 상태 ([AR-081 spec](../dashboard-task-specs/AR-081-strength-q-v1.md) — 직관 이름 naming 단일 출처) |
| 인용 의무 | `action_space_type=bounded_continuous_u` + `u_grid` + u_max + one-shot scope 명시. RL-2 구계열(3/5-level·G2·NetGain) 혼합 인용 금지 (§6-15). v1(u≤1) 결과와 v2(u≤2) 결과는 u_max 로 구분 인용 |

**AR-085 개방 근거** (사용자 directive 2026-07-22 "열어보자 — 과보정이 아닐 수도 있어"): u=1 의 접지-일관 제약은 **우리 접지 추정**에 의존하며, 추정이 stance frame 을 놓치거나 median/interp 로 target 을 낮추면 u=1 이 여전히 under-correction 일 수 있음. u>1 개방은 "u=1 이 상한이다"라는 미검증 가정을 제거하고 데이터로 최적 strength 를 발견 (AR-085 사전등록 검정). Neural Additive Models (Agarwal et al., NeurIPS 2021) — additive Q 해석 근거.

## 6. 외부 공개 인용 의 의무 (AGENTS.md §3-21)

본 프로젝트 의 action space 의 인용 시 의무:

| Item | 의무 |
|---|---|
| **Action space type 명시** | `discrete_3level`, `discrete_5level`, `dense_grid_proxy`, `bounded_continuous_u` 중 하나 |
| **Grid / domain 명시** | "3-level prototype", "5-level", "`u_grid={0.0,0.1,...,1.0}`", "`u∈`0,1]`" 등 |
| **STOP action 포함 여부** | "10 actions (STOP + 3 tool × 3 strength)" 또는 "bounded u with STOP/no-op anchor" 명시 |
| **사용 stage** | RL-0 oracle / RL-1 imitation / RL-2 learned / Stage A transition / Stage B mining / continuous argmax 등 |
| **Validation mode** | direct / ranked / risk-filtered / gated / oracle diagnostic 중 어느 mode 인지 명시 |

본 의무 위반 시 silent invalidation ([AGENTS.md §6-5](../../../AGENTS.md) 동질).

---

## 7. References (간단)

### 7-1. Parameterized / Hybrid Action

- **Masson et al. 2016**, "Reinforcement Learning with Parameterized Actions", **AAAI 2016** — parameterized action space (discrete + continuous parameter).
- **Hausknecht & Stone 2016**, "Deep Reinforcement Learning in Parameterized Action Space", **ICLR 2016** — PA-DDPG.
- **Wei et al. 2018**, "Hierarchical Approaches for Reinforcement Learning in Parameterized Action Space" — H-PA-DDPG, H-MPO.

### 7-2. Continuous Action Discretization

- **Tang & Agrawal 2020**, "Discretizing Continuous Action Space for On-Policy Optimization" — continuous → discrete grid.
- **Tavakoli et al. 2018**, "Action Branching Architectures for Deep Reinforcement Learning" — multi-dim action discretization.

### 7-3. Motion Generation 의 Discrete vs Continuous Control

- **Holden et al. 2020**, "Learned Motion Matching", **ACM TOG 2020** — motion-level discrete vs continuous trade-off.
- **Peng et al. 2021**, "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control" — continuous action 의 motion control.

본 references 는 본 프로젝트 의 5-level discretization 의 spirit 의 정당화. 본 프로젝트 의 grid (factor 0.2-1.0) 의 정확한 formulation 은 본 프로젝트 자체 변형.

---

## 8. 본 문서 의 유지 의무

- 본 action_space_provenance.md 는 **action space 의 single source**.
- Tool / strength / grid 변경 시 본 표 + AGENTS.md §3-21 + 부록 II 동시 갱신 의무.
- 외부 공개 (논문·발표) 시 본 표 의 grid + RL stage 명시 의무.
