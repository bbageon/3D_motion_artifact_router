# Action Space Provenance — ArtifactRouter (2026-05-25)

> 본 문서는 사용자 directive (2026-05-25): "5-level strength 의 연구적 위치 문서화 — strength 는 motion quality score 가 아니라 correction intensity parameter. tool 선택 = discrete action, strength = ordinal/continuous action parameter. 근거: parameterized action / hybrid action / continuous action discretization 연구." 정식 박제.
>
> [AGENTS.md §3-21 (Action Space Provenance Gate)](../AGENTS.md) 의 정식 출처. [`docs/metric_provenance.md`](metric_provenance.md) 와 동등 reference (metric 의 provenance vs action 의 provenance).

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
| **Continuous (future)** | float ∈ [0.0, 1.0+] | RL-2 후속 의 continuous policy 가능성 | parameterized / continuous action 연구 (DDPG, SAC, PA-DDPG 등) |

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
| `correction_tools/base.py` 의 `Strength` Literal type | 3-level + 5-level token 의 union: `Literal["small", "medium", "large", "xsmall", "small5", "medium5", "large5", "xlarge"]` |
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
| **RL-2 (Q-learning / value iteration)** | **TBD — 5-level oracle 결과 후 결정** | 10 or 16 |
| (future) Continuous action | tool × strength_value (float ∈ [0, 1+]) | discrete tool + continuous param (hybrid) |

### 5-1. RL-2 의 action space 결정 분기 (사용자 directive)

| Case | 조건 | RL-2 action space |
|---|---|---|
| **A** | 5-level oracle > 3-level oracle 의미 있게 우월 | **5-level (16 actions)** |
| **B** | NetGain 비슷, fidelity / correction magnitude 감소 | 5-level (over-modification 감소용) |
| **C** | 차이 거의 없음 | **3-level 유지** (10 actions), 5-level 은 appendix / future |

---

## 6. 외부 공개 인용 의 의무 (AGENTS.md §3-21)

본 프로젝트 의 action space 의 인용 시 의무:

| Item | 의무 |
|---|---|
| **Grid 명시** | "3-level prototype (small/medium/large)" 또는 "5-level (xsmall/small5/medium5/large5/xlarge)" 명시 |
| **STOP action 포함 여부** | "10 actions (STOP + 3 tool × 3 strength)" 또는 "16 actions (STOP + 3 × 5)" 명시 |
| **사용 stage** | RL-0 oracle / RL-1 imitation / RL-2 (TBD) 의 어느 stage 의 action space 인지 명시 |

본 의무 위반 시 silent invalidation ([AGENTS.md §6-5](../AGENTS.md) 동질).

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
