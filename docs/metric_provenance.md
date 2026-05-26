# Metric Provenance Table — ArtifactRouter (2026-05-25)

> 본 문서는 사용자 directive (2026-05-25 Metric Citation Gate) 의 정식 reference. 본 프로젝트 의 모든 평가 / reward metric 의 출처 + 본 프로젝트 구현 변형 + 최종 평가 / proxy 분류 의 단일 출처.
>
> **AGENTS.md §3-20 (Metric Citation Gate)** 의 정식 출처. 외부 공개 (논문·발표) 의 metric 인용 시 본 문서 reference.
> [`docs/current_research_position.md`](current_research_position.md) 의 §2 (Evaluator), §3 (NetGain) 도 본 문서와 동기화.

---

## 1. 본 문서 의 목적

ArtifactRouter 의 모든 metric 을 다음 **3 Category** 로 분류:

| Category | 정의 | 외부 공개 적절성 |
|---|---|---|
| **A. Standard metric** | top-tier 논문 (CVPR, NeurIPS, ICCV, ECCV, ACM MM 등) 의 standard metric. 본 프로젝트 가 동일 정의 사용. | **외부 공개 최종 성능 근거 가능** (논문 인용). |
| **B. Variant metric** | top-tier 논문 의 metric 을 명확히 변형. 변형 의도 + 차이점 명시. | 외부 공개 가능, 단 **variant 임을 명시 의무**. |
| **C. Proxy metric** | 본 프로젝트 자체 정의 — internal routing reward / diagnostic 용. standard reference 없음 또는 매우 약함. | **외부 공개 최종 성능 근거 금지**. 내부 reward / diagnostic only. |

본 분류 의 의무 — [AGENTS.md §3-20](../AGENTS.md):
- 모든 평가 / reward metric 은 본 표에 등록.
- 새 metric 추가 시 본 표 update 의무.
- Category C 의 metric 결과 는 외부 공개 의 sole evidence 금지.

---

## 2. Category A — Standard Metric (top-tier 논문 인용)

본 절 의 metric 은 **외부 공개 최종 성능 근거 로 인용 가능**. 단 본 프로젝트 가 **아직 정식 구현 / 측정 안 함** — Step 8/9 의 후속 작업 의 의무.

### 2-1. FID_motion (Fréchet Inception Distance for motion)

| Item | Value |
|---|---|
| **Source (general)** | Heusel et al. 2017, "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (NeurIPS 2017) — FID 원조 |
| **Source (motion)** | Guo et al. 2022, "Generating Diverse and Natural 3D Human Motions from Text" (CVPR 2022) — HumanML3D paper, motion VAE latent 기반 FID |
| **Source (motion 후속)** | Tevet et al. 2023 (MDM), Jiang et al. 2023 (MotionGPT, NeurIPS 2023) |
| **Formula (original)** | `FID = ||μ_real - μ_gen||^2 + Tr(Σ_real + Σ_gen - 2(Σ_real Σ_gen)^(1/2))` 의 motion VAE latent space |
| **본 프로젝트 구현** | **미구현** — Step 9 (G1 generator + final evidence) 의 의무 |
| **분류** | **A. standard** (구현 후) |
| **용도** | distributional fidelity (real vs generated motion distribution) — 최종 성능 |
| **Caveat** | HumanML3D 의 official motion VAE 사용 의무 (다른 latent 으로 변형 시 variant 분류) |

### 2-2. R-Precision

| Item | Value |
|---|---|
| **Source** | Guo et al. 2022 (CVPR 2022 HumanML3D) |
| **Formula** | text-motion retrieval 의 top-K accuracy. ground truth text 가 generated motion 의 top-K closest text candidate 에 포함 되는 비율. |
| **본 프로젝트 구현** | **미구현** — text-to-motion alignment 평가 의 standard |
| **분류** | **A. standard** (구현 후) |
| **용도** | text-motion semantic alignment |
| **Caveat** | T2M motion encoder 의 retrieval 의무 (HumanML3D official) |

### 2-3. Matching Score / MM-Dist (Multi-modality Distance)

| Item | Value |
|---|---|
| **Source** | Guo et al. 2022 (CVPR 2022 HumanML3D), Petrovich et al. 2023 (TMR) |
| **Formula** | text feature 와 motion feature 의 multimodal latent distance (R-precision 의 기반 distance) |
| **본 프로젝트 구현** | **미구현** |
| **분류** | **A. standard** (구현 후) |
| **용도** | text-motion alignment |

### 2-4. Diversity

| Item | Value |
|---|---|
| **Source** | Guo et al. 2022 (HumanML3D), Lee et al. 2019 ("Dancing to Music", NeurIPS 2019) |
| **Formula** | sample 간 latent feature pairwise distance 평균 |
| **본 프로젝트 구현** | **미구현** |
| **분류** | **A. standard** (구현 후) |
| **용도** | generated motion 의 다양성 |

### 2-5. Multimodality

| Item | Value |
|---|---|
| **Source** | Guo et al. 2022 (HumanML3D) |
| **Formula** | 같은 text 에 대한 multiple generated sample 의 latent distance — text 별 sample diversity |
| **본 프로젝트 구현** | **미구현** |
| **분류** | **A. standard** (구현 후) |
| **용도** | text-conditioned diversity |

### 2-6. MPJPE (Mean Per-Joint Position Error)

| Item | Value |
|---|---|
| **Source** | Ionescu et al. 2014 (Human3.6M), Pavllo et al. 2019 (VideoPose3D), HumanML3D paper |
| **Formula** | `MPJPE = mean_{t,j} || p_pred(t,j) - p_GT(t,j) ||` |
| **본 프로젝트 구현** | ✅ 구현 (FidelityLoss Protocol A 의 기반 — [reproducibility-checklist §3-3](../.claude/skills/reproducibility-checklist/SKILL.md)) |
| **분류** | **A. standard** |
| **용도** | per-joint position error (Protocol A: vs clean GT; Protocol B simplified: vs original generator output) |
| **Caveat** | 본 프로젝트 의 Protocol B simplified 는 "MPJPE vs original_g2" — original 이 GT 아님. **variant 로 표기 의무** ([§3 의 Protocol B simplified](#3-2-fidelity-loss-protocol-b-simplified) 참조). |

---

## 3. Category B — Variant Metric (top-tier 변형)

본 절 의 metric 은 top-tier 의 metric 을 변형. 외부 공개 시 **variant 명시 의무**.

### 3-1. Foot Skating / Sliding / Contact Error

| Item | Value |
|---|---|
| **Source (original)** | Holden et al. 2017 (PFNN), Karunratanakul et al. 2023 (GMD) — foot contact + sliding metric. Tevet et al. 2023 (MDM) — foot skating. |
| **Source (recent comprehensive)** | **PP-Motion: Physical-Perceptual Fidelity Evaluation for Human Motion Generation** (ACM MM 2025) — foot contact / skating / penetration 의 unified evaluation framework. |
| **본 프로젝트 변형** | **FootFloatingEvaluator** (`evaluators/foot_floating_evaluator.py`) — simple Y-threshold (tau_float=0.05). **Sliding 의 horizontal velocity component 미포함**. |
| **분류** | **B. variant** (현재 구현은 partial), **C. proxy** (외부 공개 시 caveat) |
| **용도** | foot artifact diagnostic |
| **CAVEAT (부록 Z 발견)** | 현재 구현 의 corruption robustness 부족 — synthetic `inject_foot_floating(0.08)` 이 evaluator max score 거의 안 올림. **standard foot skating / sliding metric 으로 보강 의무** (Item 6). |

### 3-2. Fidelity Loss Protocol B simplified

| Item | Value |
|---|---|
| **Source (original)** | MPJPE (Ionescu 2014) |
| **본 프로젝트 변형** | `FidelityLoss_B_simplified = MPJPE(refined, original_g2)` — clean GT 대신 generator output 을 reference 로. |
| **분류** | **B. variant** |
| **용도** | G2 natural 의 correction magnitude proxy (clean GT 없음) |
| **Caveat** | 외부 공개 시 "Protocol B simplified — MPJPE vs original generator output (variant)" 명시 의무. |

### 3-3. Bone Length Variation

| Item | Value |
|---|---|
| **Source (related)** | Petrovich et al. 2021 (ACTOR, ICCV 2021) — skeleton consistency. Holden et al. 2016 — bone length consistency. |
| **본 프로젝트 변형** | **BoneLengthEvaluator** — per-bone normalized variation (1st frame 또는 canonical reference). |
| **분류** | **B. variant** |
| **용도** | skeleton consistency (bone stretch / shrink artifact) |
| **Caveat** | 일부 motion generation 논문 의 skeleton constraint 와 유사 spirit, 단 본 프로젝트 의 per-bone scoring 의 정확한 formula 는 본 프로젝트 변형. |

### 3-4. Acceleration / Jerk (Temporal Smoothness)

| Item | Value |
|---|---|
| **Source** | Flash & Hogan 1985 ("Coordination of Arm Movements" — minimum-jerk principle). Holden et al. 2017 (PFNN). Zhang et al. 2022 (MotionDiffuse) — jerk-based smoothness. |
| **본 프로젝트 변형** | **VelocityJitterEvaluator** — per-joint mean acceleration norm. AccelerationJerk — per-joint mean jerk. |
| **분류** | **B. variant** (close to standard) |
| **용도** | temporal smoothness diagnostic |
| **Caveat** | mean acceleration norm 의 normalization 차이 (per-joint 평균 vs total) — 외부 공개 시 정확한 formula 인용 의무. |

### 3-5. PhysicalGateV0 — 5 Evaluator (Safe Orchestration Layer B, 2026-05-26 revised)

본 절 의 evaluator 는 [`current_research_position.md §0-4`](current_research_position.md) 의 Safe Orchestration architecture 의 **Evaluator Layer B (Physical Constraint Gate)**. **NetGain 의 weight term 이 아닌 hard gate 의 decision (accept / repair / rollback / STOP)** 로 사용.

**Calibration approach** (사용자 directive): HumanML3D clean N=500~1000 sample 의 distribution → p50/p90/p95/p99 추출 → unsafe threshold = p99 (또는 p95 보수적).

**Gate decision rule** (예):
```python
if Penetrate_after > max(Penetrate_before + eps, CleanP99_Penetrate):
    rollback
if BoneLengthCV_after > max(BoneLengthCV_before + eps, CleanP99_BoneCV):
    rollback
if JerkSpike_after > max(JerkSpike_before * 1.05, CleanP99_Jerk):
    rollback
```

#### 3-5-1. PenetrateEvaluator (gate)

| Item | Value |
|---|---|
| **Source (recent SOTA, 2020+)** | **PhysDiff** (Yuan et al. 2023, **ICCV**) — "Physics-Guided Human Motion Diffusion Model". ground penetration 의 physics-guided projection. **HuMoR** (Rempe et al. 2021, **ICCV**) — ground-aware fitting. |
| **본 프로젝트 정의** | foot/ankle joint Y < ground_y - penetrate_eps 의 ratio (per-frame, per-foot). |
| **분류** | **B. variant** |
| **용도** | Layer B gate decision. accept/rollback boundary 의 정량 기준. |
| **Caveat** | ground_y estimator (Skeleton Normalizer 또는 motion 의 minimum Y 의 heuristic) 의 정확성 의존. |

#### 3-5-2. FloatEvaluator (gate)

| Item | Value |
|---|---|
| **Source (recent, 2020+)** | **PhysDiff** (Yuan et al. 2023, **ICCV**) — foot floating. **MDM** (Tevet et al. 2023, **ICLR**) — foot contact loss. |
| **본 프로젝트 정의** | contact 추정 frame 중 foot height > float_threshold 의 ratio. 기존 FootFloatingEvaluator 의 gate-form (threshold 의 의미 정정 + clean calibration 의무). |
| **분류** | **B. variant (gate-form)** |
| **용도** | Layer B gate decision. |
| **Caveat** | contact heuristic 의 robust 함이 prerequisite (현재 velocity_based v1.2.0). Item 6 contact estimator 도입 후 정식 보강. |

#### 3-5-3. SkateEvaluator (gate)

| Item | Value |
|---|---|
| **Source (recent SOTA, 2020+)** | **PhysDiff** (Yuan et al. 2023, **ICCV**) — foot sliding metric. **MDM** (Tevet et al. 2023, **ICLR**) — foot skating evaluation. |
| **본 프로젝트 정의** | contact 추정 frame 중 foot horizontal velocity > skate_threshold (예: 0.05 m/frame) 의 ratio. |
| **분류** | **B. variant** |
| **용도** | Layer B gate decision. foot sliding artifact 의 정량 (foot floating 과 orthogonal 차원). |
| **Caveat** | skate threshold 의 walking 의 stance phase 와 swing phase 의 boundary 의존 — clean calibration 의무. |

#### 3-5-4. JerkSpikeEvaluator (gate)

| Item | Value |
|---|---|
| **Source (recent, 2020+)** | **MDM** (Tevet et al. 2023, **ICLR**) — velocity smoothness loss. **VIBE** (Kocabas et al. 2020, **CVPR**) — temporal smoothness baseline. **TCMR** (Choi et al. 2021, **CVPR**) — acceleration/jerk metric 의 정식 formulation. |
| **본 프로젝트 정의** | per-joint acceleration norm 의 p95 (motion-wide, normalized). 기존 VelocityJitterEvaluator (mean) 의 spike-only (p95) 변형. |
| **분류** | **B. variant** |
| **용도** | Layer B gate decision. spike 가 mean 보다 over-modification 식별 적합. |
| **Caveat** | normalization (per-joint vs total) 의 정확한 formula — 외부 공개 시 명시. |

#### 3-5-5. BoneLengthCVEvaluator (gate)

| Item | Value |
|---|---|
| **Source (recent, 2020+)** | **HuMoR** (Rempe et al. 2021, **ICCV**) — bone length consistency. **MDM** (Tevet et al. 2023, **ICLR**) — geometric loss. **ACTOR** (Petrovich et al. 2021, **ICCV**) — skeleton constraint spirit. |
| **본 프로젝트 정의** | per-bone length 의 coefficient of variation (std/mean) across frames, max over bones. 기존 BoneLengthEvaluator (mean variation) 의 gate-form (CV-based + max-bone). |
| **분류** | **B. variant (gate-form)** |
| **용도** | Layer B gate decision. bone stretch/shrink 의 정량. |
| **Caveat** | CV (std/mean) 의 robust 함이 mean=0 의 case 에서 한계 (실제로는 bone length 가 0 인 case 없음). |

---

## 4. Category C — Proxy Metric (본 프로젝트 자체 정의)

본 절 의 metric 은 **외부 공개 최종 성능 근거 금지**. 내부 routing reward / diagnostic only.

### 4-1. NetGain (calibrated_protocol_a_v1)

| Item | Value |
|---|---|
| **Source** | **본 프로젝트 자체 정의 — no direct standard reference**. |
| **Related framework (spirit, not direct)** | Holden et al. 2020 ("Learned Motion Matching") — multi-cost framework. Ng et al. 1999 ("Policy Invariance Under Reward Transformations") — reward shaping. |
| **Formula** | `NetGain = ArtifactReduction - α·FidelityLoss - β·CorrectionMag - γ·ToolCost` (α=5.0, β=γ=0, calibrated_protocol_a_v1). |
| **본 프로젝트 구현** | ✅ ([reproducibility-checklist §3-4](../.claude/skills/reproducibility-checklist/SKILL.md)). |
| **분류** | **C. proxy** — **internal routing reward only**. |
| **용도** | **policy optimization reward** for RL-1 / RL-2. **NOT final motion quality metric**. |
| **CAVEAT** | NetGain 의 결과 의 외부 공개 인용 시 의무 동반: "NetGain is a proxy reward, not a standard motion quality metric. Final quality is validated by standard metrics (FID, R-Precision, MM-Dist) + visual/perceptual rating." |

### 4-2. FootFloatingEvaluator (current)

| Item | Value |
|---|---|
| **Source** | 본 프로젝트 자체 — simple Y-threshold |
| **Formula** | `score = mean_t I(foot_y(t) > tau_float=0.05)` |
| **분류** | **C. proxy (diagnostic only)** |
| **CAVEAT** | 부록 Z 발견: corruption robustness 부족, evaluator limitation. **외부 공개 시 'diagnostic proxy, not final foot quality metric' 명시 의무**. **Item 6 contact estimator 도입 후 Category B 의 foot skating / sliding metric 으로 대체 / 보강**. |

### 4-3. BoneLengthEvaluator (current)

| Item | Value |
|---|---|
| **Source** | 본 프로젝트 자체 — per-bone std variation |
| **분류** | **C. proxy** (Category B 의 ACTOR 의 skeleton constraint spirit 와 close, 단 정확한 formula 는 본 프로젝트 변형) |
| **CAVEAT** | 외부 공개 시 'bone consistency diagnostic proxy' 명시. ACTOR / SMPL 의 standard formulation 으로 대체 가능 (후속). |

### 4-4. VelocityJitterEvaluator (current)

| Item | Value |
|---|---|
| **Source** | 본 프로젝트 자체 (Category B 의 jerk metric spirit 와 close) |
| **Formula** | per-joint mean acceleration norm |
| **분류** | **C. proxy** (Category B 와 close, 단 정확한 normalization 차이) |
| **CAVEAT** | 외부 공개 시 'temporal smoothness diagnostic proxy' 명시. Standard jerk metric 의 정확한 formulation 으로 보강 가능. |

### 4-5. Total Artifact Score

| Item | Value |
|---|---|
| **Source** | 본 프로젝트 자체 — `sum_m metric_m` (sum of all evaluator max scores) |
| **분류** | **C. proxy** |
| **용도** | refinement loop 의 Score 비감소 의무 (AGENTS.md §3-4) 의 monotonicity check |
| **CAVEAT** | 외부 공개 인용 금지. |

---

## 5. 향후 evidence 보강 plan (사용자 directive 박제)

### 5-1. Standard metric 도입 의무 (외부 공개 prerequisite)

| Order | Metric | Source | 작업 |
|---|---|---|---|
| 1 | **FID_motion** | Guo 2022 (HumanML3D), Tevet 2023 (MDM) | HumanML3D official motion VAE encoder 도입 + FID 측정 |
| 2 | **R-Precision + Matching Score** | Guo 2022 | HumanML3D official text-motion encoder 도입 |
| 3 | **Diversity + Multimodality** | Guo 2022 | sample-level pairwise distance |
| 4 | **Foot skating / sliding / contact error** | PP-Motion (ACM MM 2025), MDM | FootFloating 대체 / 보강 (Item 6 contact estimator 결합) |

### 5-2. 본 프로젝트 의 proxy metric 의 위치 정정

| Metric | 현재 위치 | 정정 후 위치 |
|---|---|---|
| FootFloatingEvaluator | "evaluator score" | "diagnostic proxy — Category C" |
| BoneLengthEvaluator | "evaluator score" | "diagnostic proxy — Category C" |
| VelocityJitterEvaluator | "evaluator score" | "diagnostic proxy — Category C" |
| NetGain | "performance metric" | **"internal routing reward — Category C"** |

### 5-3. RL-2 진입 전 의 정식 prerequisite

본 directive 박제 후 RL-2 진입 의 의무:
- **NetGain = reward only (Category C)**. RL-2 의 reward.
- **최종 성능 (RL-2 vs baseline) 의 evidence = Category A (FID, R-Precision, MM-Dist) + visual/perceptual rating (perceptual pilot Step 6-7) + Category B variants (foot skating, jerk)**.
- Category C metric 의 결과 만으로 "ArtifactRouter 우월" 단정 금지.

---

## 6. 본 문서 의 유지 의무

- 본 metric_provenance.md 는 **외부 공개 evidence 의 single source**.
- 새 metric 추가 / 변경 시 본 표 + AGENTS.md §3-20 동시 갱신 의무.
- 외부 공개 (논문·발표·README) 결과 인용 시 본 표 의 Category 명시 의무.

---

## 7. References (간단 인용)

- **Guo et al. 2022**, "Generating Diverse and Natural 3D Human Motions from Text", **CVPR 2022** — HumanML3D paper.
- **Tevet et al. 2023**, "Human Motion Diffusion Model" (MDM), **ICLR 2023**.
- **Jiang et al. 2023**, "MotionGPT: Human Motion as a Foreign Language", **NeurIPS 2023**.
- **Heusel et al. 2017**, "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", **NeurIPS 2017** — FID 원조.
- **Ionescu et al. 2014**, "Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments", **IEEE TPAMI** — MPJPE.
- **Pavllo et al. 2019**, "3D Human Pose Estimation in Video with Temporal Convolutions and Semi-Supervised Training", **CVPR 2019** — VideoPose3D.
- **Holden et al. 2017**, "Phase-Functioned Neural Networks for Character Control", **ACM TOG 2017** — foot contact / sliding.
- **Holden et al. 2016**, "A Deep Learning Framework for Character Motion Synthesis and Editing", **ACM TOG 2016** — bone length.
- **Holden et al. 2020**, "Learned Motion Matching", **ACM TOG 2020** — multi-cost framework (NetGain related spirit).
- **Karunratanakul et al. 2023**, "Guided Motion Diffusion for Controllable Human Motion Synthesis" (GMD), **ICCV 2023** — foot sliding.
- **Petrovich et al. 2021**, "Action-Conditioned 3D Human Motion Synthesis with Transformer VAE" (ACTOR), **ICCV 2021** — skeleton consistency.
- **Petrovich et al. 2023**, "TMR: Text-to-Motion Retrieval" — Matching Score 후속.
- **Yoon et al. 2020**, "Speech Gesture Generation from the Trimodal Context" — FGD.
- **Akhter & Black 2015**, "Pose-Conditioned Joint Angle Limits for 3D Human Pose Reconstruction", **CVPR 2015** — anatomical joint limits.
- **Loper et al. 2015**, "SMPL: A Skinned Multi-Person Linear Model", **ACM TOG 2015** — anatomical joint limits.
- **Flash & Hogan 1985**, "The Coordination of Arm Movements: An Experimentally Confirmed Mathematical Model", **Journal of Neuroscience 1985** — minimum-jerk principle.
- **Lee et al. 2019**, "Dancing to Music", **NeurIPS 2019** — Diversity in motion synthesis.
- **Yuan et al. 2023**, "PhysDiff: Physics-Guided Human Motion Diffusion Model", **ICCV 2023** — ground penetration / floating / foot sliding 의 physics-guided projection. PhysicalGateV0 의 Penetrate / Float / Skate 의 직접 motivation.
- **Rempe et al. 2021**, "HuMoR: 3D Human Motion Model for Robust Pose Estimation", **ICCV 2021** — ground-/contact-aware fitting + bone length consistency. PhysicalGateV0 의 BoneLengthCV + contact-aware Penetrate/Float 의 motivation.
- **Kocabas et al. 2020**, "VIBE: Video Inference for Human Body Pose and Shape Estimation", **CVPR 2020** — temporal smoothness 평가 baseline. PhysicalGateV0 의 JerkSpike 의 motivation.
- **Choi et al. 2021**, "Beyond Static Features for Temporally Consistent 3D Human Pose" (TCMR), **CVPR 2021** — acceleration/jerk metric 의 정식 formulation. PhysicalGateV0 의 JerkSpike 의 보조 reference.
- **Guo et al. 2024**, "MoMask: Generative Masked Modeling of 3D Human Motions", **CVPR 2024** — HumanML3D standard metric (FID/R-Prec/MM-Dist/Diversity) 의 최신 application. Step E (Standard Metric Integration) reference.
- **Zhang et al. 2022**, "MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model" — jerk / foot artifact.
- **PP-Motion 2025**, "PP-Motion: Physical-Perceptual Fidelity Evaluation for Human Motion Generation", **ACM MM 2025** — foot artifact unified evaluation.
- **Ng et al. 1999**, "Policy Invariance Under Reward Transformations", **ICML 1999** — reward shaping (NetGain related spirit).
- **CVPR 2025 examples** — recent T2M generation papers using HumanML3D / KIT-ML standard metrics.
