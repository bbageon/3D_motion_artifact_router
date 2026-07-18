# Current Research Position — ArtifactRouter (2026-05-25, revised 2026-05-26)

> 본 문서는 사용자 directive (2026-05-25, "NetGain validity 검증 plan") 의 Step 1: 현재 연구 의 위치 를 단일 문서로 고정. 외부 공개 (논문·발표·README) 의 evidence 인용 시 본 문서를 reference.
>
> AGENTS.md §3-17 (evidence 계층) + §3-18 (baseline family protocol) + 부록 P/U/Y/Z/AA/BB 의 통합 reference.
>
> **2026-05-26 revision (사용자 directive — "safe orchestration framing"):**
> - 본 프로젝트 의 정식 framing 은 **"NetGain-only"** 가 아니라 **"safe orchestration"**.
> - **NetGain = internal routing reward only**, 최종 quality 지표 아님 — §0 신설 (최상위 framing).
> - **RL-2 objective = `maximize artifact_improvement subject to physical_validity + no_harm`**, NOT `maximize NetGain`.
> - **Physical constraint gate** (BoneLengthViolation / GroundPenetration / ContactConsistency / JerkSpike) 의 의무 — §3-6 신설.
> - Step 7 (다음 우선순위) 의 분기 변경 — Step A-F (사용자 6-step priority).

---

## 0-0. 확정 연구 가닥 — Artifact–Tool Pair 구조 (2026-07-08 사용자 directive)

> 사용자 결정 (2026-07-08): "명확한 pair 1개 + 보조 pair 4개로 연구 가닥을 좁힌다."
> 본 축소는 타협이 아니라 **측정이 만든 정리** — 각 행이 실측 근거와 1:1 대응.
> 6축 균등 취급 framing 을 대체. 가설 본문(H-2026-204/205/206)은 무변경 (§3-11 비대상 —
> evidence claim 의 범위 조정). P5 (AR-058-5) 의 주장 구조는 본 절을 따른다.

| 위상 | Pair | 실측 근거 | Caveat |
|---|---|---|---|
| **명확 (primary) — ⚠️ 2026-07-12 지각 검증 실패, 대증요법 재해석** | **foot skating → contact-aware cleanup / combo(+BoneProj)** | MDM CI-clean 개선 (P1~P4 완주) — **물리 지표(Cat B) 개선은 유효** | **A/B b1 2회 모두 우연** (v1 11/20 슬로모·v2 combo 10/20 실속도, 마지막 재검정 소진) → anchoring 계열 = **대증요법** (증상 억제, 병인 미해결). 병인 후보 = root 전진 부족 (AR-071: MDM v_root = GT의 42%, 100% prompt, 부족↔fs ρ=+0.34 p<1e-5; 방향 서명은 불확정) |
| **핵심 pair — 완주 (2026-07-13 P5 최종화)** | **root progression collapse (MDM) → root-aware correction (AR-072) + generator/state 조건부 적용** | 기전 0.42→1.02 (GT-free) · Cat-A 삼중 유의 개선 (R@1 +0.047·MM −1.0·FID 7.2→3.3, R-Prec 수정 후) · **blind A/B 19/20** · **VQ 는 deficit 없음 + 고정 적용 시 손상 (AR-077, pool-scoped)** — 고정 적용 불가의 직접 근거 | anchoring(증상) 우연 vs root(병인) 압도 = intervention evidence (인과 아님). **per-motion gate 미완 (AR-078 "한계 확정"**: richer state AUC 0.722 vs 0.674 유의 개선하나 harm@30% 감소 비유의; 순위≠안전 축 분리) — "routing 작동" 표현 금지. b1(AR-073 진행)·MDM·locomotion·단일 벤치마크·δ 조건부. 종합 = [P5 최종본](../../../evals/reports/2026-07-12_poc_necessity_p1_p5.md) |
| 보조 | penetration → ground projection | 유병률 실측 희소 (occur ≤1.8%, AR-063) | **tool 미구현 정당** — 유병률이 실재할 때만 설계 |
| 보조 | BoneCV → BoneProjectionTool | 메커니즘 정합 (audit 2-4) | side-effect guard 의무 (§6-12) + gate 무판별 → **raw 값 guard 로만** |
| 보조 | jitter → VelocitySmoothingTool | 메커니즘 정합 (audit 2-5) | **secondary 위상** — smoothing 보상 함정 기지 |
| 보류 | floating → (FootLock, raw guard 로만) | intent 오염 (AR-062 A-6) + P3 미검 | AR-066-lite 경량 판정 + human 라벨 후 재평가 — 그 전 routing 주장에 불사용 |

**운영 함의**: (a) perceptual A/B 검증은 primary pair 에 집중 (보조 pair 는 지각 claim 자체를 안 함). (b) routing 주장 = "primary 에서 조건부 보정 + 그 외에서 STOP/guard" — Safe Orchestration §0 framing 과 정합 (보정하지 않을 때를 아는 것이 시스템의 절반). (c) 버린 축 없음 — 각 보조/보류 pair 는 측정된 이유와 함께 보존 (§3-13).

## 0. 정식 Framing — **Safe Orchestration (NOT NetGain-only)** (사용자 directive 2026-05-26)

### 0-1. 본 framing 의 정식 정의

**ArtifactRouter 의 정식 framing** (2026-05-26 박제):

> Generator-agnostic motion refinement 의 **safe orchestration** —
> artifact reduction 을 추구하되, **physical validity + no-harm** 의 hard safety constraint 를 만족하는 조건 하에서.

**핵심 변경**:
1. NetGain (Category C internal routing reward) 만으로는 motion 의 **physical validity 또는 perceptual quality 를 보장하지 않음**.
2. NetGain reward hacking 가능 — artifact score 만 줄이고 motion 을 왜곡하는 over-modification 경로.
3. 따라서 RL-2 objective 는 **constrained optimization**:
   ```
   maximize    artifact_improvement      (proxy via NetGain or similar)
   subject to  physical_validity_gate    (hard, NOT in reward weight)
               no_harm_gate              (semantic / perceptual)
   ```
4. Physical gate 결정: `accept` / `repair` / `rollback` / `STOP`.

### 0-2. 본 framing 의 evidence motivation (사용자 directive)

> "G2에서 artifact profile은 관찰되었다. NetGain 기준으로 artifact reduction은 가능하다. 하지만 NetGain-only correction은 정성적으로 왜곡/과보정/physical inconsistency를 만들 수 있다."

본 우려 의 검증 의무 (사용자 directive Step B, 우선순위 최상):
- G2 top correction case (motion_006, _007, _008, _028 — G2 natural n=50 의 oracle non-STOP top 4) 의 **side-by-side 시각화** (overlay 아닌 좌우).
- NetGain 높은 G2 보정이 실제로 좋아 보이는지 vs 다리 길이/자세/리듬 왜곡 가시 확인.

**중요 caveat — diagnostic vs representative** (사용자 directive 2026-05-26 박제):
- 본 4 sample 은 **NetGain top 4 의 의도적 enriched subset** — diagnostic, NOT representative.
- 적절한 인용: "NetGain-top G2 correction case 의 visual distortion sanity check (n=4 diagnostic)".
- 부적절한 인용 (금지): "G2 에서 보정이 잘 된다" (representative claim 금지).
- 전부 good 이어도 Step E (Standard Metric Integration) 의 의무 유지.

### 0-3. 본 framing 의 reference papers (2020+ peer-reviewed top-tier, AGENTS.md §3-22 의무)

본 safe orchestration framing 의 정식 정량 motivation — **7 papers** (revised 2026-05-26):

| Ref | 빌려온 개념 | 본 프로젝트 적용 |
|---|---|---|
| **MDM** (Tevet et al. 2023, **ICLR**) — `openreview](https://openreview.net/forum?id=SJ1kSyO2jwu) | motion generation 에서 geometric loss / velocity / foot contact 의 **별도 축** 처리 | NetGain 의 ArtifactReduction / FidelityLoss 분리 (Protocol A/B) 의 spirit 일관. Physical gate evaluator (BoneLengthCV / JerkSpike) 가 fidelity-orthogonal 차원. |
| **PhysDiff** (Yuan et al. 2023, **ICCV**) — `CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_PhysDiff_Physics-Guided_Human_Motion_Diffusion_Model_ICCV_2023_paper.html) | floating / foot sliding / ground penetration 의 **physics-guided projection**. Physical plausibility 는 scalar reward 안 넣고 **별도 constraint / guidance** 로 처리. | §3-6 PhysicalGateV0 의 PenetrateEvaluator / FloatEvaluator / SkateEvaluator 의 직접 motivation. NetGain reward weight 가 아닌 **hard gate** 로 분리하는 결정 의 근거. |
| **HuMoR** (Rempe et al. 2021, **ICCV**) — `project](https://geometry.stanford.edu/projects/humor/) | ground-/contact-aware fitting + bone length consistency | §3-6 의 BoneLengthCVEvaluator 의 직접 motivation + Penetrate / Float 의 contact-aware spirit. |
| **VIBE** (Kocabas et al. 2020, **CVPR**) — `CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Kocabas_VIBE_Video_Inference_for_Human_Body_Pose_and_Shape_Estimation_CVPR_2020_paper.html) | temporal smoothness 평가 의 baseline (jerk-based) | §3-6 의 JerkSpikeEvaluator 의 직접 motivation. |
| **TCMR** (Choi et al. 2021, **CVPR**) — `CVF PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_Beyond_Static_Features_for_Temporally_Consistent_3D_Human_Pose_and_CVPR_2021_paper.pdf) | acceleration / jerk metric 의 정식 formulation (temporal consistency) | §3-6 의 JerkSpikeEvaluator 의 보조 정량 reference. |
| **HumanML3D** (Guo et al. 2022, **CVPR**) — `CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html) | **FID, R-Precision, MM-Dist, Diversity, Multimodality** standard metrics 의 motion quality 평가. Artifact score 만으로 최종 품질 단정 안 함. | Step E (Standard Metric Integration) 의 직접 의무 — 외부 공개 prerequisite. NetGain (Category C) ≠ 최종 quality 의 정식 근거. |
| **MoMask** (Guo et al. 2024, **CVPR**) — `CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_MoMask_Generative_Masked_Modeling_of_3D_Human_Motions_CVPR_2024_paper.pdf) | HumanML3D standard metric 의 최신 application — generative motion SOTA pipeline. FID / R-Prec / MM-Dist / Diversity 의 의무 column. | Step E 의 reference implementation (HumanML3D official evaluator 재사용 가능성). |

**7 papers 의 통합 message**: motion quality 의 정식 평가 = (a) standard metric (FID/R-Prec/MM-Dist/Diversity) + (b) physical plausibility (penetrate/float/skate/jerk/bone consistency) + (c) generation diversity. NetGain (Category C internal routing reward) 단독 인용 = misalignment with field-standard evaluation framework.

### 0-4. Safe Orchestration Architecture — 정식 정의 (2026-05-26 신설)

본 §0 의 framing 이 **architecture diagram** 으로 명시되며, 외부 공개 (논문·발표·README) 의 architecture figure reference 의무.

```
`Generator Output]
    G1 (diffusion, future) / G2 (MotionGPT, current)
            ↓
`Canonical Motion Normalizer]
    SMPL-22, root-relative, fps=20  (AGENTS.md §3-1)
            ↓
`Evaluator Layer]
    A. Artifact Evaluators        (current: FootFloating / BoneLength / VelocityJitter)
    B. Physical Constraint Gate   (new: Penetrate / Float / Skate / JerkSpike / BoneLengthCV)
    C. Final Quality Metrics      (future: FID / R-Precision / MM-Dist / Diversity / Multimodality + perceptual)
            ↓
`Artifact Router / Policy]
    state  = artifact_scores + physical_scores + prev_action + remaining_budget
    action = STOP or correction_tool × strength  (16 actions, AGENTS.md §3-21)
            ↓
`Correction Candidate]
    FootLock / BoneProjection / VelocitySmoothing / future repair tools
            ↓
`Physical Gate Decision]
    if safe       → accept
    elif repair   → physical refinement agent (strength↓ or alt-tool)
    else          → rollback or STOP
            ↓
`Closed-loop Re-evaluation]
    repeat until STOP / budget / no improvement  (AGENTS.md §3-4)
```

**Role separation table** (사용자 directive 박제):

| 구성 | 역할 | 연구적 의미 |
|---|---|---|
| Artifact Evaluator | 국소 artifact 감지 | "어디가 문제인가?" |
| NetGain | internal routing reward (Category C) | "개입할 가치가 있는가?" |
| Physical Gate | 안전성 필터 (post-correction) | "고쳤지만 망가지지 않았는가?" |
| Orchestrator / RL Policy | action sequence 선택 | "무엇을, 얼마나, 몇 번?" |
| Standard Metrics | 최종 품질 평가 (Category A) | "전체 motion quality 가 유지/개선됐는가?" |
| Perceptual Rating | 사람 판단 검증 | "눈으로 봐도 납득 가능한가?" |

### 0-5. RL-2 의 새 Objective — Constrained Optimization (2026-05-26 신설)

기존: `maximize NetGain` → **수정**: **Constrained Optimization**:

```
maximize    ArtifactReduction - cost
subject to  Penetrate     <= CleanP99_Penetrate     + eps
            Float         <= CleanP99_Float         + eps
            Skate         <= CleanP99_Skate         + eps
            JerkSpike     <= CleanP99_JerkSpike     * 1.05
            BoneLengthCV  <= CleanP99_BoneLengthCV  + eps
            FidelityLoss  <= max_fidelity_loss
```

**Penalty reward form (alternative formulation)**:

```
R = + ArtifactReduction
    - λ₁ FidelityLoss
    - λ₂ CorrectionMagnitude
    - λ₃ ToolCallCost
    - λ₄ PhysicalViolation       (gate violation count or magnitude)
    - λ₅ RollbackPenalty         (gate-triggered rollback count)
```

**논문 표현 권장** (사용자 directive): **constrained optimization 쪽이 더 안전**.
- "NetGain 을 올렸다" → claim weak.
- **"unsafe correction 을 배제하면서 artifact intervention 을 선택했다"** → claim strong.

### 0-3. 본 framing 의 외부 공개 의무 (AGENTS.md §3-17 일관)

- **"NetGain-only" 표현 금지** — 모든 5단계 리포트 / 논문 의 결론 절은 "safe orchestration" framing 으로 진술.
- **NetGain 비교 결과 단독 인용 금지** — physical validity gate 의 evidence 또는 perceptual rating 의 evidence 동반 의무.
- Physical gate 미도입 상태 의 RL-2 결과 = "preliminary diagnostic" only — 외부 공개 인용 보류.

---

---

## 1. 본 연구 의 정체성 (한 줄)

ArtifactRouter 는 **canonicalized motion artifact state 위에서 cost · risk 를 고려해 correction intervention 또는 STOP 을 선택하는 generator-agnostic, tool-extensible decision system** 이다. 새 motion generator 도, 새 단일 correction algorithm 도 개발하지 않는다.

## 2. Evaluator 의 위치 — **Category C Proxy Metric (AGENTS.md §3-20)**

### 2-1. 현재 evaluator 의 정의

- **FootFloatingEvaluator**: simple Y-threshold (tau_float=0.05) — **Category C proxy**.
- **BoneLengthEvaluator**: per-bone normalized variation — **Category C proxy** (Category B 의 ACTOR skeleton constraint spirit 와 close, 단 정확한 formula 변형).
- **VelocityJitterEvaluator**: per-joint mean acceleration norm — **Category C proxy** (Category B 의 jerk metric 와 close, 단 normalization 차이).

(상세 — [`.claude/docs/governance/metric_provenance.md`](metric_provenance.md), [reproducibility-checklist SKILL §3](../../skills/reproducibility-checklist/SKILL.md))

### 2-2. 현재 evaluator 의 위치 (AGENTS.md §3-20)

- **Category C — Diagnostic proxy metric**. 최종 motion quality 의 ground truth 아님.
- 본 프로젝트 의 routing policy / RL agent 의 reward signal 로 사용 가능 (proxy).
- **외부 공개 (논문·발표) 에서 본 evaluator 의 결과를 "motion quality" 또는 "최종 성능 근거" 로 인용 금지**.
- 본 evaluator score 의 변화가 시각/사람 평가 와 일치하는지는 별도 검증 (Step 6 perceptual pilot).

### 2-3. 발견된 limitation (2026-05-25)

- **FootFloatingEvaluator 의 corruption robustness 부족** (부록 Z): synthetic `inject_foot_floating(0.08)` 이 max score 를 거의 안 올림 (synthetic median 0.005 < clean median 0.021). 글로벌 Y shift 가 evaluator metric 과 mismatch. **Item 6 (contact estimator) 의 정량 motivation**.
- 본 limitation 후 외부 공개 시 FootFloating 결과 의 caveat 동반 의무.

### 2-4. 향후 보강 (AGENTS.md §3-20 + metric_provenance.md §5-1)

본 evaluator 의 Category C 위치 정정 후, 외부 공개 prerequisite:
- **Category A / B 의 standard metric 도입**:
  - Foot skating / sliding / contact error (PP-Motion ACM MM 2025, MDM) — FootFloating 보완 / 대체.
  - Bone length consistency 의 ACTOR / SMPL 의 standard formulation.
  - Jerk metric 의 정확한 formulation (Flash & Hogan 1985).
- 본 도입 후 외부 공개 의 최종 성능 evidence 가능.

## 3. NetGain 의 위치 — **Category C Internal Routing Reward (AGENTS.md §3-20)** + **NOT 최종 quality** (§0 일관)

### 3-1. NetGain 의 정의 (calibrated_protocol_a_v1)

```
NetGain = ArtifactReduction - α·FidelityLoss - β·CorrectionMag - γ·ToolCost
α = 5.0, β = 0.0, γ = 0.0 (synthetic Protocol A grid search, 부록 D)
```

### 3-2. NetGain 의 위치 (AGENTS.md §3-20 박제)

- **Category C — Internal Routing Reward** (외부 공개 최종 성능 근거 금지).
- **RL-1 / RL-2 policy 가 argmax 하는 objective** (policy optimization reward).
- **NOT 최종 motion quality metric**. NetGain median 비교 만으로 "ArtifactRouter 가 우월" 단정 금지.
- α=5.0 의 threshold sensitivity 는 robust (부록 AA, α=1-20 range).

### 3-3. NetGain 의 외부 공개 인용 시 의무

- **인용 표기**: "NetGain is a proxy reward, not a standard motion quality metric. Final quality is validated by standard metrics (FID, R-Precision, MM-Dist) + visual/perceptual rating + Category B variants (foot skating, jerk)."
- **단독 성능 근거 인용 금지** — Category A / B / quality-validated evidence 동반 의무.

### 3-4. NetGain 의 originality (부록 BB)

- **본 프로젝트 자체 정의** — direct standard reference 없음.
- Related framework (spirit, not direct): Holden et al. 2020 (Learned Motion Matching), Ng et al. 1999 (reward shaping).
- 본 프로젝트의 unique contribution.

### 3-5. RL-2 진입 전 prerequisite (사용자 directive 2026-05-25, revised 2026-05-26)

- **NetGain = reward only** (Category C internal routing reward).
- **RL-2 objective = constrained optimization** (§0-1):
  - `maximize artifact_improvement` (NetGain or revised proxy).
  - `subject to physical_validity_gate + no_harm_gate` (hard constraints, NOT reward weight).
- **최종 성능 (RL-2 vs baseline) 의 evidence**: **Category A + visual/perceptual rating + Category B variants + physical validity gate output**.
- 본 prerequisite 충족 전 RL-2 결과 의 외부 공개 인용 금지.

### 3-6. Physical Constraint Gate — **별도 평가, NetGain weight 에 넣지 말 것** (사용자 directive 2026-05-26 신설, **PhysDiff ICCV 2023 + MDM ICLR 2023 motivation**)

본 §0 의 framing 의 의무 mechanism. **NetGain 의 추가 negative weight 로 처리하면 reward hacking 잔존** — 따라서 hard gate 로 분리.

**근거 논문** (AGENTS.md §3-22):
- **PhysDiff** (Yuan et al. 2023, ICCV) — physical artifact (foot sliding / ground penetration / floating) 를 physics-guided projection 으로 다루는 SOTA 의 framework. **scalar reward 안 넣고 별도 constraint** 의 핵심 motivation.
- **MDM** (Tevet et al. 2023, ICLR) — motion generation 에서 geometric loss / velocity / foot contact 를 **별도 축** 으로 처리하는 design. fidelity 와 physical plausibility 의 dimensional separation.

#### 3-6-1. Gate 의 최소 구성

| Evaluator (proposed) | 측정 | Category | 정량 근거 (2020+ peer-reviewed) |
|---|---|---|---|
| **BoneLengthViolation** | per-bone length 의 frame-to-frame 상대 변화 (clean reference 대비) | B (variant) | ACTOR (Petrovich et al. 2021, ICCV) skeleton constraint spirit. 본 프로젝트 의 strict variant. |
| **GroundPenetration** | foot Y < ground threshold 의 ratio | B (variant) | **PhysDiff (Yuan et al. 2023, ICCV)** 의 ground penetration metric. 신규 또는 기존 FootFloating 의 strict 버전. |
| **ContactConsistency** | contact label 의 frame-to-frame 일관성 (foot velocity ≈ 0 when in contact) | B (variant) | **PhysDiff (Yuan 2023)** + **HumanML3D (Guo et al. 2022, CVPR)** 의 contact 추정 spirit. Item 6 의 contact estimator 결합. |
| **JerkSpike** | acceleration 의 95-th percentile (per-joint normalized) | B (variant) | **MDM (Tevet et al. 2023, ICLR)** 의 acceleration / velocity smoothness loss spirit. 기존 VelocityJitter 의 spike-only 버전. |

#### 3-6-2. Gate 의 decision

| Decision | 조건 | 정책 |
|---|---|---|
| **accept** | 모든 gate evaluator < threshold | tool output 채택 |
| **repair** | 일부 gate violation, strength 감소 가능 | strength `large5` → `medium5` 또는 같은 tool 다른 strength 재시도 |
| **rollback** | gate violation 심함 + 직전 step 의 score 가 더 나음 | 직전 step 으로 복귀 |
| **STOP** | tool 적용 reset 후에도 gate violation 또는 budget 소진 | 종료 |

#### 3-6-3. NetGain reward weight 가 아닌 hard gate 의 이유

- (a) Weight 로 넣으면 — reward maximizer 가 weight 와 reward 의 trade-off 학습, weight 조정 시 결과 robust 하지 않음.
- (b) Hard gate 로 분리 — policy 의 action space 가 `{accept, repair, rollback, STOP}` 또는 그 subset 으로 명확.
- (c) 외부 공개 시 "physical safety 가 정량 보장됨" 의 직접 evidence.

#### 3-6-3-1. Gate threshold 는 **regression-based (before+eps) 의무, absolute clean_p99 금지** (Step E-2.6, 2026-05-27 신설)

Step E-2.6 (BoneLengthCV threshold sensitivity, n=300) 의 정량 발견:

- **Absolute clean_p99 thresholding (after > clean_p99) = 46% violation rate, 109 false positive** — 자연적으로 bone CV 높은 G2 motion (generation artifact) 이 correction 으로 worsen 안 됐는데도 flag.
- **Regression-based (after > before + eps) = 9.7%, real violations only** — correction 이 실제로 metric 증가시킨 경우만 flag.

**의무**: gate decision 의 unsafe threshold 는 `max(before + eps, clean_p99)` (regression-dominant) 사용. **pure absolute clean_p99 단독 사용 금지** (FP-heavy). 본 logic 은 ``tools/safe_sequence_oracle_run.py`](../../../tools/safe_sequence_oracle_run.py) 의 `_gate_violation` 에 구현됨.

**Robustness** (Step E-2.6): regression-based threshold (before+eps → relative 10%) 의 violation rate 는 4.3~9.7% stable band + FootLock dominance 86~100% 유지 — threshold 완화 에 robust. 상세: ``reports/2026-05-27.md §7`](../../../reports/2026-05-27.md).

#### 3-6-4. Gate 의 비교 의무 (사용자 Step D)

같은 sample 에 대해:
- **A. NetGain-only oracle / policy** (현재).
- **B. NetGain + physical gate oracle / policy** (신규).

비교 측정:
- ArtifactReduction 의 감소량.
- BoneLengthViolation / GroundPenetration / ContactConsistency / JerkSpike 의 감소.
- over-modification 의 정량 (correction magnitude 의 분포 shift).
- STOP rate 의 증가.

→ Physical gate 가 distortion 을 줄이면 본 framing 의 직접 evidence.

## 4. Synthetic Corruption 의 위치 — **Controlled Diagnostic Only**

### 4-1. Synthetic 의 정의 + 적절한 용도

- HumanML3D clean + `inject_foot_floating(0.08)` + `inject_jitter(0.05)` (Step 3 multi-artifact recipe).
- **적절한 용도** (AGENTS.md §3-17):
  - tool 작동 검증.
  - evaluator artifact 잡기 검증.
  - oracle headroom 측정.
  - policy 의 known artifact 학습 능력 검증.
  - ablation / regression test.

### 4-2. Synthetic 의 limit — **최종 성능 sole evidence 금지**

본 프로젝트 의 이미 관찰된 evidence (synthetic 일반화 위험 4 정량 evidence):

1. **B2 NetGain**: synthetic mean +0.187 vs G2 mean -0.017 (정반대 sign, 부록 V).
2. **Best strength**: synthetic large 76% vs G2 small 100% (oracle, 부록 V).
3. **Synthetic-trained B7 → G2 zero-shot fail** (부록 I, Step 5 transfer diagnostic).
4. **RL-1 imitation**: synthetic sub-B2 (-15.6% closure, 부록 S) vs G2 92% closure (정반대 mechanism, 부록 X).

### 4-3. Synthetic vs G2 magnitude (부록 Z)

- BoneLength: synthetic median 0.291 vs G2 median 0.032 — **9.21x**.
- VelocityJitter: synthetic median 0.199 vs G2 median 0.000 — **~40x (mean ratio)**.
- → "Synthetic 우수 ≠ G2 우수" 의 robust 정량 증거.

## 5. Real-Distribution Evidence — **G2/G1 Natural + Perceptual**

### 5-1. 현재 active scope

- **G2 (MotionGPT)**: external_assets/g2_generated_v1/ (50 motion, t2m.txt 50 prompt).
- **G1 (MDM/MLD)**: **미도입** — Week 3+ MVP plan. real-distribution evidence 확장 의 다음 의무.

### 5-2. 외부 공개 evidence 5 의무 (AGENTS.md §3-17)

1. **G2 natural** — paired test, multi-seed, n ≥ 30 (부록 L 의 일부 충족).
2. **G1 natural** — 미수행 (Week 3+).
3. **Multi-prompt category coverage** — t2m.txt 50 prompts 의 category 분포 분석 미수행. **사용자 Step 2 의 일반 동작 10 prompt** 가 첫 정식 시도.
4. **시각화 sanity check** — 부록 W (n=2 sample, preliminary).
5. **Perceptual rating / motion quality metric** — **미수행** (사용자 Step 5 의 정식 진입).

### 5-3. 본 프로젝트 의 G2 측면 결과 (real-distribution evidence)

- **HGB G2 abstain learning 거의 perfect** (부록 X): STOP recall 100% (held-out + closed-loop), B2-family-best 도 strict surpass (15/15, p=3e-05, d=+0.47), oracle G2 98% gap closure.
- [H-2026-203](../../../evals/hypotheses/H-2026-203.md) (No-harm) 의 정식 real-distribution evidence supports.

## 6. Baseline Family Protocol — **§3-18**

### 6-1. B2 의 정정 (사용자 directive 2026-05-25)

- B2 는 **"대표 baseline" 아닌 "fixed smoothing diagnostic baseline family"**.
- B2-family: B2-small / B2-medium / B2-large / B2-val-best.
- 성공 기준: **"B2-medium 초과" 아닌 "fixed smoothing family 대비 우월"**.

### 6-2. B2-family sweep 결과 (부록 V)

| Distribution | B2-family ceiling | 함의 |
|---|---|---|
| Synthetic | B2-medium ≈ B2-val-best (medium 90% sample best) | 기존 B2-medium 비교 robust |
| G2 | **B2-val-best = B2-small (100% sample best)** | B2-medium 의 임의성 advantage 발견 |

### 6-3. RL-2 평가의 baseline family 의무

RL-2 평가 시 비교 baseline:
- **B2-family-best** (per-sample best strength).
- **B5** (rule-based).
- **RL-1 best** (synthetic: MLP, G2: HGB).
- **Sequence oracle** (closed-loop ceiling).

## 7. 현재 Evidence Status (2026-05-25 시점)

### 7-1. 완료된 evidence

| Section | Evidence |
|---|---|
| Step RL-0 (synthetic) | Sequence oracle > B2 multi (부록 M, d=+1.71, +0.124 mean gap) |
| Step RL-0 보강 (G2) | Sequence oracle > B2 G2 (부록 N, d=+0.726, STOP-best 54%) |
| Step RL-1 (imitation policy) | G2 abstain success (부록 O, X), synthetic partial (부록 R, S) |
| Step 5-B v2 (small-calibration) | n_eval=30 × 3 seed, all p<0.001, d=+0.317 (부록 L) |
| RL-1 진단 | 4 origin verdict (부록 R): RF dominant + data partial + imitation paradigm (부록 T) |
| B2-family sweep | §3-18 정식 (부록 V) |
| Visual sanity | 6 GIF (부록 W, preliminary, n=2 sample) |
| 3 분포 분리 | bone 9.21x / jitter 40x (부록 Z) — framing supports |
| α sensitivity | default α=5.0 robust 1-20 (부록 AA) |
| Metric references | reproducibility-checklist §3 확장 (부록 BB) |

### 7-2. 미수행 evidence (외부 공개 의무)

| Evidence | 우선순위 | 작업 |
|---|---|---|
| **NetGain validity 검증** | **다음 (사용자 Steps 3-7)** | G2 general-prompt pilot + perceptual rating |
| G1 natural transfer | Week 3+ | G1 wrapper |
| Multi-prompt category | Step 3 (사용자 plan) | 일반 동작 10 prompt |
| Perceptual rating | Step 5 (사용자 plan) | 20-30 pairwise A/B |
| Contact estimator | Item 6 (큰 작업) | evaluator 정밀화 |
| FootFloating limitation 해소 | Item 6 결합 | contact estimator 도입 |

## 8. 핵심 분기점 (사용자 Step 6, revised 2026-05-26 — Safe Orchestration framing)

본 프로젝트 의 다음 분기 — 사용자 directive 2026-05-26 의 6-step priority 로 정정.

### 8-1. **새 우선순위 — 사용자 directive 2026-05-26**

| Step | 작업 | 의미 |
|---|---|---|
| **A** | current_research_position.md 정정 (본 §0 박제) | "safe orchestration" framing 의 단일 출처 |
| **B** | **G2 top correction (motion_006/007/008/028) side-by-side 시각화** | NetGain 높은 G2 보정 의 distortion 검증 — **최우선** |
| **C** | Physical constraint evaluator/gate 설계 (§3-6) | 본 framing 의 mechanism |
| **D** | NetGain-only vs NetGain+physical-gated oracle 비교 | gate 의 effect 정량 evidence |
| **E** | Standard metric integration (FID/R-Prec/MM-Dist/Diversity) | 외부 공개 prerequisite |
| **F** | RL-2 constrained policy (Step C-E 후) | 본 framing 의 정식 RL implementation |

### 8-2. Step B 의 정식 motivation + 3-way classification (사용자 directive 2026-05-26)

> "G2 top correction sample 4개를 side-by-side로 보고, NetGain이 높은 보정이 실제로 왜곡을 만드는지 확인한다."

#### 8-2-1. Diagnostic caveat (representative 아님)

본 4 sample 은 **NetGain top 4 의 의도적 enriched subset** (motion_006/007/008/028, NetGain 범위 +0.092 ~ +0.173). G2 distribution 의 representative 가 아니라 **NetGain-high case 의 visual distortion sanity check**.

#### 8-2-2. Visual inspection checklist (사용자 directive)

panel 1 (Original) vs panel 3 (5-level oracle) 의 차이 의 5 항목 점검:

1. 다리/팔 길이가 늘어나 보이는가?
2. 발이 바닥에 말이 되게 붙는가?
3. ground penetration 이 보이는가?
4. 움직임 리듬이 죽었는가?
5. prompt 의미와 상체/하체 동작이 유지되는가?

#### 8-2-3. 3-way classification + 분기

| 결과 | 정의 | 분기 |
|---|---|---|
| **good** | panel 3 이 original 보다 명확히 나음 | Step E (standard metric) + Step F (RL-2) 우선. Step C 의 priority 낮춤. |
| **distorted** | artifact 줄었지만 physical/posture 왜곡 (다리 길이 / 발 위치 / 리듬 / 상체 의미) | **Step C (Physical Constraint Gate) 의 high priority**. RL-2 reward 재설계 의 직접 motivation. |
| **ambiguous** | 차이 작음 또는 판단 어려움 | Step C 의 evaluator 별 threshold ablation 우선. |

**중요 결정 rule** (사용자 directive):
- **distorted 1개라도 있으면 Step C 우선**.
- **전부 good 이어도 Step E 의 의무 보존** (diagnostic subset 의 generalization 불가).

#### 8-2-4. motion_007 의 special case

motion_007 의 oracle sequence = `VelocitySmoothing-medium5 → FootLock-xlarge` — 두 강한 strength 의 연속, NetGain 높지만 **distortion 위험 후보**. 본 sample 의 panel 3 vs panel 1 비교 가 framework 의 핵심 검증 case.

### 8-3. 이전 Branch A/B 의 위치 정정 (2026-05-26)

이전 (2026-05-25) 의 Branch A/B (NetGain validity 통과/부족) 는 **본 §0 의 framing 하에서 의 sub-decision** 으로 격하:
- **이전 Branch A (NetGain validity 통과)** = "NetGain proxy 가 perceptual 과 reasonably 일치" — 본 framing 의 Step F (RL-2) 진입 조건 의 일부.
- **이전 Branch B (NetGain validity 부족)** = "NetGain proxy 가 perceptual 과 미일치" — Step C (physical gate) 의 motivation 의 일부.

본 분기 는 **Step C-D 완료 후 의 RL-2 의 reward design 의 분기**.

## 9. 분포별 RL-1 의 역할 정식 박제 (부록 X)

| Distribution | RL-1 의 역할 | 핵심 mechanism |
|---|---|---|
| **G2 low-severity natural** | **abstain policy (STOP) — 거의 perfect** | HGB STOP recall 100%, oracle 98% closure (부록 X) |
| **Synthetic high-severity** | **부족** — RL-2 motivation | rollback 33-44% (paradigm 한계), oracle gap -16% ~ -6% (부록 R, S, T) |

## 10. 외부 공개 evidence keyword 의무 (AGENTS.md §3-17)

5단계 리포트의 결론 절에 다음 keyword 명시:
- **"controlled diagnostic finding"** — synthetic-only evidence.
- **"real-distribution evidence"** — G1 / G2 natural.
- **"quality-validated evidence (preliminary)"** — visual sanity (n=수개).
- **"quality-validated evidence"** — perceptual rating (n≥20, 정식).

본 keyword 누락 결과는 silent invalidation (§6-5 동질) 으로 취급.

## 11. 본 문서의 유지 의무

- 외부 공개 (논문·발표·README) 결과 인용 시 본 문서를 reference.
- 본 문서 의 어느 항목 변경 시 → AGENTS.md / reproducibility-checklist / 5단계 리포트 의 일관성 동시 갱신.
- 본 문서 자체 의 갱신은 사용자 framing 정정 또는 큰 evidence 추가 시.
