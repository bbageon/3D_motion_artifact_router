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

(상세 — [`docs/metric_provenance.md`](metric_provenance.md), [reproducibility-checklist SKILL §3](../.claude/skills/reproducibility-checklist/SKILL.md))

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

### 3-6. Physical Constraint Gate — **별도 평가, NetGain weight 에 넣지 말 것** (사용자 directive 2026-05-26 신설)

본 §0 의 framing 의 의무 mechanism. **NetGain 의 추가 negative weight 로 처리하면 reward hacking 잔존** — 따라서 hard gate 로 분리.

#### 3-6-1. Gate 의 최소 구성

| Evaluator (proposed) | 측정 | Category | 비고 |
|---|---|---|---|
| **BoneLengthViolation** | per-bone length 의 stride 별 상대 변화 (clean 대비) | B (ACTOR spirit) | 기존 BoneLengthEvaluator 의 strict 버전 |
| **GroundPenetration** | foot Y < ground threshold 의 ratio | B (PP-Motion partial) | 신규 또는 FootFloating 의 strict 버전 |
| **ContactConsistency** | contact label 의 frame-to-frame 일관성 | B (HumanML3D contact spirit) | Item 6 의 contact estimator 결합 |
| **JerkSpike** | acceleration 의 95-th percentile | B (Flash & Hogan 1985) | 기존 VelocityJitter 의 spike 버전 |

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
- [H-2026-203](../evals/hypotheses/H-2026-203.md) (No-harm) 의 정식 real-distribution evidence supports.

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

### 8-2. Step B 의 정식 motivation

> "G2 top correction sample 4개를 side-by-side로 보고, NetGain이 높은 보정이 실제로 왜곡을 만드는지 확인한다."

본 검증 의 결과가 분기:
- **B 의 결과 = NetGain 높은 보정이 visually 좋음** → physical gate 의 priority 낮춤, RL-2 의 NetGain proxy reward 가 OK.
- **B 의 결과 = NetGain 높은 보정이 왜곡** → physical gate 의 필수 mechanism, Step C-D 의 high priority.

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
