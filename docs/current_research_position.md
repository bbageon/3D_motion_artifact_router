# Current Research Position — ArtifactRouter (2026-05-25)

> 본 문서는 사용자 directive (2026-05-25, "NetGain validity 검증 plan") 의 Step 1: 현재 연구 의 위치 를 단일 문서로 고정. 외부 공개 (논문·발표·README) 의 evidence 인용 시 본 문서를 reference.
>
> AGENTS.md §3-17 (evidence 계층) + §3-18 (baseline family protocol) + 부록 P/U/Y/Z/AA/BB 의 통합 reference.

---

## 1. 본 연구 의 정체성 (한 줄)

ArtifactRouter 는 **canonicalized motion artifact state 위에서 cost · risk 를 고려해 correction intervention 또는 STOP 을 선택하는 generator-agnostic, tool-extensible decision system** 이다. 새 motion generator 도, 새 단일 correction algorithm 도 개발하지 않는다.

## 2. Evaluator 의 위치 — **Diagnostic Proxy Metric**

### 2-1. 현재 evaluator 의 정의

- **FootFloatingEvaluator**: simple Y-threshold (tau_float=0.05).
- **BoneLengthEvaluator**: per-bone normalized variation.
- **VelocityJitterEvaluator**: per-joint mean acceleration norm.

(상세 — [reproducibility-checklist SKILL §3](../.claude/skills/reproducibility-checklist/SKILL.md))

### 2-2. 현재 evaluator 의 위치

- **Diagnostic proxy metric — 최종 motion quality 의 ground truth 아님**.
- 본 프로젝트 의 routing policy / RL agent 의 reward signal 로 사용 가능 (proxy).
- 단 외부 공개 (논문·발표) 에서 본 evaluator 의 결과를 **"motion quality" 로 인용 금지**.
- 본 evaluator score 의 변화가 시각/사람 평가 와 일치하는지는 별도 검증 (Step 5 perceptual pilot).

### 2-3. 발견된 limitation (2026-05-25)

- **FootFloatingEvaluator 의 corruption robustness 부족** (부록 Z): synthetic `inject_foot_floating(0.08)` 이 max score 를 거의 안 올림 (synthetic median 0.005 < clean median 0.021). 글로벌 Y shift 가 evaluator metric 과 mismatch. **Item 6 (contact estimator) 의 정량 motivation**.
- 본 limitation 후 외부 공개 시 FootFloating 결과 의 caveat 동반 의무.

## 3. NetGain 의 위치 — **Proxy Reward for RL/Routing Policy**

### 3-1. NetGain 의 정의 (calibrated_protocol_a_v1)

```
NetGain = ArtifactReduction - α·FidelityLoss - β·CorrectionMag - γ·ToolCost
α = 5.0, β = 0.0, γ = 0.0 (synthetic Protocol A grid search, 부록 D)
```

### 3-2. NetGain 의 위치

- **Proxy reward** — RL/routing policy 가 argmax 하는 objective.
- **최종 motion quality 의 정의 아님**. NetGain median 비교 만으로 "ArtifactRouter 가 우월" 단정 금지.
- α=5.0 의 threshold sensitivity 는 robust (부록 AA, α=1-20 range).
- **NetGain proxy 의 perceptual validity 는 별도 검증 의무** (Step 5 perceptual pilot).

### 3-3. NetGain 의 originality (부록 BB)

- **본 프로젝트 자체 정의** — direct standard reference 없음.
- Related framework (spirit, not direct): Holden et al. 2020 (Learned Motion Matching), Ng et al. 1999 (reward shaping).
- 본 프로젝트의 unique contribution.

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

## 8. 핵심 분기점 (사용자 Step 6)

본 프로젝트 의 다음 분기:

**Branch A (NetGain validity 통과)**:
- NetGain rank ≈ visual/perceptual rank (Step 5 pilot 통과).
- → **RL-2 (trajectory-level value learning) 진입**.
- RL-2 의 reward = NetGain (proxy). 결과 = visual/perceptual 로 검증.

**Branch B (NetGain validity 부족)**:
- NetGain 과 사람 판단 자주 어긋남.
- → **Reward 수정 우선**:
  - FootFloating weight 낮추기 (부록 Z limitation 결합).
  - jerk / foot sliding 추가.
  - fidelity penalty 재조정.
  - contact estimator 개선 (Item 6).
  - correction magnitude penalty 복원.
- → **RL-2 진입 보류**.

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
