# Finding — MotionGPT(G2)에 대한 본 연구의 시사점

> 대상: MotionGPT (Jiang et al., NeurIPS 2023) 출력(G2) 에 ArtifactRouter 를 적용해 얻은 발견과 그 연구적 함의.
> 데이터: [../dataset/motiongpt_g2_pool.md](../dataset/motiongpt_g2_pool.md). 근거 규약: AGENTS.md §3-17(evidence tier)·§3-20(metric category)·§3-22(2020+ 근거).
> 핵심 한 줄: **MotionGPT 는 분포적으로 이미 near-GT(VQ codebook) 이라 real refinement headroom 이 작고, tool 의 효용은 artifact 프로파일(=generator)에 의존한다. 따라서 MotionGPT 에서 framework 의 올바른 행동은 대부분 STOP(no-harm) 이며, 품질 gain 은 artifact-rich generator 에서 기대된다.**

---

## 0. 평가 단위 — 대표(representative) 결과 + no-harm (헤드라인)

> **방법론(§3-22)**: stress 는 artifact proxy(=router 의 단서)로 고른 enriched subset → 단독 보고는 selection bias(낙관적). **대표 성능·no-harm 주장은 generator 의 실제 출력 분포(전체 비율)로** 한다 — HumanML3D 평가 프로토콜(Guo CVPR2022)·MDM(ICLR2023)·MoMask(CVPR2024) 모두 full test set 보고. stress/natural 은 **사전 등록된(split frozen) 부차 진단**.

**Band 실제 비율** (full G2 pool n=600): stress **124(20.7%)** / normal 353 / clean_like 123 → low-artifact **476(79.3%)**. ⚠️ stress 만 보거나 holdout 65:99 로 concat(=stress **39.6%**) 하면 **흠집 소수를 ~2배 과대표집**.

### 대표 헤드라인 (prevalence-weighted, [snapshot](../../evals/snapshots/representative_aggregate_g2_v1.json))

| 지표 (category) | M0(safe) | heuristic | oracle | 대표 판정 |
|---|---|---|---|---|
| **FID** (A, standard) | +0.004 | +0.002 | +0.006 | **NEUTRAL** (개선도 손상도 없음) |
| **R@1** (A) | -0.0015 | +0.001 | +0.001 | NEUTRAL |
| accel (B, smoothness) | -0.3% | -2.7% | -0.6% | 미미 개선 |
| float (B) | -9.1% | -10.2% | -10.2% | 개선 |
| **foot_skate** (B, 표준) | +9.9% | +20.4% | +20.4% | **악화 (trade-off)** |
| artifact_total (C, **proxy**) | -11.5% | -24.6% | -29.2% | 감소 (단 내부 proxy·stress 집중) |
| **no-harm** (clean GT Δ) | **0.00000** | **0.00000** | **0.00000** | **완전 보존** |

(physical/artifact 모두 bootstrap CI 가 0 제외 = 유의. FID 는 set-level → per-band Δ 의 prevalence-weighted 근사.)

### 대표 결론 (정직)

1. **standard quality(Category A)는 대표 분포에서 NEUTRAL** — refinement 가 FID/R-Prec 를 **개선하지도, 손상하지도 않는다**. F1(near-GT, headroom 없음)과 정합.
2. **no-harm 확정** — clean GT Δ = 정확히 0, low-artifact band(79.3%)도 거의 0. framework 가 깨끗한 모션을 건드리지 않음.
3. **artifact proxy(Category C) 감소는 stress 소수(20.7%)에 집중** — 대표 분포로 보면 stress-only view(-18~-40%) 보다 **희석**(-11~-29%)되고, **내부 proxy 라 최종 quality 근거 아님**(§3-20).
4. **physical(Category B)은 trade-off·미미** — smoothness/float 소폭↓ vs 표준 foot_skate↑.

→ **MotionGPT 에서의 정직한 contribution = "safe no-harm refinement (standard quality NEUTRAL 보존) + 내부 artifact proxy 의 minority(stress) 감소"**. **"real quality 향상" 은 주장 불가.** stress-only 수치(AR-029/AR-016)는 mechanism **진단**으로만 인용.

---

## F1. MotionGPT 는 분포적으로 이미 near-GT 다 (real FID headroom 작음)

**결론**: g2_natural 의 FID 0.846 ≈ clean GT 0.717. MotionGPT 출력은 자연 분포에 매우 가깝다. VQ codebook(clean motion 으로 학습한 token 사전)으로 양자화하는 구조라, 통계적으로 그럴듯한 token 으로 snap → 분포적 결함이 적다.

- **Evidence**: [standard_metric_closed_loop_v1](../../evals/snapshots/standard_metric_closed_loop_v1.json) (Category A). [survey §3](../generator_failure_mode_survey.md) (VQ vs diffusion paradigm).
- **함의**: tool 이 FID(분포 거리)를 개선할 여지가 구조적으로 작다. "tool 적용 → FID 개선 없음" 의 1차 원인.
- **불확실성**: holdout FID 는 small-n upward bias (§3-9). 단 natural≈clean 의 방향은 robust.

## F2. FID 는 MotionGPT 의 local 물리 artifact 에 둔감하다 (proxy-standard misalignment)

**결론**: g2_natural 이 FID 로 near-GT 여도, AR-043 proxy 로는 FootFloating 등 local artifact 가 실재(stress 80%, natural 10%). FID 는 분포 거리라 한 clip 의 발 미끄러짐/뜸 같은 per-frame 위반을 거의 못 잡는다.

- **Evidence**: [AR-043](../../evals/snapshots/dataset_issue_prevalence_audit_v1.json) (Category C proxy). [survey §1-1](../generator_failure_mode_survey.md).
- **함의**: "FID 중립 = 문제 없음" 이 아니다. FID 가 보는 층(분포/semantic)과 tool 이 고치는 층(local 물리)이 **다르다**. 효과 판정에 FID 단독 사용 금지.
- **불확실성**: artifact proxy = Category C. local artifact 가 perceptual 로 유의한지는 미검증(b2/b3, AR-023 필요).

## F3. tool 의 효용은 artifact-profile(=generator)에 의존한다 — MotionGPT 엔 foot headroom 이 작다

**결론**: 표준 physical metric(Category B)으로 재측정(AR-029)해도 MotionGPT 의 foot-artifact headroom 은 작다. tool 은 자기가 노리는 artifact 가 있어야 쓸모 있는데, MotionGPT(VQ)는 그 artifact 를 적게 만든다.

real g2_stress n=65, original 대비 paired Δ ([physical_metric_g2_stress_v1](../../evals/snapshots/physical_metric_g2_stress_v1.json)):

| 축 (낮을수록 좋음) | M0(safe) | heuristic | oracle | 방향 |
|---|---|---|---|---|
| **acceleration** (smoothness, ground-INDEP) | d=-0.16 (42%) | d=-0.24 (63%) | d=-0.45 (63%) | **소폭 개선 (유의)** |
| float (planted foot height) | d=-0.49 (26%) | d=-0.67 (43%) | d=-0.70 (55%) | 개선 (적용 subset) |
| **foot_skate** (GMD/EDGE 표준) | d=+0.40 (2%) | d=+0.43 (2%) | d=+0.66 (3%) | **악화 (trade-off)** |

- **해석**: ① 떠 있는 발을 내리면(float↓) 그 발의 잔여 수평이동이 표준 foot_skate 로 등록 → **skate↑** (foot artifact 간 trade-off). ② MotionGPT foot_skate original=0.0039 로 **이미 극히 낮아 headroom 없음**. ③ ground-independent 한 acceleration 만 깨끗하게 개선되나 **효과 작음**.
- **Evidence**: AR-029 (Category B) + F1 closed_loop FID.
- **§6-10 caveat**: 본 결론은 **MotionGPT 단독** 실측. "tool 효용이 generator-의존" 의 대조 입증은 ≥2 generator(diffusion 실측) 필요 — 현재는 1 generator + 문헌(PhysDiff) 기대.

## F4. 공격적 적용은 FID 까지 악화시킨다 — 안전한 free win 이 아니다

**결론**: heuristic/oracle 처럼 tool 을 많이 적용하면 artifact proxy 는 줄지만 FID 가 악화된다(g2_stress: heuristic ΔFID +0.040, oracle +0.019). 즉 physical 일부 개선 ↔ standard quality 손상이 동시에 일어난다.

- **Evidence**: standard_metric_closed_loop_v1 (Category A).
- **함의**: MotionGPT 에서 "무지성 일괄 보정(fixed post-processing)" 은 손해. → 조건부 선택의 필요성(F6).

## F5. MotionGPT 에서 framework 의 올바른 행동은 대부분 STOP(no-harm)이다

**결론**: 고칠 artifact 가 적은 분포에서는 "아무것도 안 하기(STOP)" 가 최적이다. 안전 정책 M0 는 실제로 대부분 STOP 하며, clean_noharm 에서는 모든 metric Δ=정확히 0 (no-harm 통과).

- **Evidence**: closed_loop clean_noharm 全 Δ=0.000; M0 의 stress Δartifact 작음(-0.014, 대부분 STOP).
- **함의**: framework 가 "망치지 않음"(no-harm)을 지키는 정상 동작. gain 부재 ≠ framework 결함. gain 은 artifact-rich regime 에서 발생.
- **불확실성**: STOP 비율의 정밀 분해(selection_mode breakdown)는 별도 보고 권장(§3-25).

## F6. framework 시사점 — 이 generator-의존성이 곧 조건부 라우팅의 존재 이유

**결론**: 만약 모든 tool 이 모든 generator/동작에 똑같이 들었다면 일괄 처리로 충분하다. **generator·동작마다 artifact 가 다르기 때문에** "이 artifact 엔 이 tool, 없으면 STOP" 의 조건부 선택(artifact-conditioned routing)이 가치를 갖는다.

- **함의**: 본 발견은 [H-2026-204](../../evals/hypotheses/H-2026-204.md)(artifact-conditioned + closed-loop > fixed post-proc)·[H-2026-206](../../evals/hypotheses/H-2026-206.md)(generator-agnostic)를 **뒷받침**한다. "generator 마다 적합 tool 이 다르다" = "조건부 라우팅 > 일괄 처리" 의 근거.
- **단**: H status 전환은 사용자 승인 게이트(§3-11). 본 문서는 supports 신호이지 status 변경이 아니다.

## F7. 우리가 정의한(문헌 인용) 문제들이 MotionGPT 에 절대 기준으로 거의 없다

**결론**: artifact/physical 8개 evaluator 의 **절대 발생률**(AR-043, SEV_LOW / clean-p99 기준)로 보면, MotionGPT 출력에 정의된 문제가 거의 안 나타난다. 우리 evaluator 리스트는 주로 **diffusion 기반 문헌(PhysDiff 등)**에서 인용한 것이라 token/VQ 와 맞지 않는다.

| 정의 문제 (인용) | full | natural | stress | clean GT | 판정 |
|---|---|---|---|---|---|
| FootFloating (PhysDiff/MDM) | 23% | 10% | 80% | 26% | ✅ 유일 reliable·discriminating |
| BoneLength / CV (HuMoR/ACTOR) | 76~100% | 84~100% | 91~100% | 0~2% | 🔶 G2≫GT(체계적)이나 saturated·모호 |
| **Skate** (PhysDiff/GMD ★헤드라인) | **0%** | **0%** | 0% | 0% | ❌ 부재 |
| **Penetrate** (PhysDiff) | **0%** | **0%** | 0% | 0% | ❌ 부재(+min-Y degenerate) |
| Float·JerkSpike·VelocityJitter | 0~3% | 0~4% | 0~5% | 0~8% | ❌ 거의 부재 |

- **Evidence**: [AR-043](../../evals/snapshots/dataset_issue_prevalence_audit_v1.json) (reliable=FootFloating 1 / moderate 2 / weak 5). AR-029 의 독립 표준 foot_skate=0.0039 도 skate 부재 확증.
- **함의**: tool 이 고칠 문제 자체가 거의 없음 = headroom 없음의 **근본 원인**. "tool 이 안 들어서"가 아니라 "문제가 없어서".
- **불확실성**: 0% 일부는 evaluator 둔감(Penetrate=min-Y ground degenerate) 가능 — 단 skate 는 독립 metric 으로도 ≈0 이라 진짜 부재. **AR-043 GO 결정에 본 caveat 필수.**

## F8. Category A 위반은 *의미(semantic)* 축에 집중 — refinement 범위 밖 → MotionGPT 단독 필요성 약함

**결론**: 권위 지표(Category A)로 GT 대비 gap 을 보면, MotionGPT 의 위반은 **거의 전부 text-motion 의미 정렬 축**(R-Precision/MM-Dist)이며, 이는 국소 기하 보정(refinement)이 **고칠 수 없는** 생성(generation) 문제다.

| 지표 | clean GT | g2_natural | gap | 축 | refinement 가 고치나 |
|---|---|---|---|---|---|
| FID ↓ | 0.717 | 0.846 | +0.13 (작음) | 물리/분포 | ❌ (ΔFID≈0) |
| **R@1** ↑ | 0.563 | 0.424 | **−0.14 (큼)** | **의미** | ❌ (ΔR@1≈0) |
| **MM-Dist** ↓ | 2.775 | 4.133 | **+1.36 (큼)** | **의미** | ❌ |

- **§3-22 판단**: MotionGPT 는 권위 지표를 **위반하긴 하나(의미 축)**, refinement 가 고치는 **물리 축엔 위반이 거의 없다**(F7). → **MotionGPT 단독으로는 본 refinement 연구의 필요성(necessity)이 약하다.** AGENTS.md §10 Stop/축소 신호(전제 약함·고칠 대상 부재)에 해당.
- **근거**: [standard_metric_closed_loop_v1](../../evals/snapshots/standard_metric_closed_loop_v1.json) (Category A) + [representative_aggregate_g2_v1](../../evals/snapshots/representative_aggregate_g2_v1.json) (ΔR@1≈0).
- **결정 권한**: **Go/Stop 자체는 사용자 승인 게이트**(AGENTS.md §10, §3-11). 본 문서는 evidence 박제이지 결정이 아니다.
- **불확실성**: R-Prec/MM-Dist gap 은 강한 generator 라도 GT 미달이 정상(MotionGPT R@1 0.424 는 보고 SOTA 근처). small-n.

## F9. MotionGPT 를 고칠 tool 은 *다른 class* (re-rank / edit / RL) — 우리 기하 보정 아님

**결론**: MotionGPT 의 큰 문제(의미)를 고칠 method 는 문헌에 존재하나, ArtifactRouter 의 "기하 국소 보정" class 가 아니라 세 다른 class 다.

| class | 방식 | post-hoc? | 대표 (★=peer-reviewed) |
|---|---|---|---|
| ① 선택 (re-ranking) | N 후보 생성 → text-motion 정합 최고 선택 | ✅ | best-of-N (TMR 점수 기반) |
| ② 편집 (editing) | 모션을 텍스트에 맞게 수정 | ✅ | ★Iterative Motion Editing (SIGGRAPH 2024) · PGR2M · IRG-MotionLLM (preprint) |
| ③ 생성측 RL/DPO | 보상/선호로 generator 정렬 | ❌ (재학습) | MotionRL · MoDiPO · ReAlign (preprint) |
| (참고) 물리 post-hoc | 물리 artifact 보정 (=우리 class) | ✅ | DMC (preprint) — 단 MotionGPT 엔 고칠 물리 거의 없음 |

- **§3-22**: ②의 SIGGRAPH 2024 만 peer-reviewed, 나머지 preprint → 단독 근거 금지(방향 근거).
- **함의 (방향 갈림길)**: **①re-ranking 이 ArtifactRouter 에 가장 자연스럽게 흡수 가능**(post-hoc·model-agnostic·R-Precision 직접 개선). ②/③ 은 framework 정체성을 "기하 보정"→"선택/편집/생성"으로 확장 → 새 가설·구조 변경(§3-11 게이트).

## F10. 메커니즘 — VQ codebook 이 실데이터로 snap → 물리 clean ↔ 의미 손해 (fidelity↔controllability trade-off)

**결론**: MotionGPT(VQ)는 **실데이터에서 학습한 token 사전(codebook)으로 양자화**하므로, 출력이 실모션 manifold 로 끌려간다. → **물리적으로 그럴듯(큰 물리 오류 적음)** 하지만, **요청이 codebook 에 정확히 없으면 가장 가까운 token 으로 대체 → 의미 정확도 손해**. 즉 "정확함"을 일부 포기하고 "그럴듯함"을 얻는 설계.

- **근거**: MotionGPT (Jiang NeurIPS 2023) · T2M-GPT (Zhang CVPR 2023) · MoMask (Guo CVPR 2024) = VQ codebook 구조. 대조: PhysDiff (Yuan ICCV 2023) = diffusion 은 codebook 없음 → 자유롭지만 물리 artifact 多.
- **정밀화 ①**: 완벽 clean 아님 — token 이음새 재구성 오차로 **bone 변동(G2 ≫ GT)** 잔존(F7).
- **정밀화 ②**: 균일하지 않음 — 어려운 요청 ~20%(stress, FID 8.9)는 manifold 에서 더 벗어남.
- **함의**: VQ↔diffusion 의 trade-off 가 "generator 마다 적합 tool 이 다른" 근본 이유(F3·F6). **우리 물리 보정 tool 은 diffusion 에 적합, MotionGPT 엔 부적합.**

---

## 결론적으로 주장 가능 / 불가능 (정직 경계)

| ✅ MotionGPT 에서 주장 가능 | ❌ MotionGPT 에서 주장 불가 |
|---|---|
| safe **no-harm** refinement (clean Δ=0, 망치지 않음) | "MotionGPT 의 real quality(FID/R-Prec) 를 향상시킨다" |
| 강한 **controlled diagnostic** (synthetic ΔFID -48.5, 복원 능력) | synthetic 결과를 real 성능 근거로 인용 (§3-17 위반) |
| 표준 metric 으로 **소폭 smoothness 개선** (Category B, AR-029) | "큰 품질 gain" — 효과 크기 작고 trade-off 동반 |
| **conditional routing 의 필요성** 정황(F6) | learned policy 단독 성능 우위 (snapshot≥2 + 사용자 승인 필요) |
| MotionGPT 는 **물리적으로 깨끗**(F7: skate/penetrate≈0) | "MotionGPT 는 흠 없는 generator" (의미 위반 큼, F8) |
| 위반은 **의미 축**에 집중 (F8: R@1 −0.14, MM-Dist +1.36) | 우리 **기하 보정 tool 로 그 의미 위반을 고친다** (F9: 다른 class 필요) |

## 한 줄 종합 (necessity)

> **MotionGPT 는 흠 없는 generator 가 아니다 — 가장 큰 문제는 "의미 정렬"(F8)이다. 단 그 문제는 우리 기하 보정 tool 의 범위 밖(F9)이고, 우리 tool 이 고치는 물리 축엔 위반이 거의 없다(F7, VQ 메커니즘 F10). 따라서 *MotionGPT 단독으로는 본 refinement 연구의 필요성이 약하다*. 필요성은 물리 위반이 실재하는 diffusion 에서 세워야 한다.** (Go/Stop 결정 = 사용자 게이트, §10)

## 다음 단계 (두 갈림길 — 결정은 사용자)

| 길 | 내용 | 비고 |
|---|---|---|
| **(A)** 정체성 유지 → diffusion | real refinement headroom 큰 generator(MDM/MoMask)에서 AR-029/AR-043 반복 → MotionGPT 와 paired 대조 → "tool 효용=generator-의존" 입증(§6-10) | 사용자가 generator 별도 확보 중(AR-022/045) |
| **(B)** action space 확장 → re-ranking | F9 의 ①re-ranking 을 새 action 으로 → MotionGPT 처럼 물리 clean·의미 약한 generator 에도 ArtifactRouter 가 쓸모 | framework 변경 = **새 가설·승인 게이트(§3-11)** |

## 남는 불확실성 종합 (§3-22)

- 모든 G2 결론은 **단일 generator** 실측 (diffusion 대조 미확보) — necessity 판단도 1 generator 근거.
- artifact proxy = Category C; perceptual(b2/b3) 미검증.
- foot_skate 는 ground=minY 추정 민감 — 절대 ground 기준 재확인 권장.
- holdout FID/R-Prec small-n (n=99/100); physical metric 효과 크기 작음.
- F9 의 re-ranking/edit/RL 효용은 **본 프로젝트 미측정** (문헌 기대, 다수 preprint).
