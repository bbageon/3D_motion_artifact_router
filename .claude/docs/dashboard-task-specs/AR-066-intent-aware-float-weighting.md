# AR-066 - Intent-Aware Float Weighting (Text-Conditioned Floating Judgment)

Status: backlog  
Epic: Evaluator  
Priority: 🟠  
Parent: AR-063 (A-6 잔여) / 사용자 제안 (2026-07-07): "float 에 대해서는 text 의 intent 에 맞게 가중치를 조절할 필요가 있어보이는데?"

## Goal

Floating 판정의 변별력 부족 (clean p99 0.70 — 자연 동작이 float band 에 걸림) 을 **"이 프롬프트에서 발이 떠 있는 것이 의도된 동작인가"** 로 조건화해 해소한다. 점프·기어가기·앉기 프롬프트의 발 들림은 정상, 정지·보행 프롬프트의 지속 부양은 artifact.

## Why This Exists

- AR-063 실측: Float gate-fire ≤1.1% (occur 61~68% 인데) — **정의가 자연 동작(heel roll·toe-off·점프)과 artifact 부양을 같은 밴드에서 구분 못 함**. 데이터 문제가 아니라 정의의 해상도 문제.
- MotionGPT(VQ) 의 유일한 reliable 물리 축이 FootFloating (AR-043) — float 정의가 날카로워지면 VQ 쪽 routing 근거가 정밀해진다.
- 비유: 작문 채점에서 "문장이 짧다"를 감점하려면 **시(詩)인지 논설문인지**를 먼저 알아야 한다 — 같은 현상도 의도에 따라 정상/결함이 갈린다.

## Component Design (직교 성분 — 순차 의무 없음; 2026-07-08 사용자 지적으로 개정)

> 개정 사유: (a) 지속시간 성분과 intent 성분은 **서로 다른 혼동 요인**을 고치는 직교 관계 —
> 지속시간 가중은 순간적 자연 들림(heel roll)만 제거하고 점프·앉기의 "지속된 의도적 부양"은
> 못 거름 → intent 를 뒤로 미룰 이유 없음. (b) text-motion 정렬은 기존 연구 표준 도구가
> 확립돼 있고 (아래 grounding) 본 프로젝트에 이미 인프라 존재 (R-Precision/MM-Dist 용
> Guo evaluator 텍스트 인코더) → keyword 휴리스틱을 1차로 둘 이유 없음. 동시 도입 가능하되
> **ablation 으로 성분별 기여 분리 보고** (§3-25 정신).

| 성분 | 방식 | 근거 도구 |
|---|---|---|
| **A. Intent 조건화 (주축)** | prompt 임베딩을 anchor 문장 집합 (airborne/acrobatic · seated/lying · locomotion · stationary) 과 비교하는 **zero-shot 분류** — 기존 text-motion contrastive 인코더 재사용 (추가 학습 없음). category 별 float 채점 억제 또는 별도 calibration | Guo CVPR2022 evaluator (보유) / TMR ICCV2023 (업그레이드 후보). keyword rule 은 **fallback** 으로 강등 (engineering heuristic) |
| **B. 지속시간 가중 (보조·직교)** | float segment 를 persistence 로 가중 (<3 frame 순간 들림 무시) — heel-roll·toe-off 노이즈 제거 | motion-only |
| **C. 학습 intent-aware contact (미래)** | text-conditioned frame-level contact-intent estimator (provenance "Item 6" + AR-059/VisRAG 계열) — A/B 는 sequence/segment-level 까지만 가능하고 **frame-level 국소화는 본 성분 전까지 불가** (한계 명시) | human 라벨 축적 후 |

## Validation Plan

- 성분 A/B 적용 후 clean 재calibration → **p99 가 판별 가능한 수준으로 내려오는지** + 생성물 fire-rate 가 generator 간 분리를 보이는지. **A/B 각각 단독 + 결합의 3-arm ablation** 으로 기여 분리.
- AR-058-3f human pack 의 `prompt_intent_conflict` / `dominant_artifact=floating` 라벨 (응답 수집 시) 을 external validation 으로 사용.
- **순환 금지**: bucket rule·가중치를 평가 결과(generator 간 차이가 나오도록)에 맞춰 tuning 하지 않는다 — clean + human 라벨로만 calibrate (HARKing 차단, §6-4 정신).

## Research Grounding (§3-22)

**Text-motion 부합 평가는 확립된 표준 축** (2026-07-08 보강 — "기존 연구 없나?" 질의 답):

- **HumanML3D** (Guo et al., CVPR 2022) — **R-Precision/MM-Dist**: text-to-motion 분야의 사실상 표준 text-부합 평가 protocol (contrastive text-motion 임베딩). 본 프로젝트 Category A 로 이미 사용 중 (MotionGPT 의미 위반 판정의 근거).
- **TMR** (Petrovich et al., ICCV 2023) — contrastive text-motion retrieval 임베딩 (Guo evaluator 보다 강한 공간) — intent 분류기 업그레이드 후보.
- **Voas et al.** (SIGGRAPH Asia 2023) — "What is the Best Automated Metric for Text to Motion Generation?": 자동 metric ↔ 인간 text-부합 판단 상관을 직접 연구 (MoBERT) — "정렬 점수를 판정에 쓸 수 있다"의 직접 근거.
- **MotionCLIP** (Tevet et al., ECCV 2022) / **TEMOS** (Petrovich et al., ECCV 2022) — 모션-텍스트 공동 임베딩 계열의 기반.
- **한계 (본 task 가 채우는 gap)**: 위 전부 **sequence-level** — "frame t 에서 발이 떠야 한다"는 frame-level 국소화는 제공하지 않음 → 성분 A 는 prompt/segment-level 조건화까지만 claim.

물리/contact 측:
- **HuMoR** (Rempe et al., ICCV 2021) — contact 를 관측이 아니라 **추정 대상**으로 다루는 근거.
- **PhysDiff** (Yuan et al., ICCV 2023) — float/slide/penetration 이 생성물의 실제 실패 유형.
- **PP-Motion** (ACM MM 2025) — 물리 지표는 지각 판단과 연결돼야 함.
- keyword fallback rule 은 **engineering heuristic** (논문 근거 아님 — 명시 의무). anchor 문장 집합의 구성도 heuristic 성분 포함 — human 라벨로 검증.

## Claim Boundary

허용: "intent 조건화가 float 판정의 변별력을 개선하는지 측정한다" (설계 후).
금지: intent 가중치를 외부 성능 claim 에 단독 사용 / human 라벨 없이 "지각적 artifact" 단정 / 평가 결과에 맞춘 가중치 튜닝.

## Note

Skate 축은 본 task 와 무관하게 이미 판별력 확보 (AR-063) — 본 task 는 **float 축 전용**. coordinate cleanup tool (AR-060/061) 의 claim boundary "text intent 처리 안 함" 은 유지 — intent 는 **evaluator/판정 층**에서 다룬다 (역할 분리).
