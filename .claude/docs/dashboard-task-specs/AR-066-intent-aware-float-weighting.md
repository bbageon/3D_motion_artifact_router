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

## Tiered Design (싼 것부터 — 순서 고정)

| Tier | 방식 | text 사용 | 비용 |
|---|---|---|---|
| **0. 지속시간 가중** | float segment 를 **persistence** 로 가중 (예: <3 frame 순간 들림 무시 / 지속 부양만 점수) — heel-roll·toe-off 노이즈 제거 | 없음 (motion-only) | 최소 — 먼저 실행 |
| **1. Intent 층화(stratification)** | prompt 를 keyword rule 로 bucket 화 — airborne/acrobatic (jump·hop·leap·climb·swim·crawl·sit·lie…) / locomotion / stationary — bucket 별 float 채점 억제 또는 **별도 calibration** | 약 (keyword) | 낮음 — prompt bank (AR-049 annotation) 재사용 |
| **2. 학습 intent-aware contact** | text-conditioned contact-intent estimator (provenance "Item 6 contact estimator" + AR-059/VisRAG 계열) | 강 (학습) | 높음 — 미래 |

## Validation Plan

- Tier 0/1 적용 후 clean 재calibration → **p99 가 판별 가능한 수준으로 내려오는지** + 생성물 fire-rate 가 generator 간 분리를 보이는지.
- AR-058-3f human pack 의 `prompt_intent_conflict` / `dominant_artifact=floating` 라벨 (응답 수집 시) 을 external validation 으로 사용.
- **순환 금지**: bucket rule·가중치를 평가 결과(generator 간 차이가 나오도록)에 맞춰 tuning 하지 않는다 — clean + human 라벨로만 calibrate (HARKing 차단, §6-4 정신).

## Research Grounding (§3-22)

- **HumanML3D** (Guo et al., CVPR 2022) — text-motion 대응 관계가 평가의 1급 신호 (R-Precision/MM-Dist) : intent 신호의 출처.
- **HuMoR** (Rempe et al., ICCV 2021) — contact/ground-aware plausibility : contact 를 관측이 아니라 **추정 대상**으로 다루는 근거.
- **PhysDiff** (Yuan et al., ICCV 2023) — float/slide/penetration 이 생성물의 실제 실패 유형.
- **PP-Motion** (ACM MM 2025) — physical-perceptual fidelity 정합 : 물리 지표는 지각 판단과 연결돼야 함.
- Tier 1 keyword bucket rule 자체는 **engineering heuristic** (논문 근거 아님 — 명시 의무).

## Claim Boundary

허용: "intent 조건화가 float 판정의 변별력을 개선하는지 측정한다" (설계 후).
금지: intent 가중치를 외부 성능 claim 에 단독 사용 / human 라벨 없이 "지각적 artifact" 단정 / 평가 결과에 맞춘 가중치 튜닝.

## Note

Skate 축은 본 task 와 무관하게 이미 판별력 확보 (AR-063) — 본 task 는 **float 축 전용**. coordinate cleanup tool (AR-060/061) 의 claim boundary "text intent 처리 안 함" 은 유지 — intent 는 **evaluator/판정 층**에서 다룬다 (역할 분리).
