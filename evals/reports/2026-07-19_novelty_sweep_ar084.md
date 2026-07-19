# AR-084 — 신규성 sweep: "MDM root collapse + contact-consistent root 재구성"의 기존 보고 여부

> 목적: 논문 가치의 관건인 ④ (기전 발견·처방이 이미 보고됐는가) 확인. web 조사 기반 — 아래 "남은 확인" 2건은 원문 정독 필요.

## 1. 조사 결과 — 가장 가까운 기존 연구 5건

| 논문 | 무엇을 말하나 | 우리와의 거리 |
|---|---|---|
| **LongDanceDiff** (arXiv 2308.11945, dance) | "foot sliding 은 global root trajectory 와 local rotation 의 **misalignment 에서 발생**" — 단 **정량화 없음** (원문 확인: 통계·측정 전무), 해결은 학습 시 loss/모듈(GTM), post-hoc 아님, **dance 한정** | 개념 한 줄이 선행 — **정량 진단·post-hoc 처방은 없음** |
| **Li et al. 2024** (Computer Animation & Virtual Worlds, foot-constrained ST-transformer) | keyframe 기반 합성에서 **"differentiable root trajectory reconstruction 을 post-process 로 수행해 root 를 posture 에 맞춰 foot-sliding 제거"** | **방법 조각이 가장 근접** — 단 자기 파이프라인 내장(decoder 출력 기반)·keyframe in-betweening 맥락. 인용 필수 |
| **RoHM** (CVPR 2024, 재구성) | diffusion 의 velocity-적분 root 표현이 **누적 오차로 drift** → absolute+relative 표현으로 완화 | 다른 설정(재구성)·**다른 병리(drift)** — 우리 C-검정은 drift 를 **기각**했고 발견은 "요구-무관 상수 붕괴" |
| Back to Basics (arXiv 2512.04499) | 표현 선택이 skating 에 영향 (global root 포함 표현이 심함) | 원문 확인: root 붕괴 진단·정량·post-hoc 재구성 **전무** |
| UnderPressure (2022) + Kovar et al. 2002 (고전, 배경) | IK 기반 footskate cleanup (발 고정 계열) | 우리의 "증상 도구" 가족 — 본 연구가 대비군으로 실측한 바로 그 계열 |

## 2. 신규성 판정 (현 시점)

**유지되는 신규성 (논문의 심장):**
1. **진단**: text-to-motion diffusion(MDM)의 root 속도가 프롬프트 요구와 무관한 **상수로 붕괴**함을 정량 규명 — 요구-추종 회귀(기울기 0.027)·분산 비율(5.7%)·3-generator 동일 검정 대비(VQ 0.59/0.63·95~99%)·drift(적분 누적) 기각. 문헌에는 정성적 한 줄(LongDanceDiff)과 다른-설정 drift(RoHM)만 존재 — **이 정량 진단 형태는 미보고로 판단**.
2. **사슬**: 증상 도구(발 고정)의 Cat-A 무효 vs 기전 도구(root 재구성)의 Cat-A 삼중 개선의 **통제 대비** + **generator-조건부 no-harm** (VQ 손상) + **강도 정책 Q(s,u)** — 이 인과 사슬 전체가 신규.

**하향되는 신규성 (표현 조정 필요):**
- "contact 기반 root 재구성" **방법 자체는 최초가 아님** (Li et al. 2024 계열) → 우리 표현은 "새 알고리즘"이 아니라 **"training-free·generator-무관 post-hoc tool 로 분리 + 강도 매개변수화(u) + 붕괴 진단에 근거한 적용"** 으로 위치 지정. 인용 의무.

## 3. 남은 확인 (원문 정독 2건 — 판정 뒤집힘 리스크 관리)

1. Li et al. 2024 원문 — root 재구성의 정확한 입력(contact? posture?)·독립 적용 가능성.
2. GMD/OmniControl/PriorMDM 계열 (trajectory **controllability** 연구) — 자발적 root 붕괴의 **진단** 서술이 서론에 있는지 (현재까지의 조사로는 제어 가능성 동기이지 붕괴 정량 아님).

## Sources

- [LongDanceDiff (arXiv 2308.11945)](https://arxiv.org/abs/2308.11945) · [ar5iv 원문 확인](https://ar5iv.labs.arxiv.org/html/2308.11945)
- [Li et al. 2024, CAVW](https://onlinelibrary.wiley.com/doi/10.1002/cav.2217)
- [RoHM (arXiv 2401.08570)](https://arxiv.org/pdf/2401.08570)
- [Back to Basics (arXiv 2512.04499)](https://arxiv.org/html/2512.04499v1)
- [UnderPressure (arXiv 2208.04598)](https://arxiv.org/abs/2208.04598)
- [GMD (arXiv 2305.12577)](https://arxiv.org/pdf/2305.12577)

## Claim boundary

web 조사 기반 판단 — "미보고로 판단"은 전수 조사가 아니라 표적 검색의 결론. 남은 확인 2건 전까지 논문 서론에 "최초" 표현 금지, "to our knowledge + 인용 대비" 형식 권장.
