# AR-079 - Harm-Averse Selective Gate

Status: in-progress (2026-07-13 착수 — 사용자 우선순위 2)  
Epic: RL-Q  
Priority: 🔴  
Parent: AR-078 ("한계 확정" — benefit 분류 gate 는 harm 을 못 낮춤; 순위≠안전 축 분리)

## Goal

AR-078 의 발견 — benefit 을 잘 맞히는 score 와 harm 을 피하는 score 는 다른 축 — 에 따라,
**harm 을 직접 예측·회피하는 gate** 가 같은 held-out 에서 harmful-apply 를 낮추는지 확인.
질문: "해로운 적용을 사전에 막을 수 있나?"

## 사전등록 (2026-07-13 — 결과 보기 전 고정)

### 데이터·라벨

- AR-078 과 동일: 두 CSV join, 동일 split(calib/holdout), δ = calibration VQ |ΔMM| median.
- benefit = ΔMM < −δ, **harm = ΔMM > +δ** (별도 라벨 — 처음으로 직접 예측 대상).

### Gate

| gate | 구성 |
|---|---|
| (참조) G2 skate / G5 benefit-LR | AR-078 그대로 재현 |
| H-LR | **harm 예측기**: LR(같은 pre-action feature, label=harm) — harm 이 예측 가능한가 (harm-AUC) |
| **HA (harm-averse)** | score = P(benefit) − λ·P(harm). λ ∈ {0.25, 0.5, 1, 2, 4} 를 **calibration 에서만** 선택: calib harm@30 최소화, 단 calib capture@30 ≥ 0.8×G5-calib. λ 고정 후 holdout 평가 |

### 지표 (holdout) + 판정 기준 (사전 고정)

- harm-AUC (H-LR): **harm 자체가 예측 가능한가** — ≤0.55 면 접근 자체가 어렵다는 정직한 결과.
- harm@적용률·capture@적용률 (G2/G5/HA), bootstrap paired diff (B=1000, multiplicity 보존).
- **지지** = holdout 에서 (a) harm@30(HA) − harm@30(G5) diff CI 상한 < 0 **그리고** (b) capture@30(HA) ≥ 0.8 × capture@30(G5).
- **기각/한계** = 미충족 (그 자체로 유효 — "harm 은 현 state 로 예측 불가"의 특정).
- 전 수치 δ 조건부 exploratory (AR-077/078 caveat 승계). "routing 작동" 표현은 여전히 금지 (AR-073 + 지지 시에만 재논의).

## Claim Boundary

허용: "harm 직접 예측/회피 gate 의 held-out 성능 — harm 감소 여부와 capture 대가".
금지: MM-Dist proxy 를 지각 harm 으로 / λ·τ 를 holdout 에서 고르는 사후 조정 / 단일 벤치마크 밖 일반화.

## Grounding (§3-22)

Gangrade et al. AISTATS 2021 (selective prediction — abstain 원리) · Guo HumanML3D CVPR2022 (MM-Dist).
