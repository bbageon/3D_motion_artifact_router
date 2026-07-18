# AR-079 — Harm-Averse Selective Gate: 판정 "기각/한계", 단 trade-off 지형 확보

- 사전등록: [spec](../../.claude/docs/dashboard-task-specs/AR-079-harm-averse-gate.md) · Raw: [harm_averse_gate_ar079_v1.json](../snapshots/harm_averse_gate_ar079_v1.json) · harness: [tools/harm_averse_gate_ar079.py](../../tools/harm_averse_gate_ar079.py)
- 질문: "해로운 적용을 사전에 막을 수 있나?" — harm(ΔMM>+δ)을 **직접 예측**하는 H-LR + 결합 score HA = P(benefit) − λ·P(harm) (λ=2.0, calibration 에서만 선택 — 제약: capture ≥ 0.8×G5). **전 수치 δ 조건부 exploratory.**

## 1. 핵심 결과 (holdout 1,350)

**harm 예측 가능성**: harm-AUC **0.663** [0.615, 0.708] — harm 은 현 pre-action state 로 **중간 수준 예측 가능** ("예측 불가" 아님).

적용률별 (harm / capture):

| gate | 10% | 30% | 50% |
|---|---|---|---|
| G2 skate | 0.170 / 0.225 | 0.360 / 0.451 | 0.393 / 0.641 |
| G5 benefit-LR | 0.185 / 0.221 | 0.291 / 0.511 | 0.375 / 0.685 |
| **HA harm-averse** | 0.193 / 0.213 | **0.262** / 0.468 | **0.265** / 0.618 |
| (참고) 순수 low-harm | **0.067** / 0.050 | 0.153 / 0.215 | 0.219 / 0.478 |

harm@30 paired diff: **HA−G5 = −0.024 [−0.077, +0.025]** (비유의) · HA−G2 = **−0.088 [−0.153, −0.025]** (유의).

## 2. 사전등록 판정: **"기각/한계"**

기준 = (a) harm@30 HA−G5 diff CI 상한<0 **AND** (b) capture ≥ 0.8×G5. **(a) 미충족** (방향은 개선이나 CI 가 0 포함) → 등록 분기대로 기각/한계. **per-motion no-harm gate 는 여전히 미완** — AR-078 결론 유지.

## 3. 판정 밖 관측 (exploratory — 운용 설계의 재료)

1. **skate 단일 gate 대비는 유의하게 안전** (−0.088 CI-clean) — harm-aware 결합이 무가치하지 않음.
2. **고적용률에서 harm 평탄화**: 적용률 30→50% 로 늘려도 HA 는 harm ~0.26 유지 (G5 는 0.29→0.375 로 상승) — 많이 적용할수록 harm-averse 항의 가치가 커짐.
3. **보수 운용 모드 존재**: 순수 low-harm 정렬은 적용률 10% 에서 **harm 6.7%** 달성 (capture 5%) — "적게, 확실한 것만 고치는" 운용점이 실측으로 존재. no-harm 을 최우선하는 배포에서 선택 가능한 corner.
4. λ=4 는 calib harm 0.156 까지 낮추나 capture 제약(≥0.8×G5) 위반 — trade-off 는 연속적이며 공짜가 아님.

## 4. 함의

우선순위 표의 질문 "해로운 적용을 사전에 막을 수 있나?"의 답: **"부분적으로 — 통계적으로 유의한 수준으로 G5 를 이기진 못하나(δ 조건부), harm 은 예측 가능(0.66)하고 적용률-harm trade-off 지형(보수 corner 포함)은 확보됐다."** AR-075 재개 시 "검증된 gate" 는 여전히 부재하나, **보수 운용 모드**(저적용률 low-harm 정렬)는 운용 문서화 후보. 다음 개선 방향은 feature 가 아니라 (i) 더 큰 표본 (비유의가 검정력 문제인지), (ii) 지각 기반 harm label (MM proxy 한계) — 둘 다 AR-073/b2 이후 판단.

## Claim Boundary

허용: "harm 은 pre-action state 로 중간 수준 예측 가능(AUC 0.66); harm-averse 결합은 skate 대비 유의 개선, benefit-LR 대비 비유의; 적용률-harm trade-off 지형과 보수 corner 실측".
금지: "harm gate 성립" (기준 미충족) / MM-Dist proxy 를 지각 harm 으로 / λ 사후 조정 / 단일 벤치마크·δ 밖 일반화.

## Grounding (§3-22)

Gangrade et al. AISTATS 2021 (selective prediction) · Guo HumanML3D CVPR2022 (MM-Dist).
