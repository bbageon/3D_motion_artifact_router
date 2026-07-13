# AR-065 - Effect-Aware State Rebuild on Repaired Evaluators

Status: **in-progress** (2026-07-13 착수 — 4차 피드백: P5 최종화보다 우선)  
Epic: RL-Q  
Priority: 🟠  
Parent: AR-063 (gate 수리 후속) / AR-077 (routing gate 재료 — AR-078 선행)

## Pre-action State Feature 목록 (4차 피드백 채택 — 결과 보기 전 고정)

전부 **원본 모션 + prompt 만으로, tool 적용 전** 계산 (GT-free, after-action leakage 금지):

| # | feature | 정의 (pre-action) |
|---|---|---|
| 1 | `generator_id` | 배포 시 알려진 메타데이터 |
| 2 | `foot_skate` | v2 정의 (원본 관측) |
| 3 | `root_gait_mismatch` | **counterfactual**: 접지발의 골반-상대 후류 속도(=gait 함의 속도) − 실제 root 속도. induced_disp/path_gain 의 pre-action 대체물 |
| 4 | `contact_persistence` | 접지 run 평균 길이 + 접지 비율 |
| 5 | `trajectory_shape` | 경로 길이 + 직진도(net displacement / path length) |
| 6 | `locomotion_intent` | prompt keyword bucket (경량 rule — AR-066 계열) |
| 7 | (optional) `support/uncertainty` | kNN/ensemble — 본 착수 범위 밖, AR-042 연계 |

**보정 후 값 (Cat-A 변화·path_gain·induced_disp) = label 또는 최종 guard 로만** — state 로 사용 금지.

## Goal

기존 learned ranker (AR-040/AR-052 계열) 의 state feature 가 **v0.1.0 gate evaluator** 로 산출되어 Skate/Penetrate 축이 **항상 0 (무신호)** 이었던 문제를 해소 — 수리된 v0.2.0 evaluator 로 feature 를 재산출하고 re-train 필요 여부를 판단한다.

## Why This Exists

AR-062 audit: Skate/Penetrate gate 는 구조적 vacuity → effect-aware state 의 해당 feature 는 학습에 기여한 적이 없음 (탈락 feature 와 동일). AR-063 수리 후 Skate 는 실제 판별력 (MDM fire 23.1% vs VQ 1.3~1.7%) 을 가지므로, state 에 **처음으로 skate 신호**가 들어올 수 있다.

## Scope

1. effect_aware_state 의 feature 를 v0.2.0 evaluator (trajectory 입력 포함) 로 재산출.
2. 구/신 feature 분포 비교 (어떤 축이 얼마나 달라졌나 — 특히 skate/penetrate).
3. 기존 학습 결과 (AR-040/052) 의 결론이 feature 교체로 뒤집히는지 spot-check → re-train 여부 판단 (re-train 자체는 별도 결정).
4. **AR-078 routing gate 용 richer pre-action state 후보 확장** (AR-077 후속): root-deficit proxy 등.

## ⚠️ Counterfactual Feature 원칙 (AR-077 2차 피드백)

routing gate 의 state 는 **correction 을 적용하기 전(pre-action)에 원본 모션만으로 계산 가능** 해야 한다. `path_gain`·`induced_disp` 는 **실제 correction 결과에서 꺼내면 after-action leakage** (결과를 보고 결과를 예측). AR-065/AR-078 에서 이들을 state 로 쓰려면:
- **pre-action counterfactual 로 정의**: 원본 모션의 접지 패턴에서 "root solve 를 하면 얼마나 움직일지" 를 tool 적용 없이 추정 (예: 접지 구간의 발-골반 상대 변위 합 = 예상 root 보정량 proxy) — 실제 apply 전에 계산.
- foot_skate 는 원본 관측이라 이미 pre-action ✓. path_gain/induced_disp 만 counterfactual 화 필요.
- gate 학습·평가 시 feature 가 label(correction benefit) 산출에 쓰인 정보를 미리 보지 않도록 분리 (leakage 재차 차단).

## Success Criteria

- feature 재산출 + 분포 비교 리포트.
- "기존 learned 결과 해석 시 'skate/penetrate state 무신호' caveat" 의 구체 영향 범위 확정.

## Claim Boundary

금지: 재산출만으로 기존 learned 결과 무효 주장 (분포 비교 근거 필요) / re-train 없이 신규 성능 claim.

## Note

기존 snapshot 은 무수정 (§6-2) — severity_version 으로 구분. trajectory 입력 protocol (AR-048/AR-062 A-4) 준수.
