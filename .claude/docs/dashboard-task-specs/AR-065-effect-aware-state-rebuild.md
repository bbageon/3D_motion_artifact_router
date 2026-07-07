# AR-065 - Effect-Aware State Rebuild on Repaired Evaluators

Status: backlog  
Epic: RL-Q  
Priority: 🟠  
Parent: AR-063 (gate 수리 후속)

## Goal

기존 learned ranker (AR-040/AR-052 계열) 의 state feature 가 **v0.1.0 gate evaluator** 로 산출되어 Skate/Penetrate 축이 **항상 0 (무신호)** 이었던 문제를 해소 — 수리된 v0.2.0 evaluator 로 feature 를 재산출하고 re-train 필요 여부를 판단한다.

## Why This Exists

AR-062 audit: Skate/Penetrate gate 는 구조적 vacuity → effect-aware state 의 해당 feature 는 학습에 기여한 적이 없음 (탈락 feature 와 동일). AR-063 수리 후 Skate 는 실제 판별력 (MDM fire 23.1% vs VQ 1.3~1.7%) 을 가지므로, state 에 **처음으로 skate 신호**가 들어올 수 있다.

## Scope

1. effect_aware_state 의 feature 를 v0.2.0 evaluator (trajectory 입력 포함) 로 재산출.
2. 구/신 feature 분포 비교 (어떤 축이 얼마나 달라졌나 — 특히 skate/penetrate).
3. 기존 학습 결과 (AR-040/052) 의 결론이 feature 교체로 뒤집히는지 spot-check → re-train 여부 판단 (re-train 자체는 별도 결정).

## Success Criteria

- feature 재산출 + 분포 비교 리포트.
- "기존 learned 결과 해석 시 'skate/penetrate state 무신호' caveat" 의 구체 영향 범위 확정.

## Claim Boundary

금지: 재산출만으로 기존 learned 결과 무효 주장 (분포 비교 근거 필요) / re-train 없이 신규 성능 claim.

## Note

기존 snapshot 은 무수정 (§6-2) — severity_version 으로 구분. trajectory 입력 protocol (AR-048/AR-062 A-4) 준수.
