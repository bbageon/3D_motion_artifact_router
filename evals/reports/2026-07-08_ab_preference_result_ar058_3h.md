# AR-058-3h — A/B 선호 검정 결과 (b1, 사전등록 fail 분기 발동)

- 사전등록: [ab_preference_pack_ar058_3h_v1.json](../snapshots/ab_preference_pack_ar058_3h_v1.json) (판정 기준 결과 수집 전 박제)
- 결과 raw: [ab_preference_result_ar058_3h_v1.json](../snapshots/ab_preference_result_ar058_3h_v1.json) · 응답: [rater_sheet_filled_rater1.csv](../../reports/figures/2026-07-08/ar058_3h_ab_preference_pack/rater_sheet_filled_rater1.csv)
- Tier: **b1** (단일 평가자, blind 유지 — answer_key 미열람 판정). quality-validated (b2/b3) 아님.

## 결과

| 항목 | 값 |
|---|---|
| 보정본 선호 | **11/20 (55%)** — 이항 양측 p≈0.824, **우연과 구분 불가** |
| 사전등록 분기 | **≤12/20 fail** → primary pair (foot skating ↔ coord cleanup) **지각 claim 불성립 (b1)** → **AGENTS §10 Go/Stop 정식 소집** (사용자 게이트) |

물리 지표와의 대비: 20쌍 전부 foot_skate_world 는 개선(평균 −15%)이었으나 지각 선호는 우연 수준 — **"물리 지표 개선 ≠ 지각 개선"이 primary pair 의 worst-tail 에서 b1 실측으로 확인됨.** 사용자의 사전 직관("명확한 문제로 안 보인다")과 일치.

## 탐색적 관찰 (post-hoc — 확증 아님, 인용 금지)

| 그룹 (ΔboneCV 0.03 기준) | 보정본 선호 |
|---|---|
| 왜곡 작은 쌍 (12) | 8/12 (67%) |
| 왜곡 큰 쌍 (8) | 3/8 (38%) |

- 가설 후보: **propagation bone 왜곡(AR-064 대상)이 skate 개선의 지각 이득을 상쇄.**
- 반례: pair_15 (왜곡 없음 + fs −51% 인데 원본 선호) — 긴 turning 동작에서 발 anchoring 의 뻣뻣함이 별도 부작용 후보.
- 확인 경로: **bone-preserving 보정(AR-064) 적용 후 새 사전등록 A/B 재검정** — 그 전까지 본 subgroup 수치는 어떤 주장에도 사용 금지.

## 함의 (Go/Stop 소집 시 검토 재료)

1. 사전등록 그대로: b1 에서 지각 claim 불성립 — P5 를 "지각 선호 확인" 어조로 쓸 수 없음.
2. 선택지 (사용자 결정): (a) §10 Stop/축소, (b) **repair-then-retest** — AR-064 (bone-preserving) 구현 후 재검정 1회 (탐색 관찰이 동기이나 결과 보장 없음), (c) 평가자 추가 (b2) — 단 b1 이 우연 수준이라 기대값 낮음.
3. 어느 경우든 물리 지표(Category B) 수준의 결과·방법론 기여(vacuity 감사, 좌표 protocol, 조건부 효과)는 유효 — claim 을 "지각 검증" 없이 "물리 지표 개선 + 지각 미확인"으로 정직 표기하는 경로도 존재.
