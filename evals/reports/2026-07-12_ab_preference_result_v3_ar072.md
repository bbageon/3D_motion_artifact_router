# AR-072 ④ — A/B v3 결과 (19/20 지지: 기전 처방이 지각 회복)

- 사전등록: [pack prereg](../snapshots/ab_preference_pack_v3_ar072_v1.json) · 결과 raw: [result](../snapshots/ab_preference_result_v3_ar072_v1.json) · 응답: [rater_sheet_filled_rater1.csv](../../reports/figures/2026-07-12/ar072_ab_v3_pack/rater_sheet_filled_rater1.csv)
- 자극: 실속도(20fps), blind (answer_key 미열람 판정), locomotion 한정 신규 20쌍.

## 결과

| 항목 | 값 |
|---|---|
| root-aware 보정본 선호 | **19/20 (95%)** — 이항 양측 **p ≈ 4×10⁻⁵**, 우연 강하게 기각 |
| 사전등록 분기 | **≥15/20 지지** → root pair 지각 회복 확인 (b1) |

## 동일 조건 대비 (처방만 교체 — rater·protocol·generator 동일)

| A/B | 처방 | 다루는 것 | 보정본 선호 |
|---|---|---|---|
| v1 (AR-058-3h) | cleanup (발 고정) | 증상 | 11/20 (p=0.82, 우연) |
| v2 (AR-058-3i) | combo (발+뼈) | 증상 | 10/20 (p=1.0, 우연) |
| **v3 (AR-072)** | **root solve (몸통 이동)** | **병인** | **19/20 (p<1e-4, 지지)** |

**동일 rater·동일 blind protocol·동일 MDM worst-tail에서 처방만 바꿨다.** 증상(발) 처방 2회는 우연, 병인(root) 처방은 압도 — "artifact score 최적화가 아니라 기전(root/gait consistency)을 고치면 지각 품질이 회복된다"의 강한 개입 증거(intervention evidence).

## 전 관문 종합 (AR-072)

| 관문 | 결과 |
|---|---|
| 불변성 (local·bone·y) | 0/900 위반 (구조 보장) |
| 기전 표적 | v_root/GT 0.419 → 1.018 (GT-free 복원) |
| R-Precision @1 | 0.276 → 0.319 (유의 개선) |
| MM-Dist | 4.89 → 4.29 (유의 개선) |
| FID(vs GT) | 7.19 → 3.31 (유의 개선, 절반↓) |
| root accel p95 | 0.0055 < GT 0.0064 (자연 동역학 회복) |
| **blind A/B** | **19/20 (p<1e-4)** |

물리 + 표준 지표(Cat-A) + 지각(b1)이 **모두 같은 방향** — anchoring 계열이 한 번도 못 이룬 정합.

## 탐색 관찰 (post-hoc — 인용 금지)

- 유일한 원본 선호 **pair_16**: ratio 회복이 가장 약하고(0.35→0.53) fs 개선도 최소(−0.0016) — "보정이 약했던" case. 처방 강도와 선호가 정합 (강할수록 선호).
- over-correction 꼬리 5쌍(ratio>1.5) **전부 보정본 선호** — 정지 인상보다 과속이 덜 거슬렸을 가능성 (n=5, 확증 아님). GT 대비 과속이 지각적으로 치명적이지 않다는 **약한 신호** — 단 A/B 표본이 worst-tail 편향이라 일반화 금지.

## Claim Boundary

**허용**: "b1 blind A/B 에서 root-aware 보정이 압도적 선호(19/20), 증상 처방(우연)과 정면 대비 — mechanism-aware correction 의 intervention evidence. 물리·Cat-A·지각이 모두 일치."
**금지**: "인과 증명" (개입이 경로·속도·리듬 동시 변경 → intervention evidence 까지) / "foot skating 해결" (단일 generator MDM · locomotion 한정 · b1) / b1 을 확정 지각 근거로 외부 인용 (b2/b3 필수, AR-023) / VQ·비-locomotion·타 데이터셋 일반화.

## 함의 / 다음

1. **P5 종합 승격**: "증상 최적화 ≠ 지각 품질" 위에 **"기전(root/gait)을 고치면 지각·표준지표·물리가 모두 회복"** 절 추가 — 연구의 품질 개선 목표가 기전 경로로 복원됨.
2. **AR-071 가설 격상**: root/gait mismatch 가 상관(ρ=0.34)을 넘어 **개입 증거**로 지지 (인과는 아직).
3. 후속(별도 게이트): b2/b3 다중 평가자 (AR-023 연계) · VQ/비-locomotion/over-correction 경계 조건 · orchestrator 통합 (root tool = KDG 최상위 노드).
