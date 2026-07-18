# Hypotheses Summary — ArtifactRouter 연구 가설 통합

> 연구를 진행하며 산출한 가설을 **한곳에 모은 요약/index**. **canonical 은 [`evals/hypotheses/<h_id>.md`](../../../evals/hypotheses/)** (append-only registry, [hypothesis-registry SKILL](../../skills/hypothesis-registry/SKILL.md) 규약). 본 문서는 참조용 요약이며 본문을 대체·수정하지 않는다. status 전환은 **사용자 승인 게이트** (AGENTS.md §3-11) — 본 문서에서 임의 변경 금지.

## 1. 한눈에 (status) — 2026-07-13 통합 검토 반영

| h_id | RQ | 한 줄 | status | evidence 현황 (2026-07-13) |
|---|---|---|---|---|
| [H-2026-207](../../../evals/hypotheses/H-2026-207.md) | RQ1+RQ2 재작성 | **기전-겨냥 + generator/state 조건부 refinement** (Cat-A+blind 지각 통화) | `active` | **post-hoc registration** — 생성 근거 E8·E10 존재; supports 전환은 독립 재현(AR-024/067/080) 담당 |
| [H-2026-205](../../../evals/hypotheses/H-2026-205.md) | RQ3 | learnable routing > rule-based | `active` | **inconclusive** — 구 12/12 재현 증거는 v0.1.0 무신호 feature·Cat-C 통화·단일 gen 위 (supported-전환 철회); 신 체계 순위 신호 있음(AUC 0.72)·no-harm gate 3연속 미달. 재평가 = AR-065 후 |
| [H-2026-206](../../../evals/hypotheses/H-2026-206.md) | RQ4 | 학습 selector 의 generator-agnostic transfer | `active` | **미착수** (선행 H-205 미성립). 해석 note: framework 는 agnostic, 처방은 조건부 (E10) |
| [H-2026-203](../../../evals/hypotheses/H-2026-203.md) | secondary | high-quality output No-harm | `active` | **inconclusive + 전제 반전** — clean 실체는 G1 이 아니라 VQ (AR-077); 등록 절차 미평가 |

> 공식 status 전환 (2026-07-13, 사용자 게이트 통과): **H-204 → superseded (H-207 로 대체)**. 그 외 3건 active 유지 — [통합 검토 리포트](../../../evals/reports/2026-07-13_hypothesis_status_review.md) 단일 출처.

## 2. 활성 가설 상세 요약

### H-2026-203 — No-harm on high-quality (secondary)
- **주장**: 이미 좋은 motion(G1 high-quality)에 orchestrator 적용 후 분포·semantic·fidelity 훼손 없음.
- **현황**: clean_noharm_holdout(HumanML3D clean)에서 모든 metric Δ=0 (Step 6 part 2) — 단 clean=GT이지 진짜 G1 generator 아님. **H-203의 "high-quality generator" 조건 미충족**. AR-026.
- track_scope: G1.

### H-2026-207 — Mechanism-targeted + conditional refinement (RQ1+RQ2 재작성, H-204 supersede)
- **주장**: refinement 는 (a) 증상 점수가 아니라 생성 기전(예: MDM root progression collapse)을 겨냥할 때, (b) generator/state 조건부로 적용될 때 품질 개선. 통화 = Cat-A (R-Prec/MM-Dist/FID) + 사전등록 blind 지각.
- **현황**: post-hoc registration (E8: MDM Cat-A 3지표 개선 + blind 19/20 / E10: VQ 고정 적용 시 악화). **supports 전환 = 독립 재현 표본에서만** (AR-024 새 pool / AR-067 KIT-ML / AR-080 MLD).
- track_scope: G1, G2. canonical: [H-2026-207.md](../../../evals/hypotheses/H-2026-207.md).

### H-2026-205 — Learnable routing > rule-based (RQ3)
- **주장**: supervised/contextual-bandit(RL-style) selector가 rule-based baseline 대비 net gain 의미 있게 개선.
- **현황 (2026-07-13 inconclusive)**: 구 증거(M0 12/12 재현)는 (i) v0.1.0 무신호 feature (AR-062), (ii) NetGain Cat-C 통화, (iii) G2 단일 위에서 측정 — **supported-전환 제안 철회**. 신 체계: 순위 학습성 신호 재확인 (richer-state benefit-AUC 0.722, CI-clean) / 실용 no-harm gate 3연속 미달 (AR-078/079).
- 재평가 조건: AR-065 잔여 (수리된 v0.2.0 state 재산출) 후 학습 재검 + no-harm 기준 통과.
- track_scope: G1, G2. canonical 기각조건·표본요건은 [H-2026-205.md](../../../evals/hypotheses/H-2026-205.md).

### H-2026-206 — Generator-agnostic 일반화 (RQ4)
- **주장**: generator A 학습 selector가 generator B output에 zero-shot/small-calibration transfer.
- 측정: Transfer NetGain ratio = NetGain(A-trained on B) / NetGain(B-trained on B). 임계 zero-shot ratio ≥ 0.7, small-cal ratio ≥ 0.9.
- **현황**: **미검증** — G1(MDM/MLD) 미구축. G2 단독으로 generator-agnostic 단정 금지 (§6-10, 최소 2 generator + paired test 의무). AR-022.
- track_scope: G1↔G2 bidirectional.

## 3. 종결 가설

| h_id | → 후속 | 종결 일 | 사유 |
|---|---|---|---|
| H-2026-200 | H-2026-204 | 2026-05-15 | 초기 등록 `track_scope:[G1,G2,G3]` misregistration 정정 (행정적) |
| H-2026-201 | H-2026-205 | 2026-05-15 | 부모 supersede 상속, RL refinement가 핵심 contribution 명시 (행정적) |
| H-2026-202 | H-2026-206 | 2026-05-15 | 동일 정정, transfer pair를 G1↔G2 bidirectional로 한정 (행정적) |
| **H-2026-204** | **H-2026-207** | **2026-07-13** | **실질적 supersede**: 등록 통화 NetGain 의 §3-20 Cat-C 강등 + 실측 경로 변화 (rule-based selection 지각 실패 → 기전-겨냥 보정). [통합 검토](../../../evals/reports/2026-07-13_hypothesis_status_review.md) |

## 4. 운영 규약 (요약)

- 가설은 **결과 보기 전 사전 등록** (HARKing 차단, Kerr 1998). append-only.
- status 전환·supersede promote = 사용자 승인 게이트 (AGENTS.md §3-11, §6-4).
- 판정은 [eval-compare SKILL §6](../../skills/eval-compare/SKILL.md) 5단계 리포트에서만, trial≥20 + snapshot≥2 하에서.
- 단일 trial/sample로 supports/contradicts 단정 금지 (§3-9, §6-7).
- 상세 절차: [hypothesis-registry SKILL](../../skills/hypothesis-registry/SKILL.md). 신규 등록: [`evals/hypotheses/_index.md`](../../../evals/hypotheses/_index.md)에서 다음 h_id 확인.
