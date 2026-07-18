# 가설 status 통합 검토 (H-203/204/205/206) — 5단계 평가 + 제안 draft

> Board: AR-020 (통합 확장) · **status 전환·supersede promote = §3-11 사용자 승인 게이트 — 본 문서는 draft·권고까지만.** 평가 기준 = 각 가설의 **등록 시점 기각 조건 원문** (사후 수정 없음, HARKing 차단).

## 공통 배경 — 등록(5월) 이후 바뀐 두 가지 평가 전제

1. **NetGain 의 지위 하락**: 가설들의 1차 통화(currency)였던 NetGain 은 이후 metric provenance 게이트(§3-20)에서 **Category C (내부 proxy — 외부 최종 성능 근거 금지)** 로 분류됨. 최종 평가 통화는 Cat-A (R-Prec/MM-Dist/FID) + 지각(blind A/B)으로 이동.
2. **State feature 결함 발견**: 6월까지의 learned selector 증거는 **v0.1.0 gate evaluator** 로 산출된 state 를 사용 — AR-062 감사에서 Skate/Penetrate 축이 구조적 무신호(항상 0)였음이 판명 (수리 = AR-063; 재산출 = AR-065 진행 중).

이 두 가지는 가설 본문의 잘못이 아니라 **측정 도구의 사후 발견** — 아래 평가에 그대로 반영한다.

---

## H-2026-204 — 조건부 선택 + closed-loop > fixed post-proc (NetGain)

1. **제시한 가설**: artifact-conditioned tool selection + closed-loop 이 fixed post-processing 보다 NetGain 우위 (B5>B2 ≥+5%, closed-loop ≥+3%; 기각 5조건).
2. **실험 평가**: 2026-05-19 RQ1 리포트 (B8/B5/B2, G2) + E-chain (P4 고정 실패 / E5 조건부 물리 / E10 generator-조건부 Cat-A / E6·E8 지각) — raw 전부 snapshot 박제.
3. **실험적 근거**:
   - 등록 문구 기준: **임계 1 supports** (oracle B8 > B2 — 좋은 action 존재), **임계 2 contradicts** (rule-based B5 ≤ B2, 2026-05-19 — 기각조건 3 에 접근). closed-loop ≥+3% 는 신 체계에서 정식 미측정.
   - 신 증거 기준 (등록 통화 밖): "고정 균일 처리보다 조건부" 는 **Cat-A 로 강하게 지지** — 같은 보정이 MDM 개선(R@1+0.047·MM−1.0·FID−7.3) vs VQ 손상(E10, pool-scoped) → 고정 적용 불가. 단 그 조건부의 실체는 등록 당시 상정(rule-based tool selection)이 아니라 **기전-인식 보정 + generator/state 조건부 적용**.
4. **가설 평가**: **혼합 — 등록 문구(NetGain·B5 rule-based)로는 partial-contradicts, 핵심 메시지(조건부>고정)로는 신 통화에서 supports.** 등록된 임계·통화가 사후 게이트(§3-20)로 무력화되어 원문 그대로의 판정이 더 이상 연구를 대표하지 못함.
5. **다음 스텝 — 권고: supersede draft (아래 H-2026-207 안), 사용자 승인 시 promote.**

### H-2026-207 draft (promote 는 승인 후 — 본문 요지)

> **"Post-generation refinement 는 (a) 증상 점수가 아니라 생성 기전을 겨냥할 때, (b) generator/state 조건부로 적용될 때 품질을 개선한다 — 평가 통화는 Cat-A (R-Prec/MM-Dist/FID) + 사전등록 blind 지각."**
> 사전 정의(요지): 지지 = 기전 보정이 대상 generator 에서 Cat-A ≥2/3 유의 개선 + blind A/B ≥15/20; 비대상 generator 고정 적용 시 Cat-A 악화 (조건부 필요성). 기각 = 기전 보정이 Cat-A 개선 없음, 또는 조건부와 고정의 차이 없음. 현 증거 (E8·E10) 는 등록 시점에 이미 충족 상태이므로 promote 시 **"등록과 동시에 supports (post-hoc registration 명시)"** 로 정직하게 표기 — 신규 예측력은 KIT-ML(AR-067)·MLD(AR-080)·독립 pool(AR-024) 재현이 담당.

## H-2026-205 — learnable routing > rule-based

1. **제시한 가설**: learned selector (B6) 가 rule-based (B5) 대비 NetGain ≥+5%, oracle gap closure ≥30%, agreement ≥50%.
2. **실험 평가**: 2026-06-01 G2 learned vs baseline (12/12 재현 — AR-020 의 원 근거) + AR-052 계열 + 신 체계 AR-078/079 (routing gate 3연속 사전등록 검정).
3. **실험적 근거**:
   - 구 증거: G2 에서 learned 가 rule-based 를 개선 (12/12 재현, snapshot≥2) — **단, (i) state 의 Skate/Penetrate 축 무신호(v0.1.0), (ii) NetGain(Cat-C) 통화, (iii) G2 단일 generator.**
   - 신 증거: 순위 학습성은 지지 (richer state benefit-AUC 0.722 vs 단일 0.674, CI-clean) — "학습 가능한 신호 존재" 는 참. 그러나 실용 결정 기준(no-harm)은 3회 검정 모두 미달 (AR-078 "한계 확정", AR-079 "기각/한계").
4. **가설 평가**: **inconclusive (supports 전환 요건 미충족).** 학습성의 신호는 재확인됐으나, 구 supporting evidence 는 측정 도구 결함 + proxy 통화 위에 있고, 신 체계의 실용 판정은 미성립.
5. **다음 스텝 — 권고: `active` 유지 + 원 AR-020 의 `supported` 전환 제안을 철회 (진행하지 않음).** 재평가 조건: AR-065 잔여(수리된 state 로 재산출) 후 학습 재검 + no-harm gate 기준 통과.

## H-2026-206 — 학습 selector 의 generator-agnostic transfer

1. **제시한 가설**: A-학습 selector 가 B generator 에서 zero-shot ratio ≥0.7 등.
2. **실험 평가**: **등록된 transfer 실험 미착수** (선행 조건 H-205 supports 미성립).
3. **실험적 근거**: 간접 관측만 — framework(canonical 좌표·evaluator·state)는 3 generator 에 적용됐고, foot_skate 신호는 generator 를 AUC 0.83~0.88 로 구분. 단 이는 "selector transfer" 측정이 아님.
4. **가설 평가**: **평가 불가 (미착수)** — 전제 미충족 상태에서 판정하지 않음.
5. **다음 스텝 — 권고: `active` 유지 (미착수 명시).** 해석 note 만 기록: 신 증거는 "generator-agnostic = 같은 처방이 어디서나 통함"이 아니라 "**framework 는 agnostic, 처방은 조건부**"임을 시사 (E10) — 착수 시 이 해석으로 transfer 정의 재확인 필요.

## H-2026-203 — high-quality generator 에 no-harm (secondary)

1. **제시한 가설**: 학습된 라우팅이 high-quality 입력(G1 상정)에서 STOP/small-strength 로 분포·의미 보존 (FID ≤+5% 등 5 기각조건).
2. **실험 평가**: 등록된 형태(학습 selector + G1 50 sample + user study)로는 미착수. 관련 실측 = E10 (VQ 에 고정 적용 시 Cat-A 악화) + AR-074 (non-loco 0.85m push) + AR-078/079 (per-motion gate 미완).
3. **실험적 근거**: **등록 전제의 반전 발견** — 등록 시 "high-quality = G1(MDM/MLD 디퓨전)" 상정이었으나, 실측은 MDM 이 root 결함 generator 이고 **clean 축(root/skate)의 실체는 VQ**. VQ 에 무개입(STOP)이면 no-harm 자명 충족; 고정 개입 시 위반(E10); per-motion 학습 gate 는 아직 no-harm 보장 못 함 (harmful-apply 37%).
4. **가설 평가**: **inconclusive** — 등록된 절차로 미평가 + 전제(어느 generator 가 high-quality 인가) 재정의 필요.
5. **다음 스텝 — 권고: `active` 유지 + 전제 반전 note 기록.** 재개 시 "high-quality 입력" 을 실측 기준(VQ 계열 root/skate-clean)으로 재정의하는 supersede 를 그때 draft.

---

## 결정 요청 (§3-11 — 사용자 승인 게이트, 4건)

| # | 가설 | 권고 | 승인 시 처리 |
|---|---|---|---|
| 1 | H-2026-204 | **supersede** → H-2026-207 promote | H-207 파일 등록(위 draft 본문 정식화) + H-204 status `superseded` |
| 2 | H-2026-205 | `active` 유지 + **AR-020 supported-전환 철회** | 변경 이력에 본 리포트 링크 append |
| 3 | H-2026-206 | `active` 유지 (미착수 명시) | 변경 이력 append |
| 4 | H-2026-203 | `active` 유지 (전제 반전 note) | 변경 이력 append |

승인 형식: "1,2,3,4 승인" / 개별 지정 / 보류 지시 모두 가능. 승인 전까지 가설 파일은 무변경.

## Claim boundary

본 리포트 = 검토·draft. 어떤 status 도 아직 전환되지 않음. H-207 draft 의 "등록과 동시에 supports" 는 post-hoc registration 임을 promote 시 본문에 명시 (예측력 검증은 AR-024/067/080 재현이 담당).
