# Hypotheses Summary — ArtifactRouter 연구 가설 통합

> 연구를 진행하며 산출한 가설을 **한곳에 모은 요약/index**. **canonical 은 [`evals/hypotheses/<h_id>.md`](../../evals/hypotheses/)** (append-only registry, [hypothesis-registry SKILL](../skills/hypothesis-registry/SKILL.md) 규약). 본 문서는 참조용 요약이며 본문을 대체·수정하지 않는다. status 전환은 **사용자 승인 게이트** (AGENTS.md §3-11) — 본 문서에서 임의 변경 금지.

## 1. 한눈에 (status)

| h_id | RQ | 한 줄 | status | evidence 현황 |
|---|---|---|---|---|
| [H-2026-203](../../evals/hypotheses/H-2026-203.md) | secondary | high-quality generator output No-harm (분포·semantic·fidelity 훼손 없음) | `active` | **미충족** — clean(GT) no-harm만 확인, 진짜 G1 필요 |
| [H-2026-204](../../evals/hypotheses/H-2026-204.md) | RQ1+RQ2 | artifact-conditioned tool selection + closed-loop > fixed post-processing (NetGain) | `active` | 부분 — B2-family 대비 우위 evidence 있음, 5단계 평가 미완 (AR-025) |
| [H-2026-205](../../evals/hypotheses/H-2026-205.md) | RQ3 | learnable routing (supervised/contextual-bandit) > rule-based (net gain) | `active` | **강함** — M0 > random 전 holdout 유의, snapshot 2개 12/12 재현 (AR-020) |
| [H-2026-206](../../evals/hypotheses/H-2026-206.md) | RQ4 | 학습 selector의 generator-agnostic 일반화 (G1↔G2 bidirectional) | `active` | **미검증** — G1 미구축 (AR-022) |

> 공식 status 전환된 가설 = **0건** (4 active 모두). H-205는 요건(snapshot≥2 + 재현) 충족, 승인만 대기.

## 2. 활성 가설 상세 요약

### H-2026-203 — No-harm on high-quality (secondary)
- **주장**: 이미 좋은 motion(G1 high-quality)에 orchestrator 적용 후 분포·semantic·fidelity 훼손 없음.
- **현황**: clean_noharm_holdout(HumanML3D clean)에서 모든 metric Δ=0 (Step 6 part 2) — 단 clean=GT이지 진짜 G1 generator 아님. **H-203의 "high-quality generator" 조건 미충족**. AR-026.
- track_scope: G1.

### H-2026-204 — Artifact-conditioned selection + closed-loop > fixed post-processing (RQ1+RQ2)
- **주장**: artifact 조건부 tool 선택 + closed-loop refinement가 고정 후처리(B2-family fixed smoothing)보다 NetGain 우위.
- **현황**: B2 fixed smoothing의 해악(100% BoneLengthCV violation, FID 악화) 정량 확립. closed-loop 0% violation + FID 보존. **단 H-204 5단계 평가(B2-family vs M0+gate, snapshot≥2)는 AR-025로 미완**.
- 기각 조건: NetGain·artifact reduction 임계 미달. Oracle best-tool baseline type(single-step/sequence) 명시 의무 (§3-16).
- track_scope: G1, G2.

### H-2026-205 — Learnable routing > rule-based (RQ3, 1차 contribution)
- **주장**: supervised/contextual-bandit(RL-style) selector가 rule-based baseline 대비 net gain 의미 있게 개선.
- **현황 (가장 강함)**: per-state paired test — M0(broad-support) > random+gate 전 real holdout 유의 (g2_stress p=1.2e-6 d=0.47, g2_natural p=1.6e-17 d=0.87). M0 vs heuristic: natural/synthetic 유의, stress NetGain 동률+FID 우위. **snapshot 2개(v2 seed20260531, v3 seed20260615) 12/12 방향+유의성 재현**.
- 사전 정의 기각 조건: paired Wilcoxon p≥0.05 또는 effect size 미달. snapshot≥2 의무.
- **status 전환(active→supported)**: 요건 충족, **사용자 승인 게이트 대기** (AR-020).
- track_scope: G1, G2. canonical 기각조건·표본요건은 [H-2026-205.md](../../evals/hypotheses/H-2026-205.md).

### H-2026-206 — Generator-agnostic 일반화 (RQ4)
- **주장**: generator A 학습 selector가 generator B output에 zero-shot/small-calibration transfer.
- 측정: Transfer NetGain ratio = NetGain(A-trained on B) / NetGain(B-trained on B). 임계 zero-shot ratio ≥ 0.7, small-cal ratio ≥ 0.9.
- **현황**: **미검증** — G1(MDM/MLD) 미구축. G2 단독으로 generator-agnostic 단정 금지 (§6-10, 최소 2 generator + paired test 의무). AR-022.
- track_scope: G1↔G2 bidirectional.

## 3. 종결 가설 (행정적 supersede, 2026-05-15)

| h_id | → 후속 | 사유 |
|---|---|---|
| H-2026-200 | H-2026-204 | 초기 등록 `track_scope:[G1,G2,G3]` misregistration 정정 (G3는 본 프로젝트 scope 외) |
| H-2026-201 | H-2026-205 | 부모 supersede 상속, RL refinement가 핵심 contribution 명시 |
| H-2026-202 | H-2026-206 | 동일 정정, transfer pair를 G1↔G2 bidirectional로 한정 |

> supersede 공통 사유: 본 저장소는 독립 프로젝트이며 이전 저장소 LLM motion 실험의 후속 아님. **결과 기반 결론이 아닌 행정적 정정**.

## 4. 운영 규약 (요약)

- 가설은 **결과 보기 전 사전 등록** (HARKing 차단, Kerr 1998). append-only.
- status 전환·supersede promote = 사용자 승인 게이트 (AGENTS.md §3-11, §6-4).
- 판정은 [eval-compare SKILL §6](../skills/eval-compare/SKILL.md) 5단계 리포트에서만, trial≥20 + snapshot≥2 하에서.
- 단일 trial/sample로 supports/contradicts 단정 금지 (§3-9, §6-7).
- 상세 절차: [hypothesis-registry SKILL](../skills/hypothesis-registry/SKILL.md). 신규 등록: [`evals/hypotheses/_index.md`](../../evals/hypotheses/_index.md)에서 다음 h_id 확인.
