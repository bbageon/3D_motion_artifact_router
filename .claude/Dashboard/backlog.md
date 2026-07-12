# 🗂️ Backlog (미래)

> 미착수·미래 작업. **연구방향이 자주 바뀌므로** 방향 변경 시 재정렬. [index](README.md) · 다음 상태: [ready](ready.md).

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| AR-065 | RL-Q | **① Effect-aware state 재구축** — 수리된 v0.2.0 state + **counterfactual root-deficit proxy** feature. ⚠️ path_gain/induced_disp 는 correction 결과에서 꺼내면 after-action leakage → **원본 모션에서 tool 적용 전 계산 가능한 pre-action counterfactual** 로 정의 | 🟠 | [spec](../docs/dashboard-task-specs/AR-065-effect-aware-state-rebuild.md). AR-063/077 후속. AR-078 의 richer-state 재료 |
| AR-078 | RL-Q | **② Routing gate 비교 (held-out benefit-AUC)** — generator-only / skate-only(AR-077: benefit-AUC ~0.68 δ조건부, false-apply 50% 불충분) / **richer-state**(AR-065) gate 를 same held-out 에서 benefit-AUC·false-apply 로 비교 | 🔴 | AR-077 2차 피드백 결정: 단일 skate gate no-harm 불충분 → richer pre-action state 필요. **AR-065 선행** |
| AR-075 | RL-Q | **③ Orchestrator 통합 — 검증된 gate 만** — RootGaitConsistencyTool = KDG 최상위 + STOP gate. ⚠️ foot-skate 단일 gate 통합 금지 (AR-077) → AR-078 에서 held-out 개선 확인된 gate 만 편입 | 🟡 | AR-078 결과 대기. registry 확장 = §4. Safe Orchestration no-harm gate |
| AR-066 | Evaluator | Intent-aware float weighting — **경량 5-rule 확정**: prompt 4-bucket keyword 분류(ground-required/airborne/seated-lying/ambiguous) → ground-required 만 강한 float 판정, airborne/seated 제외·약가중, ambiguous 보류, raw score 전량 기록 | 🟡 | 07-08 경량화 (floating 은 축 하나 — 과투자 방지). [spec](../docs/dashboard-task-specs/AR-066-intent-aware-float-weighting.md). 임베딩 분류는 spot-check 실패 시만 optional |
| AR-064 | Tool | Bone-preserving propagation — coord cleanup 의 leg-chain 선형 propagation(NOT IK)이 u 비례 BoneCV 상승(+0.045~0.053 @u=1.0) 유발 → IK 또는 사후 bone re-projection 으로 대체 | 🟠 | [spec](../docs/dashboard-task-specs/AR-064-bone-preserving-propagation.md). AR-061 후속. u=0.25 는 +0.002 로 미미 — 저강도 우선 운용 가능 |
| AR-059 | Future-RL | RS-GRPO-style scoped reward / intent evaluator — 현재는 오버엔지니어링, 먼 훗날 human labels 축적 후 reward-model/RL 학습 후보 | ⚪ | 지금은 scope 분리 철학만 AR-058-3e human rubric에 반영 |
| AR-041 | RL-Q | v1 target-aware ranker — target proposal + local/relation feature 구현 | 🟠 | AR-037 후속. action=`(tool,target,u)`, g2_stress oracle gap 회수 목적 |
| AR-042 | RL-Q | v2 support-aware ranker — data support/uncertainty state + OOD ablation | 🟡 | AR-037 후속. kNN/ensemble 기반 support, offline value overestimation 억제 |
| AR-025 | Evidence | H-2026-204 (artifact-conditioned + closed-loop > fixed post-proc) 5단계 평가 | 🟡 | B2-family vs M0+gate, snapshot≥2 |
| AR-026 | Evidence | H-2026-203 (high-quality generator no-harm) — **진짜 G1 필요** (clean=GT 는 미충족) | ⚪ | AR-022 의존 |
| AR-027 | RL-Q | Stage 4 constrained offline RL (CQL/IQL) — reranking 넘어 policy optimization | ⚪ | Stage 3 안정 후 |
| AR-028 | RL-Q | sequence oracle (multi-step ceiling) vs step-1 oracle gap 측정 | ⚪ | 현재 oracle = step-1 dense |
| AR-032b | Harness | (선택) gitignored regenerable 디스크 정리 (final_motions 96MB 등) | ⚪ | git 추적 밖. 재현 시 재생성 필요 → 보류, 디스크 압박 시만 |
| AR-067 | Evidence | KIT-ML 데이터셋 일반화 — 현재 결론은 HumanML3D 단일 생태계(시험지·정답지·채점기·체크포인트) 범위. KIT-ML 에서 유병률·tool 효과 방향 재현 확인 | ⚪ | 사용자 directive 07-08: **등록만, 착수 나중**. [spec](../docs/dashboard-task-specs/AR-067-kit-ml-dataset-generalization.md). MMM 21-joint retargeting = §3-1/§4 게이트 대상. 외부 공개 시 그전까지 "단일 벤치마크 범위" 명시 |

---

운영: 우선순위 오르면 [ready.md](ready.md) 로 승격. 폐기 시 [cancelled.md](cancelled.md).
