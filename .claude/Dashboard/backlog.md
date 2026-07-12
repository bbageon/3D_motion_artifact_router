# 🗂️ Backlog (미래)

> 미착수·미래 작업. **연구방향이 자주 바뀌므로** 방향 변경 시 재정렬. [index](README.md) · 다음 상태: [ready](ready.md).

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| AR-073 | Perceptual | AR-072 b2/b3 다중 평가자 (3명+ inter-rater) — root-aware A/B v3 를 외부 공개급 evidence 로. AR-023 연계 | 🟠 | b1(19/20) → 외부 claim 은 3+ 필요. 동일 pack 재사용 가능 |
| AR-074 | Tool | Root-aware 경계조건 — over-correction 꼬리(ratio>1.5 18.2%, "walk in place" 급소) + VQ + 비-locomotion 에서 solve 안전성. contact confidence 가중 여부 | 🟠 | [AR-072 결과](../../evals/reports/2026-07-12_ab_preference_result_v3_ar072.md) 후속. 조건부 적용/STOP 설계 |
| AR-075 | RL-Q | Orchestrator 통합 — RootGaitConsistencyTool = KDG 최상위 노드로 routing 에 편입 (root-first → 말단 tool 순서). registry candidate-set 확장 = §4 change-obligation | 🟡 | AR-072 tool 은 현재 export 만 (freeze). Safe Orchestration 계층에 병인 tool 편입 |
| AR-066 | Evaluator | Intent-aware float weighting — **경량 5-rule 확정**: prompt 4-bucket keyword 분류(ground-required/airborne/seated-lying/ambiguous) → ground-required 만 강한 float 판정, airborne/seated 제외·약가중, ambiguous 보류, raw score 전량 기록 | 🟡 | 07-08 경량화 (floating 은 축 하나 — 과투자 방지). [spec](../docs/dashboard-task-specs/AR-066-intent-aware-float-weighting.md). 임베딩 분류는 spot-check 실패 시만 optional |
| AR-065 | RL-Q | Effect-aware state 재구축 검토 — 기존 learned ranker (AR-040/052 계열) 는 v0.1.0 무신호 feature (Skate/Penetrate ≡ 0) 로 학습됨 → 수리된 v0.2.0 state 로 feature 재산출 + re-train 여부 판단 | 🟠 | [spec](../docs/dashboard-task-specs/AR-065-effect-aware-state-rebuild.md). AR-063 후속 |
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
