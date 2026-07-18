# 🗂️ Backlog (미래)

> 미착수·미래 작업. **연구방향이 자주 바뀌므로** 방향 변경 시 재정렬. [index](README.md) · 다음 상태: [ready](ready.md).

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| AR-075 | RL-Q | **③ Orchestrator 통합 — 보류** (AR-078 "한계 확정": 사전등록 기준 넘은 gate 없음 → 편입 대상 없음). 안전 운용 문서화 = generator-level rule (MDM 만 apply). 재개 조건 = harm-averse gate (AR-079 후보) 가 held-out 기준 통과 | 🟡 | "검증된 gate 만 통합" 원칙 유지. registry 확장 = §4 |
| AR-082 | Perceptual | (후보) **u\*(s) 정책 지각 검증** — Q(s,u) 정책 보정본 vs 원본 blind A/B (신규 prompt, 사전등록 1회). "정책 작동" 표현의 관문 (AR-081 spec §4 유보 해제 조건) | 🔴 | 착수 = 사용자 게이트 (A/B 판정 필요). pack 제작은 Agent 가능 |
| AR-083 | RL-Q | (후보) Closed-loop strength-Q — AR-081 spec §5 의 history 상태 (iteration_remaining·last_u·same_pair_count·누적 관측치·KDG mask) 추가 + transition 보존 검정 | 🟡 | one-shot 지지 후속. 착수 시 사전등록 |
| AR-080 | Evidence | MLD 재현 — root progression collapse 가 **diffusion 계열 현상**인지 (두 번째 diffusion 으로 E9 의 n=1 해소). 동일 회귀·분산 검정 (AR-076 harness 재사용) | 🟡 | **환경 구축 필요** (MLD 체크포인트·의존성·wrapper — G1 계열 AR-022 연계). 구축 후 착수 |
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
