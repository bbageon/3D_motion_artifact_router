# 🗂️ Backlog (미래)

> 미착수·미래 작업. **연구방향이 자주 바뀌므로** 방향 변경 시 재정렬. [index](README.md) · 다음 상태: [ready](ready.md).

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| AR-064 | Tool | Bone-preserving propagation — coord cleanup 의 leg-chain 선형 propagation(NOT IK)이 u 비례 BoneCV 상승(+0.045~0.053 @u=1.0) 유발 → IK 또는 사후 bone re-projection 으로 대체 | 🟠 | [AR-061 snapshot](../../evals/snapshots/coordinate_footskate_effect_ar061_v1.json) guard 결과. u=0.25 는 +0.002 로 미미 — 저강도 우선 운용 가능 |
| AR-059 | Future-RL | RS-GRPO-style scoped reward / intent evaluator — 현재는 오버엔지니어링, 먼 훗날 human labels 축적 후 reward-model/RL 학습 후보 | ⚪ | 지금은 scope 분리 철학만 AR-058-3e human rubric에 반영 |
| AR-041 | RL-Q | v1 target-aware ranker — target proposal + local/relation feature 구현 | 🟠 | AR-037 후속. action=`(tool,target,u)`, g2_stress oracle gap 회수 목적 |
| AR-042 | RL-Q | v2 support-aware ranker — data support/uncertainty state + OOD ablation | 🟡 | AR-037 후속. kNN/ensemble 기반 support, offline value overestimation 억제 |
| AR-025 | Evidence | H-2026-204 (artifact-conditioned + closed-loop > fixed post-proc) 5단계 평가 | 🟡 | B2-family vs M0+gate, snapshot≥2 |
| AR-026 | Evidence | H-2026-203 (high-quality generator no-harm) — **진짜 G1 필요** (clean=GT 는 미충족) | ⚪ | AR-022 의존 |
| AR-027 | RL-Q | Stage 4 constrained offline RL (CQL/IQL) — reranking 넘어 policy optimization | ⚪ | Stage 3 안정 후 |
| AR-028 | RL-Q | sequence oracle (multi-step ceiling) vs step-1 oracle gap 측정 | ⚪ | 현재 oracle = step-1 dense |
| AR-032b | Harness | (선택) gitignored regenerable 디스크 정리 (final_motions 96MB 등) | ⚪ | git 추적 밖. 재현 시 재생성 필요 → 보류, 디스크 압박 시만 |

---

운영: 우선순위 오르면 [ready.md](ready.md) 로 승격. 폐기 시 [cancelled.md](cancelled.md).
