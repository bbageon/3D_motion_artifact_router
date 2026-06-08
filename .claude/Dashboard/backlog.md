# 🗂️ Backlog (미래)

> 미착수·미래 작업. **연구방향이 자주 바뀌므로** 방향 변경 시 재정렬. [index](README.md) · 다음 상태: [ready](ready.md).

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| AR-049 | Generator | hard-tier prompt bank (n=300, primary) + tiered 표본 (soft 50 diagnostic) | 🟠 | 실사용 evidence는 hard-tier에서. **설계(A) 확정**: soft 50 = **per-sample(B/C) 전용·FID/R-Prec 인용 금지**, tier 비교는 per-sample 지표만. **hard 300=primary**(full Category-A, per-gen 동일·3seed·paired Δ+CI). in-distribution HumanML3D만(GT 유지). complexity 기준은 **추후 결정**. [spec](../docs/dashboard-task-specs/AR-049-hard-tier-prompt-bank.md) |
| AR-040 | RL-Q | v0 global quality-aware ranker — `q_proxy` 정의 + v0 state vector + M1 dense effect oracle | 🟠 | **AR-044 이후 착수**. G2 real problem taxonomy/headroom 결과로 q_proxy 정의. action=`(tool,u)`, target-aware claim 없음 |
| AR-041 | RL-Q | v1 target-aware ranker — target proposal + local/relation feature 구현 | 🟠 | AR-037 후속. action=`(tool,target,u)`, g2_stress oracle gap 회수 목적 |
| AR-042 | RL-Q | v2 support-aware ranker — data support/uncertainty state + OOD ablation | 🟡 | AR-037 후속. kNN/ensemble 기반 support, offline value overestimation 억제 |
| AR-025 | Evidence | H-2026-204 (artifact-conditioned + closed-loop > fixed post-proc) 5단계 평가 | 🟡 | B2-family vs M0+gate, snapshot≥2 |
| AR-026 | Evidence | H-2026-203 (high-quality generator no-harm) — **진짜 G1 필요** (clean=GT 는 미충족) | ⚪ | AR-022 의존 |
| AR-027 | RL-Q | Stage 4 constrained offline RL (CQL/IQL) — reranking 넘어 policy optimization | ⚪ | Stage 3 안정 후 |
| AR-028 | RL-Q | sequence oracle (multi-step ceiling) vs step-1 oracle gap 측정 | ⚪ | 현재 oracle = step-1 dense |
| AR-032b | Harness | (선택) gitignored regenerable 디스크 정리 (final_motions 96MB 등) | ⚪ | git 추적 밖. 재현 시 재생성 필요 → 보류, 디스크 압박 시만 |

---

운영: 우선순위 오르면 [ready.md](ready.md) 로 승격. 폐기 시 [cancelled.md](cancelled.md).
