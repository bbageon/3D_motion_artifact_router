# 🗂️ Backlog (미래)

> 미착수·미래 작업. **연구방향이 자주 바뀌므로** 방향 변경 시 재정렬. [index](README.md) · 다음 상태: [ready](ready.md).

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| AR-051 | Evidence | Cross-generator refinement effect validation — tool 적용 전후 Δ + Category-A 품질 보존 + 시각 사례 | 🟠 | **AR-050 이후 착수**(완료). AR-044 taxonomy의 실제 tool headroom을 검증하고 `q_proxy` 근거를 만든다. native + sensitivity(짧은 sample 포함/제외) 분리 보고. [spec](../docs/dashboard-task-specs/AR-051-cross-generator-refinement-effect.md) |
| AR-052 | Generator | (선택) MotionGPT length-conditioned generation 실험 — generate_conditional(lengths) 검증 + token trace(Part B/C) | ⚪ | AR-050 후속. service 수정+재생성 필요. 별도 protocol/version. 품질 부작용·target-match 측정. cross-gen은 sensitivity로 진행 가능하므로 enhancement |
| AR-040 | RL-Q | v0 global quality-aware ranker — `q_proxy` 정의 + v0 state vector + M1 dense effect oracle | 🟠 | **AR-051 이후 착수**. 실제 tool 전후 effect와 quality trade-off로 q_proxy 정의. action=`(tool,u)`, target-aware claim 없음 |
| AR-041 | RL-Q | v1 target-aware ranker — target proposal + local/relation feature 구현 | 🟠 | AR-037 후속. action=`(tool,target,u)`, g2_stress oracle gap 회수 목적 |
| AR-042 | RL-Q | v2 support-aware ranker — data support/uncertainty state + OOD ablation | 🟡 | AR-037 후속. kNN/ensemble 기반 support, offline value overestimation 억제 |
| AR-025 | Evidence | H-2026-204 (artifact-conditioned + closed-loop > fixed post-proc) 5단계 평가 | 🟡 | B2-family vs M0+gate, snapshot≥2 |
| AR-026 | Evidence | H-2026-203 (high-quality generator no-harm) — **진짜 G1 필요** (clean=GT 는 미충족) | ⚪ | AR-022 의존 |
| AR-027 | RL-Q | Stage 4 constrained offline RL (CQL/IQL) — reranking 넘어 policy optimization | ⚪ | Stage 3 안정 후 |
| AR-028 | RL-Q | sequence oracle (multi-step ceiling) vs step-1 oracle gap 측정 | ⚪ | 현재 oracle = step-1 dense |
| AR-032b | Harness | (선택) gitignored regenerable 디스크 정리 (final_motions 96MB 등) | ⚪ | git 추적 밖. 재현 시 재생성 필요 → 보류, 디스크 압박 시만 |

---

운영: 우선순위 오르면 [ready.md](ready.md) 로 승격. 폐기 시 [cancelled.md](cancelled.md).
