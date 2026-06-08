# 📋 Ready (선택됨)

> 착수 준비된 작업 (사용자 신호 대기). [index](README.md) · 인접 상태: [backlog](backlog.md) → 여기 → [in-progress](in-progress.md).

| ID | Epic | Title | Priority | Blocked by / Note |
|---|---|---|---|---|
| AR-044 | Evidence | G2 real problem taxonomy + visual/tool headroom audit | 🟠 | AR-043 후속. RL/q_proxy 전 실제 MotionGPT 문제·시각 사례·tool headroom 확인. [spec](../docs/dashboard-task-specs/AR-044-g2-real-problem-taxonomy-headroom.md) |
| AR-045 | Generator | MoMask 확보 → masked-token generator stress pool 구축 | 🟠 | acquisition + n=50 smoke OK (prelim: FootFloating 48% > MotionGPT 10%). **큰 stress pool 확장은 AR-048(protocol correction) freeze 후**. CVPR 2024. [spec](../docs/dashboard-task-specs/AR-045-momask-generator-acquisition.md) |
| AR-020 | Evidence | H-2026-205 status `active`→`supported` 검토 | 🟠 | **사용자 승인 게이트** (AGENTS.md §3-11). snapshot≥2 + 12/12 재현 요건 충족 ([evals/reports/2026-06-01_g2_learned_vs_baseline.md](../../evals/reports/2026-06-01_g2_learned_vs_baseline.md)) |
| AR-021 | RL-Q | g2_stress Stage 3 — continuous argmax 로 oracle headroom (d≈0.43) 회수 | 🟠 | FID 보존 조건. line search / Bayesian opt |
| AR-023 | Perceptual | perceptual b2/b3 (3명+ inter-rater) — quality-validated evidence | ⚪ | 현재 b1 (GIF) 만. 사용자 작업 |
| AR-024 | Evidence | 독립 pool 재현 (snapshot 3) — 새 generation pool (현재 v2/v3 = 동일 600 pool partition) | ⚪ | 더 강한 재현 |

---

운영: [backlog.md](backlog.md) 에서 우선순위 오른 항목을 본 파일로 승격. 착수 시 [in-progress.md](in-progress.md) 로 이동.
