# 📋 Ready (선택됨)

> 착수 준비된 작업 (사용자 신호 대기). [index](README.md) · 인접 상태: [backlog](backlog.md) → 여기 → [in-progress](in-progress.md).

| ID | Epic | Title | Priority | Blocked by / Note |
|---|---|---|---|---|
| AR-058-3h | Perceptual | **A/B 선호 검정 — pack 완성(8028714, marker 제거 재렌더 302da51), 사용자 판정 대기** (20쌍, ~10분). [index](../../reports/figures/2026-07-08/ar058_3h_ab_preference_pack/index.md) 열고 [rater_sheet.csv](../../reports/figures/2026-07-08/ar058_3h_ab_preference_pack/rater_sheet.csv) 작성. ⚠️ answer_key 는 판정 전 열람 금지 | 🔴 | 사전 등록 기준: ≥15/20 지지 / ≤12/20 → §10 Go/Stop 소집 / 13~14 → 평가자 추가. [spec](../docs/dashboard-task-specs/AR-058-3h-ab-preference-perceptual-test.md) · [snapshot](../../evals/snapshots/ab_preference_pack_ar058_3h_v1.json) |
| AR-058-3f | Evidence | Foot-skate human feedback pilot execution — 사람이 보기에도 품질 저하인지 검증하고 정의 freeze | 🔴 | Pack 준비 완료 + skate-segment GIF 18개 추가(0c70293). 사용자/평가자 응답 대기 (3h 의 A/B 와 상보 — 3f=가시성 절대평가, 3h=보정 선호). [rater index](../../reports/figures/2026-07-02/ar058_3f_footskate_human_feedback_pack/rater_index.md), [spec](../docs/dashboard-task-specs/AR-058-3f-human-feedback-pilot-execution.md) |
| AR-058-5 | Evidence | P5 - artifact-aware state-conditioned routing 필요성 종합 | 🔴 | P1~P4 evidence chain을 도입부 주장/그림으로 통합. [spec](../docs/dashboard-task-specs/AR-058-5-p5-routing-necessity-synthesis.md) |
| AR-020 | Evidence | H-2026-205 status `active`→`supported` 검토 | 🟠 | **사용자 승인 게이트** (AGENTS.md §3-11). snapshot≥2 + 12/12 재현 요건 충족 ([evals/reports/2026-06-01_g2_learned_vs_baseline.md](../../evals/reports/2026-06-01_g2_learned_vs_baseline.md)) |
| AR-021 | RL-Q | g2_stress Stage 3 — continuous argmax 로 oracle headroom (d≈0.43) 회수 | 🟠 | FID 보존 조건. line search / Bayesian opt |
| AR-023 | Perceptual | perceptual b2/b3 (3명+ inter-rater) — quality-validated evidence | ⚪ | 현재 b1 (GIF) 만. 사용자 작업 |
| AR-024 | Evidence | 독립 pool 재현 (snapshot 3) — 새 generation pool (현재 v2/v3 = 동일 600 pool partition) | ⚪ | 더 강한 재현 |

---

운영: [backlog.md](backlog.md) 에서 우선순위 오른 항목을 본 파일로 승격. 착수 시 [in-progress.md](in-progress.md) 로 이동.
