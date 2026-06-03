# `.claude/Dashboard/` — 작업 board (index)

> Jira / Plane 스타일 작업 board. **상태별로 파일 분리** (각 state = 한 파일). 본 README 가 **index** (운영 규약 + state 링크 + epic + 결론 overview). 상세 evidence 는 `reports/` (일지) + `evals/` + git commit 이 single source — board 는 그 위의 index 일 뿐 중복 서술하지 않는다.

## 상태별 파일 (Plane 컬럼)

| State | 파일 | 현황 |
|---|---|---|
| 🗂️ Backlog | [backlog.md](backlog.md) | 미착수·미래 |
| 📋 Todo | [todo.md](todo.md) | 착수 준비 (ready) |
| 🟢 In Progress | [in-progress.md](in-progress.md) | 진행 중 (한 번에 1개) |
| ✅ Done | [done.md](done.md) | 완료 (commit 링크 박제) |
| 🚫 Cancelled | [cancelled.md](cancelled.md) | 폐기 (사유 박제) |

흐름: `Backlog → Todo → In Progress → Done / Cancelled`.

## 사용 규약

- **Issue ID**: `AR-NNN` (append-only 증가, 닫혀도 재사용 금지). **Priority**: 🔴 Urgent / 🟠 High / 🟡 Medium / ⚪ Low. **Epic**: 큰 묶음 (아래).
- **작업 시작**: [todo.md](todo.md) 항목을 [in-progress.md](in-progress.md) 로 이동(pull, 한 번에 1개 권장).
- **작업 종료(commit)**: [in-progress.md](in-progress.md) → [done.md](done.md) + commit 해시/링크 박제. 새 후속 작업은 [backlog.md](backlog.md).
- 자동 reminder: [.claude/settings.json](../settings.json) SessionStart(시작) / PostToolUse git commit(종료) 훅.
- 본 board ≠ 가설 레지스트리 ([`evals/hypotheses/`](../../evals/hypotheses/)) — 가설 status 전환은 사용자 승인 게이트 (AGENTS.md §3-11).

## 🧭 Epics

| Epic | 의미 | 상태 |
|---|---|---|
| **Framing** | Safe Orchestration 정식화·governance | 안정 |
| **Gate** | Physical Constraint Gate (accept/repair/rollback/STOP) | 안정 |
| **RL-F** | RL-2 safe imitation (discrete, F-1~F-5) | 완료 → RL-Q |
| **RL-Q** | bounded continuous Q_safe(s,tool,u) (Stage 1/A/A+/B) | active |
| **G2-stress** | G2 real stress 중심 pipeline (Stage 2-G) | active (핵심) |
| **Evidence** | 통계·재현·결과·가설 평가 | active |
| **Quality** | standard metric (FID/R-Prec) + perceptual | active |
| **Generator** | generator-agnostic (G1/MDM transfer) | 미착수 |
| **Harness** | governance/tooling 유지보수 | 상시 |

## 📌 현재 연구 결론 상태 (한눈에)

- **공식 status 전환 가설 = 0건** (203/204/205/206 모두 `active`). 200/201/202 = 행정적 supersede.
- **Evidence 강함 (미전환)**: H-205 (learnable routing > rule-based) — snapshot 2개 12/12 재현, closed-loop 0% violation + final quality 보존/개선.
- **미검증**: generator-agnostic (G1 미구축, AR-022), high-quality no-harm (진짜 G1 필요, AR-026), perceptual b2/b3 (AR-023).
- 상세: [results_draft_g2_stage2.md §7](../../docs/results_draft_g2_stage2.md) · [가설 요약](../docs/hypotheses-summary.md) · [용어 사전](../docs/glossary.md).
