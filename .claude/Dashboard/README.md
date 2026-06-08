# `.claude/Dashboard/` — 작업 board (index)

> Jira / Plane 스타일 작업 board. **상태별로 파일 분리** (각 state = 한 파일). 본 README 가 **index** (운영 규약 + state 링크 + epic + 결론 overview). 상세 evidence 는 `reports/` (일지) + `evals/` + git commit 이 single source — board 는 그 위의 index 일 뿐 중복 서술하지 않는다.

## 상태별 파일 (Plane 컬럼)

| State | 파일 | 현황 |
|---|---|---|
| 🗂️ Backlog | [backlog.md](backlog.md) | 미착수·미래 |
| 📋 Ready | [ready.md](ready.md) | 착수 준비 |
| 🟢 In Progress | [in-progress.md](in-progress.md) | 진행 중 (한 번에 1개) |
| ✅ Done | [done.md](done.md) | 완료 (commit 링크 박제) |
| 🚫 Cancelled | [cancelled.md](cancelled.md) | 폐기 (사유 박제) |

흐름: `Backlog → Ready → In Progress → Done / Cancelled`.

## 사용 규약

- **Issue ID**: 기본 권장 형식은 `AR-NNN` 이지만, 핵심은 **Dashboard 에 등록된 row** 다. 닫힌 ID/row 는 재사용 금지. **Priority**: 🔴 Urgent / 🟠 High / 🟡 Medium / ⚪ Low. **Epic**: 큰 묶음 (아래).
- **작업 시작**: [ready.md](ready.md) 항목을 [in-progress.md](in-progress.md) 로 이동(pull, 한 번에 1개 권장).
- **작업 종료(commit)**: [in-progress.md](in-progress.md) → [done.md](done.md) + commit 해시/링크 박제. 새 후속 작업은 [backlog.md](backlog.md).
- **Board row 는 1줄 index**: 작업의 자세한 배경·명세·실험 설계는 [`.claude/docs/dashboard-task-specs/`](../docs/dashboard-task-specs/) 의 `<dashboard-id>-*.md` 로 분리한다.
- **순서 제시 원칙**: 예정사항·다음 순서·우선순위는 반드시 본 Dashboard 에 등록된 row 기준으로 제시한다. Dashboard 에 없는 작업은 먼저 [backlog.md](backlog.md) 에 등록한 뒤 언급한다.
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
| **Generator** | multi-generator refinement core: MotionGPT + MDM + MoMask | active |
| **Harness** | governance/tooling 유지보수 | 상시 |

## 📌 현재 연구 결론 상태 (한눈에)

- **공식 status 전환 가설 = 0건** (203/204/205/206 모두 `active`). 200/201/202 = 행정적 supersede.
- **Evidence 강함 (미전환)**: H-205 (learnable routing > rule-based) — snapshot 2개 12/12 재현, closed-loop 0% violation + final quality 보존/개선.
- **미검증**: generator-agnostic (MotionGPT 외 MDM/MoMask 구축 진행, AR-022/AR-045), high-quality no-harm (진짜 G1 필요, AR-026), perceptual b2/b3 (AR-023).
- 상세: [results_draft_g2_stage2.md §7](../../docs/results_draft_g2_stage2.md) · [가설 요약](../docs/hypotheses-summary.md) · [용어 사전](../docs/glossary.md).
