# `.claude/Dashboard/` — 작업 흐름·상태 추적 (board)

> Dashboard = **작업 흐름 / 상태** 를 한눈에 보는 board. 연구 evidence 자체는 `reports/` (일지) + `evals/` (snapshot/report) + git commit 이 single source 이며, 본 폴더는 그 위의 **index/board** 다.

## 본 폴더 (상태별 파일 분리)

| 파일 | 용도 |
|---|---|
| [PROJECT_BOARD.md](PROJECT_BOARD.md) | **index** — 운영 규약 + state 파일 링크 + epic + 결론 overview |
| [backlog.md](backlog.md) | 🗂️ 미착수·미래 |
| [todo.md](todo.md) | 📋 착수 준비 (ready) |
| [in-progress.md](in-progress.md) | 🟢 진행 중 (한 번에 1개) |
| [done.md](done.md) | ✅ 완료 (commit 링크 박제) |
| [cancelled.md](cancelled.md) | 🚫 폐기 (사유 박제) |

## 운영

- 작업 **시작**: `todo.md` 항목을 `in-progress.md` 로 이동(pull).
- 작업 **종료**(commit): `in-progress.md` → `done.md` 이동 + commit 링크 박제. 새 후속 작업은 `backlog.md`.
- 자동 reminder: [.claude/settings.json](../settings.json) 의 SessionStart(시작) / PostToolUse git commit(종료) 훅.
