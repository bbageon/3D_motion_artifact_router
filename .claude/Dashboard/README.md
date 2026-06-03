# `.claude/Dashboard/` — 작업 흐름·상태 추적 (board)

> Dashboard = **작업 흐름 / 상태** 를 한눈에 보는 board. 연구 evidence 자체는 `reports/` (일지) + `evals/` (snapshot/report) + git commit 이 single source 이며, 본 폴더는 그 위의 **index/board** 다.

## 본 폴더

| 파일 | 용도 |
|---|---|
| [PROJECT_BOARD.md](PROJECT_BOARD.md) | Jira/Plane 스타일 board (Backlog → Todo → In Progress → Done / Cancelled). AR-NNN issue + Epic + priority. |

## 운영

- 작업 **시작** 시 board 를 읽고 착수 항목을 Todo→In Progress 로 pull.
- 작업 **종료**(commit) 시 In Progress→Done 이동 + commit 링크 박제.
- 자동 reminder: [.claude/settings.json](../settings.json) 의 SessionStart / PostToolUse(git commit) 훅.
