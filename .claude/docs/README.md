# `.claude/docs/` — Agent 참조 문서 (reference)

> **docs = Agent 가 참조하는 문서** (skills = Agent 가 실행하는 절차·명세). 본 폴더는 실행 절차가 아니라, Agent 가 판단·서술 시 인용하는 **reference material** 을 둔다.

## 본 폴더의 문서

| 문서 | 용도 |
|---|---|
| [glossary.md](glossary.md) | 본 프로젝트 전용 용어·약어 사전 (G1/G2, NetGain, KDG, Q_safe, u-grid 등). 첫 등장 용어 풀이의 단일 출처. |
| [hypotheses-summary.md](hypotheses-summary.md) | 연구 가설 H-2026-200~206 의 한곳 통합 요약 (canonical 은 [`evals/hypotheses/`](../../evals/hypotheses/) append-only registry). |

## docs vs skills vs Dashboard (역할 분리)

| 위치 | 역할 | 예 |
|---|---|---|
| `.claude/skills/` | **Agent 가 실행** 하는 구체적 절차·checklist·명세 | eval-compare, hypothesis-registry, intent-reconciliation |
| `.claude/docs/` | **Agent 가 참조** 하는 문서 (용어·요약·배경) | glossary, hypotheses-summary |
| `.claude/Dashboard/` | **작업 흐름·상태** 추적 (board) | PROJECT_BOARD.md |
| `docs/` (repo root) | 연구 provenance·framing 단일 출처 (metric/action_space/research position) | metric_provenance.md, action_space_provenance.md |

본 분리 기준은 [AGENTS.md §3-24](../../AGENTS.md) (Harness Rule vs Skill 분리) 와 일관.
