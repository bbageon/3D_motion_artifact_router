# ✅ Done

> 완료 작업 (commit 링크 박제). 20+ 누적 시 하단 `## Archive` 로 압축 이동. [index](README.md).

| ID | Epic | Title | 완료 | Evidence |
|---|---|---|---|---|
| AR-043 | Evidence | Dataset predefined issue prevalence audit — stress/natural 4.1x + evaluator 신뢰도 등급(reliable=FootFloating 1) + ar040_decision=GO+조건4 (5조건 충족) | 06-03 | b835735 |
| AR-037 | RL-Q | Effect-Aware Ranker state ablation 설계 — v0/v1/v2 schema 고정 (40/64/71) + before-action observable 검증 (금지 0 leak, v0 NaN 0) + dim 보고 | 06-03 | b715433 |
| AR-039 | Harness | §3 절대규칙 평가 → 저위험 정리 (effect 규약 명확화 #1 / §3↔§6 dual-view #2 / 3-cluster index A·B·C #4). 규칙 내용 불변 | 06-03 | 81e70a8 |
| AR-038 | Harness | AGENTS.md vs 01-instructions 경계 명확화 (WHAT=규칙 원본 / WHO·HOW=Role·적용) + 우선순위 중복 제거 | 06-03 | 5488bd0 |
| AR-036 | Harness | PROJECT_BOARD.md 삭제 → README.md 가 단일 index (상태별 파일과 중복 제거) + 참조 7곳 수정 | 06-03 | db1adcc |
| AR-035 | Harness | Dashboard 상태별 파일 분리 (backlog/ready/in-progress/done/cancelled + index) + 훅 갱신 | 06-03 | a569556 |
| AR-034 | Harness | AGENTS.md 619→146줄 + .claude/{docs,Dashboard} 분리 (skills/docs/Dashboard) | 06-03 | a1a9b6c |
| AR-033 | Harness | 보드 자동화 훅 (SessionStart=시작 / PostToolUse git commit=종료 reminder) | 06-03 | [.claude/settings.json](../settings.json) · 774be46 |
| AR-031 | Harness | 하네스 평가 + Plane 스타일 보드 신설 | 06-03 | 1bb3e57 |
| AR-032 | Harness | 무참조 throwaway 3개 삭제 (`_axis_check`/`_mpl_check`/`_inspect_b6_closed_loop_trace`) | 06-03 | 1bb3e57 |
| AR-019 | Evidence | 결과 섹션 초안 (4 result 통합 + claim 범위) | 06-01 | [docs/results_draft_g2_stage2.md](../../docs/results_draft_g2_stage2.md) · e055604 |
| AR-018 | Evidence | 재현 snapshot 2 (split v3) — **12/12 방향+유의성 재현** | 06-01 | [snapshot](../../evals/snapshots/rl2_g2_paired_stats_v3.json) · e1a9746 |
| AR-017 | Evidence | G2 paired 통계 (M0 vs baselines, Wilcoxon+d+bootstrap) | 06-01 | [snapshot](../../evals/snapshots/rl2_g2_paired_stats_v1.json) · f2f8139 |
| AR-016 | G2-stress | Step 6 part 2 — standard metric (FID/R-Prec, Δ vs original) | 06-01 | [snapshot](../../evals/snapshots/standard_metric_closed_loop_v1.json) · d1a443a |
| AR-015 | G2-stress | Step 6 part 1 — closed-loop real-gate-ON ablation (M0 vs baselines) | 06-01 | [snapshot](../../evals/snapshots/rl2_closed_loop_ablation_stage2_v1.json) · 43fa24c |
| AR-014 | G2-stress | Step 3-5 — G2 transitions + M0/M1/M2/M3 + offline gen eval | 05-31 | [snapshot](../../evals/snapshots/rl2_offline_gen_eval_stage2_v1.json) · 433578c |
| AR-013 | G2-stress | Stage 2-G v2 — G2 pool 600 stratified + split v2 (stress in train) | 05-31 | [split](../../evals/splits/g2_real_stress_split_v2.json) · 7d49279 |
| AR-012 | G2-stress | Stage 2-G v1 — G2 real stress profile + split freeze (pivot) | 05-31 | c7f6cbc |
| AR-011 | RL-Q | Stage B-1 hard mining (synthetic boundary, 4160 transitions) | 05-31 | 720597e |
| AR-010 | RL-Q | Stage B-0 P_safe Bias Audit (root cause = argmax+leaky P_safe) | 05-30 | [reports/2026-05-30.md](../../reports/2026-05-30.md) · bf47ade |
| AR-009 | RL-Q | Stage A+ continuous argmax confirmation | 05-29 | 1bb3ce6 |
| AR-008 | RL-Q | Stage A action-effect transition dataset (dense u-grid) | 05-29 | 2a072e1 |
| AR-007 | RL-Q | RL-2 Stage 1 — bounded continuous Q_safe(s,tool,u) | 05-29 | 363a7cc |
| AR-006 | Quality | Step G-3/G-4 — synthetic severe FID + perceptual GIF | 05-29 | 975a134 · 8a5c727 |
| AR-005 | Quality | Step G-1/G-2 — FID + R-Precision (dual B2) | 05-28 | a2c6c0c · 8f4cc70 |
| AR-004 | RL-F | Step F-1~F-5 — RL-2 safe imitation (16-action) + violation decomp | 05-27/28 | d2671c3 외 |
| AR-003 | Gate | Step E — Safe Sequence Oracle (gate-aware DFS) | 05-26/27 | d2f6163 외 |
| AR-002 | Gate | PhysicalGateV0 (5 evaluator, regression-based threshold) | 05-26 | — |
| AR-001 | Framing | Safe Orchestration framing 고정 (NOT NetGain-only) | 05-26 | [docs/current_research_position.md](../../docs/current_research_position.md) |

---

## Archive

> (현재 비어있음 — Done 20+ 누적 시 오래된 항목 압축 이동.)
