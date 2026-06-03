# ArtifactRouter — Project Board

> Jira / Plane 스타일 작업흐름 보드. **연구방향이 자주 바뀌므로** 본 보드는 "지금 무엇이 backlog / 진행중 / 완료 / 기록(history)인가" 를 한눈에 본다. 상세 evidence 는 `reports/<date>.md` (일지) + `evals/` (snapshot/report) + git commit 이 single source — 본 보드는 그 위의 **index/board** 일 뿐 중복 서술하지 않는다.

## 사용 규약 (운영 방법)

- **State (Plane 스타일)**: `Backlog` (미착수·미래) → `Todo` (선택됨·ready) → `In Progress` (진행중) → `Done` (완료) / `Cancelled` (폐기).
- **Priority**: 🔴 Urgent / 🟠 High / 🟡 Medium / ⚪ Low.
- **Issue ID**: `AR-NNN` (ArtifactRouter, append-only 증가). 닫혀도 ID 재사용 금지.
- **Epic**: 큰 작업 묶음 (label). issue 는 한 epic 에 속함.
- **갱신 시점**: 매 작업 turn 종료 시 해당 issue 의 state 갱신 + 새 작업은 Backlog 에 issue 추가. Done 으로 옮길 때 commit/report 링크 박제.
- **History**: Done 이 누적되면 본 문서 하단 `## History (archive)` 로 옮겨 압축 (board 상단은 최근 상태만).
- 본 보드 ≠ 가설 레지스트리 ([`evals/hypotheses/`](evals/hypotheses/)) — 가설 status 전환은 사용자 승인 게이트 (AGENTS.md §3-11), 본 보드는 작업 추적용.

---

## 🟢 In Progress

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| _(없음)_ | | 다음 작업 선택 대기 (Todo 에서 pull) | | |

---

## 📋 Todo (선택됨, ready — 사용자 신호 대기)

| ID | Epic | Title | Priority | Blocked by / Note |
|---|---|---|---|---|
| AR-020 | Evidence | H-2026-205 status `active`→`supported` 검토 | 🟠 | **사용자 승인 게이트** (§3-11). snapshot≥2 + 12/12 재현 요건 충족 ([evals/reports/2026-06-01_g2_learned_vs_baseline.md](evals/reports/2026-06-01_g2_learned_vs_baseline.md)) |
| AR-021 | RL-Q | g2_stress Stage 3 — continuous argmax 로 oracle headroom (d≈0.43) 회수 | 🟠 | FID 보존 조건. line search / Bayesian opt |
| AR-022 | Generator | MDM/MLD (G1) 구축 → generator-agnostic transfer (H-2026-206) | 🟡 | heavy setup (clone+dep+ckpt+wrapper+pool). 사용자 결정: "G2 정리 후" |
| AR-023 | Perceptual | perceptual b2/b3 (3명+ inter-rater) — quality-validated evidence | ⚪ | 현재 b1 (GIF) 만. 사용자 작업 |
| AR-024 | Evidence | 독립 pool 재현 (snapshot 3) — 새 generation pool (현재 v2/v3 = 동일 600 pool partition) | ⚪ | 더 강한 재현 |

---

## 🗂️ Backlog (미래 — 방향 바뀌면 재정렬)

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| AR-025 | Evidence | H-2026-204 (artifact-conditioned + closed-loop > fixed post-proc) 5단계 평가 | 🟡 | B2-family vs M0+gate, snapshot≥2 |
| AR-026 | Evidence | H-2026-203 (high-quality generator no-harm) — **진짜 G1 필요** (clean=GT 는 미충족) | ⚪ | AR-022 의존 |
| AR-027 | RL-Q | Stage 4 constrained offline RL (CQL/IQL) — reranking 넘어 policy optimization | ⚪ | Stage 3 안정 후 |
| AR-028 | RL-Q | sequence oracle (multi-step ceiling) vs step-1 oracle gap 측정 | ⚪ | 현재 oracle = step-1 dense |
| AR-029 | Quality | metric 보강 — standard foot skating/sliding (FootFloating proxy 한계 부록 Z) | ⚪ | Category B 승격 |
| AR-032b | Harness | (선택) gitignored regenerable 디스크 정리 (final_motions 96MB 등) | ⚪ | git 추적 밖. 재현 시 재생성 필요 → 보류, 디스크 압박 시만 |

---

## ✅ Done (최근 — 누적 시 History 로 이동)

| ID | Epic | Title | 완료 | Evidence |
|---|---|---|---|---|
| AR-031 | Harness | 하네스 평가 + Plane 스타일 보드 신설 | 06-03 | 본 문서 |
| AR-032 | Harness | 무참조 throwaway 3개 삭제 (`_axis_check`/`_mpl_check`/`_inspect_b6_closed_loop_trace`) — evidence 인용 `_*` 는 유지 | 06-03 | (본 commit) |
| AR-019 | Evidence | 결과 섹션 초안 (4 result 통합 + claim 범위) | 06-01 | [docs/results_draft_g2_stage2.md](docs/results_draft_g2_stage2.md) · e055604 |
| AR-018 | Evidence | 재현 snapshot 2 (split v3) — **12/12 방향+유의성 재현** | 06-01 | [evals/snapshots/rl2_g2_paired_stats_v3.json](evals/snapshots/rl2_g2_paired_stats_v3.json) · e1a9746 |
| AR-017 | Evidence | G2 paired 통계 (M0 vs baselines, Wilcoxon+d+bootstrap) | 06-01 | [snapshot](evals/snapshots/rl2_g2_paired_stats_v1.json) · f2f8139 |
| AR-016 | G2-stress | Step 6 part 2 — standard metric (FID/R-Prec, Δ vs original) | 06-01 | [snapshot](evals/snapshots/standard_metric_closed_loop_v1.json) · d1a443a |
| AR-015 | G2-stress | Step 6 part 1 — closed-loop real-gate-ON ablation (M0 vs baselines) | 06-01 | [snapshot](evals/snapshots/rl2_closed_loop_ablation_stage2_v1.json) · 43fa24c |
| AR-014 | G2-stress | Step 3-5 — G2 transitions + M0/M1/M2/M3 + offline gen eval | 05-31 | [snapshot](evals/snapshots/rl2_offline_gen_eval_stage2_v1.json) · 433578c |
| AR-013 | G2-stress | Stage 2-G v2 — G2 pool 600 stratified + split v2 (stress in train) | 05-31 | [split](evals/splits/g2_real_stress_split_v2.json) · 7d49279 |
| AR-012 | G2-stress | Stage 2-G v1 — G2 real stress profile + split freeze (pivot) | 05-31 | c7f6cbc |
| AR-011 | RL-Q | Stage B-1 hard mining (synthetic boundary, 4160 transitions) | 05-31 | 720597e |
| AR-010 | RL-Q | Stage B-0 P_safe Bias Audit (root cause = argmax+leaky P_safe) | 05-30 | [reports/2026-05-30.md](reports/2026-05-30.md) · bf47ade |
| AR-009 | RL-Q | Stage A+ continuous argmax confirmation | 05-29 | 1bb3ce6 |
| AR-008 | RL-Q | Stage A action-effect transition dataset (dense u-grid) | 05-29 | 2a072e1 |
| AR-007 | RL-Q | RL-2 Stage 1 — bounded continuous Q_safe(s,tool,u) | 05-29 | 363a7cc |
| AR-006 | Quality | Step G-3/G-4 — synthetic severe FID + perceptual GIF | 05-29 | 975a134 · 8a5c727 |
| AR-005 | Quality | Step G-1/G-2 — FID + R-Precision (dual B2) | 05-28 | a2c6c0c · 8f4cc70 |
| AR-004 | RL-F | Step F-1~F-5 — RL-2 safe imitation (16-action) + violation decomp | 05-27/28 | d2671c3 외 |
| AR-003 | Gate | Step E — Safe Sequence Oracle (gate-aware DFS) | 05-26/27 | d2f6163 외 |
| AR-002 | Gate | PhysicalGateV0 (5 evaluator, regression-based threshold) | 05-26 | — |
| AR-001 | Framing | Safe Orchestration framing 고정 (NOT NetGain-only) | 05-26 | [docs/current_research_position.md](docs/current_research_position.md) |

---

## 🧭 Epics (label 정의)

| Epic | 의미 | 상태 |
|---|---|---|
| **Framing** | Safe Orchestration 정식화·governance | 안정 |
| **Gate** | Physical Constraint Gate (accept/repair/rollback/STOP) | 안정 |
| **RL-F** | RL-2 safe imitation (discrete, F-1~F-5) | 완료 → RL-Q 로 전환 |
| **RL-Q** | bounded continuous Q_safe(s,tool,u) (Stage 1/A/A+/B) | active |
| **G2-stress** | G2 real stress 중심 pipeline (Stage 2-G) | active (핵심) |
| **Evidence** | 통계·재현·결과·가설 평가 | active |
| **Quality** | standard metric (FID/R-Prec) + perceptual | active |
| **Generator** | generator-agnostic (G1/MDM transfer) | 미착수 |
| **Harness** | governance/tooling 유지보수 | 상시 |

---

## 📌 현재 연구 결론 상태 (한눈에)

- **공식 status 전환된 가설: 0건** (203/204/205/206 모두 `active`). 200/201/202 = 행정적 supersede.
- **Evidence 강함 (status 미전환)**: H-205 방향 (learnable routing > rule-based) — snapshot 2개 12/12 재현. closed-loop 0% violation + final quality 보존/개선.
- **미검증**: generator-agnostic (G1 미구축, AR-022), high-quality no-harm (진짜 G1 필요, AR-026), perceptual b2/b3 (AR-023).
- 상세: [docs/results_draft_g2_stage2.md](docs/results_draft_g2_stage2.md) §7 claim scope.

---

## History (archive)

> Done 이 20+ 누적되면 오래된 항목을 본 절로 압축 이동. (현재 비어있음 — Done 이 위에 유지.)
