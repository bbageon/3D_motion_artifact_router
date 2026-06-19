# ✅ Done

> 완료 작업 (commit 링크 박제). 20+ 누적 시 하단 `## Archive` 로 압축 이동. [index](README.md).

| ID | Epic | Title | 완료 | Evidence |
|---|---|---|---|---|
| AR-054 | Evidence | Generator artifact 100-case library — representative pool에서 MotionGPT/MDM/MoMask별 artifact 예시 100개씩 수집·정리. 총 300개, generator별 unique prompt 100개, qualitative library claim boundary 명시 | 06-19 | de2efdd · [spec](../docs/dashboard-task-specs/AR-054-generator-artifact-100case-library.md) · [snapshot](../../evals/snapshots/generator_artifact_100case_library_v1.json) · [summary](../../reports/figures/2026-06-19/ar054_generator_artifact_100cases/summary.md) · [일지](../../reports/2026-06-19.md) |
| AR-053 | Evidence | Generator artifact visual pack — MotionGPT FootFloating, MDM world foot-skate, MoMask FootFloating 대표 사례 GIF 3개 생성. 정량 taxonomy의 정성 확인, 성능 claim 아님 | 06-19 | [spec](../docs/dashboard-task-specs/AR-053-generator-artifact-visual-pack.md) · [snapshot](../../evals/snapshots/generator_artifact_visual_pack_v1.json) · [visual](../../reports/figures/2026-06-19/ar053_generator_artifacts/manifest.json) · [일지](../../reports/2026-06-19.md) |
| AR-052 | RL-Q | v0 learned ranker train/eval — prompt-level split overlap 0. RF pooled R2=0.788, top1=0.489, top3=0.811, regret 0.116 vs random 0.928. 학습기는 의미 있지만 false-improve 17% + top1 미흡 → AR-041 target-aware 필요 | 06-19 | [spec](../docs/dashboard-task-specs/AR-052-v0-learned-ranker-train-eval.md) · [snapshot](../../evals/snapshots/effect_aware_v0_learned_ranker_v1.json) · [일지](../../reports/2026-06-19.md) |
| AR-040 | RL-Q | v0 global quality-aware ranker — `q_proxy_v0.1` 정의 + 2700 state / 27000 transition / 900 prompt M1 dense oracle. FootLock failure는 q_proxy에서 강한 음수 처리, v0 oracle은 smoothing 위주 선택. learned policy 성능 claim 없음 | 06-19 | [spec](../docs/dashboard-task-specs/AR-040-v0-quality-aware-ranker.md) · [snapshot](../../evals/snapshots/effect_aware_v0_oracle_v1.json) · [일지](../../reports/2026-06-19.md) |
| AR-051 | Evidence | Cross-generator refinement effect validation — representative-300 × 3 generator에서 fixed tool 적용 Δ + Category-A + 시각 pack. 결론: FootLock은 proxy 개선과 foot-skate 악화를 동시에 만들 수 있고, fixed tool만으로 품질 개선 보장 불가 → quality-aware ranker 필요 | 06-19 | [physical/proxy](../../evals/snapshots/representative_refinement_effect_v1.json) · [Category-A](../../evals/snapshots/standard_metric_representative_refinement_v1.json) · [visual](../../reports/figures/2026-06-19/ar051_refinement_visual_pack/manifest.json) · [일지](../../reports/2026-06-19.md) |
| AR-050 | Generator | MotionGPT length-control audit (Part A~D 완료) — exact 21.9%, 4-frame=seed-specific 희귀(011743 다른seed=188). 원인: generate_direct에 length 미전달→확률적 EOS. **Part C: length-conditioned(with_len)도 개선 안 됨(exact 1.7% vs 16.7%, negative)** → native primary + sensitivity 분리 protocol | 06-09 | 67c1f4d+3b1bd82 · [audit](../../evals/snapshots/motiongpt_length_audit_v1.json) · [compare](../../evals/snapshots/motiongpt_length_conditioned_compare_v1.json) · [일지](../../reports/2026-06-09.md) |
| AR-044 | Evidence | representative-300 3-generator 문제 taxonomy — MDM world foot-skate 0.0108(약 2x)·MotionGPT FootFloating 41%. **문제량 확인 완료, 실제 tool 효과/시각 검증은 AR-051로 분리** | 06-09 | 72f9bc1 · [snapshot](../../evals/snapshots/representative_pool_measure_v1.json) · [일지](../../reports/2026-06-09.md) |
| AR-022 | Generator | Representative-300 shared pool 생성·검증 (MDM) — 900 모션 fail 0, target-match 100%. freeze 8/8 PASS | 06-08 | ee10ee9 · [freeze](../../evals/snapshots/representative_pool_freeze_v1.json) |
| AR-045 | Generator | Representative-300 shared pool 생성·검증 (MoMask) — 900 모션 fail 0, target-match 100% | 06-08 | ee10ee9 · [freeze](../../evals/snapshots/representative_pool_freeze_v1.json) |
| AR-049 | Generator | Representative-300 prompt bank + a-priori complexity annotation — hardness 선별(길이편향 82.7%) 대신 representative 무작위 300(원분포, 180-199frame 46%); 4축 complexity는 annotation(raw/z+quartile)·hard-300은 challenge candidate 보존 | 06-08 | fa54f1c · [bank](../../evals/prompts/protocol_rep_test_300_seed20260608.json) · [spec](../docs/dashboard-task-specs/AR-049-representative-prompt-bank.md) |
| AR-048 | Generator | Generation Protocol Correction — 3 generator trajectory/local 이중표현 통일 + full-motion/mirror/문장 dedup + GT길이 target + multi-seed + ground≠minY + semantic floor 0.8. 450 모션, 30 선별, **7/7 PASS** | 06-08 | 8e40db1 · [freeze](../../evals/snapshots/protocol_freeze_v1.json) · [일지](../../reports/2026-06-08.md) |
| AR-047 | Infra | Generator GPU Docker 마이크로서비스화 (3 generator → FastAPI GPU 서비스 8001/8002/8003, cu128/sm120, checkpoint host read-only 마운트) | 06-07 | caf468b · 3 서비스 health ok ([check](../../tools/check_generator_availability.py)) |
| AR-046 | Evidence | MotionGPT 시사점 정리(F0~F10) + docs/{findings,dataset} 신설 + 대표(prevalence-weighted) 재집계 (FID NEUTRAL + no-harm 확정) | 06-03 | 4a53c37 · [findings](../../docs/findings/motiongpt_implications.md) |
| AR-029 | Quality | standard physical metric (foot_skate GMD/EDGE + accel MDM) 으로 real g2_stress 재측정 — accel 소폭개선(ground-indep) + float 개선 vs foot_skate 악화(trade-off) | 06-03 | 4a53c37 · [snapshot](../../evals/snapshots/physical_metric_g2_stress_v1.json) |
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
