# AGENTS.md — motion-artifact-router (ArtifactRouter)

> **본 파일 = 규칙 원본 (WHAT — 무엇이 규칙인가).** 위반 시 결과 invalidation 되는 **절대 규칙·게이트(invariant)** 만 둔다 (~200줄). Agent 의 **Role·페르소나·가치 우선순위·의사결정 권한·커뮤니케이션 스타일 (WHO/HOW — 누가 어떻게 적용하는가)** 는 본 문서가 아니라 [`.claude/rules/phase/01-instructions.md`](.claude/rules/phase/01-instructions.md) 가 owner (본 문서는 Role 을 정의하지 않는다; 둘이 합쳐 하네스 4계층의 "01 지침 레이어").
>
> 실행 절차 = `.claude/skills/`, 프로젝트 참조 문서 = `.claude/docs/`, 범용 템플릿 = `docs/harness-research-template/`, 작업 board = `.claude/Dashboard/` (분배 기준 §3-24, [phase 01 §7-1](.claude/rules/phase/01-instructions.md)). 템플릿 상속: [`docs/harness-research-template/01-instructions.md §2`](docs/harness-research-template/01-instructions.md).

## 목차
[1 컨텍스트](#1-시스템-컨텍스트) · [2 빌드&실행](#2-빌드--실행) · [3 절대 규칙](#3-절대-규칙) · [4 경로별 분기](#4-경로별-조건-분기) · [5 실패 대응](#5-실패-대응) · [6 위험 행동](#6-위험-행동) · [7 디렉토리](#7-디렉토리별-규칙) · [8 참조](#8-참조-전용)

---

## 1. 시스템 컨텍스트

**ArtifactRouter** = 외부 motion generator (MotionGPT·MDM·MLD 등) output 위에서 **artifact state → correction action 매핑**을 routing 문제로 정식화한 generator-agnostic, tool-extensible decision system. 새 generator·새 단일 calibrator 개발 안 함. cost·risk (NetGain) 고려 + STOP(abstain) 포함. 정식 framing = **Safe Orchestration** (NOT NetGain-only): RL objective = `maximize artifact_improvement s.t. physical_validity + no_harm` ([.claude/docs/governance/current_research_position.md §0](.claude/docs/governance/current_research_position.md)).

가치 우선순위 (상충 시 좌선): 연구 정직성 > 재현성 > 비교 가능성 > 효율성 > 편의성. **Role·정의·근거 = [01-instructions §2-2](.claude/rules/phase/01-instructions.md) owner** (본 문서는 규칙만, 가치관 정의 안 함).

**핵심 가설** (canonical [`evals/hypotheses/`](evals/hypotheses/) append-only, 요약 [.claude/docs/reference/hypotheses-summary.md](.claude/docs/reference/hypotheses-summary.md)): [H-2026-207](evals/hypotheses/H-2026-207.md) (RQ1+2 재작성 — 기전-겨냥+조건부 refinement, Cat-A+지각 통화; post-hoc registration) · [H-2026-205](evals/hypotheses/H-2026-205.md) (RQ3 learnable routing — inconclusive) · [H-2026-206](evals/hypotheses/H-2026-206.md) (RQ4 generator-agnostic — 미착수) · [H-2026-203](evals/hypotheses/H-2026-203.md) (secondary no-harm — 전제 반전 note). 종결: H-200/201/202 (2026-05-15) · H-204 (2026-07-13 supersede→H-207). **가설 본문 수정·status 전환 = §3-11 사용자 승인 게이트.**

**기술 스택**: Python 3.10, conda env `motion-router` (메인) + `mgpt` (G2 inference). torch≥2.4 / transformers≥5.7 / scikit-learn 등. 외부 generator: **G2** (공식 MotionGPT, active) + **G1** (MDM/MLD, 미구축). 설치 상세 [`.claude/docs/operations/setup.md`](.claude/docs/operations/setup.md).

**소스 디렉토리** ([§7](#7-디렉토리별-규칙)): `generators/` `skeleton_normalizer/` `evaluators/` `correction_tools/` `orchestrator/` `refinement_loop/` `tools/` `evals/` `reports/` `docs/` `.claude/` `external_assets/`. 용어는 [.claude/docs/reference/glossary.md](.claude/docs/reference/glossary.md) 단일 출처.

본 저장소 = **독립 프로젝트** (이전 저장소 후속 아님). import 자산 = public dataset + 시각화 utility 한정.

---

## 2. 빌드 & 실행

- **환경**: 두 conda env (`motion-router` 메인 + `mgpt` G2). 설치·데이터 자산 확보 상세 → [`.claude/docs/operations/setup.md`](.claude/docs/operations/setup.md).
- **실행** (예): `python -m generators.motiongpt_wrapper --prompt "walking" --n-frames 40` · `python -m refinement_loop.loop --generator G2 --max-iterations 5`. 도구별 CLI 는 각 tool docstring.
- **센서** (코드·tool·평가 변경 시): `python -m py_compile <file>` · `pytest tests/unit/ -v` · `pytest tests/integration/ -v` · skeleton round-trip. 상세 [`02-sensor.md`](.claude/rules/phase/02-sensor.md).
- **평가(L4)**: 관측 전용 (머지 비차단). 회귀 항목은 [`eval-compare SKILL §6`](.claude/skills/eval-compare/SKILL.md) 5단계 리포트로 본 문서/phase 갱신.

---

## 3. 절대 규칙

> **IMPORTANT:** 본 절 위반 = (a) 비교 가능성 파괴 (b) 재현 불가 (c) 가설 평가 오염 (d) loop 안정성 손상 중 하나. 예외 없이 준수. 절대 규칙 추가는 메트릭 근거(반복 회귀) 또는 사용자 directive 시에만 (§3-24).
>
> **위반 effect 규약**: 각 규칙 = invariant(1줄) + 상세 링크. effect 는 — (i) `위반 = §6-5` 명시 규칙 = silent invalidation (결과 무효), (ii) 미명시 = 위 공통 (a)~(d) + [§6 위험 행동](#6-위험-행동)의 대응 prohibition. **§6 은 §3 obligation 의 금지형(prohibition) view — 둘은 같은 규칙의 양면**이며 §6 단독은 새 규칙이 아니다.
>
> **규칙 cluster** (방향 변경 시 주로 **C** 만 손봄; A/B 는 framework·정직성 invariant 로 안정):
>
> | cluster | 성격 | 규칙 |
> |---|---|---|
> | **A. Framework invariant** | 방향 무관 안정 (format·interface·loop) | §3-1 · §3-2 · §3-3 · §3-4 · §3-5 |
> | **B. Research-integrity gate** | 연구 정직성·평가 무결성 | §3-7 · §3-8 · §3-9 · §3-11 · §3-12 · §3-13 · §3-14 · §3-16 · §3-17 · §3-18 · §3-22 · §3-23 |
> | **C. Provenance & metadata/protocol** | 기록·provenance·실험 protocol (가변) | §3-6(+6-1) · §3-10 · §3-15 · §3-19 · §3-20 · §3-21 · §3-24 · §3-25 · §3-26 |

- **§3-1 Canonical Motion Format**: 모든 motion = canonical SMPL 22-joint `[T,22,3]`, fps=20, root-relative (PELVIS=origin). joint 순서 단일 출처 [`canonical_smpl_22.py`](skeleton_normalizer/canonical_smpl_22.py).
- **§3-2 Tool/Evaluator 인터페이스**: CorrectionTool `apply(motion,target_part,target_joints,frame_range,strength,metadata)` + Evaluator output schema (`agent·error_type·body_part·frames·score·severity·recommendation`) 준수. 변경 시 [`base.py`](correction_tools/base.py)+모든 구현체 동시 갱신.
- **§3-3 KDG Ordering**: tool 선택·ordering 은 [`orchestrator/kdg.py`](orchestrator/kdg.py) Kinematic Dependency Graph 위반 금지 (root/상위 node 먼저; 동 depth 는 hard-constraint→soft; ancestor-descendant joint 동 step 병렬 금지). 변경 = §3-11 게이트.
- **§3-4 Closed-loop Score 비감소**: [`convergence.py`](refinement_loop/convergence.py) 종합 Score step별 non-decreasing. 악화 호출 reject/rollback. same `(tool,target)` 재호출은 strength 감소만 (oscillation 방지).
- **§3-5 Generator Quality-tier 분리**: G1/G2 결과는 디렉토리·파일명·`generator_id` 로 분리. 한 generator 결과를 다른 generator 평가기 입력 금지. active scope = G1+G2 ; G3 (`external_assets/local_lora_g3/`) import·실행 금지.
- **§3-6 평가 기록 의무**: 추론·평가 시 동시 기록 — generator id+hash+prompt / evaluator·tool config hash / model_card hash / artifact metric / NetGain / FidelityLoss(Protocol A/B/C) / efficiency / tool call trace. **policy/risk head/gate/heuristic/oracle 가 selection 에 관여 시 추가 field → [policy-validation-traceability](.claude/docs/governance/policy-validation-traceability.md) (§3-25)**. 누락 record 는 비교 근거 인용 금지. 기록 = raw record (`evals/raw/`) + 일지 (`reports/<date>.md`) 동시.
- **§3-6-1 연구일지 의무**: 핵심 경로 변경 또는 generator inference·평가 1회+ 실행 시 `reports/<YYYY-MM-DD>.md` 작성 (정량지표·시각화·실험메타·raw cross-link 4항목). 상세 [research-journal SKILL](.claude/skills/research-journal/SKILL.md).
- **§3-7 자가 수정 메타 규칙**: 동일 센서 실패 3회+ 또는 동일 회귀 2 스냅샷+ 시 재발 방지 규칙을 §3 또는 phase 에 추가. **메트릭 근거 없이 절대 규칙 추가 금지.**
- **§3-8 검증 질의 ≠ 구현 지시**: 사용자 검증 질의("맞아?","왜?")는 답변만. 구현 지시로 확장 금지.
- **§3-9 단일 sample/trial 결론 금지**: 단일 sample·trial 로 성능 결론 금지. split 전체 또는 다중 generator 분포 + paired test (trial≥20, snapshot≥2).
- **§3-10 데이터·모델 버전 관리**: 새 산출물은 [data-versioning SKILL](.claude/skills/data-versioning/SKILL.md) metadata 규약. metadata 불일치 상태 결과 산출 금지.
- **§3-11 가설 사전 등록·보수적 수정**: 가설 `evals/hypotheses/<h_id>.md` 사전 등록 + append-only. **status 전환·supersede promote = 사용자 승인 게이트, Agent 단독 금지** ([hypothesis-registry SKILL](.claude/skills/hypothesis-registry/SKILL.md)).
- **§3-12 외부 공개 재현성 체크리스트**: 논문·발표·README 전 [reproducibility-checklist §2](.claude/skills/reproducibility-checklist/SKILL.md) 14+2 항목. 지표는 동 §3 단일 출처. 자체 재정의 금지.
- **§3-13 Negative Result 보존**: 폐기·기각 시도는 일지 "실패한 시도" 절 + raw record `negative_result:true`.
- **§3-14 우회 기록 의무**: 정공법 실패 우회는 즉시 [workaround-tracking SKILL §4](.claude/skills/workaround-tracking/SKILL.md) `evals/workarounds/<W-id>.md` (append-only). `open`+`critical` 1개+ 시 외부 공개 보류.
- **§3-15 Raw record metadata**: 모든 raw record 에 `severity_versions` + `split_id` (calibration↔holdout silent leakage 차단) + `evaluator_config_hashes` 누락 없이. 상세 [eval-collect SKILL](.claude/skills/eval-collect/SKILL.md). 누락 record 인용 금지.
- **§3-16 Oracle type 명시**: oracle best-tool baseline 은 `oracle_type` field 로 **single-step** vs **sequence(=closed-loop, ≥single-step)** 명시. 두 type 혼합 인용 금지.
- **§3-17 Synthetic vs Real 분리**: 모든 결과를 3-tier 로 분류·인용 — **controlled diagnostic** (synthetic, 최종 성능 sole evidence 금지) / **real-distribution** (G1/G2 natural) / **quality-validated** (perceptual b1/b2/b3). 결론 절에 keyword 명시 의무. 상세 [current_research_position §0](.claude/docs/governance/current_research_position.md). 위반 = §6-5 silent invalidation.
- **§3-18 Baseline Family Protocol**: "B2" 단독 표기 금지 → "B2-medium"/"B2-family" (fixed smoothing diagnostic family). 성공 기준 = "fixed smoothing family 대비 우월". B5/B6/B7 도 family. 상세 [reproducibility-checklist §3](.claude/skills/reproducibility-checklist/SKILL.md).
- **§3-19 GIF/MP4 Axis Convention**: motion = Y-up (HEAD_y>PELVIS_y>FOOT_y). GIF/MP4/3D PNG 작성·수정 시 `ax.view_init(vertical_axis="y")` + 첫 frame inspection 의무. 상세 [02-sensor §1-2](.claude/rules/phase/02-sensor.md). 2026-05-25 이전 GIF = axis bug, `*_yup_fix/` 만 정식.
- **§3-20 Metric Citation Gate**: 모든 metric 은 [metric_provenance.md](.claude/docs/governance/metric_provenance.md) 등록 + Category **A** (standard, 외부 근거 가능) / **B** (variant, 명시 의무) / **C** (proxy, 외부 최종 성능 근거 금지). NetGain=C. 외부 근거 = A (FID/R-Prec/MM-Dist) + perceptual 동반. 위반 = §6-5.
- **§3-21 Action Space Provenance**: 모든 RL/Q-surface stage 의 action space 는 [action_space_provenance.md](.claude/docs/governance/action_space_provenance.md) 등록. 리포트·외부 공개 시 `action_space_type` (`discrete_3level`/`discrete_5level`/`dense_grid_proxy`/`bounded_continuous_u`) + stage + STOP 포함 + u-mapper version 명시. 다른 grid 직접 비교 시 같은 sample/reference/config. **RL-2 historical: learned primary=3-level; 5-level=oracle ceiling** ; continuous-u 는 별도 stage 로 기록. 위반 = §6-5.
- **§3-22 Research Grounding Gate**: 연구 설계 피드백/평가 해석/metric·baseline·algorithm 선택/외부 공개 판단 시 **2020+ peer-reviewed top-tier 논문 근거** 동반 (arXiv-only 단독 금지). 응답 4항목 (판단/근거논문/적용범위/불확실성). 근거 부족 시 `engineering heuristic`/`internal proxy assumption`/`pilot-only finding` 명시 (외부 단독 근거 금지). 상세 [04-evaluation §7-0](.claude/rules/phase/04-evaluation.md). 단순 버그·경로·테스트는 대상 아님.
- **§3-23 Intent-Reconciliation Loop**: 새 evaluator/tool/oracle/baseline/policy/snapshot/framework doc commit 직전 5-step self-check → verdict `aligned`/`partial`/`misaligned`/`scope_creep`/`unintended_side_effect`. 뒤 3개는 사용자 보고 (commit 보류/confirm). 상세 [intent-reconciliation SKILL](.claude/skills/intent-reconciliation/SKILL.md).
- **§3-24 Harness Rule vs Skill/Doc 분리**: AGENTS=invariant·필수 metadata field·evidence tier·silent invalidation·승인 게이트. **skills**=실행 절차/checklist/예시. **docs**=참조(용어·요약·spec). **Dashboard**=board. Dashboard row 는 1줄 index 로 유지하고 상세 작업 명세는 `.claude/docs/dashboard-task-specs/<dashboard-id>-*.md` 로 분리한다. 예정사항·다음 순서·우선순위는 Dashboard 에 등록된 row 기준으로만 제시하며, Dashboard 에 없는 작업은 먼저 backlog 에 등록한다. AGENTS 는 아키텍처 역할을 과도하게 고정하지 않고 **configuration 과 claim 일치**를 강제. `heuristic`/`proxy`/`pilot` 근거는 절대 규칙 직접 승격 금지 → skills/docs 먼저 ([phase 01 §7-1](.claude/rules/phase/01-instructions.md)).
- **§3-25 Policy-Validation Traceability**: learned policy/Q/risk head/heuristic/oracle/gate 결과 인용 시 **무엇이 품질 향상에 기여했는지 분리 가능**해야 함. `selection_mode`·`candidate_trace`·`gate_recheck`·`policy_contribution_baseline` 기록 ; `gate_recheck=false`=diagnostic only ; baseline 비교 없으면 `policy contribution not isolated`. field·claim rule 상세 [policy-validation-traceability](.claude/docs/governance/policy-validation-traceability.md). 위반 = §6-5.
- **§3-26 Action-Effect Coverage / Hard-Example Provenance**: continuous-u/Q-surface 단위 = `(state,tool,u,after_state,gate_result,utility)` transition. transition/hard-mining dataset 은 `transition_dataset_id`·`u_grid`·`seed`·`mining_reason` 등 기록 + hard-mined 는 natural 과 분리 보고. 상세 [policy-validation-traceability §3](.claude/docs/governance/policy-validation-traceability.md).

---

## 4. 경로별 조건 분기

`<변경> → <대응 검증>`:

| 변경 경로 | 대응 |
|---|---|
| `skeleton_normalizer/` (canonical/root-relative) | G1·G2 output round-trip 재검증 + artifact coverage 재산출 |
| `evaluators/<name>_evaluator.py` | historical raw record 재계산 가능성 확인 ; 비호환 시 `aggregation_rule_version` ↑ + SEVERITY_VERSION bump + reproducibility-checklist §3 갱신 |
| `correction_tools/<name>_tool.py` | [`base.py`](correction_tools/base.py)+모든 구현체+orchestrator scoring 동시 갱신 + integration smoke |
| `orchestrator/kdg.py`·`scoring.py` | **§3-11 사용자 승인 게이트** + 변경 사유·baseline 커밋 기록 |
| `refinement_loop/loop.py`·`convergence.py` | max_iter/종료 조건 변경 사유 + 기존 trial reanalysis |
| `generators/<name>_wrapper.py` | [`base.py`](generators/base.py) interface 준수 + canonical round-trip |
| `requirements.txt` | `motion-router` 에서 install 선행 + 결정성 영향 확인 (torch/transformers 등) |
| `external_assets/**` | freeze 취급, 직접 수정 금지 (변경 시 사용자 승인) |

---

## 5. 실패 대응

| 시그니처 | 대응 |
|---|---|
| `external_assets/` 손실 | [`.claude/docs/operations/setup.md §2`](.claude/docs/operations/setup.md) 옵션 A(재다운로드)/B(robocopy) 복구 |
| generator 비결정성 | sampling/cuda non-determinism → N≥3 평균 또는 `torch.use_deterministic_algorithms(True)` |
| Tool conflict | KDG ConflictScore threshold 초과 reject → `A(t)` 매핑·tool 분류 점검 |
| loop oscillation | same `(tool,target)` 반복 → `convergence.py` `same_pair_strength_decay` 점검 |

연구 차원 리스크 (artifact 정의 도메인 의존성 / tool effect generator dependency / no-harm 보수성) → [reproducibility-checklist §3](.claude/skills/reproducibility-checklist/SKILL.md) 지표 사전 + phase 02-sensor §5.

---

## 6. 위험 행동 (금지)

> 본 절 = §3 절대 규칙의 **금지형(prohibition) view** (둘은 같은 규칙의 양면, §3 도입부 effect 규약). "근거" 열이 §3-N 인 항목은 그 §3 obligation 의 위반에 해당.

| # | 금지 | 근거 |
|---|---|---|
| 6-1 | 거대 산출물 (generator output·LoRA·dataset GB) git 커밋 | `.gitignore` 등재 |
| 6-2 | 기존 `reports/<날짜>.md` 수치 덮어쓰기 | 새 파일·행 추가 |
| 6-3 | 임계값(NetGain·artifact·fidelity) 완화로 회귀 회피 | 원인 수정 또는 §3 강화로만 |
| 6-4 | 가설 사후 수정 (HARKing, Kerr 1998) | append-only ; 새 draft→승인→새 H-id |
| 6-5 | metadata 우회로 산출물 동질화 (silent invalidation) | §3-10/15 위반 |
| 6-6 | 우회 미기록·등급 임의 하향 | §3-14 |
| 6-7 | 단일 trial 가설 평가 | §3-9 (trial≥20·snapshot≥2) |
| 6-8 | KDG ordering bypass | §3-3 (conflict_risk=∞) |
| 6-9 | Tool conflict 무시 강행 | §3-3 |
| 6-10 | generator transfer 단일 generator 일반화 | §6-10: 최소 2 generator (G1↔G2) + paired test |
| 6-11 | provisional NetGain weight tagless 인용 | `netgain_weight_status` 명시 ; provisional 은 calibrated 처럼 외부 인용 금지 |
| 6-12 | cross-evaluator side effect 미기록 | tool effect matrix 에 모든 evaluator before/after (`cross_evaluator_effects`) |
| 6-13 | integration smoke 의 가설 근거 인용 | 5단계 리포트(trial≥20/paired)에서만 |
| 6-14 | learned safety (P_safe/Q_safe/risk) gate-free 인용 | `diagnostic_no_gate` + false-safe rate ([§3-25 doc](.claude/docs/governance/policy-validation-traceability.md)) |
| 6-15 | action space evidence 혼합 인용 | `action_space_type`/`u_grid`/mapper version 분리 (§3-21) |
| 6-16 | gate-only improvement 를 policy contribution 오인 | baseline 비교 없으면 `policy contribution not isolated` |

---

## 7. 디렉토리별 규칙

| 디렉토리 | 책임 (단일 출처) |
|---|---|
| `generators/` | 외부 generator wrapper (`base.py` Generator interface). 새 generator 구현 안 함 |
| `skeleton_normalizer/` | output → canonical SMPL 22 (`canonical_smpl_22.py`). retargeting 직접 구현 안 함 |
| `evaluators/` | artifact + physical gate evaluator (`base.py` schema 일관) |
| `correction_tools/` | `CorrectionTool.apply(...)` ; correction magnitude·modified joints report |
| `orchestrator/` | evaluator+history → decision. kdg/scoring/rule_based/supervised/contextual_bandit (동일 I/O) |
| `refinement_loop/` | closed-loop 종료·convergence ; tool call trace |
| `tools/` | 시각화·실험 도구 (synthetic injection / tool effect matrix / perceptual) |
| `evals/` | `raw/` (Collect) · `snapshots/` · `reports/<period>.md` (5단계) · `hypotheses/`·`workarounds/` (append-only) |
| `reports/` | 일자별 일지 + `figures/<date>/` |
| `.claude/` | `rules/phase/` (지침) · `skills/` (실행) · `docs/` (참조) · `Dashboard/` (board, 상태별 파일 분리: backlog/ready/in-progress/done/cancelled) — §3-24 |
| `external_assets/` | public dataset + vestigial archive. read-only, 직접 수정 금지 (§4) |
| `experiments/` | MVP feasibility (Week 1-4) segregated workspace |

---

## 8. 참조 전용

- **구현 레시피** (evaluator/tool/generator/가설/우회 추가, 네이밍·포맷): [.claude/docs/reference/implementation-recipes.md](.claude/docs/reference/implementation-recipes.md).
- **용어 사전**: [.claude/docs/reference/glossary.md](.claude/docs/reference/glossary.md). **가설 요약**: [.claude/docs/reference/hypotheses-summary.md](.claude/docs/reference/hypotheses-summary.md). **작업 board**: [.claude/Dashboard/](.claude/Dashboard/README.md) (상태별 파일).
- **연구 provenance** (단일 출처): [metric_provenance](.claude/docs/governance/metric_provenance.md) · [action_space_provenance](.claude/docs/governance/action_space_provenance.md) · [current_research_position](.claude/docs/governance/current_research_position.md) · [motion_research_strategy_summary](.claude/docs/research/motion_research_strategy_summary.md).
- **데이터셋 카드**: [.claude/docs/dataset/](.claude/docs/dataset/README.md) (G2=MotionGPT pool 등). **발견·시사점**: [.claude/docs/findings/](.claude/docs/findings/README.md). **generator 실패 유형**(문헌): [.claude/docs/generator/generator_failure_mode_survey.md](.claude/docs/generator/generator_failure_mode_survey.md).
- **phase 지침**: [`.claude/rules/phase/`](.claude/rules/phase/) 01~04.
