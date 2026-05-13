# AGENTS.md — motion-artifact-router (ArtifactRouter)

> 본 파일은 [`docs/harness-research-template/AGENTS.template.md`](docs/harness-research-template/AGENTS.template.md) 를 본 프로젝트 (ArtifactRouter — artifact-conditioned tool selection for generator-agnostic human motion refinement) 에 적용한 결과이다. 작성 원칙·해석 규약은 [`docs/harness-research-template/01-instructions.md §2`](docs/harness-research-template/01-instructions.md) 를 그대로 상속한다.

## 목차

- [1. 시스템 컨텍스트](#1-시스템-컨텍스트)
- [2. 빌드 & 실행](#2-빌드--실행)
- [3. 절대 규칙](#3-절대-규칙)
- [4. 경로별 조건 분기](#4-경로별-조건-분기)
- [5. 실패 대응 & 흔한 실수](#5-실패-대응--흔한-실수)
- [6. 위험 행동](#6-위험-행동)
- [7. 디렉토리별 상세 규칙](#7-디렉토리별-상세-규칙)
- [8. 참고 전용 (아래 섹션)](#8-참고-전용-아래-섹션)
- [9. 구현 레시피](#9-구현-레시피)
- [10. 아키텍처 제약 & 네이밍](#10-아키텍처-제약--네이밍)

---

## 1. 시스템 컨텍스트

본 프로젝트는 **ArtifactRouter** — 기존/미래 motion generator (MotionGPT·T2M-GPT·MDM·MLD 등) 의 출력 skeleton motion을 입력으로 받아, artifact 종류·body part·frame range 에 따라 적절한 evaluator 와 correction tool 을 선택·조합·재평가 하는 generator-agnostic, tool-extensible orchestration harness 다.

본 연구는 새로운 motion generator를 개발하지 않는다. 새로운 단일 correction algorithm 도 개발하지 않는다. **artifact 상태 → correction action 매핑을 학습 가능한 routing 문제로 정식화**해 fixed post-processing 또는 monolithic refinement 대비 net gain 우위를 정량 입증하는 것이 본 연구의 핵심 contribution.

연구의 우선순위는 (1) 연구 정직성 (가설 사전 등록·HARKing 차단·negative result 보존·우회 ledger 의무), (2) 재현성 (시드·환경·산출물 버전·tool call trace 추적), (3) 비교 가능성 (G1/G2/G3 generator quality-tier 분리·동일 metric 사전), (4) 효율성 (refinement stage FLOPs·tool call count·wall-clock), (5) 편의성 순이다.

핵심 가설은 다음과 같다 (상세는 [`evals/hypotheses/`](evals/hypotheses/) 에 사전 등록).

- [H-2026-200](evals/hypotheses/H-2026-200.md) — Artifact-conditioned tool selection 이 fixed post-processing 보다 효과적이며 closed-loop refinement 가 single-step 보다 artifact reduction 과 motion fidelity 의 trade-off 를 더 잘 관리한다.
- [H-2026-201](evals/hypotheses/H-2026-201.md) — Artifact state → correction action 매핑은 학습 가능한 routing 문제. supervised / contextual-bandit selector 가 rule-based baseline 대비 net gain 을 의미 있게 개선한다.
- [H-2026-202](evals/hypotheses/H-2026-202.md) — 학습된 selector 는 새로운 generator output 에 대해서도 generator-agnostic 일반화 가능 (zero-shot + small-calibration transfer).
- [H-2026-203](evals/hypotheses/H-2026-203.md) (secondary) — High-quality motion (SOTA generator output) 에 대해 No-harm 운영 특성 — 분포 (FID_motion / FGD) · semantic · fidelity 훼손 없음.

가설 본문 수정·status 전환은 §3-11 에 따라 [`.claude/skills/hypothesis-registry/SKILL.md §4`](.claude/skills/hypothesis-registry/SKILL.md) 사용자 승인 게이트를 거친다. AGENTS.md §1 핵심 가설은 본 등록 가설과 동기화되어야 한다.

기술 스택은 다음과 같다:

- 언어/런타임은 Python 3.10 (메인 환경 `motion-router`, conda) 이다.
- 핵심 라이브러리는 `torch>=2.4.0`, `transformers>=5.7.0`, `peft>=0.13.0`, `bitsandbytes>=0.43.0`, `numpy`, `matplotlib`, `Pillow`, `scipy`, `einops`, `networkx`, `scikit-learn` 이다 ([requirements.txt](requirements.txt)).
- 사용 generator (외부 의존, [`generators/`](generators/) 에 wrapper):
  - **G1** — high-quality SOTA generator (예: MDM, MLD — diffusion-based).
  - **G2** — token-based generator (예: MotionGPT, T2M-GPT).
  - **G3** — legacy/lightweight artifact-rich generator (본 프로젝트의 default — 이전 저장소의 `external_assets/local_lora_g3` LoRA-fine-tuned Gemma 4 E4B-it).
- 빌드 도구: `pyproject.toml` + pip. 별도 빌드 단계 없음.
- 정형 테스트 러너: **pytest**. round-trip + integration smoke + end-to-end refinement loop 검증.

본 저장소는 이전 저장소 [`3D-Motion-Trajectory-prediction`](../3D-Motion-Trajectory-prediction/) 의 sanity check 결과 위에 시작한 **독립 후속 프로젝트** 이며, 이전 저장소의 자산을 다음과 같이 import 한다:

- `external_assets/processed_noaug/` — HumanML3D root-relative + sliding window JSON (이전 저장소 directory junction).
- `external_assets/local_lora_g3/` — 이전 저장소의 학습된 LoRA adapter (G3 generator 로 사용).
- `external_assets/HumanML3D/` — 원본 HumanML3D 데이터셋 (GT 분포).
- `external_assets/code/` — `plot_3d_motion.py`, `stage1_mpjpe_and_gif.py`, `stage1_autoregressive_chain.py`, `make_prompt_3d.py`, `convert_3d_delta.py` (이전 저장소에서 복사).

소스 디렉토리 구조는 다음과 같다:

- [`generators/`](generators/) — Base Motion Generator wrapper (§6.1).
- [`skeleton_normalizer/`](skeleton_normalizer/) — Skeleton Normalizer (§6.1).
- [`evaluators/`](evaluators/) — Evaluator Tool Registry (§6.2).
- [`correction_tools/`](correction_tools/) — Correction Tool Registry (§6.3).
- [`orchestrator/`](orchestrator/) — Orchestrator + KDG + Scoring + rule-based / supervised / contextual-bandit (§6.4).
- [`refinement_loop/`](refinement_loop/) — Closed-loop refinement (§6.5).
- [`tools/`](tools/) — 시각화·실험 도구 (artifact injection, before/after GIF, tool effect matrix 등).
- [`evals/hypotheses/`](evals/hypotheses/) — 사전 등록 가설 (append-only).
- [`evals/raw/`](evals/raw/) — Collect 산출물.
- [`evals/workarounds/`](evals/workarounds/) — 우회 ledger.
- [`reports/`](reports/) — 사람-가독 연구일지.
- [`experiments/`](experiments/) — MVP feasibility study (Week 1-4, 명세 §12).
- [`docs/harness-research-template/`](docs/harness-research-template/) — 4계층 하네스 원본 (이전 저장소에서 복사).
- [`docs/motion_research_strategy_summary.md`](docs/motion_research_strategy_summary.md) — 연구 명세 (단일 출처).
- [`.claude/`](.claude/) — Claude Code 운영 자산 (phase + skill 적용본).
- [`external_assets/`](external_assets/) — 이전 저장소 자산 (junction + 복사).

프로파일/환경 구성:

- `motion-router` — 메인 환경 (conda, Python 3.10). 모든 컴포넌트 실행 + 평가.
- (필요 시) `torch_render` — HumanML3D 전처리용 (이전 저장소에서 상속). 본 저장소에서는 보통 미사용.

---

## 2. 빌드 & 실행

### 2-1. 환경 준비

```
conda create -n motion-router python=3.10 -y
conda activate motion-router
pip install -r requirements.txt
```

이전 저장소 자산 import (Windows directory junction, admin 권한 불필요):

```
cd external_assets
cmd /c "mklink /J processed_noaug ..\..\3D-Motion-Trajectory-prediction\processed_noaug"
cmd /c "mklink /J local_lora_g3 ..\..\3D-Motion-Trajectory-prediction\lora_3d_delta_h2026004_singlesample_002989_stage1_seed42"
cmd /c "mklink /J HumanML3D ..\..\3D-Motion-Trajectory-prediction\data\HumanML3D"
```

### 2-2. 핵심 실행 명령어

```
# G3 generator (이전 저장소 LoRA) inference
python -m generators.local_lora_wrapper --prompt "walking" --n-frames 40

# 외부 generator wrapper (MotionGPT 등) — wrapper 추가 후
python -m generators.motiongpt_wrapper --prompt "walking" --n-frames 40

# Skeleton normalization
python -m skeleton_normalizer.normalizer --input <motion.npy> --output <normalized.json>

# Evaluator 단위 실행
python -m evaluators.contact_evaluator --input <motion.json>

# Correction tool 단위 적용
python -m correction_tools.foot_lock_tool --input <motion.json> --target-frames 10:30

# End-to-end refinement loop
python -m refinement_loop.loop --generator G3 --prompt "walking" --max-iterations 5
```

### 2-3. 센서 명령어

코드·tool·평가 정의 변경 시 수행하는 센서 명령어 (상세 [`.claude/rules/phase/02-sensor.md`](.claude/rules/phase/02-sensor.md)).

```
# 정적 임포트·구문 검사
python -m py_compile <변경 파일.py>

# Unit 테스트 (component 단위)
pytest tests/unit/ -v

# Integration smoke (end-to-end refinement)
pytest tests/integration/ -v

# Skeleton Normalizer round-trip 검사
python -m tests.unit.skeleton_normalizer.test_roundtrip
```

### 2-4. 평가 레이어(L4) 실행

추론·평가 산출물 생성 후 평가 메트릭 기록 (상세 [`.claude/skills/eval-collect/SKILL.md`](.claude/skills/eval-collect/SKILL.md) — 작성 후).

본 평가는 머지를 차단하지 않는 관측 전용이며, 실패해도 커밋을 진행한다. 단 회귀 리포트 (`evals/reports/<period>.md`) 항목이 있으면 [`.claude/skills/eval-compare/SKILL.md`](.claude/skills/eval-compare/SKILL.md) 5단계 리포트 절차로 본 AGENTS.md 또는 phase 문서를 갱신한다.

---

## 3. 절대 규칙

> **IMPORTANT:** 본 섹션 항목은 위반 시 (a) 실험 결과 비교 가능성 파괴, (b) 학습/추론 재현 불가, (c) 가설 평가 근거 오염, (d) refinement loop 안정성 손상 중 하나를 유발한다. 예외 없이 준수하라.

### 3-1. Canonical Motion Format

모든 motion 데이터는 **canonical SMPL 22-joint, [T, J=22, 3], fps=20, root-relative (PELVIS=origin)** 로 정규화 후 컴포넌트 간 전달한다. Skeleton Normalizer 이후의 motion은 본 format을 깨지 않는다. joint 순서는 [`skeleton_normalizer/canonical_smpl_22.py`](skeleton_normalizer/canonical_smpl_22.py) 단일 출처.

### 3-2. Tool Registry 인터페이스

CorrectionTool (명세 §6.3 공통 인터페이스) 의 `apply(motion, target_part, target_joints, frame_range, strength, metadata)` signature 를 모든 correction tool 이 준수한다. 변경 시 [`correction_tools/base.py`](correction_tools/base.py) 와 모든 구현체를 동시 갱신 — [`.claude/skills/change-obligation/SKILL.md`](.claude/skills/change-obligation/SKILL.md) (작성 후).

Evaluator 의 output schema (`agent`·`error_type`·`body_part`·`frames`·`score`·`severity`·`recommendation`) 도 동일 의무 적용 — [`evaluators/base.py`](evaluators/base.py) 단일 출처.

### 3-3. KDG Ordering 의무

Orchestrator 의 tool 선택·ordering 은 [`orchestrator/kdg.py`](orchestrator/kdg.py) 의 Kinematic Dependency Graph (명세 §6.4.0) ordering rule 을 위반하지 마라.

- Root 또는 상위 kinematic node 수정 tool 이 먼저 호출.
- 동일 depth 에서는 hard-constraint tool (contact projection / joint angle clamp) → soft tool (smoothing / learned calibrator) 순.
- KDG ancestor-descendant 관계 joint 를 modify 하는 두 tool 은 동일 step 병렬 적용 금지.

KDG ordering rule 변경은 사용자 승인 게이트 ([§3-11](#3-11-가설-사전-등록과-보수적-수정) 의 가설 본문 수정과 동급).

### 3-4. Closed-loop Score 비감소

[`refinement_loop/convergence.py`](refinement_loop/convergence.py) 의 종합 Score (명세 §6.4.1: `expected_artifact_reduction - fidelity_risk - conflict_risk - tool_cost`) 는 step별 비감소 (non-decreasing) 여야 한다. Score 가 악화되는 tool 호출은 reject 또는 rollback. Same `(tool, target)` 재호출은 strength 를 줄여서만 허용 (oscillation 방지).

### 3-5. Generator Quality-tier 분리

G1 (high-quality) · G2 (token-based) · G3 (legacy artifact-rich) generator 결과는 결과 디렉토리·파일명·raw record 의 generator id 로 명확히 분리한다. 한 generator 결과를 다른 generator 평가기에 입력 금지. 명세 §9.2 quality-tier coverage 의무.

### 3-6. 평가 기록 의무

추론·평가 수행 시 다음 항목 동시 기록:

- 사용한 generator id + version hash + prompt (있는 경우).
- 사용한 evaluator·correction_tool registry 의 config hash.
- Skeleton Normalizer 의 `model_card.json` hash.
- artifact metric (명세 §9.3.1 정의).
- NetGain (명세 §9.4: ArtifactReduction - α·FidelityLoss - β·CorrectionMagnitude - γ·ToolCallCost).
- FidelityLoss (Protocol A/B/C 중 어느 것).
- Efficiency metric (FLOPs / runtime / tool call count).
- Tool call trace (모든 step 의 tool·target·strength·before/after metric).

위 항목 중 하나라도 누락된 평가 결과는 비교 근거로 인용하지 마라. 기록은 (a) 기계 처리 raw record (`evals/raw/<timestamp>.json` — [`.claude/skills/eval-collect/SKILL.md`](.claude/skills/eval-collect/SKILL.md), 작성 후) + (b) 사람-가독 연구일지 (`reports/<YYYY-MM-DD>.md` — [`.claude/skills/research-journal/SKILL.md`](.claude/skills/research-journal/SKILL.md)) 에 동시 보존.

### 3-6-1. 연구일지 작성 의무

핵심 경로 ([`.claude/rules/phase/02-sensor.md §2`](.claude/rules/phase/02-sensor.md)) 변경, 또는 generator inference·평가 1회 이상 실행 시 그 날의 `reports/<YYYY-MM-DD>.md` 작성. 4개 필수 항목: (a) 정량 지표 (generator·tool registry 별 분리), (b) 시각화 (다중 sample 우선 + tool call trace), (c) 실험 메타 (재현 가능 명령어·시드·환경), (d) raw record cross-link.

### 3-7. 자가 수정 메타 규칙

동일 유형의 센서 실패 3회 이상 반복 또는 회귀 리포트의 동일 회귀가 둘 이상 연속 스냅샷 확인 시 재발 방지 규칙을 AGENTS.md §3 또는 [`.claude/rules/phase/`](.claude/rules/phase/) 해당 phase 에 추가. 메트릭 근거 없이 절대 규칙 추가 금지.

### 3-8. 검증 질의와 구현 지시의 분리

사용자 검증 질의 ("이 부분이 맞아?", "왜 이렇게 했어?") 는 답변만. 구현 지시로 확장 해석 금지.

### 3-9. 단일 sample · 단일 trial 결론 금지

단일 sample 시각화나 한 trial 의 NetGain 만으로 framework 성능 결론을 내리지 마라. 최소한 검증 split 전체 또는 동일 quality-tier 의 다중 generator output 분포 + paired test ([`.claude/skills/reproducibility-checklist/SKILL.md §3-7`](.claude/skills/reproducibility-checklist/SKILL.md)) 사용.

### 3-10. 데이터·모델 버전 관리 의무

새로 생성하는 산출물 (generator output JSON, normalized motion, evaluator config, tool registry config, LoRA adapter) 은 [`.claude/skills/data-versioning/SKILL.md §2`](.claude/skills/data-versioning/SKILL.md) (작성 후) metadata 규약 따른다. 학습·추론·평가 시작 전 정합성 검증 통과 — metadata 불일치 상태에서 결과 산출 금지.

### 3-11. 가설 사전 등록과 보수적 수정

연구 가설은 [`.claude/skills/hypothesis-registry/SKILL.md`](.claude/skills/hypothesis-registry/SKILL.md) 에 따라 `evals/hypotheses/<h_id>.md` 로 사전 등록, 본문 영구 보존 (append-only). status 전환 (`active` → `supported|rejected|superseded|withdrawn`) 과 가설 supersede 에 따른 새 가설 promote 는 **사용자 승인 게이트** 거친다. Agent 단독 진행 금지.

### 3-12. 외부 공개 시 재현성 체크리스트

논문·발표·외부 README 결과 공개 전 [`.claude/skills/reproducibility-checklist/SKILL.md §2`](.claude/skills/reproducibility-checklist/SKILL.md) 14+1 항목 점검. 평가 지표는 동 skill §3 정의 단일 출처 인용. 자체 재정의 또는 공식과 다른 변형을 외부 산출물에 포함 금지.

### 3-13. Negative Result 보존 의무

폐기·발산·기각된 시도 (tool 적용 후 artifact 증가, refinement loop 발산, 가설 contradicts 등) 는 일지의 "실패한 시도" 절 ([`.claude/skills/research-journal/SKILL.md §3-5`](.claude/skills/research-journal/SKILL.md)) 에 명시. raw record 에 `negative_result: true` 표기.

### 3-14. 우회·간접 해결 기록 의무

정공법으로 해결 못 하고 우회·간접 해결 (라이브러리 미호환·하드웨어 한계·tool 인터페이스 제약·임시 generator 사용 등) 한 사항은 발견 즉시 [`.claude/skills/workaround-tracking/SKILL.md §4`](.claude/skills/workaround-tracking/SKILL.md) 에 따라 `evals/workarounds/<W-id>.md` 등록 (append-only). `status: open` + `severity: critical` 인 항목 1개라도 있으면 외부 공개 보류.

---

## 4. 경로별 조건 분기

본 섹션은 특정 파일·디렉토리 변경 시 추가 검증·재생성을 강제. 모든 항목은 `<변경 조건> → <대응 검증>` 형식.

### 4-1. `skeleton_normalizer/`

canonical format 정의 또는 root-relative 로직 변경 → 전체 generator (G1/G2/G3) output 의 round-trip 재검증. 충분성 지표 (artifact metric coverage) 재산출.

### 4-2. `evaluators/<name>_evaluator.py`

evaluator 정의 (artifact metric 계산식 또는 severity threshold) 변경 → 모든 historical raw record 의 해당 metric 재계산 가능 여부 확인. 직전 스냅샷과 호환되지 않으면 `aggregation_rule_version` 상승. [`.claude/skills/reproducibility-checklist/SKILL.md §3`](.claude/skills/reproducibility-checklist/SKILL.md) 지표 정의 사전 동시 갱신.

### 4-3. `correction_tools/<name>_tool.py`

CorrectionTool interface 변경 (`apply` signature, output schema) → [`correction_tools/base.py`](correction_tools/base.py) 와 모든 구현체 + orchestrator scoring 동시 갱신. integration smoke 재실행.

### 4-4. `orchestrator/kdg.py` 또는 `orchestrator/scoring.py`

KDG nodes/edges/weights 또는 ScoreFunction 정의 변경 → 사용자 승인 게이트 (§3-11 가설 본문 수정과 동급). 변경 사유와 비교 baseline 을 커밋 메시지에 기록.

### 4-5. `refinement_loop/loop.py` 또는 `refinement_loop/convergence.py`

max_iterations 또는 종료 조건 변경 → 변경 사유 + 기존 trial 결과의 reanalysis 검토. 정합성 검증.

### 4-6. `generators/<name>_wrapper.py`

새 generator wrapper 추가 → [`generators/base.py`](generators/base.py) Generator interface 준수 검증. 출력이 canonical SMPL 22 format으로 변환 가능한지 [`skeleton_normalizer/`](skeleton_normalizer/) 와 함께 round-trip 검증.

### 4-7. `requirements.txt`

변경 시 → `motion-router` 환경 에서 `pip install -r requirements.txt` 먼저 수행. 변경된 패키지가 학습/추론 결정성에 영향 주는지 확인 (특히 `torch`, `transformers`, `peft`, `bitsandbytes`).

### 4-8. `external_assets/**`

이전 저장소의 자산이 junction 으로 참조됨. 본 디렉토리 직접 수정 금지 — 이전 저장소가 freeze 상태라야 본 저장소 결과 재현 가능. 이전 저장소 변경 필요 시 사용자 승인 후 진행.

---

## 5. 실패 대응 & 흔한 실수

### 5-1. `external_assets/` junction 깨짐

이전 저장소 디렉토리 이동·이름 변경 시 junction 깨짐. 본 저장소의 `external_assets/<name>` 을 삭제 후 재생성 (cmd `mklink /J`).

### 5-2. NF4 inter-session 비결정성 (G3 generator)

이전 저장소의 LoRA 를 G3 generator 로 사용 시 같은 seed·같은 입력에서 ±0.02 정도 결과 변동 발생 가능 ([이전 저장소 W-2026-010](external_assets/code/../../../3D-Motion-Trajectory-prediction/evals/workarounds/W-2026-010.md) 참고). 본 프로젝트의 G3 inference 는 N≥3 generation 평균 또는 `torch.use_deterministic_algorithms(True)` 적용 권장.

### 5-3. Tool conflict 의도치 않은 발생

orchestrator 가 같은 joint 에 두 tool 을 연속 호출 시 KDG ConflictScore 가 threshold 초과 → tool 호출 reject. 원인 분석: KDG affected joint set `A(t)` 정의 누락 또는 tool 분류 오류. [`orchestrator/kdg.py`](orchestrator/kdg.py) 의 affected joints 매핑 점검.

### 5-4. Refinement loop oscillation

같은 `(tool, target)` 가 반복 호출되며 같은 artifact 가 추가·삭제 반복 → §3-4 Score 비감소 + strength reduction 적용 안 되는 경우. `refinement_loop/convergence.py` 의 `same_pair_strength_decay` 로직 점검.

### 5-5. 연구 차원 리스크

- **Artifact 정의의 도메인 의존성**: foot sliding · jitter 등 metric 의 임계가 motion category (걷기 vs 점프 등) 에 따라 다를 수 있음. 명세 §9.3.1 의 정의를 단일 출처로 유지하되, 일부 metric 의 robust 검증 필요. [`reproducibility-checklist SKILL §3`](.claude/skills/reproducibility-checklist/SKILL.md) 지표 정의 사전에 적용 도메인 명시.
- **Tool effect의 generator dependency**: 명세 §9.2 G1/G2/G3 별로 효과 다를 수 있음. tool effect matrix (E2 ablation) 로 정량.
- **No-harm 평가의 보수성**: G1 high-quality generator output 에 orchestrator 가 적용된 후 motion 이 미세하게라도 손상되면 H-2026-203 contradicts. fid threshold·user study non-inferiority margin 사전 등록 필수.

---

## 6. 위험 행동

### 6-1. 거대 산출물 git 커밋

generator output · LoRA · HumanML3D 데이터 (GB 단위) 를 git 에 커밋 금지. `.gitignore` 등재 + symlink/junction 경유 참조.

### 6-2. 평가 결과 재기록

기존 `reports/<날짜>.md` 의 수치를 새 실험 결과로 덮어쓰지 마라. generator·tool registry 변경·체크포인트별로 새 파일·새 행 추가.

### 6-3. 임계값 완화로 회귀 회피

NetGain 임계, artifact reduction 임계, fidelity loss 상한 등을 낮춰 회귀 통과시키지 마라. 회귀는 원인 수정 또는 §3 절대 규칙 강화로만 해소.

### 6-4. 가설 사후 수정 (HARKing)

결과 본 뒤 H-2026-200~203 본문을 사후 수정해 supports 로 보이게 만들지 마라 (Kerr 1998 HARKing). 가설은 append-only. 변경은 새 draft → 사용자 승인 → 새 H-id 로만.

### 6-5. metadata 우회로 산출물 동질화

서로 다른 evaluator config · tool registry · generator output 을 동일 raw record id 로 덮어쓰거나 metadata 사후 편집해 hash 맞추지 마라. silent invalidation 은 §3-10 위반.

### 6-6. 우회 미기록·등급 임의 하향

정공법 실패 사항을 ledger 등록 없이 진행 또는 등급 (critical/material/low) 을 결과에 맞춰 임의 하향 금지.

### 6-7. 단일 trial 가설 평가

단일 trial 결과로 H-2026-200~203 supports/contradicts 단정 금지. 판정은 [`.claude/skills/eval-compare/SKILL.md §6`](.claude/skills/eval-compare/SKILL.md) 5단계 리포트에서만, trial ≥ 20 + snapshot ≥ 2 하에서.

### 6-8. KDG ordering rule bypass

`orchestrator/kdg.py` 의 ordering rule 을 우회해 tool 을 임의 순서로 호출 금지. ordering violation 은 §3-3 위반이며 conflict_risk = ∞ 로 처리.

### 6-9. Tool conflict 무시

같은 frame_range·target_part 에 두 tool 이 연속 호출될 때 conflict_risk 가 threshold 초과한 결과를 무시하고 강행 금지.

### 6-10. Generator transfer 결과 임의 일반화

H-2026-202 (generator-agnostic) 검증 시 1-2개 generator 결과만으로 일반화 단정 금지. 최소 3 generator (G1/G2/G3) 의 다양한 quality-tier 확인 + paired test.

---

## 7. 디렉토리별 상세 규칙

### 7-1. `generators/`

외부 motion generator 의 wrapper 디렉토리. 각 wrapper 는 `generators/base.py` 의 Generator interface 를 구현. 출력은 항상 `[T, 22, 3]` numpy + metadata.

본 저장소가 새 motion generator 를 구현하지 않는다 — 외부 generator 의 wrapping 만.

### 7-2. `skeleton_normalizer/`

외부 generator output → canonical SMPL 22 변환. canonical format 정의는 `canonical_smpl_22.py` 단일 출처. root-relative 로직·contact label 추정·ground plane 추정 모듈 분리.

본 저장소는 retargeting tool (SMPL-X 등) 을 직접 구현하지 않고 외부 라이브러리 또는 이전 저장소 코드를 import.

### 7-3. `evaluators/`

명세 §6.2 7 evaluator (contact / temporal / skeletal / root_torso / upper_limb / lower_limb / coordination) registry. 각 evaluator 는 `base.py` interface 준수. output schema 일관성 필수.

### 7-4. `correction_tools/`

명세 §6.3 correction tool registry. 각 tool 은 `base.py` 의 `CorrectionTool.apply(...)` signature 준수. correction magnitude 와 modified joints 를 report 에 명시.

### 7-5. `orchestrator/`

evaluator reports + tool history → tool decision. `kdg.py` (Kinematic Dependency Graph), `scoring.py` (Score function), `rule_based.py` (Phase 1), `supervised_selector.py` + `contextual_bandit.py` (Phase 2). 각 알고리즘은 동일 input/output interface.

### 7-6. `refinement_loop/`

closed-loop refinement 의 종료·convergence 검증. max_iterations · Score 비감소 · oscillation 방지 룰. tool call trace 저장.

### 7-7. `tools/`

시각화·실험 도구. 명세 §9 의 protocol (synthetic injection, tool effect matrix, perceptual rating) 구현.

### 7-8. `evals/`

평가 파이프라인의 기계 처리 산출물.
- `evals/raw/<timestamp>_<task_id>_<trial_id>.json` — Collect.
- `evals/normalized/`, `evals/graded/`, `evals/snapshots/`, `evals/reports/<period>.md` — 후속.
- `evals/hypotheses/<h_id>.md` — append-only.
- `evals/workarounds/<W-id>.md` — append-only.

### 7-9. `reports/`

연구일지 + 시각화. 일자별 일지 (`reports/<YYYY-MM-DD>.md`) + figures (`reports/figures/<YYYY-MM-DD>/*.png|.gif`).

### 7-10. `.claude/`

Claude Code 운영 자산. `rules/phase/` 는 phase 판단 규칙, `skills/` 는 실행 절차·평가 파이프라인.

### 7-11. `external_assets/`

이전 저장소 자산 import. junction 경유 read-only 참조. 본 디렉토리 직접 수정 금지 (§4-8).

### 7-12. `experiments/`

MVP feasibility study (Week 1-4) 의 segregated workspace. 각 Week 의 산출물은 해당 sub-디렉토리에.

---

## 8. 참고 전용 (아래 섹션)

위 (1~7) 는 모든 변경에 필수. 아래 (9~10) 는 특정 구현 작업 시에만 참조.

---

## 9. 구현 레시피

### 9-1. 새 evaluator 추가

1. `evaluators/<name>_evaluator.py` 작성 — `Evaluator` base class 상속.
2. `evaluate(motion) -> EvaluatorReport` 구현 (명세 §6.2 출력 schema).
3. 단위 테스트 `tests/unit/evaluators/test_<name>.py` 추가.
4. evaluator registry config (`evaluators/__init__.py` 또는 dedicated config) 갱신.
5. [reproducibility-checklist §3 지표 정의 사전](.claude/skills/reproducibility-checklist/SKILL.md) 에 metric 추가.
6. integration smoke 재실행.

### 9-2. 새 correction tool 추가

1. `correction_tools/<name>_tool.py` 작성 — `CorrectionTool` base class 상속 (명세 §6.3).
2. `apply(motion, target_part, target_joints, frame_range, strength, metadata)` 구현.
3. KDG affected joints `A(t)` 와 propagation weights 등록 (`orchestrator/kdg.py`).
4. artifact-tool compatibility table (rule-based orchestrator) 에 매핑 추가.
5. 단위 테스트 + integration smoke + tool effect matrix 재실행.

### 9-3. 새 generator wrapper 추가

1. `generators/<name>_wrapper.py` 작성 — `Generator` base class 상속.
2. `generate(prompt, n_frames) -> (motion, metadata)` 구현. motion 은 `[T, 22, 3]` 또는 wrapper 내부에서 변환.
3. skeleton normalizer 와의 round-trip 검증.
4. G1/G2/G3 quality-tier 분류 결정 (명세 §9.2).
5. evals/raw 의 `generator_id` 필드에 등록.

### 9-4. 신규 가설 등록

1. `evals/hypotheses/_index.md` 에서 다음 h_id 확인.
2. [`hypothesis-registry SKILL §2-3`](.claude/skills/hypothesis-registry/SKILL.md) 형식으로 `H-YYYY-NNN.md` 작성.
3. `사전 정의`·`기각 조건`·`표본 요건` 모두 채운다.
4. `_index.md` 갱신.

### 9-5. 우회 발견 시 등록

1. 정공법 실패 확정 즉시 [`workaround-tracking SKILL §4`](.claude/skills/workaround-tracking/SKILL.md) 절차로 W-id 부여.
2. ledger 파일 작성 (`severity`·`status`·`resolution_target` 포함).
3. 관련 가설·일지·model_card 에 W-id cross-link.
4. critical 인 경우 사용자에게 발견 시점 알림.

### 9-6. 외부 공개 전 점검

1. [`reproducibility-checklist §2`](.claude/skills/reproducibility-checklist/SKILL.md) 14+1 항목 점검.
2. `evals/workarounds/_index.md` 에 `status: open` + `severity: critical` 이 0개임을 확인.
3. `reports/checklists/<period>_<scope>.md` 에 점검 결과 기록.

### 9-7. MVP Feasibility Study 단계 (명세 §12)

- **Week 1**: generator output 확보 + skeleton format 통일 + visualization.
- **Week 2**: evaluator metrics 구현 (foot, jitter, bone 최소 3종).
- **Week 3**: oracle best-tool 실험 + tool-effect matrix + rule-based orchestrator.
- **Week 4**: fixed smoothing baseline + synthetic injection + perceptual rating + Go/Stop 결정 (명세 §11).

본 단계는 [`experiments/week<N>_*/`](experiments/) 디렉토리 단위로 segregate.

---

## 10. 아키텍처 제약 & 네이밍

### 10-1. 파일 네이밍

- Evaluator: `evaluators/<artifact-category>_evaluator.py` (예: `contact_evaluator.py`).
- Correction tool: `correction_tools/<tool-name>_tool.py` (예: `foot_lock_tool.py`).
- Generator wrapper: `generators/<generator-name>_wrapper.py` (예: `motiongpt_wrapper.py`).
- 단위 테스트: `tests/unit/<component>/test_<name>.py`.

### 10-2. 데이터 포맷

- 내부 motion: `[T, 22, 3]` numpy float64 + 메타데이터 dict.
- Evaluator report: schema 명세 §6.2 직접 인용.
- Tool report: schema 명세 §6.3 직접 인용.
- Orchestrator decision: schema 명세 §6.4 직접 인용.
- raw record schema: [`eval-collect SKILL §5`](.claude/skills/eval-collect/SKILL.md) (작성 후).

### 10-3. 결과 파일 분리

- generator (G1/G2/G3) · evaluator config · tool registry config · seed 별로 결과 파일 분리.

### 10-4. 참고 문서

- 하네스 4계층 원본: [`docs/harness-research-template/01-instructions.md`](docs/harness-research-template/01-instructions.md), [`02-sensor.md`](docs/harness-research-template/02-sensor.md), [`03-test.md`](docs/harness-research-template/03-test.md), [`04-evaluation.md`](docs/harness-research-template/04-evaluation.md).
- 하네스 적용본: [`.claude/rules/phase/`](.claude/rules/phase/), [`.claude/skills/`](.claude/skills/).
- 연구 명세 (단일 출처): [`docs/motion_research_strategy_summary.md`](docs/motion_research_strategy_summary.md).
- 사전 등록 가설: [`evals/hypotheses/_index.md`](evals/hypotheses/_index.md).
- 평가 지표 정의 사전: [`.claude/skills/reproducibility-checklist/SKILL.md §3`](.claude/skills/reproducibility-checklist/SKILL.md).
- 우회 ledger: [`evals/workarounds/_index.md`](evals/workarounds/_index.md).
- 이전 저장소 (sanity check 결과): [`../3D-Motion-Trajectory-prediction/`](../3D-Motion-Trajectory-prediction/).
- MotionGPT 공식: <https://github.com/OpenMotionLab/MotionGPT>.
- HumanML3D 공식: <https://github.com/EricGuo5513/HumanML3D>.
