# 03. 테스트(Test) Phase — ArtifactRouter

> 본 phase 문서는 [`docs/harness-research-template/03-test.md`](../../../docs/harness-research-template/03-test.md) 의 "03. 테스트 레이어 — Research Track" 규격을 본 프로젝트(ArtifactRouter)에 적용한 결과이다. 분류·오라클·더블·flaky·충분성 원칙의 정의는 상위 템플릿을 상속하며, 본 문서는 차이만 기술한다. 본 프로젝트는 pytest 기반 unit/integration 테스트와 end-to-end refinement loop 검증을 결합한다.

---

## 1. 테스트 분류 (본 프로젝트 적용)

상위 템플릿 §1의 4 분류 (Unit/Integration/Contract/Regression) 를 ArtifactRouter 컴포넌트에 매핑.

### 1-1. Unit — round-trip + 단위 검증

각 컴포넌트를 외부 의존 없이 검증한다.

- **Skeleton Normalizer**: 임의 `[T, 22, 3]` motion → root-relative 정규화 → 역변환 (epsilon 이내).
- **Evaluator (각각)**: synthetic motion에 대한 expected metric 값 (예: foot이 ground 위 10cm 떠 있으면 FootFloating > 0.05).
- **Correction Tool (각각)**: target artifact가 인위적으로 주입된 motion에 tool 적용 → 해당 artifact metric 감소 확인.
- **Orchestrator/Scoring**: 알려진 (artifact state, history) 입력에 대한 Score 출력이 결정적이고 예측 가능.
- **Orchestrator/KDG**: 알려진 (tool_a, tool_b, history) 쌍에 대한 ConflictScore 결정적.

테스트 위치: `tests/unit/<component>/test_<name>.py`. pytest 표준 형태.

### 1-2. Integration — refinement loop end-to-end

generator → normalize → evaluate → correct → re-evaluate가 결합된 경계에서 동작 검증.

- **형태**: G1 또는 G3 generator의 짧은 motion (8-32 frames) 으로 closed-loop refinement 1회 실행.
- **검증**: 충분성 지표 (tool selection 충분성, format integrity) 가 0이 아닌지·기본 분포가 깨지지 않았는지.
- 실패 시 [`sufficiency-metric SKILL §4`](../../skills/sufficiency-metric/SKILL.md) 원인 분류·복구 절차.

### 1-3. Contract — 컴포넌트 인터페이스

외부 인터페이스 합의 검증.

- **Generator interface**: `generators/base.py` 의 `Generator.generate(prompt, n_frames)` 가 `[T, 22, 3]` numpy array + metadata 반환.
- **Skeleton Normalizer interface**: `normalize(motion, fps)` 가 canonical SMPL 22-joint format + ground plane + contact label 반환.
- **Evaluator interface**: `evaluate(motion)` 이 evaluator output schema (`agent`, `error_type`, `body_part`, `frames`, `score`, `severity`, `recommendation`) 준수.
- **CorrectionTool interface (명세 §6.3)**: `apply(motion, target_part, target_joints, frame_range, strength, metadata)` 가 `(corrected_motion, report)` tuple 반환.
- **Orchestrator decision schema**: `decision`, `primary_error`, `selected_tool`, `target_part`, `target_frames`, `strength`, `next_step`.

본 분류 검증은 [`change-obligation SKILL`](../../skills/change-obligation/SKILL.md) 매핑 규약과 결합 (skill 작성 후).

### 1-4. Regression — 평가 메트릭 비교

과거 NetGain·artifact reduction·FidelityLoss 회귀가 재현되지 않는지 감시.

- 평가기 출력의 generator (G1/G2/G3) 별 NetGain·artifact metric의 직전 스냅샷 대비 비교.
- 다중 trial 평균·CI로 비교. 단일 trial로 회귀 단정 금지 ([`AGENTS.md §3-9`](../../../AGENTS.md)).

### 1-5. 비교 트랙 (Generator quality-tier coverage)

본 프로젝트는 비교 대상이 **delta/absolute** 가 아니라 **G1 (high-quality) / G2 (token-based) / G3 (legacy artifact-rich)** generator 의 quality-tier. 변경 영향 선정 시 다음을 추가 강제:

- skeleton_normalizer·evaluator·orchestrator 변경 → **G1/G2/G3 generator 출력 모두에 대해 round-trip + smoke 동시 수행**.
- 한 generator에 대한 결과만 측정한 경우 → 다른 generator도 정합성 검증.
- 파일명 substring 매칭만으로 영향 범위를 결정하지 마라.

### 1-6. 금지 규칙

- 단위 round-trip 검사를 빼먹고 integration smoke만으로 변환기·파서를 검증하지 마라.
- 단일 sample 시각화로 회귀 결론 금지.
- pytest 인프라 도입 후에도 ad-hoc 검사를 허용하지만 결과를 커밋 메시지·`reports/`에 남긴다.

---

## 2. 오라클 (통계적 보강)

상위 템플릿 §2의 3 오라클 (reference/property/metamorphic) 원칙 그대로. 본 프로젝트 보강:

### 2-1. 통계적 오라클

모델 출력 (refined motion) 의 회귀 판정은 단일 trial 임계 비교가 아니라:

- **paired test** — 같은 generator output에 대해 (rule-based orchestrator vs learned orchestrator) 또는 (with vs without tool X) 페어링한 NetGain에 Wilcoxon signed-rank.
- **bootstrap CI** — N=1000 resampling, 95% CI.
- **effect size** — Cohen's d 또는 ratio of NetGain.

본 통계적 오라클은 [`reproducibility-checklist SKILL §3-7`](../../skills/reproducibility-checklist/SKILL.md) 을 단일 출처로 인용.

### 2-2. 컴포넌트 오라클

- **Skeleton Normalizer**: property (round-trip), metamorphic (좌표 평행이동 시 root-relative 좌표 불변).
- **Evaluator**: reference (synthetic motion의 GT artifact value 와 비교), property (motion 전체 평행이동 시 metric 불변 또는 예측 가능 변화).
- **Correction Tool**: property (target artifact metric 단조 감소), reference (synthetic injection 후 복원 시 GT와 비교).
- **Orchestrator**: reference (mock state에서 expected decision 비교), property (KDG ordering 위반 시 무조건 conflict_risk > threshold).

### 2-3. 금지 규칙

- "예외 없이 실행됨" 만을 모델 오라클로 삼지 마라.
- 컴포넌트 코드의 계산 로직을 테스트에서 복붙해 오라클로 만들지 마라 (동일 결함 양쪽 재현).
- 단일 motion 시각화로 합격 판정 금지.
- 내부 attention·hidden state·logit 같은 구현 세부에 묶인 오라클 금지.

---

## 3. 테스트 더블

상위 템플릿 §3 그대로. 본 프로젝트 추가:

- **fixture는 결정적 시드와 묶는다** — synthetic artifact injector의 seed, generator inference seed, dataloader seed를 fixture 정의에 포함.
- **Skeleton Normalizer는 실제 사용** — fake normalizer로 대체 시 contact label·ground plane 추정이 깨짐.
- **통합 smoke와 회귀 평가는 실제 generator + LoRA로** — 변환기·evaluator·correction_tool 단위 검증에서는 mock motion으로 가능, 통합·회귀는 실제 모델.

### 3-1. 금지 규칙

- skeleton_normalizer의 내부 함수에 mock 주입해 round-trip 결과 가리지 마라.
- Generator wrapper를 fake로 대체하지 마라 (실제 generator의 artifact 분포가 본 연구의 평가 대상).
- 통합 smoke에서 evaluator·correction_tool을 stub으로 대체 금지.

---

## 4. 변경 영향 테스트 선정 (Generator quality-tier 분리)

본 프로젝트는 G1/G2/G3 generator quality-tier가 동시 운영. 변경 영향 선정 시 추가 강제:

- skeleton_normalizer·evaluator·correction_tool·orchestrator 변경 → **G1 + G2 + G3 generator output 모두에 대해 round-trip + smoke 동시 수행**.
- G1만 변경된 경우 → G2/G3에도 정합성 확인 (interface 변경이 leak되지 않았는지).

safe vs precise trade-off (국소 precise, 전체 safe) 그대로.

---

## 5. flaky 대응

본 절은 원칙만 선언. 상세는 [`flaky-handling SKILL`](../../skills/flaky-handling/SKILL.md) (skill 작성 후) 또는 [`harness-research-template`의 SW 트랙 flaky-handling](../../../docs/harness-research-template/../harness-template/skillSample/flaky-handling.md) 위임.

본 프로젝트 추가 의무:

- 동일 seed·동일 입력의 refinement loop를 N≥3회 재실행.
- flaky 후보 범주 분류:
  - seed 고정 누락.
  - CUDA/GPU 비결정성 (특히 NF4 양자화 generator).
  - tool registry 순서 의존.
  - dataloader 순서 의존.
- flaky 후보를 일지의 "실패한 시도" 절([`research-journal SKILL §3-5`](../../skills/research-journal/SKILL.md)) 에 cross-link.

**IMPORTANT:** "한 번 더 돌려보니 괜찮았다" 로 종결 금지. 통과한 재실행은 **재현되지 않은 실패**.

---

## 6. 충분성 지표 — 도메인 의존

본 프로젝트의 충분성 지표:

- **Artifact metric coverage**: evaluator registry가 명세 §6.2의 7 evaluator (contact / temporal / skeletal / root_torso / upper_limb / lower_limb / coordination) 의 artifact를 cover하는 비율.
- **Tool selection 충분성**: orchestrator decision이 candidate tool의 artifact-tool compatibility를 명세 §6.4의 rule 따라 만든 비율.
- **Refinement loop convergence**: max_iterations 안에 Score 비감소 + STOP에 도달한 비율.

임계·복구 절차는 [`sufficiency-metric SKILL §2`](../../skills/sufficiency-metric/SKILL.md).

정형 mutation testing류 미도입.

### 6-1. 기준 변경 규칙

- 임계값을 **하향 조정 금지** (회귀 회피 차단).
- 상향 허용. 단 sufficiency-metric SKILL과 본 phase의 서술을 동시 갱신.
- "100% artifact reduction" 같은 수치 목표 금지 (motion fidelity 손상 위험).

---

## 7. Agent 표준 실행 흐름

1. 변경 대상 분류 (§1) 선택.
2. 오라클 (§2) 설계 — 컴포넌트는 property·reference 우선, refined motion은 통계적 오라클.
3. 더블 (§3) 구성 — skeleton_normalizer·실제 generator는 fake 대체 안 함.
4. change-obligation이 지시한 누락 검증 먼저 작성.
5. 변경 영향 검사 (§4) 실행 — G1/G2/G3 동시.
6. 실패 시 → 재현성 검사 (§5) → 결정적 실패 / flaky 분류.
7. end-to-end refinement loop 검증 (전체 게이트).
8. 시각화·사람 검증이 포함되면 다중 sample 평균과 함께 확인.
9. 전체 게이트 통과 후 커밋.

**IMPORTANT:** flaky 분류 상태로 커밋 금지.

---

## 8. 유지보수 규칙

- 분류·오라클·더블 원칙을 변경했으면 → §7 흐름과 [`change-obligation SKILL §6`](../../skills/change-obligation/SKILL.md) 복구 절차 동시 갱신.
- 충분성 기준을 조정했으면 → [`02-sensor.md §1-1`](./02-sensor.md)·[`sufficiency-metric SKILL §2`](../../skills/sufficiency-metric/SKILL.md) 동시 갱신.
- pytest 디렉터리 규약을 변경했으면 → §1과 [`AGENTS.md §2-2`](../../../AGENTS.md) 센서 명령어 동시 갱신.
- 본 문서 용어가 [`02-sensor.md`](./02-sensor.md)·[`04-evaluation.md`](./04-evaluation.md)·하위 skill과 불일치 → `02`를 지도, 본 문서를 방법론 원본으로 간주.
