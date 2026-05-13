---
description: ArtifactRouter의 도메인 충분성 지표(artifact metric coverage, tool selection 충분성, refinement loop convergence)를 정의하고 임계값·복구 절차를 제공하는 skill이다. line coverage 대신 도메인 지표를 사용한다.
metadata:
  scope:
    paths:
      - "evaluators/**"
      - "correction_tools/**"
      - "orchestrator/**"
      - "refinement_loop/**"
  activation:
    keywords:
      - "충분성"
      - "sufficiency"
      - "coverage"
      - "convergence"
    when_to_use: 평가 게이트에서 충분성 지표 임계 미달 시. registry 확장 검토 시.
  constraints:
    allowed_tools:
      - "Read"
      - "Write"
      - "Edit"
      - "Glob"
    risk_level: medium
---

# Sufficiency Metric — ArtifactRouter

본 skill은 [`docs/harness-research-template/skillSample/sufficiency-metric.md`](../../../docs/harness-research-template/skillSample/sufficiency-metric.md) 의 절차를 본 프로젝트에 적용한다.

---

## 1. 목적

다음을 보장한다.

- line coverage 가 적용되지 않는 ArtifactRouter 컴포넌트에 대한 도메인 충분성 지표 정의.
- 충분성 임계 미달 시 복구 절차 제공.
- registry 확장 (evaluator·correction_tool 추가) 시 coverage 변화 정량.

---

## 2. 충분성 지표 정의

### 2-1. Artifact Metric Coverage

명세 §6.2 의 7 evaluator 가 의도된 artifact 종류를 cover 하는 비율.

```
ArtifactCoverage = (수행한 evaluator 종류) / 7
```

목표: ≥ 7/7 = 100%. 단 일부 평가 단계 (MVP Week 2) 에서 점진 확장.

**임계**:
- Week 2 종료 시점: ≥ 3 evaluator (foot / jitter / bone).
- Week 4 (MVP 종료) 시점: ≥ 5 evaluator.
- 본격 학습 진입 시점: 7/7.

### 2-2. Tool Selection 충분성

명세 §6.3 의 9 correction tool 중 orchestrator 가 candidate 로 등록한 tool 의 비율.

```
ToolSelectionSufficiency = (orchestrator candidate set 의 tool 수) / 9
```

**임계**:
- MVP Week 4 종료 시: ≥ 4 tool.
- 본격 학습: ≥ 6 tool.

### 2-3. Refinement Loop Convergence Rate

max_iterations 안에 STOP 도달 + Score 비감소 만족한 trial 의 비율.

```
ConvergenceRate = (정상 종료 trial) / (total trial)
```

**임계**:
- 정상 종료: max_iterations 도달 전 STOP + Score 비감소.
- 비정상 종료: max_iterations 도달 (수렴 안 됨) 또는 Score 감소 발생.

**임계값**:
- MVP: ≥ 70%.
- 본격 학습: ≥ 90%.

### 2-4. Tool Conflict Rate

KDG ConflictScore 가 threshold 초과해 reject 된 tool 호출 의 비율.

```
ConflictRate = (rejected tool calls) / (total tool calls attempted)
```

**임계**:
- MVP: 정상 범위 5-15% (적절한 conflict 감지).
- < 5% → KDG 가 too permissive (threshold 너무 큼).
- > 20% → KDG 가 too strict (threshold 너무 작음) 또는 affected joint 매핑 오류.

### 2-5. Format Integrity Rate

refinement loop 종료 후 motion 이 canonical SMPL 22 format 을 유지한 trial 비율.

```
FormatIntegrityRate = (format 무결 trial) / (total trial)
```

**임계**: 100%. 100% 미만이면 hard fail (skeleton normalizer 또는 correction tool 의 format 위반).

---

## 3. 임계 미달 시 복구 절차

### 3-1. ArtifactCoverage 미달

원인:
- 미구현 evaluator.
- evaluator config 등록 실패.

복구:
1. 누락 evaluator 식별 (명세 §6.2 의 7 종류와 비교).
2. 우선순위: contact (foot sliding/floating/penetration) > temporal (jitter/jerk) > skeletal (bone length/joint angle).
3. 새 evaluator 구현 + 단위 테스트 작성.
4. integration smoke 재실행.

### 3-2. ToolSelectionSufficiency 미달

원인:
- 미구현 correction tool.
- artifact-tool compatibility matrix 누락.

복구:
1. 누락 tool 식별 (명세 §6.3 의 9 종류와 비교).
2. 우선순위: foot_lock > velocity_smoothing > bone_projection > joint_angle_clamp.
3. 새 correction tool 구현 + KDG affected joints 등록.
4. tool effect matrix (E2 ablation) 로 효과 검증.

### 3-3. ConvergenceRate 미달

원인:
- max_iterations 너무 작음.
- Score function 의 weight 가 unstable.
- Tool 들 간 oscillation.

복구:
1. max_iterations 증가 검토.
2. Score function weight (α, β, γ) 조정 (사용자 승인 게이트 — KDG/scoring 변경은 §3-11 가설 본문 수정과 동급).
3. same-pair strength decay 강도 증가.
4. oscillation 발견 시 tool ordering rule 강화 (KDG).

### 3-4. ConflictRate 비정상

원인:
- KDG affected joint 매핑 부정확.
- ConflictScore threshold 잘못 설정.

복구:
1. KDG nodes/edges 재정의.
2. 각 tool 의 `A(t)` (affected joints) + propagation weights 검증.
3. ConflictScore threshold 보정 (history 분석).

### 3-5. FormatIntegrity 위반

원인 (hard fail):
- Skeleton Normalizer 의 round-trip 실패.
- Correction tool 이 joint 수 또는 좌표계 변경.

복구:
1. 위반 사건 isolate.
2. 위반 tool 또는 normalizer 수정.
3. unit test 추가.
4. integration smoke 재실행.

---

## 4. 측정 절차

각 평가 trial 후 충분성 지표를 raw record 에 기록:

```python
"sufficiency_metrics": {
    "artifact_coverage": 5 / 7,
    "tool_selection_sufficiency": 6 / 9,
    "convergence_rate": 0.85,
    "conflict_rate": 0.08,
    "format_integrity_rate": 1.0
}
```

snapshot 에서 trial 단위 분포로 집계.

---

## 5. 기준 변경 규칙

- 임계값 **하향 조정 금지** (회귀 회피 차단).
- 상향 조정 허용. 단 본 SKILL §2 와 [`02-sensor.md §1-1`](../../rules/phase/02-sensor.md), [`03-test.md §6`](../../rules/phase/03-test.md) 동시 갱신.
- 새 충분성 지표 추가 시 본 SKILL §2 에 정의 추가.

---

## 6. 금지 규칙

- 임계값 완화로 회귀 회피 금지.
- 충분성 지표 미달 상태에서 외부 결과 인용 금지.
- 100% 목표 임의 적용 금지 (특히 ArtifactCoverage — 모든 artifact 가 모든 motion 에 발생하지 않음).
