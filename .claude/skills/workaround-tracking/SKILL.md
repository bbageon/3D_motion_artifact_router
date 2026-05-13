---
description: ArtifactRouter 진행 중 정공법으로 해결 못 하고 우회·간접 해결한 사항(라이브러리 미호환·하드웨어 한계·tool 인터페이스 제약·temporary generator 등)을 발견 즉시 ledger에 등록하고, 결론·외부 공개 시점에 통합 보고하는 skill이다.
metadata:
  scope:
    paths:
      - "evals/workarounds/**"
  activation:
    keywords:
      - "우회"
      - "workaround"
      - "정공법"
      - "ledger"
    when_to_use: 정공법 실패 사항 발견 즉시. 외부 공개 전 critical W-id 점검.
  constraints:
    allowed_tools:
      - "Read"
      - "Write"
      - "Edit"
      - "Glob"
    risk_level: high
---

# Workaround Tracking — ArtifactRouter

본 skill은 [`docs/harness-research-template/skillSample/workaround-tracking.md`](../../../docs/harness-research-template/skillSample/workaround-tracking.md) 의 절차를 본 프로젝트에 적용한다.

---

## 1. 목적

다음을 보장한다.

- 정공법 실패 사항을 발견 즉시 ledger 등록 (append-only).
- severity (critical / material / low) 분류.
- 본격 학습·외부 공개 진입 전 critical W-id 해소 또는 명시.
- 모든 평가 결과 인용 시 관련 W-id cross-link.

---

## 2. severity 분류

| severity | 정의 | 외부 공개 영향 |
|---|---|---|
| **critical** | 결과 의미를 바꾸거나 외부 재현 거의 불가능 | 외부 공개 전 반드시 해소 또는 supersede |
| **material** | 환경·도구 의존성으로 결과 시간/방식 다르지만 수치 동일 | 진입 전 검증 + 일지 명시 |
| **low** | 시각화·디버그 또는 사전 등록 임시 정책 변경 | 가설 변경 이력 기록 후 진행 |

---

## 3. 등록 형식

각 우회는 `evals/workarounds/W-2026-NNN.md`:

```markdown
---
w_id: W-2026-NNN
status: open
discovered_at: YYYY-MM-DDTHH:MM:SSZ
discovered_during: <H-id Stage·Week 1 generator setup 등>
severity: critical | material | low
domain: <예: peft·env·orchestrator·skeleton-normalizer·generator>
related_hypotheses: [<H-id list>]
related_artifacts:
  - <경로>
resolution_target: <H-id 본격 진입 전 또는 외부 공개 전>
---

# W-2026-NNN — <한 줄 요약>

## 본래 의도

<정공법으로 무엇을 하려 했는지>

## 정공법 시도 결과

<실패 양상 + 정량 근거>

## 채택한 우회 방식

<우회의 구체적 구현>

## 재현율 영향 분석 — <severity> 사유

<왜 본 severity 등급인지>

## 본격 학습·외부 공개 진입 전 의무

<다음 단계로 진입하기 위해 무엇을 해야 하는지>

## 영향받는 가설·산출물

- 가설: <H-id list>
- 산출물: <경로 list>
- 일지 cross-link: <reports/<YYYY-MM-DD>.md>

## 변경 이력 (append-only)

- YYYY-MM-DD — 등록.
```

---

## 4. 등록 절차

1. 정공법 실패 확정 즉시 W-id 부여 (`evals/workarounds/_index.md` 에서 다음 NNN 확인).
2. ledger 파일 §3 형식 작성.
3. severity 분류 (§2 기준 엄격 적용 — 결과 따라 임의 하향 금지).
4. 관련 가설 (`evals/hypotheses/<H-id>.md`) 본문에 본 W-id cross-link.
5. 관련 일지 (`reports/<YYYY-MM-DD>.md`) §3-4-1 우회 발견 절에 등록.
6. critical 인 경우 사용자에게 발견 시점 알림.
7. `_index.md` 의 Active 표에 한 줄 추가.

---

## 5. Status 전환 — 사용자 승인 게이트

### 5-1. 가능한 전환

- `open` → `resolved` (정공법으로 우회 해소).
- `open` → `accepted-permanent` (영구적 환경 의존성으로 수용).

### 5-2. 절차

1. **draft 작성**: 해소 근거 (정량 검증 + 5단계 리포트 인용) + 새 코드/환경 변경 명시.
2. **사용자 승인 요청**:
   - critical 필수.
   - material 권장.
   - low 는 권장 사항.
3. **승인 시 적용**: frontmatter `status` 변경, `resolved_at` 또는 `accepted_at` 추가, 변경 이력 한 줄 추가, `_index.md` 의 Resolved 표 또는 Accepted-permanent 표로 이동.

### 5-3. 금지

- 결과에 맞춰 severity 임의 하향 금지.
- 사용자 승인 없이 critical status 전환 금지.
- 외부 공개 전 일괄 등록 금지 (append-only 의미 손상).

---

## 6. 결론·외부 공개 시 의무

다음 시점에 모든 활성 W-id 통합 보고:

### 6-1. [`eval-compare SKILL §6`](../eval-compare/SKILL.md) 5단계 리포트의 부록 A

```markdown
## 부록 A — 우회·간접 해결 ledger 인용

| W-id | severity | status | 한 줄 | 본격 학습 영향 |
|---|---|---|---|---|
| W-2026-001 | critical | open | ... | ... |
```

### 6-2. [`research-journal SKILL §3-4-1`](../research-journal/SKILL.md) 일지

오늘 등록·전환된 W-id 한 줄씩.

### 6-3. [`reproducibility-checklist SKILL §2 #14`](../reproducibility-checklist/SKILL.md)

외부 공개 점검: `status: open` + `severity: critical` 항목 0개 확인.

### 6-4. [`hypothesis-registry SKILL §4`](../hypothesis-registry/SKILL.md) supersede draft

관련 W-id 처리 계획 명시.

---

## 7. ArtifactRouter 특수 사항

본 프로젝트의 우회 예상 영역:

### 7-1. 외부 generator 의존성

- G1/G2 generator 의 checkpoint 가 외부 저장소 dependency.
- 외부 generator 의 버전 hash 가 본 평가 결과의 재현성에 영향.
- W-id 등록 형식: `severity: material`, `domain: env·external-generator`.

### 7-2. NF4 inter-session 비결정성 (G3)

- 이전 저장소의 W-2026-010 — 본 프로젝트에서도 적용.
- G3 inference 시 ±0.02 noise → N≥3 generation 평균 또는 deterministic mode 적용.

### 7-3. KDG affected joint 매핑 불확실성

- Correction tool 의 forward kinematics propagation 정확 추정 어려움.
- KDG weight 가 heuristic 인 경우 → W-id 로 명시.

### 7-4. Tool 효과 generator dependency

- 같은 correction tool 이 G1 에서는 효과 있고 G2 에서는 효과 작은 경우.
- compatibility matrix 의 generator-specific 차이를 W-id 로 명시.

### 7-5. Synthetic injection 의 real artifact 와의 gap

- §9.3 Protocol A (synthetic injection) 결과가 Protocol B (real generator output) 와 일치하지 않을 가능성.
- 본 gap 자체를 W-id 로 등록 가능.

---

## 8. 금지 규칙

- 우회 미기록 진행 금지 ([`AGENTS.md §6-6`](../../../AGENTS.md)).
- severity 결과에 맞춰 하향 조정 금지.
- 외부 공개 직전 일괄 등록 금지 (append-only 의미 손상).
- 사용자 승인 없이 critical status 전환 금지.
