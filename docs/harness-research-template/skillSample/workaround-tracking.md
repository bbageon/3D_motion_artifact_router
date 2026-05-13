---
description: 정공법으로 해결 못 하고 우회·간접 해결한 사항을 발견 즉시 ledger에 기록하고, 결론·외부 공개 시점에 통합 보고하는 skill이다. HARKing 차단·재현율 보존을 목적으로 한다.
metadata:
  scope:
    paths:
      - "evals/workarounds/**"
      - "**/*.py"
  activation:
    keywords:
      - "우회"
      - "workaround"
      - "fallback"
      - "monkey-patch"
      - "version-pin"
      - "skip module"
      - "bypass"
      - "ignore warning"
    when_to_use: 정공법 시도가 호환·환경·라이브러리 한계로 실패해 다른 방법으로 우회하거나, 표준 의도와 다른 임시 형태로 진행하기로 결정한 즉시 참조한다.
  constraints:
    allowed_tools: ["Read", "Write", "Edit", "Glob"]
    risk_level: high
  artifacts:
    inputs:
      - "본래 의도한 절차·표준 패턴"
      - "정공법 시도가 실패한 에러·증거(log, traceback, OOM, ...)"
      - "채택한 우회 방식 본문"
    outputs:
      - "evals/workarounds/<W-id>.md (append-only ledger entry)"
      - "evals/workarounds/_index.md 갱신"
      - "관련 가설/일지/체크리스트의 cross-link"
---

# Workaround Tracking Skill — Research Track Template

본 skill은 **정공법으로 해결 못 한 사항을 결론에서 잃지 않게 만드는** 절차이다. ML/연구 프로젝트는 라이브러리·하드웨어 호환 한계로 자주 우회를 사용하지만, 결론 시점에 우회 사실이 보고되지 않으면 (a) 외부 재현이 깨지고, (b) 본격 학습으로 확장될 때 같은 우회가 재현되지 않을 수 있으며, (c) 학습 결과를 본래 패턴으로 잘못 해석할 수 있다.

본 skill은 [`01-instructions.md §2`](../01-instructions.md)의 문장 형식 규칙을 그대로 따른다.

---

## 1. 목적

본 skill은 다음을 보장한다.

- 정공법으로 해결 못 한 우회·간접 해결을 발견 즉시 ledger에 기록한다(append-only).
- 우회의 재현율 영향을 분류해 본격 학습/외부 공개 진입 전 처리 의무를 부여한다.
- 결론·외부 공개 시점에 누락 없이 통합 보고한다.
- 재현율 평가([`reproducibility-checklist.md §2-5`](./reproducibility-checklist.md))에 open critical workaround 점검 항목을 강제한다.

### 1-1. 본 skill이 맡지 않는 것

- 우회의 기술적 해소 — 별도 작업으로 진행. 본 skill은 "발견·기록·분류·결론 보고"만 맡는다.
- 가설 본문 수정 — [`hypothesis-registry.md §4`](./hypothesis-registry.md) 사용자 승인 게이트.
- 시각화·일지 — [`research-journal.md`](./research-journal.md) 보조.

---

## 2. 적용 대상

다음 중 하나에 해당하면 우회로 간주해 즉시 등록.

- **버전·라이브러리 호환 우회** — 표준 패턴이 라이브러리 버전 미지원·미호환으로 실패해 비표준 패턴으로 대체.
- **하드웨어 한계 우회** — VRAM·메모리 한계로 swap·offload·precision 변경 등 환경 의존적 절충.
- **기능 누락 우회** — 본래 사용하려던 라이브러리 기능이 없거나 깨져 직접 구현 또는 다른 도구 대체.
- **경고 무시** — `UserWarning`/`FutureWarning`을 명시적으로 수용한 채 진행.
- **임시 데이터 정책 변경** — sanity check·디버깅 목적의 데이터 축소/단순화(가설에 명시되어 있어도 ledger 기록).
- **기능적 동등성 가정 채택** — A 대신 B를 사용하면서 "결과 의미는 같다"고 단정한 모든 결정.

### 2-1. 등록 면제 조건

- 단순 코드 리팩터링 또는 명시적 hyperparam 튜닝(uses별 trade-off 문서화 시).
- AGENTS.md §3·§4 절대 규칙·경로 분기로 이미 표준화된 절차의 정상 적용.
- 가설 사전 정의의 임의 분기 중 명시적으로 선택된 분기(가설 본문에 이미 변형이 등록된 경우).

### 2-2. 금지 규칙

- 우회를 발견하고도 ledger 등록 없이 진행하지 마라.
- "잘 작동하니까 괜찮다"는 이유로 critical 등급 우회를 본격 학습으로 확장하지 마라.
- 우회 항목의 등급을 임의로 낮추지 마라.

---

## 3. 분류 — 재현율 영향 등급

본 등급은 **외부에서 같은 결과를 재현할 수 있는가**를 기준으로 부여한다.

### 3-1. critical

다음 중 하나라도 충족:

- 학습 dynamics 자체를 본질적으로 바꿈(trainable params, optimizer state, gradient flow 등).
- 평가 결과 해석을 바꿈(예: catastrophic forgetting 가능성, base model identity 변경).
- 외부 환경에서 같은 결과 재현이 거의 불가능.

본격 학습/외부 공개 진입 **전 반드시 해소** 또는 가설 supersede로 명시. 미해소 상태에서 외부 결과 인용 금지.

### 3-2. material

다음 중 하나라도 충족:

- 환경·도구 의존성으로 결과 시간·방식·정밀도가 달라짐(수치는 같음).
- adapter 저장·로드의 호환성 영향.
- 메모리·연산 효율의 비표준 처리.

본격 학습/외부 공개 진입 **전 검증 + 일지 명시 의무**. 환경 매트릭스(VRAM·OS·driver·라이브러리 버전) 호환성 점검 후 인용 가능.

### 3-3. low

다음 중 하나에 해당:

- 시각화·디버그 도구 변경(학습 결과에 영향 없음).
- 사전 등록 가설의 명시적 임시 정책.
- 연구 진행 비용을 줄이기 위한 명시적 trade-off 선택(가설 변경 이력에 인용된 것).

가설 변경 이력에 기록 후 진행 가능. 결론 시점 보고 의무는 동일.

---

## 4. 식별·등록 절차

1. **즉시 인지** — 정공법 시도가 실패해 다른 방법을 채택한 시점에 본 skill 트리거.
2. **W-id 부여** — `evals/workarounds/_index.md`에서 다음 순번 확인. `W-YYYY-NNN` 형식.
3. **ledger 파일 생성** — `evals/workarounds/<W-id>.md`에 §5 형식으로 작성.
4. **분류** — §3 등급 기준으로 critical/material/low 부여.
5. **본격 학습 진입 전 의무 명시** — 해소·검증·재시도 중 하나.
6. **cross-link** — 관련 가설·일지·model_card·5단계 리포트에 W-id 인용.
7. **`_index.md` 갱신** — 새 entry 한 줄 추가.
8. **사용자 보고는 결론 시점**에 수행 — 발견 즉시 별도 보고 불필요. 단 critical은 발견 시점에 사용자에게 알릴 가치 있음.

---

## 5. ledger 파일 형식

```markdown
---
w_id: W-YYYY-NNN
status: open | resolved | accepted-permanent
discovered_at: <ISO 8601>
discovered_during: <H-id 또는 작업 단계>
severity: critical | material | low
domain: <도메인 분류 — 예: quantization | peft | tokenizer | viz | env | data-policy | adapter | other>
related_hypotheses: [<H-id 목록>]
related_artifacts: [<산출물 경로 목록>]
resolution_target: <해소 시점·범위>
---

# W-YYYY-NNN — <한 줄 요약>

## 본래 의도

<원래 어떻게 하려 했는가. 표준 패턴 인용.>

## 정공법 시도 결과

<무엇을 시도했고 어떻게 실패했는가. 에러 메시지·로그 인용.>

## 채택한 우회 방식

<어떻게 우회했는가. 코드 위치·의존 인용.>

## 재현율 영향 분석

<왜 critical/material/low인가. §3 기준으로 명시.>

## 본격 학습 진입 전 의무

<해소·검증 절차. 누가·언제·어떻게.>

## 영향받는 가설·산출물

- 가설: <H-id>
- model_card: <경로>
- 일지: reports/<YYYY-MM-DD>.md

## 변경 이력 (append-only)

- <ISO 8601> — 등록(severity=..., status=open).
- <ISO 8601> — status=resolved 또는 accepted-permanent (사용자 승인 후).
```

### 5-1. status 정의

- `open` — 등록되었으나 아직 해소 안 됨. 기본값.
- `resolved` — 해소됨(정공법 복귀 또는 호환 가능한 대체 도입). 변경 이력에 해소 방식·시점·증거 인용.
- `accepted-permanent` — 영구 수용. 사유와 본격 학습/외부 공개 영향 분석 동봉. 사용자 승인 필요.

`open` → `resolved`/`accepted-permanent` 전환은 **사용자 승인 게이트**(critical 필수, material 권장).

---

## 6. 결론·외부 공개 시점 보고 의무

### 6-1. 5단계 비교 리포트 부록

[`evalSample/eval-compare.md §6`](../evalSample/eval-compare.md) 5단계 리포트에 부록 §"우회·간접 해결 ledger 인용"을 둔다(필수).

```markdown
## 부록 — 우회·간접 해결 ledger 인용

본 비교 기간 활성·해소된 우회 항목:

| W-id | severity | status | 한 줄 요약 | 본격 학습 진입 전 의무 |
|---|---|---|---|---|
| <W-id> | <등급> | <상태> | <요약> | <의무> |

**critical 미해소 항목이 있으면 본 비교 결과를 외부에 인용하지 마라** — <H-id>의 supports/contradicts 판정에 우회 영향이 분리되지 않은 상태이다.
```

### 6-2. 재현성 체크리스트 통합

[`reproducibility-checklist.md §2-5`](./reproducibility-checklist.md) #15 항목에서 강제: `status: open` + `severity: critical`인 항목이 0개여야 외부 공개 가능.

### 6-3. 일지 통합

[`research-journal.md §3-4-1`](./research-journal.md) "우회 발견" 절에 그 날 등록·해소된 W-id를 한 줄씩 인용.

### 6-4. 가설 supersede 시 인용 의무

[`hypothesis-registry.md §4`](./hypothesis-registry.md) 사용자 승인 게이트의 supersede draft 본문에 다음 항목 추가:

> **우회 ledger 점검** — 본 가설과 관련된 우회 항목(`related_hypotheses`)을 모두 나열하고, 각 항목의 처리 계획(해소·영구 수용)을 명시한다. open critical이 있으면 supersede가 그것을 해소하는지 답한다.

---

## 7. 금지 규칙

- 등급을 결과에 맞춰 임의로 낮추지 마라(예: critical → material).
- 우회 ledger를 외부 공개 직전에 일괄 생성하지 마라(append-only 의미 손실).
- 동일 W-id를 재사용하지 마라(해소된 entry도 보존).
- ledger 본문을 사후 수정하지 마라(append-only). 변경 사항은 변경 이력에 새 줄로 추가.
- 5단계 리포트의 "부록" 절을 빈 채로 두지 마라(없으면 "(없음)"으로 표기).

---

## 8. 유지보수 규칙

- §3 등급 기준을 변경했으면 → 기존 entry의 등급 재분류는 새 entry(append) 또는 변경 이력에 인용. 본문 직접 수정 금지.
- 새 도메인(`domain` 필드의 enum)을 추가했으면 → §5 frontmatter 예시 갱신.
- 본 skill의 등록 결과가 04-evaluation.md §6의 회귀 분류에 반복 인용되면 → AGENTS.md §3-7 자가 수정 메타 규칙으로 새 절대 규칙 검토.

<IMPORTANT>
우회 기록은 "선택지"가 아니라 **연구 정직성·재현율의 1차 방어선**이다. 본 ledger 없이는 외부 공개 결과의 재현 책임이 잠재적 우회의 누적 효과로 끊긴다. 본 skill의 §6 결론 보고는 [`01-instructions.md §2-2`](../01-instructions.md) Role 우선순위의 1번(연구 정직성)·2번(재현성) 항목을 기술적으로 강제하는 메커니즘이다.
</IMPORTANT>
