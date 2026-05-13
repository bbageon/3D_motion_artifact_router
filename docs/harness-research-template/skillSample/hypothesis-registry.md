---
description: 연구 가설을 evals/hypotheses/<h_id>.md로 사전 등록하고, 사후 수정·폐기는 사용자 승인 게이트를 거쳐 보수적으로만 허용하는 skill이다. HARKing(Hypothesizing After Results are Known — Kerr 1998) 차단을 목적으로 한다.
metadata:
  scope:
    paths:
      - "evals/hypotheses/**"
      - "AGENTS.md"
  activation:
    keywords:
      - "가설 등록"
      - "hypothesis"
      - "pre-registration"
      - "가설 수정"
      - "가설 폐기"
      - "HARKing"
    when_to_use: 새 연구 가설을 등록하거나, 결과를 보고 기존 가설을 수정·폐기하려 할 때 참조한다. 가설 변경은 사용자 승인 없이 진행하지 않는다.
  constraints:
    allowed_tools: ["Read", "Write", "Edit", "Glob"]
    risk_level: high
  artifacts:
    inputs:
      - "AGENTS.md §1 핵심 가설"
      - "회귀 리포트 (evals/reports/<period>.md)"
      - "스냅샷 (evals/snapshots/{daily,weekly}/*.json)"
    outputs:
      - "evals/hypotheses/<h_id>.md (등록 가설)"
      - "evals/hypotheses/<h_id>.draft.md (수정/폐기 제안 — 사용자 승인 대기)"
      - "evals/hypotheses/_index.md (가설 인덱스)"
---

# Hypothesis Registry Skill — Research Track Template

본 skill은 연구 가설의 **사전 등록(pre-registration)** 과 **사후 수정·폐기의 보수적 게이트**를 운영한다. Kerr 1998(*HARKing: Hypothesizing After the Results are Known*) 이래 HARKing은 결과의 통계적 신뢰성을 파괴하는 가장 흔한 결함으로 보고된다. Lipton & Steinhardt 2018(*Troubling Trends in ML Scholarship*)·Ioannidis 2005도 같은 결함을 ML/통계 분야에서 광범위하게 보고한다. 본 skill은 가설을 **append-only**로 보존하고, 변경은 사용자 승인을 거친 새 레코드로만 가능하게 한다.

본 skill은 [`01-instructions.md §2`](../01-instructions.md)의 문장 형식 규칙을 그대로 따른다.

---

## 1. 목적

본 skill은 다음을 보장한다.

- 모든 활성 가설이 단일 위치(`evals/hypotheses/`)에 등록된다.
- 등록된 가설의 본문은 영구 보존되며, 결과를 본 뒤 사후 수정되지 않는다.
- 가설 수정·폐기·승계(supersede)는 사용자 승인 게이트를 통과해야만 적용된다.
- 평가 리포트([`evalSample/eval-compare.md`](../evalSample/eval-compare.md))의 5단계 출력 중 §1·§4·§5에서 본 가설들이 명시 인용된다.

### 1-1. 본 skill이 맡지 않는 것

- 가설의 통계적 평가(supports / contradicts / inconclusive 판정) — `eval-compare`.
- 학습/추론 메타(체크포인트의 `active_hypotheses`) 동봉 — [`data-versioning.md`](./data-versioning.md).
- 일지 본문 — [`research-journal.md`](./research-journal.md).

본 skill은 **가설의 등록·식별·수명주기 관리**만 수행한다.

---

## 2. 가설 식별·디렉토리 규약

### 2-1. h_id 부여 규칙

가설 식별자는 `H-YYYY-NNN` 형식이다. 예: `H-2026-001`.

- `YYYY`는 등록 연도.
- `NNN`은 해당 연도의 등록 순번(zero-padded 3자리).
- 한 번 부여된 h_id는 재사용하지 않는다(폐기·기각된 가설의 h_id도 회수하지 않는다).

### 2-2. 가설 파일 위치

- `evals/hypotheses/<h_id>.md` — 등록된 가설.
- `evals/hypotheses/<new_h_id>.draft.md` — 수정·폐기 제안. 사용자 승인 후 `<new_h_id>.md`로 promote.
- `evals/hypotheses/_index.md` — 가설 인덱스(전체 활성/승계/기각 목록과 관계 그래프).

### 2-3. 가설 파일 형식

```markdown
---
h_id: H-YYYY-NNN
status: active
registered_at: <ISO 8601>
parent: null
supersedes: null
related: []
domain: <도메인 분류 — 예: representation|tokenization|model|dataset|protocol>
track_scope: [<비교 트랙 목록>]
---

# H-YYYY-NNN — <한 줄 가설 제목>

## 본문 (등록 시점 그대로)

<가설 진술 — 단일 단언문>

## 사전 정의

- 측정할 지표: <예: 도메인 메트릭 + 충분성 지표>
- 임계 효과 크기: <예: 메트릭 ΔX 이상의 차이를 의미 있는 차이로 본다>
- 비교 대상: <예: 트랙 A vs 트랙 B의 지표 Y>
- 기각 조건: <예: paired Wilcoxon p > 0.05 이거나 effect size 임계 미만이면 기각>
- 표본 요건: trial ≥ <N>, n_seeds ≥ <K>.

## 등록 근거

<왜 이 가설을 세웠는가. 사전 문헌·예비 실험·이론적 근거.>

## 평가 대상 산출물

- 사용할 트랙: <트랙 목록>.
- 사용할 split: val / test.
- 사용할 체크포인트 정책: <예: 동일 step, 동일 시드 분포>.

## 변경 이력 (append-only)

- <ISO 8601> — 등록.
```

`status`는 다음 다섯 중 하나이다.

- `active` — 평가 대상.
- `supported` — 표본 충족 후 supports 판정으로 마감.
- `rejected` — 표본 충족 후 contradicts 판정으로 마감.
- `superseded` — 새 가설(`<new_h_id>`)로 대체됨.
- `withdrawn` — 사전 정의 결함·평가 불가능 등으로 자진 철회됨.

`active` 외 모든 전환은 §4 사용자 승인 게이트를 거친다.

---

## 3. 등록 절차 (신규 가설)

신규 가설은 사용자 승인 없이 등록 가능하다(append-only이며 후속 평가에 영향을 줄 뿐 기존 결과를 바꾸지 않는다).

1. `evals/hypotheses/_index.md`에서 사용 중인 h_id를 확인하고 다음 순번 결정.
2. `evals/hypotheses/<h_id>.md`를 §2-3 형식으로 작성.
3. `domain`·`track_scope`·`사전 정의`·`기각 조건`·`표본 요건`을 모두 채운다.
4. `_index.md`에 한 줄 추가.
5. 신규 가설이 AGENTS.md §1 핵심 가설을 변경하는 의미를 가지면 → 별개 절차 §4를 거친다. 신규 등록과 §1 갱신은 동시에 일어나지 않는다.

### 3-1. 금지 규칙

- `사전 정의`·`기각 조건`·`표본 요건` 중 하나라도 비워두지 마라. 가설은 **선검증 가능한 단언**이어야 한다.
- 결과를 본 뒤(평가 리포트가 출력된 뒤) 새 가설을 등록해 그 결과를 가설로 바꾸지 마라(HARKing).
- 가설 본문에 "성능이 좋아졌다" 같은 모호한 표현을 쓰지 마라. 측정 가능한 지표·임계로 진술한다.

---

## 4. 수정·폐기 절차 (사용자 승인 필수)

기존 가설의 `status`를 `active`에서 다른 값으로 전환하거나, 가설을 새 가설로 승계할 때 다음 순서를 따른다.

### 4-1. 단계

1. **수정 사유 식별** — 다음 중 어느 것에 해당하는지 명시.
   - 평가 결과가 contradicts로 판정했고 사전 정의된 기각 조건을 충족했다 → `rejected` 후보.
   - supports 판정 + 표본 요건 충족 → `supported` 후보(완결).
   - 가설의 사전 정의 자체에 결함 발견 → `withdrawn` 후보.
   - 새로운 가설로 정제·세분화할 필요 → `superseded` 후보.
2. **draft 작성** — supersede·withdrawn·rejected이면 → 새 h_id 부여 + `<new_h_id>.draft.md` 작성. 본문에 (a) 변경 사유, (b) 인용 리포트 경로, (c) 새 가설 본문(supersede인 경우), (d) 영향받는 다른 가설, (e) 관련 우회 ledger 항목의 처리 계획([`workaround-tracking.md §6-4`](./workaround-tracking.md))을 포함.
3. **사용자 승인 게이트** — 사용자에게 draft 경로와 변경 요약 제시 후 명시적 승인("진행"·"OK"·"approve" 등) 대기. 침묵·맥락 추론으로 promote하지 마라.
4. **promote** — 승인 후:
   - draft를 `<new_h_id>.md`로 rename.
   - 기존 가설 파일의 frontmatter `status`를 갱신, 본문 끝의 `변경 이력`에 한 줄 추가(본문 자체는 수정 금지).
   - `_index.md` 갱신.
5. **commit** — 가설 파일·`_index.md`·관련 phase·skill 갱신을 한 커밋에 묶는다.

### 4-2. 사용자 승인이 면제되는 경우

- 오타·서식 수정(본문 의미 보존).
- `related` 필드에 다른 h_id를 추가.
- `_index.md`의 시각적 정렬·요약 갱신.

위 외의 모든 status 전환·본문 의미 변경은 §4-1을 거친다.

### 4-3. 금지 규칙

- 등록된 가설 파일의 본문(`# H-...` 이하)을 직접 수정하지 마라. 변경은 새 draft → 사용자 승인 → 새 파일로만 가능하다.
- 사용자 승인 없이 `status`를 전환하지 마라.
- 결과 리포트가 contradicts인데 가설 본문을 사후 수정해 supports로 보이게 만들지 마라(Kerr 1998 HARKing).
- "사용자 승인을 받았다"고 가정하지 마라.

**IMPORTANT:** 본 skill의 사용자 승인 게이트는 AGENTS.md §3-11과 함께 **연구 정직성(research integrity)** 의 1차 방어선이다. 어떤 자동화도 본 게이트를 우회해서는 안 된다.

---

## 5. 평가 파이프라인과의 연결

- [`evalSample/eval-collect.md`](../harness-template/evalSample/eval-collect.md) — raw record에 `active_hypotheses` 필드.
- [`data-versioning.md §2-3`](./data-versioning.md) — `model_card.json.active_hypotheses`로 학습 시점 활성 가설 보존.
- [`evalSample/eval-compare.md`](../evalSample/eval-compare.md) — 비교 리포트 5단계 중 §1·§4·§5에서 h_id 명시 인용.
- [`research-journal.md`](./research-journal.md) — 일지에 활성 가설 ID 인용.

---

## 6. 유지보수 규칙

- 가설 파일 형식(§2-3)을 변경했으면 → 기존 가설은 그대로 두고 새 가설부터 새 형식으로 등록(append-only 원칙).
- 사용자 승인 절차(§4-1)를 변경했으면 → AGENTS.md 절대 규칙에서 본 skill을 인용하는 절을 동시 갱신.
- 본 skill의 사용자 승인 게이트가 우회되는 사례가 발견되면 → 회귀 리포트에 등재하고 eval-compare 피드백 루프로 처리.

<IMPORTANT>
가설은 "결과를 보고 다듬는 진술"이 아니라 **"결과를 보기 전에 기각 조건을 명시한 사전 등록 단언"** 이다. 사용자 승인 게이트(§4-1) 없이 status를 전환하거나 본문을 수정하면 AGENTS.md §3-7 자가 수정 메타 규칙의 메트릭 근거가 회고적 합리화(post-hoc rationalization)로 오염된다 — Kerr 1998 / Lipton & Steinhardt 2018.
</IMPORTANT>
