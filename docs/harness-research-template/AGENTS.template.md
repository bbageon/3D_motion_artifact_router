# AGENTS.md — <project-name>

> 본 파일은 [`harness-research-template/AGENTS.template.md`](./AGENTS.template.md)를 본 프로젝트(연구 트랙)에 적용한 결과이다. 모든 `<...>` placeholder는 프로젝트 컨텍스트로 교체한다. 일반 SW 트랙 적용은 [`harness-template/AGENTS.template.md`](../harness-template/AGENTS.template.md)을 사용한다.

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

이 프로젝트는 `<프로젝트 한 줄 설명. 연구 도메인·데이터셋·핵심 방법 포함>`이다.

연구의 우선순위는 (1) 데이터 형식 일관성, (2) 학습/평가 재현 가능성, (3) 실험 결과 비교 가능성, (4) 편의성 순이다.

핵심 가설은 다음과 같다 (상세는 [`evals/hypotheses/`](evals/hypotheses/)에 사전 등록).

- `<H-YYYY-001>` — `<한 줄 가설>`.
- `<H-YYYY-002>` — `<한 줄 가설>`.

가설 본문 수정·status 전환은 §3-11에 따라 [`.claude/skills/hypothesis-registry/SKILL.md §4`](.claude/skills/hypothesis-registry/SKILL.md) 사용자 승인 게이트를 거친다.

기술 스택은 다음과 같다:
- 언어/런타임은 `<언어 + 버전>`이다.
- 핵심 라이브러리는 `<목록과 버전>`이다.
- 베이스 모델(있는 경우)은 `<모델 이름·hash·license>`이다.
- 빌드/패키지 도구는 `<도구 또는 "없음. 모듈 실행이 빌드 대체">`이다.
- 정형 테스트 러너는 `<도구 또는 "미도입. 검증은 round-trip + smoke + 검증 split 평가로 구성">`이다.

소스 디렉토리 구조는 다음과 같다:

- `<경로>` — `<역할 한 줄>`이다.
- ...
- [`evals/hypotheses/`](evals/hypotheses/) — 사전 등록 가설 (append-only).
- [`evals/raw/`](evals/raw/) — Collect 산출물.
- [`evals/workarounds/`](evals/workarounds/) — 우회 ledger.
- [`reports/`](reports/) — 사람-가독 연구일지.

프로파일/환경 구성은 다음과 같다:

- `<env-name>` — `<환경 차이 요약>`이다.

---

## 2. 빌드 & 실행

### 2-1. 환경 준비

`<환경 생성 + 기본 의존성 설치 명령>`.

### 2-2. 핵심 실행 명령어

`<데이터 전처리 명령>` / `<학습 명령>` / `<추론 명령>` / `<평가 명령>`을 각각 한 줄로 기록한다(시드·체크포인트 인자 포함, 그대로 재실행 가능한 형태).

### 2-3. 센서 명령어

코드·데이터·프롬프트 변경 시 수행하는 센서 명령어는 다음과 같다 (상세는 [`.claude/rules/phase/02-sensor.md`](.claude/rules/phase/02-sensor.md)).

- 정적 임포트·구문 검사: `<lint 또는 py_compile 명령>`.
- round-trip 검사 (인코더·파서 변경 시): `<encode→decode→equality 명령>`.
- smoke 추론 (학습·추론 코드 변경 시): `<quick smoke 스크립트>`.
- 충분성 지표 산출 (검증 split): `<평가기 명령>` — 임계 정책은 [`.claude/skills/sufficiency-metric/SKILL.md`](.claude/skills/sufficiency-metric/SKILL.md).

### 2-4. 평가 레이어(L4) 실행

추론·평가 산출물을 생성한 후 다음 단계로 Agent 수행 메트릭과 모델 메트릭을 함께 기록한다 (상세는 [`.claude/skills/eval-collect/SKILL.md`](.claude/skills/eval-collect/SKILL.md)). 평가는 머지를 막지 않는 관측용이며, 실패해도 커밋을 진행한다.

회귀 리포트(`evals/reports/*.md`)에 항목이 있으면 [`.claude/skills/eval-compare/SKILL.md`](.claude/skills/eval-compare/SKILL.md)의 5단계 리포트 절차에 따라 본 AGENTS.md 또는 phase 문서를 갱신한다.

---

## 3. 절대 규칙

> **IMPORTANT:** 이 섹션의 모든 항목은 위반 시 (a) 실험 결과 비교 가능성 파괴, (b) 학습/추론 재현 불가, (c) 가설 평가 근거 오염 중 하나를 유발한다. 예외 없이 준수하라.

### 3-1. 데이터 형식 일관성

`<프로젝트 데이터 스키마(JSON 구조·관절·좌표계 등)>`를 깨지 마라. 위반 시 변환기·학습기·추론기·평가기 사이에서 키가 어긋나 평가 결과가 무의미해진다.

### 3-2. 토큰·표현 포맷

`<프로젝트 토큰/표현 포맷 정의>`이다. 인코더와 파서 양쪽을 동시에 수정한다 — [`.claude/skills/change-obligation/SKILL.md`](.claude/skills/change-obligation/SKILL.md).

### 3-3. 학습용 프롬프트·입력 형식

`<프로젝트 학습 입력 형식 정의>`. 형식 변경 시 변환기·학습기 래퍼·추론기 입력 형식 세 곳을 동시 갱신한다.

### 3-4. 베이스 모델·체크포인트 경로

`<프로젝트 베이스 모델/체크포인트 경로 규약>`. 학습은 항상 `<양자화/precision 정책>`로 수행한다.

### 3-5. 실험 트랙 분리

비교 대상 트랙(예: `delta`/`absolute`, `scaled`/`unscaled`)은 결과 디렉토리·파일명으로 명확히 분리한다. 한 트랙 결과를 다른 트랙 평가기에 입력하지 마라.

### 3-6. 평가 기록 의무

추론·평가를 수행했으면 → 다음 항목을 동시 기록한다.
- 사용한 모델/체크포인트 경로 + `model_card.json` hash.
- 사용한 테스트 split 경로 + `_meta` hash.
- 충분성 지표 (parse success·calibration·도메인 지표).
- 모델 메트릭 (도메인별: MPJPE/MSE/F1/AUROC 등).
- (delta·누적 표현인 경우) 절대 좌표/원시 단위로 복원한 후의 지표.

위 항목 중 하나라도 누락된 평가 결과는 비교 근거로 인용하지 마라. 기록은 (a) 기계 처리용 raw record (`evals/raw/<timestamp>.json` — [`.claude/skills/eval-collect/SKILL.md`](.claude/skills/eval-collect/SKILL.md)) + (b) 사람-가독 연구일지 (`reports/<YYYY-MM-DD>.md` — [`.claude/skills/research-journal/SKILL.md`](.claude/skills/research-journal/SKILL.md))에 동시 보존한다.

### 3-6-1. 연구일지 작성 의무

핵심 경로([`.claude/rules/phase/02-sensor.md §2`](.claude/rules/phase/02-sensor.md))에 속한 파일을 변경했거나, 학습·추론·평가를 1회 이상 실행했거나, 신규 트랙·체크포인트·평가 지표를 도입했거나, 회귀·개선 신호를 관찰했으면 → 그 날의 `reports/<YYYY-MM-DD>.md`를 작성한다. 일지는 (a) 정량 지표(트랙별 분리), (b) 시각화 (다중 샘플 우선), (c) 실험 메타(재현 가능 명령어·시드·환경), (d) raw record cross-link 네 항목을 모두 포함한다.

### 3-7. 자가 수정 메타 규칙

동일 유형의 센서 실패가 3회 이상 반복되거나, 회귀 리포트(`evals/reports/*.md`)에 동일 회귀가 둘 이상의 연속 스냅샷에서 확인되면, 재발 방지 규칙을 본 AGENTS.md 또는 [`.claude/rules/phase/`](.claude/rules/phase/)에 추가한다. 규칙 추가·수정 시 반드시 근거 리포트 경로를 인용한다. 메트릭 근거 없이 절대 규칙을 추가하지 마라.

### 3-8. 검증 질의와 구현 지시의 분리

사용자의 검증 질의("이 부분이 맞아?", "왜 이렇게 했어?")를 구현 지시로 확장 해석하지 마라. 검증 질의에는 답변만 한다. 구현은 사용자가 명시적으로 지시했을 때만 착수한다.

### 3-9. 단일 샘플·단일 trial 결론 금지

단일 샘플 시각화나 한 trial의 지표로 모델 성능 결론을 내리지 마라. 최소한 검증 split 전체 또는 동일 정책으로 잘라낸 다중 trial 분포 + paired test([`.claude/skills/reproducibility-checklist/SKILL.md §3-7`](.claude/skills/reproducibility-checklist/SKILL.md))를 사용한다.

### 3-10. 데이터·모델 버전 관리 의무

새로 생성하는 데이터 산출물과 모델 체크포인트는 [`.claude/skills/data-versioning/SKILL.md §2`](.claude/skills/data-versioning/SKILL.md)의 metadata 규약을 따른다 (`_meta` 레코드·`model_card.json` 동봉). 학습·추론·평가 시작 전 정합성 검증을 통과해야 한다 — metadata 불일치 상태에서 결과를 산출하지 마라.

### 3-11. 가설 사전 등록과 보수적 수정

연구 가설은 [`.claude/skills/hypothesis-registry/SKILL.md`](.claude/skills/hypothesis-registry/SKILL.md)에 따라 `evals/hypotheses/<h_id>.md`로 사전 등록하고, 본문은 영구 보존(append-only)한다. `status` 전환(`active` → `supported|rejected|superseded|withdrawn`)과 가설 supersede에 따른 새 가설 promote는 **사용자 승인 게이트**를 거친다. Agent가 단독으로 진행하지 마라.

### 3-12. 외부 공개 시 재현성 체크리스트

논문·발표·외부 README 등으로 결과를 공개하기 전 [`.claude/skills/reproducibility-checklist/SKILL.md §2`](.claude/skills/reproducibility-checklist/SKILL.md)의 14+1개 항목을 점검한다. 평가 지표는 동일 skill §3의 정의를 단일 출처로 인용한다.

### 3-13. Negative result 보존 의무

폐기·발산·기각된 시도(parse 0%, OOM, 학습 발산, 가설 contradicts 등)는 일지의 "실패한 시도" 절([`.claude/skills/research-journal/SKILL.md §3-5`](.claude/skills/research-journal/SKILL.md))에 명시한다. raw record에는 `negative_result: true`로 표기한다.

### 3-14. 우회·간접 해결 기록 의무 (workaround ledger)

정공법으로 해결 못 하고 우회·간접 해결한 사항은 발견 즉시 [`.claude/skills/workaround-tracking/SKILL.md §4`](.claude/skills/workaround-tracking/SKILL.md)에 따라 `evals/workarounds/<W-YYYY-NNN>.md`에 등록한다(append-only). `status: open` + `severity: critical`인 항목이 1개라도 있으면 외부 공개를 보류한다.

---

## 4. 경로별 조건 분기

본 섹션은 특정 파일 변경 시 추가 검증·재생성을 강제하는 조건이다. 모든 항목은 `<변경 조건> → <대응 검증>` 형식이다.

### 4-1. `<데이터 전처리 스크립트 경로>`

변경 시 → `<전처리 산출물 디렉토리>`를 재생성하고, 학습 JSONL 변환을 다시 실행해 새 산출물의 `_meta`가 정합되는지 확인한다.

### 4-2. `<토크나이저 확장 스크립트>`

변경 시 → 토크나이저 확장본을 재생성하고, 신규 토큰이 변환기·학습기·추론기에서 동일하게 처리되는지 확인한다.

### 4-3. `<인코더/파서 스크립트>`

토큰 포맷 또는 프롬프트 템플릿을 변경했으면 → 동일 변경을 대응 파서·평가기에 동시 적용한다. round-trip 검사(임의 입력 → encode → decode → equality)를 수행한다.

### 4-4. `<학습 스크립트>`

`<핵심 hyperparam(예: max-length·epochs·learning-rate·target modules)>`을 변경했으면 → 변경 사유와 비교 baseline을 커밋 메시지에 기록한다. 디폴트 값을 무근거로 변경하지 마라.

### 4-5. `<평가 스크립트>`

평가 지표 정의(계산식)를 변경했으면 → 동일 변경을 모든 트랙 평가기에 동시 적용하고, [`.claude/skills/reproducibility-checklist/SKILL.md §3`](.claude/skills/reproducibility-checklist/SKILL.md) 지표 정의 사전을 갱신한다. `aggregation_rule_version`을 올린다.

### 4-6. `<의존성 매니페스트(requirements.txt 등)>`

변경 시 → 환경에서 의존성 재설치를 먼저 수행하고, 변경된 패키지가 학습/추론 결과의 결정성에 영향을 주는지 확인한다.

---

## 5. 실패 대응 & 흔한 실수

### 5-1. `<에러 시그니처>`

`<에러 패턴>`이 발생했으면 → `<원인>`을 확인하고 `<대응 절차>`를 수행한다.

(이 절은 프로젝트별 흔한 환경 이슈·의존성 충돌·하드웨어 한계로 채운다. 우회를 채택해야 하면 §3-14에 따라 ledger에 등록한다.)

---

## 6. 위험 행동

### 6-1. 거대 산출물 git 커밋

학습 JSONL(GB 단위)·체크포인트·원본 데이터셋을 git에 커밋하지 마라. `.gitignore`에 등재되어야 한다.

### 6-2. 평가 결과 재기록

기존 `reports/<날짜>.md`의 수치를 새 실험 결과로 덮어쓰지 마라. 트랙·체크포인트별로 새 파일·새 행을 추가한다.

### 6-3. 임계값 완화로 회귀 회피

평가 지표 회귀가 발생했을 때 임계값을 낮추는 방식으로 통과시키지 마라. 회귀는 원인 수정 또는 절대 규칙 강화로만 해소한다.

### 6-4. 가설 사후 수정 (HARKing)

결과를 본 뒤 가설 본문을 사후 수정해 supports로 보이게 만들지 마라(Kerr 1998 HARKing). 등록된 가설은 append-only이며, 변경은 새 draft → 사용자 승인 → 새 가설(`<new_h_id>`)로만 가능하다.

### 6-5. metadata 우회로 산출물 동질화

서로 다른 정책으로 만들어진 산출물을 동일 파일명으로 덮어쓰거나 metadata를 사후 편집해 hash를 맞추지 마라. silent invalidation은 §3-10 위반이며, 이후 모든 비교를 무의미하게 만든다.

### 6-6. 우회 미기록·등급 임의 하향

정공법으로 해결 못 한 사항을 ledger에 등록하지 않고 진행하거나, 등급(critical/material/low)을 결과에 맞춰 임의로 낮추지 마라.

### 6-7. 단일 trial로 가설 평가

단일 trial 결과로 가설 supports/contradicts를 단정하지 마라. 본 판정은 [`.claude/skills/eval-compare/SKILL.md §6`](.claude/skills/eval-compare/SKILL.md) 5단계 리포트에서만, 충분한 표본(trial ≥ 20·snapshot ≥ 2) 하에서만 수행한다.

---

## 7. 디렉토리별 상세 규칙

본 섹션은 디렉토리 단위의 책임을 정의한다.

### 7-1. `<프로젝트 코드 디렉토리>`

`<역할 한 줄>`. 도메인 모듈이나 헬퍼 라이브러리는 `<sub-dir>`에 둔다.

### 7-2. `evals/`

평가 파이프라인의 기계 처리 산출물 디렉토리이다.

- `evals/raw/<timestamp>_<task_id>_<trial_id>.json` — Collect 산출물.
- `evals/normalized/`, `evals/graded/`, `evals/snapshots/{daily,weekly}/`, `evals/reports/<period>.md` — 후속 단계 산출물.
- `evals/hypotheses/<h_id>.md` — 사전 등록 가설 (append-only).
- `evals/hypotheses/_index.md` — 가설 인덱스.
- `evals/workarounds/<W-id>.md` — 우회 ledger (append-only).
- `evals/workarounds/_index.md` — 우회 인덱스.

### 7-3. `reports/`

연구일지 디렉토리이다. 일자별 일지(`reports/<YYYY-MM-DD>.md`)·시각화(`reports/figures/<YYYY-MM-DD>/`)·체크리스트(`reports/checklists/<period>_<scope>.md`)가 들어간다.

### 7-4. `.claude/`

Claude Code 운영 자산이다. `rules/phase/`는 phase 판단 규칙, `skills/`는 실행 절차·평가 파이프라인을 담는다.

---

## 8. 참고 전용 (아래 섹션)

위(1~7)는 모든 변경에 필수로 따르는 구간이다. 아래(9~10)는 특정 구현 작업 시에만 참조하는 구간이다.

---

## 9. 구현 레시피

### 9-1. 신규 가설 등록

1. `evals/hypotheses/_index.md`에서 다음 h_id를 확인한다.
2. `evals/hypotheses/<h_id>.md`를 [`hypothesis-registry §2-3`](.claude/skills/hypothesis-registry/SKILL.md) 형식으로 작성한다.
3. `사전 정의`·`기각 조건`·`표본 요건`을 모두 채운다.
4. `_index.md`에 한 줄 추가한다.

### 9-2. 신규 데이터 산출물 생성

1. 산출물 직전에 `git rev-parse HEAD`·`git status --porcelain`로 dirty 여부를 확인한다.
2. dirty 상태이면 커밋 후 산출물 생성, 또는 `<commit>-dirty` 표시를 metadata에 포함한다.
3. [`data-versioning §2`](.claude/skills/data-versioning/SKILL.md)의 스키마를 산출물에 동봉한다.
4. 다음 단계 산출물에서 참조할 sha256을 기록한다.

### 9-3. 신규 평가 지표 추가

1. 도메인별 정의·계산식·해석을 [`reproducibility-checklist §3`](.claude/skills/reproducibility-checklist/SKILL.md) 지표 정의 사전에 추가한다.
2. 평가기·raw record 스키마·snapshot 스키마에 필드를 추가한다.
3. 직전 스냅샷과 호환되지 않으면 `aggregation_rule_version`을 올린다.

### 9-4. 우회 발견 시 등록

1. 정공법 시도 실패가 확정되면 [`workaround-tracking §4`](.claude/skills/workaround-tracking/SKILL.md)에 따라 즉시 W-id를 부여한다.
2. ledger 파일을 §5 형식으로 작성한다(`severity`·`status`·`resolution_target` 포함).
3. 관련 가설·일지·model_card에 W-id를 cross-link한다.
4. critical인 경우 사용자에게 발견 시점에 알린다.

### 9-5. 외부 공개 전 점검

1. [`reproducibility-checklist §2`](.claude/skills/reproducibility-checklist/SKILL.md) 14+1 항목을 점검한다.
2. `evals/workarounds/_index.md`에 `status: open` + `severity: critical`이 0개임을 확인한다.
3. `reports/checklists/<period>_<scope>.md`에 점검 결과를 기록한다.

---

## 10. 아키텍처 제약 & 네이밍

### 10-1. 파일 네이밍

`<프로젝트 파일/모듈 네이밍 규약>`. 트랙 모듈은 `<track>/<순번>_<역할>_<track>.<ext>` 형식을 권장한다.

### 10-2. 데이터 포맷

- 내부 데이터 스키마: §3-1 참조.
- 학습 JSONL: 첫 줄 `_meta` + 이후 학습 레코드 (학습기는 첫 줄을 skip한다).
- 체크포인트 디렉토리: `model_card.json` 동봉.

### 10-3. 결과 파일 분리

- 트랙별·체크포인트별·split별로 결과 파일을 분리한다.

### 10-4. 참고 문서

- 하네스 4계층 원본: [`docs/harness-research-template/01-instructions.md`](docs/harness-research-template/01-instructions.md), [`02-sensor.md`](docs/harness-research-template/02-sensor.md), [`03-test.md`](docs/harness-research-template/03-test.md), [`04-evaluation.md`](docs/harness-research-template/04-evaluation.md).
- 사전 등록 가설: [`evals/hypotheses/_index.md`](evals/hypotheses/_index.md).
- 평가 지표 정의 사전: [`.claude/skills/reproducibility-checklist/SKILL.md §3`](.claude/skills/reproducibility-checklist/SKILL.md).
- 우회 ledger: [`evals/workarounds/_index.md`](evals/workarounds/_index.md).
