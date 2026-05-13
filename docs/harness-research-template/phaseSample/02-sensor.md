# 02. 센서(Sensor) Phase — `<project-name>` (Research Track Sample)

> 본 phase 문서는 [`../02-sensor.md`](../02-sensor.md)의 "02. 센서 레이어 (Research Track)" 규격을 본 프로젝트에 적용한 sample이다. 감지 대상·게이트 계층·판정 의미 정의는 상위 원본을 상속한다.

---

## 1. 감지 대상 (본 프로젝트 적용)

[`../02-sensor.md §1`](../02-sensor.md)의 baseline을 본 프로젝트 산출물에 결합한다.

- **정적 적합성** — `<lint 명령 또는 py_compile>`로 임포트·구문 검사.
- **구축 가능성** — `<베이스 모델/토크나이저 로드 명령>`이 성공하는지.
- **행동 회귀**
  - 포맷 회귀 — `<변환기/파서>` round-trip 검사.
  - 모델 회귀 — `<평가기>` 출력의 직전 스냅샷 대비 회귀.
- **변경 의무** — 토큰 포맷·프롬프트 포맷·데이터 스키마 변경의 일부 적용 상태. [`changeObligationSample`](../../harness-template/changeObligationSample.md)에 위임.

### 1-1. 확장 감지 대상

- **시험 충분성** — `<도메인 충분성 지표(예: parse success rate / domain pass rate)>`. 임계는 [`../skillSample/sufficiency-metric.md`](../skillSample/sufficiency-metric.md).
- **결정성 이탈(flaky)** — [`flaky-handling`](../../harness-template/skillSample/flaky-handling.md) 상속.
- **실험 정합성** — [`../skillSample/data-versioning.md §3`](../skillSample/data-versioning.md)에 위임.

---

## 2. 핵심 경로 정의

다음을 핵심 경로(core path)로 지정한다.

- `<데이터 전처리 스크립트 경로>` — 데이터 포맷 단일 진입점.
- `<토크나이저/특수 토큰 확장 스크립트>` — 모델·토크나이저 계약 원본.
- `<인코더 스크립트>` / `<파서 스크립트>` — 토큰 포맷·프롬프트 템플릿 원본.
- `<학습 스크립트>` — 학습 절차 원본.
- `<추론기·평가기 경로>` — 모델 회귀 관찰 출구.
- `evals/hypotheses/**` — 사용자 승인 게이트 대상.
- `evals/workarounds/**` — 우회 ledger.

본 목록은 AGENTS.md §7과 일관 유지.

---

## 3. 게이트 계층

### 3-1. 빠른 국소 게이트

- 변경된 `.py` 임포트/구문 검사: `<명령>`.
- 변환기·파서 변경 → round-trip 검사 (32샘플).
- 추론기·학습 코드 변경 → quick smoke 추론 (8~32 샘플).
- 변경 의무 판정.
- 산출물 metadata 정합성 확인.

### 3-2. 전체 게이트

- 핵심 경로 변경 포함 시 → 비교 트랙 모두의 검증 split 평가 1회 이상.
- 평가 기록 의무(AGENTS.md §3-6) 충족.
- 충분성 지표 임계 판정 (sufficiency-metric §2).
- 우회 발견 즉시 ledger 등록 확인.

### 3-3. 국소 → 전체 격상 조건

다음 중 하나라도 해당하면 격상:

- `<토크나이저 확장 스크립트>` 변경.
- `<인코더/파서>`의 토큰 포맷·프롬프트 템플릿 변경.
- `<데이터 전처리>`의 윈도·split·feature set 변경.
- `<학습 스크립트>`의 `<핵심 hyperparam>` 변경.
- 평가 지표 정의 변경.
- `evals/hypotheses/**` 변경(이 경우 hypothesis-registry §4 사용자 승인 게이트 동시 적용).

---

## 4. 판정 의미

[`../02-sensor.md §4`](../02-sensor.md)의 세 판정을 본 프로젝트 게이트에 부여한다.

- **hard fail** — 구문/임포트 에러, round-trip 불일치, smoke 0% 충분성, AGENTS.md §3 절대 규칙 위반, change-obligation 미충족, metadata 불일치.
- **soft fail** — flaky 후보, 충분성 격상 조건 도달, 트랙 비대칭.
- **관측 전용** — 평가 산출물, 단일 시각화, 분포 통계.

가설 status 전환·우회 등급 변경은 사용자 승인 게이트 대상으로, 본 절의 판정과 별도.

---

## 5. Agent 표준 실행 흐름

[`../02-sensor.md §5`](../02-sensor.md)의 10단계를 본 프로젝트에 그대로 적용한다.

---

## 6. 설계 원칙 (결정성)

- 학습·추론·평가에 고정 시드 사용: `<seed 변수 목록>`.
- 시드 변경은 변경 단위로 기록.
- GPU 결정성은 평가 재현 단계에서만 강제 (학습 비용 trade-off).

---

## 7. 유지보수 규칙

- 감지 대상·핵심 경로·게이트 구성 변경 → AGENTS.md §7과 §2 센서 명령어 정의 동시 갱신.
- 격상 조건 확장 → 근거 회귀 리포트 또는 장애 사례를 커밋 메시지에 인용.
- 정형 lint·typecheck 도입 시 → §1과 AGENTS.md §2-2 동시 갱신.
