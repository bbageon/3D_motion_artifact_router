# 02. 센서(Sensor) 레이어 — Research Track

본 문서는 [`../harness-template/02-sensor.md`](../harness-template/02-sensor.md)의 SW 트랙 센서 레이어 원본을 **연구 트랙**에 맞춰 확장한 변형이다. baseline 감지 대상(정적 적합성·구축 가능성·행동 회귀·변경 의무) 정의·게이트 계층·판정 의미는 SW 트랙 원본을 그대로 상속한다. 본 문서는 차이만 기술한다.

본 문서가 추가하는 것은 다음 다섯 가지이다.

- **확장 감지 대상의 연구 트랙 구체화** — 도메인 충분성 지표·결정성 이탈·실험 정합성.
- **연구 트랙 핵심 경로 정의** — 변환기·인코더·파서·학습기·평가기·가설 등록·우회 ledger를 핵심 경로로 둔다.
- **격상 조건의 연구 트랙 baseline** — 토크나이저·토큰 포맷·평가 지표 변경 시 자동 격상.
- **충분성 지표의 도메인 의존성** — line coverage가 적용되지 않는 ML/통계 파이프라인은 [`skillSample/sufficiency-metric.md`](./skillSample/sufficiency-metric.md)로 위임.
- **결정성 의무의 시드 정책** — 학습·추론·평가 전반에 고정 시드 의무.

본 문서는 [`01-instructions.md §2`](./01-instructions.md)의 작성 원칙을 그대로 따른다.

---

## 1. 감지 대상 (연구 트랙 확장)

SW 트랙 원본의 4 baseline(정적 적합성·구축 가능성·행동 회귀·변경 의무)을 본 트랙에서는 다음과 같이 구체화한다.

### 1-1. 정적 적합성

본 트랙은 정형 lint/typecheck 인프라가 없는 ML 저장소에도 적용 가능해야 하므로 다음을 baseline으로 둔다.

- 변경된 `.py` (또는 해당 언어 파일)의 임포트/구문 검사 — `python -m py_compile <파일>` 또는 동등 명령.
- 정형 lint·typecheck 도구가 도입되면 → 본 절을 갱신한다.

### 1-2. 구축 가능성

- 베이스 모델·토크나이저·체크포인트의 로드 성공 여부.
- 변환기·학습기·추론기·평가기 사이의 토크나이저 vocab id 일치.

### 1-3. 행동 회귀

본 트랙은 행동 회귀를 두 부류로 분리한다.

- **포맷 회귀** — 토큰 포맷·프롬프트 포맷·데이터 스키마의 round-trip 일관성. 임의 입력 → encode → decode → equality 비교.
- **모델 회귀** — 충분성 지표(parse success·calibration·도메인 지표)와 모델 메트릭(MPJPE/MSE/F1/AUROC 등)의 직전 스냅샷 대비 회귀.

### 1-4. 변경 의무

토큰 포맷·프롬프트 포맷·데이터 스키마 변경이 변환기·파서·평가기 중 한 곳에만 적용된 상태이다. 신규 트랙·신규 평가 지표·신규 핵심 스크립트도 의무 대상이다. 세부 절차는 [`changeObligationSample.md`](../harness-template/changeObligationSample.md)을 그대로 상속한다.

### 1-5. 확장 감지 대상

- **구조 적합성** — 트랙별 모듈 분리·환경 분리(예: 학습 환경 vs 데이터 전처리 환경) 위반을 수동 리뷰로 감지한다.
- **시험 충분성** — 본 트랙은 line coverage가 아니라 **도메인 충분성 지표**(parse success rate·calibration·도메인 임계 합격률)를 사용한다. 정의·임계·복구 절차는 [`skillSample/sufficiency-metric.md`](./skillSample/sufficiency-metric.md)에 위임한다.
- **결정성 이탈(flaky)** — 동일 시드·동일 입력에서 결과가 달라지는 상태. 재현·격리는 [`../harness-template/skillSample/flaky-handling.md`](../harness-template/skillSample/flaky-handling.md)을 그대로 상속한다.
- **실험 정합성** — 산출물의 metadata와 현재 코드 정의가 일치하는지(silent invalidation 차단). 검증 절차는 [`skillSample/data-versioning.md §3`](./skillSample/data-versioning.md)에 위임한다.

---

## 2. 핵심 경로 정의 (연구 트랙 baseline)

다음을 **핵심 경로(core path)** baseline으로 지정한다. 충분성 지표와 변경 의무는 본 목록을 우선 대상으로 삼는다.

- 데이터 전처리 스크립트 — 데이터 포맷의 단일 진입점. 윈도·joint·split 정책의 원본.
- 토크나이저/특수 토큰 확장 스크립트 — 모델·토크나이저 계약의 원본.
- 인코더/파서 스크립트 — 토큰 포맷·프롬프트 템플릿의 원본.
- 학습 스크립트 — 학습 절차의 원본.
- 추론기·평가기 — 모델 회귀를 관찰하는 출구.
- 가설 디렉토리 (`evals/hypotheses/`) — append-only. 본 디렉토리의 변경은 [`skillSample/hypothesis-registry.md §4`](./skillSample/hypothesis-registry.md) 사용자 승인 게이트의 대상이다.
- 우회 ledger 디렉토리 (`evals/workarounds/`) — append-only. 변경은 [`skillSample/workaround-tracking.md`](./skillSample/workaround-tracking.md)에 위임.

본 목록은 적용본의 디렉토리 구조에 맞춰 구체화한다. AGENTS.md §7 디렉토리 규칙과 일관되게 유지한다.

---

## 3. 게이트 계층 (연구 트랙 격상 조건)

SW 트랙의 2-tier 계층(빠른 국소 게이트 / 전체 게이트)을 그대로 사용한다. 본 트랙은 다음 격상 조건을 추가한다.

### 3-1. 빠른 국소 게이트

- 변경된 파일 임포트/구문 검사.
- 변환기·파서를 변경했으면 → round-trip 검사(임의 입력 32샘플 정도).
- 추론기·학습 코드를 변경했으면 → quick smoke 추론(8~32 샘플) — 충분성 지표(parse success·domain pass rate)가 0%이면 hard fail.
- 변경 의무 판정.
- 산출물 metadata 정합성 확인([`skillSample/data-versioning.md §3`](./skillSample/data-versioning.md)).

빠른 국소 게이트는 precise 우선이다. 약간의 누락을 허용한다.

### 3-2. 전체 게이트

- 핵심 경로 변경을 포함한 커밋이면 → 검증 split 평가를 1회 이상 수행한다(트랙별 분리).
- 평가 결과 기록 의무([`AGENTS.template.md §3-6`](./AGENTS.template.md))를 충족한다.
- 충분성 지표가 임계값을 넘는지 [`skillSample/sufficiency-metric.md §2`](./skillSample/sufficiency-metric.md)로 판정.
- 산출물 metadata 정합성 검증.
- 우회 발견 즉시 ledger 등록 확인([`skillSample/workaround-tracking.md §4`](./skillSample/workaround-tracking.md)).

전체 게이트는 safe 우선이다. 누락을 허용하지 않는다.

### 3-3. 국소 → 전체 격상 조건 (연구 트랙 baseline)

다음 중 하나라도 해당하면 국소 게이트를 전체 게이트로 격상한다.

- 토크나이저 확장 스크립트를 변경했다.
- 인코더·파서의 토큰 포맷·프롬프트 템플릿을 변경했다.
- 데이터 전처리의 윈도·joint·split 정책을 변경했다.
- 학습 스크립트의 핵심 hyperparam (max-length·epochs·learning-rate·LoRA target 등)을 변경했다.
- 평가 지표 정의(계산식)를 변경했다.
- 가설 디렉토리(`evals/hypotheses/`)에 변경이 있다 — 이 경우 동시에 [`skillSample/hypothesis-registry.md §4`](./skillSample/hypothesis-registry.md) 사용자 승인 게이트 적용.

**IMPORTANT:** 국소 게이트만 통과한 상태로 핵심 경로 변경을 커밋하지 마라. 전체 게이트 통과가 커밋의 필요조건이다.

---

## 4. 판정 의미 (연구 트랙 baseline)

SW 트랙의 3 판정(hard fail / soft fail / informational)을 그대로 사용한다. 본 트랙의 추가 baseline은 다음과 같다.

- **hard fail** — round-trip 불일치, smoke 0% 파싱, 산출물 metadata 불일치, AGENTS.md §3 절대 규칙 위반, change-obligation 미충족.
- **soft fail** — flaky 후보, 충분성 지표 임계 미달(직전 스냅샷 대비 격상 조건 도달), 트랙 비대칭 검출.
- **관측 전용(informational)** — 평가 파이프라인 산출물, 단일 시각화 결과, 토큰 길이 분포 통계.

가설 status 전환·우회 등급 변경 같은 **사용자 승인 게이트** 대상은 본 절의 판정 체계와 별도로 운영된다 — Agent가 hard/soft fail로 판단하지 않고 사용자 결정을 대기한다.

---

## 5. Agent 표준 실행 흐름 (연구 트랙)

변경 후 Agent는 다음 순서를 따른다.

1. 변경 영향 범위 식별. §3-3 격상 조건에 해당하면 처음부터 전체 게이트 흐름으로 이동.
2. 빠른 국소 게이트(§3-1) 실행.
3. 변환기·파서·평가기·추론기 중 하나 이상을 변경했으면 → change-obligation 매핑 충족 확인.
4. 실패했으면 → 판정에 따라 분기. hard fail은 즉시 수정 후 재실행. soft fail은 [`../harness-template/skillSample/flaky-handling.md`](../harness-template/skillSample/flaky-handling.md) 또는 [`skillSample/sufficiency-metric.md`](./skillSample/sufficiency-metric.md)을 수행.
5. 국소 게이트 통과 후 → 전체 게이트(§3-2) 실행.
6. 시각화·사람이 보는 산출물이 변경되었으면 → 단일 샘플로 결론 내지 말고 다중 샘플 평균을 함께 확인.
7. 전체 게이트 통과 후 → 평가 수집을 먼저 실행([`evalSample/eval-collect.md`](../harness-template/evalSample/eval-collect.md)).
8. 학습·추론·평가를 1회 이상 수행한 날이거나 핵심 경로 변경이 포함되었으면 → 연구일지 작성([`skillSample/research-journal.md`](./skillSample/research-journal.md), [`AGENTS.template.md §3-6-1`](./AGENTS.template.md)).
9. 정공법 실패 우회를 채택했으면 → [`skillSample/workaround-tracking.md §4`](./skillSample/workaround-tracking.md)에 등록.
10. 일지·raw record가 모두 있으면 → 커밋.

**IMPORTANT:** 센서가 hard fail 상태일 때 커밋하지 마라. 임계값 완화나 검사 비활성화로 회피하는 것을 금지한다.

---

## 6. 설계 원칙 (결정성 의무)

본 트랙 센서는 SW 트랙의 7 원칙(결정성·국소성·명확성·독립성·구성 가능성·추적 가능성·확장 가능성)을 그대로 상속한다. 결정성에 한해 다음 추가 의무를 둔다.

- 학습·추론·평가에는 고정 시드(`torch.manual_seed`, `numpy.random.seed`, `random.seed` 또는 도메인 동등 시드)를 사용한다.
- 시드를 바꾸면 평가 회귀 판정에 영향을 주므로 시드 변경 자체를 변경 단위로 기록한다.
- GPU/하드웨어 결정성(`torch.use_deterministic_algorithms(True)`·`CUBLAS_WORKSPACE_CONFIG` 등)은 학습 비용과 trade-off가 있으므로 평가 재현 단계에서만 강제한다.

---

## 7. 유지보수 규칙

- 본 트랙의 핵심 경로 baseline(§2)을 변경했으면 → AGENTS.md §7과 §2 센서 명령어 정의를 동시 갱신한다.
- 격상 조건(§3-3)을 확장했으면 → 근거 회귀 리포트 또는 장애 사례를 커밋 메시지에 인용한다.
- 정형 lint·typecheck 도구를 도입했으면 → §1-1의 정적 적합성 절과 AGENTS.md §2-2의 센서 명령어를 동시 갱신한다.
- 본 문서 용어가 [`03-test.md`](./03-test.md)·[`04-evaluation.md`](./04-evaluation.md)·하위 skill과 불일치하면 → 본 문서를 지도의 원본으로 간주하고 하위 문서를 갱신한다.
