# 03. 테스트(Test) 레이어 — Research Track

본 문서는 [`../harness-template/03-test.md`](../harness-template/03-test.md)의 SW 트랙 테스트 레이어 원본을 **연구 트랙**에 맞춰 확장한 변형이다. 분류(Unit/Integration/Contract/Regression)·오라클(reference/property/metamorphic)·더블·flaky 대응 원칙은 SW 트랙 원본을 그대로 상속한다. 본 문서는 차이만 기술한다.

본 문서가 추가하는 것은 다음 다섯 가지이다.

- **연구 트랙 테스트 형태의 매핑** — 정형 단위테스트 인프라가 없는 ML/통계 저장소에서 round-trip 검사·smoke 추론·검증 split 평가가 어떻게 분류에 매핑되는지.
- **오라클 강도의 통계적 보강** — 모델 출력에는 paired test·effect size·CI를 오라클의 보조로 둔다.
- **충분성 지표의 도메인 의존성** — line coverage 대신 도메인 충분성 지표(parse success·calibration 등) 사용.
- **재현성 검증의 결정성 의무** — 시드·환경 fixture와 결합한 재현 절차.
- **변경 영향 선정의 트랙 분리** — delta·absolute·track A·track B 같은 비교 트랙에 동시 적용.

본 문서는 [`01-instructions.md §2`](./01-instructions.md)의 작성 원칙을 그대로 따른다.

---

## 1. 테스트 분류의 연구 트랙 매핑

SW 트랙 원본의 4 부류(Unit/Integration/Contract/Regression)를 본 트랙은 다음 형태로 운영한다.

### 1-1. Unit — round-trip 검사

변환기·인코더·파서를 외부 의존 없이 검증한다.

- 형태: 임의 입력 → encode → decode → equality(부동소수 오차 1e-5 또는 도메인 적정값).
- 정형 test 디렉터리 없이도 ad-hoc 검증 또는 일회성 스크립트로 수행 가능. 검사 결과를 커밋 메시지에 인용한다.
- 정형 테스트 러너(pytest 등)가 도입되면 → 본 절을 갱신한다.

### 1-2. Integration — smoke 추론

토크나이저·모델·파서·평가기가 결합된 경계에서 동작을 검증한다.

- 형태: 검증 split의 소량 샘플(8~32개)을 추론하고, 충분성 지표(parse success·domain pass rate)가 0이 아닌지·기본 길이/형식 분포가 깨지지 않았는지 확인.
- 실패 시 [`skillSample/sufficiency-metric.md §4`](./skillSample/sufficiency-metric.md)의 원인 분류·복구 절차로 위임.

### 1-3. Contract — 토크나이저·프롬프트·데이터 스키마

외부 인터페이스 합의를 검증한다.

- 토크나이저 확장본이 변환기·학습기·추론기 모두에서 동일한 vocab id를 반환하는지.
- 학습 JSONL 레코드가 `system`/`prompt`/`completion` 또는 도메인 동등 구조를 모두 가지는지.
- 내부 데이터 레코드가 사전 정의 스키마를 유지하는지.

본 분류의 검증은 [`changeObligationSample.md`](../harness-template/changeObligationSample.md) 매핑 규약과 결합된다.

### 1-4. Regression — 검증 split 평가

과거 모델 회귀가 재현되지 않는지 감시한다.

- 평가기 출력의 충분성 지표·모델 메트릭을 직전 스냅샷과 비교.
- 다중 trial 평균·CI로 비교. 단일 trial로 회귀 단정 금지([`AGENTS.template.md §3-9`](./AGENTS.template.md)).

### 1-5. 금지 규칙

- 단위 round-trip 검사를 빼먹고 통합 smoke만으로 변환기·파서를 검증하지 마라. 결함의 국소화 비용이 폭증한다.
- 단일 샘플 시각화로 회귀 결론을 내리지 마라.
- 정형 테스트 러너 부재를 핑계로 검증을 생략하지 마라. ad-hoc 검사라도 결과를 커밋 메시지·`reports/`에 남긴다.

---

## 2. 오라클 (통계적 보강)

SW 트랙의 3 오라클(reference/property/metamorphic) 원칙을 그대로 사용한다. 본 트랙은 모델 출력에 대해 다음을 보강한다.

### 2-1. 통계적 오라클

모델 출력의 회귀 판정은 단일 trial의 임계 비교가 아니라 다음 통계적 오라클로 수행한다 — Demšar 2006.

- **paired test** — 동일 split·동일 시드 페어링한 두 트랙(또는 두 체크포인트) 결과에 Wilcoxon signed-rank 또는 paired t-test 적용. p-value < 0.05 + effect size 임계 충족이 있어야 의미 있는 차이.
- **bootstrap CI** — N=1000 resampling 기준 95% CI. n_trials 또는 n_seeds 표본에 적용.
- **effect size** — Cohen's d 또는 ratio of means. 단위 없는 차이 크기.

본 통계적 오라클은 [`skillSample/reproducibility-checklist.md §3-7`](./skillSample/reproducibility-checklist.md)을 단일 출처로 인용한다.

### 2-2. 변환기·파서 오라클

변환기·파서는 SW 트랙의 property 오라클(round-trip)을 우선한다.

- `decode(encode(x)) == x` (도메인 적정 오차 이내).
- 메타모픽: 좌표 평행이동·축 반전 같은 입력 변형에 대한 출력 변화가 예측 가능한지.

### 2-3. 금지 규칙

- "예외 없이 실행됨"만을 모델 오라클로 삼지 마라. 그것은 스모크 체크이다.
- 변환기 코드의 계산 로직을 테스트에서 복붙해 오라클로 만들지 마라.
- 단일 시퀀스 시각화로 모델 회귀의 합격 판정을 내리지 마라.
- 모델 내부 attention·logit 같은 구현 세부에 묶이는 오라클을 만들지 마라.

---

## 3. 테스트 더블

SW 트랙 원본 §4를 그대로 사용한다. 본 트랙은 다음을 추가한다.

- **fixture는 결정적 시드와 묶는다** — fixture에 시드 값과 인덱스 분포를 포함한다. 시드를 바꾸면 fixture 정의가 바뀐 것이다.
- **토크나이저는 실제 확장본을 사용한다** — 가짜 토크나이저는 토큰 ID 계약을 깨뜨린다.
- **통합 smoke와 회귀 평가는 실제 모델로** — 변환기·파서 단위 검증에서는 모델을 stub으로 대체할 수 있으나, 통합·회귀에서는 실제 학습된 모델을 사용한다.

### 3-1. 금지 규칙

- 변환기·파서의 내부 함수에 mock을 주입해 round-trip 결과를 가리지 마라.
- 토크나이저 확장본을 가짜로 대체하지 마라.
- 통합 smoke에서 모델을 stub으로 대체하지 마라.

---

## 4. 변경 영향 테스트 선정 (트랙 분리)

본 트랙은 비교 대상 트랙(예: `delta`/`absolute`, `treatment A`/`treatment B`)이 동시에 운영된다. 변경 영향 선정 시 다음을 추가 강제한다.

- 변환기·토크나이저·평가기 변경 → **모든 비교 트랙**의 round-trip + smoke를 동시 수행.
- 한 트랙의 평가기·인코더만 수정한 경우 → 다른 트랙도 정합성 검증(다른 트랙이 동일 변경을 빠뜨리지 않았는지).
- 파일명 substring 매칭만으로 영향 범위를 결정하지 마라. 두 트랙이 변환기·토크나이저 변경에 동시에 영향받는다.

SW 트랙의 safe vs precise trade-off(국소 게이트는 precise 우선, 전체 게이트는 safe 우선)는 그대로 상속한다.

---

## 5. flaky 대응

본 절은 원칙만 선언한다. 상세 절차는 [`../harness-template/skillSample/flaky-handling.md`](../harness-template/skillSample/flaky-handling.md)에 위임한다. 연구 트랙의 추가 의무는 다음과 같다.

- 동일 시드·동일 입력의 추론을 N≥3회 재실행한다.
- flaky 후보는 시드 고정 누락·CUDA/GPU 비결정성·tokenizer side effect·dataloader 순서 의존 중 어느 범주인지 분류한다.
- flaky 후보를 일지의 "실패한 시도" 절([`skillSample/research-journal.md §3-5`](./skillSample/research-journal.md))에 cross-link한다.

**IMPORTANT:** "한 번 더 돌려보니 괜찮았다"로 종결하지 마라. 통과한 재실행은 **재현되지 않은 실패**이지 해결이 아니다.

---

## 6. 충분성 지표 — 도메인 의존

본 트랙은 line coverage가 적용되지 않는 ML/통계 파이프라인에 대해 다음 원칙을 따른다.

- 충분성 지표는 **도메인이 정의한다** — parse success rate, calibration error, prediction interval coverage, domain pass rate 등.
- 임계값·복구 절차는 [`skillSample/sufficiency-metric.md §2`](./skillSample/sufficiency-metric.md)에 위임.
- 정형 mutation testing류는 ML 모델에 적용이 어렵다 — 본 트랙은 미도입.

### 6-1. 기준 변경 규칙

- 임계값을 **하향 조정하지 마라**. 회귀 해소를 위해 기준을 낮추는 것을 금지한다.
- 상향 조정은 허용한다. 단, 충분성 skill과 본 phase의 서술을 동시 갱신한다.
- "100% parse success" 같은 수치 목표를 잡지 마라. 100%는 모델 품질을 보장하지 않으며, 출력을 단순화시키는 학습 왜곡을 유발할 수 있다.

---

## 7. Agent 표준 실행 흐름

1. 변경 대상 분류(§1) 선택.
2. 오라클(§2) 설계 — 변환기·파서는 property 우선, 모델은 통계적 오라클.
3. 더블(§3) 구성 — 토크나이저·실제 모델은 fake로 대체하지 않음.
4. change-obligation이 지시한 누락 검증을 먼저 작성.
5. 변경 영향 검사(§4) 실행 — 비교 트랙 동시.
6. 실패했으면 → 재현성 검사(§5) → 결정적 실패 / flaky 후보 분류.
7. 검증 split 평가로 전체 게이트 실행.
8. 시각화·사람 검증이 포함되면 다중 샘플 평균과 함께 확인.
9. 전체 게이트 통과 후 커밋.

**IMPORTANT:** 검사가 flaky로 분류된 상태에서 커밋하지 마라. 격리 또는 수정이 선행되어야 한다.

---

## 8. 유지보수 규칙

- 분류·오라클·더블 원칙을 변경했으면 → §7 표준 흐름과 change-obligation 복구 절차를 동시 갱신한다.
- 충분성 기준을 조정했으면 → [`02-sensor.md §1-5`](./02-sensor.md)·[`skillSample/sufficiency-metric.md §2`](./skillSample/sufficiency-metric.md) 두 곳을 동시 갱신한다.
- 정형 테스트 러너(pytest 등)를 도입하면 → §1의 분류와 디렉터리 규약, AGENTS.md §2-2의 센서 명령어를 동시 갱신한다.
- 본 문서 용어가 [`02-sensor.md`](./02-sensor.md)·[`04-evaluation.md`](./04-evaluation.md)·하위 skill과 불일치하면 → `02`를 지도, 본 문서를 방법론 원본으로 간주한다.
