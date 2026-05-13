# 03. 테스트(Test) Phase — `<project-name>` (Research Track Sample)

> 본 phase 문서는 [`../03-test.md`](../03-test.md)의 "03. 테스트 레이어 (Research Track)" 규격을 본 프로젝트에 적용한 sample이다. 분류·오라클·더블·flaky·충분성 원칙은 상위 원본을 상속한다.

본 프로젝트는 정형 단위테스트 인프라가 `<없음 / 도입 예정 / pytest 등>`이므로 "테스트"는 (a) round-trip 검사, (b) smoke 추론, (c) 검증 split 평가의 세 형태로 운영된다.

---

## 1. 테스트 분류 (본 프로젝트 적용)

- **단위(Unit) — round-trip 검사**
  - 대상: `<인코더·파서 모듈 경로>`.
  - 형태: 임의 입력 N=`<32 또는 1k>` → encode → decode → equality(오차 `<1e-5>`).
  - `<정형 test 디렉터리 사용 또는 ad-hoc 인라인 검증>`.
- **통합(Integration) — smoke 추론**
  - 대상: `<smoke 스크립트 경로>`.
  - 형태: 검증 split 소량 샘플 추론. 충분성 지표가 0이 아닌지 확인.
- **계약(Contract) — 토크나이저·프롬프트·스키마**
  - 토크나이저 확장본의 vocab id 일치.
  - 학습 JSONL의 `system`/`prompt`/`completion` 필드(또는 도메인 동등) 유지.
  - 내부 데이터 스키마 유지.
- **회귀(Regression) — 검증 split 평가**
  - 대상: `<트랙별 평가기 경로>`.
  - 비교 단위: 충분성 지표·도메인 메트릭의 직전 스냅샷.

### 1-1. 비교 트랙

본 프로젝트는 `<트랙 A>` / `<트랙 B>` / `<트랙 N>`을 동시 운영한다. 회귀 테스트는 활성 트랙 모두에 동일 윈도 정책으로 잘라낸 검증 split을 사용한다.

### 1-2. 금지 규칙

[`../03-test.md §1-5`](../03-test.md)를 상속한다.

---

## 2. 오라클 설계

- 변환기·파서 → property 오라클(`decode(encode(x)) == x`) + metamorphic 오라클(좌표 평행이동 등).
- 모델 출력 → 통계적 오라클(paired Wilcoxon · effect size · CI).
- 단위 결정성 검사(부동소수 오차 `<도메인 적정값>`).

[`../03-test.md §2`](../03-test.md)·[`../skillSample/reproducibility-checklist.md §3-7`](../skillSample/reproducibility-checklist.md)을 단일 출처로 인용.

---

## 3. 테스트 더블

- **fixture** — 검증용 소량 샘플(`<8~32 샘플>`)을 고정 인덱스로 추출. 시드와 인덱스를 fixture 정의에 포함.
- **stub / mock** — 변환기·파서 단위 검증에서는 모델을 stub 또는 fake greedy decoder로 대체 가능. 통합 smoke와 회귀 평가는 실제 모델.
- **fake** — 토크나이저는 실제 확장본 사용 (가짜 토크나이저 금지).

---

## 4. 변경 영향 테스트 선정

국소 게이트:

1. 핵심 경로(02 §2)에 대응하는 검증.
   - 변환기 변경 → 모든 트랙의 인코더 round-trip + 트랙 대응 파서 round-trip.
   - 파서 변경 → 트랙 대응 인코더 출력으로 파서 검증.
   - 토크나이저 변경 → 변환기·학습기·추론기의 vocab 일치 확인.
   - 학습 코드 변경 → smoke 추론 + 토큰/출력 분포 확인.
2. 변경된 파일을 import/의존하는 상위 모듈 검증.
3. change-obligation 복구 절차로 신규 생성·보완된 검사.

전체 게이트: 활성 트랙 모두의 검증 split 평가 1회 이상.

---

## 5. flaky 대응

[`../03-test.md §5`](../03-test.md)를 따른다. 본 프로젝트는 다음 원인 범주를 적용한다.

- 시드 고정 누락.
- CUDA/GPU 비결정성.
- tokenizer side effect.
- dataloader 순서 의존.

flaky 후보는 일지 §3-5에 cross-link.

---

## 6. 충분성 지표

본 프로젝트의 충분성 지표는 `<도메인 충분성 지표 — 예: parse success rate>`이다(line coverage 미적용).

- smoke 게이트 → 0% 금지(hard fail).
- 전체 게이트 → 검증 split 임계 + 격상 조건(sufficiency-metric §2).

### 6-1. 기준 변경 규칙

[`../03-test.md §6-1`](../03-test.md)을 상속한다. 하향 금지·상향 허용.

---

## 7. Agent 표준 실행 흐름

[`../03-test.md §7`](../03-test.md)의 9단계를 본 프로젝트에 그대로 적용한다.

---

## 8. 유지보수 규칙

- 분류·오라클·더블 원칙 변경 → §7 표준 흐름과 change-obligation 복구 절차 동시 갱신.
- 충분성 기준 조정 → 02-sensor.md §1-5와 sufficiency-metric.md §2 동시 갱신.
- 정형 테스트 러너 도입 시 → §1의 분류·디렉터리 규약·AGENTS.md §2-2의 센서 명령어 동시 갱신.
