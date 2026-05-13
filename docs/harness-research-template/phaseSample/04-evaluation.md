# 04. 평가(Evaluation) Phase — `<project-name>` (Research Track Sample)

> 본 phase 문서는 [`../04-evaluation.md`](../04-evaluation.md)의 "04. 평가 레이어 (Research Track)" 규격을 본 프로젝트에 적용한 sample이다. 파이프라인 5단계·지표 부류·5단계 사이클·부트스트랩 임계 정의는 상위 원본을 상속한다.

---

## 1. 본 phase의 위치 (관측 전용)

평가는 관측 전용(meta) 레이어이다. 커밋 차단은 [`./02-sensor.md`](./02-sensor.md)가 담당한다.

**IMPORTANT:** 본 phase의 산출물은 어떤 경우에도 머지를 직접 차단하지 않는다.

---

## 2. 평가 단위

- **Task** — 사용자 요청 단위. 트랙 정보 포함.
- **Trial** — 핵심 경로 변경에 따른 검증 split 평가 1회 또는 smoke 1회.
- **Outcome** — gate 수준(02 §4의 세 판정) + trial 수준([`eval-grade`](../../harness-template/evalSample/eval-grade.md))의 두 계층 병렬.
- **Snapshot** — 일·주 단위 집계. 일별 day 1부터, 주별 일별 ≥ 14개 누적 시.

---

## 3. 파이프라인 위치 (점진 가동)

- Collect → [`../../harness-template/evalSample/eval-collect.md`](../../harness-template/evalSample/eval-collect.md). day 1부터.
- Normalize → [`../../harness-template/evalSample/eval-normalize.md`](../../harness-template/evalSample/eval-normalize.md). raw 1개 생성 즉시.
- Grade → [`../../harness-template/evalSample/eval-grade.md`](../../harness-template/evalSample/eval-grade.md). Normalize와 동시.
- Aggregate → [`../../harness-template/evalSample/eval-aggregate.md`](../../harness-template/evalSample/eval-aggregate.md). 일별 day 1부터, 주별 일별 ≥ 14개 누적 시.
- Compare → [`../evalSample/eval-compare.md`](../evalSample/eval-compare.md). **trial ≥ 20개 + 비교 가능 스냅샷 ≥ 2개**일 때 본격 가동.

### 3-1. 부트스트랩 임계

[`../04-evaluation.md §3-1`](../04-evaluation.md)의 문헌 근거(Shewhart 1931 / Daly et al. ICPE 2020 / Foo et al. ICSE 2015 / CLT n≥30 / Wilcoxon n≥20)를 상속한다. trial cadence 측정 후 보정.

---

## 4. 출력 경로

### 4-1. 기계 처리 트랙

- `evals/raw/<timestamp>_<task_id>_<trial_id>.json`
- `evals/normalized/`, `evals/graded/`, `evals/snapshots/{daily,weekly}/`
- `evals/reports/<period>.md` (5단계 리포트)
- `evals/hypotheses/<h_id>.md`·`_index.md`
- `evals/workarounds/<W-id>.md`·`_index.md`
- `evals/checklists/<period>_<scope>.md`

### 4-2. 사람-가독 트랙

- `reports/<YYYY-MM-DD>.md`
- `reports/figures/<YYYY-MM-DD>/`

두 트랙은 raw record 경로로 cross-link.

---

## 5. 측정 지표

### 5-1. 수행 품질 지표 (Agent)

- Task 성공률 / 국소 게이트 pass@1 / 평균 재시도 횟수.

### 5-2. 검증 품질 지표 (모델)

- 게이트별 통과율 / 규칙 위반 카테고리.
- 충분성 지표: `<도메인 충분성 지표>`.
- 도메인 모델 메트릭: `<지표 1>`, `<지표 2>`, ... (트랙·체크포인트·split별 분리).
- horizon-wise·segment-wise·class-wise error.
- flaky 발생률·격리 수명.

### 5-2-1. 정성·시각화 보조 지표

[`../04-evaluation.md §5-2-1`](../04-evaluation.md)을 상속한다. 본 프로젝트의 도메인 시각화는 `<주요 시각화 종류>`이다.

### 5-3. 개선 효과 지표

- 지침 개정 전후 위반 카테고리 변화.
- 포맷·hyperparam 변경 전후 충분성·모델 메트릭 변화.
- 반복 실패 카테고리 집중도.

### 5-4. Agent 실행 효율 지표 (선택)

[`../../harness-template/04-evaluation.md §6-4`](../../harness-template/04-evaluation.md)를 도입 시 동일 적용.

---

## 6. 회귀·개선 판정

[`../04-evaluation.md §6`](../04-evaluation.md)의 4 수준을 그대로 사용. 본 프로젝트 승격 기준 초기값:

- 개별 → 반복: 동일 카테고리 연속 N=3회 또는 7일 윈도 내 M=5건.
- 반복 → 구조적 회귀: 둘 이상의 일별 스냅샷 또는 1주별 + 1일별.
- 개선 신호: 같은 지표가 두 윈도에서 동조 개선.

표본 부족 시 회귀 판정 보류.

---

## 7. 피드백 루프 — 5단계 사이클

Compare 단계의 출력은 [`../evalSample/eval-compare.md §6`](../evalSample/eval-compare.md)의 5단계 형식을 따른다.

1. 제시한 가설 — `evals/hypotheses/`의 활성 h_id 인용.
2. 실험 평가 — trial·snapshot·체크포인트·split·raw record 명시.
3. 실험적 근거 — 트랙별 지표 변화량·CI·effect size·paired test (Demšar 2006).
4. 가설 평가 — supports/contradicts/inconclusive.
5. 다음 스텝 또는 가설 사후 검증 — supports/contradicts/inconclusive 별 옵션.

### 7-1. 회귀 해소

[`../04-evaluation.md §7-3`](../04-evaluation.md)의 자가 수정 메타 규칙을 상속한다. 회귀 회피를 위한 임계 완화 금지.

### 7-2. 가설 사후 수정의 보수성

[`../04-evaluation.md §7-2`](../04-evaluation.md)의 사용자 승인 게이트를 상속한다(Kerr 1998 HARKing 차단).

---

## 8. Agent 관여 범위

- raw record에 task_id·trial_id·트랙·체크포인트 model_card hash 기록.
- 구조적 회귀 리포트가 생성되면 원인 카테고리 분류 + 규칙 개정안 초안.

**금지:** 평가 지표를 근거 없이 인용하지 마라. 모든 인용은 `evals/snapshots/` 또는 `evals/reports/` 경로 참조.

---

## 9. 유지보수 규칙

- 지표·임계·승격 기준 변경 → 과거 스냅샷과의 호환성 주석. `aggregation_rule_version` 인상.
- 평가 판정을 관측 전용 이외로 격상 금지.
- 부트스트랩 임계(§3-1)는 trial cadence 측정 후 한 번 보정.
