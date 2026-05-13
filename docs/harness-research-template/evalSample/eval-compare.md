---
description: 현재 스냅샷과 기준 스냅샷을 비교해 활성 가설을 통계적으로 평가하고, "1. 제시한 가설 → 2. 실험 평가 → 3. 실험적 근거 → 4. 가설 평가 → 5. 다음 스텝 또는 가설 사후 검증" 5단계로 출력하는 skill이다.
when_to_use: evaluation comparison, hypothesis evaluation, regression detection, improvement detection, snapshot diff, 평가 비교, 가설 평가, 회귀 감지, 개선 신호 판정
allowed-tools:
  - Read
  - Grep
  - Bash
  - Write
  - Edit
---

# Eval Compare — Research Track (5-Step Cycle)

본 skill은 [`04-evaluation.md §7-1`](../04-evaluation.md) 평가 파이프라인의 Phase 5 (Compare)를 연구 트랙에 적용한다. 입력은 [`eval-aggregate`](../../harness-template/evalSample/eval-aggregate.md) 산출 스냅샷과 [`hypothesis-registry`](../skillSample/hypothesis-registry.md)의 활성 가설이며, 출력은 **연구 관점의 5단계 리포트**(가설 → 실험 평가 → 실험적 근거 → 가설 평가 → 다음 스텝/사후 검증)이다.

본 skill의 출력은 일반 SW 트랙의 4판정(`stable`/`improvement`/`regression`/`inconclusive`)이 아니라 5단계 사이클이다. 4판정은 5단계 중 §4(가설 평가)의 입력으로 사용된다.

본 skill은 [`01-instructions.md §2`](../01-instructions.md)의 문장 형식 규칙을 그대로 따른다.

---

## 1. 목적

본 skill은 다음을 보장한다.

- 최근 변화가 실제 회귀·개선인지 판단한다.
- 활성 가설을 통계적으로 평가한다(supports/contradicts/inconclusive).
- 가설 사후 수정 옵션(폐기·수정·추가 trial)을 사용자 승인 게이트로만 promote.
- 출력은 항상 5단계 구조를 따라 가설·근거·평가·후속 행동이 분리 보존된다.

### 1-1. 본 skill이 맡지 않는 것

- 가설 등록·수정·폐기의 사용자 승인 절차 — [`hypothesis-registry §4`](../skillSample/hypothesis-registry.md).
- 우회 ledger 등록·등급 분류 — [`workaround-tracking`](../skillSample/workaround-tracking.md).
- 데이터·모델 metadata 정합성 검증 — [`data-versioning §3`](../skillSample/data-versioning.md).
- 일지 본문 — [`research-journal`](../skillSample/research-journal.md).

본 skill은 **스냅샷 비교 + 가설 평가 + 후속 행동 제안**만 수행한다.

---

## 2. 입력

- 현재 스냅샷(`evals/snapshots/{daily,weekly}/<current>.json`).
- 기준 스냅샷(`evals/snapshots/{daily,weekly}/<reference>.json`).
- 활성 가설 목록(`evals/hypotheses/_index.md` + 각 `<h_id>.md`).
- 활성 우회 ledger(`evals/workarounds/_index.md`).
- 비교 기준 기간·회귀 임계값·low-sample 경고 여부.

---

## 3. 부트스트랩 임계 (본격 가동 조건)

본 skill은 다음을 모두 만족할 때만 본격 가동한다 — [`04-evaluation.md §3-1`](../04-evaluation.md).

- trial ≥ 20개 누적.
- 비교 가능한 스냅샷 ≥ 2개.

위 조건을 충족하지 않은 단계에서는 informational 출력만 작성하며, 가설 평가는 보류한다("inconclusive — low sample").

---

## 4. 비교 대상 기본 지표

- 충분성 지표(parse success / domain pass rate 등).
- 도메인 모델 메트릭(트랙별·체크포인트별·split별 분리).
- horizon-wise / segment-wise / class-wise error.
- effect size · CI · paired test 결과.
- 수행 품질 지표(pass@1, retry count).

---

## 5. 통계적 평가

본 skill의 §3(실험적 근거)과 §4(가설 평가)는 다음 통계적 절차를 따른다 — Demšar 2006, [`reproducibility-checklist.md §3-7`](../skillSample/reproducibility-checklist.md).

- **paired test** — 동일 split·동일 시드 페어링한 두 트랙 결과에 Wilcoxon signed-rank 또는 paired t-test 적용. p-value < 0.05 + effect size 임계 충족이 있어야 의미 있는 차이.
- **bootstrap CI** — N=1000 resampling 기준 95% CI.
- **effect size** — Cohen's d 또는 ratio of means.
- **다중 비교 보정** — 가설이 여러 개라면 Benjamini-Hochberg 또는 Bonferroni로 false discovery rate 보정.

---

## 6. 출력 형식 (5단계 리포트)

`evals/reports/<period>.md`에 다음 구조로 작성한다. 5단계 중 하나라도 빠지면 자가 수정 메타 규칙(AGENTS.md §3-7)의 근거로 인용할 수 없다.

```markdown
# Compare Report — <period> (<scope>)

- 생성 시점: <ISO 8601>
- 비교 기간: <reference snapshot> → <current snapshot>
- 표본: trial=<N>, snapshot=<M>, n_seeds=<K>
- 활성 가설: <H-id 목록>

---

## 1. 제시한 가설

- <H-id> — <한 줄 가설 본문>. (`evals/hypotheses/<h_id>.md`)
- ...

본 단계에서 새 가설을 만들지 마라. 결과를 보고 가설을 손질하는 행위는 HARKing이며, hypothesis-registry §4 사용자 승인 게이트로만 가능하다.

## 2. 실험 평가

| 항목 | 값 |
|---|---|
| 사용 트랙 | <트랙 목록> |
| 시드 분포 | <시드 목록> |
| 체크포인트 model_card hash | <hash 목록> |
| split | <val / test> |
| raw record 경로 | `evals/raw/<timestamp>_*.json` 다중 |

## 3. 실험적 근거

다음은 사실 기록이다. supports/contradicts를 본 단계에서 단정하지 마라.

| 비교 | 지표 | <트랙 A> | <트랙 B> | Δ | CI(95%) | effect size | p-value (paired) |
|---|---|---|---|---|---|---|---|
| <비교 1> | <지표> | <값> | <값> | <Δ> | [<lo>, <hi>] | <d> | <p> |

다중 비교 보정: <Benjamini-Hochberg / Bonferroni / 없음>.

## 4. 가설 평가

가설별 supports/contradicts/inconclusive:

- <H-id> → **supports** | **contradicts** | **inconclusive** — 근거: §3의 <비교 N>. 사전 정의된 기각 조건 충족 여부: <충족/미충족>.
- ...

본 단계의 판정은 사전 정의된 기각 조건([`hypothesis-registry §2-3`](../skillSample/hypothesis-registry.md))과 §5의 통계적 임계를 모두 충족할 때만 부여한다.

## 5. 다음 스텝 또는 가설 사후 검증

가설 평가 결과에 따라 다음 옵션 중 **하나 이상**을 제시한다. 본 절은 제안이며, 가설 status 전환·새 trial 실행은 **사용자 승인 후**에만 진행한다.

- supports → 옵션 A: 다음 연구 단계 진행 (구체 단계 명시). 옵션 B: `status: supported` 마감(사용자 승인 필요).
- contradicts → 옵션 C: 가설 폐기 draft (`hypothesis-registry §4`, 사용자 승인 필수). 옵션 D: 가설 수정 draft (사용자 승인 필수). 옵션 E: 평가 절차 결함 의심 → 02·03 재검토.
- inconclusive → 옵션 F: 추가 trial 수집 (필요 표본 수 명시). 옵션 G: low-sample 경고 유지.

위 옵션 중 B·C·D는 hypothesis-registry §4 사용자 승인 게이트 통과 후에만 promote 가능하다.

## 부록 A — 우회·간접 해결 ledger 인용

본 비교 기간 활성·해소된 우회 항목:

| W-id | severity | status | 한 줄 요약 | 본격 학습 진입 전 의무 |
|---|---|---|---|---|
| <W-id> | <등급> | <상태> | <요약> | <의무> |

**critical 미해소 항목이 있으면 본 비교 결과를 외부에 인용하지 마라** — [`workaround-tracking §6-1`](../skillSample/workaround-tracking.md).

(우회가 없으면 "(없음)"으로 표기. 부록을 빈 채로 두지 마라.)

## 부록 B — low-sample 경고 (해당 시)

표본 부족으로 §4 판정이 보류된 경우 본 절을 채운다.

- 현재 trial: <N> (임계 20).
- 현재 스냅샷: <M> (임계 2).
- 보완 계획: <추가 trial 일정·예상 시점>.
```

---

## 7. 사용자 승인 게이트와의 연결

본 skill의 §5 출력 중 다음 옵션은 사용자 승인 없이 promote하지 마라.

- 옵션 B (`status: supported` 마감) — hypothesis-registry §4.
- 옵션 C (가설 폐기 draft) — hypothesis-registry §4.
- 옵션 D (가설 수정 draft) — hypothesis-registry §4.
- 외부 공개에 본 리포트 인용 — reproducibility-checklist §4.

승인은 명시적 메시지로만 인정.

---

## 8. 금지 규칙

- §1~§5 중 하나라도 빠진 리포트를 자가 수정 메타 규칙(AGENTS.md §3-7)의 근거로 인용하지 마라.
- 단일 trial 결과로 supports/contradicts를 단정하지 마라.
- 가설 본문을 본 skill에서 직접 수정하지 마라.
- 회귀 회피를 위해 회귀 임계값을 하향 조정하지 마라.
- 부록 A를 빈 채로 두지 마라.

---

## 9. 유지보수 규칙

- 출력 형식(§6)을 변경했으면 → [`04-evaluation.md §7-1`](../04-evaluation.md)과 동시 갱신.
- 통계적 임계(§5)를 변경했으면 → [`reproducibility-checklist §3-7`](../skillSample/reproducibility-checklist.md)와 동시 갱신.
- 본 skill의 부록 A 누락이 반복되면 → 워크플로 자동화 검토(템플릿화).
- 본 문서 용어가 hypothesis-registry·workaround-tracking과 불일치하면 → AGENTS.md를 원본으로 갱신.

<IMPORTANT>
본 skill의 5단계 출력은 "프로젝트 개발의 다음 작업 지시"가 아니라 **"연구 관점의 가설 평가 사이클(가설 → 실험 → 근거 → 평가 → 후속 검증)"** 이다. 5단계 중 하나라도 빠지면 AGENTS.md §3-7 자가 수정 메타 규칙의 메트릭 근거가 회고적 합리화(post-hoc rationalization)로 오염된다 — Kerr 1998 HARKing. 가설 사후 검증의 옵션 C/D(폐기·수정)는 hypothesis-registry §4 사용자 승인 게이트를 우회할 수 없다.
</IMPORTANT>
