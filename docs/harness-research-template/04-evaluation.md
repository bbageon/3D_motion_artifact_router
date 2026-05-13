# 04. 평가(Evaluation) 레이어 — Research Track

본 문서는 [`../harness-template/04-evaluation.md`](../harness-template/04-evaluation.md)의 SW 트랙 평가 레이어 원본을 **연구 트랙**에 맞춰 확장한 변형이다. 평가 단위(Task/Trial/Outcome/Snapshot)·파이프라인 5단계(Collect/Normalize/Grade/Aggregate/Compare)·관측 전용 경계 원칙은 SW 트랙 원본을 그대로 상속한다. 본 문서는 차이만 기술한다.

본 문서가 추가하는 것은 다음 다섯 가지이다.

- **부트스트랩 임계의 문헌 근거** — trial ≥ 20, snapshot ≥ 2의 통계적 근거(Shewhart 1931 / Daly et al. ICPE 2020 / Foo et al. ICSE 2015 / CLT n≥30 / Wilcoxon n≥20).
- **기계 트랙 vs 사람-가독 트랙의 분리** — `evals/*` 기계 처리와 `reports/<YYYY-MM-DD>.md` 사람-가독 일지의 cross-link 의무.
- **Compare 단계의 5단계 연구 사이클 출력** — 4판정(`stable`/`improvement`/`regression`/`inconclusive`)이 가설 평가의 입력으로 들어가는 상위 구조.
- **가설 사후 수정의 보수성** — HARKing 차단을 위한 사용자 승인 게이트.
- **사람 트랙의 negative result·우회 보존** — 일지에 누적되는 정성적 정보가 후속 메타 분석의 자원.

본 문서는 [`01-instructions.md §2`](./01-instructions.md)의 작성 원칙을 그대로 따른다.

---

## 1. 본 phase의 위치 (관측 전용)

평가는 관측 전용(meta) 레이어이다. 커밋 차단은 [`02-sensor.md`](./02-sensor.md)가 담당한다. 본 phase의 산출물은 어떤 경우에도 머지를 직접 차단하지 않는다 — 머지 차단이 필요하면 02-sensor.md §3-3의 격상 조건으로 상승시킨다.

**IMPORTANT:** 평가 임계값 완화로 회귀를 회피하지 마라.

---

## 2. 평가 단위 (연구 트랙 baseline)

- **Task** — 사용자 요청 단위. 하나의 커밋 또는 PR과 일치하도록 식별자 부여. 비교 트랙 정보를 task 메타에 포함.
- **Trial** — 핵심 경로 변경에 따른 검증 split 평가 1회 또는 smoke 추론 1회.
- **Outcome** — gate 수준과 trial 수준의 두 계층을 병렬 보존. gate 수준은 02-sensor.md §4의 세 판정. trial 수준은 [`evalSample/eval-grade.md`](../harness-template/evalSample/eval-grade.md)가 정의.
- **Snapshot** — 일·주 단위로 집계. 일별은 day 1부터 가동, 주별은 일별 스냅샷 ≥ 14개 누적 시 가동.

---

## 3. 파이프라인 위치 (점진 가동)

SW 트랙 원본의 5단계(Collect/Normalize/Grade/Aggregate/Compare)를 그대로 사용한다. 본 트랙은 부트스트랩을 **달력 시간이 아니라 trial·snapshot 수**로 통제한다.

- Collect → day 1부터 가동.
- Normalize → raw record가 1개라도 생성되면 즉시 가동.
- Grade → Normalize와 동시 가동.
- Aggregate (일별) → day 1부터.
- Aggregate (주별) → 일별 스냅샷 ≥ 14개 누적 시.
- Compare → trial ≥ 20개 + 비교 가능한 스냅샷 ≥ 2개일 때 본격 가동. 그 이전에는 informational 출력만.

본 트랙은 평가 파이프라인 전용 스크립트가 부트스트랩 초기에 없을 수 있다. 초기에는 Agent가 수동으로 raw record를 기록한다. 사용 빈도가 늘면 `scripts/eval-*.py`로 분리한다.

### 3-1. 부트스트랩 임계의 문헌 근거

위 임계값(trial ≥ 20, snapshot ≥ 2)은 다음 문헌을 따른다.

- **Statistical Process Control (Shewhart 1931 / Montgomery, *Introduction to Statistical Quality Control*)** — 안정된 control limit 산출에 20–25 subgroup이 표준. Compare 본격 가동의 trial ≥ 20 기준.
- **Daly et al., ICPE 2020 *Use of Statistical Methods to Track Performance*** — change-point detection의 reference window로 ~20–30 historical run.
- **Foo et al., ICSE 2015 *An Industrial Case Study on the Automated Detection of Performance Regressions in Heterogeneous Environments*** — 7–14일 rolling window. 주별 스냅샷의 14 일별 스냅샷 누적 기준.
- **CLT n≥30 rule / Wilcoxon-Mann-Whitney 표본 하한 n≥20** — 임계값 보정에 snapshot ≥ 20을 요구하는 근거.

본 임계값은 프로젝트의 trial cadence에 맞춰 적용본에서 보정한다.

---

## 4. 출력 경로 (듀얼 트랙)

본 트랙은 동일 작업을 두 트랙으로 보존한다.

### 4-1. 기계 처리 트랙

- `evals/raw/<timestamp>_<task_id>_<trial_id>.json` — Collect 산출물.
- `evals/normalized/<timestamp>.json` — Normalize 산출물.
- `evals/graded/<timestamp>.json` — Grade 산출물.
- `evals/snapshots/{daily,weekly}/<period>.json` — Aggregate 산출물.
- `evals/reports/<period>.md` — Compare 산출물(5단계 리포트).
- `evals/hypotheses/<h_id>.md` — 사전 등록 가설(append-only).
- `evals/hypotheses/_index.md` — 가설 인덱스.
- `evals/workarounds/<W-id>.md` — 우회 ledger(append-only).
- `evals/workarounds/_index.md` — 우회 인덱스.
- `evals/checklists/<period>_<scope>.md` — 재현성 체크리스트.

회귀·개선 판정과 자가 수정 메타 규칙의 근거는 본 트랙에서만 인용한다.

### 4-2. 사람-가독 트랙

- `reports/<YYYY-MM-DD>.md` — 일자별 연구일지.
- `reports/figures/<YYYY-MM-DD>/` — 시각화 산출물.

정성·시각·서술 보존이 목적이며, 작성 의무는 [`skillSample/research-journal.md`](./skillSample/research-journal.md)이 정의한다.

### 4-3. cross-link 의무

두 트랙은 raw record 경로(`evals/raw/<timestamp>_<task_id>_<trial_id>.json`)로 cross-link되어야 한다. cross-link이 끊긴 일지·raw record는 회귀 판정의 근거로 인용할 수 없다.

---

## 5. 측정 지표 (연구 트랙 baseline)

### 5-1. 수행 품질 지표 (Agent)

- Task 성공률 — 변경이 전체 게이트 통과로 도달한 비율.
- 국소 게이트 pass@1 — 첫 시도에서 round-trip + smoke를 통과한 비율.
- 평균 재시도 횟수.

### 5-2. 검증 품질 지표 (모델)

- 게이트별 통과율.
- 규칙 위반 카테고리.
- 충분성 지표 (parse success·calibration·domain pass rate 등 — 도메인 정의).
- 모델 메트릭 (도메인별: MPJPE/MSE/F1/AUROC/BLEU 등 — 트랙·체크포인트·split별 분리).
- horizon-wise·joint-wise·class-wise·segment-wise error (도메인에 따라 적용).
- flaky 발생률·격리 수명.

### 5-2-1. 정성·시각화 보조 지표

정량 지표 외에 다음을 정성·시각화 보조 지표로 함께 관찰한다. 본 항목들은 informational로만 기록한다.

- 누적 오차·drift 추세.
- 분포의 비대칭·꼬리 두께.
- 표현 형식 파손 빈도(parse 실패 사례 분류).
- 사람 검증이 가능한 도메인 시각화(예: 시퀀스 예측에서 ground-truth 대비 오버레이).

### 5-3. 개선 효과 지표

- 지침 개정(AGENTS.md, phase, skills) 전후 위반 카테고리 변화.
- 토큰 포맷·프롬프트 포맷 변경 전후 충분성 지표·모델 메트릭 변화.
- 반복 실패 카테고리의 집중도.

### 5-4. Agent 실행 효율 지표 (선택)

본 트랙도 SW 트랙 원본 §6-4의 토큰 소비·도구 호출 지표를 동일하게 적용 가능하다. 도입 시 [`../harness-template/04-evaluation.md §6-4`](../harness-template/04-evaluation.md)를 그대로 인용한다.

---

## 6. 회귀·개선 판정

SW 트랙의 4 수준(개별/반복/구조적 회귀/개선 신호)을 그대로 사용한다. 본 트랙은 다음 baseline을 둔다.

- 개별 → 반복: 동일 카테고리에서 연속 N=3회 또는 7일 윈도 내 M=5건 이상.
- 반복 → 구조적 회귀: 둘 이상의 일별 스냅샷 또는 1주별 + 1일별 두 윈도에서 연속 확인.
- 개선 신호: 같은 지표가 두 윈도에서 동조 개선.

표본이 부족한 초기 운영(trial < 20 또는 비교 가능 스냅샷 < 2)에서는 회귀 판정을 보류하고 informational로만 기록한다 — §3-1.

---

## 7. 피드백 루프 — 5단계 연구 사이클

본 트랙의 Compare 단계는 SW 트랙의 4판정을 **가설 평가의 입력**으로 받아 5단계 사이클로 출력한다. 본 형식 의무는 [`evalSample/eval-compare.md §6`](./evalSample/eval-compare.md)에 위임한다.

### 7-1. 5단계 사이클

1. **제시한 가설** — `evals/hypotheses/`의 활성 h_id 인용. 본 단계에서 새 가설을 만들지 마라.
2. **실험 평가** — 어떤 trial·snapshot이 본 비교에 사용되었는지 명시(트랙·시드 분포·체크포인트 model_card hash·split·raw record 경로).
3. **실험적 근거** — 트랙별 지표 변화량·CI·effect size·paired test(Demšar 2006). 사실 기록만 한다.
4. **가설 평가** — 가설별 `supports`/`contradicts`/`inconclusive`. §3 인용 의무.
5. **다음 스텝 또는 가설 사후 검증** —
   - supports → 다음 연구 단계 진행 또는 `status: supported` 마감(사용자 승인).
   - contradicts·inconclusive → (A) 추가 trial 수집, (B) 가설 폐기 draft, (C) 가설 수정 draft, (D) 평가 절차 결함 의심. **B·C는 사용자 승인 게이트** 통과 후에만 promote.

위 5단계 중 하나라도 빠진 비교 리포트는 자가 수정 메타 규칙(AGENTS.md §3-7)의 근거로 인용하지 마라.

### 7-2. 가설 사후 수정의 보수성 (HARKing 차단)

Kerr 1998·Lipton & Steinhardt 2018·Ioannidis 2005가 보고하듯, 결과를 본 뒤 가설을 손질해 supports로 보이게 만드는 HARKing은 발견의 통계적 신뢰성을 파괴한다. 본 트랙은 다음 조건을 모두 만족할 때만 사용자 승인을 요청한다.

- 사전 정의된 기각 조건이 명확히 충족되었거나, 평가 절차 결함이 식별되었다.
- 변경 사유가 [`evalSample/eval-compare.md`](./evalSample/eval-compare.md) 5단계 리포트에 인용되어 있다.
- 새 draft가 [`skillSample/hypothesis-registry.md §4`](./skillSample/hypothesis-registry.md) 형식으로 작성되었다.

위 조건이 충족되어도 사용자 승인 없이 promote하지 마라.

### 7-3. 회귀 해소

- 구조적 회귀가 감지되었으면 → AGENTS.md §3 절대 규칙·02-sensor.md·03-test.md·해당 skill에 대응 규칙을 **추가 또는 강화**한다. 본 절차는 AGENTS.md §3-7 자가 수정 메타 규칙과 같다.
- 회귀 해소를 위해 센서·평가 임계값을 **완화하지 마라**.
- 규칙 추가·수정 시 근거 리포트 경로를 커밋 메시지에 인용한다.
- 동일 회귀가 규칙 개정 없이 반복되면 → 센서 또는 검사 자체의 결함을 의심하고 02·03을 재검토한다.

---

## 8. Agent 관여 범위

- task_id·trial_id·변경 요약·비교 트랙·체크포인트 경로를 raw record에 남긴다.
- 구조적 회귀 리포트가 생성되었으면 → 원인 카테고리(토큰 포맷 회귀 / 학습 분포 변화 / 파서 결함 / 시드·결정성 결함 등)를 분류하고 규칙 개정안 초안 작성. 개정안은 근거 리포트 경로를 본문에 인용한다.

**금지:** 평가 지표를 근거 없이 인용하지 마라. 모든 인용은 `evals/snapshots/` 또는 `evals/reports/`의 구체 경로를 참조한다.

---

## 9. 유지보수 규칙

- 지표·임계값·승격 기준을 변경했으면 → 과거 스냅샷과의 호환성을 주석으로 남긴다. `aggregation_rule_version`을 올린다.
- 평가 판정을 관측 전용 이외로 격상하지 마라.
- 부트스트랩 임계(§3-1: trial ≥ 20, snapshot ≥ 2)는 적용본의 trial cadence가 측정된 뒤 한 번 보정한다. 보정 시 §3-1 인용 문헌 또는 누적된 `evals/snapshots/`의 분포를 근거로 인용한다.
- 본 문서 용어가 02-sensor.md·03-test.md·하위 `eval-*` skill과 불일치하면 → `02`를 지도, `03`을 방법론, 본 문서를 관측 레이어로 간주하고 본 문서를 갱신한다.
