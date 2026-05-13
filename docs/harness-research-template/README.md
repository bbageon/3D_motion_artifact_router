# Harness Engineering Template — Research Track

이 템플릿은 [`docs/harness-template/`](../harness-template/README.md)의 일반 SW 하네스를 **연구 프로젝트**(가설 검증 중심·재현성 1차 의무·실험 메트릭 비교 가능성 필수)에 맞춰 재구성한 변형이다. 두 템플릿은 동시 공존하며 — 본 변형은 ML/통계/실험 연구 저장소를 위한 starting point이다.

본 변형은 **본 저장소의 적용본**(3D Motion Trajectory Prediction — `AGENTS.md` + `.claude/rules/phase/` + `.claude/skills/`)에서 일반화 추출되었으며, 다음 외부 연구를 인용해 보강되었다.

- Kerr 1998 — *HARKing: Hypothesizing After the Results are Known* (HARKing 정의·차단의 원논문).
- Ioannidis 2005 — *Why Most Published Research Findings Are False* (사후 가설 수정·작은 표본·다중 비교가 결과 신뢰성에 미치는 영향).
- Pineau et al. 2021 — *Improving Reproducibility in Machine Learning Research* (NeurIPS Reproducibility Checklist 2.0의 코드·데이터·hyperparam·시드 의무).
- Dodge et al. 2019 — *Show Your Work: Improved Reporting of Experimental Results* (학습 비용·hyperparam 분포·표본 수 보고 의무).
- Lipton & Steinhardt 2018 — *Troubling Trends in Machine Learning Scholarship* (ML 논문의 흔한 결함: 설명 vs 추측 혼동·HARKing·misleading benchmarks).
- Sambasivan et al. 2021 — *"Everyone wants to do the model work, not the data work"* (data versioning 부재가 ML 재현성의 최대 단일 위험).
- Demšar 2006 — *Statistical Comparisons of Classifiers over Multiple Data Sets* (paired Wilcoxon · effect size · multiple-comparison 보정의 합의된 절차).
- Shewhart 1931 / Montgomery — *Statistical Process Control* (안정된 control limit 산출에 20~25 subgroup 표준).
- Daly et al. ICPE 2020 — *Use of Statistical Methods to Track Performance in MongoDB* (change-point detection에 ~20~30 historical run).
- Foo et al. ICSE 2015 — *Industrial Case Study on Automated Detection of Performance Regressions* (7~14일 rolling window).

---

## 1. 어떤 프로젝트에 적합한가

본 템플릿은 다음 조건을 하나 이상 만족하는 프로젝트에 적용한다.

- 사전 등록 가설을 평가하는 ML/통계/실험 연구.
- 결과를 논문·발표·외부 보고서로 공개할 예정인 저장소.
- 데이터 산출물(전처리 JSON·학습 JSONL·체크포인트 등)의 정의가 바뀔 수 있는 저장소.
- 회귀·개선 판정을 정량 지표 + 통계적 검정으로 수행하려는 저장소.
- 표준 line coverage 기준이 적용되지 않는 ML/통계 파이프라인(line coverage 대신 도메인 충분성 지표가 필요).

일반 SW 개발(웹·서버·클라이언트 등)은 [`docs/harness-template/`](../harness-template/README.md) 원본을 사용한다.

---

## 2. 일반 SW 하네스와의 차이

본 변형은 일반 SW 템플릿에 다음 다섯 축을 추가한다.

- **Role(연구자 페르소나)** — `01-instructions.md`에 정체성·우선순위(연구 정직성 > 재현성 > 비교 가능성 > 효율성 > 편의성)·의사결정 권한·커뮤니케이션 스타일 절을 둔다.
- **연구 정직성 게이트** — 가설 사전 등록(append-only)·HARKing 차단·우회 ledger·negative result 보존을 절대 규칙으로 둔다.
- **사용자 승인 게이트** — 가설 status 전환·평가 임계 완화·metadata 우회·외부 공개 결과 인용은 사용자 명시 승인 후에만 진행.
- **데이터·모델 버전 관리** — 모든 데이터 산출물에 `_meta`/`model_card.json` 동봉 의무. silent invalidation 차단.
- **5단계 비교 리포트** — Phase 5(Compare)의 출력 형식이 4판정(`stable`/`improvement`/`regression`/`inconclusive`)이 아니라 "가설 → 실험 평가 → 실험적 근거 → 가설 평가 → 다음 스텝/사후 검증" 5단계로 고정된다.

---

## 3. 폴더 구성

```
harness-research-template/
├─ README.md                          # 본 문서
├─ 01-instructions.md                 # 1. 지침 레이어 원본 (Role/persona 절 추가)
├─ 02-sensor.md                       # 2. 센서 레이어 원본 (도메인 충분성 지표 슬롯 추가)
├─ 03-test.md                         # 3. 테스트 레이어 원본 (round-trip · smoke · 검증 split 회귀 분류 추가)
├─ 04-evaluation.md                   # 4. 평가 레이어 원본 (부트스트랩 임계 + 5단계 사이클 + 듀얼 트랙)
├─ AGENTS.template.md                 # AGENTS.md 골격 (연구 정직성 절대 규칙 포함)
├─ phaseSample/                       # 4 레이어의 적용본 샘플
│  ├─ 01-instructions.md
│  ├─ 02-sensor.md
│  ├─ 03-test.md
│  └─ 04-evaluation.md
├─ skillSample/                       # 연구 트랙 skill 샘플
│  ├─ hypothesis-registry.md          # 가설 사전 등록·사용자 승인 게이트
│  ├─ data-versioning.md              # _meta·model_card.json·정합성 검증
│  ├─ reproducibility-checklist.md    # NeurIPS 14항목 + 평가 지표 정의 사전
│  ├─ research-journal.md             # 일자별 사람-가독 일지
│  ├─ workaround-tracking.md          # 우회 ledger (W-id)
│  └─ sufficiency-metric.md           # 도메인 충분성 지표 (coverage 대체)
└─ evalSample/                        # 평가 파이프라인 5단계
   └─ eval-compare.md                 # 5단계 사이클 출력 (다른 4개는 일반 템플릿 재사용)
```

다음 파일은 일반 템플릿에서 그대로 가져온다 (변경 불필요):

- `frontmatterSample.md` → [`../harness-template/frontmatterSample.md`](../harness-template/frontmatterSample.md)
- `changeObligationSample.md` → [`../harness-template/changeObligationSample.md`](../harness-template/changeObligationSample.md)
- `CLAUDE.template.md` → [`../harness-template/CLAUDE.template.md`](../harness-template/CLAUDE.template.md) (얇은 `@AGENTS.md` 포인터)
- `skillSample/flaky-handling.md` → [`../harness-template/skillSample/flaky-handling.md`](../harness-template/skillSample/flaky-handling.md)
- `evalSample/eval-collect.md`, `eval-normalize.md`, `eval-grade.md`, `eval-aggregate.md` → [`../harness-template/evalSample/`](../harness-template/evalSample/) (단 본 템플릿 `evalSample/eval-compare.md`만 5단계 사이클 형식으로 재정의)

---

## 4. 도입 절차

연구 프로젝트는 다음 순서로 적용한다. 모든 단계를 한 번에 갖추지 않아도 된다 — 평가 표본이 누적되면서 점진 가동한다.

1. **AGENTS.md 작성** — `AGENTS.template.md`를 프로젝트 루트의 `AGENTS.md`로 복사하고 `<...>` placeholder를 프로젝트 컨텍스트로 교체한다. `CLAUDE.template.md`(일반 템플릿)를 `CLAUDE.md`로 복사한다.
2. **Role(§2)·핵심 가설 등록** — `phaseSample/01-instructions.md` §2를 적용해 Agent의 페르소나·우선순위를 고정한다. 1차 핵심 가설을 `evals/hypotheses/H-YYYY-001.md`에 사전 등록한다 ([`skillSample/hypothesis-registry.md`](./skillSample/hypothesis-registry.md)).
3. **데이터·모델 버전 관리 가동** — 새 데이터 산출물·체크포인트를 만들기 전에 [`skillSample/data-versioning.md`](./skillSample/data-versioning.md)의 `_meta`/`model_card.json` 규약을 적용한다.
4. **센서·테스트 phase 적용** — `phaseSample/02-sensor.md`·`03-test.md`를 복사하고 프로젝트의 변환기·파서·평가기 경로로 채운다. 도메인 충분성 지표(parse success·calibration·domain-specific metric)는 [`skillSample/sufficiency-metric.md`](./skillSample/sufficiency-metric.md)로 정의한다.
5. **연구일지 시작** — 학습·추론·평가 1회 이상 실행한 날부터 [`skillSample/research-journal.md`](./skillSample/research-journal.md)로 `reports/<YYYY-MM-DD>.md`를 누적한다.
6. **우회 ledger 가동** — 정공법 실패 우회를 발견 즉시 [`skillSample/workaround-tracking.md`](./skillSample/workaround-tracking.md)로 등록한다(append-only).
7. **평가 파이프라인 점진 가동** — `phaseSample/04-evaluation.md`를 복사한 뒤 `evalSample/*`을 `.claude/skills/eval-*/SKILL.md`로 분배한다. 처음에는 Collect만 가동하고, trial ≥ 20개 + 비교 가능 스냅샷 ≥ 2개가 누적되면 Compare 단계의 5단계 리포트를 본격 가동한다 (`04-evaluation.md §3-1`의 부트스트랩 임계).
8. **외부 공개 직전 점검** — 논문·발표·README 작성 직전에 [`skillSample/reproducibility-checklist.md`](./skillSample/reproducibility-checklist.md) 14+1항목을 점검한다. `status: open` + `severity: critical` 우회가 1개라도 있으면 외부 공개를 보류한다.

---

## 5. 핵심 운영 원칙

본 템플릿이 일반 SW 템플릿과 공통으로 따르는 운영 원칙은 [`../harness-template/README.md §6`](../harness-template/README.md)을 그대로 상속한다. 본 변형은 추가로 다음을 강제한다.

- **연구 정직성이 첫 번째 우선순위**이다. 효율성·편의성은 연구 정직성과 충돌하면 양보된다 — `phaseSample/01-instructions.md §2-2`.
- **결과를 본 뒤 가설을 손질하지 않는다**(HARKing 차단). 가설 본문은 append-only이며, 수정·폐기는 사용자 승인 후 새 h_id로만 가능하다 — `skillSample/hypothesis-registry.md §4`.
- **회귀·개선 판정은 5단계 리포트 외 위치에서 단정하지 않는다**. 일지·커밋 메시지·다른 skill 문서에서 "성능이 향상되었다"는 표현을 단정으로 쓰지 않는다 — `evalSample/eval-compare.md §6`.
- **정공법 실패 우회는 발견 즉시 ledger에 등록한다**. 외부 공개 직전 일괄 등록은 append-only 의미를 잃는 것이다 — `skillSample/workaround-tracking.md §7`.
- **단일 샘플 시각화·단일 trial 결과로 결론 내리지 않는다**. 최소한 검증 split 전체 또는 다중 trial 분포·CI·effect size·paired test를 함께 본다 — `skillSample/reproducibility-checklist.md §3-7`.

---

## 6. 최종 정의

연구 트랙 하네스의 목적은 Agent를 더 똑똑하게 만드는 것이 아니라, **연구 정직성·재현성·비교 가능성을 자동화 게이트로 보장**해 외부 인용 가능한 결과를 생산하는 것이다.

- Instructions는 규칙과 Role을 정의한다.
- Sensor는 정합성·충분성·결정성·변경 의무를 게이트로 강제한다.
- Test는 round-trip·smoke·검증 split 회귀를 동적 검증으로 구성한다.
- Evaluation은 5단계 리포트로 가설 수명주기와 회귀를 관찰한다.
- Skill은 가설 등록·우회 ledger·재현성 점검·일지·데이터 버전 관리 같은 연구 의무를 자동화한다.
