---
description: NeurIPS reproducibility checklist(Pineau et al. 2021 ML Reproducibility Checklist 2.0)·Dodge et al. 2019 *Show Your Work*의 14+1 항목을 본 트랙에 적응한 점검 절차이며, 각 평가 지표 정의·계산식·해석을 함께 제공한다.
metadata:
  scope:
    paths:
      - "reports/**/*.md"
      - "evals/**"
      - "AGENTS.md"
  activation:
    keywords:
      - "reproducibility"
      - "재현성 체크리스트"
      - "metric definition"
      - "지표 정의"
      - "논문 작성"
      - "NeurIPS checklist"
    when_to_use: 실험 결과를 외부에 공개·발표·논문 게재하기 전, 또는 재현성에 의문이 제기될 때 참조한다. 평가 지표 정의를 공식적으로 인용해야 할 때도 본 skill의 §3을 인용한다.
  constraints:
    allowed_tools: ["Read", "Write", "Edit"]
    risk_level: medium
  artifacts:
    inputs:
      - "AGENTS.md (시스템 컨텍스트·핵심 가설·학습 디폴트)"
      - "evals/hypotheses/<h_id>.md"
      - "evals/raw/*.json·evals/snapshots/*.json"
      - "<체크포인트 디렉토리>/model_card.json"
      - "evals/workarounds/_index.md"
    outputs:
      - "reports/checklists/<period>_<scope>.md (체크리스트 점검 결과)"
      - "각 평가 지표 정의 사전(본 문서 §3) 인용"
---

# Reproducibility Checklist Skill — Research Track Template

본 skill은 Pineau et al. 2021(*Improving Reproducibility in Machine Learning Research — NeurIPS Reproducibility Checklist 2.0*)·Dodge et al. 2019(*Show Your Work: Improved Reporting of Experimental Results*)의 점검 항목을 본 트랙에 적응한 결과이다. 각 평가 지표의 정의·계산식·해석을 함께 사전(dictionary)으로 묶어 외부 인용·논문·발표 시점의 단일 출처로 사용한다.

본 skill은 [`01-instructions.md §2`](../01-instructions.md)의 문장 형식 규칙을 그대로 따른다.

---

## 1. 목적

본 skill은 다음을 보장한다.

- 외부 공개 시점(논문·발표·README)에 코드·데이터·모델·hyperparam·평가 절차가 모두 명시 인용 가능하다.
- 평가 지표의 정의·계산식·해석이 단일 위치(§3)에 정의되어 모든 인용이 동일 기준을 가리킨다.
- 본 프로젝트의 결과가 다른 연구자에게 재현 가능하다.

### 1-1. 본 skill이 맡지 않는 것

- 가설 수명주기 — [`hypothesis-registry.md`](./hypothesis-registry.md).
- 데이터·모델 metadata 동봉 — [`data-versioning.md`](./data-versioning.md).
- 회귀 판정·통계적 비교 — [`evalSample/eval-compare.md`](../evalSample/eval-compare.md).
- 일지 본문 — [`research-journal.md`](./research-journal.md).

본 skill은 **점검 항목 목록**과 **지표 정의 사전**만 제공한다.

---

## 2. 체크리스트 항목 (14+1)

외부 공개 시점에 모든 항목이 충족되었는지 점검한다. 각 항목은 (a) 본 skill이 강제, (b) 다른 skill이 강제, (c) 외부 인용 시 명시 의무로 분류된다.

### 2-1. 코드·데이터 공개

1. **코드 저장소 URL** — git remote 또는 archive(예: Zenodo DOI).
2. **데이터 라이선스** — 상위 데이터셋 라이선스와 sub-dataset 라이선스 명시.
3. **본 프로젝트 라이선스** — 코드·문서·모델 가중치 각각.

### 2-2. 데이터 전처리

4. **전처리 정책** — windowing/split/feature set/normalization 파라미터. 출처: `_meta` 레코드.
5. **좌표계·정규화** — 정의와 단위.
6. **feature/joint set** — 사용된 입력 변수.
7. **train/val/test split** — 정의와 sample 수.

### 2-3. 모델·학습

8. **베이스 모델** — 모델 이름·hash·tokenizer hash.
9. **hyperparam 전체** — 모든 학습/추론 hyperparam.
10. **시드·결정성** — `torch.manual_seed`/`numpy.random.seed`/`random.seed` + 도메인 동등 + `torch.use_deterministic_algorithms` + cuDNN 설정.
11. **하드웨어·소프트웨어** — GPU 모델, CUDA·cuDNN, 핵심 라이브러리 버전.
12. **학습 비용** — 총 step, GPU시간, 추정 에너지(Dodge et al. 2019 요구).

### 2-4. 평가·통계

13. **평가 절차** — n_seeds, n_trials, 사용 split, 지표 정의(본 문서 §3 인용). paired test 사용 시 검정 통계량과 p-value 명시.
14. **negative result** — 폐기한 표현·발산한 학습·기각된 가설(`status: rejected|withdrawn`) 명시.

### 2-5. 우회·재현 위험

15. **우회·간접 해결 ledger 점검** — `evals/workarounds/_index.md`에서 `status: open` + `severity: critical`인 항목이 **0개**여야 외부 공개 가능. 모든 W-id가 본 보고에서 인용되었는지 cross-link 확인. material 항목은 환경 매트릭스와 함께 명시 — [`workaround-tracking.md §6-2`](./workaround-tracking.md).

---

## 3. 평가 지표 정의 사전

본 절은 본 프로젝트의 모든 평가 지표를 단일 출처로 정의한다. 외부 인용은 항상 본 절을 참조한다. 본 템플릿은 **연구 도메인에 흔히 등장하는 지표 family**의 정의 패턴을 제공하며, 적용본은 도메인 지표를 본 형식으로 추가한다.

### 3-1. 충분성 지표 (parse success / domain pass rate 등)

- **정의** — 모델이 생성한 응답 중 도메인 파서·검증기가 통과시킨 비율.
- **계산식** — `pass_rate = passed / attempted`.
- **단위** — 비율 ∈ [0, 1].
- **임계 정책** — [`sufficiency-metric.md §2`](./sufficiency-metric.md).
- **금지 사항** — 모델 출력의 후처리(정규식 보정 등)로 본 지표를 인위적으로 끌어올리지 마라.

### 3-2. 도메인 모델 메트릭 (예: MPJPE / MSE / F1 / AUROC / BLEU 등)

적용본은 도메인 지표의 **정의·계산식·단위·해석·트랙별 처리·참고 문헌**을 본 절에 한 번만 정의한다. 예시 패턴:

```markdown
### 3-X. <지표 이름> (Mean Per Joint Position Error 등)

- 정의: <한 줄 정의>.
- 계산식: <LaTeX 또는 의사 코드>.
- 단위: <단위>.
- 트랙별 처리: <비교 트랙 간 차이 — 예: delta는 누적 적분 후 absolute 공간에서 측정>.
- 해석: <낮을수록 좋다 / 분포의 어떤 측면을 보는가>.
- 참고: <원논문 인용>.
- 금지 사항: <지표 정의 변경 시 동시 갱신 의무 — aggregation_rule_version 인상>.
```

### 3-3. horizon-wise / segment-wise error

- **정의** — 시퀀스/시간/segment 단위로 산출한 모델 메트릭.
- **해석** — drift 누적·시간 의존 오차의 분포.

### 3-4. 분포·꼬리 지표

- **정의** — class-wise / joint-wise / strata-wise 평균 + 분포의 꼬리(p95·p99) 추가 보고.
- **해석** — 평균값 단독은 분포의 비대칭을 가린다.

### 3-5. effect size·CI·paired test (통계적 보조 지표)

- **effect size** — 두 트랙 또는 두 체크포인트의 차이를 단위 없는 크기로 표기. Cohen's d 또는 ratio of means.
- **bootstrap confidence interval** — N=1000 resampling 기준 95% CI. n_trials 또는 n_seeds 표본에 적용.
- **paired test** — Wilcoxon signed-rank test. 동일 split·동일 시드 페어링한 두 트랙 결과에 적용. p-value < 0.05 + effect size 임계 충족이 있어야 의미 있는 차이.
- **참고** — Demšar 2006(*Statistical Comparisons of Classifiers over Multiple Data Sets*).
- **사용 위치** — [`evalSample/eval-compare.md §3·§5`](../evalSample/eval-compare.md) 5단계 출력 중 §3(실험적 근거)·§4(가설 평가).

---

## 4. 점검 절차

외부 공개 시점에 다음 순서로 점검한다.

1. `evals/hypotheses/_index.md`에서 활성·종결 가설 모두 식별.
2. 외부 공개 산출물(논문·README·블로그)이 어떤 가설을 인용하는지 확인.
3. 인용된 가설에 대해 §2의 14+1 항목 점검.
4. 평가 지표를 인용하는 모든 위치가 §3을 가리키는지 확인(자체 정의 금지).
5. `reports/checklists/<period>_<scope>.md`에 점검 결과 기록.
6. 누락 항목이 있으면 → 외부 공개 전 보완.

### 4-1. 출력 형식

```markdown
# Reproducibility Checklist — <기간 또는 외부 공개 대상>

- 점검 시점: <ISO 8601>
- 점검 대상: <논문 v1.0 / 발표 / README v2 등>
- 인용 가설: <h_id 목록>

## 항목별 점검

| # | 항목 | 출처 | 충족 여부 | 비고 |
|---|---|---|---|---|
| 1 | 코드 저장소 URL | <URL> | ✓ | |
| ... | | | | |
| 15 | 우회 ledger 점검 | evals/workarounds/_index.md | ✓ | open+critical 0개 확인 |

## 누락 항목

- <항목명>: <누락 원인 및 보완 계획>

## 지표 정의 인용

본 보고에 사용된 모든 평가 지표는 `.claude/skills/reproducibility-checklist/SKILL.md §3`을 따른다.
```

---

## 5. 금지 규칙

- 평가 지표 정의를 본 §3 외 위치에서 새로 작성하지 마라.
- §2 항목 중 하나라도 누락된 결과를 외부에 공개하지 마라.
- 체크리스트를 결과에 맞게 사후 수정하지 마라(append-only). 누락이 있으면 누락 사실 자체를 기록한다.
- 가설 인용 없이 외부 결과를 발표하지 마라.

---

## 6. 유지보수 규칙

- §3에 새 지표를 추가했으면 → [`04-evaluation.md §5-2`](../04-evaluation.md)·트랙 평가기·`aggregation_rule_version`을 동시 갱신.
- §2 점검 항목을 변경했으면 → 직전 체크리스트와의 호환성을 비고에 명시.
- 외부 인용에서 본 §3과 다른 정의가 발견되면 → 외부 산출물을 정정하고 본 §3을 단일 출처로 유지.

<IMPORTANT>
재현성 체크리스트는 "공개 직전 한 번 채우는 양식"이 아니라 **"실험 시작 시점부터 누적해 외부 공개 시점에 통합되는 메타데이터의 합"** 이다 — Pineau et al. 2021. 누락 항목이 외부 공개 후 발견되면 AGENTS.md §3-7 자가 수정 메타 규칙의 근거 사슬이 사후적으로 끊긴다.
</IMPORTANT>
