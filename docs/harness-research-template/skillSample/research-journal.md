---
description: 일자별 연구일지(reports/<YYYY-MM-DD>.md)를 하네스 평가 파이프라인과 연결해 작성·검증하는 skill이다. 시각화·정량 지표·실험 메타·raw record 링크·우회·실패한 시도를 모두 포함한다.
metadata:
  scope:
    paths:
      - "reports/**/*.md"
      - "reports/figures/**"
      - "evals/raw/**/*.json"
  activation:
    keywords:
      - "연구일지"
      - "daily report"
      - "research journal"
      - "일지 작성"
      - "experiment log"
      - "시각화 첨부"
    when_to_use: 실험·학습·추론·평가를 1회 이상 수행한 날 마무리 시점에 참조한다. 일지가 없는 날 핵심 경로 변경을 커밋하지 않는다.
  constraints:
    allowed_tools: ["Read", "Write", "Edit", "Bash", "Glob"]
    risk_level: medium
  artifacts:
    inputs:
      - "오늘 수행한 명령어·스크립트·파라미터"
      - "트랙별 평가기 출력 (충분성·모델 메트릭)"
      - "변경 파일 목록 (git diff --name-only)"
      - "시각화 산출물"
      - "raw record 경로 (evals/raw/<timestamp>.json)"
    outputs:
      - "reports/<YYYY-MM-DD>.md (사람 가독 일지)"
      - "reports/figures/<YYYY-MM-DD>/*.png|.gif (시각화)"
      - "raw record와의 cross-link"
---

# Research Journal Skill — Research Track Template

본 skill은 [`04-evaluation.md §4`](../04-evaluation.md) 평가 파이프라인의 **사람-가독 트랙**으로서 연구일지를 운영하는 절차이다. 기계 처리용 raw record는 eval-collect가 담당하고, 본 skill은 같은 작업을 사람이 읽을 수 있는 정성·시각·서술 형태로 보존한다.

본 skill은 [`01-instructions.md §2`](../01-instructions.md)의 문장 형식 규칙을 그대로 따른다.

---

## 1. 목적

본 skill은 다음을 보장한다.

- 실험 1회당 사람-가독 일지 1개와 기계 처리용 raw record 1개가 동시에 남는다.
- 일지에 시각화·정량 지표·실험 메타·재현 명령어가 모두 포함되어 논문·리팩토링·재현 시 참고 가능하다.
- 일지와 raw record가 cross-link되어 자가 수정 메타 규칙(AGENTS.md §3-7)의 근거 추적이 끊기지 않는다.

### 1-1. 본 skill이 맡지 않는 것

- raw record 직렬화 — eval-collect.
- 회귀·개선 판정 — eval-compare.
- 작성 양식 자체 — `reports/report_rules.md`(적용본의 원본 규약).

본 skill은 양식 위에서 **하네스 흐름과의 연결·시각화 의무·트랙별 분리·우회·실패한 시도**를 추가로 강제한다.

---

## 2. 작성 시점·범위

연구일지는 다음 조건 중 하나라도 해당하면 작성한다.

- 핵심 경로([`02-sensor.md §2`](../02-sensor.md))에 속한 파일을 변경했다.
- 학습·추론·평가를 1회 이상 실행했다.
- 실험 트랙·체크포인트·평가 지표 중 하나 이상이 새로 도입되었다.
- 회귀 또는 개선 신호([`04-evaluation.md §6`](../04-evaluation.md))를 관찰했다.

문서·README만 변경한 경우 일지를 생략할 수 있다. 단, AGENTS.md·phase·skill 변경은 일지 대상(자가 수정 메타 규칙의 흔적).

---

## 3. 필수 항목

`reports/<YYYY-MM-DD>.md`는 적용본의 `reports/report_rules.md` §3 섹션 구조를 따르되, 본 skill은 다음 5+1 항목을 **추가로** 요구한다.

### 3-1. 정량 지표 (트랙별 분리)

다음 표를 일지에 포함한다(트랙·체크포인트·split별).

```markdown
| 트랙 | 체크포인트 | split | <충분성 지표> | <도메인 메트릭 1> | <도메인 메트릭 2> | ... |
|---|---|---|---|---|---|---|
| <트랙 A> | <ckpt path> | val | 0.93 | <값> | <값> | ... |
| <트랙 B> | <ckpt path> | val | 0.95 | <값> | <값> | ... |
```

직전 일지 대비 변동(`Δ`)을 함께 기록한다(예: `<지표> 0.142 (Δ -0.008)`). 변동의 통계적 의미는 eval-compare가 판정한다 — 일지 본문에서 회귀·개선을 단정하지 마라.

지표는 AGENTS.md §3-6 평가 기록 의무를 모두 충족해야 한다.

### 3-2. 시각화 이미지

도메인에 적합한 시각화(예: 시퀀스 prediction 오버레이·메트릭 곡선·실패 사례·분포)를 포함한다.

- 저장 위치: `reports/figures/<YYYY-MM-DD>/<설명>.png` 또는 `.gif`.
- 본문에 다음 형식으로 임베드.

```markdown
![<설명>](figures/<YYYY-MM-DD>/<filename>.png)
```

- 시각화는 **다중 샘플** 결과를 우선한다. 단일 샘플 시각화만 첨부하지 마라 — AGENTS.md §3-9.
- 정성 보조 지표(분포 꼬리·drift 추세 등)에 대한 시각화는 [`04-evaluation.md §5-2-1`](../04-evaluation.md)의 항목과 매핑되어야 한다.

### 3-3. 실험 메타

다음 항목을 표 또는 리스트로 기록.

- 사용 명령어(인자 포함, 그대로 재실행 가능한 형태).
- 사용 시드(`torch.manual_seed`·`numpy.random.seed`·`random.seed`).
- 사용 conda/venv 환경 이름.
- GPU·CUDA·cuDNN 버전.
- 사용 체크포인트 step.
- 변경 파일 목록(`git diff --name-only HEAD~1`).
- 신규 파일 목록(`git diff --diff-filter=A --name-only`).
- 학습 trial이면 — loss 곡선 또는 최종 loss, 학습 시간.

### 3-4. raw record cross-link

일지 끝에 다음 줄을 추가해 raw record와 양방향 추적을 만든다.

```markdown
## raw record

- `evals/raw/<timestamp>_<task_id>_<trial_id>.json`
- task_id: `<task_id>`
- trial_id: `<trial_id>`
- 활성 가설: <H-id 목록> (`evals/hypotheses/<h_id>.md`)
```

raw record가 없으면 → eval-collect를 먼저 수행한다. 일지 단독 기록은 자가 수정 메타 규칙의 메트릭 근거로 인용할 수 없다.

### 3-4-1. 우회 발견 (workaround-tracking ledger)

오늘 등록·전환된 우회 항목을 한 줄씩 인용한다 — [`workaround-tracking.md §6-3`](./workaround-tracking.md).

```markdown
## 우회 발견 (workaround-tracking ledger)

오늘 등록·전환된 항목:
- <W-id> (severity: <등급>) — <한 줄 요약> — `evals/workarounds/<W-id>.md`
- (없으면 "(없음)"으로 표기)
```

본 절은 발견 즉시 기록 의무가 있는 절이며, **결론 시점에 eval-compare 5단계 리포트의 "부록 A — 우회·간접 해결 ledger 인용"이 본 절을 누적 인용**한다.

### 3-5. 실패한 시도 (Negative Results)

오늘 실험·학습·평가 중 폐기·중단·발산한 시도를 명시. 본 절은 빈 채로 두지 마라(없으면 "없음"으로 표기). negative result 보존은 후속 연구·메타 분석·HARKing 차단의 핵심 자원 — AGENTS.md §3-13.

```markdown
## 실패한 시도

- **<요약>** — 어떤 시도였는지 한 줄.
  - 수행 명령: <재현 가능한 명령>.
  - 실패 양상: 발산 / parse 0% / OOM / 학습 중단 / 결과가 가설 기각 등.
  - 정량 근거: <지표 또는 raw record 경로>.
  - 폐기 사유: <왜 이 시도를 후속에서 더 추구하지 않는지>.
  - 관련 가설(있는 경우): <H-id>.
```

특히 다음을 포함한다.

- 실험 트랙·hyperparam·표현 중 폐기한 후보.
- 시드를 바꿨을 때만 사라지는 결과(flaky 후보 — flaky-handling과 cross-link).
- 가설을 기각한 결과(가설 평가는 eval-compare가 단정하므로, 일지에서는 "기각 후보로 분류" 수준의 사실 기록만 한다).

---

## 4. 작성 절차

1. 오늘 작업의 변경 파일·실행 명령·결과 산출물 수집.
2. eval-collect를 먼저 실행해 raw record 생성.
3. `reports/figures/<YYYY-MM-DD>/`를 생성하고 시각화 산출물 저장(다중 샘플 우선).
4. `reports/<YYYY-MM-DD>.md`를 적용본의 `report_rules.md` §3 섹션 + 본 skill §3 추가 항목으로 작성.
5. §5 검증 체크리스트 통과 확인.
6. 통과하면 → 커밋. 회귀·개선 후속 조치는 eval-compare가 다음 스냅샷에서 처리.

---

## 5. 검증 체크리스트

일지 커밋 전 다음을 모두 만족해야 한다.

- [ ] 적용본 `report_rules.md` §3의 모든 섹션이 채워져 있다(해당 없음은 "없음"으로 표기).
- [ ] §3-1 정량 지표 표에 트랙·체크포인트·split이 분리되어 있다.
- [ ] §3-1 표에 AGENTS.md §3-6의 평가 기록 의무 항목이 모두 들어 있다.
- [ ] §3-2 시각화 이미지가 1개 이상 포함되어 있고 다중 샘플 기반이다.
- [ ] §3-3 실험 메타에 재현 가능한 명령어와 시드가 포함되어 있다.
- [ ] §3-4 raw record cross-link가 유효하고 활성 가설 ID가 명시되어 있다.
- [ ] §3-4-1 우회 발견 절이 채워져 있다("없음" 표기 포함).
- [ ] §3-5 실패한 시도 절이 채워져 있다("없음" 표기 포함).
- [ ] 회귀·개선·가설 supports/contradicts를 본문에서 단정하지 않았다(eval-compare에 위임).
- [ ] 가설 본문 수정·폐기 작업을 사용자 승인 없이 수행하지 않았다(hypothesis-registry §4).

---

## 6. 금지 규칙

- 시각화 첨부 없이 일지를 마무리하지 마라(실험 수행 일은 §3-2 의무).
- 단일 샘플 시각화만으로 일지 결론을 만들지 마라.
- 트랙 평균값을 단일 행으로 표기하지 마라(트랙 비대칭은 eval-aggregate §6의 보존 원칙).
- 직전 일지의 수치를 새 결과로 덮어쓰지 마라.
- raw record 없이 일지 단독으로 회귀·개선을 단정하지 마라.
- 시각화 이미지를 git에 직접 커밋하기 전에 크기를 확인한다. 100MB 이상 단일 파일은 LFS 또는 외부 저장소를 검토한다.

---

## 7. 유지보수 규칙

- 본 skill의 추가 필수 항목(§3-1~§3-5)을 변경했으면 → 적용본 `report_rules.md`와 [`04-evaluation.md §5`](../04-evaluation.md) 보조 지표 절을 동시 갱신.
- 시각화 디렉토리 규약을 변경했으면 → AGENTS.md §7을 동시 갱신.
- 본 skill 결과(일지)가 eval-compare의 회귀 분류 근거로 반복 인용되지 않으면 → 일지 양식이 raw record와 cross-link를 충분히 만들지 못한 신호. §3-4 재검토.

<IMPORTANT>
연구일지는 "오늘 한 일을 적는 일기"가 아니라 **"논문·재현·하네스 자가 수정의 1차 기록"** 이다. 시각화·정량 지표·실험 메타·raw record cross-link·우회·실패한 시도 중 하나라도 빠지면 04-evaluation.md §6의 회귀·개선 판정과 AGENTS.md §3-7 자가 수정 메타 규칙의 근거 사슬이 끊긴다.
</IMPORTANT>
