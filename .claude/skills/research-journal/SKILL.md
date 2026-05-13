---
description: 일자별 연구일지(reports/<YYYY-MM-DD>.md)를 하네스 평가 파이프라인과 연결해 작성·검증하는 skill이다. 시각화(tool call trace + before/after motion)·정량 지표(generator·tool registry 별)·실험 메타·raw record 링크를 모두 포함한다.
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
      - "일지"
    when_to_use: 실험·refinement·평가를 1회 이상 수행한 날 마무리 시점. 일지 없는 날 핵심 경로 변경 커밋 금지.
  constraints:
    allowed_tools:
      - "Read"
      - "Write"
      - "Edit"
      - "Bash"
      - "Glob"
    risk_level: medium
---

# Research Journal — ArtifactRouter

본 skill은 [`docs/harness-research-template/skillSample/research-journal.md`](../../../docs/harness-research-template/skillSample/research-journal.md) 의 절차를 본 프로젝트에 적용한다.

---

## 1. 목적

다음을 보장한다.

- 실험 1회당 사람-가독 일지 1개 + 기계 처리 raw record 1개 동시 보존.
- 일지에 시각화 (tool call trace + before/after motion) · 정량 지표 · 실험 메타 · 재현 명령어 포함.
- 일지와 raw record 가 cross-link 되어 자가 수정 메타 규칙 ([`AGENTS.md §3-7`](../../../AGENTS.md)) 의 근거 추적 끊기지 않음.

### 1-1. 본 skill이 맡지 않는 것

- raw record 직렬화 — [`eval-collect SKILL`](../eval-collect/SKILL.md) (작성 후) 또는 직접 작성.
- 회귀·개선 판정 — [`eval-compare SKILL`](../eval-compare/SKILL.md).
- 우회 ledger 등록 — [`workaround-tracking SKILL`](../workaround-tracking/SKILL.md).

---

## 2. 작성 시점·범위

다음 조건 중 하나라도 해당 시 일지 작성:

- 핵심 경로 ([`02-sensor.md §2`](../../rules/phase/02-sensor.md)) 파일 변경.
- generator inference·evaluator·correction tool 또는 refinement loop 를 1회 이상 실행.
- evaluator·correction_tool registry 또는 orchestrator 알고리즘 변경.
- 회귀·개선 신호 관찰.

문서·README만 변경 시 일지 생략 가능. 단 AGENTS.md·phase·skill 변경은 일지 대상.

---

## 3. 필수 항목

`reports/<YYYY-MM-DD>.md` 는 다음 항목을 모두 포함.

### 3-1. 정량 지표 (Generator · Tool Registry 별 분리)

다음 표 양식 (직전 일지 대비 변동 `Δ` 함께).

```markdown
| Generator | tool registry | orchestrator | ArtifactReduction | FidelityLoss | NetGain | tool calls |
|---|---|---|---|---|---|---|
| G1 (MDM v0.3) | full registry | rule_based | 0.123 | 0.012 | 0.099 (Δ +0.005) | 3.2 |
| G2 (MotionGPT v1) | full registry | rule_based | 0.234 | 0.025 | 0.187 | 4.5 |
| G3 (local_lora) | full registry | rule_based | 0.412 | 0.087 | 0.281 | 6.1 |
```

회귀·개선·H-2026-200~203 supports/contradicts 단정은 [`eval-compare SKILL`](../eval-compare/SKILL.md) 5단계 리포트에 위임. 일지에서 단정 금지.

지표 정의는 [`reproducibility-checklist SKILL §3`](../reproducibility-checklist/SKILL.md) 단일 출처 인용.

### 3-2. 시각화 이미지

- **3D skeleton 애니메이션 — before vs after refinement** (다중 sample 우선).
- **Tool call trace 시각화** — refinement loop 의 step별 (tool · target · score) bar chart 또는 timeline.
- **Artifact metric 곡선** — refinement step별 metric 변화.
- **per-frame error 분포** — distribution plot.

저장 위치: `reports/figures/<YYYY-MM-DD>/<descriptive_name>.png|.gif`. 일지 본문에 embed:

```markdown
![G3 walking refinement (before gray vs after orange, tool calls bar)](figures/2026-MM-DD/g3_walking_refinement.gif)
```

단일 sample 시각화만 첨부 금지 — [`AGENTS.md §3-9`](../../../AGENTS.md). 다중 sample (≥3) 또는 다중 generator (G1+G2+G3) 함께.

### 3-3. 실험 메타

- **사용 명령어** (재현 가능 형태).
- **사용 시드** (torch / numpy / random).
- **사용 conda 환경**.
- **GPU·CUDA·hardware 정보** (가능한 경우).
- **사용 generator id + version hash + checkpoint hash**.
- **사용 evaluator config hash · correction_tool config hash · orchestrator config hash**.
- **변경 파일 목록** (`git diff --name-only HEAD~1`).
- **신규 파일 목록** (`git diff --diff-filter=A --name-only`).
- **trial 단위 통계**: trial 수, generator 별 분포, seed 별 분포.

### 3-4. raw record cross-link

```markdown
## raw record

- `evals/raw/<timestamp>_<task_id>_<trial_id>.json`
- task_id: `<task_id>`
- trial_id: `<trial_id>`
- 활성 가설: H-2026-200, H-2026-201, ... (`evals/hypotheses/<h_id>.md`)
- 부수 우회: W-2026-NNN, ...
```

raw record 없으면 일지 단독 기록은 회귀 판정 근거 인용 불가.

### 3-4-1. 우회 발견 (workaround-tracking ledger)

```markdown
## 우회 발견 (workaround-tracking ledger)

오늘 등록·전환된 항목:
- W-2026-NNN (severity: critical) — <한 줄 요약> — `evals/workarounds/W-2026-NNN.md`
- (없으면 "(없음)" 표기)
```

### 3-5. 실패한 시도 (Negative Results)

오늘 폐기·중단·발산한 시도. 빈 채로 두지 마라 (없으면 "없음" 표기).

```markdown
## 실패한 시도

- **<요약>** — 어떤 시도였는지.
  - 수행 명령: <재현 가능 명령>.
  - 실패 양상: refinement loop 발산 / tool 적용 후 artifact 증가 / orchestrator decision 비정상 / format 불일치 등.
  - 정량 근거: <metric 또는 raw record 경로>.
  - 폐기 사유.
  - 관련 가설.
```

### 3-6. ArtifactRouter 특화 — Tool Call Trace 요약

각 refinement 의 tool call sequence + step별 score change 요약:

```markdown
## Tool Call Trace (G3 sample 0)

| step | tool | target | strength | score_before | score_after | Δ |
|---|---|---|---|---|---|---|
| 1 | foot_lock | right_foot | medium | 0.521 | 0.412 | -0.109 |
| 2 | velocity_smoothing | upper_body | small | 0.412 | 0.398 | -0.014 |
| 3 | bone_projection | full_body | small | 0.398 | 0.395 | -0.003 |
| STOP | — | — | — | — | — | — |
```

---

## 4. 작성 절차

1. 오늘 변경 파일·실행 명령·결과 산출물 수집.
2. raw record 먼저 생성 (Collect 단계).
3. `reports/figures/<YYYY-MM-DD>/` 생성 + 시각화 저장 (다중 sample 우선).
4. `reports/<YYYY-MM-DD>.md` 본 SKILL §3 항목 충족.
5. §5 검증 체크리스트 통과 확인.
6. 통과 시 → 커밋.

---

## 5. 검증 체크리스트

일지 커밋 전 다음 모두 충족:

- [ ] §3-1 정량 지표 표가 generator·tool registry·orchestrator 별 분리.
- [ ] §3-1 표에 [`AGENTS.md §3-6`](../../../AGENTS.md) 항목 모두 포함.
- [ ] §3-2 시각화 ≥ 1개, 다중 sample 또는 다중 generator 기반.
- [ ] §3-3 실험 메타에 재현 가능 명령어 + 시드 포함.
- [ ] §3-4 raw record cross-link 유효 + 활성 가설 id 명시.
- [ ] §3-4-1 우회 발견 절 채워짐 ("없음" 표기 포함).
- [ ] §3-5 실패한 시도 절 채워짐 ("없음" 표기 포함).
- [ ] §3-6 tool call trace 요약 포함.
- [ ] 회귀·개선·가설 supports/contradicts 본문 단정 안 함 (eval-compare 위임).

---

## 6. 금지 규칙

- 시각화 없이 일지 마무리 금지 (§3-2 의무).
- 단일 sample 시각화만으로 결론 금지.
- generator 평균값 단일 행 기록 금지 (G1/G2/G3 분리 보존).
- 직전 일지 수치 덮어쓰기 금지 (새 행 추가).
- raw record 없이 회귀·개선 단정 금지.
- 100MB 이상 단일 시각화는 git LFS 또는 외부 저장소 검토.
