---
description: ArtifactRouter 연구 가설을 evals/hypotheses/<h_id>.md로 사전 등록하고, status 전환·supersede는 사용자 승인 게이트를 거쳐 보수적으로만 허용하는 skill이다. HARKing(Hypothesizing After Results are Known) 차단을 목적으로 한다.
metadata:
  scope:
    paths:
      - "evals/hypotheses/**"
  activation:
    keywords:
      - "가설"
      - "hypothesis"
      - "supersede"
      - "사전 등록"
      - "preregister"
    when_to_use: 신규 가설 등록·기존 가설 status 전환·supersede 검토 시. ArtifactRouter Main RQ(H-2026-200~203) 평가 결과를 정리하는 모든 단계에서 참조.
  constraints:
    allowed_tools:
      - "Read"
      - "Write"
      - "Edit"
      - "Bash"
      - "Glob"
    risk_level: high
  artifacts:
    inputs:
      - "Main RQ 정의 (명세 §8)"
      - "사전 정의·기각 조건·표본 요건"
      - "관련 우회·이전 결과·관련 가설"
    outputs:
      - "evals/hypotheses/<h_id>.md (append-only)"
      - "evals/hypotheses/_index.md"
---

# Hypothesis Registry — ArtifactRouter

본 skill은 [`docs/harness-research-template/skillSample/hypothesis-registry.md`](../../../docs/harness-research-template/skillSample/hypothesis-registry.md) 의 가설 사전 등록 절차를 본 프로젝트(ArtifactRouter)에 적용한다.

---

## 1. 목적

본 skill은 다음을 보장한다.

- 연구 가설을 **결과를 보기 전에 등록** 한다 (HARKing 차단).
- 가설 본문은 영구 보존(append-only).
- `status` 전환·supersede 는 **사용자 승인 게이트** 를 거친다.
- AGENTS.md §1 핵심 가설과 등록 가설은 동기화된다.

### 1-1. 본 skill이 맡지 않는 것

- 가설별 실험 평가 — [`eval-compare SKILL`](../eval-compare/SKILL.md).
- 가설 결과의 정량 산출 — [`reproducibility-checklist SKILL §3`](../reproducibility-checklist/SKILL.md).
- 일지 작성 — [`research-journal SKILL`](../research-journal/SKILL.md).

---

## 2. 등록 형식

각 가설은 `evals/hypotheses/<h_id>.md` 파일로 작성. h_id 는 `H-YYYY-NNN` 형식 (예: H-2026-200).

본 프로젝트의 h_id 시리즈:
- **H-2026-200~209**: ArtifactRouter Main RQ.
- **H-2026-210~299**: 후속 가설 (추가 motion category, ablation 등).

### 2-1. 본문 구조

```markdown
---
h_id: H-2026-NNN
status: active
registered_at: YYYY-MM-DDTHH:MM:SSZ
parent: <상위 가설 (있으면)>
supersedes: <대체된 가설 (있으면)>
related: [<관련 가설>]
domain: <영역 — orchestration / routing / generator-transfer / no-harm 등>
track_scope: [<적용 generator quality-tier — G1/G2/G3>]
---

# H-2026-NNN — <한 줄 가설>

## 본문 (등록 시점 그대로)

<가설 본문. 이 부분은 영구 보존. 변경 금지>

## 사전 정의

### 측정할 지표

- <명세 §9.3 의 metric 인용, 예: NetGain · ArtifactReduction · FidelityLoss · FID_motion>

### 임계 효과 크기

- <예: NetGain median Δ ≥ 0.05, paired Wilcoxon p < 0.05, Cohen's d ≥ 0.3>

### 비교 대상

- <baseline: B1-B8 (명세 §9.1) 중 어느 것>
- <generator quality-tier: G1/G2/G3>
- <n_seeds·n_samples 표본 요건>

### 기각 조건 (다음 중 하나라도 충족)

- <명확한 기각 시그니처. 명세 §11 Stop 기준 또는 가설 특화>

### 표본 요건

- trial ≥ N per (generator × seed)
- 또는 명세 §11 Go 기준 충족 표본

## 평가 대상 산출물

- generator: <G1/G2/G3 + version>
- evaluator config: <hash>
- tool registry config: <hash>
- orchestrator: <rule-based / supervised / contextual-bandit>
- 사용할 split: <train/val/test>
- 비교 절차: [`eval-compare SKILL §6`](../eval-compare/SKILL.md) 5단계 리포트

## Stage 정의 (선택)

본 가설이 단계적 검증이 필요한 경우.

- Stage 0: <첫 검증 단계>
- Stage 1: <확장 단계>
- Stage 2: <본격 평가>

## 변경 이력 (append-only)

- YYYY-MM-DD — 등록.
```

---

## 3. 등록 절차

1. `evals/hypotheses/_index.md` 에서 다음 h_id 확인.
2. `evals/hypotheses/H-2026-NNN.md` 파일 §2 형식 작성.
3. 사전 정의 모든 항목 충족 확인:
   - 측정 지표 (metric 정의 단일 출처 인용).
   - 임계 효과 크기 (effect size · p-value 사전 명시).
   - 비교 대상 (baseline 정의).
   - 기각 조건.
   - 표본 요건.
4. `_index.md` 의 활성 가설 표에 한 줄 추가.
5. 등록 자체는 사용자 승인 불필요 (append-only).

---

## 4. Status 전환 — 사용자 승인 게이트

### 4-1. 가능한 status 전환

- `active` → `supported` (가설 검증 완료, 기각 조건 미충족).
- `active` → `rejected` (기각 조건 충족).
- `active` → `superseded` (새 가설로 대체).
- `active` → `withdrawn` (가설 자체 폐기).

### 4-2. 절차

1. **draft 작성**: 전환 사유 + 5단계 리포트 인용 + 새 가설 본문 (supersede 시) 작성.
2. **사용자 승인 요청**: 명시적 메시지 ("진행" / "OK" / "approve") 로만 인정.
3. **승인 시 적용**: 기존 가설 frontmatter `status` 변경, 변경 이력에 한 줄 추가, `_index.md` 갱신.
4. **본문 보존**: 기존 가설 본문은 **수정 금지** (append-only). 변경은 새 H-id 로 새 가설 promote.

### 4-3. supersede 절차

1. 기존 가설 (예: H-2026-200) 의 결과가 contradicts → 새 가설 (예: H-2026-220) 작성.
2. 새 가설 frontmatter 에 `supersedes: H-2026-200` 기록.
3. 사용자 승인 후 H-2026-200 의 `status: superseded`, supersede target 명시.
4. `_index.md` 의 종결 가설 표로 이동.

### 4-4. HARKing 차단

다음 조건 모두 만족 시에만 사용자 승인 요청:

- 사전 정의된 기각 조건 명확 충족 (또는 평가 절차 결함 식별).
- 변경 사유가 [`eval-compare SKILL`](../eval-compare/SKILL.md) 5단계 리포트에 인용.
- 새 draft 가 본 SKILL §2 형식.

위 조건 미충족 + 사용자 승인 없이 status 전환·본문 수정 금지.

---

## 5. `_index.md` 양식

```markdown
# Hypothesis Index

## 활성 가설

| h_id | 한 줄 요약 | 등록 | domain | track_scope |
|---|---|---|---|---|
| H-2026-200 | ... | 2026-MM-DD | orchestration | G1/G2/G3 |

## 종결 가설

| h_id | status | 종결 일 | 후속 | 사유 |
|---|---|---|---|---|

## 승계 그래프

```
H-2026-200 (active) ─...
```
```

---

## 6. 금지 규칙

- 결과 본 뒤 본문 사후 수정 금지 (HARKing).
- status 전환 단독 진행 금지 (사용자 승인 필수).
- 사전 정의 항목 누락한 가설 등록 금지.
- 단일 trial 결과로 supports/contradicts 판정 금지 ([`AGENTS.md §3-9`](../../../AGENTS.md), [`AGENTS.md §6-7`](../../../AGENTS.md)).

---

## 7. 본 프로젝트 활성 가설 사전 등록 (필수 항목)

본 프로젝트 시작 시점에 다음 가설 사전 등록 필수:

- **H-2026-200** (Main RQ1+RQ2): Artifact-conditioned tool selection + closed-loop refinement 가 fixed post-processing 보다 NetGain 우위.
- **H-2026-201** (Main RQ3): Learnable Routing — supervised/contextual-bandit selector 가 rule-based 대비 net gain 개선.
- **H-2026-202** (Main RQ4): Generator transfer — zero-shot + small-calibration.
- **H-2026-203** (secondary, Analysis Q4): No-harm on high-quality generator (FID_motion·correction magnitude·user study non-inferiority).

각 가설의 본문은 명세 §4.1 (Core Hypothesis) + §8 (Main RQ) 를 단일 출처로 인용.
