---
description: 산출물 (evaluator / tool / oracle / baseline / RL policy / snapshot / framework doc) 의 commit 직전 의무 5-step Intent-Reconciliation agentic loop. Intent (사용자 directive / hypothesis / paper) 와 Actual (실제 metric / behavior) 의 정량·정성 비교 → aligned / partial / misaligned / scope_creep / unintended_side_effect 분류 → 일지 박제 + 사용자 보고.
metadata:
  scope:
    paths:
      - "evaluators/**/*.py"
      - "correction_tools/**/*.py"
      - "generators/**/*.py"
      - "orchestrator/**/*.py"
      - "tools/oracle_*.py"
      - "tools/baseline_*.py"
      - "tools/safe_*.py"
      - "evals/snapshots/*.json"
      - "evals/hypotheses/**/*.md"
      - "docs/current_research_position.md"
      - "docs/metric_provenance.md"
      - "docs/action_space_provenance.md"
      - "reports/**/*.md"
  activation:
    keywords:
      - "intent reconciliation"
      - "intent-reconciliation"
      - "의도 부합"
      - "intent vs actual"
      - "산출물 검증"
    when_to_use: 새 산출물 (data / feature / oracle / policy / framework / doc) 의 commit 직전. 또는 산출물 의 사용자 보고 직전.
  constraints:
    allowed_tools:
      - "Read"
      - "Write"
      - "Edit"
      - "Bash"
      - "Glob"
      - "Grep"
    risk_level: medium
---

# Intent-Reconciliation Loop — ArtifactRouter

본 skill 은 [AGENTS.md §3-23](../../../AGENTS.md) 의 Intent-Reconciliation Loop 의무 의 구체 절차.

> 사용자 directive (2026-05-27): "지금 하네스 루프에 어떤 데이터나 기능을 만들고 나면 초기 만들고자 했던 의도와 부합한지 추가적인 에이전틱 루프를 돌리게 설정해줘."

---

## 1. 목적

다음을 보장한다:

- **Silent drift 차단**: Agent 가 만든 산출물 이 사용자 의도 / hypothesis / 외부 근거 와 **silent 하게 발산** 하지 않음.
- **Side-effect 의 명시 박제**: 의도 외 의 (긍정 / 부정) effect 도 발견 즉시 사용자 보고.
- **Self-correction loop 의 정식화**: 산출물 commit 직전 의 routine self-check 로 [§3-7 자가 수정 메타 규칙](../../../AGENTS.md) 의 day-level 적용.

### 1-1. 본 skill 이 맡지 않는 것

- 가설 사후 수정 (HARKing) — [hypothesis-registry SKILL](../hypothesis-registry/SKILL.md) 의 사용자 승인 게이트.
- 외부 공개 의 reproducibility checklist — [reproducibility-checklist SKILL](../reproducibility-checklist/SKILL.md).
- 우회 ledger — [workaround-tracking SKILL](../workaround-tracking/SKILL.md).

본 skill 은 **산출물 commit 직전** 의 routine self-check 만 담당.

---

## 2. 트리거 — 의무 적용 대상

[AGENTS.md §3-23](../../../AGENTS.md) 의 트리거 list 일관:

1. 새 evaluator / correction tool / generator wrapper.
2. 새 oracle / baseline / policy.
3. 새 evaluation snapshot (`evals/snapshots/*.json`).
4. 새 framework / pipeline / decision rule doc.
5. 새 가설 본문 / status 전환.

**예외**: trivial 산출물 (typo / lint / 형식 / repo-local bookkeeping). 이 경우 본 skill 의무 아님.

---

## 3. 5-Step Loop — 구체 절차

### 3-1. Step 1: Intent 식별

본 산출물 의 motivation source 를 명확 파악. 다음 4 source 중 하나 (또는 조합):

| Source | 식별 방법 | 인용 형식 |
|---|---|---|
| **사용자 directive** | 사용자 메시지 의 verbatim 인용 (날짜 + 핵심 문장) | "2026-05-27: 사용자 directive 'X' 박제" |
| **사전 등록 가설** | `evals/hypotheses/H-YYYY-NNN.md` 의 RQ + 조건 | "H-2026-204 RQ1 의 fixed post-processing 비교 조건" |
| **외부 논문 / framework** | 2020+ peer-reviewed top-tier (AGENTS.md §3-22) | "PhysDiff (Yuan et al. 2023, ICCV) 의 physics-guided projection" |
| **자가 메타 규칙** | AGENTS.md §3 의 절대 규칙 또는 자가 수정 | "AGENTS.md §3-21 (Action Space Provenance) 의 action space spec 의무" |

**검증**: Intent 가 위 4 source 중 하나 로 명확 trace 가능 한가? 그렇지 않으면 산출물 자체 의 motivation 의심 — 사용자 confirm 또는 폐기.

### 3-2. Step 2: Acceptance criteria 식별

본 산출물 의 success threshold — 정량 + 정성 분리:

#### 3-2-1. 정량 criteria 예시

- NetGain 임계 (예: `B2-medium 대비 +0.10 이상`).
- gate violation rate (예: `clean p99 기준 0%`).
- coverage (예: `7 evaluator 의 4 category 충족`).
- p-value / effect size (예: `paired Wilcoxon p < 0.05, Cohen d > 0.5`).
- sample size (예: `n ≥ 20 per [reproducibility-checklist §3-7](../reproducibility-checklist/SKILL.md)`).

#### 3-2-2. 정성 criteria 예시

- "사용자 framing 의 직접 supports".
- "X behavior 가 visible / observable".
- "Y side-effect 가 없음".
- "외부 논문 의 결과 와 일관".

**검증**: criteria 가 사전 정의 인가 (HARKing 차단), 또는 사후 수정 인가? 사후 수정 이면 [hypothesis-registry SKILL §4](../hypothesis-registry/SKILL.md) 사용자 승인 게이트.

### 3-3. Step 3: Actual outcome 측정

산출물 의 실제 behavior 의 정량 / 정성 결과 수집:

- **정량**: snapshot 의 metric, sample size, p-value, effect size, gate violation rate, NetGain change 등.
- **정성**: 사용자 framing 과의 일치, side-effect 발견, 외부 논문 결과 와의 일관성 등.

**검증**: 측정 결과 가 reproducible 한가? raw record cross-link 있는가 ([eval-collect SKILL](../eval-collect/SKILL.md), 작성 후)?

### 3-4. Step 4: Intent vs Actual 비교 — 5 카테고리 분류

| Category | 정의 | 분기 |
|---|---|---|
| **`aligned`** | 의도 와 결과 일치 + criteria 충족 | commit 진행 + 일지 박제 |
| **`partial`** | 일부 의도 충족 (정량 임계 미달 또는 일부 case 만 작동) | commit + caveat 박제 (사용자 보고) |
| **`misaligned`** | 의도 와 결과 불일치 (정량 임계 미충족 + 다른 방향) | **commit 보류** + 사용자 보고 + 정정 plan 또는 폐기 |
| **`scope_creep`** | 산출물 이 의도 보다 큰/작은 scope | **사용자 confirm 의무** (scope 변경 승인) |
| **`unintended_side_effect`** | 의도 외 의 (긍정/부정) effect 발견 | 별도 박제 + 사용자 보고 (긍정 이면 새 H-id, 부정 이면 ledger) |

### 3-5. Step 5: 결과 박제 + 분기

#### 3-5-1. 의무 박제 위치

산출물 의 type 별 박제 위치:

| 산출물 type | 박제 위치 |
|---|---|
| evaluator / tool / oracle / baseline | 일지 (`reports/<YYYY-MM-DD>.md`) 의 본 산출물 절 의 sub-section `**Intent-Reconciliation**` |
| snapshot (calibration, sweep) | 일지 + snapshot 의 `metadata.intent_reconciliation` field |
| 가설 본문 / status 전환 | `evals/hypotheses/<h_id>.md` 의 reconciliation section + [hypothesis-registry SKILL](../hypothesis-registry/SKILL.md) 게이트 |
| framework / doc | doc 본문 의 revision log 절 + 일지 cross-link |

#### 3-5-2. 박제 format (의무 4-line)

```markdown
**Intent-Reconciliation** (AGENTS.md §3-23):
- **Intent**: <source + verbatim 인용 또는 H-id + criteria>
- **Actual**: <정량 결과 + 정성 관찰>
- **Verdict**: `aligned` / `partial` / `misaligned` / `scope_creep` / `unintended_side_effect`
- **Side-effect** (있으면): <별도 박제 내용 + 사용자 보고 status>
```

#### 3-5-3. 사용자 보고 의무

다음 verdict 의 경우 사용자 보고 의무 (commit 보류 또는 confirm 요청):

- `misaligned`: commit 보류, 정정 plan 또는 폐기 의 사용자 결정.
- `scope_creep`: scope 변경 의 사용자 confirm.
- `unintended_side_effect`: 별도 박제 + 사용자 보고 (긍정 / 부정 에 따라 후속 action).

`aligned` / `partial` 은 commit + 일지 박제 진행 (사용자 보고 권장 but not 의무).

---

## 4. 적용 예 — 본 프로젝트 의 retrospective application

### 4-1. PhysicalGateV0 5 evaluator (commit c5f8665, 2026-05-26)

- **Intent**: 사용자 directive (2026-05-26): "Physical constraint evaluator/gate 설계 — BoneLengthViolation / GroundPenetration / ContactConsistency / JerkSpike + accept / repair / rollback / STOP". + PhysDiff (Yuan 2023 ICCV) 의 physics-guided projection framework.
- **Actual**: 5 evaluator (Penetrate / Float / Skate / JerkSpike / BoneLengthCV) 구현 + HumanML3D clean n=493 calibration → 4/5 evaluator p99 매우 tight (0~5.5e-6), Float 만 loose (0.80).
- **Verdict**: `aligned` (5 evaluator 모두 구현 + calibration 완료).
- **Side-effect**: Float evaluator 의 known FP issue 의 정량 confirm (부록 Z) — Item 6 contact estimator 후속 보강 의무.

### 4-2. Safe Sequence Oracle (commit 78d51fc + 22058d5, 2026-05-26)

- **Intent**: 사용자 directive (2026-05-26): "physical hard violation path prune, safe path 중 NetGain 최대 선택. RL-2 의 새 teacher / upper bound." + PhysDiff + HuMoR + MDM framework.
- **Actual**: top-4 의 2/4 (motion_007/028) shifted → STOP. G2 natural n=50 의 7/50 (14%) shifted, NetGain SUM drop -39.4% (median 의 drop 작음, top-2 가 93% contribution).
- **Verdict**: `aligned` + `unintended_side_effect`.
- **Side-effect**: **All 7 violations involve FootLock tool** — FootLock 의 kinematic chain side-effect 가 systematic. 본 발견 은 의도 outside — FootLock tool 의 algorithm 정정 의 motivation 의 후속 분기 (별도 작업 후보).

---

## 5. 본 skill 의 의무 (메타 의무)

- 본 skill 적용 시 5-step 모두 완료 박제. 누락 시 commit 의 일지 의무 (AGENTS.md §3-6-1) 미충족 으로 취급.
- `misaligned` / `scope_creep` / `unintended_side_effect` 의 경우 사용자 보고 의무.
- 산출물 의 incremental update (예: same evaluator 의 SEVERITY_VERSION bump) — simplified version (Intent 변경 없으면 Actual 만 갱신) 허용.

---

## 6. Cross-link

- 본 skill 의 의무 source: [AGENTS.md §3-23](../../../AGENTS.md).
- 본 skill 의 사용자 directive (verbatim): 2026-05-27 messages.
- 관련 skill:
  - [research-journal SKILL](../research-journal/SKILL.md) — 일지 의 박제 format.
  - [hypothesis-registry SKILL](../hypothesis-registry/SKILL.md) — `misaligned` 또는 가설 사후 수정 의 사용자 승인 게이트.
  - [reproducibility-checklist SKILL](../reproducibility-checklist/SKILL.md) — 외부 공개 의 정식 check.
  - [workaround-tracking SKILL](../workaround-tracking/SKILL.md) — 부정 side-effect 의 ledger.
- 관련 AGENTS.md 절:
  - §3-7 (자가 수정 메타 규칙) — 본 skill 의 day-level application.
  - §3-22 (Research Grounding Gate) — Intent 의 source 의 일부.
  - §6-4 (HARKing 차단) — Step 2 criteria 의 사전 정의 의무.
