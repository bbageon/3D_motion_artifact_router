---
description: ArtifactRouter의 현재 스냅샷과 기준 스냅샷을 비교해 활성 가설(H-2026-200~203)을 통계적으로 평가하고, "1. 제시한 가설 → 2. 실험 평가 → 3. 실험적 근거 → 4. 가설 평가 → 5. 다음 스텝" 5단계로 출력하는 skill이다.
metadata:
  scope:
    paths:
      - "evals/reports/**"
      - "evals/snapshots/**"
      - "evals/hypotheses/**"
  activation:
    keywords:
      - "eval-compare"
      - "5단계"
      - "비교 리포트"
      - "regression"
      - "improvement"
    when_to_use: 충분한 trial·snapshot 누적 후 가설 평가 시점. trial ≥ 20 + snapshot ≥ 2 본격 가동.
  constraints:
    allowed_tools:
      - "Read"
      - "Write"
      - "Edit"
      - "Glob"
    risk_level: high
---

# Eval Compare — ArtifactRouter

본 skill은 [`docs/harness-research-template/evalSample/eval-compare.md`](../../../docs/harness-research-template/evalSample/eval-compare.md) 의 절차를 본 프로젝트에 적용한다.

---

## 1. 목적

다음을 보장한다.

- 평가 파이프라인의 5단계 (Collect/Normalize/Grade/Aggregate/Compare) 중 Compare 단계를 가설 평가의 입력으로 활용.
- 4 판정 (`stable`/`improvement`/`regression`/`inconclusive`) 을 가설별 supports/contradicts/inconclusive 판정으로 변환.
- 5단계 연구 사이클 (가설 → 실험 평가 → 실험적 근거 → 가설 평가 → 다음 스텝) 출력.
- HARKing 차단 — 가설 사후 수정은 사용자 승인 게이트.

---

## 2. 가동 조건

상위 템플릿의 부트스트랩 임계 그대로:

- **trial ≥ 20** 누적.
- **비교 가능 snapshot ≥ 2**.

그 이전에는 informational 출력만. 본격 supports/contradicts 판정 불가.

본 프로젝트 MVP feasibility (4주, 명세 §12) 동안 trial 누적이 빠르면 Week 4 종료 시점에 본격 가동. 그 전에는 oracle best-tool / rule-based / learned selector 의 비교 결과만 informational 보존.

---

## 3. 입력

다음 산출물을 사용:

- `evals/snapshots/{daily,weekly}/<period>.json` — Aggregate 산출물.
- `evals/hypotheses/<h_id>.md` — 활성 가설 (H-2026-200~203).
- `evals/raw/<timestamp>.json` — trial 단위 raw record (paired test 용 pairing key 추출).
- `evals/workarounds/_index.md` — 부록 A 인용.

---

## 4. 4판정 → 가설 평가 매핑

snapshot 의 4 판정을 가설별로 매핑:

| 4 판정 | 의미 | 가설 평가 시그널 |
|---|---|---|
| `improvement` | 동조 개선 | 가설 supports 후보 (단 효과 크기 + 통계 검증) |
| `regression` | 구조적 회귀 | 가설 contradicts 또는 inconclusive |
| `stable` | 변화 없음 | inconclusive |
| `inconclusive` | 표본 부족·noise | inconclusive |

본 매핑이 직접 status 전환을 만들지 않는다 — [`hypothesis-registry SKILL §4`](../hypothesis-registry/SKILL.md) 사용자 승인 게이트 필수.

---

## 5. 통계 절차

[`reproducibility-checklist SKILL §3-7`](../reproducibility-checklist/SKILL.md) 그대로:

- **Paired Wilcoxon signed-rank** — 동일 generator·동일 sample paired 비교.
- **Effect size (Cohen's d)** — `(mean_A - mean_B) / pooled_std`.
- **Bootstrap CI** (N=1000, 95%) — n_trials 또는 n_seeds 표본 적용.
- **Non-inferiority margin δ** — H-2026-203 No-harm 검증 시.

---

## 6. 5단계 사이클 출력 형식

`evals/reports/<period>.md` 의 본문 구조:

```markdown
# eval-compare — <period>

본 리포트는 [`docs/harness-research-template/evalSample/eval-compare.md`](../../../docs/harness-research-template/evalSample/eval-compare.md) 의 5단계 형식을 ArtifactRouter 도메인에 적용한 결과이다.

## 1. 제시한 가설

본 비교는 다음 활성 가설을 평가한다:

- [H-2026-200](../hypotheses/H-2026-200.md) — Artifact-conditioned tool selection + closed-loop refinement net gain.
- [H-2026-201](../hypotheses/H-2026-201.md) — Learnable Routing net gain.
- [H-2026-202](../hypotheses/H-2026-202.md) — Generator transfer.
- [H-2026-203](../hypotheses/H-2026-203.md) — No-harm on high-quality.

## 2. 실험 평가

본 비교에 사용된 trial·snapshot:

- 기준 snapshot: `evals/snapshots/<period_baseline>.json` (trial N=<n>, generator G1/G2/G3 분포).
- 현재 snapshot: `evals/snapshots/<period_current>.json` (trial N=<n>).
- 사용 generator id + version hash + 사용 evaluator/tool registry config hash.
- 사용 split.
- 비교 페어링 key (예: same prompt + same seed).
- raw record 경로 list.

## 3. 실험적 근거

(사실 기록만. supports/contradicts 단정 금지 — §4 가설 평가 절에서만.)

### Generator 별 NetGain 변화

| Generator | baseline NetGain | current NetGain | Δ | paired Wilcoxon p | Cohen's d | 95% CI |
|---|---|---|---|---|---|---|
| G1 (MDM) | ... | ... | ... | ... | ... | ... |
| G2 (MotionGPT) | ... | ... | ... | ... | ... | ... |
| G3 (local_lora) | ... | ... | ... | ... | ... | ... |

### Artifact reduction 분포

(per-artifact metric 표 또는 시각화 인용)

### FidelityLoss 분포

(Protocol A/B/C 별)

### Tool Effect Matrix (해당 ablation 이 있는 경우)

(E2 — 모든 sample 에 모든 tool 적용 후 best tool 식별)

## 4. 가설 평가

각 가설별 supports/contradicts/inconclusive + 정량 근거 + §3 인용.

### H-2026-200 — Artifact-conditioned tool selection

- **판정**: supports / contradicts / inconclusive.
- **정량 근거**: <NetGain · ArtifactReduction · FidelityLoss 변화 + p-value + effect size>.
- **§3 인용**: <표 또는 시각화 경로>.

### H-2026-201 — Learnable Routing

(동일 형식)

### H-2026-202 — Generator transfer

(동일 형식)

### H-2026-203 — No-harm on high-quality

(동일 형식. non-inferiority margin δ 충족 여부.)

## 5. 다음 스텝

판정에 따라:

- **supports**: 다음 연구 단계 진행 또는 status `supported` 마감 (사용자 승인 게이트).
- **contradicts**: (A) 추가 trial 수집 / (B) 가설 폐기 draft / (C) 가설 수정 draft / (D) 평가 절차 결함 의심.
- **inconclusive**: 추가 trial 또는 평가 절차 점검.

B·C 는 사용자 승인 게이트 필수.

## 부록 A — 우회·간접 해결 ledger 인용

| W-id | severity | status | 한 줄 | 본격 학습 영향 |
|---|---|---|---|---|
| W-2026-NNN | <severity> | <status> | <한 줄> | <영향> |

## 부록 B — Tool Call Trace 통계

(per-generator · per-tool · per-target 의 호출 빈도·평균 strength·평균 score Δ 분포)
```

---

## 7. 5단계 위반 시 인용 불가

위 5단계 중 하나라도 빠진 비교 리포트는 자가 수정 메타 규칙 ([`AGENTS.md §3-7`](../../../AGENTS.md)) 의 근거로 인용 불가.

---

## 8. 가설 사후 수정의 보수성 (HARKing 차단)

[`hypothesis-registry SKILL §4`](../hypothesis-registry/SKILL.md) 그대로:

- 사전 정의된 기각 조건 명확 충족 (또는 평가 절차 결함 식별).
- 변경 사유가 본 5단계 리포트에 인용.
- 새 draft 가 hypothesis-registry §2 형식.

위 조건 충족 + 사용자 승인 후에만 promote.

---

## 9. 금지 규칙

- 5단계 중 하나라도 빠진 리포트를 자가 수정 메타 규칙의 근거로 인용 금지.
- 단일 trial 또는 표본 부족 (trial < 20) 상태로 supports/contradicts 단정 금지.
- 가설 status 단독 promote 금지 (사용자 승인 게이트).
- 통계 절차 (§5) 없이 변화 단정 금지.
- 부록 A (W-id 인용) 누락 금지.

---

## 10. 작성 절차

1. 가동 조건 (§2) 확인 — trial ≥ 20, snapshot ≥ 2.
2. snapshot 의 4 판정 추출.
3. 통계 절차 (§5) 적용.
4. §6 5단계 형식으로 `evals/reports/<period>.md` 작성.
5. 부록 A (W-id) 채움.
6. 판정에 따라 사용자 승인 게이트 (status 전환 draft) 작성 — 직접 promote 금지.
