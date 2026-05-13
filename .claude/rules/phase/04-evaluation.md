# 04. 평가(Evaluation) Phase — ArtifactRouter

> 본 phase 문서는 [`docs/harness-research-template/04-evaluation.md`](../../../docs/harness-research-template/04-evaluation.md) 의 "04. 평가 레이어 — Research Track" 규격을 본 프로젝트(ArtifactRouter)에 적용한 결과이다. 파이프라인 5단계·지표 부류·회귀 판정 정의는 상위 템플릿을 상속하며, 본 문서는 차이만 기술한다. 본 프로젝트의 평가는 (a) Agent 수행 메트릭과 (b) ArtifactRouter 모델 메트릭 (artifact reduction · NetGain · FidelityLoss · 효율) 의 두 축을 함께 다룬다.

---

## 1. 본 phase의 위치

평가는 관측 전용(meta) 레이어. 커밋 차단은 [`02-sensor.md`](./02-sensor.md) 가 담당. 본 phase 산출물은 머지를 직접 차단하지 않는다.

**IMPORTANT:** 평가 임계값 완화로 회귀 회피 금지.

---

## 2. 평가 단위

- **Task** — 사용자 요청 단위. 커밋 또는 PR과 일치. generator (G1/G2/G3) 와 가설 (H-2026-200~203) 정보를 task 메타에 포함.
- **Trial** — refinement loop 1회 실행 (single sample × single generator).
- **Outcome** — gate 수준 (02-sensor.md §4) + trial 수준의 두 계층 병렬 보존.
- **Snapshot** — 일·주 단위 집계.

---

## 3. 파이프라인 위치 (점진 가동)

상위 템플릿의 5단계 (Collect/Normalize/Grade/Aggregate/Compare) 그대로. 부트스트랩은 trial·snapshot 수 기준.

- Collect → day 1 가동.
- Normalize → raw record 1개라도 생성되면 즉시 가동.
- Grade → Normalize와 동시.
- Aggregate (일별) → day 1.
- Aggregate (주별) → 일별 ≥ 14개.
- Compare → trial ≥ 20 + 비교 가능 snapshot ≥ 2 일 때 본격. 그 전 informational.

본 프로젝트 MVP feasibility (4주, 명세 §12) 동안 trial 누적이 빠르게 진행되므로 Compare 본격 가동은 MVP 종료 시점 (Week 4 끝) 또는 본격 학습 진입 후 expected.

### 3-1. 부트스트랩 임계 근거

상위 템플릿 §3-1 그대로 적용:
- Shewhart 1931 / Montgomery — 20-25 subgroup이 표준.
- Daly et al. ICPE 2020 — change-point detection 20-30 historical run.
- Foo et al. ICSE 2015 — 7-14일 rolling window.
- CLT n≥30 / Wilcoxon n≥20.

본 프로젝트의 trial cadence는 generator inference 비용에 따라 다름 — MVP 결과 보고 보정.

---

## 4. 출력 경로 (듀얼 트랙)

### 4-1. 기계 처리 트랙

- `evals/raw/<timestamp>_<task_id>_<trial_id>.json` — Collect.
- `evals/normalized/<timestamp>.json` — Normalize.
- `evals/graded/<timestamp>.json` — Grade.
- `evals/snapshots/{daily,weekly}/<period>.json` — Aggregate.
- `evals/reports/<period>.md` — Compare (5단계 리포트).
- `evals/hypotheses/<h_id>.md` — 사전 등록 (append-only).
- `evals/hypotheses/_index.md` — 가설 인덱스.
- `evals/workarounds/<W-id>.md` — 우회 ledger (append-only).
- `evals/workarounds/_index.md` — 우회 인덱스.
- `evals/checklists/<period>_<scope>.md` — 재현성 체크리스트 (필요 시).

회귀·개선 판정과 자가 수정 메타 규칙의 근거는 본 트랙에서만 인용.

### 4-2. 사람-가독 트랙

- `reports/<YYYY-MM-DD>.md` — 일자별 연구일지.
- `reports/figures/<YYYY-MM-DD>/` — 시각화 (특히 tool call trace + before/after motion GIF).

### 4-3. cross-link 의무

두 트랙은 raw record 경로 (`evals/raw/<timestamp>_<task_id>_<trial_id>.json`) 로 cross-link. cross-link 끊긴 일지·raw record는 회귀 판정 근거로 인용 불가.

---

## 5. 측정 지표

### 5-1. 수행 품질 지표 (Agent)

- Task 성공률 — 변경이 전체 게이트 통과로 도달한 비율.
- 국소 게이트 pass@1 — 첫 시도에서 round-trip + smoke 통과 비율.
- 평균 재시도 횟수.

### 5-2. 검증 품질 지표 (ArtifactRouter)

명세 §9.3 의 metric을 그대로 사용:

**Artifact metrics** (명세 §9.3.1):
- FootSliding distance.
- GroundPenetration ratio.
- FootFloating ratio.
- BoneLengthVariation.
- JointAngleViolation rate.
- VelocityJitter (mean acceleration norm).
- AccelerationJerk (mean jerk norm).

**Normalized·Total artifact score** (명세 §9.3):
```
NormalizedArtifactScore_m = (metric_m - reference_mean_m) / (reference_std_m + eps)
TotalArtifactScore = sum_m w_m * NormalizedArtifactScore_m
```

**Fidelity metrics** (명세 §9.3):
- **Protocol A (Synthetic injection)**: MPJPE(refined, clean GT) - MPJPE(corrupted, clean GT).
- **Protocol B (Generator output)**: correction magnitude, modified frame ratio, semantic consistency, user study MOS.
- **Protocol C (Distributional)**: FID_motion, FGD, Diversity, MM-Dist.

**Efficiency metrics** (명세 §9.3):
- number of tool calls.
- iteration count.
- wall-clock runtime per sample (ms).
- FLOPs per sample (refinement stage only).
- FPS (effective throughput).
- Relative cost vs single learned calibrator.

**NetGain** (명세 §9.4):
```
NetGain = ArtifactReduction - alpha * FidelityLoss - beta * CorrectionMagnitude - gamma * ToolCallCost
```

지표 정의는 [`reproducibility-checklist SKILL §3`](../../skills/reproducibility-checklist/SKILL.md) 을 단일 출처로 인용.

### 5-2-1. 정성·시각화 보조 지표

- 누적 artifact 추세 (refinement step별).
- foot trajectory visualization.
- bone length variation over time.
- tool call trace + before/after motion GIF.
- user study MOS sample (명세 §9.3 Protocol B).

본 항목들은 informational 기록만. 본 [`AGENTS.md §3-9`](../../../AGENTS.md) 단일 sample 결론 금지 원칙 따라 다중 sample 시각화 우선.

### 5-3. 개선 효과 지표

- 지침 개정 (AGENTS.md, phase, skill) 전후 위반 카테고리 변화.
- evaluator·correction_tool registry 확장 전후 artifact coverage 변화.
- orchestrator 알고리즘 변경 (rule-based → contextual-bandit) 전후 NetGain 변화.

### 5-4. Agent 실행 효율 지표 (선택)

상위 템플릿 §5-4 그대로 — 토큰 소비·도구 호출·캐시 구조 비율 등. ArtifactRouter는 LLM-based calibrator (optional plug-in) 사용 시 본 지표 도입.

---

## 6. 회귀·개선 판정

상위 템플릿 §6 그대로:
- 개별 → 반복: 동일 카테고리 연속 N=3회 또는 7일 윈도 내 M=5건 이상.
- 반복 → 구조적 회귀: 둘 이상 일별 또는 1주별+1일별 두 윈도.
- 개선 신호: 같은 지표 두 윈도에서 동조 개선.

표본 부족 초기 (trial < 20, snapshot < 2) — 회귀 판정 보류, informational 기록만.

---

## 7. 피드백 루프 — 5단계 연구 사이클

Compare 단계는 4 판정 (`stable`/`improvement`/`regression`/`inconclusive`) 을 가설 평가의 입력으로 받아 5단계로 출력. 형식 의무는 [`eval-compare SKILL §6`](../../skills/eval-compare/SKILL.md).

### 7-1. 5단계 사이클

1. **제시한 가설** — H-2026-200~203 중 활성 h_id 인용.
2. **실험 평가** — 어떤 trial·snapshot 사용했는지 명시 (generator·시드·split·raw record 경로).
3. **실험적 근거** — generator (G1/G2/G3) 별 NetGain·artifact reduction·FidelityLoss 변화량·CI·effect size·paired test (Demšar 2006). 사실 기록만.
4. **가설 평가** — H-2026-200~203 각각 `supports`/`contradicts`/`inconclusive`. §3 인용 의무.
5. **다음 스텝 또는 가설 사후 검증** —
   - supports → 다음 연구 단계 또는 status `supported` 마감 (사용자 승인).
   - contradicts·inconclusive → (A) 추가 trial, (B) 가설 폐기 draft, (C) 가설 수정 draft, (D) 평가 절차 결함 의심.
   - **B·C는 사용자 승인 게이트** 통과 후 promote.

### 7-2. 가설 사후 수정의 보수성 (HARKing 차단)

상위 템플릿 §7-2 그대로. Kerr 1998·Lipton & Steinhardt 2018·Ioannidis 2005 인용. 본 프로젝트의 가설은 다음 조건 모두 만족 시 사용자 승인 요청:

- 사전 정의된 기각 조건 (NetGain·artifact reduction 임계) 명확 충족 또는 평가 절차 결함 식별.
- 변경 사유가 5단계 리포트에 인용.
- 새 draft가 [`hypothesis-registry SKILL §4`](../../skills/hypothesis-registry/SKILL.md) 형식.

### 7-3. 회귀 해소

- 구조적 회귀 감지 시 → AGENTS.md §3 절대 규칙·02·03·skill에 대응 규칙 추가/강화. 본 절차는 AGENTS.md §3-7 자가 수정 메타 규칙과 같다.
- 임계값 완화 금지.
- 규칙 추가·수정 시 근거 리포트 경로를 커밋 메시지에 인용.
- 동일 회귀 반복 → 센서 또는 검사 자체 결함 의심, 02·03 재검토.

---

## 8. Agent 관여 범위

- task_id·trial_id·변경 요약·generator (G1/G2/G3)·active 가설 (H-2026-200~203)·tool call trace를 raw record에 남긴다.
- 구조적 회귀 발견 시 → 원인 카테고리 (evaluator 정의 회귀 / correction tool 효과 변화 / orchestrator decision 회귀 / KDG conflict 변화 / generator output 변화) 분류, 규칙 개정안 초안 작성.

**금지:** 평가 지표를 근거 없이 인용 금지. 모든 인용은 `evals/snapshots/` 또는 `evals/reports/` 구체 경로 참조.

---

## 9. 유지보수 규칙

- 지표·임계값·승격 기준을 변경했으면 → 과거 스냅샷 호환성 주석. `aggregation_rule_version` 올린다.
- 평가 판정을 관측 전용 이외로 격상 금지.
- 부트스트랩 임계 (§3-1: trial ≥ 20, snapshot ≥ 2) 는 MVP feasibility 결과 보고 보정.
- 본 문서 용어가 02-sensor.md·03-test.md·하위 `eval-*` skill 과 불일치 → `02` 지도, `03` 방법론, 본 문서 관측 레이어로 간주 후 본 문서 갱신.

---

## 10. ArtifactRouter 특화 — 명세 §11 Go/Stop 기준

본 프로젝트의 MVP feasibility (4주, 명세 §12) 종료 시점에 다음 명세 §11 Go/Stop 기준을 적용. 본 phase의 회귀·개선 판정과 별개로 운영.

### Go 기준 (명세 §11)

1. 생성 motion에 artifact metric이 의미 있게 측정됨 (최소 2 generator).
2. correction tool 적용 후 target artifact 평균 15% 이상 감소.
3. tool 적용 후 motion fidelity 손상 < 10% (correction magnitude).
4. artifact별 effective tool이 다르게 나타남 (최소 2 artifact type).
5. fixed smoothing 대비 net gain ≥ 10% 우세.
6. Oracle best-tool과 rule-based orchestrator 간 충분한 gap (learned orchestrator로 채울 여지).

### Stop 또는 축소 기준

1. 모든 tool에서 target artifact 감소율 < 5%.
2. global smoothing이 oracle best-tool 의 90% 이상 도달.
3. 단일 calibrator 또는 fixed pipeline이 proposed 와 비슷하거나 우세.
4. generator 한 종류에서만 작동 (일반화 어려움).
5. 모든 artifact에 단일 best tool 우세 (conditional selection 가치 부재).

본 Go/Stop 결정 자체는 사용자 승인 게이트 (가설 status 전환). Agent는 결정 권한 없음.
