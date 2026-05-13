# 02. 센서(Sensor) Phase — ArtifactRouter

> 본 phase 문서는 [`docs/harness-research-template/02-sensor.md`](../../../docs/harness-research-template/02-sensor.md) 의 "02. 센서 레이어 — Research Track" 규격을 본 프로젝트(ArtifactRouter)에 적용한 결과이다. 감지 대상·게이트 계층·판정 의미의 정의는 상위 템플릿을 상속하며, 본 문서는 중복 서술하지 않는다. 본 프로젝트의 핵심 컴포넌트는 **generators / skeleton_normalizer / evaluators / correction_tools / orchestrator / refinement_loop** 이며, 본 phase는 이 컴포넌트들에 대한 센서 baseline을 정의한다.

---

## 1. 감지 대상 (본 프로젝트 적용)

상위 템플릿 §1의 baseline을 ArtifactRouter 산출물에 결합한다.

- **정적 적합성** — Python 임포트·구문 오류. `python -m py_compile <파일>` 검사. 정형 lint(`ruff`·`mypy`) 도입 시 본 절 갱신.
- **구축 가능성** — generator wrapper 로드 성공, skeleton_normalizer가 외부 generator output을 `[T, 22, 3]` canonical format으로 정규화 성공, evaluator/correction_tool registry 등록 성공.
- **행동 회귀** — 두 부류:
  - **포맷 회귀** — Skeleton Normalizer의 round-trip 검사 (외부 generator output → canonical → 원본 좌표 복원 시 epsilon 이내), evaluator output schema 일관성.
  - **메트릭 회귀** — artifact metric (foot sliding/jitter/bone length 등), NetGain, FidelityLoss 의 직전 스냅샷 대비 회귀.
- **변경 의무** — evaluator/correction_tool/orchestrator의 인터페이스 변경이 다른 컴포넌트에만 적용된 상태. 신규 evaluator·correction_tool·generator wrapper 추가는 의무 대상.

### 1-1. 확장 감지 대상

- **구조 적합성** — `generators/`, `evaluators/`, `correction_tools/`, `orchestrator/` 디렉토리 경계 위반 (예: orchestrator가 evaluator 내부 구현 import) 을 수동 리뷰로 감지.
- **시험 충분성** — **artifact metric coverage** + **tool selection 충분성** 을 도메인 충분성 지표로 사용. 정의·임계·복구 절차는 [`sufficiency-metric SKILL`](../../skills/sufficiency-metric/SKILL.md).
- **결정성 이탈 (flaky)** — 같은 generator output·같은 evaluator 입력·같은 seed인데 결과가 달라지는 상태. 본 프로젝트는 외부 generator (NF4 양자화 모델 등) 의 inter-session noise 가능성을 고려해야 함.
- **실험 정합성** — generator output·tool registry config·LoRA adapter의 metadata와 현재 코드 정의 일치 (silent invalidation 차단). 검증 절차는 [`data-versioning SKILL §3`](../../skills/data-versioning/SKILL.md).

---

## 2. 핵심 경로 정의

다음을 **핵심 경로** 로 지정한다. 충분성 지표와 변경 의무는 본 목록을 우선 대상으로 삼는다.

- [`skeleton_normalizer/`](../../../skeleton_normalizer/) — 외부 generator output → canonical SMPL 22 변환. 데이터 단일 진입점.
- [`generators/base.py`](../../../generators/base.py) — Generator interface. 모든 generator wrapper 의 계약 원본.
- [`evaluators/base.py`](../../../evaluators/base.py) — Evaluator interface. 모든 evaluator 의 계약 원본.
- [`correction_tools/base.py`](../../../correction_tools/base.py) — CorrectionTool interface (명세 §6.3 공통 인터페이스). 모든 correction tool 의 계약 원본.
- [`orchestrator/`](../../../orchestrator/) — orchestrator 의사결정 로직. KDG / scoring / rule_based / supervised_selector / contextual_bandit 등.
- [`refinement_loop/`](../../../refinement_loop/) — closed-loop refinement 의 종료 조건·convergence 검증.
- [`evals/hypotheses/`](../../../evals/hypotheses/) — append-only. 변경은 사용자 승인 게이트.
- [`evals/workarounds/`](../../../evals/workarounds/) — append-only. 변경은 [`workaround-tracking SKILL`](../../skills/workaround-tracking/SKILL.md).

본 목록은 [`AGENTS.md §7`](../../../AGENTS.md) 디렉토리 규칙과 일관되게 유지한다.

---

## 3. 게이트 계층

상위 템플릿의 2-tier 계층 (빠른 국소 게이트 / 전체 게이트) 을 그대로 사용한다.

### 3-1. 빠른 국소 게이트

- 변경된 `.py` 임포트/구문 검사.
- skeleton_normalizer 또는 generator wrapper를 변경했으면 → round-trip 검사 (canonical format 재정규화 후 좌표 epsilon 이내).
- evaluator를 변경했으면 → 단위 테스트 (synthetic motion 입력에 대한 expected metric 값 비교).
- correction_tool을 변경했으면 → tool 적용 전후 target artifact metric 감소 확인 (단순 단위 검증).
- orchestrator를 변경했으면 → mock evaluator reports + mock tool registry 로 decision 생성 검증.
- 변경 의무 매핑 충족.
- 산출물 metadata 정합성 ([`data-versioning SKILL §3`](../../skills/data-versioning/SKILL.md)).

### 3-2. 전체 게이트

- 핵심 경로 변경을 포함한 커밋이면 → end-to-end refinement loop를 1회 이상 수행 (generator output → normalize → evaluate → correct → re-evaluate).
- 평가 결과는 generator (G1/G2/G3) 별로 분리 기록 ([`AGENTS.md §3-6`](../../../AGENTS.md) 평가 기록 의무).
- 충분성 지표 (artifact coverage, tool selection 충분성) 가 임계를 넘는지 [`sufficiency-metric SKILL §2`](../../skills/sufficiency-metric/SKILL.md) 판정.
- 산출물 metadata 정합성 검증.
- 우회 발견 즉시 ledger 등록 확인 ([`workaround-tracking SKILL §4`](../../skills/workaround-tracking/SKILL.md)).

### 3-3. 국소 → 전체 격상 조건

다음 중 하나라도 해당하면 전체 게이트로 격상:

- `skeleton_normalizer/` 의 canonical format 정의 (joint 순서·좌표계·contact label 추정) 를 변경.
- `evaluators/base.py` 의 Evaluator interface 또는 evaluator output schema 변경.
- `correction_tools/base.py` 의 CorrectionTool interface 변경.
- `orchestrator/scoring.py` 의 ScoreFunction 또는 `orchestrator/kdg.py` 의 ConflictScore 정의 변경.
- `refinement_loop/loop.py` 의 종료 조건·convergence 조건 변경.
- 평가 지표 정의 (foot sliding / jitter / bone length 등의 계산식) 변경.
- `evals/hypotheses/` 변경 — 동시에 [`hypothesis-registry SKILL §4`](../../skills/hypothesis-registry/SKILL.md) 사용자 승인 게이트 적용.

**IMPORTANT:** 국소 게이트만 통과한 상태로 핵심 경로 변경을 커밋하지 마라.

---

## 4. 판정 의미

상위 템플릿의 3 판정 (hard fail / soft fail / informational) 그대로 사용. 본 프로젝트 추가:

- **hard fail** — Python 구문/임포트 에러, Skeleton Normalizer round-trip 불일치, evaluator output schema 위반, CorrectionTool interface 위반, KDG ordering 위반, refinement loop convergence 위반 (Score 비감소 깨짐), AGENTS.md §3 절대 규칙 위반, change-obligation 미충족.
- **soft fail** — flaky 후보, 충분성 지표 임계 미달, generator 비대칭 (G1만 측정하고 G2/G3 누락 등), tool conflict_risk 임계 초과.
- **관측 전용** — refinement loop 의 step별 metric trace, tool call trace, 시각화 산출물.

가설 status 전환·우회 등급 변경 등 사용자 승인 게이트 대상은 본 판정 체계와 별도 — Agent가 hard/soft fail로 판단하지 않고 사용자 결정을 대기.

---

## 5. Agent 표준 실행 흐름

변경 후 Agent는:

1. 변경 영향 범위 식별. §3-3 격상 조건 해당 시 처음부터 전체 게이트.
2. 빠른 국소 게이트 (§3-1) 실행.
3. 변환기·evaluator·correction_tool·orchestrator 중 하나 이상 변경 시 → [`change-obligation SKILL`](../../skills/change-obligation/SKILL.md) 매핑 충족 확인 (skill 작성 후).
4. 실패 시 → hard fail은 즉시 수정 후 재실행. soft fail은 flaky-handling 또는 sufficiency-metric SKILL 수행.
5. 국소 통과 후 → 전체 게이트 (§3-2).
6. 시각화·사람이 보는 산출물이 변경되면 → 단일 sample 결론 금지, 다중 sample 평균 함께 확인.
7. 전체 게이트 통과 후 → 평가 수집 (`eval-collect`).
8. 학습·추론·평가 1회 이상 또는 핵심 경로 변경이면 → 연구일지 작성 ([`research-journal SKILL`](../../skills/research-journal/SKILL.md)).
9. 정공법 실패 우회 채택 시 → [`workaround-tracking SKILL §4`](../../skills/workaround-tracking/SKILL.md) 등록.
10. 일지·raw record 모두 존재 시 → 커밋.

**IMPORTANT:** 센서 hard fail 상태로 커밋 금지.

---

## 6. 설계 원칙 (결정성 의무)

상위 템플릿의 7 원칙 (결정성·국소성·명확성·독립성·구성 가능성·추적 가능성·확장 가능성) 상속. 결정성 추가 의무:

- 학습·추론·평가는 고정 시드 (`torch.manual_seed`, `numpy.random.seed`, `random.seed`).
- generator·evaluator·correction_tool 호출 시 random 의존이 있으면 명시.
- GPU 결정성 (`torch.use_deterministic_algorithms(True)`) 은 평가 재현 단계에서만 강제 — 학습 비용 ↑ trade-off.
- 외부 generator (특히 NF4 양자화 모델) 의 inter-session noise는 known issue로 ledger에 명시 후 통계적 처리 (N≥3 generation 평균).

---

## 7. 유지보수 규칙

- 핵심 경로 (§2) 를 변경했으면 → [`AGENTS.md §7`](../../../AGENTS.md) 디렉토리 규칙과 §2 센서 명령어 정의를 동시 갱신.
- 격상 조건 (§3-3) 을 확장했으면 → 근거 회귀 리포트 또는 장애 사례를 커밋 메시지에 인용.
- 정형 lint·typecheck 도구를 도입하면 → §1 정적 적합성 절과 [`AGENTS.md §2-2`](../../../AGENTS.md) 센서 명령어를 동시 갱신.
- 본 문서 용어가 [`03-test.md`](./03-test.md)·[`04-evaluation.md`](./04-evaluation.md)·하위 skill과 불일치 → 본 문서를 원본으로 하위 문서 정정.
