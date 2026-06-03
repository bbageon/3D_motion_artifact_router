# Implementation Recipes & Architecture Constraints (reference)

> [AGENTS.md](../../AGENTS.md) 의 구현 레시피(구 §9) + 아키텍처·네이밍(구 §10) 을 분리한 **참조 문서**. 특정 구현 작업 시에만 참조 (모든 변경 필수 아님). 절대 규칙은 AGENTS.md §3.

## 1. 구현 레시피

### 1-1. 새 evaluator 추가
1. `evaluators/<name>_evaluator.py` — `Evaluator` base class 상속. 2. `evaluate(motion) -> EvaluatorReport` (명세 §6.2 schema). 3. 단위 테스트 `tests/unit/evaluators/test_<name>.py`. 4. registry config (`evaluators/__init__.py`) 갱신. 5. [reproducibility-checklist §3 지표 정의 사전](../skills/reproducibility-checklist/SKILL.md) metric 추가. 6. integration smoke 재실행.

### 1-2. 새 correction tool 추가
1. `correction_tools/<name>_tool.py` — `CorrectionTool` 상속 (명세 §6.3). 2. `apply(motion, target_part, target_joints, frame_range, strength, metadata)` 구현. 3. KDG affected joints `A(t)` + propagation weights 등록 (`orchestrator/kdg.py`). 4. artifact-tool compatibility table 매핑 추가. 5. 단위 테스트 + integration smoke + tool effect matrix 재실행.

### 1-3. 새 generator wrapper 추가
1. `generators/<name>_wrapper.py` — `Generator` 상속. 2. `generate(prompt, n_frames) -> (motion, metadata)`, motion `[T,22,3]`. 3. skeleton normalizer round-trip 검증. 4. G1(diffusion)/G2(token) quality-tier 분류 (명세 §9.2). 5. evals/raw `generator_id` 등록.

### 1-4. 신규 가설 등록
1. [`evals/hypotheses/_index.md`](../../evals/hypotheses/_index.md) 에서 다음 h_id 확인. 2. [hypothesis-registry SKILL §2-3](../skills/hypothesis-registry/SKILL.md) 형식으로 `H-YYYY-NNN.md`. 3. `사전 정의`·`기각 조건`·`표본 요건` 모두 채움. 4. `_index.md` + [hypotheses-summary.md](hypotheses-summary.md) 갱신.

### 1-5. 우회 발견 시 등록
1. 정공법 실패 확정 즉시 [workaround-tracking SKILL §4](../skills/workaround-tracking/SKILL.md) 로 W-id 부여. 2. ledger (`severity`·`status`·`resolution_target`). 3. 관련 가설·일지·model_card 에 W-id cross-link. 4. critical 이면 사용자 알림.

### 1-6. 외부 공개 전 점검
1. [reproducibility-checklist §2](../skills/reproducibility-checklist/SKILL.md) 14+2 항목. 2. `evals/workarounds/_index.md` 의 `status:open`+`severity:critical` 0개 확인. 3. `reports/checklists/<period>_<scope>.md` 기록.

### 1-7. MVP Feasibility Study 단계 (명세 §12)
- Week 1: generator output + skeleton format 통일 + viz. Week 2: evaluator metrics (foot/jitter/bone 최소 3). Week 3: oracle best-tool + tool-effect matrix + rule-based orchestrator. Week 4: fixed smoothing baseline + synthetic injection + perceptual + Go/Stop (명세 §11). → `experiments/week<N>_*/` segregate.

## 2. 아키텍처 제약 & 네이밍

### 2-1. 파일 네이밍
- Evaluator: `evaluators/<artifact-category>_evaluator.py`. Correction tool: `correction_tools/<tool-name>_tool.py`. Generator wrapper: `generators/<generator-name>_wrapper.py`. 단위 테스트: `tests/unit/<component>/test_<name>.py`.

### 2-2. 데이터 포맷
- 내부 motion: `[T,22,3]` numpy float64 + metadata dict. Evaluator/Tool/Orchestrator report schema: 명세 §6.2/§6.3/§6.4 직접 인용. raw record schema: [eval-collect SKILL §5](../skills/eval-collect/SKILL.md).

### 2-3. 결과 파일 분리
- generator (G1/G2) · evaluator config · tool registry config · seed 별로 결과 파일 분리.

## 3. 참고 문서 (외부 + 하네스)
- 하네스 4계층 원본: [`docs/harness-research-template/`](../../docs/harness-research-template/) 01~04. 적용본: [`.claude/rules/phase/`](../rules/phase/) + [`.claude/skills/`](../skills/).
- 연구 명세 단일 출처: [`docs/motion_research_strategy_summary.md`](../../docs/motion_research_strategy_summary.md). provenance: [metric_provenance](../../docs/metric_provenance.md) · [action_space_provenance](../../docs/action_space_provenance.md) · [current_research_position](../../docs/current_research_position.md).
- 사전 등록 가설: [`evals/hypotheses/_index.md`](../../evals/hypotheses/_index.md). 우회 ledger: [`evals/workarounds/_index.md`](../../evals/workarounds/_index.md).
- MotionGPT: <https://github.com/OpenMotionLab/MotionGPT>. HumanML3D: <https://github.com/EricGuo5513/HumanML3D>.
