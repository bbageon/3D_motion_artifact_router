"""Orchestrator — 명세 §6.4.

evaluator reports + tool call history → tool decision.
rule_based / supervised_selector / contextual_bandit / multi_step_rl 알고리즘은
모두 orchestrator.base.Orchestrator 인터페이스를 구현한다.

본 모듈은 (a) base interface + KDG ordering violation 예외, (b) Week 4 prototype 의
rule_based orchestrator + tool effect matrix 도구 + oracle best-tool baseline (선택)
의 registry 를 노출한다.

### Oracle best-tool 의 type 명시 의무 (AGENTS.md §3-16)

[H-2026-204](../evals/hypotheses/H-2026-204.md) 등에서 baseline B8 로 쓰는 oracle 은
두 종류로 명확히 구분한다:

  - **single-step oracle**: 후보 tool 을 개별로 한 번만 적용 후 NetGain best 선택.
  - **sequence oracle (closed-loop oracle)**: tool 시퀀스의 모든 path 의 최종 NetGain best.

본 저장소에서 oracle 을 구현할 때는 (orchestrator/oracle_*.py 또는 별도 도구) 반드시
`oracle_type` field 를 raw record·summary 에 박제하고, 두 type 의 결과를 동일 baseline
으로 묶어 인용하지 않는다. AGENTS.md §3-16 (oracle type 명시 의무) 와 §6-5 (metadata
우회로 산출물 동질화) 위반 차단.
"""
from orchestrator.base import KDGOrderingViolation, Orchestrator, OrchestratorDecision
from orchestrator.oracle_single_step import (
    OracleCandidate,
    OracleSelection,
    select_best_tool_single_step,
)
from orchestrator.rule_based import RuleBasedOrchestrator, compute_tool_effect_matrix

__all__ = [
    "Orchestrator",
    "OrchestratorDecision",
    "KDGOrderingViolation",
    "RuleBasedOrchestrator",
    "compute_tool_effect_matrix",
    "OracleCandidate",
    "OracleSelection",
    "select_best_tool_single_step",
]
