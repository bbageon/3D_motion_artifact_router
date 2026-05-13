"""Closed-loop refinement — 명세 §6.5.

RefinementLoop은 generator output에 evaluator + orchestrator + correction tool을
반복 적용해 refined motion + tool call trace를 생성한다.

AGENTS.md §3-4 Closed-loop Score 비감소 의무.
"""
from refinement_loop.loop import RefinementLoop, RefinementResult

__all__ = ["RefinementLoop", "RefinementResult"]
