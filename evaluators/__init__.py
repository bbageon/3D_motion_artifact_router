"""Evaluator tool registry — 명세 §6.2.

각 evaluator 는 evaluators.base.Evaluator 인터페이스를 구현하며 EvaluatorReport
list 를 반환한다. 본 모듈은 (a) base interface 와 (b) Week 2 prototype evaluator 3
종 (foot_floating, bone_length, velocity_jitter) 의 registry 를 노출한다.

명세 §6.2 Evaluator Tool Registry.
AGENTS.md §3-2 Tool Registry 인터페이스 의무.
"""
from evaluators.base import Evaluator, EvaluatorReport
from evaluators.bone_length_evaluator import BoneLengthEvaluator
from evaluators.foot_floating_evaluator import FootFloatingEvaluator
from evaluators.velocity_jitter_evaluator import VelocityJitterEvaluator

#: 본 저장소의 default evaluator registry. orchestrator 가 본 list 를 iter 하며 evaluate().
DEFAULT_EVALUATORS: list[Evaluator] = [
    FootFloatingEvaluator(),
    BoneLengthEvaluator(),
    VelocityJitterEvaluator(),
]

__all__ = [
    "Evaluator",
    "EvaluatorReport",
    "FootFloatingEvaluator",
    "BoneLengthEvaluator",
    "VelocityJitterEvaluator",
    "DEFAULT_EVALUATORS",
]
