"""Evaluator tool registry — 명세 §6.2 + Safe Orchestration Layer B (PhysicalGateV0).

Layer A — Artifact Evaluators (current):
  FootFloatingEvaluator, BoneLengthEvaluator, VelocityJitterEvaluator.

Layer B — Physical Constraint Gate (PhysicalGateV0, 2026-05-26 신설):
  PenetrateEvaluator, FloatEvaluator, SkateEvaluator, JerkSpikeEvaluator,
  BoneLengthCVEvaluator.

각 evaluator 는 evaluators.base.Evaluator 인터페이스를 구현하며 EvaluatorReport
list 를 반환한다.

명세 §6.2 Evaluator Tool Registry.
AGENTS.md §3-2 Tool Registry 인터페이스 의무.
docs/current_research_position.md §0-4 Safe Orchestration architecture.
"""
from evaluators.base import Evaluator, EvaluatorReport
from evaluators.bone_length_evaluator import BoneLengthEvaluator
from evaluators.foot_floating_evaluator import FootFloatingEvaluator
from evaluators.physical_gate import (
    DEFAULT_PHYSICAL_GATE_EVALUATORS,
    BoneLengthCVEvaluator,
    FloatEvaluator,
    JerkSpikeEvaluator,
    PenetrateEvaluator,
    SkateEvaluator,
)
from evaluators.velocity_jitter_evaluator import VelocityJitterEvaluator

#: Layer A — Artifact Evaluators (orchestrator 가 본 list 를 iter 하며 evaluate).
DEFAULT_EVALUATORS: list[Evaluator] = [
    FootFloatingEvaluator(),
    BoneLengthEvaluator(),
    VelocityJitterEvaluator(),
]

__all__ = [
    "Evaluator",
    "EvaluatorReport",
    # Layer A — Artifact Evaluators
    "FootFloatingEvaluator",
    "BoneLengthEvaluator",
    "VelocityJitterEvaluator",
    "DEFAULT_EVALUATORS",
    # Layer B — Physical Constraint Gate (PhysicalGateV0)
    "PenetrateEvaluator",
    "FloatEvaluator",
    "SkateEvaluator",
    "JerkSpikeEvaluator",
    "BoneLengthCVEvaluator",
    "DEFAULT_PHYSICAL_GATE_EVALUATORS",
]
