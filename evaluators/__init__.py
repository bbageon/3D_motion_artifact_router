"""Evaluator tool registry — 명세 §6.2.

각 evaluator는 evaluators.base.Evaluator 인터페이스를 구현하며 EvaluatorReport를 반환한다.
"""
from evaluators.base import Evaluator, EvaluatorReport

__all__ = ["Evaluator", "EvaluatorReport"]
