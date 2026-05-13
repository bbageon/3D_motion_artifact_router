"""Correction tool registry — 명세 §6.3.

각 tool은 correction_tools.base.CorrectionTool 인터페이스를 구현해 motion을 국소 보정한다.
"""
from correction_tools.base import CorrectionReport, CorrectionTool

__all__ = ["CorrectionTool", "CorrectionReport"]
