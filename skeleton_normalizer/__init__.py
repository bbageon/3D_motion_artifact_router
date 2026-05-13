"""Skeleton Normalizer — 명세 §6.1.

외부 generator output을 canonical SMPL 22-joint, [T, 22, 3], fps=20, root-relative format으로 정규화.

AGENTS.md §3-1 Canonical Motion Format 의무.
"""
from skeleton_normalizer.canonical_smpl_22 import SMPL_22, NAME_TO_IDX

__all__ = ["SMPL_22", "NAME_TO_IDX"]
