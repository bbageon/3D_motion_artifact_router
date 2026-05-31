"""Shared metadata helpers for experiment snapshots.

These helpers keep Step E/F snapshot metadata aligned with AGENTS.md gates
without rewriting historical numeric results.
"""
from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

from evaluators.base import Evaluator


STRENGTHS_5LEVEL = ("xsmall", "small5", "medium5", "large5", "xlarge")
STRENGTHS_3LEVEL = ("small", "medium", "large")
TOOLS_ORDER = ("FootLockTool", "BoneProjectionTool", "VelocitySmoothingTool")

CONTROLLED_DIAGNOSTIC = "controlled diagnostic finding"
REAL_DISTRIBUTION = "real-distribution evidence"
QUALITY_VALIDATED = "quality-validated evidence"


def _severity_version(ev: Evaluator) -> str:
    module = importlib.import_module(type(ev).__module__)
    return str(getattr(module, "SEVERITY_VERSION", "unknown"))


def evaluator_config_hashes(evaluators: Iterable[Evaluator]) -> dict[str, str]:
    return {ev.name: ev.evaluator_class_hash() for ev in evaluators}


def evaluator_severity_versions(evaluators: Iterable[Evaluator]) -> dict[str, str]:
    return {ev.name: _severity_version(ev) for ev in evaluators}


def dedupe_evaluators(*groups: Iterable[Evaluator]) -> list[Evaluator]:
    by_name: dict[str, Evaluator] = {}
    for group in groups:
        for ev in group:
            by_name.setdefault(ev.name, ev)
    return list(by_name.values())


def action_space_grid(
    *,
    grid: str = "5-level",
    stage: str = "RL-2",
    include_stop: bool = True,
) -> dict[str, Any]:
    # Action space grid 의 분류 (AGENTS.md §3-21 의 4 category):
    #   - discrete: 3-level / 5-level / discrete_Nlevel
    #   - bounded_continuous_u: u∈[0,1] normalized intensity (action_space_provenance §5-2)
    #   - dense_grid_proxy: dense sampled u-grid (continuous 의 진단 proxy)
    #   - N/A: profile / split / non-action context (helper 비활성, descriptor 만 반환)
    grid_l = grid.lower()
    if grid in ("5-level", "discrete_5level", "discrete-5level"):
        strengths = STRENGTHS_5LEVEL
    elif grid in ("3-level", "discrete_3level", "discrete-3level"):
        strengths = STRENGTHS_3LEVEL
    elif grid_l.startswith("continuous") or grid_l.startswith("bounded_continuous") or grid_l.startswith("bounded-continuous") or grid_l.startswith("dense_grid_proxy") or grid_l.startswith("dense-grid-proxy"):
        # Bounded continuous / dense-grid-proxy intensity (action_space_provenance §5-2).
        return {
            "grid": grid,
            "stage": stage,
            "include_stop": include_stop,
            "action_type": "discrete_tool_x_continuous_intensity",
            "tools": list(TOOLS_ORDER),
            "intensity": "u in [0,1] (normalized); FootLock/BoneProjection factor=u, VelocitySmoothing sigma=2.0*u",
        }
    elif grid_l.startswith("n/a") or grid_l == "none" or grid_l == "":
        # Profile / split / non-action context — descriptor only, action space irrelevant.
        return {
            "grid": grid, "stage": stage, "include_stop": include_stop,
            "action_type": "not_applicable",
        }
    else:
        # Permissive fallback: 새 descriptor 도 raise 하지 않고 generic record (helper = mutable, AGENTS §3-24).
        return {
            "grid": grid, "stage": stage, "include_stop": include_stop,
            "action_type": "custom",
            "tools": list(TOOLS_ORDER),
        }
    n_actions = len(TOOLS_ORDER) * len(strengths) + (1 if include_stop else 0)
    return {
        "grid": grid,
        "stage": stage,
        "include_stop": include_stop,
        "n_actions": n_actions,
        "tools": list(TOOLS_ORDER),
        "strengths": list(strengths),
    }


def common_snapshot_metadata(
    *,
    split_id: str,
    evidence_tier: str | list[str],
    oracle_type: str,
    action_grid: str = "5-level",
    stage: str = "RL-2",
    evaluators: Iterable[Evaluator] = (),
    gate_evaluators: Iterable[Evaluator] = (),
) -> dict[str, Any]:
    all_evaluators = dedupe_evaluators(evaluators, gate_evaluators)
    return {
        "split_id": split_id,
        "oracle_type": oracle_type,
        "action_space_grid": action_space_grid(grid=action_grid, stage=stage),
        "evidence_tier": evidence_tier,
        "evaluator_config_hashes": evaluator_config_hashes(all_evaluators),
        "evaluator_severity_versions": evaluator_severity_versions(all_evaluators),
    }
