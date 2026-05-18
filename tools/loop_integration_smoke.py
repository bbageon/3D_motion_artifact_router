"""Refinement loop integration smoke — 1 sample × 1 corrupted version × closed-loop.

⚠️ AGENTS.md §6-13 의무 ⚠️
================================================
본 도구의 결과는 **integration test 결과로만** 기록된다. refinement_loop 파이프
라인이 끝까지 죽지 않고 도는지 (build_data → evaluator → orchestrator → tool
apply → re-evaluate → score 비감소 검증 → STOP/revise/rollback) 만 확인.

가설 supports/contradicts 의 근거로 **인용 금지**. 가설 평가의 근거는
[`eval-compare SKILL §6`](../.claude/skills/eval-compare/SKILL.md) 5단계 리포트
의 trial ≥ 20 / paired test / effect size 조건 충족 결과에서만.

본 CLI 의 raw record schema 는 `record_type="loop_integration_smoke"` 로 명시,
가설 근거 인용 검사 (외부 공개 시) 시 본 type 의 record 는 제외해야 한다.
================================================

본 도구는 다음을 수행한다:
  1. HumanML3D GT 1 sample 로딩.
  2. synthetic injection 1 종 (foot_floating, default) 으로 corrupted motion 생성.
  3. RuleBasedOrchestrator + 3 종 evaluator + 3 종 correction tool 로 RefinementLoop 실행.
  4. 결과 (refined motion · tool_history · decision_history · score_trace) 를
     raw record 1 개로 저장 — `record_type="loop_integration_smoke"`.

CLI 예:
    python -m tools.loop_integration_smoke --sample-id 000000 \\
        --artifact foot_floating \\
        --max-iterations 5 \\
        --output evals/raw/<auto>_loop_integration_smoke_<sample>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from correction_tools import (
    DEFAULT_CORRECTION_TOOLS,
    BoneProjectionTool,
    CorrectionTool,
    FootLockTool,
    VelocitySmoothingTool,
)
from evaluators import DEFAULT_EVALUATORS
from orchestrator import RuleBasedOrchestrator
from refinement_loop import RefinementLoop
from tools.synthetic_injection import (
    inject_bone_stretch,
    inject_foot_floating,
    inject_jitter,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "loop_integration_smoke"  # AGENTS.md §6-13 — 가설 근거 인용 금지 type

INJECTORS = {
    "foot_floating": lambda m, seed: inject_foot_floating(m, lift_height=0.08, seed=seed),
    "bone_stretch_right_arm": lambda m, seed: _partial_bone_stretch(m, seed),
    "global_jitter": lambda m, seed: inject_jitter(m, noise_std=0.05, seed=seed),
}


def _partial_bone_stretch(motion: np.ndarray, seed: int) -> np.ndarray:
    T = motion.shape[0]
    half = max(1, T // 2)
    return np.concatenate(
        [inject_bone_stretch(motion[:half], chain_label="right_arm", stretch_factor=1.30, seed=seed),
         motion[half:]],
        axis=0,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _get_severity_version(evaluator: Any) -> str:
    mod = sys.modules.get(type(evaluator).__module__)
    if mod is None:
        return "unversioned"
    return getattr(mod, "SEVERITY_VERSION", "unversioned")


def _serialize_decision(d: Any) -> dict[str, Any]:
    """OrchestratorDecision → JSON-serializable dict."""
    obj = asdict(d)
    if obj.get("target_frames") is not None:
        obj["target_frames"] = list(obj["target_frames"])
    return obj


def _serialize_correction(r: Any) -> dict[str, Any]:
    obj = asdict(r)
    if obj.get("frame_range") is not None:
        obj["frame_range"] = list(obj["frame_range"])
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RefinementLoop integration smoke — single sample × single corrupted "
                    "version × closed-loop. AGENTS.md §6-13 의무: 가설 근거 인용 금지."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--sample-id", type=str, default="000000",
                        help="HumanML3D sample id (stem).")
    parser.add_argument("--artifact", type=str, default="foot_floating",
                        choices=list(INJECTORS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--task-id", type=str, default="loop_integration_smoke_v1")
    parser.add_argument("--split-id", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="raw record output JSON. None 이면 auto path.")
    args = parser.parse_args()

    split_id = args.split_id if args.split_id is not None else args.task_id

    sample_path = args.data_dir / f"{args.sample_id}.npy"
    if not sample_path.exists():
        raise FileNotFoundError(f"sample not found: {sample_path}")

    clean = np.load(str(sample_path)).astype(np.float64)
    if clean.ndim != 3 or clean.shape[1] != 22 or clean.shape[2] != 3:
        raise ValueError(f"sample shape {clean.shape} not [T, 22, 3]")

    injector = INJECTORS[args.artifact]
    corrupted = injector(clean, args.seed)

    # tools dict for loop.
    tools_dict: dict[str, CorrectionTool] = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "BoneProjectionTool": BoneProjectionTool(),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
    }

    orchestrator = RuleBasedOrchestrator(
        tool_registry=list(tools_dict.values()),
        stop_severity_threshold="low",
    )
    loop = RefinementLoop(
        evaluators=list(DEFAULT_EVALUATORS),
        correction_tools=tools_dict,
        orchestrator=orchestrator,
        max_iterations=args.max_iterations,
    )

    result = loop.run(corrupted)

    evaluator_config_hashes = {
        ev.name: ev.evaluator_class_hash() for ev in DEFAULT_EVALUATORS
    }
    evaluator_severity_versions = {
        ev.name: _get_severity_version(ev) for ev in DEFAULT_EVALUATORS
    }
    tool_class_hashes = {name: t.tool_class_hash() for name, t in tools_dict.items()}

    timestamp = _now_iso()
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,  # AGENTS.md §6-13 — 가설 근거 인용 금지 type
        "timestamp": timestamp,
        "task_id": args.task_id,
        "split_id": split_id,
        "trial_id": args.sample_id,
        "sample_path": str(sample_path),
        "generator_id": "humanml3d_gt",
        "artifact_kind": args.artifact,
        "seed": int(args.seed),
        "evaluator_config_hashes": evaluator_config_hashes,
        "evaluator_severity_versions": evaluator_severity_versions,
        "tool_registry_config_hashes": tool_class_hashes,
        "skeleton_normalizer_model_card_hash": None,
        "motion_shape": list(clean.shape),
        "fps": 20,
        "loop_config": {
            "orchestrator": orchestrator.name,
            "max_iterations": args.max_iterations,
            "score_increase_tolerance": loop.score_increase_tolerance,
        },
        "result": {
            "converged": result.converged,
            "max_iterations_reached": result.max_iterations_reached,
            "rolled_back": result.rolled_back,
            "n_tool_applications": len(result.tool_history),
            "n_decisions": len(result.decision_history),
            "score_trace": result.score_trace,
            "score_initial": result.score_trace[0],
            "score_final": result.score_trace[-1],
            "score_total_change": result.score_trace[-1] - result.score_trace[0],
            "tool_call_trace": [_serialize_correction(r) for r in result.tool_history],
            "decision_trace": [_serialize_decision(d) for d in result.decision_history],
            "metadata": result.metadata,
        },
        "metrics_not_applicable": {
            "NetGain": "integration smoke; NetGain is not the test target",
            "netgain_weight_status": "n/a (integration smoke)",
        },
        "agents_md_obligation_note": (
            "AGENTS.md §6-13 — 본 record 의 record_type='loop_integration_smoke' 는 "
            "가설 supports/contradicts 의 근거로 인용 금지. integration smoke 통과 사실은 "
            "파이프라인이 죽지 않고 도는지의 확인일 뿐, 통계적 근거 아님."
        ),
        "negative_result": False,
    }

    out = args.output
    if out is None:
        raw_dir = REPO_ROOT / "evals" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        out = raw_dir / f"{timestamp}_{args.task_id}_{args.sample_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    # stdout summary (ASCII-safe — Windows cp949 stdout 환경 회피).
    print(f"[OK] integration smoke complete - record: {out}")
    print(f"  artifact: {args.artifact}")
    print(f"  converged: {result.converged} / max_iter_reached: {result.max_iterations_reached} / rolled_back: {result.rolled_back}")
    print(f"  n_tool_applications: {len(result.tool_history)}")
    print(f"  score_trace: {result.score_trace}")
    print(f"  stop_reason: {result.metadata.get('stop_reason')}")
    print(f"  AGENTS.md 6-13 - integration smoke (not hypothesis evidence).")


if __name__ == "__main__":
    main()
