"""AR-072 ③-물리: RootGaitConsistencyTool 효과 측정 + Cat-A 입력용 보정본 dump.

사전등록 guard (spec ①) 중 즉시 측정분:
  - 기전 표적: v_root/GT ratio 가 0.42 → 1.0 방향 이동하는가 (핵심)
  - fs (Cat-B): 감소 (정의상 기대 — 보고만, 성공 기준 아님)
  - 불변성: local(root-relative) 완전 불변 assert (전 sample)
  - smoothness: root 수평 accel p95 악화 감시
Cat-A (R-Prec/MM-Dist) 는 mgpt env 별도 단계 — 본 도구가 보정 trajectory 를
external_assets/ar072_root_corrected_mdm/ (gitignored, 재생성 가능) 에 dump.

CLI (motion3d env):
    python tools/root_correction_effect_ar072.py --limit 12   # preflight
    python tools/root_correction_effect_ar072.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import RootGaitConsistencyTool
from tools.coords_protocol import PELVIS
from tools.representative_pool_measure import foot_skate_world

POOL = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608" / "mdm"
GT_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
DUMP = REPO_ROOT / "external_assets" / "ar072_root_corrected_mdm"
LOCO_THRESH = 0.010


def _root_speed(traj):
    return float(np.linalg.norm(np.diff(traj[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean())


def _root_accel_p95(traj):
    a = np.diff(traj[:, PELVIS, :][:, [0, 2]], axis=0, n=2)
    return float(np.percentile(np.linalg.norm(a, axis=1), 95)) if len(a) else 0.0


def _boot_ci(vals, rng, n=1000):
    nv = len(vals)
    bs = [float(np.mean(vals[rng.integers(0, nv, nv)])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "root_correction_effect_ar072_v1.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tool = RootGaitConsistencyTool()
    DUMP.mkdir(parents=True, exist_ok=True)

    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(POOL.glob("*.json"))
             if p.name != "_pool_summary.json"]
    if args.limit:
        metas = metas[:args.limit]

    per_prompt = defaultdict(lambda: defaultdict(list))
    gt_speed: dict[str, float] = {}
    gt_accel: dict[str, float] = {}
    inv_violations = 0
    for m in metas:
        traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        T = traj.shape[0]
        corr, rep = tool.apply(traj, target_part="root", target_joints=[],
                               frame_range=(0, T - 1), strength="large",
                               metadata={"coord_space": "trajectory"})
        # 불변성 assert (구조 보장 검증 — 전 sample).
        lb = traj - traj[:, PELVIS:PELVIS + 1, :]
        la = corr - corr[:, PELVIS:PELVIS + 1, :]
        if not np.allclose(la, lb, atol=1e-9):
            inv_violations += 1
        np.save(DUMP / f"{m['sample_id']}__seed{m['seed']}.trajectory.npy", corr.astype(np.float32))

        sid = m["sample_id"]
        d = per_prompt[sid]
        d["fs_b"].append(foot_skate_world(traj)); d["fs_a"].append(foot_skate_world(corr))
        d["vr_b"].append(_root_speed(traj)); d["vr_a"].append(_root_speed(corr))
        d["acc_b"].append(_root_accel_p95(traj)); d["acc_a"].append(_root_accel_p95(corr))
        d["cff"].append(rep.metadata["constrained_frame_frac"])
        if sid not in gt_speed:
            gt = np.load(GT_DIR / f"{sid}.npy").astype(np.float64)
            gt_speed[sid] = float(np.linalg.norm(
                np.diff(gt[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean())
            gt_accel[sid] = _root_accel_p95(gt)

    rows = []
    for sid, d in per_prompt.items():
        rows.append({
            "sid": sid, "gt": gt_speed[sid],
            "fs_b": float(np.mean(d["fs_b"])), "fs_a": float(np.mean(d["fs_a"])),
            "vr_b": float(np.mean(d["vr_b"])), "vr_a": float(np.mean(d["vr_a"])),
            "acc_b": float(np.mean(d["acc_b"])), "acc_a": float(np.mean(d["acc_a"])),
            "cff": float(np.mean(d["cff"])),
        })
    loco = [r for r in rows if r["gt"] > LOCO_THRESH]
    ratio_b = np.array([r["vr_b"] / r["gt"] for r in loco])
    ratio_a = np.array([r["vr_a"] / r["gt"] for r in loco])
    fs_d = np.array([r["fs_a"] - r["fs_b"] for r in rows])
    acc_d = np.array([r["acc_a"] - r["acc_b"] for r in rows])

    out = {
        "schema_version": "1.0.0", "record_type": "root_correction_effect", "board_id": "AR-072",
        "treatment": "RootGaitConsistencyTool u=1.0 (contact-consistent root solve, GT-free)",
        "split_id": "protocol_rep_pool_seed20260608 mdm (300 prompts x 3 seeds)",
        "invariance_check": {"n_samples": len(metas), "local_violations": inv_violations},
        "mechanism_target": {
            "ratio_before": {"mean": round(float(ratio_b.mean()), 4), "ci95": _boot_ci(ratio_b, rng)},
            "ratio_after": {"mean": round(float(ratio_a.mean()), 4), "ci95": _boot_ci(ratio_a, rng)},
            "n_locomotion": len(loco),
            "note": "GT 는 판정 참조로만 — 보정 자체는 GT-free (constrained_frame_frac 참조)"},
        "foot_skate_world": {
            "delta_mean": round(float(fs_d.mean()), 5), "ci95": _boot_ci(fs_d, rng),
            "frac_improved": round(float((fs_d < 0).mean()), 3),
            "note": "정의상 감소 기대 — 성공 기준 아님 (사전등록)"},
        "root_accel_p95": {
            "delta_mean": round(float(acc_d.mean()), 5), "ci95": _boot_ci(acc_d, rng),
            "frac_worsened": round(float((acc_d > 0).mean()), 3),
            "before_mean": round(float(np.mean([r["acc_b"] for r in rows])), 5),
            "after_mean": round(float(np.mean([r["acc_a"] for r in rows])), 5),
            "gt_reference_mean": round(float(np.mean(list(gt_accel.values()))), 5),
            "note": "원본 root 는 과소 이동이라 accel 도 과소 — GT 기준과 비교해 '자연 동역학 회복'인지 '과도 jitter'인지 판정"},
        "constrained_frame_frac_mean": round(float(np.mean([r["cff"] for r in rows])), 4),
        "corrected_dump": str(DUMP.relative_to(REPO_ROOT)),
        "next": "Cat-A guard (R-Prec/MM-Dist, mgpt env) — 기각 조건: R@1 CI-밖 하락 또는 MM-Dist 유의 악화",
        "claim_boundary": "물리+기전 표적 측정 — 지각/semantic 판정 아님 (Cat-A 와 A/B 이후)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"invariance violations: {inv_violations}/{len(metas)}")
    print(f"mechanism ratio: {ratio_b.mean():.3f} -> {ratio_a.mean():.3f} (target 1.0, n_loco={len(loco)})")
    print(f"fs delta: {fs_d.mean():+.5f} (improved {(fs_d<0).mean()*100:.0f}%)")
    print(f"root accel p95 delta: {acc_d.mean():+.5f} (worsened {(acc_d>0).mean()*100:.0f}%)")
    print(f"[OK] wrote {args.output}; corrected dump -> {DUMP}")


if __name__ == "__main__":
    main()
