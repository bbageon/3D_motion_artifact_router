"""AR-074: root-aware correction 의 STOP 경계 + GT-free 판정 신호.

질문: root correction 이 어디서 harm(=STOP 대상)이고, 추론 시 GT 없이 어떻게 판정하나.

측정 (원본 MDM 전 300 prompt, GT 는 label/검증용):
  - 상태 라벨: is_locomotion = GT root speed > 0.01.
  - correction 유도 root 변위 (GT-free 관측): induced_disp = mean|offset|,
    root_path_before/after (절대 m).
  - GT-free 후보 신호: constrained_frame_frac(cff), path_gain = path_after/max(path_before,eps),
    induced_disp.
  - self-STOP 검정: non-loco 에서 induced_disp 가 loco 대비 작은가 (자동 no-op)?
  - over-correction: ratio_after>1.5 를 GT-free 신호로 예측 가능한가 (분리도).
  - GT-free STOP rule 후보 + no-harm(non-loco 에서 밀어버림 여부).

CLI (motion3d env):
    python tools/root_stop_boundary_ar074.py
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
LOCO_THRESH = 0.010


def _root_path(m):
    return float(np.linalg.norm(np.diff(m[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).sum())


def _root_speed(m):
    return float(np.linalg.norm(np.diff(m[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean())


def _boot_ci(a, rng, n=1000):
    a = np.asarray(a, float); m = len(a)
    if m == 0:
        return [None, None]
    bs = [float(np.mean(a[rng.integers(0, m, m)])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "root_stop_boundary_ar074_v1.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tool = RootGaitConsistencyTool()

    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(POOL.glob("*.json"))
             if p.name != "_pool_summary.json"]
    if args.limit:
        metas = metas[:args.limit]

    per = defaultdict(lambda: defaultdict(list))
    gt_speed = {}
    for m in metas:
        traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        T = traj.shape[0]
        corr, rep = tool.apply(traj, target_part="root", target_joints=[],
                               frame_range=(0, T - 1), strength="large",
                               metadata={"coord_space": "trajectory"})
        offset = (corr[:, PELVIS, :] - traj[:, PELVIS, :])[:, [0, 2]]
        sid = m["sample_id"]
        d = per[sid]
        d["induced_disp"].append(float(np.linalg.norm(offset, axis=1).mean()))
        d["path_b"].append(_root_path(traj)); d["path_a"].append(_root_path(corr))
        d["vr_a"].append(_root_speed(corr))
        d["cff"].append(rep.metadata["constrained_frame_frac"])
        d["fs_b"].append(foot_skate_world(traj)); d["fs_a"].append(foot_skate_world(corr))
        if sid not in gt_speed:
            gt = np.load(GT_DIR / f"{sid}.npy").astype(np.float64)
            gt_speed[sid] = _root_speed(gt)

    rows = []
    for sid, d in per.items():
        gt = gt_speed[sid]
        pb = float(np.mean(d["path_b"]))
        rows.append({
            "sid": sid, "gt_speed": gt, "is_loco": gt > LOCO_THRESH,
            "induced_disp": float(np.mean(d["induced_disp"])),
            "path_b": pb, "path_a": float(np.mean(d["path_a"])),
            "path_gain": float(np.mean(d["path_a"])) / max(pb, 1e-6),
            "vr_a": float(np.mean(d["vr_a"])), "ratio_after": float(np.mean(d["vr_a"])) / max(gt, 1e-9),
            "cff": float(np.mean(d["cff"])),
            "fs_b": float(np.mean(d["fs_b"])), "fs_a": float(np.mean(d["fs_a"])),
        })
    loco = [r for r in rows if r["is_loco"]]
    nonloco = [r for r in rows if not r["is_loco"]]

    # ---- self-STOP: non-loco 에서 유도 변위가 loco 대비 작은가.
    disp_loco = np.array([r["induced_disp"] for r in loco])
    disp_non = np.array([r["induced_disp"] for r in nonloco])
    self_stop = {
        "loco_induced_disp_mean": round(float(disp_loco.mean()), 5), "loco_ci95": _boot_ci(disp_loco, rng),
        "nonloco_induced_disp_mean": round(float(disp_non.mean()), 5), "nonloco_ci95": _boot_ci(disp_non, rng),
        "ratio_nonloco_over_loco": round(float(disp_non.mean() / max(disp_loco.mean(), 1e-9)), 3),
        "nonloco_root_path_after_mean": round(float(np.mean([r["path_a"] for r in nonloco])), 4),
        "verdict": None,
    }
    # non-loco 유도 변위가 loco 의 절반 미만이면 부분 self-STOP; 절대 root path_after 작으면 no-harm.
    self_stop["verdict"] = (
        "self-STOP 경향 (non-loco 변위 << loco)" if self_stop["ratio_nonloco_over_loco"] < 0.5
        else "harm 위험 (non-loco 도 큰 변위 — 명시적 gate 필요)")

    # ---- over-correction 예측: ratio_after>1.5 를 GT-free 신호로 분리 (path_gain, induced_disp, cff).
    over = np.array([r["ratio_after"] > 1.5 for r in loco])
    def _auc(sig):
        s = np.array([r[sig] for r in loco]); y = over
        # AUC via rank (Mann-Whitney).
        if y.sum() == 0 or y.sum() == len(y):
            return None
        order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
        auc = (ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
        return round(float(auc), 3)
    over_pred = {"n_over": int(over.sum()), "n_loco": len(loco),
                 "auc_path_gain": _auc("path_gain"), "auc_induced_disp": _auc("induced_disp"),
                 "auc_cff": _auc("cff"),
                 "note": "AUC>0.7 = 해당 GT-free 신호로 over-correction 사전 식별 가능"}

    # ---- GT-free STOP rule 후보: induced_disp 임계로 (already-good/self-STOP) 걸러내기.
    # apply 가치 = fs 개선. STOP 이 맞는 케이스 = 유도 변위 큰데 fs 개선 작거나 non-loco.
    all_disp = np.array([r["induced_disp"] for r in rows])
    fs_improve = np.array([r["fs_b"] - r["fs_a"] for r in rows])  # 양수 = 개선
    corr_disp_fs = float(np.corrcoef(all_disp, fs_improve)[0, 1])

    out = {
        "schema_version": "1.0.0", "record_type": "root_stop_boundary", "board_id": "AR-074",
        "n_prompts": len(rows), "n_loco": len(loco), "n_nonloco": len(nonloco),
        "self_stop_test": self_stop,
        "over_correction_prediction": over_pred,
        "gt_free_signal": {
            "induced_disp_vs_fs_improve_corr": round(corr_disp_fs, 3),
            "note": "induced_disp = 추론 시 관측 가능 (GT 불요). fs 개선과 상관 높으면 apply 가치 proxy."},
        "stop_rule_candidate": {
            "rule": "STOP if (non-locomotion-like: 접지 backward-flow 미약 → induced_disp 작음 = self-STOP 자동) "
                    "OR (over-correction risk: path_gain 극단). apply otherwise.",
            "status": "관측 기반 rule 후보 — 학습 policy 아님. non-loco Cat-A/A/B 는 필요 시 후속."},
        "claim_boundary": "관측 진단 + rule 후보. 지각 harm 단정은 A/B 필요. MDM 단일.",
        "grounding": ["Safe Orchestration no-harm (position §0)", "PhysDiff ICCV2023 (post-proc side effect)"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8")
    print(f"self-STOP: loco disp {self_stop['loco_induced_disp_mean']} vs non-loco {self_stop['nonloco_induced_disp_mean']} "
          f"(ratio {self_stop['ratio_nonloco_over_loco']}) -> {self_stop['verdict']}")
    print(f"  non-loco root path after: {self_stop['nonloco_root_path_after_mean']} m (작을수록 no-harm)")
    print(f"over-correction AUC: path_gain {over_pred['auc_path_gain']} disp {over_pred['auc_induced_disp']} cff {over_pred['auc_cff']} (n_over {over_pred['n_over']}/{over_pred['n_loco']})")
    print(f"induced_disp vs fs_improve corr: {corr_disp_fs:.3f}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
