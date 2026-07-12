"""AR-077 보완(피드백 5): GT-free STOP 분리 검증.

질문: GT root ratio(추론 시 관측 불가)가 아니라 **관측 가능한 신호만**(foot_skate,
path_gain, induced_disp)으로 MDM=apply / VQ=STOP 이 실제로 갈리는가.

AR-074 GT-free STOP rule: apply IF (foot_skate 높음 AND path_gain 정상) else STOP.
본 도구는 3 generator locomotion 165 prompt 에 대해 이 신호들의 분포를 비교하고,
"skate 高" gate 하나만으로 generator 가 분리되는지(AUC) 측정.

CLI (motion3d env):
    python tools/gtfree_stop_separation_ar077.py
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

POOL_ROOT = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608"
GT_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints"
GENS = ("mdm", "motiongpt", "momask")
LOCO_THRESH = 0.010


def _rs(m):
    return float(np.linalg.norm(np.diff(m[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean())


def _rp(m):
    return float(np.linalg.norm(np.diff(m[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).sum())


def _ci(a, rng, n=1000):
    a = np.asarray(a, float); m = len(a)
    bs = [float(np.mean(a[rng.integers(0, m, m)])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]


def _auc(pos, neg):
    """pos(MDM) 가 neg(VQ) 보다 큰지 — Mann-Whitney AUC."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(allv); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return round(float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "gtfree_stop_separation_ar077_v1.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tool = RootGaitConsistencyTool()

    gt_cache = {}
    sig = {g: defaultdict(list) for g in GENS}
    for gen in GENS:
        pool = POOL_ROOT / gen
        metas = [json.load(open(p, encoding="utf-8")) for p in sorted(pool.glob("*.json"))
                 if p.name != "_pool_summary.json"]
        if args.limit:
            metas = metas[:args.limit]
        per = defaultdict(lambda: defaultdict(list))
        for m in metas:
            sid = m["sample_id"]
            if sid not in gt_cache:
                gt = np.load(GT_DIR / f"{sid}.npy").astype(np.float64)
                gt_cache[sid] = _rs(gt)
            traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
            T = traj.shape[0]
            corr, rep = tool.apply(traj, target_part="root", target_joints=[],
                                   frame_range=(0, T - 1), strength="large",
                                   metadata={"coord_space": "trajectory"})
            off = (corr[:, PELVIS, :] - traj[:, PELVIS, :])[:, [0, 2]]
            d = per[sid]
            d["skate"].append(foot_skate_world(traj))
            d["path_gain"].append(_rp(corr) / max(_rp(traj), 1e-6))
            d["induced"].append(float(np.linalg.norm(off, axis=1).mean()))
            d["gt"].append(gt_cache[sid])
        for sid, d in per.items():
            if float(np.mean(d["gt"])) <= LOCO_THRESH:
                continue
            sig[gen]["skate"].append(float(np.mean(d["skate"])))
            sig[gen]["path_gain"].append(float(np.mean(d["path_gain"])))
            sig[gen]["induced"].append(float(np.mean(d["induced"])))

    # AR-074 skate-high gate: MDM 의 skate 하위 25% 를 apply 임계로 (관측 기반). VQ 통과율?
    mdm_skate = np.array(sig["mdm"]["skate"])
    thr = float(np.percentile(mdm_skate, 25))
    per_gen = {}
    for g in GENS:
        s = np.array(sig[g]["skate"])
        per_gen[g] = {
            "n": len(s),
            "skate": {"mean": round(float(s.mean()), 5), "ci95": _ci(s, rng)},
            "path_gain": {"mean": round(float(np.mean(sig[g]["path_gain"])), 3)},
            "induced_disp": {"mean": round(float(np.mean(sig[g]["induced"])), 4)},
            "frac_above_skate_thr": round(float((s > thr).mean()), 3),  # apply 후보 비율
        }

    out = {
        "schema_version": "1.0.0", "record_type": "gtfree_stop_separation", "board_id": "AR-077",
        "question": "GT root ratio 없이 관측 신호(skate/path_gain/induced_disp)만으로 MDM=apply, VQ=STOP 분리되나",
        "skate_apply_threshold": {"value": round(thr, 5), "def": "MDM foot_skate 25th pct (AR-074 skate-high gate proxy)"},
        "per_generator": per_gen,
        "separation_auc_MDM_vs_VQ": {
            "skate": {"vs_motiongpt": _auc(sig["mdm"]["skate"], sig["motiongpt"]["skate"]),
                      "vs_momask": _auc(sig["mdm"]["skate"], sig["momask"]["skate"])},
            "induced_disp": {"vs_motiongpt": _auc(sig["mdm"]["induced"], sig["motiongpt"]["induced"]),
                             "vs_momask": _auc(sig["mdm"]["induced"], sig["momask"]["induced"])},
        },
        "interpretation": "frac_above_skate_thr: MDM 높음(apply)·VQ 낮음(STOP) 이면 skate gate 하나로 GT-free 분리 성립. "
                          "AUC≈1 = 신호가 MDM/VQ 를 강하게 분리.",
        "claim_boundary": "GT-free 신호 분리 검증 (관측). generator 인과(VQ 구조) 주장 금지. MDM·단일 벤치마크.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    import sys as _s; _s.stdout.reconfigure(encoding="utf-8")
    for g in GENS:
        pg = per_gen[g]
        print(f"[{g}] skate {pg['skate']['mean']} ci{pg['skate']['ci95']} | induced {pg['induced_disp']['mean']} "
              f"| path_gain {pg['path_gain']['mean']} | frac>skate_thr {pg['frac_above_skate_thr']}")
    print(f"skate AUC MDM vs MGPT {out['separation_auc_MDM_vs_VQ']['skate']['vs_motiongpt']} "
          f"vs MoMask {out['separation_auc_MDM_vs_VQ']['skate']['vs_momask']}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
