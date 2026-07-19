"""AR-077: VQ(MotionGPT/MoMask) root 상태 + no-harm 진단 (generator-conditioned routing).

프레이밍 (사전등록 spec): "VQ 에도 필요한가 vs STOP 이 정답인가". 3-시나리오
(깔끔/애매/놀람) 은 결과 보기 전 고정 — 성공 기준 = "결정이 generator/state 에 따라 다름".

측정 (MDM + MotionGPT + MoMask, locomotion 한정 = GT root speed>0.01, 동일 필터):
  - root ratio (v_root/GT) — VQ 에 deficit 있나 (AR-076 대비).
  - RootGaitConsistencyTool 적용 후 physical: fs Δ, induced_disp, path_gain (AR-074 GT-free 신호).
  - 보정본 dump (Cat-A 는 mgpt env 별도 단계 — vq_root_catA_ar077.py).
  - routing 판정: AR-074 STOP rule (apply IF skate高 AND path_gain 정상) 을 각 generator 에.

CLI (motion3d env):
    python tools/vq_root_noharm_ar077.py
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
DUMP_ROOT = REPO_ROOT / "external_assets" / "ar077_root_corrected"
GENS = ("mdm", "motiongpt", "momask")
LOCO_THRESH = 0.010


def _root_speed(m):
    return float(np.linalg.norm(np.diff(m[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).mean())


def _root_path(m):
    return float(np.linalg.norm(np.diff(m[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).sum())


def _boot_ci(a, rng, n=1000):
    a = np.asarray(a, float); m = len(a)
    bs = [float(np.mean(a[rng.integers(0, m, m)])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main() -> None:
    global POOL_ROOT, DUMP_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260719)
    ap.add_argument("--pool-root", type=Path, default=None,
                    help="독립 재현용 다른 pool (AR-024). 기본 = seed20260608")
    ap.add_argument("--dump-root", type=Path, default=None,
                    help="보정본 dump 위치 (pool 별 분리)")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "vq_root_noharm_ar077_v1.json")
    args = ap.parse_args()
    if args.pool_root is not None:
        POOL_ROOT = args.pool_root
    if args.dump_root is not None:
        DUMP_ROOT = args.dump_root
    rng = np.random.default_rng(args.seed)
    tool = RootGaitConsistencyTool()

    gt_speed_cache = {}
    results = {}
    for gen in GENS:
        pool = POOL_ROOT / gen
        dump = DUMP_ROOT / gen
        dump.mkdir(parents=True, exist_ok=True)
        metas = [json.load(open(p, encoding="utf-8")) for p in sorted(pool.glob("*.json"))
                 if p.name != "_pool_summary.json"]
        if args.limit:
            metas = metas[:args.limit]
        per = defaultdict(lambda: defaultdict(list))
        for m in metas:
            sid = m["sample_id"]
            if sid not in gt_speed_cache:
                gt = np.load(GT_DIR / f"{sid}.npy").astype(np.float64)
                gt_speed_cache[sid] = _root_speed(gt)
            traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
            T = traj.shape[0]
            corr, rep = tool.apply(traj, target_part="root", target_joints=[],
                                   frame_range=(0, T - 1), strength="large",
                                   metadata={"coord_space": "trajectory"})
            np.save(dump / f"{sid}__seed{m['seed']}.trajectory.npy", corr.astype(np.float32))
            offset = (corr[:, PELVIS, :] - traj[:, PELVIS, :])[:, [0, 2]]
            pb = _root_path(traj)
            d = per[sid]
            d["vr"].append(_root_speed(traj)); d["gt"].append(gt_speed_cache[sid])
            d["fs_b"].append(foot_skate_world(traj)); d["fs_a"].append(foot_skate_world(corr))
            d["induced"].append(float(np.linalg.norm(offset, axis=1).mean()))
            d["path_gain"].append(_root_path(corr) / max(pb, 1e-6))
            d["ratio_after"].append(_root_speed(corr) / max(gt_speed_cache[sid], 1e-9))

        loco = []
        for sid, d in per.items():
            gt = float(np.mean(d["gt"]))
            if gt <= LOCO_THRESH:
                continue
            loco.append({
                "ratio_before": float(np.mean(d["vr"])) / gt,
                "ratio_after": float(np.mean(d["ratio_after"])),
                "fs_delta": float(np.mean(d["fs_a"])) - float(np.mean(d["fs_b"])),
                "induced": float(np.mean(d["induced"])),
                "path_gain": float(np.mean(d["path_gain"])),
            })
        rb = np.array([r["ratio_before"] for r in loco])
        fsd = np.array([r["fs_delta"] for r in loco])
        ind = np.array([r["induced"] for r in loco])
        pg = np.array([r["path_gain"] for r in loco])
        results[gen] = {
            "n_locomotion": len(loco),
            "root_ratio_before": {"mean": round(float(rb.mean()), 4), "ci95": _boot_ci(rb, rng),
                                  "p50": round(float(np.percentile(rb, 50)), 4)},
            "fs_delta": {"mean": round(float(fsd.mean()), 5), "ci95": _boot_ci(fsd, rng)},
            "induced_disp": {"mean": round(float(ind.mean()), 4), "ci95": _boot_ci(ind, rng)},
            "path_gain": {"mean": round(float(pg.mean()), 3), "p95": round(float(np.percentile(pg, 95)), 3),
                          "frac_gt_1.5": round(float((rb * 0 + (np.array([r['ratio_after'] for r in loco]) > 1.5)).mean()), 3)},
            "corrected_dump": (str(dump.resolve().relative_to(REPO_ROOT))
                               if dump.resolve().is_relative_to(REPO_ROOT) else str(dump)),
        }
        print(f"[{gen}] n={len(loco)} root_ratio={results[gen]['root_ratio_before']['mean']} "
              f"ci{results[gen]['root_ratio_before']['ci95']} | fsΔ={results[gen]['fs_delta']['mean']:+.5f} "
              f"induced={results[gen]['induced_disp']['mean']} pathgain={results[gen]['path_gain']['mean']}")

    # ---- 시나리오 판정 (VQ vs MDM).
    mdm_r = results["mdm"]["root_ratio_before"]["mean"]
    scen = {}
    for gen in ("motiongpt", "momask"):
        r = results[gen]["root_ratio_before"]["mean"]
        ci = results[gen]["root_ratio_before"]["ci95"]
        if ci[0] > 0.9:
            s = "깔끔 (deficit 없음 → STOP 정답)"
        elif r > mdm_r + 0.1:
            s = "애매 (덜 부족 → state-conditioned)"
        else:
            s = "놀람 (MDM 수준 deficit)"
        scen[gen] = {"root_ratio": r, "ci95": ci, "scenario": s, "vs_mdm": round(r - mdm_r, 3)}

    out = {
        "schema_version": "1.0.0", "record_type": "vq_root_noharm", "board_id": "AR-077",
        "framing": "generator-conditioned routing — VQ apply vs STOP (사전등록 3-시나리오, HARKing 차단)",
        "split_id": "protocol_rep_pool_seed20260608 (3 gen) + HumanML3D GT, locomotion 한정",
        "results": results,
        "scenario_verdict": scen,
        "mdm_reference": {"root_ratio": mdm_r, "note": "AR-076: root progression collapse (기울기 0.027)"},
        "routing_implication": "MDM 은 apply (deficit 큼); VQ 판정은 scenario_verdict 참조. Cat-A no-harm 은 vq_root_catA_ar077.py (mgpt env).",
        "next": "Cat-A (R-Prec/MM-Dist/FID) 로 no-harm 확정 — VQ 에 correction 적용 시 악화하면 STOP 근거.",
        "claim_boundary": "diagnostic — 지각 harm 은 A/B 필요 (애매/놀람 시). 3-시나리오 사후 재프레임 금지.",
        "grounding": ["Guo MoMask CVPR2024", "Jiang MotionGPT NeurIPS2023", "Tevet MDM ICLR2023"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n=== 시나리오 판정 (VQ vs MDM root_ratio) ===")
    print(f"  MDM(ref) root_ratio={mdm_r}")
    for gen, sv in scen.items():
        print(f"  {gen:<10} root_ratio={sv['root_ratio']} ci{sv['ci95']} (vs MDM {sv['vs_mdm']:+.3f}) -> {sv['scenario']}")
    print(f"[OK] wrote {args.output}; dumps -> {DUMP_ROOT}")


if __name__ == "__main__":
    main()
