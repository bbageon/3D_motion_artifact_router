"""Step G-1 (사용자 directive 2026-05-28): standard metric FID + Diversity (HumanML3D tm2t evaluator).

사용자 directive:
> "Step G는 3-level primary로 진행. FID / R-Precision / MM-Dist / Diversity / physical
>  violation rate / perceptual sanity. 3-level learned / 5-level learned / safe oracle /
>  B2-family를 같이 비교."

본 도구는 HumanML3D 의 official tm2t evaluator (movement_encoder + motion_encoder,
deps/t2m/t2m/text_mot_match/model/finest.tar) 로 motion embedding 을 계산 → FID + Diversity.

FID(method, GT): 각 correction method 의 output 분포 가 natural HumanML3D 분포 와
얼마나 가까운가 (lower = 자연 분포 에 가까움). NetGain (Category C proxy) 과 별개 의
**Category A standard metric** (AGENTS.md §3-20).

Methods (G2 test300):
  - GT_ref: HumanML3D clean motions (natural distribution reference)
  - noop: G2 original (correction 없음)
  - B2-best: G2 + best VelocitySmoothing strength
  - safe_oracle_3level / safe_oracle_5level: gate-aware oracle corrected

NOTE: mgpt env 필요 (torch + motion_process + tm2t_evaluator). R-Precision/MM-Dist
(text co-embedding) 는 Step G-2 별도.

CLI (mgpt env):
    conda activate mgpt
    python tools/standard_metric_fid.py \
        --output evals/snapshots/standard_metric_fid_v1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MGPT = REPO_ROOT / "external_assets" / "MotionGPT"
sys.path.insert(0, str(MGPT))

import torch

# chumpy/numpy alias patch (same as _smpl_fit_inline).
import builtins as _b
for _n in ("bool", "int", "float", "complex", "object", "str"):
    if not hasattr(np, _n):
        setattr(np, _n, getattr(_b, _n))

from mGPT.data.humanml.common.skeleton import Skeleton
from mGPT.data.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
import mGPT.data.humanml.scripts.motion_process as mp
from mGPT.archs.tm2t_evaluator import MovementConvEncoder, MotionEncoderBiGRUCo

T2M_DIR = MGPT / "deps" / "t2m" / "t2m"
CHECKPOINT = T2M_DIR / "text_mot_match" / "model" / "finest.tar"
MEAN_PATH = T2M_DIR / "Comp_v6_KLD005" / "meta" / "mean.npy"
STD_PATH = T2M_DIR / "Comp_v6_KLD005" / "meta" / "std.npy"
TEMPLATE = REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints" / "000021.npy"
UNIT_LEN = 4


def _setup_motion_process_globals():
    """Set the module globals process_file relies on (from motion_process __main__)."""
    mp.fid_r, mp.fid_l = [8, 11], [7, 10]
    mp.face_joint_indx = [2, 1, 17, 16]
    mp.r_hip, mp.l_hip = 2, 1
    mp.l_idx1, mp.l_idx2 = 5, 8
    mp.joints_num = 22
    mp.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    mp.kinematic_chain = t2m_kinematic_chain
    example = np.load(str(TEMPLATE)).reshape(-1, 22, 3)
    tgt_skel = Skeleton(mp.n_raw_offsets, mp.kinematic_chain, "cpu")
    mp.tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(example[0]))


def _to_features(positions: np.ndarray) -> np.ndarray | None:
    """xyz (T,22,3) → HumanML3D 263-dim feature. None if too short."""
    if positions.shape[0] < 8:
        return None
    try:
        data, _, _, _ = mp.process_file(positions.astype(np.float64), 0.002)
        return data  # (T-1, 263)
    except Exception as e:
        print(f"[WARN] process_file: {e}")
        return None


def _load_encoders():
    move = MovementConvEncoder(259, 512, 512)
    motion = MotionEncoderBiGRUCo(512, 1024, 512)
    ck = torch.load(str(CHECKPOINT), map_location="cpu")
    move.load_state_dict(ck["movement_encoder"])
    motion.load_state_dict(ck["motion_encoder"])
    move.eval(); motion.eval()
    return move, motion


def _embed(feats_list: list[np.ndarray], mean, std, move, motion) -> np.ndarray:
    """List of (T,263) → (N, emb_dim) embeddings."""
    embs = []
    with torch.no_grad():
        for feat in feats_list:
            f = (feat - mean) / std
            t = torch.from_numpy(f).float().unsqueeze(0)  # (1, T, 263)
            mov = move(t[..., :-4])  # (1, T', 512)
            m_len = torch.tensor([mov.shape[1]])
            emb = motion(mov, m_len)  # (1, emb)
            embs.append(emb.squeeze(0).numpy())
    return np.array(embs)


def _fid(mu1, cov1, mu2, cov2):
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * np.trace(covmean))


def _diversity(embs, n_pairs=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(embs)
    if n < 2:
        return 0.0
    idx1 = rng.choice(n, n_pairs); idx2 = rng.choice(n, n_pairs)
    return float(np.mean(np.linalg.norm(embs[idx1] - embs[idx2], axis=1)))


def _stats(embs):
    return embs.mean(axis=0), np.cov(embs, rowvar=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--gt-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--oracle-3level", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_g2_hml3d_test300_3level_v1.json")
    parser.add_argument("--oracle-5level", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_g2_hml3d_test300_v1.json")
    parser.add_argument("--n-gt", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "standard_metric_fid_v1.json")
    args = parser.parse_args()

    _setup_motion_process_globals()
    mean = np.load(str(MEAN_PATH)); std = np.load(str(STD_PATH))
    move, motion = _load_encoders()
    print(f"[INFO] encoders loaded. mean {mean.shape}, std {std.shape}")

    # Correction tools (motion-router logic — reuse via direct import from repo).
    sys.path.insert(0, str(REPO_ROOT))
    from correction_tools import FootLockTool, BoneProjectionTool, VelocitySmoothingTool
    tools = {"FootLockTool": FootLockTool(default_ground_y=0.0),
             "BoneProjectionTool": BoneProjectionTool(),
             "VelocitySmoothingTool": VelocitySmoothingTool()}

    def _apply_seq(m, seq):
        T = m.shape[0]; out = m.copy()
        for s in seq:
            if s[0] not in tools: break
            out, _ = tools[s[0]].apply(out, target_part=s[1], target_joints=[], frame_range=(0, T-1), strength=s[2])
        return out

    o3 = {p["trial_id"]: p for p in json.load(open(args.oracle_3level, encoding="utf-8"))["per_sample"]}
    o5 = {p["trial_id"]: p for p in json.load(open(args.oracle_5level, encoding="utf-8"))["per_sample"]}

    # GT reference: random HumanML3D clean motions.
    rng = np.random.default_rng(args.seed)
    gt_files = sorted(args.gt_dir.glob("*.npy"))
    gt_chosen = [gt_files[i] for i in rng.choice(len(gt_files), args.n_gt, replace=False)]

    methods = {"GT_ref": [], "noop": [], "b2_netgain_best": [], "b2_artifact_best": [],
               "safe_oracle_3level": [], "safe_oracle_5level": []}
    # NetGain (Protocol B, vs original): -target_delta - alpha*mpjpe. alpha=5.0.
    from evaluators import DEFAULT_EVALUATORS as _DE
    _ALPHA = 5.0
    def _target_g2(mot):
        return float(np.mean([max((r.score for r in ev.evaluate(mot)), default=0.0)
                              for ev in _DE if ev.name in ("FootFloatingEvaluator","BoneLengthEvaluator","VelocityJitterEvaluator")]))
    def _netgain_b2(corrected, original):
        return -(_target_g2(corrected) - _target_g2(original)) - _ALPHA * float(np.mean(np.linalg.norm(corrected-original, axis=-1)))
    def _artifact_b2(mot):
        return sum(max((r.score for r in ev.evaluate(mot)), default=0.0) for ev in _DE)

    print("[INFO] converting GT reference...")
    for p in gt_chosen:
        f = _to_features(np.load(str(p)))
        if f is not None: methods["GT_ref"].append(f)

    g2_files = sorted(args.g2_batch_dir.glob("motion_*.npy"))
    print(f"[INFO] converting {len(g2_files)} G2 methods...")
    for i, p in enumerate(g2_files, 1):
        tid = p.stem
        m = np.load(str(p)).astype(np.float64)
        # noop
        f = _to_features(m)
        if f is not None: methods["noop"].append(f)
        # 두 B2 variant: b2_netgain_best (Protocol B NetGain, F-3 일관) + b2_artifact_best (artifact-min, F-5 일관).
        ng_best_m, ng_best = m, 0.0  # noop NetGain = 0
        art_best_m, art_best = m, None
        for st in ("small", "medium", "large"):
            out, _ = tools["VelocitySmoothingTool"].apply(m, target_part="full_body", target_joints=[], frame_range=(0, m.shape[0]-1), strength=st)
            ng = _netgain_b2(out, m)
            if ng > ng_best:
                ng_best, ng_best_m = ng, out
            art = _artifact_b2(out)
            if art_best is None or art < art_best:
                art_best, art_best_m = art, out
        fng = _to_features(ng_best_m); fart = _to_features(art_best_m)
        if fng is not None: methods["b2_netgain_best"].append(fng)
        if fart is not None: methods["b2_artifact_best"].append(fart)
        # safe oracle 3-level / 5-level
        for okey, od in (("safe_oracle_3level", o3), ("safe_oracle_5level", o5)):
            sb = od.get(tid, {}).get("safe_best")
            cm = _apply_seq(m, sb["sequence"]) if (sb and sb["length"] > 0) else m
            f = _to_features(cm)
            if f is not None: methods[okey].append(f)
        if i % 50 == 0:
            print(f"   {i}/{len(g2_files)}")

    # Embed + FID + Diversity.
    print("[INFO] embedding + FID...")
    emb = {k: _embed(v, mean, std, move, motion) for k, v in methods.items() if v}
    gt_mu, gt_cov = _stats(emb["GT_ref"])
    results = {}
    for k, e in emb.items():
        if k == "GT_ref":
            results[k] = {"n": len(e), "diversity": _diversity(e)}
            continue
        mu, cov = _stats(e)
        results[k] = {"n": len(e), "FID_vs_GT": _fid(gt_mu, gt_cov, mu, cov), "diversity": _diversity(e)}

    out = {
        "schema_version": "1.0.0", "record_type": "standard_metric_fid",
        "task_id": "standard_metric_fid_v1",
        "evaluator": "HumanML3D tm2t (movement+motion encoder, t2m finest.tar)",
        "metric_category": "A (standard, HumanML3D/Guo 2022 CVPR)",
        "gt_reference": "HumanML3D clean new_joints (n={})".format(args.n_gt),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Step G-1: Standard Metric FID + Diversity ===")
    print(f"  {'method':<22} {'n':<6} {'FID_vs_GT':<12} {'Diversity'}")
    for k, r in results.items():
        fid = f"{r['FID_vs_GT']:.4f}" if "FID_vs_GT" in r else "(ref)"
        print(f"  {k:<22} {r['n']:<6} {fid:<12} {r['diversity']:.4f}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
