"""Step G-3 (사용자 directive 2026-05-28): synthetic severe standard metric (FID/Diversity + R-Prec/MM-Dist).

사용자 directive:
> "G-3 synthetic severe는 그 다음. b2_netgain_best degradation 이 여기서 보일 것
>  (NetGain-best 가 over-smooth 하는 영역)."

G-1/G-2 (G2 natural, low/moderate severity) 에서는 correction 이 거의 불필요 → noop 이
이미 좋고 b2_artifact_best 만 FID 악화. 본 G-3 는 **반대 regime** (synthetic severe,
high severity) — corruption 이 분포 를 크게 밀어내므로 correction 이 실제로 필요. 이
영역에서 NetGain Protocol A best (b2_netgain_best) 가 aggressive smoothing 으로 over-correct
하는지, gate-aware safe oracle 이 더 잘 복원 하는지 비교.

Methods (synthetic severe corpus, foot_floating(0.08)+jitter(0.05) on clean HumanML3D):
  - GT_ref: clean HumanML3D (natural distribution reference)
  - noop: corrupted (correction 없음 = synthetic severe input)
  - b2_netgain_best: corrupted + best VelocitySmoothing by NetGain Protocol A (vs clean)
  - b2_artifact_best: corrupted + best VelocitySmoothing by artifact-min
  - safe_oracle_3level / safe_oracle_5level: corrupted + gate-aware safe sequence

NetGain Protocol A (clean GT reference) = -target_delta_A - alpha*fidelity_loss,
synthetic severe oracle 와 동일 (CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1, alpha=5.0).
TARGET_EVALUATORS_A = (FootFloating, VelocityJitter) — synthetic oracle 와 일치.

CAVEAT: n=60 per corrected method → FID 추정 noisy (appendix evidence, AGENTS.md §3-17
controlled diagnostic). primary evidence 는 G-1/G-2 (G2 real-distribution).

NOTE: mgpt env 필요 (motion_process + tm2t_evaluator + spacy + GloVe).

CLI (mgpt env):
    conda activate mgpt
    python tools/standard_metric_synthetic_severe.py \
        --output evals/snapshots/standard_metric_synthetic_severe_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MGPT = REPO_ROOT / "external_assets" / "MotionGPT"
sys.path.insert(0, str(MGPT))
sys.path.insert(0, str(REPO_ROOT))

import torch
import builtins as _b
for _n in ("bool", "int", "float", "complex", "object", "str"):
    if not hasattr(np, _n):
        setattr(np, _n, getattr(_b, _n))

import tools.standard_metric_fid as G
import tools.standard_metric_rprec as R
from tools.safe_sequence_oracle_synthetic_run import _multi_inject, TARGET_EVALUATORS_A
from orchestrator.oracle_single_step import CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1

TEXTS_DIR = REPO_ROOT / "external_assets" / "HumanML3D" / "texts"


def _caption(trial_id: str) -> str | None:
    f = TEXTS_DIR / f"{trial_id}.txt"
    if not f.exists():
        return None
    line = f.read_text(encoding="utf-8").splitlines()[0]
    return line.split("#")[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-3level", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_synthetic_severe_3level_v1.json")
    parser.add_argument("--oracle-5level", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_synthetic_severe_v1.json")
    parser.add_argument("--gt-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--seed", type=int, default=42, help="synthetic injection seed (oracle default=42)")
    parser.add_argument("--n-gt", type=int, default=300)
    parser.add_argument("--gt-seed", type=int, default=43)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "standard_metric_synthetic_severe_v1.json")
    args = parser.parse_args()

    G._setup_motion_process_globals()
    mean = np.load(str(G.MEAN_PATH)); std = np.load(str(G.STD_PATH))
    move, motion_enc = G._load_encoders()
    text_enc = R._load_text_encoder()
    w_vectorizer = R.WordVectorizer(str(R.GLOVE_DIR), "our_vab")
    print("[INFO] encoders + word vectorizer loaded")

    from correction_tools import FootLockTool, BoneProjectionTool, VelocitySmoothingTool
    from evaluators import DEFAULT_EVALUATORS
    tools = {"FootLockTool": FootLockTool(default_ground_y=0.0),
             "BoneProjectionTool": BoneProjectionTool(),
             "VelocitySmoothingTool": VelocitySmoothingTool()}

    def _apply_seq(m, seq):
        T = m.shape[0]; out = m.copy()
        for s in seq:
            if s[0] not in tools: break
            out, _ = tools[s[0]].apply(out, target_part=s[1], target_joints=[], frame_range=(0, T-1), strength=s[2])
        return out

    # NetGain Protocol A (clean GT reference) — synthetic oracle 와 동일.
    alpha = float(CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1["alpha"])

    def _target_a(mot):
        return float(np.mean([max((r.score for r in ev.evaluate(mot)), default=0.0)
                              for ev in DEFAULT_EVALUATORS if ev.name in TARGET_EVALUATORS_A]))

    def _mpjpe(a, b):
        return float(np.mean(np.linalg.norm(a - b, axis=-1)))

    def _netgain_a(corrected, corrupted, clean):
        td = _target_a(corrected) - _target_a(corrupted)
        fl = _mpjpe(corrected, clean) - _mpjpe(corrupted, clean)
        return -td - alpha * fl

    def _artifact(mot):
        return sum(max((r.score for r in ev.evaluate(mot)), default=0.0) for ev in DEFAULT_EVALUATORS)

    o3 = {p["trial_id"]: p for p in json.load(open(args.oracle_3level, encoding="utf-8"))["per_sample"]}
    o5 = {p["trial_id"]: p for p in json.load(open(args.oracle_5level, encoding="utf-8"))["per_sample"]}
    snap = json.load(open(args.oracle_3level, encoding="utf-8"))
    trials = [(p["trial_id"], p["sample_path"]) for p in snap["per_sample"]]
    print(f"[INFO] {len(trials)} synthetic severe samples (seed={args.seed})")

    method_keys = ["noop", "b2_netgain_best", "b2_artifact_best", "safe_oracle_3level", "safe_oracle_5level"]
    feats = {k: [] for k in method_keys}
    feats_clean = []  # for FID GT alignment if needed
    captions = []
    for i, (tid, spath) in enumerate(trials, 1):
        p = Path(spath)
        if not p.exists():
            p = args.gt_dir / f"{tid}.npy"
        if not p.exists():
            print(f"[WARN] {tid}: clean missing"); continue
        clean = np.load(str(p)).astype(np.float64)
        if clean.ndim != 3 or clean.shape[1] != 22:
            continue
        corrupted = _multi_inject(clean, seed=args.seed)
        cap = _caption(tid)
        if cap is None:
            print(f"[WARN] {tid}: caption missing"); continue
        # b2 variants on corrupted.
        ng_best_m, ng_best = corrupted, 0.0  # noop NetGain = 0
        art_best_m, art_best = corrupted, None
        for st in ("small", "medium", "large"):
            out, _ = tools["VelocitySmoothingTool"].apply(
                corrupted, target_part="full_body", target_joints=[], frame_range=(0, corrupted.shape[0]-1), strength=st)
            ng = _netgain_a(out, corrupted, clean)
            if ng > ng_best: ng_best, ng_best_m = ng, out
            art = _artifact(out)
            if art_best is None or art < art_best: art_best, art_best_m = art, out
        sb3 = o3.get(tid, {}).get("safe_best"); sb5 = o5.get(tid, {}).get("safe_best")
        m3 = _apply_seq(corrupted, sb3["sequence"]) if (sb3 and sb3["length"] > 0) else corrupted
        m5 = _apply_seq(corrupted, sb5["sequence"]) if (sb5 and sb5["length"] > 0) else corrupted
        f_noop = G._to_features(corrupted)
        f_ng = G._to_features(ng_best_m); f_art = G._to_features(art_best_m)
        f3 = G._to_features(m3); f5 = G._to_features(m5)
        if any(f is None for f in (f_noop, f_ng, f_art, f3, f5)):
            print(f"[WARN] {tid}: feature extraction failed"); continue
        feats["noop"].append(f_noop); feats["b2_netgain_best"].append(f_ng)
        feats["b2_artifact_best"].append(f_art)
        feats["safe_oracle_3level"].append(f3); feats["safe_oracle_5level"].append(f5)
        fc = G._to_features(clean)
        if fc is not None: feats_clean.append(fc)
        captions.append(cap)
        if i % 10 == 0:
            print(f"   {i}/{len(trials)}")

    # GT reference: random clean motions (stable FID reference).
    rng = np.random.default_rng(args.gt_seed)
    gt_files = sorted(args.gt_dir.glob("*.npy"))
    gt_chosen = [gt_files[k] for k in rng.choice(len(gt_files), args.n_gt, replace=False)]
    gt_feats = []
    for gp in gt_chosen:
        f = G._to_features(np.load(str(gp)))
        if f is not None: gt_feats.append(f)
    print(f"[INFO] GT_ref n={len(gt_feats)}, per-method n={len(captions)}")

    # FID + Diversity.
    gt_emb = G._embed(gt_feats, mean, std, move, motion_enc)
    gt_mu, gt_cov = G._stats(gt_emb)
    fid_results = {"GT_ref": {"n": len(gt_emb), "diversity": G._diversity(gt_emb)}}
    method_embs = {}
    for k in method_keys:
        e = G._embed(feats[k], mean, std, move, motion_enc)
        method_embs[k] = e
        mu, cov = G._stats(e)
        fid_results[k] = {"n": len(e), "FID_vs_GT": G._fid(gt_mu, gt_cov, mu, cov),
                          "diversity": G._diversity(e)}

    # R-Precision + MM-Dist (text same per sample, motion differs by method).
    text_embs = R._embed_text(captions, w_vectorizer, text_enc)
    rprec_results = {}
    for k in method_keys:
        top123, mmdist = R._rprecision_mmdist(text_embs, method_embs[k])
        rprec_results[k] = {"n": len(feats[k]), "R_precision_top1": top123[0],
                            "R_precision_top2": top123[1], "R_precision_top3": top123[2],
                            "MM_Dist": mmdist}

    out = {
        "schema_version": "1.0.0", "record_type": "standard_metric_synthetic_severe",
        "task_id": "standard_metric_synthetic_severe_v1",
        "evaluator": "HumanML3D tm2t (movement+motion+text encoder, finest.tar)",
        "metric_category": "A (standard, HumanML3D/Guo 2022 CVPR)",
        "evidence_tier": "controlled diagnostic (synthetic, AGENTS.md §3-17 — appendix only)",
        "corruption": "foot_floating(0.08) + jitter(0.05) on clean HumanML3D",
        "netgain_protocol": "A (clean GT reference, alpha=5.0)",
        "gt_reference": f"HumanML3D clean new_joints (n={len(gt_feats)})",
        "n_samples_per_method": len(captions),
        "caveat": "n per corrected method is small (~60) → FID 추정 noisy. primary evidence = G-1/G-2 (G2 real-distribution).",
        "fid_diversity": fid_results,
        "rprecision_mmdist": rprec_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Step G-3: Synthetic Severe FID + Diversity ===")
    print(f"  {'method':<22} {'n':<6} {'FID_vs_GT':<12} {'Diversity'}")
    for k, r in fid_results.items():
        fid = f"{r['FID_vs_GT']:.4f}" if "FID_vs_GT" in r else "(ref)"
        print(f"  {k:<22} {r['n']:<6} {fid:<12} {r['diversity']:.4f}")
    print("\n=== Step G-3: Synthetic Severe R-Precision + MM-Dist ===")
    print(f"  {'method':<22} {'R@1':<8} {'R@2':<8} {'R@3':<8} {'MM-Dist'}")
    for k, r in rprec_results.items():
        print(f"  {k:<22} {r['R_precision_top1']:.4f}  {r['R_precision_top2']:.4f}  {r['R_precision_top3']:.4f}  {r['MM_Dist']:.4f}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
