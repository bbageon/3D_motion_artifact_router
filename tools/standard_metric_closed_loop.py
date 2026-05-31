"""Step 6 part 2 (사용자 directive 2026-06-01): standard metric on closed-loop outputs.

사용자 directive 박제:
> "Step 6 part 1 에서 M0+gate 가 물리 제약 안 깨고 후보 고르는 능력은 확인. 남은 질문은
>  '그래서 원본보다 최종 모션 품질이 좋아졌나?'. Δ는 반드시 original/noop 대비 변화량으로.
>  절대값보다 '원본보다 좋아졌는가/망쳤는가'. NetGain 만 보면 안 됨 — 최종 주장은 '원본 생성
>  모션 대비 artifact 줄이면서, physical constraint 깨지 않고, standard quality 보존/개선'."

본 도구는 Step 6 closed-loop final motions (evals/snapshots/closed_loop_final_motions_v1/) 에
HumanML3D tm2t standard metric 적용:
  FID (vs HumanML3D clean GT reference) / R-Precision / MM-Dist / Diversity (Category A).
  + artifact_total_score (Category C proxy, 보조).
모든 metric 은 **original/noop 대비 Δ** 로 보고 (paired: 같은 state set).

비교 대상: original(noop) / M0 / M1 / M2 / M3 / random_gate / heuristic_gate / dense_oracle_step1.
group breakdown: G2 holdout 의 motion_group 별.

NOTE: mgpt env 필요 (motion_process + tm2t + spacy + GloVe).

CLI (mgpt env):
    python tools/standard_metric_closed_loop.py \
        --final-motion-dir evals/snapshots/closed_loop_final_motions_v1 \
        --split evals/splits/g2_real_stress_split_v2.json \
        --output evals/snapshots/standard_metric_closed_loop_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from tools.rl2_transition_g2_real_build import _resolve_g2
from tools.synthetic_injection import inject_foot_floating, inject_jitter

HOLDOUTS = ("g2_stress_holdout", "g2_natural_holdout", "clean_noharm_holdout", "synthetic_diag_holdout")
POLICIES = ("M0", "M1", "M2", "M3", "random_gate", "heuristic_gate", "dense_oracle_step1")
ARTIFACT_NAMES = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")


def _sid_safe(sid: str) -> str:
    return sid.replace("/", "_")


def _caption_g2(state_id, existing_pool, balanced_pool):
    pool_dir, stem = _resolve_g2(state_id, existing_pool, balanced_pool)
    meta_path = pool_dir / f"{stem}.json"
    if meta_path.exists():
        return json.load(open(meta_path, encoding="utf-8")).get("prompt", "")
    return ""


def _caption_hml3d(state_id, hml3d_root):
    tp = hml3d_root / "texts" / f"{state_id}.txt"
    if tp.exists():
        for line in open(tp, encoding="utf-8"):
            cap = line.split("#", 1)[0].strip()
            if cap:
                return cap
    return ""


def _original_motion(state_id, dist, args):
    if dist == "g2_natural":
        pool_dir, stem = _resolve_g2(state_id, args.existing_pool, args.balanced_pool)
        return np.load(str(pool_dir / f"{stem}.npy")).astype(np.float64)
    elif dist == "clean":
        return np.load(str(args.data_dir / f"{state_id}.npy")).astype(np.float64)
    elif dist == "synthetic_severe":
        clean = np.load(str(args.data_dir / f"{state_id}.npy")).astype(np.float64)
        m1 = inject_foot_floating(clean, lift_height=0.08, seed=args.synthetic_seed)
        return inject_jitter(m1, noise_std=0.05, seed=args.synthetic_seed + 1000)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-motion-dir", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "closed_loop_final_motions_v1")
    parser.add_argument("--split", type=Path,
                        default=REPO_ROOT / "evals" / "splits" / "g2_real_stress_split_v2.json")
    parser.add_argument("--g2-real", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "rl2_transition_g2_real_stage2_v1.json")
    parser.add_argument("--existing-pool", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--balanced-pool", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_balanced300_seed20260531")
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--gt-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    parser.add_argument("--n-gt", type=int, default=300)
    parser.add_argument("--gt-seed", type=int, default=44)
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "standard_metric_closed_loop_v1.json")
    args = parser.parse_args()

    # Setup tm2t.
    G._setup_motion_process_globals()
    mean = np.load(str(G.MEAN_PATH)); std = np.load(str(G.STD_PATH))
    move, motion_enc = G._load_encoders()
    text_enc = R._load_text_encoder()
    w_vectorizer = R.WordVectorizer(str(R.GLOVE_DIR), "our_vab")
    print("[INFO] tm2t encoders + word vectorizer loaded")

    # Artifact evaluators (motion-router) — repo import.
    from evaluators import DEFAULT_EVALUATORS
    art_evals = [ev for ev in DEFAULT_EVALUATORS if ev.name in ARTIFACT_NAMES]
    def _artifact_total(m):
        return float(np.mean([max((r.score for r in ev.evaluate(m)), default=0.0) for ev in art_evals]))

    split = json.load(open(args.split, encoding="utf-8"))
    g2_data = json.load(open(args.g2_real, encoding="utf-8"))
    state_map = g2_data["states"]

    # GT reference embeddings (stable FID reference).
    rng = np.random.default_rng(args.gt_seed)
    gt_files = sorted(args.gt_dir.glob("*.npy"))
    gt_chosen = [gt_files[i] for i in rng.choice(len(gt_files), args.n_gt, replace=False)]
    gt_feats = []
    for p in gt_chosen:
        f = G._to_features(np.load(str(p)))
        if f is not None: gt_feats.append(f)
    gt_emb = G._embed(gt_feats, mean, std, move, motion_enc)
    gt_mu, gt_cov = G._stats(gt_emb)
    print(f"[INFO] GT reference: n={len(gt_emb)}")

    results = {}
    for ho in HOLDOUTS:
        sids = [s for s in state_map if state_map[s]["split"] == ho]
        dist = state_map[sids[0]]["distribution"] if sids else None
        print(f"\n[INFO] {ho}: {len(sids)} states (dist={dist})")
        # Per state: original motion + caption + per-policy final motion.
        captions = []; groups = []
        feats = {"original": []}
        for pol in POLICIES:
            feats[pol] = []
        art_scores = {"original": []}
        for pol in POLICIES:
            art_scores[pol] = []
        valid_states = []
        for sid in sids:
            sinfo = state_map[sid]
            orig = _original_motion(sid, sinfo["distribution"], args)
            if orig is None:
                continue
            # caption.
            if sinfo["distribution"] == "g2_natural":
                cap = _caption_g2(sid, args.existing_pool, args.balanced_pool)
            else:
                cap = _caption_hml3d(sid, args.data_dir.parent)
            f_orig = G._to_features(orig)
            if f_orig is None or not cap:
                continue
            # per-policy final motions.
            pol_motions = {}
            ok = True
            for pol in POLICIES:
                fpath = args.final_motion_dir / f"{ho}__{pol}__{_sid_safe(sid)}.npy"
                if not fpath.exists():
                    ok = False; break
                pol_motions[pol] = np.load(str(fpath)).astype(np.float64)
            if not ok:
                continue
            pol_feats = {}
            allok = True
            for pol in POLICIES:
                pf = G._to_features(pol_motions[pol])
                if pf is None:
                    allok = False; break
                pol_feats[pol] = pf
            if not allok:
                continue
            # Commit this state.
            valid_states.append(sid)
            captions.append(cap); groups.append(sinfo.get("motion_group", "?"))
            feats["original"].append(f_orig); art_scores["original"].append(_artifact_total(orig))
            for pol in POLICIES:
                feats[pol].append(pol_feats[pol])
                art_scores[pol].append(_artifact_total(pol_motions[pol]))
        n = len(valid_states)
        print(f"   valid states: {n}")
        if n < 5:
            print(f"   [WARN] too few states for {ho}, skipping FID/R-Prec")
            results[ho] = {"n": n, "skipped": True}
            continue

        # Embed text once.
        text_embs = R._embed_text(captions, w_vectorizer, text_enc)
        # Per method: FID vs GT, R-Prec/MM-Dist, Diversity, artifact mean.
        method_metrics = {}
        for method in ["original"] + list(POLICIES):
            me = G._embed(feats[method], mean, std, move, motion_enc)
            mu, cov = G._stats(me)
            fid = G._fid(gt_mu, gt_cov, mu, cov)
            top123, mmdist = R._rprecision_mmdist(text_embs, me)
            div = G._diversity(me)
            method_metrics[method] = {
                "n": n, "FID_vs_GT": fid,
                "R_precision_top1": top123[0], "R_precision_top2": top123[1], "R_precision_top3": top123[2],
                "MM_Dist": mmdist, "Diversity": div,
                "artifact_total_mean": float(np.mean(art_scores[method])),
            }
        # Delta vs original.
        base = method_metrics["original"]
        for method in POLICIES:
            m = method_metrics[method]
            m["delta_FID"] = m["FID_vs_GT"] - base["FID_vs_GT"]
            m["delta_R1"] = m["R_precision_top1"] - base["R_precision_top1"]
            m["delta_MM_Dist"] = m["MM_Dist"] - base["MM_Dist"]
            m["delta_artifact"] = m["artifact_total_mean"] - base["artifact_total_mean"]

        # Group breakdown (G2 holdouts): artifact delta + FID per group (FID may be tiny-n noisy).
        group_breakdown = {}
        if ho.startswith("g2_"):
            by_g = defaultdict(list)
            for i, g in enumerate(groups):
                by_g[g].append(i)
            for g, idxs in by_g.items():
                if len(idxs) < 3:
                    continue
                grp = {"n": len(idxs)}
                for method in ["original"] + list(POLICIES):
                    a = float(np.mean([art_scores[method][i] for i in idxs]))
                    grp[f"{method}_artifact"] = a
                grp["M0_delta_artifact"] = grp["M0_artifact"] - grp["original_artifact"]
                grp["heuristic_delta_artifact"] = grp["heuristic_gate_artifact"] - grp["original_artifact"]
                group_breakdown[g] = grp

        results[ho] = {"n": n, "methods": method_metrics, "group_breakdown": group_breakdown}

    out = {
        "schema_version": "1.0.0", "record_type": "standard_metric_closed_loop",
        "task_id": "standard_metric_closed_loop_v1",
        "evaluator": "HumanML3D tm2t (movement+motion+text, finest.tar)",
        "metric_category": "A (standard, HumanML3D/Guo 2022 CVPR) + artifact (Category C proxy)",
        "directive": "closed-loop output 의 standard quality 가 original/noop 대비 보존/개선 되는가? Δ = vs original.",
        "gt_reference": f"HumanML3D clean new_joints (n={len(gt_emb)})",
        "policies": list(POLICIES),
        "caveat": "FID per holdout n=48-100 (작음) → noisy. Δ vs original (same state set) 가 주 신호. artifact = Category C.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Step 6 part 2 — Standard Metric on Closed-Loop Output (Δ vs original) ===")
    for ho in HOLDOUTS:
        r = results.get(ho, {})
        if r.get("skipped"):
            print(f"\n[{ho}] skipped (n={r.get('n')})"); continue
        print(f"\n[{ho}] (n={r['n']})")
        base = r["methods"]["original"]
        print(f"  original: FID={base['FID_vs_GT']:.3f} R@1={base['R_precision_top1']:.3f} MM-Dist={base['MM_Dist']:.3f} artifact={base['artifact_total_mean']:.4f}")
        print(f"  {'policy':<20} {'ΔFID':<10} {'ΔR@1':<10} {'ΔMM-Dist':<11} {'Δartifact':<12} {'FID_abs'}")
        for pol in POLICIES:
            m = r["methods"][pol]
            print(f"  {pol:<20} {m['delta_FID']:<+10.3f} {m['delta_R1']:<+10.3f} {m['delta_MM_Dist']:<+11.3f} {m['delta_artifact']:<+12.4f} {m['FID_vs_GT']:.3f}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
