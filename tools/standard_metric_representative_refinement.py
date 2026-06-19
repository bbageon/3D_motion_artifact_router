"""AR-051: Category-A standard metrics for representative refinement variants.

Uses the HumanML3D tm2t evaluator to compare original generator outputs against
selected fixed refinement variants. Seed repetitions are averaged at the motion
embedding level so the statistical/evaluation unit remains the prompt.

This script does not modify the representative pool and does not save corrected
motions. It computes corrected motions on the fly.

CLI (mgpt env recommended):
    python tools/standard_metric_representative_refinement.py \
        --output evals/snapshots/standard_metric_representative_refinement_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import tools.standard_metric_fid as F
import tools.standard_metric_rprec as R
from correction_tools import BoneProjectionTool, FootLockTool, VelocitySmoothingTool
from tools.representative_refinement_effect import _apply_variant


GENERATORS = ("motiongpt", "mdm", "momask")
METHOD_VARIANTS = {
    "original": "noop",
    "foot_lock_large": "foot_lock_large",
    "velocity_smoothing_medium": "velocity_smoothing_medium",
    "bone_projection_all_large": "bone_projection_all_large",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.load(open(path, encoding="utf-8"))


def _average_by_prompt(embs: np.ndarray, prompt_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for sid, emb in zip(prompt_ids, embs):
        buckets[sid].append(emb)
    sids = sorted(buckets)
    avg = np.stack([np.mean(buckets[sid], axis=0) for sid in sids], axis=0)
    return avg, sids


def _metrics_for_method(text_embs: np.ndarray, motion_embs: np.ndarray, gt_mu, gt_cov) -> dict[str, float]:
    mu, cov = F._stats(motion_embs)
    top123, mmdist = R._rprecision_mmdist(text_embs, motion_embs, batch_size=32, n_rep=20)
    return {
        "FID_vs_GT": F._fid(gt_mu, gt_cov, mu, cov),
        "Diversity": F._diversity(motion_embs),
        "R_precision_top1": top123[0],
        "R_precision_top2": top123[1],
        "R_precision_top3": top123[2],
        "MM_Dist": mmdist,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", type=Path, default=REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608")
    ap.add_argument("--bank", type=Path, default=REPO_ROOT / "evals" / "prompts" / "protocol_rep_test_300_seed20260608.json")
    ap.add_argument("--gt-dir", type=Path, default=REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints")
    ap.add_argument("--n-gt", type=int, default=300)
    ap.add_argument("--gt-seed", type=int, default=20260619)
    ap.add_argument("--generators", nargs="+", default=list(GENERATORS))
    ap.add_argument("--limit-prompts", type=int, default=None, help="debug only")
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "standard_metric_representative_refinement_v1.json")
    args = ap.parse_args()

    F._setup_motion_process_globals()
    mean = np.load(str(F.MEAN_PATH)); std = np.load(str(F.STD_PATH))
    move, motion_enc = F._load_encoders()
    text_enc = R._load_text_encoder()
    w_vectorizer = R.WordVectorizer(str(R.GLOVE_DIR), "our_vab")
    print("[INFO] tm2t encoders + word vectorizer loaded")

    rng = np.random.default_rng(args.gt_seed)
    gt_files = sorted(args.gt_dir.glob("*.npy"))
    gt_chosen = [gt_files[i] for i in rng.choice(len(gt_files), args.n_gt, replace=False)]
    gt_feats = [f for p in gt_chosen if (f := F._to_features(np.load(str(p)))) is not None]
    gt_emb = F._embed(gt_feats, mean, std, move, motion_enc)
    gt_mu, gt_cov = F._stats(gt_emb)
    print(f"[INFO] GT reference embeddings: {len(gt_emb)}")

    bank = _load_json(args.bank)
    bank_rows = bank["rows"][: args.limit_prompts] if args.limit_prompts else bank["rows"]
    prompt_order = [r["sample_id"] for r in bank_rows]
    prompt_meta = {r["sample_id"]: r for r in bank_rows}

    tools = {
        "FootLockTool": FootLockTool(default_ground_y=0.0),
        "VelocitySmoothingTool": VelocitySmoothingTool(),
        "BoneProjectionTool": BoneProjectionTool(),
    }
    results: dict[str, Any] = {}

    for gen in args.generators:
        metas = [
            _load_json(p)
            for p in sorted((args.pool_root / gen).glob("*.json"))
            if not p.name.startswith("_") and _load_json(p).get("sample_id") in prompt_meta
        ]
        views = {
            "native_all_embed_valid": lambda m: True,
            "sensitivity_ratio_ge_0.6": lambda m: (gen != "motiongpt") or (m["actual_generated"] / max(m["target_length"], 1) >= 0.6),
        }
        results[gen] = {}
        print(f"\n[INFO] generator={gen}, seed rows={len(metas)}")
        for view_name, keep_fn in views.items():
            feats: dict[str, list[np.ndarray]] = {m: [] for m in METHOD_VARIANTS}
            feat_sids: dict[str, list[str]] = {m: [] for m in METHOD_VARIANTS}
            captions_by_sid = {sid: prompt_meta[sid]["prompt"] for sid in prompt_order}
            skipped_short = 0
            for meta in metas:
                if not keep_fn(meta):
                    continue
                sid = meta["sample_id"]
                traj = np.load(REPO_ROOT / meta["trajectory_npy"]).astype(np.float64)
                local = np.load(REPO_ROOT / meta["local_npy"]).astype(np.float64)
                method_positions = {}
                for method, variant in METHOD_VARIANTS.items():
                    if method == "original":
                        out_traj = traj
                    else:
                        out_traj, _, _ = _apply_variant(variant, traj, local, meta, tools)
                    feat = F._to_features(out_traj)
                    if feat is None:
                        method_positions = {}
                        skipped_short += 1
                        break
                    method_positions[method] = feat
                if not method_positions:
                    continue
                for method, feat in method_positions.items():
                    feats[method].append(feat)
                    feat_sids[method].append(sid)

            method_embs = {}
            common_sids: set[str] | None = None
            for method in METHOD_VARIANTS:
                emb_seed = F._embed(feats[method], mean, std, move, motion_enc)
                emb_prompt, sids = _average_by_prompt(emb_seed, feat_sids[method])
                method_embs[method] = {"emb": emb_prompt, "sids": sids}
                common_sids = set(sids) if common_sids is None else common_sids & set(sids)
            common = sorted(common_sids or [])
            if len(common) < 32:
                print(f"  [WARN] {view_name}: too few common prompts ({len(common)}), skipping")
                results[gen][view_name] = {"skipped": True, "n_prompts": len(common)}
                continue

            # Align prompt-averaged motion embeddings and text embeddings to the common prompt set.
            text_embs = R._embed_text([captions_by_sid[sid] for sid in common], w_vectorizer, text_enc)
            aligned = {}
            for method, payload in method_embs.items():
                idx = {sid: i for i, sid in enumerate(payload["sids"])}
                aligned[method] = np.stack([payload["emb"][idx[sid]] for sid in common], axis=0)

            method_metrics = {}
            for method, emb in aligned.items():
                method_metrics[method] = _metrics_for_method(text_embs, emb, gt_mu, gt_cov)
                method_metrics[method]["n_prompts"] = len(common)
            base = method_metrics["original"]
            for method in method_metrics:
                if method == "original":
                    continue
                m = method_metrics[method]
                m["delta_FID"] = m["FID_vs_GT"] - base["FID_vs_GT"]
                m["delta_R1"] = m["R_precision_top1"] - base["R_precision_top1"]
                m["delta_MM_Dist"] = m["MM_Dist"] - base["MM_Dist"]

            results[gen][view_name] = {
                "n_prompts": len(common),
                "skipped_seed_rows_too_short_or_invalid": skipped_short,
                "methods": method_metrics,
            }
            print(f"  [{view_name}] n_prompt={len(common)} skipped_seed_rows={skipped_short}")
            for method, mm in method_metrics.items():
                fid_delta = mm.get("delta_FID", 0.0)
                r1_delta = mm.get("delta_R1", 0.0)
                print(f"    {method:<28} FID={mm['FID_vs_GT']:.3f} dFID={fid_delta:+.3f} R1={mm['R_precision_top1']:.3f} dR1={r1_delta:+.3f} MM={mm['MM_Dist']:.3f}")

    out = {
        "schema_version": "1.0.0",
        "record_type": "standard_metric_representative_refinement",
        "board_id": "AR-051",
        "evaluator": "HumanML3D tm2t evaluator (movement/motion/text encoders, finest.tar)",
        "metric_category": "A standard metrics, prompt-level seed-averaged embeddings",
        "methods": METHOD_VARIANTS,
        "views": ["native_all_embed_valid", "sensitivity_ratio_ge_0.6"],
        "gt_reference": f"HumanML3D clean new_joints n={len(gt_emb)}",
        "claim_boundary": "Standard quality check for fixed refinement variants. Learned policy superiority not claimed; prompt-level seed averaging used because seed is repeated measure.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
