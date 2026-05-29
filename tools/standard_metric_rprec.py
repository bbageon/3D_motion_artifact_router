"""Step G-2 (사용자 directive 2026-05-28): R-Precision + MM-Dist (text-motion co-embedding).

사용자 directive:
> "G-2: R-Precision / MM-Dist 먼저. correction이 prompt semantics를 깨는가? (R-Precision)
>  motion-text embedding distance가 악화되는가? (MM-Dist). safe oracle이 noop 수준으로
>  semantic preservation을 유지하는가?"

본 도구는 HumanML3D tm2t text_encoder + motion_encoder 의 co-embedding 으로 R-Precision
(top-1/2/3 retrieval) + MM-Dist (matched motion-text distance) 측정.

text 는 method 무관 동일 (caption per sample) — motion 만 method 별 다름. 따라서 spacy
tokenization 의 absolute 값 차이 가 있어도 method 간 비교 valid.

NOTE: mgpt env 필요 (motion_process + tm2t_evaluator + spacy en_core_web_sm + GloVe).

CLI (mgpt env):
    python tools/standard_metric_rprec.py --output evals/snapshots/standard_metric_rprec_v1.json
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

import tools.standard_metric_fid as G  # reuse motion pipeline
from mGPT.archs.tm2t_evaluator import TextEncoderBiGRUCo
from mGPT.data.humanml.utils.word_vectorizer import WordVectorizer, POS_enumerator

GLOVE_DIR = MGPT / "deps" / "glove"
MAX_TEXT_LEN = 20
UNIT_LEN = 4

import spacy
_NLP = spacy.load("en_core_web_sm")


def _process_text(sentence: str):
    """HumanML3D recipe: NOUN/VERB lemmatized (except 'left'), others surface. → word/POS tokens."""
    sentence = sentence.replace("-", "")
    doc = _NLP(sentence.lower())
    word_list, pos_list = [], []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == "NOUN" or token.pos_ == "VERB") and (word != "left"):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list


def _tokens_to_embeddings(caption: str, w_vectorizer: WordVectorizer):
    """caption → word_embeddings (L, 300) + pos_one_hots (L, 15) + sent_len."""
    word_list, pos_list = _process_text(caption)
    tokens = [f"{w}/{p}" for w, p in zip(word_list, pos_list)]
    if len(tokens) < MAX_TEXT_LEN:
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (MAX_TEXT_LEN + 2 - sent_len)
    else:
        tokens = tokens[:MAX_TEXT_LEN]
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
    word_embs, pos_ohots = [], []
    for tok in tokens:
        we, pe = w_vectorizer[tok]
        word_embs.append(we[None, :])
        pos_ohots.append(pe[None, :])
    return (np.concatenate(word_embs, axis=0).astype(np.float32),
            np.concatenate(pos_ohots, axis=0).astype(np.float32), sent_len)


def _load_text_encoder():
    dim_word = 300
    dim_pos = len(POS_enumerator)  # 15
    te = TextEncoderBiGRUCo(dim_word, dim_pos, 512, 512)
    ck = torch.load(str(G.CHECKPOINT), map_location="cpu")
    te.load_state_dict(ck["text_encoder"])
    te.eval()
    return te


def _embed_text(captions, w_vectorizer, text_enc):
    embs = []
    with torch.no_grad():
        for cap in captions:
            we, pe, sl = _tokens_to_embeddings(cap, w_vectorizer)
            we_t = torch.from_numpy(we).unsqueeze(0)
            pe_t = torch.from_numpy(pe).unsqueeze(0)
            # TextEncoderBiGRUCo.forward(word_embs, pos_onehot, cap_lens). sorted by length.
            emb = text_enc(we_t, pe_t, torch.tensor([sl]))
            embs.append(emb.squeeze(0).numpy())
    return np.array(embs)


def _euclidean_matrix(a, b):
    # a (N, d), b (N, d) → (N, N) pairwise euclidean.
    a2 = (a ** 2).sum(axis=1)[:, None]
    b2 = (b ** 2).sum(axis=1)[None, :]
    d2 = a2 + b2 - 2 * a.dot(b.T)
    return np.sqrt(np.maximum(d2, 0))


def _rprecision_mmdist(text_embs, motion_embs, batch_size=32, seed=0, n_rep=20):
    """Standard HumanML3D R-Precision (top-1/2/3) + MM-Dist over random 32-batches."""
    rng = np.random.default_rng(seed)
    n = len(text_embs)
    top123 = np.zeros(3)
    mm = 0.0
    count = 0
    for _ in range(n_rep):
        idx = rng.permutation(n)
        for s in range(0, n - batch_size + 1, batch_size):
            bidx = idx[s:s + batch_size]
            te = text_embs[bidx]; me = motion_embs[bidx]
            dmat = _euclidean_matrix(te, me)  # row i = text i vs all motions
            # MM-Dist = mean of matched (diagonal) distance.
            mm += np.mean(np.diag(dmat))
            # R-Precision: for each text, rank motions; correct = diagonal index.
            ranks = np.argsort(dmat, axis=1)
            for k in range(3):
                top123[k] += np.mean([(bi in ranks[i, :k+1]) for i, bi in enumerate(range(batch_size))])
            count += 1
    return (top123 / count).tolist(), float(mm / count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--oracle-3level", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_g2_hml3d_test300_3level_v1.json")
    parser.add_argument("--oracle-5level", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_g2_hml3d_test300_v1.json")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "standard_metric_rprec_v1.json")
    args = parser.parse_args()

    G._setup_motion_process_globals()
    mean = np.load(str(G.MEAN_PATH)); std = np.load(str(G.STD_PATH))
    move, motion_enc = G._load_encoders()
    text_enc = _load_text_encoder()
    w_vectorizer = WordVectorizer(str(GLOVE_DIR), "our_vab")
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

    o3 = {p["trial_id"]: p for p in json.load(open(args.oracle_3level, encoding="utf-8"))["per_sample"]}
    o5 = {p["trial_id"]: p for p in json.load(open(args.oracle_5level, encoding="utf-8"))["per_sample"]}

    g2_files = sorted(args.g2_batch_dir.glob("motion_*.npy"))
    method_keys = ["noop", "b2_netgain_best", "b2_artifact_best", "safe_oracle_3level", "safe_oracle_5level"]
    feats = {k: [] for k in method_keys}
    cap_common = []
    _ALPHA = 5.0
    def _target_g2(mot):
        return float(np.mean([max((r.score for r in ev.evaluate(mot)), default=0.0)
                              for ev in DEFAULT_EVALUATORS if ev.name in ("FootFloatingEvaluator","BoneLengthEvaluator","VelocityJitterEvaluator")]))
    def _netgain_b2(corr, orig):
        return -(_target_g2(corr) - _target_g2(orig)) - _ALPHA * float(np.mean(np.linalg.norm(corr-orig, axis=-1)))
    print(f"[INFO] processing {len(g2_files)} G2 samples...")
    for i, p in enumerate(g2_files, 1):
        tid = p.stem
        meta = json.load(open(p.with_suffix(".json"), encoding="utf-8"))
        cap = meta.get("prompt", "")
        m = np.load(str(p)).astype(np.float64)
        fn = G._to_features(m)
        if fn is None:
            continue
        # 두 B2 variant: netgain-best (F-3) + artifact-best (F-5).
        ng_best_m, ng_best = m, 0.0
        art_best_m, art_best = m, None
        for st in ("small", "medium", "large"):
            out, _ = tools["VelocitySmoothingTool"].apply(m, target_part="full_body", target_joints=[], frame_range=(0, m.shape[0]-1), strength=st)
            ng = _netgain_b2(out, m)
            if ng > ng_best: ng_best, ng_best_m = ng, out
            art = sum(max((r.score for r in ev.evaluate(out)), default=0.0) for ev in DEFAULT_EVALUATORS)
            if art_best is None or art < art_best: art_best, art_best_m = art, out
        f_ng = G._to_features(ng_best_m); f_art = G._to_features(art_best_m)
        sb3 = o3.get(tid, {}).get("safe_best"); sb5 = o5.get(tid, {}).get("safe_best")
        m3 = _apply_seq(m, sb3["sequence"]) if (sb3 and sb3["length"] > 0) else m
        m5 = _apply_seq(m, sb5["sequence"]) if (sb5 and sb5["length"] > 0) else m
        f3 = G._to_features(m3); f5 = G._to_features(m5)
        if f_ng is None or f_art is None or f3 is None or f5 is None:
            continue
        cap_common.append(cap)
        feats["noop"].append(fn); feats["b2_netgain_best"].append(f_ng); feats["b2_artifact_best"].append(f_art)
        feats["safe_oracle_3level"].append(f3); feats["safe_oracle_5level"].append(f5)
        if i % 50 == 0:
            print(f"   {i}/{len(g2_files)}")

    print(f"[INFO] {len(cap_common)} common samples. Embedding text + motion...")
    text_embs = _embed_text(cap_common, w_vectorizer, text_enc)
    results = {}
    for method, flist in feats.items():
        me = G._embed(flist, mean, std, move, motion_enc)
        top123, mmdist = _rprecision_mmdist(text_embs, me)
        results[method] = {"n": len(flist), "R_precision_top1": top123[0],
                           "R_precision_top2": top123[1], "R_precision_top3": top123[2],
                           "MM_Dist": mmdist}

    out = {
        "schema_version": "1.0.0", "record_type": "standard_metric_rprec",
        "task_id": "standard_metric_rprec_v1",
        "evaluator": "HumanML3D tm2t (text+motion co-embedding, finest.tar)",
        "metric_category": "A (standard, HumanML3D/Guo 2022 CVPR)",
        "tokenization": "spacy en_core_web_sm (HumanML3D recipe, NOUN/VERB lemmatized)",
        "note": "text 동일 per sample (method 무관), motion 만 method별. method 간 비교 valid.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Step G-2: R-Precision + MM-Dist ===")
    print(f"  {'method':<22} {'R@1':<8} {'R@2':<8} {'R@3':<8} {'MM-Dist'}")
    for k, r in results.items():
        print(f"  {k:<22} {r['R_precision_top1']:.4f}  {r['R_precision_top2']:.4f}  {r['R_precision_top3']:.4f}  {r['MM_Dist']:.4f}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
