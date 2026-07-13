"""AR-077 §6 (피드백): held-out benefit prediction — 진짜 routing gate 검증.

이전 AUC 는 label="MDM인가"(generator 분류) — benefit 예측 아님. 본 도구는:
  - benefit label (per motion) = **MM-Dist(corrected) < MM-Dist(original)** —
    이 motion 에 root correction 을 적용하면 text-motion 정렬이 좋아지는가.
    (gate 입력 foot_skate 와 독립 축 — 순환 아님.)
  - gate feature = **foot_skate (GT-free 관측)**, 보조 path_gain/induced_disp.
  - **전체 pool (3 gen, GT 필터 없음)** — 실배포 GT-free 조건 (locomotion subset
    선정에 GT 안 씀; feedback 1).
  - **prompt-split calibration/holdout** (leakage 차단): calibration 에서 foot_skate
    임계 보정(Youden J), holdout 에서 benefit-label AUC + precision/recall + no-harm.

per-motion MM = euclidean(text_emb, motion_emb) — batch 불요.

NOTE: mgpt env.
CLI:
    C:/Users/geonu/anaconda3/envs/mgpt/python.exe tools/routing_benefit_gate_ar077.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

import tools.standard_metric_fid as G
from tools.standard_metric_rprec import GLOVE_DIR, _embed_text, _load_text_encoder
from tools.representative_pool_measure import foot_skate_world
from tools.coords_protocol import PELVIS
from mGPT.data.humanml.utils.word_vectorizer import WordVectorizer  # noqa: E402

POOL_ROOT = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608"
CORR_ROOT = REPO_ROOT / "external_assets" / "ar077_root_corrected"
GENS = ("mdm", "motiongpt", "momask")


def _rpath(m):
    return float(np.linalg.norm(np.diff(m[:, PELVIS, :][:, [0, 2]], axis=0), axis=1).sum())


def _auc(score, label):
    score, label = np.asarray(score), np.asarray(label).astype(int)
    if label.sum() == 0 or label.sum() == len(label):
        return float("nan")
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label == 1].sum() - label.sum() * (label.sum() + 1) / 2) /
                 (label.sum() * (label == 0).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "evals" / "snapshots" / "routing_benefit_gate_ar077_v1.json")
    args = ap.parse_args()

    G._setup_motion_process_globals()
    mean = np.load(str(G.MEAN_PATH)); std = np.load(str(G.STD_PATH))
    move, motion_enc = G._load_encoders()
    text_enc = _load_text_encoder()
    wv = WordVectorizer(str(GLOVE_DIR), "our_vab")
    print("[INFO] encoders loaded")

    # ---- per-motion table (전체 pool, GT 필터 없음).
    recs = []
    for gen in GENS:
        pool = POOL_ROOT / gen; corr = CORR_ROOT / gen
        metas = [json.load(open(p, encoding="utf-8")) for p in sorted(pool.glob("*.json"))
                 if p.name != "_pool_summary.json"]
        fo_list, fc_list, cap_list, meta_list, sk_list, pg_list = [], [], [], [], [], []
        for m in metas:
            cp = corr / f"{m['sample_id']}__seed{m['seed']}.trajectory.npy"
            if not cp.exists():
                continue
            o = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
            c = np.load(cp).astype(np.float64)
            a, b = G._to_features(o), G._to_features(c)
            if a is None or b is None:
                continue
            fo_list.append(a); fc_list.append(b); cap_list.append(m["prompt"]); meta_list.append(m)
            sk_list.append(foot_skate_world(o)); pg_list.append(_rpath(c) / max(_rpath(o), 1e-6))
        eo = G._embed(fo_list, mean, std, move, motion_enc)
        ec = G._embed(fc_list, mean, std, move, motion_enc)
        et = _embed_text(cap_list, wv, text_enc)
        for i, m in enumerate(meta_list):
            mm_o = float(np.linalg.norm(et[i] - eo[i]))
            mm_c = float(np.linalg.norm(et[i] - ec[i]))
            recs.append({"gen": gen, "sid": m["sample_id"], "seed": m["seed"],
                         "foot_skate": sk_list[i], "path_gain": pg_list[i],
                         "mm_orig": mm_o, "mm_corr": mm_c, "benefit": int(mm_c < mm_o)})
        print(f"[INFO] {gen}: {sum(1 for r in recs if r['gen']==gen)} motions")

    skate = np.array([r["foot_skate"] for r in recs])
    dmm = np.array([r["mm_corr"] - r["mm_orig"] for r in recs])  # ΔMM (음수 = 개선)
    sids = np.array([r["sid"] for r in recs])   # 피드백 2: sample_id 단독 (generator 무관 — 같은 prompt 한쪽에)
    gens = np.array([r["gen"] for r in recs])
    N = len(recs)

    # ---- sample_id 단독 split **먼저** (δ 누수 차단 — 피드백 3차: split 후 calibration 에서만 δ 산출).
    rng = np.random.default_rng(args.seed)
    uniq = np.array(sorted(set(sids)))
    rng.shuffle(uniq)
    calib_sids = set(uniq[: len(uniq) // 2])
    is_calib = np.array([s in calib_sids for s in sids])

    # 피드백 3차: δ 는 **calibration VQ 에서만** 산출 (holdout 미사용 — benefit 경계 누수 차단).
    vq_mask = np.isin(gens, ["motiongpt", "momask"])
    delta = float(np.median(np.abs(dmm[vq_mask & is_calib])))  # calibration VQ |ΔMM| median
    def _classes(dv, dlt):
        return {"benefit": round(float((dv < -dlt).mean()), 3),
                "neutral": round(float((np.abs(dv) <= dlt).mean()), 3),
                "harm": round(float((dv > dlt).mean()), 3)}
    # δ-sensitivity 는 **참고 제시용** (고정 0.05/0.1 + calibration-δ). 전 pool 기준 (탐색).
    delta_sensitivity = {str(dlt): {g: _classes(dmm[gens == g], dlt) for g in GENS}
                         for dlt in (0.0, 0.05, 0.1, round(delta, 3))}

    # benefit label — calibration 에서 산출한 δ 로 3분류 후 benefit=1(개선).
    benefit = (dmm < -delta).astype(int)
    gen_benefit_calib_delta = {g: round(float(benefit[gens == g].mean()), 3) for g in GENS}

    # calibration: Youden J 최대화 임계 (skate > thr → apply 예측).
    cand = np.quantile(skate[is_calib], np.linspace(0.05, 0.95, 40))
    best_thr, best_j = cand[0], -1
    for t in cand:
        pred = skate[is_calib] > t
        tp = np.sum(pred & (benefit[is_calib] == 1)); fn = np.sum(~pred & (benefit[is_calib] == 1))
        tn = np.sum(~pred & (benefit[is_calib] == 0)); fp = np.sum(pred & (benefit[is_calib] == 0))
        tpr = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)
        if tpr - fpr > best_j:
            best_j, best_thr = tpr - fpr, float(t)

    # holdout 평가 (benefit label — feedback 3).
    ho = ~is_calib
    auc_ho = round(_auc(skate[ho], benefit[ho]), 3)
    pred_ho = skate[ho] > best_thr
    tp = int(np.sum(pred_ho & (benefit[ho] == 1))); fp = int(np.sum(pred_ho & (benefit[ho] == 0)))
    fn = int(np.sum(~pred_ho & (benefit[ho] == 1))); tn = int(np.sum(~pred_ho & (benefit[ho] == 0)))
    precision = round(tp / max(tp + fp, 1), 3); recall = round(tp / max(tp + fn, 1), 3)
    # 4차 피드백: false-apply 를 분해 — non-beneficial(=neutral+harm) vs **harmful(진짜 악화)만**.
    # no-harm 판단에는 harmful_apply_rate 가 더 중요 (selective prediction 원리:
    # Gangrade et al., AISTATS 2021 — 확신 낮으면 abstain, 정해진 적용률에서 harm 최소화).
    harm_lab = (dmm > delta).astype(int)           # ΔMM > +δ = 실제 품질 악화
    n_apply = max(tp + fp, 1)
    non_beneficial_apply = round(fp / n_apply, 3)  # 이득 기준 미충족 (neutral+harm)
    harmful_apply = round(float(np.sum(pred_ho & (harm_lab[ho] == 1))) / n_apply, 3)  # 진짜 악화만
    neutral_apply = round(non_beneficial_apply - harmful_apply, 3)
    false_stop = round(fn / max(fn + tn, 1), 3)

    # bootstrap holdout AUC CI (1000) — 피드백 3: **multiplicity 유지** (set 변환 금지).
    ho_sids = np.array(sorted(set(sids[ho])))
    rows_by_sid = {s: np.where((sids == s) & ho)[0] for s in ho_sids}
    boot = []
    for _ in range(1000):
        draw = rng.choice(ho_sids, len(ho_sids), replace=True)  # 중복 허용 (multiplicity 보존)
        idx = np.concatenate([rows_by_sid[s] for s in draw])   # 중복 prompt = 중복 rows
        if 0 < benefit[idx].sum() < len(idx):
            boot.append(_auc(skate[idx], benefit[idx]))
    auc_ci = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)] if boot else [None, None]

    out = {
        "schema_version": "1.1.0", "record_type": "routing_benefit_gate", "board_id": "AR-077",
        "fixes_2nd_feedback": ["benefit δ-3분류", "sample_id 단독 split(cross-gen leakage 차단)",
                               "bootstrap multiplicity 보존"],
        "n_motions": N, "pool": "전체 300 prompt × 3 seed × 3 gen (GT 필터 없음 — 실배포 GT-free 조건)",
        "benefit_label": f"ΔMM-Dist < -δ per motion. **δ={round(delta,4)} = calibration VQ |ΔMM| median** "
                         "(holdout 미사용 — 누수 차단, 피드백 3차). '부호만'(δ=0) 아님.",
        "delta_caveat": "δ 는 VQ calibration median 기반 → VQ neutral 비율이 구조적으로 ~50% 근처가 됨. "
                        "따라서 'VQ 절반이 실제 무효' / 'MDM 57%·VQ 20% benefit' / 'AUC 0.68' 은 **확정 사실 아니라 "
                        "δ 조건부 exploratory**. 정확한 진술: '내부 δ 조건에서 MDM ~57%·VQ ~20% benefit, pooled AUC ~0.68'.",
        "delta_sensitivity_benefit_neutral_harm_EXPLORATORY": delta_sensitivity,
        "gate_feature": "foot_skate (GT-free 관측)",
        "generator_benefit_rate_at_calib_delta": gen_benefit_calib_delta,
        "overall_benefit_rate_at_calib_delta": round(float(benefit.mean()), 3),
        "calibration": {"n": int(is_calib.sum()), "youden_threshold": round(best_thr, 5), "youden_J": round(best_j, 3)},
        "holdout": {"n": int(ho.sum()),
                    "benefit_auc": auc_ho, "benefit_auc_ci95_boot1000": auc_ci,
                    "precision": precision, "recall": recall,
                    "apply_decomposition_4th_feedback": {
                        "non_beneficial_apply_rate": non_beneficial_apply,
                        "harmful_apply_rate": harmful_apply,
                        "neutral_apply_rate": neutral_apply,
                        "note": "no-harm 판단엔 harmful_apply(진짜 악화)가 핵심 — non-beneficial 은 neutral 포함"},
                    "false_stop_rate": false_stop,
                    "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}},
        "interpretation": "benefit_auc 는 label='correction 이득'(generator 아님). non_beneficial_apply = APPLY 중 "
                          "최소 benefit(δ) 기준 미충족 (neutral+harm); harmful_apply = 그중 실제 악화(ΔMM>+δ)만. "
                          "selective prediction 원리(Gangrade AISTATS 2021): 전체 AUC 보다 정해진 적용률에서 "
                          "harmful-apply 가 낮은지가 no-harm 시스템의 기준. AUC/rate 모두 δ 조건부.",
        "claim_boundary": "held-out benefit-prediction gate. 완전 GT-free (pool 선정도 GT 무관). MM-Dist 를 "
                          "benefit proxy 로 사용 (지각 아님) — 지각 benefit 은 A/B 필요. MDM·단일 벤치마크. "
                          "동일 pool 재분석 = robustness evidence 이지 독립 snapshot 재현 아님.",
        "grounding": ["Guo HumanML3D CVPR2022 (MM-Dist)", "Gangrade et al. AISTATS 2021 (selective prediction)",
                      "Safe Orchestration no-harm (position §0)"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    # per-motion 레코드 dump — AR-078 gate 비교에서 재사용 (embedding 재계산 불요).
    import csv as _csv
    permotion_path = args.output.parent / "routing_benefit_permotion_ar077_v1.csv"
    with open(permotion_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["gen", "sid", "seed", "foot_skate", "path_gain_AFTER_ACTION",
                    "mm_orig", "mm_corr", "dmm", "split"])
        for i, r in enumerate(recs):
            w.writerow([r["gen"], r["sid"], r["seed"], round(r["foot_skate"], 6),
                        round(r["path_gain"], 4), round(r["mm_orig"], 4), round(r["mm_corr"], 4),
                        round(dmm[i], 4), "calib" if is_calib[i] else "holdout"])
    import sys as _s; _s.stdout.reconfigure(encoding="utf-8")
    print(f"δ(calib VQ median)={delta:.4f} | benefit@δ by gen: {gen_benefit_calib_delta} | overall {benefit.mean():.3f}")
    print(f"calib thr {best_thr:.5f} (J {best_j:.3f}) | HOLDOUT benefit AUC {auc_ho} ci{auc_ci} "
          f"prec {precision} recall {recall}")
    print(f"apply 분해: non-beneficial {non_beneficial_apply} = harmful {harmful_apply} + neutral {neutral_apply} | false_stop {false_stop}")
    print(f"[OK] wrote {args.output} + {permotion_path.name}")


if __name__ == "__main__":
    main()
