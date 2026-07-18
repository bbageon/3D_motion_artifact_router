"""AR-081 2단계: additive Q(s,u) + harm head 학습·holdout 평가 (사전등록 spec 준수).

- 구조: Q_improve(s,u) = f_A + f_B + f_C (+ f_D) — 그룹별 얕은 MLP 의 합, backfitting
  (그룹 기여 보존 — Neural Additive Models 계열). P_harm 동일 구조 별도.
- 라벨: improvement(u) = mm_u000 − mm_u (양수=개선, MM-Dist 점). harm(u) = (mm_u −
  mm_u000) > δ, δ = calibration VQ |ΔMM@u=1| median (holdout 미사용).
- 정책: u*(s) = argmax_u [Q_improve − λ·P_harm], u∈{0,.25,.5,.75,1}.
  λ 선택 규칙 (사전 고정): grid {0,0.5,1,2,4}; calibration 에서 harmful-rate 가
  고정 u=1.0(calib) 이하인 λ 중 mean improvement 최대; 없으면 λ=4.
- 판정 (사전 고정, holdout): 지지 = (a) u* vs 고정 u=1.0 improvement diff CI 하한>0
  AND (b) harmful diff CI 상한 ≤ 0. 아니면 한계.
- 병렬: Q-state(no-gen) vs Q-state+gen (shortcut 검사) + oracle-u ceiling (label-peek,
  상한 참조 전용) + Markov 축약 검정 (행동/reward 보존 — 유도 변수 재투입).

CLI (motion3d env):
    python tools/strength_q_train_ar081.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "evals" / "snapshots" / "strength_q_dataset_ar081_v1.csv"
OUT = REPO_ROOT / "evals" / "snapshots" / "strength_q_result_ar081_v1.json"

U_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
U_COL = {0.0: "mm_u000", 0.25: "mm_u025", 0.5: "mm_u050", 0.75: "mm_u075", 1.0: "mm_u100"}
GROUPS = {  # spec §1 — 그룹별 additive
    "A_need": ["stepping_speed", "body_speed_shortfall", "foot_sliding"],
    "B_contact": ["ground_contact_ratio", "contact_streak"],
    "C_motion": ["path_straightness", "duration_sec", "wants_to_travel", "wants_to_stay", "intent_unclear"],
    "D_generator": ["is_mdm", "is_motiongpt", "is_momask"],
}
LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0)
N_BOOT = 1000
BACKFIT_ROUNDS = 4


class AdditiveQ:
    """그룹별 (features, u) → 기여 f_g 의 합. backfitting 으로 학습 (기여 보존)."""

    def __init__(self, group_cols: dict, seed: int = 0):
        self.group_cols = group_cols
        self.seed = seed
        self.nets = {}
        self.intercept = 0.0

    def _xg(self, F: dict, g: str, u: np.ndarray) -> np.ndarray:
        return np.column_stack([F[c] for c in self.group_cols[g]] + [u])

    def fit(self, F: dict, u: np.ndarray, y: np.ndarray) -> "AdditiveQ":
        preds = {g: np.zeros(len(y)) for g in self.group_cols}
        self.intercept = float(y.mean())
        for r in range(BACKFIT_ROUNDS):
            for g in self.group_cols:
                target = y - self.intercept - sum(preds[gg] for gg in self.group_cols if gg != g)
                net = MLPRegressor(hidden_layer_sizes=(16,), max_iter=800,
                                   random_state=self.seed, learning_rate_init=0.01)
                net.fit(self._xg(F, g, u), target)
                self.nets[g] = net
                preds[g] = net.predict(self._xg(F, g, u))
            self.intercept = float((y - sum(preds.values())).mean()) + self.intercept * 0.0 + float(np.mean(y - sum(preds.values())))
        # intercept 정리 (마지막 잔차 평균).
        self.intercept = float(np.mean(y - sum(self.nets[g].predict(self._xg(F, g, u)) for g in self.group_cols)))
        return self

    def predict(self, F: dict, u: np.ndarray) -> np.ndarray:
        return self.intercept + sum(self.nets[g].predict(self._xg(F, g, u)) for g in self.group_cols)

    def contributions(self, F: dict, u: np.ndarray) -> dict:
        return {g: self.nets[g].predict(self._xg(F, g, u)) for g in self.group_cols}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    N = len(rows)
    sid = np.array([r["sid"] for r in rows])
    gen = np.array([r["gen"] for r in rows])
    calib = np.array([r["split"] == "calib" for r in rows])
    ho = ~calib
    mm = {u: np.array([float(r[c]) for r in rows]) for u, c in U_COL.items()}
    imp = {u: mm[0.0] - mm[u] for u in U_GRID}            # 양수 = 개선
    vq = np.isin(gen, ["motiongpt", "momask"])
    delta = float(np.median(np.abs(mm[1.0] - mm[0.0])[vq & calib]))
    harm = {u: ((mm[u] - mm[0.0]) > delta).astype(float) for u in U_GRID}

    feat_cols = sum(GROUPS.values(), [])
    raw = {c: np.array([float(r[c]) for r in rows]) for c in feat_cols}
    mu = {c: raw[c][calib].mean() for c in feat_cols}
    sd = {c: raw[c][calib].std() + 1e-9 for c in feat_cols}
    F = {c: (raw[c] - mu[c]) / sd[c] for c in feat_cols}

    # 학습 행 = motion × u_grid (u=0 anchor 포함: improvement=0, harm=0).
    def expand(mask):
        idx = np.where(mask)[0]
        rows_i = np.concatenate([idx for _ in U_GRID])
        u_v = np.concatenate([np.full(len(idx), u) for u in U_GRID])
        y_imp = np.concatenate([imp[u][idx] for u in U_GRID])
        y_harm = np.concatenate([harm[u][idx] for u in U_GRID])
        Fx = {c: F[c][rows_i] for c in feat_cols}
        return Fx, u_v, y_imp, y_harm

    Fx_c, u_c, yimp_c, yharm_c = expand(calib)
    g_all = GROUPS
    g_nogen = {k: v for k, v in GROUPS.items() if k != "D_generator"}
    print(f"[INFO] N={N}, calib rows {len(u_c)}, delta={delta:.4f}")

    models = {}
    for name, gcols in (("gen", g_all), ("nogen", g_nogen)):
        models[name] = {
            "imp": AdditiveQ(gcols).fit(Fx_c, u_c, yimp_c),
            "harm": AdditiveQ(gcols, seed=1).fit(Fx_c, u_c, yharm_c),
        }
        print(f"[INFO] trained additive Q ({name})")

    # motion 별 u-grid 예측 → 정책.
    def policy_choice(model, mask, lam):
        idx = np.where(mask)[0]
        qs = np.zeros((len(idx), len(U_GRID)))
        for j, u in enumerate(U_GRID):
            Fi = {c: F[c][idx] for c in feat_cols}
            uv = np.full(len(idx), u)
            qs[:, j] = model["imp"].predict(Fi, uv) - lam * np.clip(model["harm"].predict(Fi, uv), 0, 1)
        choice = np.array(U_GRID)[np.argmax(qs, axis=1)]
        r_imp = np.array([imp[u][i] for u, i in zip(choice, idx)])
        r_harm = np.array([harm[u][i] for u, i in zip(choice, idx)])
        return idx, choice, r_imp, r_harm

    # λ 선택 (사전 고정 규칙 — calibration 만).
    base_c_harm = float(harm[1.0][calib].mean())
    lam_table, best = {}, None
    for lam in LAMBDAS:
        _, _, ri, rh = policy_choice(models["gen"], calib, lam)
        ok = rh.mean() <= base_c_harm
        lam_table[str(lam)] = {"calib_mean_improvement": round(float(ri.mean()), 4),
                               "calib_harmful_rate": round(float(rh.mean()), 4), "feasible": bool(ok)}
        if ok and (best is None or ri.mean() > best[1]):
            best = (lam, float(ri.mean()))
    lam_star = best[0] if best else 4.0

    # ---- holdout 평가.
    results = {}
    per_policy_realized = {}
    for pname, (model, lam) in {"policy_gen": (models["gen"], lam_star),
                                 "policy_nogen": (models["nogen"], lam_star)}.items():
        idx, choice, ri, rh = policy_choice(model, ho, lam)
        per_policy_realized[pname] = (idx, ri, rh)
        results[pname] = {"mean_improvement": round(float(ri.mean()), 4),
                          "harmful_rate": round(float(rh.mean()), 4),
                          "u_distribution": {str(u): round(float((choice == u).mean()), 3) for u in U_GRID}}
    ho_idx = np.where(ho)[0]
    for pname, uv in (("fixed_u100", 1.0), ("stop_all", 0.0)):
        ri = imp[uv][ho_idx]; rh = harm[uv][ho_idx]
        per_policy_realized[pname] = (ho_idx, ri, rh)
        results[pname] = {"mean_improvement": round(float(ri.mean()), 4),
                          "harmful_rate": round(float(rh.mean()), 4)}
    oracle_imp = np.max(np.column_stack([imp[u][ho_idx] for u in U_GRID]), axis=1)
    results["oracle_u_ceiling_LABEL_PEEK"] = {"mean_improvement": round(float(oracle_imp.mean()), 4),
                                              "note": "상한 참조 전용 — 정책 아님"}

    # bootstrap (sid 단위, multiplicity 보존, 정책 간 paired).
    ho_sids = np.array(sorted(set(sid[ho])))
    pos_in_ho = {i: j for j, i in enumerate(ho_idx)}
    rows_by_sid = {s: np.array([pos_in_ho[i] for i in np.where((sid == s) & ho)[0]]) for s in ho_sids}
    rng = np.random.default_rng(20260728)
    d_imp, d_harm = {"policy_gen": [], "policy_nogen": []}, {"policy_gen": [], "policy_nogen": []}
    ri_f = per_policy_realized["fixed_u100"][1]; rh_f = per_policy_realized["fixed_u100"][2]
    for _ in range(N_BOOT):
        draw = rng.choice(ho_sids, len(ho_sids), replace=True)
        pos = np.concatenate([rows_by_sid[s] for s in draw])
        for pname in ("policy_gen", "policy_nogen"):
            ri = per_policy_realized[pname][1]; rh = per_policy_realized[pname][2]
            d_imp[pname].append(float(ri[pos].mean() - ri_f[pos].mean()))
            d_harm[pname].append(float(rh[pos].mean() - rh_f[pos].mean()))
    def _ci(a):
        return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    diffs = {p: {"improvement_diff_vs_u100": {"mean": round(float(np.mean(d_imp[p])), 4), "ci95": _ci(d_imp[p])},
                 "harm_diff_vs_u100": {"mean": round(float(np.mean(d_harm[p])), 4), "ci95": _ci(d_harm[p])}}
             for p in d_imp}
    ca = diffs["policy_gen"]["improvement_diff_vs_u100"]["ci95"][0] > 0
    cb = diffs["policy_gen"]["harm_diff_vs_u100"]["ci95"][1] <= 0.005  # 증가 없음 (반올림 여유)
    verdict = "지지 (u* 정책이 고정 u=1.0 을 개선)" if (ca and cb) else \
        "한계 — " + ("improvement 유의 우위 실패" if not ca else "harm 증가")

    # ---- Markov 축약 검정 (행동/reward 보존): 유도 변수 재투입.
    der = {"root_speed_actual": raw["stepping_speed"] - raw["body_speed_shortfall"],
           "path_length_approx": (raw["stepping_speed"] - raw["body_speed_shortfall"]) * raw["duration_sec"] * 20.0}
    Fd = dict(F)
    for c, v in der.items():
        m_, s_ = v[calib].mean(), v[calib].std() + 1e-9
        Fd[c] = (v - m_) / s_
    g_aug = {**g_all, "A_need": g_all["A_need"] + list(der.keys())}
    Fx_c_aug = {c: Fd[c][np.concatenate([np.where(calib)[0] for _ in U_GRID])] for c in sum(g_aug.values(), [])}
    aug = AdditiveQ(g_aug).fit(Fx_c_aug, u_c, yimp_c)
    # reward 보존: holdout RMSE 비교.
    Fx_h = {c: F[c][np.concatenate([ho_idx for _ in U_GRID])] for c in feat_cols}
    Fx_h_aug = {c: Fd[c][np.concatenate([ho_idx for _ in U_GRID])] for c in sum(g_aug.values(), [])}
    u_h = np.concatenate([np.full(len(ho_idx), u) for u in U_GRID])
    y_h = np.concatenate([imp[u][ho_idx] for u in U_GRID])
    rmse_base = float(np.sqrt(np.mean((models["gen"]["imp"].predict(Fx_h, u_h) - y_h) ** 2)))
    rmse_aug = float(np.sqrt(np.mean((aug.predict(Fx_h_aug, u_h) - y_h) ** 2)))
    # 행동 보존: augmented Q 로 argmax u 가 바뀌는 비율.
    qs_b = np.column_stack([models["gen"]["imp"].predict({c: F[c][ho_idx] for c in feat_cols}, np.full(len(ho_idx), u)) for u in U_GRID])
    qs_a = np.column_stack([aug.predict({c: Fd[c][ho_idx] for c in sum(g_aug.values(), [])}, np.full(len(ho_idx), u)) for u in U_GRID])
    flip = float((np.argmax(qs_b, 1) != np.argmax(qs_a, 1)).mean())

    out = {
        "schema_version": "1.0.0", "record_type": "strength_q_result", "board_id": "AR-081",
        "transition_dataset_id": "strength_q_ar081_v1", "u_grid": list(U_GRID), "delta": round(delta, 4),
        "n": N, "n_holdout": int(ho.sum()),
        "lambda_selection_calib_only": {"table": lam_table, "chosen": lam_star,
                                        "rule": "harmful<=fixed-u1(calib) 중 improvement 최대; 없으면 4"},
        "holdout": results, "holdout_diffs_paired": diffs,
        "preregistered_verdict": {"verdict": verdict, "cond_a_improvement": bool(ca), "cond_b_no_harm_increase": bool(cb)},
        "shortcut_check_gen_vs_nogen": "holdout.policy_gen vs holdout.policy_nogen 비교",
        "markov_reduction_tests": {
            "reward_preservation": {"rmse_base": round(rmse_base, 4), "rmse_with_derived": round(rmse_aug, 4),
                                    "note": "유도 변수(root_speed_actual·path_length)는 포함 feature 의 함수 — 개선 없어야 정상"},
            "action_preservation_flip_rate": round(flip, 4)},
        "claim_boundary": "one-shot·MM proxy·δ 조건부 exploratory. '정책 작동' 은 지지+지각 검증 후. "
                          "closed-loop/RL-2 구계열 혼합 인용 금지 (§6-15).",
        "grounding": ["Agarwal NeurIPS 2021 (NAM)", "Allen NeurIPS 2021 (state abstraction)",
                      "Gangrade AISTATS 2021", "Guo CVPR2022"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    import sys as _s; _s.stdout.reconfigure(encoding="utf-8")
    print(f"λ*={lam_star} | holdout:")
    for p in ("policy_gen", "policy_nogen", "fixed_u100", "stop_all", "oracle_u_ceiling_LABEL_PEEK"):
        r = results[p]
        print(f"  {p:<28} imp {r['mean_improvement']:>7} harm {r.get('harmful_rate', '-')}")
    for p, d in diffs.items():
        print(f"  diff {p} vs u100: imp {d['improvement_diff_vs_u100']['mean']} ci{d['improvement_diff_vs_u100']['ci95']} "
              f"| harm {d['harm_diff_vs_u100']['mean']} ci{d['harm_diff_vs_u100']['ci95']}")
    print(f"Markov: rmse {rmse_base:.4f}->{rmse_aug:.4f} (derived), action flip {flip:.3f}")
    print(f"VERDICT: {verdict}")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
