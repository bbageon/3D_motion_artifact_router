"""AR-082: u*(s) 정책 지각 A/B pack — 원본 vs 정책 보정본 (사전등록 spec 준수).

표본 (사전 고정): holdout · 정책 u*≥0.5 적용분 · v1~v3 sample_id 제외 · (gen,sid) 중복
제거 후 **무작위 20** (seed 20260729 — worst-tail 아님, 정책의 실제 적용 분포 대표).
보정 강도 = 정책이 고른 u* (일괄 아님). blind: 좌우 무작위 (seed 20260730), 실속도.

CLI (motion3d env):
    python tools/ab_policy_pack_ar082.py --n-pairs 2   # preflight
    python tools/ab_policy_pack_ar082.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import RootGaitConsistencyTool
from tools.coords_protocol import estimate_ground, LEFT_FOOT
from tools.footskate_segment_gif_ar058_3f import _top_skate_segment
from tools.ab_preference_pack_ar058_3h import render_pair_gif

POOL_ROOT = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608"
CHOICES = REPO_ROOT / "evals" / "snapshots" / "strength_q_choices_ar081_v1.csv"
STATE = REPO_ROOT / "evals" / "snapshots" / "preaction_state_ar065_v1.csv"
PREV_KEYS = [
    REPO_ROOT / "reports" / "figures" / "2026-07-08" / "ar058_3h_ab_preference_pack" / "answer_key_RESEARCHER_ONLY.csv",
    REPO_ROOT / "reports" / "figures" / "2026-07-12" / "ar058_3i_ab_v2_pack" / "answer_key_RESEARCHER_ONLY.csv",
    REPO_ROOT / "reports" / "figures" / "2026-07-12" / "ar072_ab_v3_pack" / "answer_key_RESEARCHER_ONLY.csv",
]
OUT = REPO_ROOT / "reports" / "figures" / "2026-07-19" / "ar082_policy_ab_pack"
SNAP = REPO_ROOT / "evals" / "snapshots" / "ab_policy_pack_ar082_v1.json"

SAMPLE_SEED = 20260729
SIDE_SEED = 20260730
PLAYBACK_FPS = 20.0
U_MIN = 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    excl = set()
    for k in PREV_KEYS:
        excl |= {r["sample_id"] for r in csv.DictReader(open(k, encoding="utf-8"))}
    print(f"[INFO] v1~v3 제외 sid {len(excl)}")

    state = {(r["gen"], r["sid"], r["seed"]): r for r in csv.DictReader(open(STATE, encoding="utf-8"))}
    cand = [r for r in csv.DictReader(open(CHOICES, encoding="utf-8"))
            if float(r["u_star"]) >= U_MIN and r["sid"] not in excl]
    # (gen,sid) 중복 제거 — prompt 중복 방지 (셔플 후 첫 항목).
    rng = np.random.default_rng(SAMPLE_SEED)
    rng.shuffle(cand)
    seen, uniq = set(), []
    for r in cand:
        k = (r["gen"], r["sid"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    picks = uniq[: args.n_pairs]
    print(f"[INFO] 후보 {len(uniq)} → 표본 {len(picks)}")

    # pool meta 인덱스.
    meta_ix = {}
    for gen in ("mdm", "motiongpt", "momask"):
        for p in sorted((POOL_ROOT / gen).glob("*.json")):
            if p.name == "_pool_summary.json":
                continue
            m = json.load(open(p, encoding="utf-8"))
            meta_ix[(gen, m["sample_id"], str(m["seed"]))] = m

    tool = RootGaitConsistencyTool()
    side_rng = np.random.default_rng(SIDE_SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i, r in enumerate(picks, 1):
        key = (r["gen"], r["sid"], r["seed"])
        m = meta_ix[key]
        traj = np.load(REPO_ROOT / m["trajectory_npy"]).astype(np.float64)
        T = traj.shape[0]
        u = float(r["u_star"])
        corr, _ = tool.apply(traj, target_part="root", target_joints=[],
                             frame_range=(0, T - 1), strength="large",
                             metadata={"coord_space": "trajectory", "continuous_u": u})
        ground = estimate_ground(traj)
        seg = _top_skate_segment(traj, ground)
        foot, s, e = (seg[0], seg[1], seg[2]) if seg is not None else (LEFT_FOOT, 0, T - 1)
        left_is = "corrected" if side_rng.random() < 0.5 else "original"
        m_left, m_right = (corr, traj) if left_is == "corrected" else (traj, corr)
        render_pair_gif(m_left, m_right, foot, s, e, ground,
                        args.out_dir / f"pair_{i:02d}.gif", fps=PLAYBACK_FPS)
        st = state.get(key, {})
        pairs.append({"pair_id": f"pair_{i:02d}", "sample_id": r["sid"], "seed": r["seed"],
                      "gen": r["gen"], "u_star": u, "left_is": left_is,
                      "body_speed_shortfall": st.get("root_gait_mismatch", ""),
                      "realized_improvement_LABEL": r["realized_improvement"],
                      "prompt": m["prompt"]})
        print(f"[OK] pair_{i:02d} {r['gen']}/{r['sid']} u*={u} left={left_is}")

    with open(args.out_dir / "rater_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "which_more_natural_A_or_B", "confidence_0_3", "notes"])
        for p in pairs:
            w.writerow([p["pair_id"], "", "", ""])
    idx = ["# AR-082 Policy A/B (blind) — u*(s) 정책 보정본 vs 원본", "",
           "> **실제 속도(20fps).** 각 쌍에서 **전체적인 자연스러움** (발-지면 접촉 + 몸이 걸음에 맞게",
           "> 전진하는가) 을 보고 A/B 강제선택 + 확신도(0~3) → [rater_sheet.csv](rater_sheet.csv).",
           "> 정답은 시트에 없습니다. answer_key 는 판정 전 열람 금지.", "",
           "| Pair | GIF |", "|---|---|"]
    for p in pairs:
        idx.append(f"| {p['pair_id']} | [gif]({p['pair_id']}.gif) |")
    (args.out_dir / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    with open(args.out_dir / "answer_key_RESEARCHER_ONLY.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(pairs[0].keys()))
        for p in pairs:
            w.writerow(list(p.values()))

    from collections import Counter
    snap = {
        "schema_version": "1.0.0", "record_type": "ab_policy_pack_preregistration", "board_id": "AR-082",
        "status": "pack_built_awaiting_ratings",
        "treatment": "RootGaitConsistencyTool @ 정책 선택 u* (AR-081 policy_gen, λ*=0.5) — 일괄 u 아님",
        "selection_rule": f"holdout · u*>={U_MIN} · v1~v3 sid 제외 · (gen,sid) dedupe · random {args.n_pairs} (seed {SAMPLE_SEED})",
        "stimulus": {"playback_fps": PLAYBACK_FPS, "side_seed": SIDE_SEED,
                     "left_corrected_count": sum(1 for p in pairs if p["left_is"] == "corrected")},
        "composition": {"by_gen": dict(Counter(p["gen"] for p in pairs)),
                        "by_u": dict(Counter(str(p["u_star"]) for p in pairs))},
        "preregistered_criterion": {"support": ">=15/20 → '정책 작동' 유보 해제 (b1)",
                                    "fail": "<=12/20 → 정책 지각 계층 불성립 (1회 원칙)",
                                    "inconclusive": "13~14 → b2 검토"},
        "claim_scope": "u*>=0.5 적용분 한정 (적용 강도 질량 ~89%); STOP 정당성은 Cat-A(E10) 몫",
        "evidence_tier": "b1 예정",
    }
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    json.dump(snap, open(SNAP, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    import sys as _s; _s.stdout.reconfigure(encoding="utf-8")
    print(f"[DONE] {len(pairs)} pairs -> {args.out_dir} | gen {snap['composition']['by_gen']} u {snap['composition']['by_u']}")
    print(f"  snapshot: {SNAP}")


if __name__ == "__main__":
    main()
