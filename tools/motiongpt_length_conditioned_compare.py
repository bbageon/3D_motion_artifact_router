"""AR-050 Part C: MotionGPT direct vs length-conditioned generation 비교.

고정 audit prompt(직접 경로 실패 사례 + 무작위 표본) × 동일 seed × {direct, conditioned} 로
MotionGPT 서비스(with_len 플래그)를 호출해 target-length match / 실제 길이 / degenerate(<=8frame) /
runtime 을 비교한다. 길이를 padding/interpolation 으로 맞추지 않음 — 순수 generation 결과.

별도 protocol(AR-050 Part C). representative pool 미변경.

CLI (motion3d env, motiongpt 서비스 재빌드 후):
    python tools/motiongpt_length_conditioned_compare.py \
        --bank evals/prompts/protocol_rep_test_300_seed20260608.json \
        --audit evals/snapshots/motiongpt_length_audit_v1.json \
        --n-random 15 --output evals/snapshots/motiongpt_length_conditioned_compare_v1.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
URL = "http://localhost:8003/generate"
SEEDS = [20260609, 20260610, 20260611]


def _gen(prompt, n_frames, seed, with_len):
    r = requests.post(URL, json={"prompt": prompt, "n_frames": int(n_frames), "seed": int(seed),
                                 "with_len": with_len}, timeout=600)
    r.raise_for_status()
    d = r.json()
    return {"actual": int(d["length_generated"]), "mode": d.get("generation_mode"),
            "elapsed": d.get("elapsed_sec")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=REPO_ROOT / "evals" / "prompts" / "protocol_rep_test_300_seed20260608.json")
    ap.add_argument("--audit", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "motiongpt_length_audit_v1.json")
    ap.add_argument("--n-random", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260609)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "evals" / "snapshots" / "motiongpt_length_conditioned_compare_v1.json")
    args = ap.parse_args()

    bank = {r["sample_id"]: r for r in json.load(open(args.bank, encoding="utf-8"))["rows"]}
    audit = json.load(open(args.audit, encoding="utf-8"))
    # audit prompts = high length-variance failures + extreme + random representative.
    fail_ids = [h["sample_id"] for h in audit["per_prompt_high_length_variance"][:5]]
    extreme_ids = [e["sample_id"] for e in audit["extreme_early_termination"]]
    rng = random.Random(args.seed)
    rest = [s for s in bank if s not in set(fail_ids + extreme_ids)]
    rand_ids = rng.sample(rest, args.n_random)
    audit_ids = list(dict.fromkeys(fail_ids + extreme_ids + rand_ids))
    print(f"[INFO] audit prompts: {len(audit_ids)} (fail {len(fail_ids)} + extreme {len(extreme_ids)} + random {len(rand_ids)})")

    rows = []
    for sid in audit_ids:
        r = bank[sid]
        tgt = r["target_length"]
        for seed in SEEDS:
            for with_len in (False, True):
                g = _gen(r["prompt"], tgt, seed, with_len)
                rows.append({"sample_id": sid, "target": tgt, "seed": seed,
                             "with_len": with_len, "actual": g["actual"], "elapsed": g["elapsed"]})
        print(f"  {sid} done (target={tgt})", flush=True)

    def agg(wl):
        sub = [x for x in rows if x["with_len"] == wl]
        n = len(sub)
        ratios = np.array([x["actual"] / x["target"] for x in sub])
        return {
            "n": n,
            "exact_match_rate": round(sum(1 for x in sub if x["actual"] == x["target"]) / n, 3),
            "ratio_in_0.8_1.25": round(sum(1 for x in sub if 0.8 <= (x["actual"] / x["target"]) <= 1.25) / n, 3),
            "degenerate_le8_rate": round(sum(1 for x in sub if x["actual"] <= 8) / n, 3),
            "ratio_median": round(float(np.median(ratios)), 3),
            "ratio_min": round(float(ratios.min()), 3),
            "actual_min": min(x["actual"] for x in sub),
            "mean_runtime_sec": round(float(np.mean([x["elapsed"] for x in sub if x["elapsed"] is not None])), 2),
        }

    direct, cond = agg(False), agg(True)
    # per-prompt: 직접 실패 prompt 가 conditioned 로 개선되나.
    per_prompt = {}
    for sid in fail_ids + extreme_ids:
        dl = sorted(x["actual"] for x in rows if x["sample_id"] == sid and not x["with_len"])
        cl = sorted(x["actual"] for x in rows if x["sample_id"] == sid and x["with_len"])
        per_prompt[sid] = {"target": bank[sid]["target_length"], "direct_lengths": dl, "conditioned_lengths": cl}

    out = {
        "schema_version": "1.0.0", "record_type": "motiongpt_length_conditioned_compare", "board_id": "AR-050",
        "policy": "별도 protocol(Part C). representative pool 미변경. padding/interpolation 미사용 (순수 generation).",
        "n_audit_prompts": len(audit_ids), "seeds": SEEDS,
        "conditioned_mechanism": "generate_conditional(task=t2m, with_len=True): 'Generate motion with <N> frames: <caption>' instruction. soft conditioning (decode 는 동일 do_sample).",
        "direct": direct, "length_conditioned": cond,
        "per_prompt_failure_cases": per_prompt,
        "claim_boundary": "MotionGPT 길이 제어 개선 가능성 진단. semantic validity 전수(R-Prec)는 미측정(AR-051). 절대 품질 claim 아님.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n=== direct vs length-conditioned (MotionGPT) ===")
    print(f"  {'metric':<22} {'direct':>10} {'conditioned':>12}")
    for k in ("exact_match_rate", "ratio_in_0.8_1.25", "degenerate_le8_rate", "ratio_median", "actual_min", "mean_runtime_sec"):
        print(f"  {k:<22} {direct[k]:>10} {cond[k]:>12}")
    print("\n  -- 직접 실패 prompt 개선 --")
    for sid, v in per_prompt.items():
        print(f"    {sid} tgt={v['target']}  direct={v['direct_lengths']}  cond={v['conditioned_lengths']}")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
