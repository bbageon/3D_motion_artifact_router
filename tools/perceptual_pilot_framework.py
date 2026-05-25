"""Step 6: Perceptual Pilot Framework — pairwise A/B comparison manifest + analysis.

사용자 directive (2026-05-25 9-step plan Step 6):
> "소규모 perceptual pilot — 20-30 pairwise A/B. method label 숨김. NetGain rank
>  vs human preference Spearman correlation."

본 도구는 두 mode:
  1. **manifest** mode — pairwise comparison manifest (blind label randomized) +
     response template (CSV) 생성.
  2. **analyze** mode — response CSV 가 채워지면 분석 (NetGain winner vs human
     preference agreement, Spearman correlation, method win rate, disagreement
     sample list).

평가 pair types (사용자 directive):
  - Original vs B2-large
  - Original vs B2-small
  - B2-small vs B2-large
  - Original/STOP vs oracle (oracle=STOP 인 sample 의 경우 identity)
  - B2-val-best vs oracle

[AGENTS.md §3-17](../AGENTS.md) 의 **quality-validated evidence (b) perceptual
rating sub-tier (정식)** 의 첫 정식 framework.

CLI:
    # Manifest 생성:
    python -m tools.perceptual_pilot_framework manifest \\
        --pilot-snapshot evals/snapshots/g2_general_pilot_v1.json \\
        --gif-dir reports/figures/2026-05-25/g2_general_gif_yup_fix \\
        --output-dir evals/perceptual/pilot_v1 \\
        --seed 42

    # Analysis (response CSV 채워진 후):
    python -m tools.perceptual_pilot_framework analyze \\
        --manifest evals/perceptual/pilot_v1/pairs_manifest_v1.json \\
        --response evals/perceptual/pilot_v1/response_v1.csv \\
        --output evals/snapshots/perceptual_pilot_v1.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

PAIR_TYPES = [
    ("Original", "B2-large"),
    ("Original", "B2-small"),
    ("B2-small", "B2-large"),
    ("Original", "oracle"),
    ("B2-val-best", "oracle"),
]


def _method_netgain_per_sample(snap: dict[str, Any]) -> dict[str, dict[str, float]]:
    """G2 general pilot 의 per-sample 의 method 별 NetGain."""
    out = {}
    for s in snap["per_sample"]:
        tid = s["trial_id"]
        # Original = 0 NetGain (baseline reference).
        out[tid] = {
            "Original": 0.0,
            "B2-small": s["B2-small"]["netgain"],
            "B2-medium": s["B2-medium"]["netgain"],
            "B2-large": s["B2-large"]["netgain"],
            "B2-val-best": s["B2-val-best"]["netgain"],
            "oracle": s["sequence_oracle"]["netgain"],
        }
    return out


def _gif_path(gif_dir: Path, trial_id: str, method: str) -> str | None:
    """주어진 trial_id + method 에 대응하는 GIF 파일 path. (없으면 None.)"""
    if method == "Original":
        # Original 만 보여주는 GIF — `vs_b2_small.gif` 의 gray 가 original.
        # 본 framework 는 overlay GIF 만 사용 — Original 의 stand-alone 은 별도 (단 부록 W 의
        # original_single 같은 형식). 본 framework 는 overlay 만 의 비교 manifest 로 작성.
        return None  # Original is referenced via overlay (gray skeleton).
    if method == "B2-small":
        return str(gif_dir / f"{trial_id}_vs_b2_small.gif")
    if method == "B2-medium":
        return str(gif_dir / f"{trial_id}_vs_b2_medium.gif")
    if method == "B2-large":
        return str(gif_dir / f"{trial_id}_vs_b2_large.gif")
    if method == "B2-val-best":
        # val-best 는 G2 의 경우 = B2-small (부록 V). overlay GIF는 same as b2_small.
        return str(gif_dir / f"{trial_id}_vs_b2_small.gif")
    if method == "oracle":
        return str(gif_dir / f"{trial_id}_vs_oracle.gif")
    return None


def _build_manifest(
    *, snap: dict[str, Any], gif_dir: Path, seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    per_sample_netgain = _method_netgain_per_sample(snap)
    available_trials = sorted(per_sample_netgain.keys())
    print(f"[INFO] available trials: {len(available_trials)} — {available_trials[:5]}...")

    pairs = []
    pair_id = 0
    for trial in available_trials:
        for method_a, method_b in PAIR_TYPES:
            # GIF availability check.
            path_a = _gif_path(gif_dir, trial, method_a)
            path_b = _gif_path(gif_dir, trial, method_b)
            # Original 의 경우: overlay GIF 의 gray skeleton 으로 reference — pair 의 한 쪽이
            # Original 이면 다른 쪽 GIF 의 gray vs orange 의 색 의 의미가 self-explanatory.
            # 본 manifest 는 그 의미를 명시.
            if method_a == "Original" and path_b is None:
                continue
            if method_b == "Original" and path_a is None:
                continue
            if path_a is None and path_b is None:
                continue
            # Compute NetGain winner per system (no human input).
            ng_a = per_sample_netgain[trial][method_a]
            ng_b = per_sample_netgain[trial][method_b]
            netgain_winner = method_a if ng_a > ng_b else (method_b if ng_b > ng_a else "tie")
            # Blind label randomization: A/B 의 method 매핑을 무작위.
            if rng.random() < 0.5:
                blind_A, blind_B = method_a, method_b
            else:
                blind_A, blind_B = method_b, method_a
            pair_id += 1
            pairs.append({
                "pair_id": f"P{pair_id:03d}",
                "trial_id": trial,
                "category_pair": f"{method_a} vs {method_b}",
                "blind_label_A": blind_A,
                "blind_label_B": blind_B,
                "method_to_gif_path": {method_a: path_a, method_b: path_b},
                "netgain": {method_a: ng_a, method_b: ng_b},
                "netgain_winner_hidden": netgain_winner,
                "note": f"Original 의 경우 overlay GIF 의 gray (do-nothing) 가 reference. method_b 의 GIF 의 gray skeleton 이 Original.",
            })

    return {
        "schema_version": "1.0.0",
        "record_type": "perceptual_pilot_manifest",
        "task_id": "perceptual_pilot_v1",
        "seed": seed,
        "pair_types": [f"{a} vs {b}" for a, b in PAIR_TYPES],
        "n_pairs": len(pairs),
        "n_trials": len(available_trials),
        "trial_ids": available_trials,
        "pairs": pairs,
    }


def _build_response_template(manifest: dict[str, Any]) -> str:
    """response CSV template 생성 — pair_id, blind labels, user choice, comments."""
    lines = ["pair_id,trial_id,category_pair,blind_label_A_gif,blind_label_B_gif,user_choice,confidence,comments"]
    for p in manifest["pairs"]:
        ga = p["method_to_gif_path"][p["blind_label_A"]] or "(Original=gray of overlay)"
        gb = p["method_to_gif_path"][p["blind_label_B"]] or "(Original=gray of overlay)"
        # user_choice: "A" or "B" or "tie" or empty (TBD).
        lines.append(f"{p['pair_id']},{p['trial_id']},\"{p['category_pair']}\",{ga},{gb},,,")
    return "\n".join(lines) + "\n"


def _analyze(manifest_path: Path, response_path: Path, output_path: Path) -> None:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    pairs_by_id = {p["pair_id"]: p for p in manifest["pairs"]}

    # Load response.
    responses = []
    with open(response_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            choice = row.get("user_choice", "").strip()
            if choice in ("A", "B", "tie"):
                responses.append(row)
    print(f"[INFO] loaded {len(responses)} responses (filtered from CSV)")
    if not responses:
        print("[WARN] no responses found — analysis skipped.")
        return

    # Agreement: NetGain winner vs human preference.
    agree_count = 0
    disagree_count = 0
    tie_count = 0
    disagreement_list = []
    pair_type_stats = defaultdict(lambda: {"agree": 0, "disagree": 0, "tie": 0})
    netgain_diffs = []
    human_winners = []
    for r in responses:
        pid = r["pair_id"]
        if pid not in pairs_by_id:
            continue
        p = pairs_by_id[pid]
        choice = r["user_choice"].strip()
        netgain_winner = p["netgain_winner_hidden"]
        # Map blind A/B → method.
        human_winner = p["blind_label_A"] if choice == "A" else (p["blind_label_B"] if choice == "B" else "tie")
        agreed = (human_winner == netgain_winner)
        category = p["category_pair"]
        if choice == "tie" or netgain_winner == "tie":
            tie_count += 1
            pair_type_stats[category]["tie"] += 1
        elif agreed:
            agree_count += 1
            pair_type_stats[category]["agree"] += 1
        else:
            disagree_count += 1
            pair_type_stats[category]["disagree"] += 1
            disagreement_list.append({
                "pair_id": pid, "trial_id": p["trial_id"],
                "category": category,
                "netgain_winner": netgain_winner,
                "human_winner": human_winner,
                "netgain_diff": p["netgain"][p["blind_label_A"]] - p["netgain"][p["blind_label_B"]],
            })
        # Spearman: NetGain difference (A-B) vs human preference (-1=B, 0=tie, +1=A).
        ng_a = p["netgain"][p["blind_label_A"]]
        ng_b = p["netgain"][p["blind_label_B"]]
        netgain_diffs.append(ng_a - ng_b)
        human_winners.append(1 if choice == "A" else (-1 if choice == "B" else 0))

    total = agree_count + disagree_count + tie_count
    agreement_rate = agree_count / max(total - tie_count, 1) if total > tie_count else 0.0

    from scipy import stats
    spearman_r, spearman_p = stats.spearmanr(netgain_diffs, human_winners) if len(netgain_diffs) > 1 else (None, None)
    kendall_tau, kendall_p = stats.kendalltau(netgain_diffs, human_winners) if len(netgain_diffs) > 1 else (None, None)

    summary = {
        "schema_version": "1.0.0",
        "record_type": "perceptual_pilot_analysis",
        "manifest_path": str(manifest_path),
        "response_path": str(response_path),
        "n_responses": len(responses),
        "agreement": {
            "agree": agree_count, "disagree": disagree_count, "tie": tie_count,
            "agreement_rate_excl_tie": float(agreement_rate),
        },
        "spearman_netgain_diff_vs_human": {
            "r": float(spearman_r) if spearman_r is not None else None,
            "p": float(spearman_p) if spearman_p is not None else None,
        },
        "kendall_tau": {
            "tau": float(kendall_tau) if kendall_tau is not None else None,
            "p": float(kendall_p) if kendall_p is not None else None,
        },
        "per_pair_type": dict(pair_type_stats),
        "disagreement_list": disagreement_list,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== Perceptual Pilot Analysis ===")
    print(f"  n_responses: {len(responses)}")
    print(f"  Agreement: {agree_count} agree, {disagree_count} disagree, {tie_count} tie")
    print(f"  Agreement rate (excl tie): {agreement_rate:.3f}")
    if spearman_r is not None:
        print(f"  Spearman (NetGain diff vs human): r={spearman_r:.3f}, p={spearman_p:.4g}")
        print(f"  Kendall tau: {kendall_tau:.3f}, p={kendall_p:.4g}")
    print(f"  Per-pair-type stats: {dict(pair_type_stats)}")
    print(f"  Disagreement count: {len(disagreement_list)}")
    print(f"[OK] wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Perceptual pilot framework (Step 6)")
    sub = parser.add_subparsers(dest="mode", required=True)

    manifest_p = sub.add_parser("manifest", help="generate pair manifest + response template")
    manifest_p.add_argument("--pilot-snapshot", type=Path, required=True)
    manifest_p.add_argument("--gif-dir", type=Path, required=True)
    manifest_p.add_argument("--output-dir", type=Path, required=True)
    manifest_p.add_argument("--seed", type=int, default=42)

    analyze_p = sub.add_parser("analyze", help="analyze filled response CSV")
    analyze_p.add_argument("--manifest", type=Path, required=True)
    analyze_p.add_argument("--response", type=Path, required=True)
    analyze_p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.mode == "manifest":
        with open(args.pilot_snapshot, encoding="utf-8") as f:
            snap = json.load(f)
        manifest = _build_manifest(snap=snap, gif_dir=args.gif_dir, seed=args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_dir / "pairs_manifest_v1.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] wrote {manifest_path} ({manifest['n_pairs']} pairs)")
        template = _build_response_template(manifest)
        template_path = args.output_dir / "response_template_v1.csv"
        template_path.write_text(template, encoding="utf-8")
        print(f"[OK] wrote {template_path}")
        # Stats.
        type_counts = Counter(p["category_pair"] for p in manifest["pairs"])
        print(f"\n  Pair type distribution: {dict(type_counts)}")
    elif args.mode == "analyze":
        _analyze(args.manifest, args.response, args.output)


if __name__ == "__main__":
    main()
