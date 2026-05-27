"""Step E-2.7 (사용자 directive 2026-05-27): representative case selection.

사용자 directive:
> "n=300에서 대표 샘플을 뽑습니다. 추천:
>  - top unsafe NetGain samples 5개
>  - FootLock violation samples 5개
>  - safe STOP shift samples 5개
>  - false-positive 의심 samples 5개
>  이건 나중에 figure / qualitative evidence로 씁니다."

본 도구는 n=300 audit snapshot + threshold sensitivity snapshot 에서 4 category 의
대표 sample 추출 + manifest 작성 (figure / qualitative evidence 용 cross-link).

4 category:
  1. top_unsafe_netgain   — unsafe NetGain 가장 높은 violation sample (큰 distortion 위험).
  2. footlock_violation   — FootLock tool 의 systematic side-effect 대표.
  3. safe_stop_shift      — unsafe→STOP 으로 shift 한 sample (gate 의 효과 명확).
  4. false_positive_susp  — clean_p99 absolute flag but NOT regression (자연 high-CV).

CLI:
    python -m tools.safe_oracle_representative_cases \
        --audit evals/snapshots/safe_sequence_oracle_g2_hml3d_test300_v1.json \
        --threshold evals/snapshots/physical_gate_threshold_sensitivity_v1.json \
        --output evals/snapshots/safe_oracle_representative_cases_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_sequence_oracle_g2_hml3d_test300_v1.json")
    parser.add_argument("--threshold", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "physical_gate_threshold_sensitivity_v1.json")
    parser.add_argument("--g2-batch-dir", type=Path,
                        default=REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "evals" / "snapshots" / "safe_oracle_representative_cases_v1.json")
    parser.add_argument("--n-per-category", type=int, default=5)
    args = parser.parse_args()

    audit = json.load(open(args.audit, encoding="utf-8"))
    thresh = json.load(open(args.threshold, encoding="utf-8"))
    table = {t["trial_id"]: t for t in audit["comparison_table"]}
    per_sample = {p["trial_id"]: p for p in audit["per_sample"]}
    cls = {c["trial_id"]: c for c in thresh["per_sample_classifications"]}

    def _prompt(tid: str) -> str:
        return per_sample[tid].get("g2_prompt", "")[:100]

    def _entry(tid: str, reason: str) -> dict:
        t = table[tid]
        c = cls.get(tid, {})
        return {
            "trial_id": tid,
            "prompt": _prompt(tid),
            "sample_npy": f"external_assets/g2_generated_hml3d_test300_seed20260527/{tid}.npy",
            "reason": reason,
            "unsafe_netgain": t["unsafe_netgain"],
            "unsafe_sequence": t["unsafe_sequence"],
            "unsafe_hard_viol": t["unsafe_hard_viol"],
            "safe_netgain": t["safe_netgain"],
            "safe_sequence": t["safe_sequence"],
            "decision_shift": t["decision_shift"],
            "bone_cv_before": c.get("before"),
            "bone_cv_after": c.get("after"),
            "bone_cv_delta": c.get("delta"),
        }

    shifted = [t for t in audit["comparison_table"] if t["decision_shift"] == "shifted"]

    # Category 1: top unsafe NetGain (among shifted = violation).
    cat1 = sorted(shifted, key=lambda t: -(t["unsafe_netgain"] or 0))[: args.n_per_category]
    cat1_entries = [_entry(t["trial_id"], "top unsafe NetGain (large distortion risk)") for t in cat1]

    # Category 2: FootLock violation (shifted + unsafe first tool FootLock), by bone CV delta.
    footlock_shifted = [t for t in shifted if t["unsafe_sequence"].startswith("FootLock")]
    cat2_sorted = sorted(footlock_shifted,
                         key=lambda t: -(cls.get(t["trial_id"], {}).get("delta") or 0))
    # Avoid duplication with cat1.
    cat1_ids = {e["trial_id"] for e in cat1_entries}
    cat2_entries = [_entry(t["trial_id"], "FootLock systematic side-effect (bone CV worst)")
                    for t in cat2_sorted if t["trial_id"] not in cat1_ids][: args.n_per_category]

    # Category 3: safe STOP shift (unsafe→STOP), by unsafe NetGain.
    stop_shift = [t for t in shifted if t["safe_sequence"] == "STOP"]
    cat3_sorted = sorted(stop_shift, key=lambda t: -(t["unsafe_netgain"] or 0))
    cat3_entries = [_entry(t["trial_id"], "safe oracle STOP shift (gate effect clear)")
                    for t in cat3_sorted][: args.n_per_category]

    # Category 4: false-positive suspects (clean_p99 flag, NOT before+eps; natural high-CV).
    fp_susp = [c for c in thresh["per_sample_classifications"]
               if c["flagged"]["clean_p99"] and not c["flagged"]["before_plus_eps"]]
    # Sort by before-CV descending (most naturally-high).
    fp_sorted = sorted(fp_susp, key=lambda c: -(c["before"] or 0))
    cat4_entries = []
    for c in fp_sorted[: args.n_per_category]:
        tid = c["trial_id"]
        # FP suspects may be unchanged samples (unsafe_best had no real regression).
        if tid in table:
            cat4_entries.append(_entry(tid, "false-positive suspect (natural high bone CV, NOT worsened by correction)"))
        else:
            cat4_entries.append({
                "trial_id": tid, "prompt": _prompt(tid),
                "sample_npy": f"external_assets/g2_generated_hml3d_test300_seed20260527/{tid}.npy",
                "reason": "false-positive suspect (natural high bone CV)",
                "bone_cv_before": c["before"], "bone_cv_after": c["after"], "bone_cv_delta": c["delta"],
            })

    out = {
        "schema_version": "1.0.0",
        "record_type": "safe_oracle_representative_cases",
        "task_id": "safe_oracle_representative_cases_v1",
        "audit_source": str(args.audit),
        "threshold_source": str(args.threshold),
        "n_per_category": args.n_per_category,
        "categories": {
            "1_top_unsafe_netgain": cat1_entries,
            "2_footlock_violation": cat2_entries,
            "3_safe_stop_shift": cat3_entries,
            "4_false_positive_suspect": cat4_entries,
        },
        "usage_note": "figure / qualitative evidence 용. Step E-2.7 (사용자 directive 2026-05-27). "
                      "side-by-side mesh GIF 생성 시 본 manifest 의 sample_npy + unsafe_sequence 사용.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Step E-2.7 Representative Cases (n=300) ===")
    for cat_name, entries in out["categories"].items():
        print(f"\n[{cat_name}] ({len(entries)} samples)")
        for e in entries:
            ng = e.get("unsafe_netgain")
            ng_str = f"{ng:+.4f}" if ng is not None else "n/a"
            delta = e.get("bone_cv_delta")
            delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
            print(f"  {e['trial_id']:<14} unsafe_ng={ng_str} bone_cv_delta={delta_str} | {e.get('unsafe_sequence','?')} -> {e.get('safe_sequence','?')}")
            print(f"      prompt: '{e['prompt'][:70]}'")
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
