"""ad-hoc: G2 general 3-level vs 5-level (g2_general_pilot_v1.json + 5level snapshot)."""
import json
import numpy as np
from scipy import stats
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
s3 = json.load(open(REPO_ROOT / "evals/snapshots/g2_general_pilot_v1.json", encoding="utf-8"))
s5 = json.load(open(REPO_ROOT / "evals/snapshots/oracle_sequence_g2_general_5level_v1.json", encoding="utf-8"))

s3_by = {ps["trial_id"]: ps["sequence_oracle"] for ps in s3["per_sample"]}
s5_by = {ps["trial_id"]: ps["best"] for ps in s5["per_sample"] if ps["best"]}

common = sorted(s3_by.keys() & s5_by.keys())
print(f"common: {len(common)} samples")

ng_3 = np.array([s3_by[t]["netgain"] for t in common])
ng_5 = np.array([s5_by[t]["netgain"] for t in common])
diff = ng_5 - ng_3
print(f"3-level median={np.median(ng_3):.5f}, mean={ng_3.mean():.5f}")
print(f"5-level median={np.median(ng_5):.5f}, mean={ng_5.mean():.5f}")
print(f"Delta(5-3) median={np.median(diff):.5f}, mean={diff.mean():.5f}")
print(f"n_5_strict/tie/loss: {(diff>1e-12).sum()}/{(np.abs(diff)<=1e-12).sum()}/{(diff<-1e-12).sum()}")
if (np.abs(diff) > 1e-12).any():
    w = stats.wilcoxon(ng_5, ng_3, alternative="greater", zero_method="wilcox")
    print(f"Wilcoxon p (5>3): {float(w.pvalue):.5g}")
    if diff.std(ddof=1) > 1e-15:
        print(f"Cohen d: {diff.mean()/diff.std(ddof=1):+.3f}")
else:
    print("(all ties, no Wilcoxon)")

len_5 = Counter(s5_by[t]["length"] for t in common)
len_3 = Counter(s3_by[t]["length"] for t in common)
print(f"best_length 5-level: {dict(sorted(len_5.items()))}")
print(f"best_length 3-level: {dict(sorted(len_3.items()))}")
stop_5 = sum(1 for t in common if s5_by[t]["length"] == 0)
stop_3 = sum(1 for t in common if s3_by[t]["length"] == 0)
print(f"STOP rate: 5-level {stop_5}/{len(common)}, 3-level {stop_3}/{len(common)}")

fa_5 = Counter((s5_by[t]["sequence"][0][0] if s5_by[t]["length"] > 0 else "STOP") for t in common)
fa_3 = Counter((s3_by[t]["sequence"][0][0] if s3_by[t]["length"] > 0 else "STOP") for t in common)
fs_5 = Counter((s5_by[t]["sequence"][0][2] if s5_by[t]["length"] > 0 else "n/a") for t in common)
fs_3 = Counter((s3_by[t]["sequence"][0][2] if s3_by[t]["length"] > 0 else "n/a") for t in common)
print(f"first_action 5-level: {dict(fa_5)}")
print(f"first_action 3-level: {dict(fa_3)}")
print(f"first_strength 5-level: {dict(fs_5)}")
print(f"first_strength 3-level: {dict(fs_3)}")

# Save summary.
summary = {
    "domain": "g2_general",
    "n_common": len(common),
    "ng_3": {"median": float(np.median(ng_3)), "mean": float(ng_3.mean())},
    "ng_5": {"median": float(np.median(ng_5)), "mean": float(ng_5.mean())},
    "diff_5_minus_3": {"median": float(np.median(diff)), "mean": float(diff.mean())},
    "n_5_strict": int((diff > 1e-12).sum()),
    "n_tie": int((np.abs(diff) <= 1e-12).sum()),
    "n_loss": int((diff < -1e-12).sum()),
    "best_length_5": dict(sorted(len_5.items())),
    "best_length_3": dict(sorted(len_3.items())),
    "stop_5": stop_5, "stop_3": stop_3,
    "first_action_5": dict(fa_5), "first_action_3": dict(fa_3),
    "first_strength_5": dict(fs_5), "first_strength_3": dict(fs_3),
}
out = REPO_ROOT / "evals/snapshots/compare_3level_5level_g2_general.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[OK] wrote {out}")
