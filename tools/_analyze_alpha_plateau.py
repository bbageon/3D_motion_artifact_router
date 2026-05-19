"""ad-hoc: v2 grid 의 alpha plateau curve 분석 (one-shot)."""
import json
import sys
from collections import defaultdict

snapshot_path = sys.argv[1] if len(sys.argv) > 1 else "evals/snapshots/netgain_weight_grid_v2.json"
with open(snapshot_path, encoding="utf-8") as f:
    s = json.load(f)

by_alpha = defaultdict(lambda: -1.0)
by_alpha_full: dict[float, tuple] = {}
by_alpha_per_art: dict[float, dict[str, float]] = defaultdict(dict)

for r in s["all_results"]:
    a = r["alpha"]
    c = r["spearman_correlation_all"]
    if c > by_alpha[a]:
        by_alpha[a] = c
        by_alpha_full[a] = (r["beta"], r["gamma"], c)
    # also per-artifact best beta/gamma per alpha.
    for art, rho in r["spearman_correlation_by_artifact"].items():
        cur = by_alpha_per_art[a].get(art, -1.0)
        if rho > cur:
            by_alpha_per_art[a][art] = rho

print("alpha  | best_rho_all | best_beta | best_gamma | rho_foot | rho_bone | rho_jitter")
print("-" * 90)
for a in sorted(by_alpha.keys()):
    b, g, c = by_alpha_full[a]
    rf = by_alpha_per_art[a].get("foot_floating", 0.0)
    rb = by_alpha_per_art[a].get("bone_stretch_right_arm", 0.0)
    rj = by_alpha_per_art[a].get("global_jitter", 0.0)
    print(f"{a:6.2f} | {c:.4f}       | {b:5.2f}     | {g:.2f}       | {rf:+.4f}  | {rb:+.4f}  | {rj:+.4f}")
