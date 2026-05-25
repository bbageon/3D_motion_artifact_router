"""ad-hoc: Generate GIF 6 only (synthetic 002806 oracle, from oracle_sequence_multi_v1)."""
import sys
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external_assets" / "code"))
from plot_3d_motion import plot_3d_motion

from correction_tools import BoneProjectionTool, FootLockTool, VelocitySmoothingTool
from tools.synthetic_injection import inject_foot_floating, inject_jitter

TOOL_BY_NAME = {
    "FootLockTool": FootLockTool(default_ground_y=0.0),
    "BoneProjectionTool": BoneProjectionTool(),
    "VelocitySmoothingTool": VelocitySmoothingTool(),
}

clean = np.load(str(REPO_ROOT / "external_assets" / "HumanML3D" / "new_joints" / "002806.npy")).astype(np.float64)
m1 = inject_foot_floating(clean, lift_height=0.08, seed=42)
corrupted = inject_jitter(m1, noise_std=0.05, seed=1042)

with open(REPO_ROOT / "evals" / "snapshots" / "oracle_sequence_multi_v1.json", encoding="utf-8") as f:
    snap = json.load(f)
trials = {r["trial_id"]: r for r in snap["per_sample"] if r.get("best_A")}
seq = trials["002806"]["best_A"]["sequence"]
print(f"[INFO] applying sequence len={len(seq)}: {seq}")

T = corrupted.shape[0]
out = corrupted.copy()
for step in seq:
    tool = TOOL_BY_NAME[step[0]]
    out, _ = tool.apply(out, target_part=step[1], target_joints=[], frame_range=(0, T - 1), strength=step[2])

stride = 2
corr = corrupted[::stride].astype(np.float32)
oracle = out[::stride].astype(np.float32)
m = min(corr.shape[0], oracle.shape[0])
corr, oracle = corr[:m], oracle[:m]
print(f"[INFO] frames after stride: {m}")

save_path = REPO_ROOT / "reports" / "figures" / "2026-05-25" / "skeleton_gif" / "synthetic_002806_vs_oracle.gif"
plot_3d_motion(
    save_path=str(save_path),
    obs_arr=np.zeros((0, 22, 3), dtype=np.float32),
    pred_gt_arr=corr,
    pred_model_arr=oracle,
    title="Synthetic 002806 corrupted gray vs oracle orange len5",
    fps=8, figsize=(6, 6), hold_last_frame_ms=600,
)
print(f"[OK] {save_path}")
