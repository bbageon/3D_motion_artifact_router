"""ad-hoc: axis convention check (Y-up vs Z-up)."""
import numpy as np

clean = np.load("external_assets/HumanML3D/new_joints/002806.npy").astype(np.float64)
print("=== HumanML3D 002806 frame 0 ===")
print(f"shape: {clean.shape}")
print(f"PELVIS (j 0): {clean[0, 0]}")
print(f"HEAD (j 15): {clean[0, 15]}")
print(f"LEFT_FOOT (j 10): {clean[0, 10]}")
print(f"RIGHT_FOOT (j 11): {clean[0, 11]}")
print(f"  -- if Y-up: HEAD_y > PELVIS_y > FOOT_y")
print(f"  X range: [{clean[0,:,0].min():.3f}, {clean[0,:,0].max():.3f}]")
print(f"  Y range: [{clean[0,:,1].min():.3f}, {clean[0,:,1].max():.3f}]")
print(f"  Z range: [{clean[0,:,2].min():.3f}, {clean[0,:,2].max():.3f}]")

# Find joint span across axes:
spans = [(clean[0,:,a].max() - clean[0,:,a].min(), a) for a in [0, 1, 2]]
spans.sort(reverse=True)
print(f"  largest axis span at axis {spans[0][1]} (span {spans[0][0]:.3f})")
print(f"     (vertical axis usually has the largest span: head-to-foot)")

g2 = np.load("external_assets/g2_general_pilot_v1/motion_010.npy").astype(np.float64)
print(f"\n=== G2 motion_010 (kick) frame 0 ===")
print(f"shape: {g2.shape}")
print(f"PELVIS: {g2[0, 0]}")
print(f"HEAD: {g2[0, 15]}")
print(f"LEFT_FOOT: {g2[0, 10]}")
print(f"RIGHT_FOOT: {g2[0, 11]}")
print(f"  X range: [{g2[0,:,0].min():.3f}, {g2[0,:,0].max():.3f}]")
print(f"  Y range: [{g2[0,:,1].min():.3f}, {g2[0,:,1].max():.3f}]")
print(f"  Z range: [{g2[0,:,2].min():.3f}, {g2[0,:,2].max():.3f}]")

spans2 = [(g2[0,:,a].max() - g2[0,:,a].min(), a) for a in [0, 1, 2]]
spans2.sort(reverse=True)
print(f"  largest axis span at axis {spans2[0][1]} (span {spans2[0][0]:.3f})")
