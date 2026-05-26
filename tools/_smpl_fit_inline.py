"""Step B-2-2: SMPL fit (joints -> SMPLify3D -> vertices) for mgpt env.

본 도구는 mgpt env 에서 실행되며, MotionGPT 의 fit.py 핵심 로직을 inline 으로
가져와 config path 의 hardcoded 절대경로를 본 프로젝트 의 deps 경로로 override.

Output: 각 input npy 옆에 {name}_mesh.npy (T, 6890, 3) + {name}_faces.npy (13776, 3).

CLI (mgpt env):
    conda activate mgpt
    python tools/_smpl_fit_inline.py \
        --dir external_assets/MotionGPT/staging/g2_top4 \
        --num-iters 50 --joint-category AMASS
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTIONGPT_ROOT = REPO_ROOT / "external_assets" / "MotionGPT"
SMPL_MODELS = MOTIONGPT_ROOT / "deps" / "smpl_models"

# chumpy (smplx dep) imports `from numpy import bool, int, ...` which are removed
# in numpy>=1.24. Monkey-patch numpy with Python builtins BEFORE chumpy import.
import builtins as _builtins
import numpy as _np
for _name in ("bool", "int", "float", "complex", "object", "str"):
    if not hasattr(_np, _name):
        setattr(_np, _name, getattr(_builtins, _name))
if not hasattr(_np, "unicode"):
    _np.unicode = str  # Python 3 unicode == str
if not hasattr(_np, "nan"):
    _np.nan = float("nan")
if not hasattr(_np, "inf"):
    _np.inf = float("inf")

# Patch sys.path before importing mGPT modules.
sys.path.insert(0, str(MOTIONGPT_ROOT))
# smplify.py does `sys.path.append(os.path.dirname(__file__))` then `import config`
# which is a SEPARATE module from mGPT.data.transforms.joints2rots.config.
# So we must inject the joints2rots dir to sys.path and patch BOTH modules.
JOINTS2ROTS_DIR = str(MOTIONGPT_ROOT / "mGPT" / "data" / "transforms" / "joints2rots")
sys.path.insert(0, JOINTS2ROTS_DIR)

# Patch hardcoded config paths (qualified module — used by our direct refs).
import mGPT.data.transforms.joints2rots.config as _cfg
_cfg.SMPL_MODEL_DIR = str(SMPL_MODELS) + os.sep
_cfg.GMM_MODEL_DIR = str(SMPL_MODELS) + os.sep
_cfg.SMPL_MEAN_FILE = str(SMPL_MODELS / "neutral_smpl_mean_params.h5")
_cfg.Part_Seg_DIR = str(SMPL_MODELS / "smplx_parts_segm.pkl")

# ALSO patch the unqualified `config` module loaded by smplify.py.
import config as _cfg2  # type: ignore
_cfg2.SMPL_MODEL_DIR = str(SMPL_MODELS) + os.sep
_cfg2.GMM_MODEL_DIR = str(SMPL_MODELS) + os.sep
_cfg2.SMPL_MEAN_FILE = str(SMPL_MODELS / "neutral_smpl_mean_params.h5")
_cfg2.Part_Seg_DIR = str(SMPL_MODELS / "smplx_parts_segm.pkl")

import h5py
import smplx
import torch

from mGPT.data.transforms.joints2rots.smplify import SMPLify3D


def fit_one(
    *, joints3d_np: np.ndarray, smplmodel, smplify: SMPLify3D, device,
    init_mean_pose: torch.Tensor, init_mean_shape: torch.Tensor,
    joint_category: str = "AMASS", num_joints: int = 22,
) -> np.ndarray:
    """Fit SMPL to (T, 22, 3) joints -> return vertices (T, 6890, 3)."""
    T = joints3d_np.shape[0]
    vertices_all = []

    pred_pose = init_mean_pose.clone()
    pred_betas = init_mean_shape.clone()
    pred_cam_t = torch.zeros(1, 3).to(device)
    keypoints_3d = torch.zeros(1, num_joints, 3).to(device)

    confidence_input = torch.ones(num_joints, device=device)
    if joint_category == "AMASS":
        # ankle / foot 의 confidence 증가.
        confidence_input[7] = 1.5
        confidence_input[8] = 1.5
        confidence_input[10] = 1.5
        confidence_input[11] = 1.5

    for idx in range(T):
        keypoints_3d[0, :, :] = torch.from_numpy(joints3d_np[idx]).float().to(device)
        (
            new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas,
            new_opt_cam_t, new_opt_joint_loss,
        ) = smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input,
        )
        # Use previous frame's fit as next init (smoother sequence).
        pred_pose = new_opt_pose
        pred_betas = new_opt_betas
        pred_cam_t = new_opt_cam_t
        # Forward to vertices.
        out = smplmodel(
            betas=new_opt_betas,
            global_orient=new_opt_pose[:, :3],
            body_pose=new_opt_pose[:, 3:],
            transl=new_opt_cam_t,
            return_verts=True,
        )
        verts = out.vertices.detach().cpu().numpy().squeeze()  # (6890, 3)
        vertices_all.append(verts)
        if (idx + 1) % 10 == 0:
            print(f"   frame {idx+1}/{T} fit (loss={float(new_opt_joint_loss):.4f})", flush=True)
    return np.stack(vertices_all, axis=0)  # (T, 6890, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True, help="Dir of input npy files.")
    parser.add_argument("--num-iters", type=int, default=50,
                        help="SMPLify3D iters/frame. 100 standard, 30-50 fast pilot.")
    parser.add_argument("--joint-category", type=str, default="AMASS")
    parser.add_argument("--num-joints", type=int, default=22)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")  # mgpt env is CPU-only.
    print(f"[INFO] device={device}, iters={args.num_iters}")
    print(f"[INFO] SMPL model dir: {_cfg.SMPL_MODEL_DIR}")

    smplmodel = smplx.create(
        _cfg.SMPL_MODEL_DIR, model_type="smpl", gender="neutral", ext="pkl", batch_size=1,
    ).to(device)
    faces = smplmodel.faces.copy()  # (13776, 3) int

    # Mean pose + shape.
    with h5py.File(_cfg.SMPL_MEAN_FILE, "r") as f:
        init_mean_pose = torch.from_numpy(f["pose"][:]).unsqueeze(0).float().to(device)
        init_mean_shape = torch.from_numpy(f["shape"][:]).unsqueeze(0).float().to(device)
    print(f"[INFO] init_mean_pose: {init_mean_pose.shape}, init_mean_shape: {init_mean_shape.shape}")

    smplify = SMPLify3D(
        smplxmodel=smplmodel, batch_size=1,
        joints_category=args.joint_category, num_iters=args.num_iters, device=device,
    )
    print("[INFO] SMPLify3D ready")

    # Save faces once (shared).
    faces_path = args.dir / "_smpl_faces.npy"
    if not faces_path.exists():
        np.save(str(faces_path), faces)
        print(f"[OK] saved faces: {faces_path}  shape={faces.shape}")

    npy_files = sorted(args.dir.glob("*.npy"))
    npy_files = [p for p in npy_files if not p.name.endswith("_mesh.npy")
                 and p.name != "_smpl_faces.npy"]
    print(f"[INFO] {len(npy_files)} input npy files to fit")

    for i, path in enumerate(npy_files, 1):
        mesh_path = path.parent / (path.stem + "_mesh.npy")
        if mesh_path.exists() and not args.overwrite:
            print(f"[SKIP {i}/{len(npy_files)}] {path.name} -> {mesh_path.name} (exists)")
            continue
        joints3d = np.load(str(path)).astype(np.float32)
        print(f"\n[{i}/{len(npy_files)}] fitting {path.name}: shape={joints3d.shape}")
        verts = fit_one(
            joints3d_np=joints3d, smplmodel=smplmodel, smplify=smplify, device=device,
            init_mean_pose=init_mean_pose, init_mean_shape=init_mean_shape,
            joint_category=args.joint_category, num_joints=args.num_joints,
        )
        np.save(str(mesh_path), verts.astype(np.float32))
        print(f"[OK {i}/{len(npy_files)}] wrote {mesh_path.name}  shape={verts.shape}")

    print(f"\n[OK] all fit complete. Output dir: {args.dir}")


if __name__ == "__main__":
    main()
