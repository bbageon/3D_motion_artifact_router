"""MotionGPT (G2) generation microservice.

Loads the MotionGPT (OpenMotionLab) model once at startup (resident on GPU) and
serves canonical ``[T, 22, 3]`` joints over HTTP. The upstream repo + flan-t5 LM
+ checkpoint are mounted read-only at /assets.

Mirrors generators/_motiongpt_inference.py but splits CLI one-shot into
load-once (startup) + generate-per-request.

Env: MGPT_ASSETS (/assets), MGPT_CFG (configs/config_h3d_stage3.yaml),
     MGPT_CKPT (optional explicit ckpt; else auto-find), MGPT_GPU_ID (0).
"""
from __future__ import annotations

import glob
import os
import sys
import time
import traceback

import numpy as np
import torch

_ORIG_TORCH_LOAD = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(*args, **kwargs)


torch.load = _patched_load  # type: ignore[assignment]

ASSETS = os.environ.get("MGPT_ASSETS", "/assets")
CFG = os.environ.get("MGPT_CFG", "configs/config_h3d_stage3.yaml")
CKPT = os.environ.get("MGPT_CKPT", "")
GPU_ID = int(os.environ.get("MGPT_GPU_ID", "0"))

sys.path.insert(0, ASSETS)
os.chdir(ASSETS)

_STATE: dict = {}


def _find_ckpt() -> str:
    if CKPT:
        return CKPT
    cands = (
        glob.glob("checkpoints/MotionGPT-base/*.ckpt")
        + glob.glob("checkpoints/MotionGPT-base/*.tar")
        + glob.glob("checkpoints/MotionGPT-base/*.bin")
    )
    if not cands:
        raise FileNotFoundError("No MotionGPT checkpoint under checkpoints/MotionGPT-base/")
    cands.sort(key=lambda p: {".ckpt": 0, ".tar": 1, ".bin": 2}.get(os.path.splitext(p)[1], 3))
    return cands[0]


def load_models() -> None:
    import pytorch_lightning as pl  # noqa: F401

    ckpt = _find_ckpt()
    # mGPT.config.parse_args reads sys.argv
    sys.argv = ["service", "--cfg", CFG]
    from mGPT.config import parse_args

    cfg = parse_args(phase="demo")
    cfg.TEST.CHECKPOINTS = ckpt

    use_cuda = torch.cuda.is_available() and GPU_ID >= 0
    device = torch.device(f"cuda:{GPU_ID}" if use_cuda else "cpu")

    from mGPT.data.build_data import build_data
    from mGPT.models.build_model import build_model

    datamodule = build_data(cfg)
    model = build_model(cfg, datamodule)
    state_dict = torch.load(ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device).eval()

    _STATE.update(cfg=cfg, model=model, device=device, ckpt=ckpt)


def generate_motion(prompt: str, n_frames: int, seed: int):
    import pytorch_lightning as pl

    pl.seed_everything(seed)
    model = _STATE["model"]
    batch = {"length": [int(n_frames)], "text": [prompt]}
    with torch.no_grad():
        outputs = model(batch, task="t2m")

    joints_obj = outputs["joints"]
    if isinstance(joints_obj, list):
        joints = joints_obj[0]
    else:
        joints = joints_obj[0] if joints_obj.dim() == 4 else joints_obj
    if hasattr(joints, "cpu"):
        joints = joints.detach().cpu().numpy()
    joints = np.asarray(joints, dtype=np.float64)

    length_field = outputs.get("length")
    try:
        length_generated = int(length_field[0])
    except (TypeError, IndexError):
        length_generated = int(joints.shape[0])
    return joints, length_generated


# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="ArtifactRouter MotionGPT service")


class GenRequest(BaseModel):
    prompt: str
    n_frames: int = 40
    seed: int = 42


try:
    load_models()
except Exception:  # noqa: BLE001
    _STATE["error"] = traceback.format_exc()
    print("[MotionGPT] load_models FAILED:\n" + _STATE["error"], flush=True)


@app.get("/health")
def health():
    return {
        "status": "ok" if "model" in _STATE else "error",
        "generator": "MotionGPT_G2",
        "cuda_available": torch.cuda.is_available(),
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "device": str(_STATE.get("device")),
        "checkpoint": _STATE.get("ckpt"),
        "error": _STATE.get("error"),
    }


@app.post("/generate")
def generate(req: GenRequest):
    if "model" not in _STATE:
        raise HTTPException(status_code=503, detail="model not loaded: " + str(_STATE.get("error")))
    t0 = time.time()
    try:
        motion_traj, length_generated = generate_motion(req.prompt, req.n_frames, req.seed)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    if motion_traj.ndim != 3 or motion_traj.shape[1] != 22 or motion_traj.shape[2] != 3:
        raise HTTPException(status_code=500, detail=f"unexpected motion shape {motion_traj.shape}")
    # MotionGPT 복원 결과는 이미 root trajectory 포함 → trajectory; local 파생 (AR-048 표준화).
    motion_local = motion_traj - motion_traj[:, 0:1, :]
    return {
        "motion": motion_local.tolist(),               # canonical (local) — 표준화 (이전엔 world 반환했음)
        "motion_trajectory": motion_traj.tolist(),      # AR-048 root-preserving
        "motion_local": motion_local.tolist(),          # AR-048 explicit
        "shape": list(motion_traj.shape),
        "fps": 20,
        "generator_id": f"G2_motiongpt_{os.path.splitext(os.path.basename(_STATE['ckpt']))[0]}",
        "compute_backend": "docker_gpu" if (torch.cuda.is_available() and GPU_ID >= 0) else "docker_cpu",
        "seed": req.seed,
        "prompt": req.prompt,
        "length_generated": length_generated,
        "elapsed_sec": round(time.time() - t0, 3),
    }
