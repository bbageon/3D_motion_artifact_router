"""MDM (G1, diffusion) generation microservice.

Loads the MDM model + diffusion + HumanML3D loader once at startup (resident on
GPU) and serves canonical ``[T, 22, 3]`` joints over HTTP. The upstream repo +
checkpoint + dataset/glove/body_models are mounted read-only at /assets.

Mirrors generators/_mdm_inference.py but splits CLI one-shot into load-once
(startup) + generate-per-request.

Env: MDM_ASSETS (/assets), MDM_MODEL_PATH (optional; else auto-find under save/),
     MDM_GPU_ID (0), MDM_GUIDANCE (2.5).
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

ASSETS = os.environ.get("MDM_ASSETS", "/assets")
MODEL_PATH = os.environ.get("MDM_MODEL_PATH", "")
GPU_ID = int(os.environ.get("MDM_GPU_ID", "0"))
GUIDANCE = float(os.environ.get("MDM_GUIDANCE", "2.5"))

os.chdir(ASSETS)
sys.path.insert(0, ASSETS)

_STATE: dict = {}


def _find_model() -> str:
    if MODEL_PATH:
        return MODEL_PATH
    cands = sorted(glob.glob("save/**/model*.pt", recursive=True))
    if not cands:
        raise FileNotFoundError("No MDM checkpoint under save/**/model*.pt")
    return cands[-1]


def load_models() -> None:
    from utils.parser_util import generate_args
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils.sampler_util import ClassifierFreeSampleModel
    from utils import dist_util
    from data_loaders.get_data import get_dataset_loader

    model_path = _find_model()
    # MDM parser reads sys.argv + args.json next to checkpoint
    old_argv = sys.argv[:]
    sys.argv = [
        "service", "--model_path", model_path,
        "--num_samples", "1", "--num_repetitions", "1",
        "--guidance_param", str(GUIDANCE), "--device", str(max(GPU_ID, 0)),
    ]
    try:
        mdm_args = generate_args()
    finally:
        sys.argv = old_argv

    dist_util.setup_dist(mdm_args.device)
    max_frames = 196 if mdm_args.dataset in ["kit", "humanml"] else 60

    data = get_dataset_loader(
        name=mdm_args.dataset, batch_size=1, num_frames=max_frames,
        split="test", hml_mode="text_only", fixed_len=0, pred_len=0,
        device=dist_util.dev(),
    )

    model, diffusion = create_model_and_diffusion(mdm_args, data)
    load_saved_model(model, model_path, use_avg=mdm_args.use_ema)
    if mdm_args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    _STATE.update(
        model=model, diffusion=diffusion, data=data, args=mdm_args,
        max_frames=max_frames, device=dist_util.dev(), model_path=model_path,
    )


def generate_motion(prompt: str, n_frames: int, seed: int) -> np.ndarray:
    from utils.fixseed import fixseed
    from data_loaders.tensors import collate
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    fixseed(seed)
    model = _STATE["model"]
    diffusion = _STATE["diffusion"]
    data = _STATE["data"]
    mdm_args = _STATE["args"]
    device = _STATE["device"]
    max_frames = _STATE["max_frames"]

    n = min(max_frames, int(n_frames))
    data.fixed_length = n

    collate_args = [{"inp": torch.zeros(n), "tokens": None, "lengths": n, "text": prompt}]
    _, model_kwargs = collate(collate_args)
    model_kwargs["y"] = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_kwargs["y"].items()
    }
    if mdm_args.guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(1, device=device) * mdm_args.guidance_param
    if "text" in model_kwargs["y"]:
        model_kwargs["y"]["text_embed"] = model.encode_text(model_kwargs["y"]["text"])

    sample = diffusion.p_sample_loop(
        model, (1, model.njoints, model.nfeats, n), clip_denoised=False,
        model_kwargs=model_kwargs, skip_timesteps=0, init_image=None,
        progress=False, dump_steps=None, noise=None, const_noise=False,
    )

    if model.data_rep == "hml_vec":
        n_joints = 22 if sample.shape[1] == 263 else 21
        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        sample = recover_from_ric(sample, n_joints)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

    rot2xyz_pose_rep = "xyz" if model.data_rep in ["xyz", "hml_vec"] else model.data_rep
    sample = model.rot2xyz(
        x=sample,
        mask=None if rot2xyz_pose_rep == "xyz" else model_kwargs["y"]["mask"].reshape(1, n).bool(),
        pose_rep=rot2xyz_pose_rep, glob=True, translation=True, jointstype="smpl",
        vertstrans=True, betas=None, beta=0, glob_rot=None, get_rotations_back=False,
    )

    motion = sample[0].detach().cpu().numpy().transpose(2, 0, 1).astype(np.float64)
    motion = motion[: int(n_frames)]
    # AR-048: return trajectory-preserving (root kept). motion_local derived in /generate.
    return motion


# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="ArtifactRouter MDM service")


class GenRequest(BaseModel):
    prompt: str
    n_frames: int = 40
    seed: int = 42


try:
    load_models()
except Exception:  # noqa: BLE001
    _STATE["error"] = traceback.format_exc()
    print("[MDM] load_models FAILED:\n" + _STATE["error"], flush=True)


@app.get("/health")
def health():
    return {
        "status": "ok" if "model" in _STATE else "error",
        "generator": "MDM_G1",
        "cuda_available": torch.cuda.is_available(),
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "device": str(_STATE.get("device")),
        "checkpoint": _STATE.get("model_path"),
        "error": _STATE.get("error"),
    }


@app.post("/generate")
def generate(req: GenRequest):
    if "model" not in _STATE:
        raise HTTPException(status_code=503, detail="model not loaded: " + str(_STATE.get("error")))
    t0 = time.time()
    try:
        motion_traj = generate_motion(req.prompt, req.n_frames, req.seed)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    if motion_traj.ndim != 3 or motion_traj.shape[1] != 22 or motion_traj.shape[2] != 3:
        raise HTTPException(status_code=500, detail=f"unexpected motion shape {motion_traj.shape}")
    motion_local = motion_traj - motion_traj[:, 0:1, :]  # AR-048 canonical root-relative
    return {
        "motion": motion_local.tolist(),               # canonical (local) — backward compat
        "motion_trajectory": motion_traj.tolist(),      # AR-048 root-preserving
        "motion_local": motion_local.tolist(),          # AR-048 explicit
        "shape": list(motion_traj.shape),
        "fps": 20,
        "generator_id": f"G1_mdm_{os.path.splitext(os.path.basename(_STATE['model_path']))[0]}",
        "compute_backend": "docker_gpu" if (torch.cuda.is_available() and GPU_ID >= 0) else "docker_cpu",
        "seed": req.seed,
        "prompt": req.prompt,
        "elapsed_sec": round(time.time() - t0, 3),
    }
