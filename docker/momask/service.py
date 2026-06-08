"""MoMask (G2) generation microservice.

Loads the MoMask masked + residual-VQ transformers once at startup (resident on
GPU) and serves canonical ``[T, 22, 3]`` joint positions over HTTP. The upstream
repo (EricGuo5513/momask-codes) + checkpoints are mounted read-only at /assets.

This mirrors generators/_momask_inference.py but splits one-shot CLI logic into
load-once (startup) + generate-per-request, so the model stays in GPU memory.

Env vars:
    MOMASK_ASSETS    mount point of momask-codes repo (default /assets)
    MOMASK_GPU_ID    cuda device id (default 0; -1 for CPU)
    MOMASK_DATASET / MOMASK_NAME / MOMASK_RES_NAME / MOMASK_CKPT_DIR
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# torch>=2.6 changed torch.load default to weights_only=True, which breaks the
# full-pickle checkpoints MoMask ships. Force the legacy behaviour.
_ORIG_TORCH_LOAD = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(*args, **kwargs)


torch.load = _patched_load  # type: ignore[assignment]

ASSETS = os.environ.get("MOMASK_ASSETS", "/assets")
DATASET = os.environ.get("MOMASK_DATASET", "t2m")
NAME = os.environ.get("MOMASK_NAME", "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns")
RES_NAME = os.environ.get("MOMASK_RES_NAME", "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw")
CKPT_DIR = os.environ.get("MOMASK_CKPT_DIR", "./checkpoints")
GPU_ID = int(os.environ.get("MOMASK_GPU_ID", "0"))

os.chdir(ASSETS)
sys.path.insert(0, ASSETS)

_STATE: dict = {}


def load_models() -> None:
    from gen_t2m import load_len_estimator, load_res_model, load_trans_model, load_vq_model
    from utils.get_opt import get_opt

    use_cuda = torch.cuda.is_available() and GPU_ID >= 0
    device = torch.device(f"cuda:{GPU_ID}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(GPU_ID)

    dim_pose = 251 if DATASET == "kit" else 263
    root_dir = Path(CKPT_DIR) / DATASET / NAME
    model_opt = get_opt(str(root_dir / "opt.txt"), device=device)

    vq_opt = get_opt(str(Path(CKPT_DIR) / DATASET / model_opt.vq_name / "opt.txt"), device=device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt = get_opt(str(Path(CKPT_DIR) / DATASET / RES_NAME / "opt.txt"), device=device)
    opt = SimpleNamespace(
        name=NAME, device=device, cond_scale=4.0, temperature=1.0,
        topkr=0.9, time_steps=18, gumbel_sample=False,
    )
    res_model = load_res_model(res_opt, vq_opt, opt)
    t2m_transformer = load_trans_model(model_opt, opt, "latest.tar")
    length_estimator = load_len_estimator(model_opt)

    for m in (t2m_transformer, vq_model, res_model, length_estimator):
        m.eval().to(device)

    meta = Path(CKPT_DIR) / DATASET / model_opt.vq_name / "meta"
    mean = np.load(meta / "mean.npy")
    std = np.load(meta / "std.npy")

    _STATE.update(
        device=device, opt=opt, t2m=t2m_transformer, vq=vq_model,
        res=res_model, mean=mean, std=std,
    )


def generate_motion(prompt: str, n_frames: int, seed: int) -> np.ndarray:
    from utils.fixseed import fixseed
    from utils.motion_process import recover_from_ric

    fixseed(seed)
    device = _STATE["device"]
    opt = _STATE["opt"]
    t2m, res, vq = _STATE["t2m"], _STATE["res"], _STATE["vq"]
    mean, std = _STATE["mean"], _STATE["std"]

    m_length = torch.LongTensor([int(n_frames)]).to(device)
    token_lens = torch.clamp(m_length // 4, min=1)
    m_length = token_lens * 4
    captions = [prompt]

    with torch.no_grad():
        mids = t2m.generate(
            captions, token_lens, timesteps=opt.time_steps, cond_scale=opt.cond_scale,
            temperature=opt.temperature, topk_filter_thres=opt.topkr, gsample=False,
        )
        mids = res.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
        pred_motions = vq.forward_decoder(mids).detach().cpu().numpy()
        data = pred_motions * std + mean

    joint_data = data[0][: int(m_length[0].item())]
    joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy().astype(np.float64)
    # AR-048: return trajectory-preserving (root kept). motion_local derived in /generate.
    return joint


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="ArtifactRouter MoMask service")


class GenRequest(BaseModel):
    prompt: str
    n_frames: int = 40
    seed: int = 42


# Load once at import (uvicorn imports service:app). Capture errors so /health
# can report them instead of crash-looping the container.
try:
    load_models()
except Exception:  # noqa: BLE001
    _STATE["error"] = traceback.format_exc()
    print("[MoMask] load_models FAILED:\n" + _STATE["error"], flush=True)


@app.get("/health")
def health():
    return {
        "status": "ok" if "t2m" in _STATE else "error",
        "generator": "MoMask_G2",
        "cuda_available": torch.cuda.is_available(),
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "device": str(_STATE.get("device")),
        "error": _STATE.get("error"),
    }


@app.post("/generate")
def generate(req: GenRequest):
    if "t2m" not in _STATE:
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
        "generator_id": f"G2_momask_{NAME}",
        "compute_backend": "docker_gpu" if (torch.cuda.is_available() and GPU_ID >= 0) else "docker_cpu",
        "seed": req.seed,
        "prompt": req.prompt,
        "elapsed_sec": round(time.time() - t0, 3),
    }
