"""MDM inference helper for ArtifactRouter.

Runs inside the external MDM-compatible conda environment. This mirrors the
upstream ``sample.generate`` path but stops before MP4 visualization and writes
only canonical joint positions.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdm-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--n-frames", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--guidance-param", type=float, default=2.5)
    args = parser.parse_args()

    mdm_root = args.mdm_root.resolve()
    os.chdir(mdm_root)
    sys.path.insert(0, str(mdm_root))

    from data_loaders.get_data import get_dataset_loader
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from data_loaders.tensors import collate
    from utils import dist_util
    from utils.fixseed import fixseed
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils.parser_util import generate_args
    from utils.sampler_util import ClassifierFreeSampleModel

    fixseed(args.seed)

    # Reuse MDM's parser so args.json next to the checkpoint is loaded exactly
    # as upstream expects.
    old_argv = sys.argv[:]
    motion_length_sec = max(float(args.n_frames) / 20.0, 0.2)
    sys.argv = [
        "artifactrouter_mdm_inference",
        "--model_path",
        str(args.model_path),
        "--text_prompt",
        args.prompt,
        "--motion_length",
        f"{motion_length_sec:.4f}",
        "--num_samples",
        "1",
        "--num_repetitions",
        "1",
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--guidance_param",
        str(args.guidance_param),
    ]
    try:
        mdm_args = generate_args()
    finally:
        sys.argv = old_argv

    max_frames = 196 if mdm_args.dataset in ["kit", "humanml"] else 60
    fps = 12.5 if mdm_args.dataset == "kit" else 20
    n_frames = min(max_frames, int(mdm_args.motion_length * fps))
    dist_util.setup_dist(mdm_args.device)

    print("Loading dataset...")
    data = get_dataset_loader(
        name=mdm_args.dataset,
        batch_size=1,
        num_frames=max_frames,
        split="test",
        hml_mode="text_only",
        fixed_len=0,
        pred_len=0,
        device=dist_util.dev(),
    )
    data.fixed_length = n_frames

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(mdm_args, data)
    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, str(args.model_path), use_avg=mdm_args.use_ema)

    if mdm_args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    collate_args = [
        {"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames, "text": args.prompt}
    ]
    _, model_kwargs = collate(collate_args)
    model_kwargs["y"] = {
        key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
        for key, val in model_kwargs["y"].items()
    }
    if mdm_args.guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(1, device=dist_util.dev()) * mdm_args.guidance_param
    if "text" in model_kwargs["y"]:
        model_kwargs["y"]["text_embed"] = model.encode_text(model_kwargs["y"]["text"])

    sample = diffusion.p_sample_loop(
        model,
        (1, model.njoints, model.nfeats, n_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )

    if model.data_rep == "hml_vec":
        n_joints = 22 if sample.shape[1] == 263 else 21
        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        sample = recover_from_ric(sample, n_joints)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

    rot2xyz_pose_rep = "xyz" if model.data_rep in ["xyz", "hml_vec"] else model.data_rep
    sample = model.rot2xyz(
        x=sample,
        mask=None if rot2xyz_pose_rep == "xyz" else model_kwargs["y"]["mask"].reshape(1, n_frames).bool(),
        pose_rep=rot2xyz_pose_rep,
        glob=True,
        translation=True,
        jointstype="smpl",
        vertstrans=True,
        betas=None,
        beta=0,
        glob_rot=None,
        get_rotations_back=False,
    )

    # Upstream shape after rot2xyz: [B, J, 3, T]. Convert to [T, J, 3].
    motion = sample[0].detach().cpu().numpy().transpose(2, 0, 1).astype(np.float64)
    motion = motion[: args.n_frames]
    motion = motion - motion[:, 0:1, :]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, motion)


if __name__ == "__main__":
    main()
