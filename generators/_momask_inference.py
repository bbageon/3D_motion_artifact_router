"""MoMask inference helper for ArtifactRouter.

This script runs inside the external ``momask`` conda environment. It avoids the
upstream BVH/MP4 visualization path and writes only canonical joint positions.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--momask-root", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--n-frames", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--dataset-name", type=str, default="t2m")
    parser.add_argument("--name", type=str, default="t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns")
    parser.add_argument("--res-name", type=str, default="tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw")
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints")
    parser.add_argument("--cond-scale", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topkr", type=float, default=0.9)
    parser.add_argument("--time-steps", type=int, default=18)
    args = parser.parse_args()

    momask_root = args.momask_root.resolve()
    os.chdir(momask_root)
    sys.path.insert(0, str(momask_root))

    from gen_t2m import load_len_estimator, load_res_model, load_trans_model, load_vq_model
    from utils.fixseed import fixseed
    from utils.get_opt import get_opt
    from utils.motion_process import recover_from_ric

    fixseed(args.seed)
    device = torch.device("cpu" if args.gpu_id == -1 else f"cuda:{args.gpu_id}")
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)

    dim_pose = 251 if args.dataset_name == "kit" else 263
    root_dir = Path(args.checkpoints_dir) / args.dataset_name / args.name
    model_opt = get_opt(str(root_dir / "opt.txt"), device=device)

    vq_opt_path = Path(args.checkpoints_dir) / args.dataset_name / model_opt.vq_name / "opt.txt"
    vq_opt = get_opt(str(vq_opt_path), device=device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt_path = Path(args.checkpoints_dir) / args.dataset_name / args.res_name / "opt.txt"
    res_opt = get_opt(str(res_opt_path), device=device)

    opt = SimpleNamespace(
        name=args.name,
        device=device,
        cond_scale=args.cond_scale,
        temperature=args.temperature,
        topkr=args.topkr,
        time_steps=args.time_steps,
        gumbel_sample=False,
    )
    res_model = load_res_model(res_opt, vq_opt, opt)
    t2m_transformer = load_trans_model(model_opt, opt, "latest.tar")
    length_estimator = load_len_estimator(model_opt)

    t2m_transformer.eval().to(device)
    vq_model.eval().to(device)
    res_model.eval().to(device)
    length_estimator.eval().to(device)

    mean = np.load(Path(args.checkpoints_dir) / args.dataset_name / model_opt.vq_name / "meta" / "mean.npy")
    std = np.load(Path(args.checkpoints_dir) / args.dataset_name / model_opt.vq_name / "meta" / "std.npy")

    captions = [args.prompt]
    m_length = torch.LongTensor([int(args.n_frames)]).to(device)
    token_lens = torch.clamp(m_length // 4, min=1)
    m_length = token_lens * 4

    # If the caller passes n_frames <= 0, estimate length from text.
    if args.n_frames <= 0:
        text_embedding = t2m_transformer.encode_text(captions)
        pred_dis = length_estimator(text_embedding)
        token_lens = Categorical(F.softmax(pred_dis, dim=-1)).sample()
        m_length = token_lens * 4

    with torch.no_grad():
        mids = t2m_transformer.generate(
            captions,
            token_lens,
            timesteps=args.time_steps,
            cond_scale=args.cond_scale,
            temperature=args.temperature,
            topk_filter_thres=args.topkr,
            gsample=False,
        )
        mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
        pred_motions = vq_model.forward_decoder(mids).detach().cpu().numpy()
        data = pred_motions * std + mean

    joint_data = data[0][: int(m_length[0].item())]
    joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy().astype(np.float64)
    joint = joint - joint[:, 0:1, :]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, joint)


if __name__ == "__main__":
    main()
