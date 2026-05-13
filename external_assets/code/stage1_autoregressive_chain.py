"""
H-2026-004 Stage 1 autoregressive chaining (다단 추론).

학습된 LoRA 모델의 출력을 다음 step의 obs로 feed-back해 horizon을 임의로 확장.
W-2026-009 resolved 후이므로 constrained decoding 기본 ON.

Workflow per chain:
  obs_frames(6) → 5 obs deltas → prompt → model.generate (constrained)
                                              ↓
                                       9 pred deltas
                                              ↓
                                       accumulate from obs_frames[-1]
                                              ↓
                                       9 new absolute frames
                                              ↓
                                       new obs_frames = last 6 of (obs+pred)
                                              ↓
                                       (next chain)

산출:
  - reports/figures/2026-05-10/h2026004_stage1_autoregressive_<N>chain.gif
  - evals/raw/h2026004_stage1_autoregressive.json

GIF에 horizon-wise drift 가 시각적으로 드러남:
  - obs (blue) + chained pred (orange) vs GT (gray)
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from plot_3d_motion import plot_3d_motion, frames_dict_to_array, SMPL_22  # noqa: E402
from stage1_mpjpe_and_gif import (  # noqa: E402
    parse_deltas_3d_dec,
    accumulate_to_absolute,
    per_joint_mpjpe,
    MotionOnlyLogitsProcessor,
    build_allowed_ids,
)


# ── token encoder (DEC_DIGITS=3 호환) ─────────────────────────────
def num_to_tokens_dec(x: float, dec_digits: int) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    int_part = int(x)
    dec_scale = 10 ** dec_digits
    dec_part = int(round((x - int_part) * dec_scale))
    if dec_part >= dec_scale:
        int_part += 1
        dec_part -= dec_scale
    return f"{sign}[NUM][INT]{int_part:03d}[SEP][DEC]{dec_part:0{dec_digits}d}[ENDNUM]"


def encode_obs_deltas_text(frames: List[Dict[str, List[float]]],
                            joint_order: List[str],
                            dec_digits: int) -> str:
    """frames(N+1) → 'JOINT:(tok,tok,tok),(tok,tok,tok),... | JOINT:...' (N deltas per joint)."""
    parts = []
    for joint in joint_order:
        if joint not in frames[0]:
            continue
        coord_strs = []
        for i in range(1, len(frames)):
            x1, y1, z1 = frames[i - 1][joint]
            x2, y2, z2 = frames[i][joint]
            dx = round(x2 - x1, dec_digits)
            dy = round(y2 - y1, dec_digits)
            dz = round(z2 - z1, dec_digits)
            coord_strs.append(
                f"({num_to_tokens_dec(dx, dec_digits)},{num_to_tokens_dec(dy, dec_digits)},{num_to_tokens_dec(dz, dec_digits)})"
            )
        parts.append(f"{joint}:{','.join(coord_strs)}")
    return " | ".join(parts)


def format_prompt_text(prompt: str) -> str:
    return f"### Instruction ###\n{prompt}\n### End Instruction ###\nAnswer:\n"


def build_long_gt_from_windows(window_files: List[str]) -> List[Dict[str, List[float]]]:
    """5 윈도(stride=5)에서 unique source frame 0-49를 stitching.

    w0000.obs[0:20]: source 0-19
    w0000.pred[0:10]: source 20-29
    w_i.pred[5:10] for i=1..4: source 30-49 (5 frames each)
    """
    long_frames: List[Dict[str, List[float]]] = []
    if not window_files:
        return long_frames
    with open(window_files[0], encoding="utf-8") as f:
        d = json.load(f)
    long_frames.extend(d["obs_sequence_3d"])  # 20
    long_frames.extend(d["pred_sequence_3d"])  # 10 → total 30
    for fp in window_files[1:]:
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        long_frames.extend(d["pred_sequence_3d"][5:10])  # 5 each
    return long_frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-windows-glob", default="processed_noaug/train/002989_w*.json")
    p.add_argument("--prompt-jsonl", default="finetune_3d_delta_h2026004_singlesample_002989_stage1.jsonl",
                   help="prompt 헤더 형식 추출용 reference JSONL")
    p.add_argument("--lora-dir", default="lora_3d_delta_h2026004_singlesample_002989_stage1_seed42")
    p.add_argument("--base-model-id", default="google/gemma-4-E4B-it")
    p.add_argument("--tokenizer-dir", default="./gemma4_e4b_tokenizerExtension")
    p.add_argument("--num-chains", type=int, default=4)
    p.add_argument("--obs-limit", type=int, default=6)
    p.add_argument("--pred-frames-per-chain", type=int, default=9,
                   help="학습된 모델이 1회 generation에서 산출하는 pred delta 프레임 수")
    p.add_argument("--dec-digits", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=8000)
    p.add_argument("--quantization", choices=["none", "nf4"], default="nf4")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--use-constrained-decoding", action="store_true", default=True)
    p.add_argument("--out-dir", default="reports/figures/2026-05-10")
    p.add_argument("--metrics-out", default="evals/raw/h2026004_stage1_autoregressive.json")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("[FATAL] CUDA required")
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = {"device_map": "auto"}
    if args.quantization == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    print("[INFO] loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, attn_implementation=args.attn_implementation, **model_kwargs
    )
    base.resize_token_embeddings(len(tok))
    print("[INFO] loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()

    # constrained decoding (default on)
    logits_processor_list = None
    if args.use_constrained_decoding:
        allowed_ids = build_allowed_ids(args.prompt_jsonl, tok, include_prompt=True)
        print(f"[INFO] constrained decoding ON: {len(allowed_ids)}/{len(tok)} tokens allowed")
        processor = MotionOnlyLogitsProcessor(allowed_ids, vocab_size=len(tok))
        logits_processor_list = LogitsProcessorList([processor])
    else:
        print("[INFO] constrained decoding OFF (W-2026-009 leak 가능)")

    # long GT (50 frames covering source 0-49)
    window_files = sorted(glob.glob(str(PROJECT_ROOT / args.source_windows_glob)))
    long_gt_frames = build_long_gt_from_windows(window_files)
    print(f"[INFO] long GT frames: {len(long_gt_frames)}")
    n_required = args.obs_limit + args.num_chains * args.pred_frames_per_chain
    if n_required > len(long_gt_frames):
        print(f"[WARN] required {n_required} GT frames but only {len(long_gt_frames)} available — "
              f"chaining will exceed GT comparison range")

    # extract prompt header from reference JSONL
    with open(PROJECT_ROOT / args.prompt_jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "_meta" in rec:
                continue
            ref_prompt = rec["prompt"]
            break
    marker = "Observed motion deltas:"
    idx = ref_prompt.find(marker)
    if idx == -1:
        print(f"[FATAL] '{marker}' not found in reference prompt")
        sys.exit(1)
    prompt_header = ref_prompt[:idx]  # ends with newline
    print(f"[INFO] prompt header ({idx} chars):")
    print("  " + prompt_header.replace("\n", "\n  ").rstrip())

    # initial obs (6 absolute frames from long GT)
    obs_frames = [
        {j: list(map(float, xyz)) for j, xyz in fr.items()}
        for fr in long_gt_frames[: args.obs_limit]
    ]

    all_predicted_frames: List[Dict[str, List[float]]] = []
    per_chain_results = []

    for chain_idx in range(args.num_chains):
        # encode 5 obs deltas from 6 frames
        obs_text = encode_obs_deltas_text(obs_frames, SMPL_22, args.dec_digits)
        body = f"Observed motion deltas: {obs_text}"
        full_prompt = prompt_header + body
        prompt_text = format_prompt_text(full_prompt)

        inputs = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_token_len = inputs["input_ids"].shape[1]

        print(f"\n=== chain {chain_idx + 1}/{args.num_chains} -- prompt tokens: {prompt_token_len} ===")

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.pad_token_id,
        )
        if logits_processor_list is not None:
            gen_kwargs["logits_processor"] = logits_processor_list

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        gen_ids = out[0][prompt_token_len:].tolist()
        gen_text = tok.decode(gen_ids, skip_special_tokens=False)

        pred_deltas = parse_deltas_3d_dec(gen_text, "Next motion deltas:", args.dec_digits)
        if not pred_deltas:
            print(f"[ERROR] chain {chain_idx + 1} parse failed")
            break
        n_frames_per_joint = max(len(v) for v in pred_deltas.values())
        n_joints = len(pred_deltas)

        # accumulate from last obs frame
        base_frame = obs_frames[-1]
        new_pred_frames_all = accumulate_to_absolute(base_frame, pred_deltas)
        new_pred_frames = new_pred_frames_all[: args.pred_frames_per_chain]

        # MPJPE for this chain vs corresponding GT slice
        gt_start = args.obs_limit + chain_idx * args.pred_frames_per_chain
        gt_end = gt_start + len(new_pred_frames)
        if gt_end <= len(long_gt_frames):
            gt_for_this = long_gt_frames[gt_start:gt_end]
            mpjpe_chain, per_frame_mpjpe, _ = per_joint_mpjpe(gt_for_this, new_pred_frames, SMPL_22)
        else:
            mpjpe_chain = float("nan")
            per_frame_mpjpe = np.array([])

        print(f"  joints in pred: {n_joints}/{len(SMPL_22)}, frames/joint: {n_frames_per_joint}")
        print(f"  chain {chain_idx + 1} MPJPE vs GT-real: {mpjpe_chain:.6f}")

        per_chain_results.append({
            "chain_index": chain_idx,
            "n_frames": len(new_pred_frames),
            "joints_count": n_joints,
            "mpjpe_vs_gt_real": float(mpjpe_chain) if not np.isnan(mpjpe_chain) else None,
            "per_frame_mpjpe": [float(v) for v in per_frame_mpjpe.tolist()],
            "gt_range": [gt_start, gt_end] if gt_end <= len(long_gt_frames) else None,
        })

        all_predicted_frames.extend(new_pred_frames)
        # update obs: last 6 frames of (obs_frames + new_pred_frames)
        combined = obs_frames + new_pred_frames
        obs_frames = combined[-args.obs_limit:]

    # final visualization
    initial_obs_arr = frames_dict_to_array(long_gt_frames[: args.obs_limit])
    n_pred_total = len(all_predicted_frames)
    pred_model_arr = frames_dict_to_array(all_predicted_frames)
    gt_compare_end = min(args.obs_limit + n_pred_total, len(long_gt_frames))
    pred_gt_arr = frames_dict_to_array(long_gt_frames[args.obs_limit: gt_compare_end])

    # truncate pred_model_arr to GT range for fair comparison in plot
    pred_model_arr = pred_model_arr[: pred_gt_arr.shape[0]]

    duration = n_pred_total / 20.0
    gif_path = out_dir / f"h2026004_stage1_autoregressive_{args.num_chains}chain.gif"
    plot_3d_motion(
        save_path=str(gif_path),
        obs_arr=initial_obs_arr,
        pred_gt_arr=pred_gt_arr,
        pred_model_arr=pred_model_arr,
        title=(f"H-2026-004 Stage 1 autoregressive chaining ({args.num_chains}-chain)\n"
               f"sid=002989 — obs 6 (blue, 0.3s) -> pred {n_pred_total} ({duration:.2f}s)\n"
               f"GT gray vs model orange — chain MPJPEs see metrics"),
        fps=4,
    )

    aggregate = {
        "schema_version": "v1",
        "h_id": "H-2026-004",
        "stage": "Stage 1 autoregressive chaining",
        "track": "delta",
        "lora_dir": args.lora_dir,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_chains": args.num_chains,
        "obs_limit": args.obs_limit,
        "pred_frames_per_chain": args.pred_frames_per_chain,
        "dec_digits": args.dec_digits,
        "use_constrained_decoding": args.use_constrained_decoding,
        "total_predicted_frames": n_pred_total,
        "duration_seconds": duration,
        "fps": 20,
        "per_chain": per_chain_results,
        "mpjpe_per_chain": [r["mpjpe_vs_gt_real"] for r in per_chain_results],
        "horizon_wise_drift_check_note": "값이 chain index 따라 단조 증가하면 drift 누적, "
                                          "일정 범위에 머물면 안정. 첫 chain이 stage1 baseline reconstruction과 같아야 함 (sanity).",
        "gif_path": str(gif_path.relative_to(PROJECT_ROOT)),
    }
    metrics_path = PROJECT_ROOT / args.metrics_out
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print()
    print(f"=== aggregate ({args.num_chains} chains) ===")
    print(f"  total predicted frames: {n_pred_total} ({duration:.2f}s)")
    print(f"  per-chain MPJPE: {[f'{m:.4f}' if m is not None else 'NaN' for m in aggregate['mpjpe_per_chain']]}")
    print(f"  GIF: {gif_path}")
    print(f"  metrics: {metrics_path}")


if __name__ == "__main__":
    main()
