"""
H-2026-004 Stage 1 정량(MPJPE) + 정성(GIF) 통합 평가.

각 윈도(5개)에 대해:
  1. 학습된 LoRA로 prompt → 토큰 생성 (greedy)
  2. 생성 텍스트를 (joint, dx, dy, dz) 시퀀스로 parse
  3. obs 마지막 프레임 절대 좌표를 base로 누적 → 모델 예측 절대 좌표 시퀀스
  4. GT pred_sequence_3d와 동일 방식으로 (델타 → 누적) 산출 → reference 시퀀스
  5. 윈도별 MPJPE 산출 (root-relative 좌표 단위)
  6. 윈도별 GIF (obs blue → pred GT gray vs 모델 orange overlay)
  7. aggregate MPJPE + 5단 GIF index

DEC_DIGITS=3 (X-full-G3) 호환.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Windows cp949 stdout 호환 (em-dash, arrow 등 unicode print)
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


class MotionOnlyLogitsProcessor(LogitsProcessor):
    """추론 시 motion grammar 외 vocab을 -inf로 마스킹해 emission 확률 0으로 강제.

    [W-2026-009](../evals/workarounds/W-2026-009.md) 우회: Both Special Only는
    lm_head 행만 frozen하고 emission 분포는 보장 안 함. 본 processor가 추론 단에서 보완.

    구현 메커니즘:
      - logits ∈ ℝ^V (V=vocab_size)에 대해 i ∉ allowed_ids이면 logits[i] += -inf
      - softmax 후 P(i ∉ allowed) = exp(-inf)/Σ = 0/Σ = 0 (수학적 보장)
      - allowed_ids는 학습 JSONL에 등장한 모든 token id의 합집합 + EOS

    Reference: XGrammar, Outlines, llguidance 등 production constrained decoding과 동일.
    """

    def __init__(self, allowed_ids: set, vocab_size: int):
        super().__init__()
        mask = torch.full((vocab_size,), float("-inf"))
        for tid in allowed_ids:
            mask[tid] = 0.0
        self._mask = mask  # CPU; lazy device move

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._mask.device != scores.device:
            self._mask = self._mask.to(scores.device)
        return scores + self._mask


def build_allowed_ids(jsonl_path: str, tok, include_prompt: bool = True) -> set:
    """학습 JSONL의 모든 record(_meta 제외)에 등장한 token id 합집합 + EOS.

    include_prompt=True: prompt + completion 모두 인코딩 (안전 마진 큼).
                  False: completion만 (motion 출력 grammar에만 정확히 한정).
    """
    allowed = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "_meta" in rec:
                continue
            if include_prompt:
                txt = rec.get("system", "") + "\n" + rec.get("prompt", "") + "\n" + rec.get("completion", "")
            else:
                txt = rec.get("completion", "")
            allowed |= set(tok.encode(txt, add_special_tokens=False))
    if tok.eos_token_id is not None:
        allowed.add(tok.eos_token_id)
    if tok.pad_token_id is not None:
        allowed.add(tok.pad_token_id)
    return allowed


def format_prompt_text(prompt: str) -> str:
    return f"### Instruction ###\n{prompt}\n### End Instruction ###\nAnswer:\n"


def parse_deltas_3d_dec(text: str, marker: str, dec_digits: int) -> Dict[str, List[Tuple[float, float, float]]]:
    """X-full-G3의 DEC_DIGITS=3 등 가변 자릿수 호환 parser."""
    start = text.find(marker)
    if start != -1:
        text = text[start + len(marker):]
    text = text.strip()
    if not text:
        return {}

    tok_pat = rf"-?\[NUM\]\[INT\]\d{{3}}\[SEP\]\[DEC\]\d{{{dec_digits}}}\[ENDNUM\]"
    frame_pat = re.compile(rf"\(\s*({tok_pat})\s*,\s*({tok_pat})\s*,\s*({tok_pat})\s*\)")
    # joint name regex: SPINE1·SPINE2·SPINE3 등 숫자 포함 joint 호환 (이전 [A-Z][A-Z_]*:은 누락)
    joint_pat = re.compile(r"([A-Z][A-Z0-9_]*):")
    decode_pat = re.compile(rf"(-?)\[NUM\]\[INT\](\d{{3}})\[SEP\]\[DEC\](\d{{{dec_digits}}})\[ENDNUM\]")

    def decode(tok: str) -> float:
        m = decode_pat.search(tok)
        if not m:
            raise ValueError(f"bad token: {tok!r}")
        sign = -1 if m.group(1) == "-" else 1
        ip = int(m.group(2))
        dp = int(m.group(3))
        return round(sign * (ip + dp / (10 ** dec_digits)), dec_digits + 1)

    result: Dict[str, List[Tuple[float, float, float]]] = {}
    matches = list(joint_pat.finditer(text))
    for i, jm in enumerate(matches):
        joint = jm.group(1)
        seg_start = jm.end()
        seg_end = matches[i + 1].start() if (i + 1) < len(matches) else len(text)
        seg = text[seg_start:seg_end]
        frames = []
        for x_tok, y_tok, z_tok in frame_pat.findall(seg):
            try:
                frames.append((decode(x_tok), decode(y_tok), decode(z_tok)))
            except ValueError:
                continue
        if frames:
            result[joint] = frames
    return result


def accumulate_to_absolute(
    base_frame: Dict[str, List[float]],
    deltas: Dict[str, List[Tuple[float, float, float]]],
) -> List[Dict[str, List[float]]]:
    """obs 마지막 프레임(base) + joint별 delta 시퀀스 → 프레임별 절대좌표 dict 리스트.
    delta 길이 N → 출력 프레임 N개 (base는 미포함, 첫 출력은 base + delta[0]).
    """
    n_frames = max((len(v) for v in deltas.values()), default=0)
    out_frames: List[Dict[str, List[float]]] = []
    cur: Dict[str, List[float]] = {j: list(map(float, xyz)) for j, xyz in base_frame.items()}
    for fi in range(n_frames):
        new_frame: Dict[str, List[float]] = {}
        for joint, joint_deltas in deltas.items():
            if joint not in cur:
                continue
            if fi >= len(joint_deltas):
                new_frame[joint] = list(cur[joint])
                continue
            dx, dy, dz = joint_deltas[fi]
            cur[joint] = [cur[joint][0] + dx, cur[joint][1] + dy, cur[joint][2] + dz]
            new_frame[joint] = list(cur[joint])
        out_frames.append(new_frame)
    return out_frames


def gt_pred_absolute(gt_pred_seq: List[Dict[str, List[float]]]) -> List[Dict[str, List[float]]]:
    """GT pred_sequence_3d는 이미 절대좌표 — 그대로 반환."""
    return [{j: list(map(float, xyz)) for j, xyz in fr.items()} for fr in gt_pred_seq]


def per_joint_mpjpe(
    gt_frames: List[Dict[str, List[float]]],
    pred_frames: List[Dict[str, List[float]]],
    joints: List[str],
) -> Tuple[float, np.ndarray, np.ndarray]:
    """프레임 정렬 후 joint별 거리 평균 → MPJPE 한 값.
    반환: (mean_mpjpe, per_frame_mpjpe[T], per_joint_mpjpe[J])
    """
    n = min(len(gt_frames), len(pred_frames))
    if n == 0:
        return float("nan"), np.array([]), np.array([])
    err = np.full((n, len(joints)), np.nan, dtype=np.float64)
    for fi in range(n):
        for ji, joint in enumerate(joints):
            if joint in gt_frames[fi] and joint in pred_frames[fi]:
                a = np.array(gt_frames[fi][joint], dtype=np.float64)
                b = np.array(pred_frames[fi][joint], dtype=np.float64)
                err[fi, ji] = float(np.linalg.norm(a - b))
    mean = float(np.nanmean(err))
    per_frame = np.nanmean(err, axis=1)
    per_joint = np.nanmean(err, axis=0)
    return mean, per_frame, per_joint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--gt-json-dir", default="processed_noaug/train")
    p.add_argument("--lora-dir", required=True)
    p.add_argument("--base-model-id", default="google/gemma-4-E4B-it")
    p.add_argument("--tokenizer-dir", default="./gemma4_e4b_tokenizerExtension")
    p.add_argument("--max-new-tokens", type=int, default=8000)
    p.add_argument("--dec-digits", type=int, default=3)
    p.add_argument("--obs-limit", type=int, default=6)
    p.add_argument("--quantization", choices=["none", "nf4"], default="nf4")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--out-dir", default="reports/figures/2026-05-06")
    p.add_argument("--metrics-out", default="evals/raw/h2026004_stage1_mpjpe.json")
    p.add_argument("--use-constrained-decoding", action="store_true",
                   help="W-2026-009 우회 — 추론 시 motion 외 vocab token을 -inf masking. "
                        "학습 데이터의 token id 합집합 + EOS만 emission 허용.")
    args = p.parse_args()

    out_dir = (PROJECT_ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = PROJECT_ROOT / args.metrics_out
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("[FATAL] CUDA required")
        sys.exit(1)

    print(f"[INFO] tokenizer: {args.tokenizer_dir}")
    print(f"[INFO] base model: {args.base_model_id}")
    print(f"[INFO] LoRA: {args.lora_dir}")

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
    print("[INFO] loaded.")

    # W-2026-009 우회: constrained decoding (옵션)
    logits_processor_list = None
    if args.use_constrained_decoding:
        allowed_ids = build_allowed_ids(args.jsonl, tok, include_prompt=True)
        print(f"[INFO] constrained decoding ON: {len(allowed_ids)} tokens allowed (vocab {len(tok)}, "
              f"masked = {len(tok) - len(allowed_ids)})")
        processor = MotionOnlyLogitsProcessor(allowed_ids, vocab_size=len(tok))
        logits_processor_list = LogitsProcessorList([processor])
    else:
        print("[INFO] constrained decoding OFF (W-2026-009 leak 가능)")

    with open(args.jsonl, encoding="utf-8") as f:
        records = []
        meta = None
        for line in f:
            data = json.loads(line)
            if "_meta" in data:
                meta = data["_meta"]
                continue
            records.append(data)
    print(f"[INFO] {len(records)} windows loaded")

    per_window_results = []
    for win_idx, rec in enumerate(records):
        sid = rec.get("sample_id", f"win{win_idx}")
        gt_path = PROJECT_ROOT / args.gt_json_dir / f"{sid}.json"
        if not gt_path.exists():
            print(f"[WARN] gt json not found: {gt_path} — skip")
            continue
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)

        obs_seq_full = gt["obs_sequence_3d"][: args.obs_limit]   # 6 frames
        gt_pred_seq = gt["pred_sequence_3d"]                      # 10 frames (절대좌표)
        base_frame = obs_seq_full[-1]                             # obs 마지막 = pred base

        prompt_text = format_prompt_text(rec["prompt"])
        inputs = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        print(f"\n=== window {win_idx} (sid={sid}) -- generating {args.max_new_tokens} tokens ===")
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
        gen_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
        gen_text = tok.decode(gen_ids, skip_special_tokens=False)

        # parse gen_text & gt completion deltas
        pred_deltas = parse_deltas_3d_dec(gen_text, "Next motion deltas:", args.dec_digits)
        gt_deltas = parse_deltas_3d_dec(rec["completion"], "Next motion deltas:", args.dec_digits)

        # 누적 → 모델 예측 절대좌표 + GT-from-deltas 절대좌표 (sanity: GT-from-deltas should equal gt_pred_seq up to 3-digit rounding)
        pred_abs_frames = accumulate_to_absolute(base_frame, pred_deltas)
        gtd_abs_frames = accumulate_to_absolute(base_frame, gt_deltas)
        gt_abs_frames_real = gt_pred_absolute(gt_pred_seq[: len(pred_abs_frames)])

        n_pred = min(len(pred_abs_frames), len(gt_abs_frames_real))

        # MPJPE: 모델 예측 vs GT (실제 좌표 사용)
        mpjpe_real, per_frame_real, per_joint_real = per_joint_mpjpe(
            gt_abs_frames_real[:n_pred], pred_abs_frames[:n_pred], SMPL_22
        )
        # MPJPE: 모델 예측 vs GT-from-deltas (3-digit 양자화 한계 검증)
        mpjpe_quant, _, _ = per_joint_mpjpe(
            gtd_abs_frames[:n_pred], pred_abs_frames[:n_pred], SMPL_22
        )
        # GT 자체 양자화 손실: GT-from-deltas vs GT-real
        mpjpe_gt_quant, _, _ = per_joint_mpjpe(
            gt_abs_frames_real[:n_pred], gtd_abs_frames[:n_pred], SMPL_22
        )

        # GIF 생성
        obs_arr = frames_dict_to_array(obs_seq_full, joint_filter=None)
        pred_gt_arr = frames_dict_to_array(gt_abs_frames_real[:n_pred], joint_filter=None)
        pred_model_arr = frames_dict_to_array(pred_abs_frames[:n_pred], joint_filter=None)
        gif_path = out_dir / f"h2026004_stage1_w{win_idx}_{sid}.gif"
        plot_3d_motion(
            save_path=str(gif_path),
            obs_arr=obs_arr,
            pred_gt_arr=pred_gt_arr,
            pred_model_arr=pred_model_arr,
            title=(f"H-2026-004 Stage 1 — sid={sid} (window {win_idx}/{len(records)-1})\n"
                   f"obs 6 (blue) → pred 9 (GT gray vs model orange) — MPJPE {mpjpe_real:.4f} (root-relative units)"),
            fps=4,
        )

        win_summary = {
            "window_index": win_idx,
            "sample_id": sid,
            "n_pred_frames": n_pred,
            "joints_in_pred": sorted(pred_deltas.keys()),
            "joints_missing_in_pred": sorted(set(SMPL_22) - set(pred_deltas.keys())),
            "mpjpe_vs_gt_real": float(mpjpe_real),
            "mpjpe_vs_gt_dequantized": float(mpjpe_quant),
            "mpjpe_gt_quantization_loss": float(mpjpe_gt_quant),
            "per_frame_mpjpe": [float(v) for v in per_frame_real.tolist()],
            "per_joint_mpjpe": {j: float(v) for j, v in zip(SMPL_22, per_joint_real.tolist())},
            "gif_path": str(gif_path.relative_to(PROJECT_ROOT)),
        }
        per_window_results.append(win_summary)
        print(f"  joints in pred: {len(pred_deltas)}/{len(SMPL_22)}")
        print(f"  MPJPE vs GT real        : {mpjpe_real:.6f}")
        print(f"  MPJPE vs GT-from-deltas : {mpjpe_quant:.6f}")
        print(f"  GT 3-digit quant loss   : {mpjpe_gt_quant:.6f}  (lower bound for any model)")

    # aggregate
    mpjpes = [w["mpjpe_vs_gt_real"] for w in per_window_results]
    aggregate = {
        "h_id": "H-2026-004",
        "stage": "Stage 1",
        "track": "delta",
        "lora_dir": args.lora_dir,
        "jsonl_path": args.jsonl,
        "jsonl_meta": meta,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_windows": len(per_window_results),
        "mpjpe_mean": float(np.mean(mpjpes)),
        "mpjpe_median": float(np.median(mpjpes)),
        "mpjpe_max": float(np.max(mpjpes)),
        "mpjpe_min": float(np.min(mpjpes)),
        "stage0_threshold_1e-3": float(np.mean(mpjpes)) < 1e-3,
        "stage1_held_out_threshold_5e-2_note": "Stage 1 spec MPJPE < 0.05는 held-out 윈도 — 본 결과는 학습 윈도 reconstruction MPJPE",
        "per_window": per_window_results,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print()
    print("=== aggregate ===")
    print(f"  N windows: {aggregate['n_windows']}")
    print(f"  MPJPE mean   : {aggregate['mpjpe_mean']:.6f}")
    print(f"  MPJPE median : {aggregate['mpjpe_median']:.6f}")
    print(f"  MPJPE min/max: {aggregate['mpjpe_min']:.6f} / {aggregate['mpjpe_max']:.6f}")
    print(f"  saved: {metrics_path}")
    print(f"  GIFs:  {out_dir}/h2026004_stage1_w*.gif")


if __name__ == "__main__":
    main()
