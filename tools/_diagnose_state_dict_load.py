"""W-2026-001 진단 B — load_state_dict missing/unexpected key + embedding shape.

사용자 directive 의 B 단계 진단:
  - language_model.shared.weight / encoder.embed_tokens.weight / decoder.embed_tokens.weight
    / lm_head.weight 의 shape
  - motion token 확장 후 vocab size
  - checkpoint 안의 해당 tensor shape
  - load 후 row 가 checkpoint 값으로 정상 들어갔는지

본 script 는 mgpt-clean env 안에서 실행되어야 한다 (transformers 5.8.1 + pl 2.6.1).

CLI:
    python _diagnose_state_dict_load.py \\
        <motiongpt_root> <cfg_path> <ckpt_path>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    if len(sys.argv) < 4:
        print("usage: python _diagnose_state_dict_load.py <motiongpt_root> <cfg> <ckpt>", file=sys.stderr)
        sys.exit(2)

    motiongpt_root = Path(sys.argv[1]).resolve()
    cfg_path = sys.argv[2]
    ckpt_path = Path(sys.argv[3]).resolve()

    sys.path.insert(0, str(motiongpt_root))
    os.chdir(motiongpt_root)

    import torch
    import pytorch_lightning as pl

    sys.argv = ["_diagnose", "--cfg", cfg_path]
    from mGPT.config import parse_args as mgpt_parse_args
    cfg = mgpt_parse_args(phase="demo")
    cfg.TEST.CHECKPOINTS = str(ckpt_path)
    cfg.SEED_VALUE = 42
    pl.seed_everything(42)

    from mGPT.data.build_data import build_data
    from mGPT.models.build_model import build_model

    print("=" * 80)
    print("[Step 1] Build datamodule + model")
    datamodule = build_data(cfg)
    model = build_model(cfg, datamodule)

    EMBED_PATTERNS = ("shared.weight", "embed_tokens", "lm_head")

    # ---- Before load_state_dict: record model param shapes + statistics ----
    print("\n" + "=" * 80)
    print("[Step 2] BEFORE load_state_dict — model parameter shapes / statistics:")
    before_state = {}
    embed_lm_param_names = []
    for name, param in model.named_parameters():
        if any(k in name for k in EMBED_PATTERNS):
            embed_lm_param_names.append(name)
            before_state[name] = param.data.detach().clone()
            print(f"  {name:60s}: shape={tuple(param.shape)} mean={param.mean().item():+.5f} std={param.std().item():.5f}")

    # ---- Inspect checkpoint state_dict keys ----
    print("\n" + "=" * 80)
    print(f"[Step 3] Load checkpoint: {ckpt_path}")
    ckpt_obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj
    print(f"  Total state_dict keys: {len(state_dict)}")
    print(f"  Keys matching {EMBED_PATTERNS}:")
    for k in sorted(state_dict.keys()):
        if any(s in k for s in EMBED_PATTERNS):
            v = state_dict[k]
            shp = tuple(v.shape) if hasattr(v, "shape") else "N/A"
            stats = ""
            if hasattr(v, "mean"):
                stats = f" mean={v.mean().item():+.5f} std={v.std().item():.5f}"
            print(f"    {k:60s}: shape={shp}{stats}")

    # ---- load_state_dict(strict=False) ----
    print("\n" + "=" * 80)
    print("[Step 4] model.load_state_dict(strict=False) + set-based key diff")
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing_keys = sorted(model_keys - ckpt_keys)
    unexpected_keys = sorted(ckpt_keys - model_keys)
    print(f"  Model state_dict keys: {len(model_keys)}")
    print(f"  Ckpt state_dict keys:  {len(ckpt_keys)}")
    print(f"  missing_keys (model needs, ckpt has not): {len(missing_keys)}")
    for k in missing_keys[:50]:
        emoji = " <-- EMBED/LM" if any(s in k for s in EMBED_PATTERNS) else ""
        print(f"    MISSING:    {k}{emoji}")
    if len(missing_keys) > 50:
        print(f"    ... ({len(missing_keys) - 50} more)")
    print(f"  unexpected_keys (ckpt has, model not): {len(unexpected_keys)}")
    for k in unexpected_keys[:50]:
        emoji = " <-- EMBED/LM" if any(s in k for s in EMBED_PATTERNS) else ""
        print(f"    UNEXPECTED: {k}{emoji}")
    if len(unexpected_keys) > 50:
        print(f"    ... ({len(unexpected_keys) - 50} more)")

    # Also try strict=False load and catch return.
    result = model.load_state_dict(state_dict, strict=False)
    if result is not None and hasattr(result, "missing_keys"):
        print(f"\n  load_state_dict result.missing_keys: {len(result.missing_keys)}")
        print(f"  load_state_dict result.unexpected_keys: {len(result.unexpected_keys)}")
    else:
        print(f"\n  load_state_dict returned: {result} (PyTorch Lightning module overrides return — use set-based diff above)")

    # ---- After load_state_dict ----
    print("\n" + "=" * 80)
    print("[Step 5] AFTER load_state_dict — embed/lm changed or unchanged?")
    for name in embed_lm_param_names:
        param = dict(model.named_parameters())[name]
        before = before_state[name]
        diff = (param.data - before).abs().mean().item()
        changed = "CHANGED" if diff > 1e-7 else "*** UNCHANGED ***"
        print(f"  {name:60s}: shape={tuple(param.shape)} mean_now={param.mean().item():+.5f} std_now={param.std().item():.5f} mean_diff={diff:.6f} [{changed}]")

    # ---- Tokenizer / vocab size ----
    print("\n" + "=" * 80)
    print("[Step 6] Tokenizer / motion vocab inspection")
    # Try common attribute paths.
    tokenizer = None
    for path in [
        ("lm", "tokenizer"),
        ("language_model", "tokenizer"),
        ("lm", "lm_head", "tokenizer"),
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            tokenizer = obj
            print(f"  tokenizer found at: model.{'.'.join(path)}")
            break
        except AttributeError:
            continue
    if tokenizer is None:
        # search modules for tokenizer attribute.
        for mod_name, mod in model.named_modules():
            if hasattr(mod, "tokenizer"):
                tokenizer = mod.tokenizer
                print(f"  tokenizer found at: model.{mod_name}.tokenizer")
                break
    if tokenizer is None:
        print("  tokenizer NOT FOUND on model — skipping vocab inspection")
    else:
        try:
            print(f"  tokenizer class: {type(tokenizer).__name__}")
            print(f"  vocab_size: {tokenizer.vocab_size}")
            vocab = tokenizer.get_vocab()
            print(f"  len(get_vocab()): {len(vocab)}")
            motion_tokens = sorted([t for t in vocab.keys() if "motion_id" in t.lower()],
                                   key=lambda s: vocab[s])
            print(f"  motion-* token count: {len(motion_tokens)}")
            if motion_tokens:
                print(f"  first 3: {motion_tokens[:3]}")
                print(f"  last 3:  {motion_tokens[-3:]}")
                print(f"  motion_id_0 token ID: {vocab.get('<motion_id_0>', 'NOT_FOUND')}")
                print(f"  motion_id_513 token ID: {vocab.get('<motion_id_513>', 'NOT_FOUND')}")
        except Exception as e:
            print(f"  tokenizer inspection failed: {e}")

    # ---- LM head / embedding consistency check ----
    print("\n" + "=" * 80)
    print("[Step 7] LM head shape vs tokenizer vocab consistency")
    try:
        if tokenizer is not None:
            tok_vocab = tokenizer.vocab_size
            for name, param in model.named_parameters():
                if name.endswith("lm_head.weight") or name.endswith("shared.weight"):
                    matches = "✓ MATCH" if param.shape[0] == tok_vocab else f"✗ MISMATCH (param={param.shape[0]} vs tok={tok_vocab})"
                    print(f"  {name}: shape={tuple(param.shape)} {matches}")
    except Exception as e:
        print(f"  consistency check failed: {e}")

    print("\n" + "=" * 80)
    print("[DONE] Diagnosis complete.")


if __name__ == "__main__":
    main()
