"""Local LoRA generator wrapper — 이전 저장소의 LLM motion experiment 자산 wrapper.

⚠️ OUT-OF-SCOPE FOR THIS PROJECT (2026-05-15) ⚠️
================================================
본 wrapper 는 **본 프로젝트 (ArtifactRouter) 의 active scope 가 아니다**.

본 저장소는 motion refinement framework 의 독립 프로젝트이며, 이전 저장소
(`3D-Motion-Trajectory-prediction`) 의 LLM-based motion generation 실험
(이전 저장소 가설 H-2026-101·102) 의 후속 연구가 아니다. 본 wrapper 는
초기 scaffold 단계에서 inadvertent inclusion 된 산출물이며, 코드 자체는
참고 자료로 보존한다.

본 프로젝트의 active generator scope: G1 (diffusion-based: MDM / MLD) +
G2 (token-based: 공식 MotionGPT). 자세한 내용은 [AGENTS.md §1](../AGENTS.md).

본 wrapper 를 active 평가·실험에 import·실행하지 마라. 후일 별도 연구
(LLM motion generation) 를 본 저장소에서 시작할 경우 새 가설 등록 +
새 wrapper 작성으로 진행한다.
================================================

(아래는 본 wrapper 가 작성된 시점의 원래 의도 — 참고용 보존)

이전 저장소 의 학습된 LoRA adapter (`external_assets/local_lora_g3/`,
Gemma 4 E4B-it + NF4 + Both Special Only, sid=002989 단일 sample × 5 windows
× 50 epochs 학습) 를 wrap 한다.

이전 저장소의 sanity check 결과:
  - short-horizon (≤1.0s) bit-exact reconstruction.
  - 1.5s+ autoregressive drift.
  - chain 5+ format collapse.

본 wrapper 는 이전 저장소의 `tools/stage1_autoregressive_chain.py` 로직을 import 해
prompt-free autoregressive generation 으로 motion 을 생성하고, generator output 을
canonical SMPL 22-joint, [T, 22, 3], root-relative format 으로 반환한다.

CLI:
    python -m generators.local_lora_wrapper --prompt "walking" --n-frames 40

명세 §6.1 Base Motion Generator 참조.
AGENTS.md §3-1 Canonical Motion Format 의무.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

from generators.base import Generator, GeneratorOutput
from skeleton_normalizer.canonical_smpl_22 import SMPL_22, NAME_TO_IDX, DEFAULT_FPS

REPO_ROOT = Path(__file__).resolve().parents[1]


class LocalLoRA_G3(Generator):
    """G3 generator — 이전 저장소의 LoRA adapter wrapper.

    Quality-tier: G3 (legacy artifact-rich). 명세 §9.2 quality-tier coverage 의무.

    Note: 본 wrapper 는 PEFT + transformers + bitsandbytes 가 필요하다.
    CUDA 가용 환경에서만 작동 (NF4 양자화).
    """

    name = "LocalLoRA_G3"

    def __init__(
        self,
        lora_dir: Path = REPO_ROOT / "external_assets" / "local_lora_g3",
        base_model_id: str = "google/gemma-4-E4B-it",
        tokenizer_dir: Optional[Path] = None,
        max_new_tokens: int = 8000,
        dec_digits: int = 3,
        obs_limit: int = 6,
        pred_frames_per_chain: int = 9,
    ) -> None:
        """
        Args:
            lora_dir: 이전 저장소의 LoRA adapter directory.
            base_model_id: Gemma 4 base 모델 (HF Hub).
            tokenizer_dir: 토크나이저 확장본 경로 (없으면 lora_dir 사용).
            max_new_tokens: generation 최대 token.
            dec_digits: 좌표 token 의 decimal 자릿수 (이전 저장소 X-full-G3 = 3).
            obs_limit: 초기 obs 프레임 수.
            pred_frames_per_chain: chain 당 prediction frame.
        """
        if not lora_dir.exists():
            raise FileNotFoundError(
                f"LoRA directory not found: {lora_dir}\n"
                "external_assets/ junction이 올바르게 설정되어 있는지 확인하세요. "
                "AGENTS.md §2-1 환경 준비 참조."
            )

        self.lora_dir = lora_dir
        self.base_model_id = base_model_id
        self.tokenizer_dir = tokenizer_dir or (
            REPO_ROOT.parent / "3D-Motion-Trajectory-prediction" / "gemma4_e4b_tokenizerExtension"
        )
        self.max_new_tokens = max_new_tokens
        self.dec_digits = dec_digits
        self.obs_limit = obs_limit
        self.pred_frames_per_chain = pred_frames_per_chain
        self._model = None
        self._tokenizer = None

    def quality_tier(self) -> str:
        return "G3"

    def _ensure_model_loaded(self) -> None:
        """Lazy load — Phase 1 MVP에서 구현."""
        if self._model is not None:
            return
        # TODO Phase 1 MVP — external_assets/code/ 의 import 경유로 모델 로드.
        # from external_assets.code.stage1_mpjpe_and_gif import MotionOnlyLogitsProcessor, ...
        # NF4 + LoRA + tokenizer 로드 (이전 저장소 tools/stage1_autoregressive_chain.py 패턴).
        raise NotImplementedError(
            "Phase 1 MVP — external_assets/code/ import 후 NF4+LoRA 로드 구현 예정. "
            "이전 저장소의 tools/stage1_autoregressive_chain.py 로딩 로직 참조."
        )

    def generate(
        self,
        prompt: Optional[str] = None,
        n_frames: int = 40,
        seed: int = 42,
        **kwargs: Any,
    ) -> GeneratorOutput:
        """Motion 생성 (autoregressive chaining).

        Args:
            prompt: caption (현재 G3 wrapper에서는 미사용 — 이전 저장소 학습은 motion-only).
            n_frames: target frame 수. 이전 저장소 학습이 obs+pred=16 frames per chain이므로
                     n_frames > 16 이면 autoregressive chaining (chain N = ceil((n_frames - obs_limit) / pred_frames_per_chain)).
            seed: random seed.
            **kwargs: 추가 인자 (구현 시 사용).

        Returns:
            GeneratorOutput — [n_frames, 22, 3] canonical motion + metadata.
        """
        self._ensure_model_loaded()
        # TODO Phase 1 MVP — autoregressive chaining 구현.
        # 1. 초기 obs frames (random 또는 HumanML3D에서 sample).
        # 2. obs deltas encoding (이전 저장소 encode_obs_deltas_text).
        # 3. constrained decoding (MotionOnlyLogitsProcessor).
        # 4. parse_deltas_3d_dec + accumulate_to_absolute.
        # 5. chain N회 반복해 n_frames 도달.
        # 6. canonical SMPL 22 format으로 반환.
        raise NotImplementedError(
            "Phase 1 MVP — autoregressive chaining 구현 예정. "
            "이전 저장소 tools/stage1_autoregressive_chain.py 로직 import."
        )

    def _generator_class_hash(self) -> str:
        """본 wrapper 코드 + LoRA model_card.json hash 결합."""
        import inspect

        wrapper_source = inspect.getsource(type(self))
        wrapper_hash = hashlib.sha256(wrapper_source.encode("utf-8")).hexdigest()
        model_card_path = self.lora_dir / "model_card.json"
        if model_card_path.exists():
            with open(model_card_path, "rb") as f:
                lora_hash = hashlib.sha256(f.read()).hexdigest()
        else:
            lora_hash = "missing"
        return f"{wrapper_hash[:16]}_{lora_hash[:16]}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LocalLoRA G3 generator (이전 저장소의 LoRA wrapper)"
    )
    parser.add_argument("--prompt", type=str, default=None, help="text caption (G3는 미사용)")
    parser.add_argument("--n-frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lora-dir",
        type=Path,
        default=REPO_ROOT / "external_assets" / "local_lora_g3",
    )
    args = parser.parse_args()

    print(f"[INFO] G3 generator — lora_dir: {args.lora_dir}")
    if not args.lora_dir.exists():
        print(f"[ERROR] external_assets/local_lora_g3 가 없음. AGENTS.md §2-1 환경 준비 참조.")
        sys.exit(1)

    print(f"[INFO] generator quality-tier: G3 (legacy artifact-rich)")
    print(f"[INFO] target n_frames: {args.n_frames}")
    print(f"[INFO] seed: {args.seed}")

    gen = LocalLoRA_G3(lora_dir=args.lora_dir)
    print(f"[INFO] generator_class_hash: {gen._generator_class_hash()}")

    try:
        out = gen.generate(prompt=args.prompt, n_frames=args.n_frames, seed=args.seed)
        print(f"[OK] motion shape: {out.motion.shape}")
        print(f"[OK] generator_id: {out.generator_id}")
    except NotImplementedError as e:
        print(f"[STUB] generation not yet implemented (Phase 1 MVP): {e}")


if __name__ == "__main__":
    main()
