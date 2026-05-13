"""Generator base interface.

모든 generator wrapper(local_lora, motiongpt, t2m_gpt, mdm 등)는 본 인터페이스를 구현한다.
출력은 canonical SMPL 22-joint, [T, J=22, 3], fps=20, root-relative format.

명세 §6.1 Base Motion Generator 참조.
AGENTS.md §3-1 Canonical Motion Format 의무.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class GeneratorOutput:
    """Generator의 표준 출력 형식.

    Attributes:
        motion: [T, 22, 3] numpy float64 array. root-relative SMPL 22-joint.
        fps: frames per second (default 20).
        prompt: text caption (있는 경우).
        generator_id: G1/G2/G3 분류 + version 식별자 (예: 'G3_local_lora_h2026004').
        generator_class_hash: generator wrapper 코드의 sha256.
        seed: 사용 seed.
        metadata: 추가 metadata (generator 별 다름).
    """

    motion: np.ndarray  # [T, 22, 3]
    fps: int = 20
    prompt: Optional[str] = None
    generator_id: str = ""
    generator_class_hash: str = ""
    seed: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.motion.ndim != 3 or self.motion.shape[1] != 22 or self.motion.shape[2] != 3:
            raise ValueError(
                f"motion shape must be [T, 22, 3], got {self.motion.shape}. "
                "AGENTS.md §3-1 Canonical Motion Format 위반."
            )


class Generator(ABC):
    """Generator abstract base class.

    Subclass는 generate()를 구현한다. wrapper는 generator-specific configuration
    을 __init__에서 받고 metadata 형성에 사용한다.
    """

    name: str = "BaseGenerator"  # subclass에서 override

    @abstractmethod
    def generate(
        self,
        prompt: Optional[str],
        n_frames: int,
        seed: int = 42,
        **kwargs: Any,
    ) -> GeneratorOutput:
        """Motion 생성.

        Args:
            prompt: text caption (text-to-motion). None이면 unconditional 또는 fixed seed.
            n_frames: 생성할 frame 수 (target T).
            seed: random seed.
            **kwargs: generator-specific arguments.

        Returns:
            GeneratorOutput — canonical [T, 22, 3] motion + metadata.

        Raises:
            ValueError: motion shape 위반 시 (AGENTS.md §3-1).
        """

    def quality_tier(self) -> str:
        """본 generator의 quality-tier 분류 (G1/G2/G3).

        명세 §9.2 quality-tier coverage 의무.

        Returns:
            'G1' (high-quality SOTA), 'G2' (token-based), 'G3' (legacy artifact-rich).
        """
        raise NotImplementedError(
            "Subclass must override quality_tier() — 명세 §9.2 quality-tier coverage 의무."
        )
