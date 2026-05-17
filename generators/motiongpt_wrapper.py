"""MotionGPT (G2) Generator wrapper — OpenMotionLab/MotionGPT 공식 구현 wrapping.

본 wrapper 는 ArtifactRouter 의 **G2 (token-based)** generator 로 공식 MotionGPT
(NeurIPS 2023, <https://github.com/OpenMotionLab/MotionGPT>) 를 사용한다.
MotionGPT 의 inference 환경 (`mgpt` conda env) 과 본 저장소의 메인 환경
(`motion-router` 또는 `motion3d`) 의 dependency 가 충돌하므로 **subprocess
+ `conda run -n mgpt`** 로 환경을 격리한다. 실제 inference 로직은
[`_motiongpt_inference.py`](_motiongpt_inference.py) 에 있다.

CLI:
    python -m generators.motiongpt_wrapper --prompt "a person walks forward" --n-frames 40

명세 §6.1 Base Motion Generator 참조.
AGENTS.md §3-1 Canonical Motion Format 의무 — MotionGPT 의 t2m 출력은 이미
canonical `[T, 22, 3]` 이므로 별도 normalize 불필요.
AGENTS.md §2-1 의 MotionGPT 설치 절차 (`mgpt` env + checkpoint) 가 선결 조건.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from generators.base import Generator, GeneratorOutput
from skeleton_normalizer.canonical_smpl_22 import DEFAULT_FPS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTIONGPT_ROOT = REPO_ROOT / "external_assets" / "MotionGPT"


class MotionGPT_G2(Generator):
    """G2 generator — 공식 OpenMotionLab/MotionGPT wrapper.

    Quality-tier: G2 (token-based, VQ-VAE codebook + T5-style LM). 명세 §9.2 quality-tier coverage 의무.

    Note: mgpt conda env + MotionGPT checkpoint 가 사전 설치되어 있어야 한다.
    설치 절차는 AGENTS.md §2-1 참조.
    """

    name = "MotionGPT_G2"

    def __init__(
        self,
        motiongpt_root: Path = DEFAULT_MOTIONGPT_ROOT,
        cfg_path: str = "configs/config_h3d_stage3.yaml",
        checkpoint_path: Optional[Path] = None,
        conda_env: str = "mgpt",
        conda_exe: Optional[Path] = None,
    ) -> None:
        """
        Args:
            motiongpt_root: external_assets/MotionGPT clone 경로.
            cfg_path: MotionGPT config (motiongpt_root 기준 상대경로).
            checkpoint_path: 학습된 ckpt/tar 파일. None 이면 checkpoints/MotionGPT-base/ 자동 탐색.
            conda_env: MotionGPT 의존성 설치된 conda env 이름 (기본 'mgpt').
            conda_exe: conda 실행 파일. None 이면 자동 탐색.
        """
        motiongpt_root = Path(motiongpt_root)
        if not motiongpt_root.exists():
            raise FileNotFoundError(
                f"MotionGPT clone not found at {motiongpt_root}.\n"
                "git clone https://github.com/OpenMotionLab/MotionGPT.git external_assets/MotionGPT\n"
                "AGENTS.md §2-1 의 MotionGPT 설치 절차 참조."
            )
        self.motiongpt_root = motiongpt_root.resolve()
        self.cfg_path = cfg_path
        self.checkpoint_path = (
            Path(checkpoint_path).resolve()
            if checkpoint_path is not None
            else self._find_checkpoint()
        )
        self.conda_env = conda_env
        self.conda_exe = Path(conda_exe).resolve() if conda_exe else self._find_conda()

    def _find_checkpoint(self) -> Path:
        ckpt_dir = self.motiongpt_root / "checkpoints" / "MotionGPT-base"
        candidates = (
            list(ckpt_dir.glob("*.ckpt"))
            + list(ckpt_dir.glob("*.tar"))
            + list(ckpt_dir.glob("*.bin"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint found in {ckpt_dir}. "
                "Run prepare/download_pretrained_models.sh first. "
                "AGENTS.md §2-1 의 MotionGPT 설치 절차 참조."
            )
        # 우선순위: .ckpt > .tar > .bin
        candidates.sort(key=lambda p: {".ckpt": 0, ".tar": 1, ".bin": 2}.get(p.suffix, 3))
        return candidates[0].resolve()

    @staticmethod
    def _find_conda() -> Path:
        env_conda = os.environ.get("CONDA_EXE")
        if env_conda and Path(env_conda).exists():
            return Path(env_conda).resolve()
        for cand in [
            Path.home() / "anaconda3" / "Scripts" / "conda.exe",
            Path.home() / "miniconda3" / "Scripts" / "conda.exe",
            Path("C:/ProgramData/anaconda3/Scripts/conda.exe"),
            Path("C:/ProgramData/miniconda3/Scripts/conda.exe"),
        ]:
            if cand.exists():
                return cand.resolve()
        raise FileNotFoundError(
            "conda.exe not found. Set CONDA_EXE env var or pass conda_exe explicitly."
        )

    def quality_tier(self) -> str:
        return "G2"

    def _generator_class_hash(self) -> str:
        """Wrapper source + checkpoint partial hash 결합 — 평가 기록 의무 (AGENTS.md §3-6).

        Checkpoint 전체 hash 는 1GB+ 라 비용 크므로 **앞 4MB** 만 hash (PyTorch state_dict
        헤더가 충분히 포함되어 학습 산출물 구분 가능).
        """
        wrapper_source = inspect.getsource(type(self))
        wrapper_hash = hashlib.sha256(wrapper_source.encode("utf-8")).hexdigest()
        ckpt_partial = b""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "rb") as f:
                ckpt_partial = f.read(4 * 1024 * 1024)  # 4MB
        ckpt_hash = hashlib.sha256(ckpt_partial).hexdigest() if ckpt_partial else "missing"
        return f"{wrapper_hash[:16]}_{ckpt_hash[:16]}"

    def generate(
        self,
        prompt: Optional[str] = None,
        n_frames: int = 40,
        seed: int = 42,
        **kwargs: Any,
    ) -> GeneratorOutput:
        """Text-to-motion 생성 (t2m task).

        Args:
            prompt: text caption (필수 — MotionGPT 는 text-conditional generator).
            n_frames: target frame 수 hint (MotionGPT 가 ignore 할 수 있음 — 실제 생성 길이는 metadata 의 length_generated 참조).
            seed: random seed.

        Returns:
            GeneratorOutput — canonical [T, 22, 3] motion + metadata.

        Raises:
            ValueError: prompt 가 None.
            RuntimeError: subprocess inference 실패.
        """
        if prompt is None:
            raise ValueError("MotionGPT (G2) requires a text prompt.")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_npy = Path(tmpdir) / "motion.npy"
            inference_script = REPO_ROOT / "generators" / "_motiongpt_inference.py"
            cmd = [
                str(self.conda_exe),
                "run",
                "-n",
                self.conda_env,
                "--no-capture-output",
                "python",
                str(inference_script),
                "--motiongpt-root",
                str(self.motiongpt_root),
                "--cfg",
                self.cfg_path,
                "--ckpt",
                str(self.checkpoint_path),
                "--prompt",
                prompt,
                "--n-frames",
                str(n_frames),
                "--seed",
                str(seed),
                "--output",
                str(out_npy),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"MotionGPT inference failed (exit {result.returncode}).\n"
                    f"--- stdout ---\n{result.stdout}\n"
                    f"--- stderr ---\n{result.stderr}\n"
                    f"--- cmd ---\n{' '.join(cmd)}"
                )
            if not out_npy.exists():
                raise RuntimeError(
                    f"MotionGPT inference produced no output ({out_npy}).\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                )

            motion = np.load(out_npy).astype(np.float64)
            metadata_path = out_npy.with_suffix(".json")
            metadata: dict[str, Any] = {}
            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)

        return GeneratorOutput(
            motion=motion,
            fps=DEFAULT_FPS,
            prompt=prompt,
            generator_id=f"G2_motiongpt_{self.checkpoint_path.stem}",
            generator_class_hash=self._generator_class_hash(),
            seed=seed,
            metadata={
                "wrapper": "MotionGPT_G2",
                "motiongpt_cfg": self.cfg_path,
                "checkpoint": str(self.checkpoint_path),
                "conda_env": self.conda_env,
                **metadata,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MotionGPT (G2) generator wrapper CLI"
    )
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text-to-motion prompt (필수)")
    parser.add_argument("--n-frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motiongpt-root", type=Path, default=DEFAULT_MOTIONGPT_ROOT)
    parser.add_argument("--cfg", type=str, default="configs/config_h3d_stage3.yaml")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="checkpoint path (None 이면 자동 탐색)")
    parser.add_argument("--conda-env", type=str, default="mgpt")
    args = parser.parse_args()

    print(f"[INFO] G2 generator (MotionGPT) — motiongpt_root: {args.motiongpt_root}")
    gen = MotionGPT_G2(
        motiongpt_root=args.motiongpt_root,
        cfg_path=args.cfg,
        checkpoint_path=args.ckpt,
        conda_env=args.conda_env,
    )
    print(f"[INFO] checkpoint: {gen.checkpoint_path}")
    print(f"[INFO] conda env: {gen.conda_env}")
    print(f"[INFO] generator_class_hash: {gen._generator_class_hash()}")
    print(f"[INFO] prompt: {args.prompt!r}, n_frames: {args.n_frames}, seed: {args.seed}")

    try:
        out = gen.generate(prompt=args.prompt, n_frames=args.n_frames, seed=args.seed)
        print(f"[OK] motion shape: {out.motion.shape}")
        print(f"[OK] generator_id: {out.generator_id}")
        print(f"[OK] metadata: {json.dumps(out.metadata, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"[ERROR] generation failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
