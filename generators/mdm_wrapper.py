"""MDM generator wrapper.

Wraps GuyTevet/motion-diffusion-model as a diffusion-family text-to-motion
generator for ArtifactRouter. The external MDM environment is isolated via
``conda run -n mdm`` because the upstream project pins Python/PyTorch versions
that are incompatible with the main router environment.
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
DEFAULT_MDM_ROOT = REPO_ROOT / "external_assets" / "motion-diffusion-model"


class MDM_G1(Generator):
    """G1 generator: MDM diffusion text-to-motion wrapper."""

    name = "MDM_G1"

    def __init__(
        self,
        mdm_root: Path = DEFAULT_MDM_ROOT,
        model_path: Optional[Path] = None,
        conda_env: str = "mdm",
        conda_exe: Optional[Path] = None,
        device: int = 0,
        backend: Optional[str] = None,
        service_url: Optional[str] = None,
    ) -> None:
        # backend: "conda" (기존) | "http" (docker 마이크로서비스).
        self.backend = backend or os.environ.get("ARTIFACTROUTER_GEN_BACKEND", "conda")
        self.service_url = service_url
        self.conda_env = conda_env
        self.device = int(device)

        if self.backend == "http":
            self.mdm_root = Path(mdm_root)
            self.model_path = Path(model_path) if model_path else Path("__http__")
            self.conda_exe = None
            return

        mdm_root = Path(mdm_root)
        if not mdm_root.exists():
            raise FileNotFoundError(
                f"MDM repo not found at {mdm_root}. Clone "
                "https://github.com/GuyTevet/motion-diffusion-model.git "
                "into external_assets/motion-diffusion-model."
            )
        self.mdm_root = mdm_root.resolve()
        self.model_path = (
            Path(model_path).resolve() if model_path is not None else self._find_checkpoint()
        )
        self.conda_exe = Path(conda_exe).resolve() if conda_exe else self._find_conda()

    def _find_checkpoint(self) -> Path:
        candidates = sorted(self.mdm_root.glob("save/**/model*.pt"))
        if not candidates:
            raise FileNotFoundError(
                "No MDM checkpoint found under external_assets/motion-diffusion-model/save/**/model*.pt. "
                "Download an official HumanML3D text-to-motion checkpoint first."
            )
        return candidates[-1].resolve()

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
        raise FileNotFoundError("conda.exe not found. Set CONDA_EXE or pass conda_exe.")

    def quality_tier(self) -> str:
        return "G1"

    def _generator_class_hash(self) -> str:
        wrapper_source = inspect.getsource(type(self))
        wrapper_hash = hashlib.sha256(wrapper_source.encode("utf-8")).hexdigest()
        ckpt_partial = b""
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                ckpt_partial = f.read(4 * 1024 * 1024)
        ckpt_hash = hashlib.sha256(ckpt_partial).hexdigest() if ckpt_partial else "missing"
        return f"{wrapper_hash[:16]}_{ckpt_hash[:16]}"

    @staticmethod
    def _as_tj3(motion: np.ndarray, length: Optional[int] = None) -> np.ndarray:
        arr = np.asarray(motion)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Expected MDM motion rank 3/4, got {arr.shape}")
        if arr.shape[0] == 22 and arr.shape[1] == 3:
            arr = arr.transpose(2, 0, 1)
        elif arr.shape[1] == 22 and arr.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Cannot convert MDM motion shape to [T,22,3]: {arr.shape}")
        if length is not None:
            arr = arr[: int(length)]
        return arr.astype(np.float64)

    def generate(
        self,
        prompt: Optional[str],
        n_frames: int,
        seed: int = 42,
        **kwargs: Any,
    ) -> GeneratorOutput:
        if prompt is None:
            raise ValueError("MDM requires a text prompt.")

        if self.backend == "http":
            from generators._http_client import generate_via_http, service_url
            base = service_url("mdm", self.service_url)
            motion, meta = generate_via_http(base, prompt, n_frames, seed)
            return GeneratorOutput(
                motion=motion,
                fps=DEFAULT_FPS,
                prompt=prompt,
                generator_id=meta.get("generator_id", "G1_mdm"),
                generator_class_hash=self._generator_class_hash(),
                seed=seed,
                metadata={"wrapper": "MDM_G1", "backend": "http", "service_url": base, **meta},
            )

        motion_length_sec = float(kwargs.get("motion_length_sec", n_frames / DEFAULT_FPS))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_npy = Path(tmpdir) / "mdm_motion.npy"
            inference_script = REPO_ROOT / "generators" / "_mdm_inference.py"
            cmd = [
                str(self.conda_exe),
                "run",
                "-n",
                self.conda_env,
                "--no-capture-output",
                "python",
                str(inference_script),
                "--mdm-root",
                str(self.mdm_root),
                "--model-path",
                str(self.model_path),
                "--prompt",
                prompt,
                "--n-frames",
                str(int(n_frames)),
                "--seed",
                str(seed),
                "--output",
                str(out_npy),
                "--device",
                str(self.device),
            ]
            result = subprocess.run(
                cmd,
                cwd=self.mdm_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"MDM inference failed (exit {result.returncode}).\n"
                    f"--- stdout ---\n{result.stdout}\n"
                    f"--- stderr ---\n{result.stderr}\n"
                    f"--- cmd ---\n{' '.join(cmd)}"
                )
            if not out_npy.exists():
                raise RuntimeError(
                    f"MDM produced no output at {out_npy}.\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                )
            motion = np.load(out_npy).astype(np.float64)

        return GeneratorOutput(
            motion=motion,
            fps=DEFAULT_FPS,
            prompt=prompt,
            generator_id=f"G1_mdm_{self.model_path.stem}",
            generator_class_hash=self._generator_class_hash(),
            seed=seed,
            metadata={
                "wrapper": "MDM_G1",
                "mdm_root": str(self.mdm_root),
                "checkpoint": str(self.model_path),
                "conda_env": self.conda_env,
                "device": self.device,
                "motion_length_sec": motion_length_sec,
                "helper": str(REPO_ROOT / "generators" / "_mdm_inference.py"),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MDM generator wrapper CLI")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--n-frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mdm-root", type=Path, default=DEFAULT_MDM_ROOT)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--conda-env", type=str, default="mdm")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    gen = MDM_G1(
        mdm_root=args.mdm_root,
        model_path=args.model_path,
        conda_env=args.conda_env,
        device=args.device,
    )
    out = gen.generate(args.prompt, n_frames=args.n_frames, seed=args.seed)
    print(f"[OK] motion shape: {out.motion.shape}")
    print(f"[OK] generator_id: {out.generator_id}")
    print(f"[OK] metadata: {json.dumps(out.metadata, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
