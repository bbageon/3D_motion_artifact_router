"""MoMask generator wrapper.

Wraps EricGuo5513/momask-codes as a masked/discrete-token text-to-motion
generator for ArtifactRouter. The external MoMask environment is isolated via
``conda run -n momask`` or another user-provided conda env.
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
DEFAULT_MOMASK_ROOT = REPO_ROOT / "external_assets" / "momask-codes"


class MoMask_G2(Generator):
    """Masked-token generator wrapper for MoMask."""

    name = "MoMask_G2"

    def __init__(
        self,
        momask_root: Path = DEFAULT_MOMASK_ROOT,
        conda_env: str = "momask",
        conda_exe: Optional[Path] = None,
        gpu_id: int = -1,
        dataset_name: str = "t2m",
        name: str = "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns",
        vq_name: str = "rvq_nq6_dc512_nc512_noshare_qdp0.2",
        res_name: str = "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw",
        backend: Optional[str] = None,
        service_url: Optional[str] = None,
    ) -> None:
        # backend: "conda" (기존 conda run) | "http" (docker 마이크로서비스).
        # default = env ARTIFACTROUTER_GEN_BACKEND or "conda" (비파괴적).
        self.backend = backend or os.environ.get("ARTIFACTROUTER_GEN_BACKEND", "conda")
        self.service_url = service_url
        self.conda_env = conda_env
        self.gpu_id = int(gpu_id)
        self.dataset_name = dataset_name
        self.model_name = name
        self.vq_name = vq_name
        self.res_name = res_name

        if self.backend == "http":
            # repo/checkpoint/conda 는 컨테이너 안에 있으므로 로컬 검증 생략.
            self.momask_root = Path(momask_root)
            self.conda_exe = None
            return

        momask_root = Path(momask_root)
        if not momask_root.exists():
            raise FileNotFoundError(
                f"MoMask repo not found at {momask_root}. Clone "
                "https://github.com/EricGuo5513/momask-codes.git into external_assets/momask-codes."
            )
        self.momask_root = momask_root.resolve()
        self.conda_exe = Path(conda_exe).resolve() if conda_exe else self._find_conda()
        self._assert_checkpoint_layout()

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

    def _assert_checkpoint_layout(self) -> None:
        ckpt_root = self.momask_root / "checkpoints" / self.dataset_name
        required = [
            ckpt_root / self.model_name / "model",
            ckpt_root / self.vq_name / "model",
            ckpt_root / self.res_name / "model",
            ckpt_root / "length_estimator" / "model",
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            missing_str = "\n".join(str(p) for p in missing)
            raise FileNotFoundError(
                "MoMask checkpoint layout is incomplete. Run prepare/download_models.sh "
                f"or place official checkpoints. Missing:\n{missing_str}"
            )

    def quality_tier(self) -> str:
        return "G2"

    def _generator_class_hash(self) -> str:
        wrapper_source = inspect.getsource(type(self))
        wrapper_hash = hashlib.sha256(wrapper_source.encode("utf-8")).hexdigest()
        marker = (
            f"{self.dataset_name}/{self.model_name}/{self.vq_name}/{self.res_name}"
        ).encode("utf-8")
        marker_hash = hashlib.sha256(marker).hexdigest()
        return f"{wrapper_hash[:16]}_{marker_hash[:16]}"

    @staticmethod
    def _find_generated_joint(result_root: Path) -> Path:
        candidates = sorted(result_root.glob("joints/*/sample*_repeat*_len*.npy"))
        if not candidates:
            raise RuntimeError(f"No MoMask generated joint npy found under {result_root}")
        return candidates[0]

    def generate(
        self,
        prompt: Optional[str],
        n_frames: int,
        seed: int = 42,
        **kwargs: Any,
    ) -> GeneratorOutput:
        if prompt is None:
            raise ValueError("MoMask requires a text prompt.")

        if self.backend == "http":
            from generators._http_client import generate_via_http, service_url
            base = service_url("momask", self.service_url)
            motion, meta = generate_via_http(base, prompt, n_frames, seed)
            return GeneratorOutput(
                motion=motion,
                fps=DEFAULT_FPS,
                prompt=prompt,
                generator_id=meta.get("generator_id", f"G2_momask_{self.model_name}"),
                generator_class_hash=self._generator_class_hash(),
                seed=seed,
                metadata={"wrapper": "MoMask_G2", "backend": "http", "service_url": base, **meta},
            )

        ext = kwargs.get("ext") or f"artifactrouter_seed{seed}_{hashlib.sha1(prompt.encode()).hexdigest()[:8]}"
        with tempfile.TemporaryDirectory() as tmpdir:
            out_npy = Path(tmpdir) / "momask_motion.npy"
            inference_script = REPO_ROOT / "generators" / "_momask_inference.py"
            cmd = [
                str(self.conda_exe),
                "run",
                "-n",
                self.conda_env,
                "--no-capture-output",
                "python",
                str(inference_script),
                "--momask-root",
                str(self.momask_root),
                "--prompt",
                prompt,
                "--n-frames",
                str(int(n_frames)),
                "--seed",
                str(seed),
                "--output",
                str(out_npy),
                "--gpu-id",
                str(self.gpu_id),
                "--dataset-name",
                self.dataset_name,
                "--name",
                self.model_name,
                "--res-name",
                self.res_name,
            ]
            result = subprocess.run(
                cmd,
                cwd=self.momask_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"MoMask inference failed (exit {result.returncode}).\n"
                    f"--- stdout ---\n{result.stdout}\n"
                    f"--- stderr ---\n{result.stderr}\n"
                    f"--- cmd ---\n{' '.join(cmd)}"
                )
            if not out_npy.exists():
                raise RuntimeError(
                    f"MoMask produced no output at {out_npy}.\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                )
            motion = np.load(out_npy).astype(np.float64)

        return GeneratorOutput(
            motion=motion,
            fps=DEFAULT_FPS,
            prompt=prompt,
            generator_id=f"G2_momask_{self.model_name}",
            generator_class_hash=self._generator_class_hash(),
            seed=seed,
            metadata={
                "wrapper": "MoMask_G2",
                "momask_root": str(self.momask_root),
                "conda_env": self.conda_env,
                "gpu_id": self.gpu_id,
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "vq_name": self.vq_name,
                "res_name": self.res_name,
                "generation_ext": str(ext),
                "helper": str(REPO_ROOT / "generators" / "_momask_inference.py"),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MoMask generator wrapper CLI")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--n-frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--momask-root", type=Path, default=DEFAULT_MOMASK_ROOT)
    parser.add_argument("--conda-env", type=str, default="momask")
    parser.add_argument("--gpu-id", type=int, default=-1)
    args = parser.parse_args()

    gen = MoMask_G2(
        momask_root=args.momask_root,
        conda_env=args.conda_env,
        gpu_id=args.gpu_id,
    )
    out = gen.generate(args.prompt, n_frames=args.n_frames, seed=args.seed)
    print(f"[OK] motion shape: {out.motion.shape}")
    print(f"[OK] generator_id: {out.generator_id}")
    print(f"[OK] metadata: {json.dumps(out.metadata, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
