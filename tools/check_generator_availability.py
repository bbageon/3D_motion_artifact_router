"""Check external generator acquisition status.

This is a lightweight setup audit. It does not run generation; it checks whether
the repo, wrapper, checkpoint layout, and conda executable are present.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# docker/compose.yaml 의 host 포트 매핑.
SERVICE_PORTS = {"mdm": 8001, "momask": 8002, "motiongpt": 8003}


def _probe_service(generator: str) -> dict:
    """docker GPU 서비스 /health 조회 (stdlib urllib, 추가 의존성 없음)."""
    host = os.environ.get(
        f"ARTIFACTROUTER_{generator.upper()}_URL",
        f"http://localhost:{SERVICE_PORTS[generator]}",
    ).rstrip("/")
    try:
        with urllib.request.urlopen(host + "/health", timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return {"url": host, "reachable": True, "status": body.get("status"),
                "device": body.get("device"), "arch_has_sm120": "sm_120" in (body.get("arch_list") or [])}
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        return {"url": host, "reachable": False, "error": str(exc)}


def _exists(path: Path) -> bool:
    return path.exists()


def _conda_exists() -> tuple[bool, str | None]:
    env_conda = os.environ.get("CONDA_EXE")
    if env_conda and Path(env_conda).exists():
        return True, str(Path(env_conda).resolve())
    for cand in [
        Path.home() / "anaconda3" / "Scripts" / "conda.exe",
        Path.home() / "miniconda3" / "Scripts" / "conda.exe",
        Path("C:/ProgramData/anaconda3/Scripts/conda.exe"),
        Path("C:/ProgramData/miniconda3/Scripts/conda.exe"),
    ]:
        if cand.exists():
            return True, str(cand.resolve())
    return False, None


def main() -> None:
    conda_ok, conda_path = _conda_exists()
    mdm_root = REPO_ROOT / "external_assets" / "motion-diffusion-model"
    momask_root = REPO_ROOT / "external_assets" / "momask-codes"
    out = {
        "MotionGPT": {
            "wrapper": _exists(REPO_ROOT / "generators" / "motiongpt_wrapper.py"),
            "repo": _exists(REPO_ROOT / "external_assets" / "MotionGPT"),
            "checkpoint_candidates": [
                str(p.relative_to(REPO_ROOT))
                for p in (REPO_ROOT / "external_assets" / "MotionGPT" / "checkpoints" / "MotionGPT-base").glob("*")
                if p.suffix in {".ckpt", ".tar", ".bin"}
            ],
        },
        "MDM": {
            "wrapper": _exists(REPO_ROOT / "generators" / "mdm_wrapper.py"),
            "repo": _exists(mdm_root),
            "checkpoint_candidates": [
                str(p.relative_to(REPO_ROOT)) for p in mdm_root.glob("save/**/model*.pt")
            ] if mdm_root.exists() else [],
        },
        "MoMask": {
            "wrapper": _exists(REPO_ROOT / "generators" / "momask_wrapper.py"),
            "repo": _exists(momask_root),
            "checkpoint_layout": {
                "masked_transformer": _exists(momask_root / "checkpoints" / "t2m" / "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns" / "model"),
                "rvq": _exists(momask_root / "checkpoints" / "t2m" / "rvq_nq6_dc512_nc512_noshare_qdp0.2" / "model"),
                "residual_transformer": _exists(momask_root / "checkpoints" / "t2m" / "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw" / "model"),
                "length_estimator": _exists(momask_root / "checkpoints" / "t2m" / "length_estimator" / "model"),
            } if momask_root.exists() else {},
        },
        "conda": {"exists": conda_ok, "path": conda_path},
        "docker_services": {g: _probe_service(g) for g in SERVICE_PORTS},
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
