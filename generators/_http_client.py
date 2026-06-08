"""HTTP client for dockerized generator services (wrapper backend='http').

각 generator 가 GPU Docker 마이크로서비스로 떠 있을 때 (docker/compose.yaml),
ArtifactRouter wrapper 는 subprocess 대신 본 클라이언트로 호출한다. 서비스는
``POST /generate {prompt,n_frames,seed}`` 에 canonical ``[T,22,3]`` motion 을 JSON
으로 반환한다 (docker/<gen>/service.py).

내부망 다른 PC 에서 호출하려면 base_url 의 host 를 서버 LAN IP 로 지정하거나
env ``ARTIFACTROUTER_<GEN>_URL`` 을 설정한다 (예: http://192.168.0.43:8002).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

# 컨테이너 포트 매핑 (docker/compose.yaml): MDM 8001 / MoMask 8002 / MotionGPT 8003
DEFAULT_URLS = {
    "mdm": "http://localhost:8001",
    "momask": "http://localhost:8002",
    "motiongpt": "http://localhost:8003",
}


def service_url(generator: str, override: Optional[str] = None) -> str:
    """generator 의 서비스 base URL 결정 (override > env > 기본 localhost 포트)."""
    if override:
        return override.rstrip("/")
    env = os.environ.get(f"ARTIFACTROUTER_{generator.upper()}_URL")
    if env:
        return env.rstrip("/")
    if generator not in DEFAULT_URLS:
        raise ValueError(f"unknown generator '{generator}' (expected one of {list(DEFAULT_URLS)})")
    return DEFAULT_URLS[generator]


def generate_via_http(
    base_url: str,
    prompt: str,
    n_frames: int = 40,
    seed: int = 42,
    timeout: float = 600.0,
) -> tuple[np.ndarray, dict]:
    """서비스에 generation 요청 → (motion [T,22,3] float64, metadata dict) 반환."""
    import requests  # lazy import (main env 에만 의존)

    resp = requests.post(
        base_url.rstrip("/") + "/generate",
        json={"prompt": prompt, "n_frames": int(n_frames), "seed": int(seed)},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    motion = np.asarray(data["motion"], dtype=np.float64)
    meta = {k: v for k, v in data.items() if k != "motion"}
    return motion, meta


def generate_dual_via_http(
    base_url: str,
    prompt: str,
    n_frames: int = 40,
    seed: int = 42,
    timeout: float = 600.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """AR-048: (motion_trajectory, motion_local, meta) 반환.

    서비스가 motion_trajectory(root 보존) + motion_local(root-relative) 둘 다 반환.
    구버전 서비스(motion_trajectory 없음) 면 motion_local 만 있고 trajectory 는 None
    → 호출측이 'dual 미지원' 으로 처리(재빌드 필요).
    """
    import requests  # lazy import

    resp = requests.post(
        base_url.rstrip("/") + "/generate",
        json={"prompt": prompt, "n_frames": int(n_frames), "seed": int(seed)},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    has_traj = "motion_trajectory" in data
    traj = np.asarray(data["motion_trajectory"], dtype=np.float64) if has_traj else None
    local = np.asarray(data.get("motion_local", data["motion"]), dtype=np.float64)
    meta = {k: v for k, v in data.items() if k not in ("motion", "motion_trajectory", "motion_local")}
    meta["dual_supported"] = bool(has_traj)
    return traj, local, meta


def health(base_url: str, timeout: float = 10.0) -> dict:
    """서비스 /health 조회 (가동/arch/error)."""
    import requests

    resp = requests.get(base_url.rstrip("/") + "/health", timeout=timeout)
    resp.raise_for_status()
    return resp.json()
