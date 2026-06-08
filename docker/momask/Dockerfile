# MoMask (G2, masked/residual-VQ token) generator service.
# 이미지 = 의존성 환경만. upstream repo(external_assets/momask-codes) + checkpoint 는
# 런타임에 /assets 로 read-only 마운트한다 (이미지에 굽지 않음).
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

# torch cu128 (RTX 5090 / sm_120 지원) — 먼저 깔아 layer 캐시 (requirements 변경 시 재사용)
RUN python -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

COPY docker/momask/requirements.txt /tmp/req.txt
RUN python -m pip install -r /tmp/req.txt
RUN python -m pip install "fastapi" "uvicorn[standard]"

# CLIP(ViT-B/32) 가중치를 build 시점에 캐시 (런타임 재다운로드 방지)
RUN python -c "import clip; clip.load('ViT-B/32', device='cpu')"

COPY docker/momask/service.py /app/service.py
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
