# MDM (G1, diffusion) generator service.
# 이미지 = 의존성 환경만. upstream repo(external_assets/motion-diffusion-model) +
# checkpoint + dataset/glove/body_models 는 런타임에 /assets 로 read-only 마운트.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN python -m pip install --upgrade pip "setuptools<58" wheel

RUN python -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# chumpy (SMPL .pkl 로딩) — numpy 먼저 + setuptools<58
RUN python -m pip install "numpy==1.23.5" \
    && python -m pip install "chumpy==0.70" --no-build-isolation

COPY docker/mdm/requirements.txt /tmp/req.txt
RUN python -m pip install -r /tmp/req.txt
# spacy 3.5.4 는 pydantic v1 (ConstrainedStr) 필요 → fastapi 도 pydantic-v1 호환 버전으로 핀.
RUN python -m pip install "pydantic<2" "fastapi>=0.95,<0.100" "uvicorn[standard]"

# CLIP(ViT-B/32) build 시점 캐시 (MDM text encoder)
RUN python -c "import clip; clip.load('ViT-B/32', device='cpu')"

COPY docker/mdm/service.py /app/service.py
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
