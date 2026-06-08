# MotionGPT (G2, token / motion-language) generator service.
# 이미지 = 의존성 환경만. upstream repo(external_assets/MotionGPT, 16GB: flan-t5 LM +
# deps + checkpoint) 는 런타임에 /assets 로 read-only 마운트 (이미지에 굽지 않음).
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git build-essential \
        libgl1 libglib2.0-0 libosmesa6 freeglut3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# setuptools<58 (chumpy build use_2to3 호환) 먼저
RUN python -m pip install --upgrade pip "setuptools<58" wheel

# torch cu128 (sm_120) — layer 캐시
RUN python -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# numpy 먼저 고정 후 chumpy build (numpy 헤더 필요 + setuptools<58)
RUN python -m pip install "numpy==1.26.4" \
    && python -m pip install "chumpy==0.70" --no-build-isolation

COPY docker/motiongpt/requirements.txt /tmp/req.txt
RUN python -m pip install -r /tmp/req.txt
RUN python -m pip install "fastapi" "uvicorn[standard]"
RUN python -m spacy download en_core_web_sm

COPY docker/motiongpt/service.py /app/service.py
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
