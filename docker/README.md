# Generator GPU Docker 마이크로서비스 (내부망 전용)

ArtifactRouter 의 3개 motion generator 를 **각각 독립 GPU Docker 서비스**로 분리한다.
모델은 컨테이너 startup 에 GPU(RTX 5090, sm_120 / cu128)에 상주하고, `POST /generate`
요청마다 canonical `[T,22,3]` (fps=20, root-relative) 모션을 JSON 으로 반환한다.

- **이미지 = 의존성 환경만.** upstream repo + checkpoint 는 host `external_assets/` 를
  컨테이너에 **read-only 마운트**(`:ro`) — 이미지에 굽지 않는다 (AGENTS.md §4 freeze 일치).
- **로컬 빌드/실행 전용** (registry push 없음). **내부망 무인증** (신뢰망 가정).

| generator | family | host 포트 | 마운트 대상 | 이미지 |
|---|---|---|---|---|
| MoMask (G2) | masked / residual-VQ | 8002 | `external_assets/momask-codes` | `artifactrouter/momask` |
| MotionGPT (G2) | token / motion-language | 8003 | `external_assets/MotionGPT` | `artifactrouter/motiongpt` |
| MDM (G1) | diffusion | 8001 | `external_assets/motion-diffusion-model` | `artifactrouter/mdm` |

## 사전 조건

- Docker Desktop + WSL2 백엔드 + NVIDIA GPU 패스스루. 확인:
  ```
  docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
  ```
  (RTX 5090 이 보여야 함.)

## 빌드

```
# 개별
docker build -f docker/momask/Dockerfile    -t artifactrouter/momask:latest .
docker build -f docker/motiongpt/Dockerfile -t artifactrouter/motiongpt:latest .
docker build -f docker/mdm/Dockerfile       -t artifactrouter/mdm:latest .
# 일괄
powershell -File docker/build_all.ps1
```

## 기동 / 종료

```
docker compose -f docker/compose.yaml up -d            # 3개 모두
docker compose -f docker/compose.yaml up -d momask     # 하나만
docker compose -f docker/compose.yaml logs -f momask   # 로그
docker compose -f docker/compose.yaml down             # 종료
```

## 호출

```
# health (모델 로딩/arch 확인)
curl http://localhost:8002/health

# generate
curl -X POST http://localhost:8002/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a person walks forward","n_frames":40,"seed":42}'
```

ArtifactRouter wrapper 에서 (motion-router env):
```python
from generators.momask_wrapper import MoMask_G2
g = MoMask_G2(backend="http")                 # 기본 localhost:8002
# 다른 서버: MoMask_G2(backend="http", service_url="http://192.168.0.43:8002")
out = g.generate("a person walks forward", n_frames=40, seed=42)
```
또는 env: `ARTIFACTROUTER_GEN_BACKEND=http`, `ARTIFACTROUTER_MOMASK_URL=http://192.168.0.43:8002`.

## 내부망 다른 PC 에서 접속

1. 서버 LAN IP 확인 (예: `192.168.0.43`).
2. 다른 PC 에서: `curl http://192.168.0.43:8002/health`.
3. **방화벽**: 접속이 안 되면 서버 Windows Defender 방화벽에서 inbound TCP 8001-8003 을
   **Private/Domain 프로필**로 허용 (Public 제외). 인터넷에는 노출하지 않는다.
4. 일부 캠퍼스 VLAN 은 단말 간 통신을 격리(client isolation)할 수 있다 — 그 경우 같은
   서브넷/관리자 정책 확인 필요.

## 비고 (연구 규칙)

- **CPU→GPU 전환은 generator output 수치를 바꾼다.** 응답 metadata 의 `compute_backend`
  (`docker_gpu`)를 기록하고, 기존 CPU pool 과 한 record 로 섞지 않는다 (AGENTS.md §3-5/§3-6/§3-10).
- 출력 계약(`[T,22,3]`, fps20, root-relative)은 conda backend 와 동일 — evaluator/orchestrator
  /refinement_loop 에 영향 없음.
- torch≥2.6 `weights_only=True` 기본값은 각 service.py 의 `torch.load` 패치로 우회.
