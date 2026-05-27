# Setup — ArtifactRouter Environment Preparation

> 본 문서는 AGENTS.md §2-1 의 install 절차 의 단일 출처. AGENTS.md 는 high-level summary + cross-link 만 유지 (architectural form, [`AGENTS.md`](../AGENTS.md)).

---

## 1. Main env (motion-router) — 메인 운영 환경

```
conda create -n motion-router python=3.10 -y
conda activate motion-router
pip install -r requirements.txt
```

본 env 는 evaluator / correction tool / orchestrator / refinement loop / 시각화 의 전체 stack.

---

## 2. 데이터 자산 준비 — `external_assets/`

본 저장소 내부 의 실제 복사본 으로 보유 (2026-05-15 마이그레이션 완료, 이전 저장소 의존 제거):

| 자산 | 크기 | 의미 |
|---|---|---|
| `external_assets/HumanML3D/` | 4.7 GB | public HumanML3D 데이터셋 (79,867 files) |
| `external_assets/processed_noaug/` | 3.0 GB | sliding window 전처리 JSON |
| `external_assets/local_lora_g3/` | 275 MB | **vestigial archive** (이전 저장소 LoRA, scope 외, 보존만) |
| `external_assets/code/` | — | 시각화 utility + 이전 저장소 LLM motion experiment 보존 코드 |

본 자산들 은 `.gitignore` 에 의해 git 추적 제외 — **fresh clone 시 자동 따라오지 않음**. 다음 두 방법 중 하나로 확보:

### 2-1. 옵션 A — HumanML3D 공식 repo 에서 재다운로드 + 전처리

<https://github.com/EricGuo5513/HumanML3D> 의 절차 따라 `data/HumanML3D` 생성. `processed_noaug` 는 본 저장소 의 전처리 스크립트 로 생성 (TBD).

### 2-2. 옵션 B — 이미 보유한 사본 에서 복사

```
robocopy <source> external_assets\HumanML3D /E /MT:8
robocopy <source> external_assets\processed_noaug /E /MT:8
```

이전 저장소 (`3D-Motion-Trajectory-prediction`) junction 은 더 이상 필요/생성 안 함. 본 저장소 는 이전 저장소 파일시스템 의존 안 함.

---

## 3. MotionGPT (G2) — 별도 conda env (`mgpt`)

generator-agnostic refinement 의 G2 evidence 의 inference env. 본 env 는 motion-router 와 **dependency 충돌 회피** 위해 분리.

### 3-1. Step 1: clone

```
git clone https://github.com/OpenMotionLab/MotionGPT.git external_assets/MotionGPT
```

(이미 본 저장소 에 있으면 skip — `external_assets/MotionGPT/` 는 `.gitignore` 제외.)

### 3-2. Step 2-3: env 생성 + setuptools pinning (chumpy 빌드 호환)

```
conda create -n mgpt python=3.10 -y
conda activate mgpt
pip install --upgrade "setuptools<58" wheel
```

### 3-3. Step 4: requirements + 누락 항목

```
cd external_assets/MotionGPT
pip install -r requirements.txt
# 위가 chumpy 에서 실패하면:
#   pip install chumpy==0.70 --no-build-isolation
#   pip install -r requirements.txt
# 누락된 m2t metric transitive import 추가:
pip install bert_score gdown
python -m spacy download en_core_web_sm
```

### 3-4. Step 4-1: transformers / tokenizers 버전 pinning (W-2026-001 정공법, 의무)

`requirements.txt` 가 transformers 버전 명시 안 함 → 최신 (5.x) 설치 → MotionGPT (2023 학습 시점, transformers 4.x) 와 incompatible.

**증상**: 최신 transformers 의 `T5ForConditionalGeneration` 이 weight tying 처리 다름 — `shared.weight` 자리에 `lm_head.weight` 잘못 load → LM input embedding broken → prompt-independent 4-frame collapse (`(1, 4, 22, 3)` 만 출력, prompt 무시).

**정공법**:

```
pip install "transformers==4.30.2" "tokenizers==0.13.3"
```

### 3-5. Step 5: 보조 자산 (T5 LM + SMPL body + t2m + glove)

#### 3-5-1. T5 (flan-t5-base, 7.8 GB) — HuggingFace LFS

```
cd deps
git lfs install
git clone https://huggingface.co/google/flan-t5-base
```

#### 3-5-2. SMPL model + t2m evaluators (Google Drive, gdown)

```
cd deps
gdown "https://drive.google.com/uc?id=1qrFkPZyRwRGd0Q3EY76K8oJaIgs_WK9i" -O smpl.tar.gz
tar xfz smpl.tar.gz   # → deps/smpl_models/
gdown "https://drive.google.com/uc?id=1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8" -O t2m.tar.gz
tar xfz t2m.tar.gz    # → deps/t2m/{glove, t2m/{kit, t2m/*/meta/mean.npy,std.npy}}
```

추출 구조에 t2m 레벨 한 번 더 들어가 있으면 평탄화:

```
mv t2m/glove glove
mv t2m/t2m _inner
rm -rf t2m
mv _inner t2m
```

### 3-6. Step 6: Pretrained checkpoint (HuggingFace LFS, 1.24 GB)

```
git lfs install
mkdir -p checkpoints && cd checkpoints
git clone https://huggingface.co/OpenMotionLab/MotionGPT-base
# → checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar
```

### 3-7. Step 7: HumanML3D dataset junction (config 의 datasets/humanml3d 경로)

```
cd external_assets/MotionGPT
mkdir -p datasets
cmd /c "mklink /J datasets\humanml3d ..\HumanML3D"
```

`texts/` 가 nested 구조 (`texts/texts/*.txt`) 라면 평탄화:

```
cd external_assets/HumanML3D
mv texts texts_outer && mv texts_outer/texts texts && rm -rf texts_outer
```

---

## 4. MotionGPT 의 known issues

### 4-1. transformers 버전 incompatibility (W-2026-001 — RESOLVED)

§3-4 의 정공법 적용 필수. 자세한 W-id 박제: [`evals/workarounds/W-2026-001.md`](../evals/workarounds/W-2026-001.md) (있으면).

### 4-2. Length 제어 (residual issue)

transformers 4.30.2 downgrade 후에도 `batch['length']` 자체는 capture only — generation 길이 는 LM 의 `do_sample` 결과 에 의존. 단 prompt 에 따라 92/176 frames 등 정상 길이 동적 생성 됨 확인됨. wrapper 는 metadata 의 `length_generated` 로 실제 길이 기록.

### 4-3. CPU-only torch (mgpt env)

본 프로젝트 의 mgpt env 는 CPU torch — SMPLify3D fit 의 iters 절감 필수 (예: iters=30 대신 default 100). 자세한 set up: [`tools/_smpl_fit_inline.py`](../tools/_smpl_fit_inline.py) 의 inline 호출 path.

---

## 5. Wrapper 사용 (motion-router env 에서)

본 저장소 의 wrapper ([`generators/motiongpt_wrapper.py`](../generators/motiongpt_wrapper.py)) 는 `conda run -n mgpt python -m generators._motiongpt_inference ...` 로 환경 격리 호출 — **motion-router env 에서 호출 무방**. checkpoint 자동 탐색 (`external_assets/MotionGPT/checkpoints/MotionGPT-base/*.tar` 또는 `*.ckpt`).

---

## 6. 본 문서 의 유지 의무

- AGENTS.md §2-1 의 install 절차 의 단일 출처.
- MotionGPT 의 install 절차 또는 known issues 변경 시 본 문서 만 갱신 (AGENTS.md 는 변경 안 함).
- 본 문서 의 step 번호 변경 시 wrapper 또는 W-id ledger 의 cross-link 동시 갱신.
