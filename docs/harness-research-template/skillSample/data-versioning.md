---
description: 데이터·모델 산출물(전처리 산출물·학습 입력·체크포인트)의 버전을 metadata·hash·model_card.json으로 추적해 silent invalidation을 차단하는 skill이다.
metadata:
  scope:
    paths:
      - "<프로젝트 데이터 전처리 스크립트>"
      - "<프로젝트 학습 입력 변환기>"
      - "<프로젝트 학습 스크립트>"
      - "<프로젝트 데이터·체크포인트 디렉토리>"
  activation:
    keywords:
      - "데이터 버전"
      - "model card"
      - "산출물 hash"
      - "정책 변경"
      - "silent invalidation"
      - "data versioning"
    when_to_use: 새 데이터 산출물을 생성하거나 모델 체크포인트를 저장하기 직전, 또는 정책·포맷 변경 직후 산출물 정합성을 확인할 때 참조한다.
  constraints:
    allowed_tools: ["Read", "Write", "Edit", "Bash", "Grep"]
    risk_level: high
  artifacts:
    inputs:
      - "현재 변환기/학습기 git hash"
      - "전처리 정책 (windowing·split·feature set 등)"
      - "토크나이저/특수 토큰 hash (해당 시)"
      - "사용 hyperparam"
    outputs:
      - "JSON/JSONL 첫 줄 _meta record"
      - "<체크포인트 디렉토리>/model_card.json"
      - "데이터·모델 정합성 검증 결과"
---

# Data Versioning Skill — Research Track Template

본 skill은 산출물의 silent invalidation(파일명은 같지만 내용 정의가 바뀐 상태)을 차단한다. Sambasivan et al. 2021(*"Everyone wants to do the model work, not the data work"*)은 데이터 버전 관리 부재가 ML 연구 재현성의 가장 큰 단일 위험으로 보고한다.

본 skill은 [`01-instructions.md §2`](../01-instructions.md)의 문장 형식 규칙을 그대로 따른다.

---

## 1. 목적

본 skill은 다음을 보장한다.

- 동일 파일명의 산출물이 서로 다른 정의에서 생성된 결과를 가리지 않게 한다(silent invalidation 차단).
- 학습 입력과 체크포인트의 입력 데이터·hyperparam·코드 버전이 추적 가능하도록 metadata를 동봉한다.
- 산출물 사용 직전 metadata와 현재 코드 정의가 일치하는지 결정론적으로 확인한다.

### 1-1. 본 skill이 맡지 않는 것

- 산출물 자체의 의미·정확성 검증 — change-obligation.
- 평가 지표 계산식 — [`reproducibility-checklist.md §3`](./reproducibility-checklist.md).
- 가설·실험 등록 — [`hypothesis-registry.md`](./hypothesis-registry.md).

본 skill은 산출물의 **메타데이터 동봉과 정합성 검증**만 수행한다.

---

## 2. 산출물별 metadata 규약

### 2-1. 내부 데이터 산출물 (전처리 결과)

전처리 스크립트가 생성하는 파일의 첫 레코드 또는 별도 `_manifest.json`에 다음을 포함한다.

```json
{
  "_meta": {
    "schema_version": "v1",
    "converter_git": "<전처리 스크립트의 git hash>",
    "<정책 파라미터 1>": <값>,
    "<정책 파라미터 2>": <값>,
    "feature_set": "<feature/joint set/column set 식별자>",
    "coordinate_system|normalization": "<좌표계 또는 정규화 방식>",
    "split": "train|val|test",
    "generated_at": "<ISO 8601>"
  }
}
```

샘플 레코드는 `_meta` 다음에 이어지며 형식은 AGENTS.md §1의 내부 스키마를 따른다.

### 2-2. 학습 입력 (JSONL 또는 동등 포맷)

학습 입력 파일의 **첫 줄**에 다음 metadata 레코드를 둔다(이후 줄은 학습 레코드).

```json
{"_meta": {"schema_version": "v1", "converter_git": "<변환기 git hash>", "track": "<비교 트랙>", "split": "train|val|test", "source_data_meta_hash": "<내부 산출물 _meta의 sha256>", "tokenizer_extension_hash": "<해당 시: tokenizer tree sha>", "input_format_version": "v1", "prompt_template_version": "v1", "generated_at": "<ISO 8601>"}}
```

학습기·추론기는 첫 줄을 `_meta`로 인식해 학습 데이터에서 제외하고 정합성 검증에 사용한다.

### 2-3. 모델 체크포인트

체크포인트 저장 시 같은 디렉토리에 `model_card.json`을 함께 저장한다.

```json
{
  "schema_version": "v1",
  "model_card_version": "v1",
  "track": "<비교 트랙>",
  "base_model": {
    "path": "<베이스 모델 경로>",
    "tokenizer_extension_hash": "<해당 시>",
    "base_model_revision": "<HF revision 또는 sha>"
  },
  "training_data": {
    "input_path": "<학습 입력 경로>",
    "input_meta_hash": "<학습 입력 _meta의 sha256>",
    "input_format_version": "v1",
    "prompt_template_version": "v1"
  },
  "hyperparameters": {
    "<hyperparam 1>": <값>,
    "<hyperparam 2>": <값>,
    "seed": <int>
  },
  "training_run": {
    "started_at": "<ISO 8601>",
    "ended_at": "<ISO 8601 또는 in-progress>",
    "total_step": <int>,
    "final_loss": <float>,
    "gpu_hours": <float>,
    "gpu_info": "<GPU 모델·CUDA·cuDNN 버전>"
  },
  "code_versions": {
    "trainer_git": "<학습 스크립트 git hash>",
    "converter_git": "<해당 트랙 변환기 git hash>"
  },
  "active_hypotheses": ["<H-id 목록>"]
}
```

`active_hypotheses`는 본 학습이 검증 대상으로 삼는 가설 ID이다 — [`hypothesis-registry.md`](./hypothesis-registry.md).

---

## 3. 정합성 검증 절차

학습·추론·평가 시작 전 Agent는 다음을 수행한다.

1. 사용할 데이터 산출물의 `_meta`를 읽는다.
2. 현재 변환기 git hash와 `_meta.converter_git`을 비교.
3. 현재 정책 파라미터(코드 디폴트 또는 CLI 인자)와 `_meta`의 정책 파라미터를 비교.
4. 토크나이저 등 외부 자산이 있다면 현재 hash와 `_meta`의 hash 비교.
5. 모델 체크포인트 사용 시 `model_card.json.training_data.input_meta_hash`와 평가에 쓸 입력의 `_meta` sha256을 비교.

검증 결과:

- 모두 일치 → 진행.
- 변환기·정책·토크나이저 중 하나라도 불일치 → hard fail. 산출물 재생성을 AGENTS.md §4 경로별 분기에 따라 수행.
- `model_card.json`이 없는 기존 체크포인트(grandfathered) → soft fail. 후처리로 model_card.json을 생성하거나 사용 시 일지에 명시 인용.

### 3-1. 판정 출력 형식

- 실패 시 출력은 "어떤 산출물에서, 어떤 metadata 항목이, 어떤 값(현재 vs 저장된)이 다른지"를 포함.
- Agent가 파싱 가능한 구조(JSON)로 출력.

**IMPORTANT:** metadata 불일치 상태에서 학습·추론·평가를 진행하지 마라. 결과는 어느 데이터·코드 정의 위에서 만들어진 것인지 추적 불가능해진다.

---

## 4. 신규 산출물 생성 절차

새 산출물을 만들 때:

1. 산출물 직전에 `git rev-parse HEAD`와 변경 파일 dirty 여부(`git status --porcelain`) 확인.
2. dirty 상태이면 → 커밋 후 산출물 생성, 또는 dirty 표시(`<commit>-dirty`)를 metadata에 포함.
3. §2의 metadata 스키마를 산출물 첫 레코드/별도 manifest/model_card에 동봉.
4. 입력의 `_meta` sha256을 함께 기록(다음 단계 산출물의 cross-link 키).
5. raw record(eval-collect §5)의 `git_metadata`·`env_metadata`에 동일 hash 기록.

### 4-1. 기존 산출물(grandfathered) 처리

본 skill 도입 이전에 생성된 산출물은 metadata가 없다.

- 기존 자산을 강제 재생성하지 마라(비용이 크고 시드·환경 재현이 어렵다).
- 사용 시점에 일지에 `legacy_artifact: true` 플래그와 추정 windowing/hash 명시.
- 회귀·개선 비교에서는 grandfathered 산출물을 기준으로 삼지 마라.
- 새 학습·평가 trial부터는 §2 규약을 강제.

---

## 5. 금지 규칙

- metadata 없이 산출물을 생성하지 마라(grandfathered 처리 외).
- metadata를 사후 편집해 hash를 맞추지 마라(불일치는 재생성으로만 해소).
- 정책이 다른 산출물을 같은 파일명으로 덮어쓰지 마라. 새 metadata와 함께 새 파일을 생성.
- model_card.json 없이 체크포인트를 다른 트랙·실험에 재사용하지 마라.
- `_meta` 레코드를 학습 입력으로 사용하지 마라(학습기·추론기는 첫 줄을 skip).

---

## 6. 유지보수 규칙

- §2 metadata 스키마를 변경했으면 → `schema_version` 또는 해당 sub-schema 버전을 올린다.
- 새 산출물 종류를 추가했으면 → §2에 절을 추가.
- 본 skill 검증이 hard fail로 반복 진입하면 → 변환기·학습기의 dirty 상태 관리(커밋 누락)가 원인일 가능성. AGENTS.md §3 작업 절차를 강화.
- 본 문서 용어가 change-obligation·eval-collect와 불일치하면 → change-obligation을 매핑 원본, eval-collect를 raw record 원본으로 본다.

<IMPORTANT>
데이터·모델 버전 관리는 "metadata 한 줄"이 아니라 **"다른 시점·다른 정의의 산출물이 같은 이름으로 섞이지 않게 만드는 약속"** 이다 — Sambasivan et al. 2021. metadata가 없는 산출물에서 도출된 모든 정량 결과는 AGENTS.md §3-7 자가 수정 메타 규칙의 근거 사슬을 끊는다.
</IMPORTANT>
