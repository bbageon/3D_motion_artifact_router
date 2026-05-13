---
description: ArtifactRouter 데이터·모델·tool registry 산출물(generator output, normalized motion, evaluator config, correction_tool config, LoRA adapter)의 버전을 metadata·hash·model_card.json으로 추적해 silent invalidation을 차단하는 skill이다.
metadata:
  scope:
    paths:
      - "external_assets/**"
      - "generators/**"
      - "skeleton_normalizer/**"
      - "evaluators/**"
      - "correction_tools/**"
      - "orchestrator/**"
      - "experiments/**"
  activation:
    keywords:
      - "metadata"
      - "model_card"
      - "version"
      - "silent invalidation"
      - "data version"
    when_to_use: 새 산출물(generator output·tool registry config·LoRA adapter·normalized motion 등) 생성 시. 학습·추론·평가 시작 전 정합성 검증.
  constraints:
    allowed_tools:
      - "Read"
      - "Write"
      - "Edit"
      - "Bash"
      - "Glob"
    risk_level: high
---

# Data Versioning — ArtifactRouter

본 skill은 [`docs/harness-research-template/skillSample/data-versioning.md`](../../../docs/harness-research-template/skillSample/data-versioning.md) 의 버전 관리 절차를 본 프로젝트에 적용한다.

---

## 1. 목적

다음을 보장한다.

- 산출물의 metadata 가 코드 정의와 일치 (silent invalidation 차단).
- 학습·추론·평가 시작 전 정합성 검증.
- raw record 가 어떤 generator·evaluator·tool registry·LoRA 로 생성됐는지 정확 추적.

---

## 2. metadata 규약

### 2-1. Generator output

```python
{
    "schema_version": "v1",
    "generator_id": "G1_MDM_v0.3" | "G2_motiongpt_v1.0" | "G3_local_lora",
    "generator_class_hash": "<sha256 of generator wrapper code>",
    "prompt": "<text caption (있으면)>",
    "n_frames": 40,
    "fps": 20,
    "seed": 42,
    "generated_at": "<ISO 8601>",
    "git_commit": "<HEAD>",
    "git_dirty": <bool>,
    "metadata_freeze": "<additional metadata>"
}
```

저장 위치: generator output `.npy` 또는 `.json` 의 sidecar `metadata.json`.

### 2-2. Normalized motion (Skeleton Normalizer 산출물)

```python
{
    "schema_version": "v1",
    "canonical_format": "SMPL_22",
    "joint_order": [<22 joint name list>],
    "root_relative": True,
    "fps": 20,
    "ground_plane_y": <float>,
    "contact_labels": <bool array (있으면)>,
    "source_generator_id": "<from §2-1>",
    "source_metadata_hash": "<sha256 of generator metadata>",
    "normalizer_class_hash": "<sha256 of normalizer code>",
}
```

### 2-3. Evaluator config

```python
{
    "schema_version": "v1",
    "evaluator_id": "<unique id>",
    "evaluator_class": "ContactEvaluator" | "TemporalEvaluator" | ...,
    "metric_definitions": {<metric_name>: {<formula 인용>}},
    "thresholds": {<metric_name>: {<severity threshold>}},
    "evaluator_class_hash": "<sha256>",
    "registered_at": "<ISO 8601>"
}
```

### 2-4. Correction tool config

```python
{
    "schema_version": "v1",
    "tool_id": "<unique id>",
    "tool_class": "FootLockTool" | "VelocitySmoothingTool" | ...,
    "kdg_affected_joints": [<joint name list>],
    "kdg_propagation_weights": {<joint>: <weight>},
    "compatibility_matrix": {<artifact_type>: <bool>},
    "tool_class_hash": "<sha256>",
    "registered_at": "<ISO 8601>"
}
```

### 2-5. LoRA adapter (이전 저장소의 G3 등)

기존 model_card.json 형식 그대로. 본 프로젝트의 G3 (LoRA wrapper) 는 [external_assets/local_lora_g3/model_card.json](../../../external_assets/local_lora_g3/model_card.json) 의 hash 를 raw record 에 인용.

### 2-6. Orchestrator state

orchestrator 가 학습된 경우 (supervised / contextual-bandit):

```python
{
    "schema_version": "v1",
    "orchestrator_id": "<unique id>",
    "algorithm": "rule_based" | "supervised" | "contextual_bandit",
    "training_data_hash": "<sha256 of training raw records>",
    "score_function_hash": "<sha256 of scoring.py>",
    "kdg_hash": "<sha256 of kdg.py>",
    "trained_at": "<ISO 8601>" (학습된 경우만),
    "checkpoint_hash": "<sha256 of model weights>" (학습된 경우만)
}
```

---

## 3. 정합성 검증 절차

학습·추론·평가 시작 전 다음 항목 검증:

1. Generator output metadata.git_commit 이 현재 HEAD 와 일치 또는 명시적 frozen 산출물.
2. Generator class hash 가 현재 코드와 일치.
3. Normalizer class hash 일치.
4. Evaluator/correction tool config 의 class hash 가 현재 코드와 일치.
5. KDG/scoring hash 일치.
6. Orchestrator config (학습된 경우) 의 training_data_hash 가 현재 활용하는 데이터셋과 호환.

불일치 발견 시 hard fail — 산출물 재생성 또는 코드 revert.

---

## 4. raw record 의 metadata cross-link

평가 raw record (`evals/raw/<timestamp>_<task_id>_<trial_id>.json`) 는 위 metadata 들의 hash 를 모두 인용:

```python
{
    ...
    "metadata_refs": {
        "generator_metadata_hash": "<from §2-1>",
        "normalized_motion_metadata_hash": "<from §2-2>",
        "evaluator_config_hashes": ["<from §2-3>"],
        "correction_tool_config_hashes": ["<from §2-4>"],
        "lora_model_card_hash": "<from §2-5>" (G3 사용 시),
        "orchestrator_config_hash": "<from §2-6>"
    },
    ...
}
```

본 cross-link 끊긴 raw record 는 회귀 판정 근거로 인용 불가.

---

## 5. 금지 규칙

- metadata 사후 편집 (silent invalidation) 금지.
- 다른 정책으로 만들어진 산출물을 동일 파일명으로 덮어쓰기 금지.
- metadata 누락한 산출물을 학습·추론·평가에 사용 금지.
- 외부 generator (G1/G2 등) 의 output 을 metadata 없이 import 금지 — wrapper 가 generator_id + metadata 자동 생성하도록 강제.

---

## 6. 본 프로젝트 산출물 metadata sidecar 위치

| 산출물 | 위치 | metadata 파일 |
|---|---|---|
| Generator output | `experiments/<week>/generators/<gen_id>/output_<N>.npy` | 같은 디렉토리의 `metadata.json` |
| Normalized motion | `experiments/<week>/normalized/<gen_id>/<id>.json` | JSON 내부 `metadata` 필드 |
| Evaluator config | `evaluators/configs/<config_id>.yaml` | 같은 파일에 `metadata:` 절 |
| Correction tool config | `correction_tools/configs/<config_id>.yaml` | 동일 |
| Orchestrator config | `orchestrator/configs/<config_id>.yaml` | 동일 |
| Refinement trace | `evals/raw/<timestamp>.json` | raw record 자체 |
