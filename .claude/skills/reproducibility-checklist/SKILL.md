---
description: NeurIPS reproducibility checklist를 본 프로젝트(ArtifactRouter)에 적응한 점검 절차이며, 각 평가 지표(artifact metric, NetGain, FidelityLoss, FID_motion 등)의 정의·계산식·해석을 함께 제공한다.
metadata:
  scope:
    paths:
      - "reports/checklists/**"
      - "evals/**"
      - "docs/**"
  activation:
    keywords:
      - "재현성"
      - "reproducibility"
      - "checklist"
      - "metric 정의"
      - "외부 공개"
    when_to_use: 외부 공개(논문·발표·README)에 결과를 인용하기 전. 또는 metric 정의를 명시할 때.
  constraints:
    allowed_tools:
      - "Read"
      - "Write"
      - "Edit"
      - "Glob"
    risk_level: high
---

# Reproducibility Checklist — ArtifactRouter

본 skill은 [`docs/harness-research-template/skillSample/reproducibility-checklist.md`](../../../docs/harness-research-template/skillSample/reproducibility-checklist.md) 의 절차를 본 프로젝트(ArtifactRouter)에 적용한다.

---

## 1. 목적

다음을 보장한다.

- 외부 공개 전 14+1 항목 점검.
- 평가 지표 정의의 단일 출처 제공.
- 통계 절차의 정합성.

---

## 2. 14+1 점검 항목

본 점검은 NeurIPS Reproducibility Checklist (Pineau et al. 2021) 에 명세 §9 ArtifactRouter 특화 항목을 추가한 것.

### 2-1. 데이터셋 (#1-3)

1. **데이터 origin**: HumanML3D 14,646 motion 클립 (Guo et al. CVPR 2022). 본 저장소는 이전 저장소의 `external_assets/HumanML3D` junction 으로 참조.
2. **Train/val/test split**: HumanML3D 표준 split 또는 본 프로젝트 정의 split. 사용 시 split id 명시.
3. **전처리**: root-relative + sliding window (이전 저장소의 02 단계). 재현 가능한 명령어 제공.

### 2-2. 모델 (#4-6)

4. **Generator (외부)**:
   - G1: MDM 또는 MLD (publicly available checkpoint, version hash).
   - G2: MotionGPT 또는 T2M-GPT (동일).
   - G3: 이전 저장소의 `external_assets/local_lora_g3` (model_card.json hash).
5. **Skeleton Normalizer / Evaluator / Correction Tool**: 본 저장소 코드 + class hash.
6. **Orchestrator**: rule_based / supervised / contextual_bandit 알고리즘 + (학습된 경우) checkpoint hash.

### 2-3. 학습 (#7-9)

7. **Optimizer / Hyperparameter**: contextual-bandit 학습 시 명시.
8. **Hyperparameter search**: 사용한 경우 search space + selection 절차.
9. **Compute budget**: GPU 시간·FLOPs.

### 2-4. 평가 (#10-12)

10. **Evaluation metric 정의**: §3 단일 출처.
11. **표본 수**: trial / generator / seed 분포.
12. **통계 절차**: paired test (Wilcoxon signed-rank), bootstrap CI (N=1000), effect size (Cohen's d). [§3-7](#3-7-통계적-오라클).

### 2-5. 결정성 (#13)

13. **Determinism**: 시드 고정 (`torch.manual_seed`, `np.random.seed`, `random.seed`). NF4 inter-session noise 시 N≥3 generation 평균 적용. 시드 정책 명시.

### 2-6. 우회 (#14)

14. **Workaround ledger**: `evals/workarounds/_index.md` 의 `status: open` + `severity: critical` 항목이 0개임을 외부 공개 전 확인.

### 2-7. ArtifactRouter 특화 (#15)

15. **Tool registry config**: 사용한 evaluator 7종 · correction tool 9종 의 config hash. KDG nodes/edges 정의의 hash. orchestrator scoring weight (α, β, γ).

---

## 3. 평가 지표 정의 사전 (단일 출처)

본 절은 ArtifactRouter 의 모든 평가 metric 의 정의 단일 출처. 외부 인용 시 본 §3 만 참조.

### 3-1. Artifact Metrics (명세 §9.3.1)

#### FootSliding distance

```
FootSliding = mean_t I(contact_foot(t)) * || p_foot_xy(t+1) - p_foot_xy(t) ||
```

`I(contact_foot(t))` 는 contact 상태 indicator, `p_foot_xy` 는 foot joint 의 horizontal projection.

#### GroundPenetration ratio

```
GroundPenetration = mean_t max(0, ground_y - min_j p_j_y(t))
```

`ground_y` 는 Skeleton Normalizer 추정.

#### FootFloating ratio

```
FootFloating = mean_t I(contact_foot(t)) * I(p_foot_y(t) - ground_y > tau_float)
```

`tau_float` 는 default 0.05m.

#### BoneLengthVariation

```
BoneVar = mean_{t,b} | length_b(t) - length_b_ref | / length_b_ref
```

`length_b_ref` 는 첫 frame 또는 canonical reference.

#### JointAngleViolation rate

```
JointViolation = mean_{t,k} I(angle_k(t) < lower_k or angle_k(t) > upper_k)
```

`(lower_k, upper_k)` 는 anatomical range — `evaluators/skeletal_evaluator.py` 의 config.

#### VelocityJitter (mean acceleration norm)

```
v_j(t+1) - v_j(t) = a_j(t)
VelocityJitter = mean_{t,j} || a_j(t) ||
```

#### AccelerationJerk (mean jerk norm)

```
a_j(t+1) - a_j(t) = jerk_j(t)
AccelerationJerk = mean_{t,j} || jerk_j(t) ||
```

### 3-2. Normalized Score (명세 §9.3)

```
NormalizedArtifactScore_m = (metric_m - reference_mean_m) / (reference_std_m + epsilon)
TotalArtifactScore = sum_m w_m * NormalizedArtifactScore_m
```

`reference_mean_m`, `reference_std_m` 은 HumanML3D test set GT motion 분포에서 추정.

### 3-3. Fidelity Metrics

#### Protocol A — Synthetic injection

```
FidelityLoss_A = MPJPE(refined, clean_GT) - MPJPE(corrupted, clean_GT)
MPJPE = mean_{t,j} || p_refined(t,j) - p_GT(t,j) ||
```

#### Protocol B — Generator output

- `correction_magnitude = mean_{t,j} || p_refined(t,j) - p_generated(t,j) ||`
- `modified_frame_ratio = (frames_modified) / T`
- `semantic_consistency = CLIP_similarity(refined_motion, text_prompt)` (text-to-motion 의 경우)

#### Protocol C — Distributional

- **FID_motion**: motion-specific Fréchet Inception Distance (HumanML3D motion feature extractor 의 latent에서).
- **FGD**: Fréchet Gesture Distance (motion auto-encoder latent).
- **Diversity**: refined motion 간 latent feature pairwise distance 평균.
- **MM-Dist**: text-motion multimodal distance (text-to-motion).

### 3-4. NetGain (명세 §9.4)

```
NetGain = ArtifactReduction - alpha * FidelityLoss - beta * CorrectionMagnitude - gamma * ToolCallCost
ArtifactReduction = TotalArtifactScore_before - TotalArtifactScore_after
```

`alpha`, `beta`, `gamma` 는 명세 §9.4 에 따라 synthetic injection protocol 의 perceptual rating 상관 최대화로 grid search. 모든 baseline / ablation 동일 weight 사용.

### 3-5. Efficiency Metrics (명세 §9.3 Efficiency)

- `number_of_tool_calls`
- `iteration_count`
- `wall-clock runtime per sample (ms)`
- `FLOPs per sample` (refinement stage, generator inference 제외)
- `FPS` (effective throughput on fixed hardware)
- `cost_ratio = cost(ArtifactRouter) / cost(single learned calibrator)`

### 3-6. Tool Call Trace

각 refinement step 의 (`step_idx`, `tool_id`, `target_part`, `frame_range`, `strength`, `score_before`, `score_after`, `score_delta`) 를 trace.json 에 저장.

### 3-7. 통계적 오라클

- **Paired test**: Wilcoxon signed-rank — 동일 sample paired 두 처리 (예: rule-based vs learned orchestrator) 비교. p-value < 0.05.
- **Effect size**: Cohen's d — `(mean_A - mean_B) / pooled_std`.
- **Bootstrap CI**: N=1000 resampling, 95% CI.
- **Non-inferiority margin**: No-harm 평가 (H-2026-203) 시 사전 등록 margin δ.

---

## 4. 점검 절차

1. 외부 공개 직전 본 §2 14+1 항목을 `reports/checklists/<period>_<scope>.md` 에 기록.
2. 각 항목별로 충족 여부 + 인용 경로 (`evals/raw/<id>.json`, `evals/snapshots/<period>.json`, `external_assets/local_lora_g3/model_card.json` 등) 명시.
3. `evals/workarounds/_index.md` 의 critical open 항목 0개 확인.
4. metric 정의는 본 §3 만 인용. 자체 재정의 또는 변형 사용 금지.
5. 누락 항목 발견 시 외부 공개 보류.

---

## 5. 금지 규칙

- 본 §3 의 metric 정의를 외부 산출물에서 재정의·변형 금지.
- Workaround 미해소 상태 (critical open) 에서 외부 공개 금지.
- 통계 절차 (§3-7) 없이 supports/contradicts 단정 금지.
- 표본 수 (#11) 부족 상태에서 일반화 단정 금지.
