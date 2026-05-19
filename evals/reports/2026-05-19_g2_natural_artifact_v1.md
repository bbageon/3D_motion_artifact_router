# eval-compare — 2026-05-19 — Action 2: G2 (MotionGPT) natural artifact 분포 측정

본 리포트는 [`eval-compare SKILL §6`](../../.claude/skills/eval-compare/SKILL.md) 의 5단계 형식을 사용자 framing 검증에 적용한 결과이다.

> **범위 한정**: 본 리포트는 [H-2026-204](../hypotheses/H-2026-204.md) 의 **G2 측 active scope** 의 자연 artifact 분포만을 측정. RQ1/RQ2 의 정량 검증은 후속 oracle (Protocol B) 측정 필요. 본 measurement 는 **informational + 사용자 framing 검증**.
>
> **사용자 framing** ([reports/2026-05-19d.md](../../reports/2026-05-19d.md)) — "큰 이득은 multi-artifact case 또는 G2 natural artifact 에서 다시 검증해야 한다". 본 리포트는 G2 natural artifact 분포의 존재 여부를 직접 측정.

---

## 1. 제시한 가설

본 비교는 다음 활성 가설의 일부 측면을 평가한다.

- [H-2026-204](../hypotheses/H-2026-204.md) **G2 active scope** — track_scope=[G1, G2] 의 G2 측 검증.
- [H-2026-203](../hypotheses/H-2026-203.md) **No-harm on high-quality** — G2 가 high-quality 인지 아니면 자연 artifact 보유인지 분류.

사용자 framing 의 measurable 형식:
- **G2 (MotionGPT) generated motion 의 자연 artifact severity 분포가 HumanML3D GT 의 그것과 다른가**?
- 다르다면 어떤 evaluator 에서 (foot / bone / jitter) 더 자주 발생하는가?

---

## 2. 실험 평가

### 사용 baseline

| 측 | Baseline ID | snapshot | n_samples |
|---|---|---|---|
| **G2 natural** | `g2_natural_artifact_v1` | [`evals/snapshots/g2_natural_artifact_v1.json`](../snapshots/g2_natural_artifact_v1.json) | 30 (split `g2_natural_v1`) |
| **HumanML3D GT (reference)** | `baseline_holdout_v2` | [`evals/snapshots/baseline_holdout_v2.json`](../snapshots/baseline_holdout_v2.json) | 50 (split `holdout_v2`) |

### Generator + inference 정보

- **G2 generator**: MotionGPT (OpenMotionLab), checkpoint `motiongpt_s3_h3d.tar` (1.24GB, sha256 `0d8a9e1a0c3bd15e...`).
- **transformers/tokenizers**: 4.30.2 / 0.13.3 ([W-2026-001](../workarounds/W-2026-001.md) resolved).
- **GPU**: NVIDIA GeForce RTX 5090 (cap 12.0), torch 2.11.0+cu128.
- **Prompts**: 공식 `external_assets/MotionGPT/demos/t2m.txt` 의 30 detailed prompts (60+ 단어급).
- **Seeds**: base_seed=42, per-sample seed = 42 + index (42-71).
- **n_frames hint**: 60 (실제로는 t2m 의 LM 이 자율적으로 60-196 frames 동적 생성).

### Evaluator / config

- **evaluator severity versions**:
  - FootFloatingEvaluator `1.2.0-2026-05-18`
  - BoneLengthEvaluator `1.0.0-2026-05-13`
  - VelocityJitterEvaluator `1.1.0-2026-05-18`
- **evaluator config hashes**: `g2_natural_artifact_v1.json` 의 `evaluator_config_hashes` (baseline_holdout_v2 와 동일 = 동일 evaluator).

### Raw records / 명령

- G2 batch: `external_assets/g2_generated_v1/motion_{001..030}.{npy,json}` + `_batch_summary.json`.
- G2 evaluator raw records: `evals/raw/*g2_natural_artifact_v1*.json` (30 records).
- 명령:
  ```
  python -m tools.g2_generate_batch --n-samples 30 --n-frames 60 --base-seed 42 \
      --output-dir external_assets/g2_generated_v1
  python -m tools.g2_evaluator_profile \
      --g2-batch-dir external_assets/g2_generated_v1 \
      --task-id g2_natural_artifact_v1 \
      --split-id g2_natural_v1 \
      --raw-output-dir evals/raw \
      --output evals/snapshots/g2_natural_artifact_v1.json
  ```

### 가동 조건 ([eval-compare SKILL §2](../../.claude/skills/eval-compare/SKILL.md))

- **trial ≥ 20**: ✓ (G2 n=30, HumanML3D GT n=50).
- **snapshot ≥ 2**: ✓ (g2_natural_artifact_v1 + baseline_holdout_v2).

본 리포트는 **informational distribution comparison** 이며 supports/contradicts 단정 아님 (NetGain 측정 안 함).

---

## 3. 실험적 근거

### 3-1. Motion length distribution (G2)

```
min:    60 frames
max:   196 frames
mean:  154.3 frames
median: 172 frames
```

→ G2 가 **정상 length 분포** (약 3-10초 motion at 20fps). [W-2026-001 RESOLVED](../workarounds/W-2026-001.md) 이전의 4-frame degenerate 와 대비.

### 3-2. Per-evaluator severity 분포

| Evaluator | G2 natural (n=30) | HumanML3D GT (n=50) | Δ |
|---|---|---|---|
| **FootFloatingEvaluator** | 17 low / 0 medium / **4 high** (n_reports=21) | 41 low / 3 medium / 4 high (n_reports=48) | G2 가 사실상 비슷한 high 빈도 (4 vs 4), 단 reports 비율 낮음 (70% vs 96%) |
| **BoneLengthEvaluator** | **28 low / 2 medium** / 0 high (n_reports=30) | **0 reports** (모든 sample 에서 bone length 위반 없음) | **결정적 차이** — G2 가 GT 에 없는 bone length artifact 다수 보유 |
| **VelocityJitterEvaluator** | 7 low / 0 medium / 0 high (n_reports=7) | 29 low / 0 medium / 0 high (n_reports=29) | G2 가 jitter 적게 발생 (23% vs 58%) |

### 3-3. Score distribution (max score per sample)

| Evaluator | G2 max score | HumanML3D GT max score |
|---|---|---|
| FootFloating median / p90 / max | 0.0250 / 0.339 / **0.583** | 0.031 / 0.207 / **0.612** |
| BoneLength median / p90 / max | 0.0316 / 0.0440 / **0.0632** | N/A (0 reports) |
| VelocityJitter median / p90 / max | 0.0214 / 0.0268 / **0.0332** | 0.023 / 0.035 / **0.047** |

해석:
- FootFloating 의 max score 가 G2 (0.583) ≈ GT (0.612) — 본질적 outlier 빈도 비슷.
- BoneLength 가 GT 0 reports → **G2 의 BoneLength artifact 는 100% generator-induced** (synthetic-like artifact 자연 생성).
- VelocityJitter 는 G2 가 GT 보다 더 적은 보고 + 더 작은 score — G2 의 generation 이 잘 smoothed 됨.

### 3-4. Multi-artifact co-occurrence (G2 sample 단위)

per-sample 의 evaluator triple 분포 (foot, bone, jitter):

| sample 별 artifact 조합 | count |
|---|---|
| foot=low/None, bone=low, jitter=None | 다수 (>15) |
| foot=high + bone=low | 3 (motion_006, 008, 028) |
| foot=high + bone=medium | 1 (motion_007) |
| foot=low + bone=medium + jitter=low | 1 (motion_019) |

→ **multi-artifact co-occurrence 가 자연적으로 발견됨** (e.g. motion_007: foot_high + bone_medium 동시). 사용자 framing 의 "multi-artifact case" 가 G2 natural distribution 에 존재.

---

## 4. 가설 평가

### H-2026-204 G2 active scope — 기본 검증

#### 판정: **G2 측 분포 측정 가능 + multi-artifact case 존재 확인**.

본 리포트는 NetGain 측정 안 함 (Protocol A 의 clean GT 없음, Protocol B oracle 후속 필요) 이므로 RQ1/RQ2 의 직접 supports/contradicts 단정 안 함. 그러나 다음 사실 확인:

1. **G2 가 evaluator-detectable artifact 보유** (특히 bone length: 30/30 reports).
2. **multi-artifact case 자연 발생** (foot+bone 동시 보유 4 samples).
3. **GT 와 다른 distribution** (bone length 만 G2 에 generator-induced).

#### §3 인용

- Motion length: [§3-1](#3-1-motion-length-distribution-g2)
- Severity 분포: [§3-2](#3-2-per-evaluator-severity-분포)
- Score 분포: [§3-3](#3-3-score-distribution-max-score-per-sample)
- Multi-artifact co-occurrence: [§3-4](#3-4-multi-artifact-co-occurrence-g2-sample-단위)

### H-2026-203 — No-harm on high-quality

#### 판정: **추가 조건 분리 필요** — G2 가 high-quality tier 가 아님이 노출됨.

- G2 의 bone length artifact 가 GT 에 없는 generator-induced artifact → G2 가 **본격적 "high-quality"** 가 아닐 가능성 시사.
- 명세 §9.2 의 quality-tier 정의 ("G1=high-quality, G2=token-based") 와 일관 — G2 는 token-based generator 이지 high-quality tier 가 아님.
- H-2026-203 의 "No-harm on high-quality" 검증은 **G1 (diffusion: MDM/MLD) 도입 후** 정식 수행. G2 는 high-quality reference 아님.

---

## 5. 다음 스텝

### 5-1. 즉시 가능한 후속 (사용자 plan 의 Step 3)

- **Action 3 진행** — rule-based (B5) measurement + B5 vs B2 paired test (RQ1 임계 2). calibration_v1 데이터 재사용으로 1-2 turn 안에 완료 가능.

### 5-2. Action 2 의 후속 (Protocol B oracle 측정)

본 리포트는 G2 natural artifact **분포** 측정만. 사용자 framing 의 "큰 이득 검증" 은 **Protocol B oracle** 필요:
- Clean GT 없음 → MPJPE 기반 FidelityLoss 불가능.
- 대안: NetGain' = ArtifactReduction - β·CorrectionMagnitude - γ·ToolCallCost (α·FidelityLoss 항 제외, 또는 Protocol C distributional).
- G2 natural artifact 가 있는 sample 들 (foot=high 4 samples, bone=medium 2 samples) 에 oracle (single_step + sequence) 측정 → closed-loop vs single-step 의 G2 natural 에서의 advantage 측정.
- 본 측정은 **별도 turn** 으로 진행 (사용자 plan 의 Action 2 deliverable 의 후속 또는 새 H-id).

### 5-3. AGENTS.md / 가설 본문 변경 여부

- H-2026-204 본문: 변경 없음. G2 측 검증의 첫 단계 (분포 측정) 완료.
- H-2026-203 본문: 변경 없음. G2 가 high-quality 가 아닌 sample 임은 명세 §9.2 와 일관, framing 자체는 그대로.

---

## 부록 A — 우회·간접 해결 ledger 인용

| W-id | severity | status | 한 줄 | 본격 학습 영향 |
|---|---|---|---|---|
| [W-2026-001](../workarounds/W-2026-001.md) | critical | **resolved** | MotionGPT 의 transformers 5.x weight tying 처리 broken → 4.30.2 + tokenizers 0.13.3 downgrade | 본 G2 측정의 사전 조건. resolved 이후 정상 측정 가능. |

## 부록 B — G2 generation 메타 (sample-level)

`evals/raw/*g2_natural_artifact_v1*.json` 의 sample 별 metadata:
- `generator_id`: `G2_motiongpt_motiongpt_s3_h3d` (모두 동일).
- `generator_class_hash`: `1effc6df6a6be0a9_f9a03a11cadf5c08` (wrapper source hash + checkpoint partial hash).
- `generator_seed`: 42-71 (per-sample).
- `generator_prompt`: t2m.txt 의 line 1-30 (60+ 단어급 detailed prompt).
- `fps`: 20.

---

## 부록 C — 본 리포트의 한계 (자기 점검)

1. **분포 측정만, NetGain 미측정** — Protocol B oracle 측정 후속 필요. RQ1/RQ2 의 정량 supports 단정 안 함.
2. **HumanML3D GT 와 split 다름** — G2 의 30 samples 와 GT 의 50 samples 가 disjoint, paired test 불가. 단 분포 자체 비교는 informational 의미 충분.
3. **G2 motion 의 quality control 없음** — MotionGPT 의 generation 은 stochastic, seed 변경 시 다른 결과. n_seeds≥3 의 robust 검증은 후속.
4. **단일 generator (G2 only)** — G1 (MDM/MLD) 도입 후 generator-tier 비교 가능. 본 리포트는 G2 측만.
