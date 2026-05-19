# baselines_current.md — Operational baseline registry

> 본 파일은 **현재 시점의 operational baselines** 를 단일 출처로 박제한다. 5단계 비교 리포트 ([`eval-compare SKILL`](../../.claude/skills/eval-compare/SKILL.md)) 의 §2 (실험 평가) 와 §3 (실험적 근거) 에서 인용할 baseline 정의는 본 registry 만 인용한다.
>
> 본 registry 는 append-only **가 아니다**. baseline 이 교체되면 본 파일을 update 하되 git diff 로 변경 추적. 과거 baseline 은 `evals/snapshots/<task_id>.json` 자체가 history 이므로 본 파일에서는 current 만 유지.
>
> **마지막 갱신**: 2026-05-19 (Step 3 of [reports/2026-05-19c.md](../../reports/2026-05-19c.md) 의 다음 단계).

---

## 1. 가설 매핑

본 registry 의 baseline 들이 어떤 가설의 어떤 RQ 를 평가하는 데 쓰이는지 명시.

| Baseline ID | 평가 대상 RQ | 가설 |
|---|---|---|
| `oracle_single_step_v2` | RQ1, RQ2 의 single-step 측 | [H-2026-204](../hypotheses/H-2026-204.md) — artifact-conditioned tool selection 우위 + closed-loop ≥ single-step |
| `oracle_sequence_v1_1_tiebreak` | RQ2 의 sequence (closed-loop fair upper bound) 측 | [H-2026-204](../hypotheses/H-2026-204.md) |
| `baseline_holdout_v2` | hold-out (50 samples) artifact 분포 reference | [H-2026-203](../hypotheses/H-2026-203.md) (no-harm 측정 기준) |
| `tool_effect_matrix_v1` | tool effect matrix (E2 ablation) | [H-2026-204](../hypotheses/H-2026-204.md) RQ1 의 ablation |
| `netgain_weight_grid_v2` | NetGain weight calibration sensitivity | calibration meta — 모든 RQ |

**가설 한 줄 풀이** (AGENTS.md §2-5-1 의무):
- **H-2026-203** — high-quality motion (SOTA generator output) 에 대한 no-harm 운영 특성 — 분포·semantic·fidelity 훼손 없음.
- **H-2026-204** — RQ1 (artifact-conditioned routing > fixed post-processing) + RQ2 (closed-loop refinement ≥ single-step 의 artifact reduction vs fidelity trade-off).

---

## 2. Operational baselines (current)

### 2-1. `oracle_single_step_v2` — single-step oracle baseline

- **snapshot**: [`evals/snapshots/oracle_single_step_v2.json`](../snapshots/oracle_single_step_v2.json).
- **raw records**: `evals/raw/*oracle_single_step_v2*.json` (30 records).
- **split_id**: `calibration_v1` (30 HumanML3D samples, deterministic by trial_id).
- **seed**: 42.
- **NetGain calibration**: `calibrated_protocol_a_v1` (α=5.0, β=0.0, γ=0.0).
- **evaluator severity versions**:
  - FootFloatingEvaluator: `1.2.0-2026-05-18` (config hash `fc58a7df...`)
  - BoneLengthEvaluator: `1.0.0-2026-05-13` (config hash `08866688...`)
  - VelocityJitterEvaluator: `1.1.0-2026-05-18` (config hash `79e098dd...`)
- **tools tested**: FootLockTool, BoneProjectionTool, VelocitySmoothingTool (3-tool registry, strengths small/medium/large).
- **artifact kinds**: foot_floating, bone_stretch_right_arm, global_jitter.
- **재현 명령**:
  ```
  python -m tools.run_oracle_single_step \
      --data-dir external_assets/HumanML3D/new_joints \
      --task-id oracle_single_step_v2 \
      --split-id calibration_v1 \
      --seed 42 \
      --n-samples 30
  ```

### 2-2. `oracle_sequence_v1_1_tiebreak` — sequence oracle (closed-loop fair upper bound)

- **snapshot**: [`evals/snapshots/oracle_sequence_v1_1_tiebreak.json`](../snapshots/oracle_sequence_v1_1_tiebreak.json).
- **raw records**: `evals/raw/*oracle_sequence_v1_1_tiebreak*.json` (30 records).
- **split_id**: `calibration_v1` (single_step 와 동일 sample set — paired test 가능).
- **seed**: 42.
- **NetGain calibration**: `calibrated_protocol_a_v1` (α=5.0, β=0.0, γ=0.0).
- **search params**: `max_depth=3`, `top_k=10`, score 비감소 pruning (`score_increase_tolerance=0.01`).
- **tie-break rule** (Step 1 fix, 2026-05-19): NetGain 동률 시 length 짧은 sequence 우선 (Occam 간결성). `key=(netgain_provisional, -length)`.
- **evaluator severity versions / tool class hashes**: oracle_single_step_v2 와 동일 (config hashes 단일 출처 보장).
- **재현 명령**:
  ```
  python -m tools.run_oracle_sequence \
      --data-dir external_assets/HumanML3D/new_joints \
      --task-id oracle_sequence_v1_1_tiebreak \
      --split-id calibration_v1 \
      --seed 42 \
      --n-samples 30 \
      --max-depth 3 --top-k 10
  ```

### 2-3. `baseline_holdout_v2` — hold-out artifact distribution

- **snapshot**: [`evals/snapshots/baseline_holdout_v2.json`](../snapshots/baseline_holdout_v2.json).
- **n_samples**: 50.
- **split_id**: hold-out (calibration_v1 와 disjoint).
- **seed**: 99.
- **목적**: H-2026-203 (no-harm) 의 분포 reference + Step 4 evaluation 의 false-positive baseline.

### 2-4. `tool_effect_matrix_v1` — tool effect matrix (E2 ablation)

- **snapshot**: [`evals/snapshots/tool_effect_matrix_v1.json`](../snapshots/tool_effect_matrix_v1.json).
- **n_samples**: 30 (calibration_v1).
- **목적**: H-2026-204 RQ1 의 ablation — 각 artifact 별 best tool 식별 + cross-evaluator side effects 측정.

### 2-5. `netgain_weight_grid_v2` — calibration sensitivity

- **snapshot**: [`evals/snapshots/netgain_weight_grid_v2.json`](../snapshots/netgain_weight_grid_v2.json).
- **grid**: α ∈ {0, 0.25, 0.5, 1, 2, 5, 10, 20, 50}, β ∈ {0, 0.25, 0.5, 1, 2}, γ ∈ {0, 0.1, 0.5, 1} (180 조합).
- **best**: α=50.0, β=1.0, γ=0.0, ρ=0.9458 (Spearman vs quality proxy = -FidelityLoss).
- **caveat — circular calibration** ([reports/2026-05-19c.md §2-2](../../reports/2026-05-19c.md) 의무 참조):
  - quality proxy = -FidelityLoss 일 때 NetGain α 가 커질수록 Spearman → 1.0 으로 trivial 수렴.
  - 따라서 **v2 best (α=50) 는 calibration constant 로 promote 하지 않는다**.
  - **현재 operational calibration 은 v1 (α=5.0)** 으로 유지 (status `calibrated_protocol_a_v1`).
  - v1 도 grid boundary 였음을 명시 인정. 정식 calibration 은 perceptual rating 수집 후 재개.

---

## 3. NetGain calibration constant (current)

`orchestrator/oracle_single_step.py` 에 정의된 상수:

```python
DEFAULT_PROVISIONAL_NETGAIN_WEIGHTS = {"alpha": 1.0, "beta": 1.0, "gamma": 0.1}
CALIBRATED_PROTOCOL_A_NETGAIN_WEIGHTS_V1 = {"alpha": 5.0, "beta": 0.0, "gamma": 0.0}
```

- **default (provisional)** — debugging / unit test 용. raw record 에 `netgain_weight_status: "provisional"` 으로 박제. AGENTS.md §6-11 에 따라 결과 인용 금지.
- **calibrated_protocol_a_v1** — Step 4 evaluation 에서 사용할 operational weight. raw record 에 `netgain_weight_status: "calibrated_protocol_a_v1"` 박제. **caveat — circular calibration source** (§2-5 참조).

**status tag hierarchy** (낮은 → 높은):
1. `provisional` — debug 만.
2. `calibrated_protocol_a_v1` — 본 registry 의 operational. 외부 인용 시 caveat 동봉.
3. (미존재) `calibrated_perceptual` — perceptual rating 기반. 후속.

---

## 4. Split definitions

| Split ID | n_samples | Source | Used by |
|---|---|---|---|
| `calibration_v1` | 30 | HumanML3D `new_joints` 의 deterministic shuffle (seed=42 sample) | oracle_single_step_v2, oracle_sequence_v1_1_tiebreak, tool_effect_matrix_v1 |
| `holdout_v2` | 50 | HumanML3D `new_joints` 의 calibration_v1 와 disjoint (seed=99) | baseline_holdout_v2 |

**중요**: calibration_v1 에서 측정한 NetGain weight 를 같은 split 으로 evaluation 하면 over-fitting. Step 4 의 본격 evaluation 은 hold-out (또는 G1/G2 generator output) 으로 진행 필수.

---

## 5. Open caveats (Step 4 evaluation 전 인지 의무)

1. **Circular calibration source** (§2-5) — Spearman vs -FidelityLoss 는 α 의 trivially optimal. v1 α=5.0 은 grid boundary 였으나 operational 으로 채택 (full sensitivity 분석은 [`per_sample_gap_alpha_sensitivity_v1.json`](../snapshots/per_sample_gap_alpha_sensitivity_v1.json)).
2. **sequence top_k truncation** ([2026-05-19c §5-1](../../reports/2026-05-19c.md)) — sequence raw 가 v1 weight 기준 top_k 만 저장. α-sensitivity 분석에서 α ≠ 5 일 때 foot_floating 의 n_loss > 0 은 truncation artifact (mathematical violation 아님).
3. **G1/G2 generator output 미측정** — 현재 모든 baseline 이 HumanML3D GT (`humanml3d_gt`) 에 synthetic injection 적용. G1 (diffusion: MDM/MLD), G2 (token-based: MotionGPT) output 의 측정은 후속.
4. **Hold-out evaluation 미실행** — calibration_v1 으로만 측정됨. Step 4 의 본격 evaluation 에서 holdout_v2 + paired test 필수.
5. **Single seed** — seed=42 (calibration), 99 (holdout). N≥3 generation 평균 (AGENTS.md §5-2 NF4 결정성 의무) 은 G3 generator 사용 시 적용 — 현재 generator 미사용으로 보류.

---

## 6. 갱신 규칙

- baseline 교체 (예: oracle_single_step_v2 → v3) 시:
  1. 새 snapshot 산출.
  2. 본 registry 의 해당 section update.
  3. 변경 commit 의 commit message 에 reason 명시.
  4. 5단계 비교 리포트 ([`eval-compare SKILL §6`](../../.claude/skills/eval-compare/SKILL.md)) 에 baseline 변경의 영향 분석 첨부.
- calibration constant 변경 (v1 → v2) 시:
  1. AGENTS.md §3-11 사용자 승인 게이트 (가설 status 전환과 동급).
  2. `orchestrator/oracle_single_step.py` constant 추가 (기존 v1 보존, append-only).
  3. 본 registry §3 update.
  4. circular calibration source 해소 증거 (perceptual rating 등) 필수.
