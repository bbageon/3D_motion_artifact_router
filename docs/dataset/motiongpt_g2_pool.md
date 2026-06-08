# Dataset Card — G2 (MotionGPT 생성 pool)

> 본 카드 = ArtifactRouter 의 primary **real-distribution** test bed 의 단일 출처. 모든 G2 실험(AR-012~018, AR-029, AR-043 등)은 본 pool/split 을 사용한다.
> 핵심: **stress / natural 은 모두 실제 MotionGPT 출력** 이며, stress 는 artifact 프로파일 기반 **selection** 이지 corruption 이 아니다 (corruption track 은 별도 synthetic).

---

## 1. 출처 (provenance)

| 항목 | 값 |
|---|---|
| **Generator** | MotionGPT (Jiang et al., **NeurIPS 2023**), 공식 구현 — token-based / VQ codebook text-to-motion |
| **실행 env** | conda `mgpt` ([generators/motiongpt_wrapper.py](../../generators/motiongpt_wrapper.py)) |
| **prompt 출처** | HumanML3D (Guo et al., CVPR 2022) test caption |
| **format** | canonical SMPL-22 `[T,22,3]`, fps=20, root-relative (PELVIS=origin) — AGENTS.md §3-1 |
| **저장** | `external_assets/g2_generated_hml3d_*/` (npy + json meta). **gitignored** (재생성 가능, §6-1) |

### 두 generation pool

| pool 디렉토리 | n | seed | 비고 |
|---|---:|---|---|
| `g2_generated_hml3d_test300_seed20260527` | 300 | 20260527 | HumanML3D test300 caption |
| `g2_generated_hml3d_balanced300_seed20260531` | 300 | 20260531 | motion_group balanced caption |
| **합계** | **600** | | trial_id 의 `/` prefix 로 pool 구분 (`_resolve_g2`) |

---

## 2. Split 구성 (`g2_real_stress_split_v2/v3`)

[evals/splits/g2_real_stress_split_v3.json](../../evals/splits/g2_real_stress_split_v3.json) — group-balanced stratified, seed=20260615.

| split | spec n | distribution | 용도 |
|---|---:|---|---|
| `train_g2_real` | 300 | g2_natural (real) | 학습 |
| `calib_g2_real` | 75 | g2_natural (real) | calibration |
| `g2_stress_holdout` | 100 | g2_natural (real), band=stress | **artifact-rich holdout** |
| `g2_natural_holdout` | 100 | g2_natural (real), band=natural | 대표 holdout |
| `reserve` | 25 | g2_natural (real) | 예비 |
| `clean_noharm_holdout` | 100 | HumanML3D GT | **no-harm reference** (corruption 금지 대상) |
| `synthetic_aux_train` / `synthetic_diag_holdout` | 160 | synthetic corruption | **controlled diagnostic** (§3-17) |

> **evaluated n ≠ spec n**: standard/physical metric eval 에서 caption·tm2t feature·전 policy motion 이 모두 valid 한 state 만 paired 사용 → **stress n=65, natural n=99** (AR-016/AR-029). 보고 시 evaluated n 사용.

### stress vs natural 의 의미 + 실제 band 비율

stress/natural 은 `g2_real_stress_profile_v2` 의 artifact 프로파일로 600개 실제 출력을 **층화**한 것. stress = 흠집 많은 실제 MotionGPT 출력, natural = 대표적 실제 출력. **둘 다 corruption 아님.**

**full pool band 실제 비율 (n=600)**: stress **124 (20.7%)** / normal 353 (58.8%) / g2_clean_like 123 (20.5%) → low-artifact(normal+clean_like) **476 (79.3%)**.

> ⚠️ **평가 단위 주의 (§3-22)**: stress 는 proxy 로 고른 enriched subset(20.7%). **대표 성능·no-harm 주장은 prevalence-weighted full 분포로** 보고하고, stress/natural 은 진단 보조표로만 인용한다. holdout 65:99 단순 concat 은 stress 39.6% 로 **~2배 과대표집**. 대표 집계: [evals/snapshots/representative_aggregate_g2_v1.json](../../evals/snapshots/representative_aggregate_g2_v1.json), 해석 [../findings/motiongpt_implications.md §0](../findings/motiongpt_implications.md).

---

## 3. 통계 (측정값)

### 3-1. Standard metric (Category A — tm2t, HumanML3D/Guo 2022) — [snapshot](../../evals/snapshots/standard_metric_closed_loop_v1.json)

| split | n | original FID↓ | R@1↑ | 해석 |
|---|---:|---:|---:|---|
| clean (HumanML3D GT) | 100 | **0.717** | 0.563 | 자연 분포 기준 |
| **g2_natural** | 99 | **0.846** | 0.424 | **clean 과 거의 동일** — MotionGPT 가 분포적으로 near-GT |
| **g2_stress** | 65 | **8.949** | 0.373 | natural 의 ~10배 — 실제로 분포적으로 나쁜 무더기 존재 |
| synthetic_diag | 48 | 55.742 | 0.145 | corruption (큰 distributional gap) |

> **caveat**: FID per-holdout n=48~100 (작음) → upward-biased/noisy. 절대값보다 split 간 방향·Δ 가 robust (§3-9).

### 3-2. Artifact proxy (Category C) — AR-043 [snapshot](../../evals/snapshots/dataset_issue_prevalence_audit_v1.json)

| split | artifact_total mean | 핵심 issue (prevalence) |
|---|---:|---|
| g2_stress | **0.0775** | FootFloating 80% · BoneLength 90.8% |
| g2_natural | **0.0189** | FootFloating 10.1% · BoneLength 83.8% |
| clean_noharm | 0.0324 | FootFloating 26% (GT 도 proxy 가 잡힘 → "artifact-free" 아님) |

**stress/natural artifact ratio = 4.10x.** evaluator 신뢰도(AR-043): **FootFloating=reliable**, BoneLength/BoneLengthCV=moderate~no-discrimination(전 split 100% fire), VelocityJitter+4 physical=weak_signal.

---

## 4. Caveat / 인용 규약

1. **stress = selection, not corruption** — 실제 MotionGPT 출력. corruption 은 synthetic track(별도).
2. **proxy category 분리** — artifact = Category C (최종 quality 단독 근거 금지, §3-20). FID/R-Prec = Category A.
3. **small-n FID** — holdout FID 는 표본 작음 → 방향 신호로만.
4. **gitignored** — npy 는 git 추적 밖. 재현 시 위 seed 로 재생성. split json 은 추적됨.
5. **single generator** — 본 pool 은 MotionGPT 단독. generator 일반화는 §6-10 (≥2 generator) — 사용자가 별도 확보 예정.

## 5. 재생성

```text
caption → MotionGPT (mgpt env, generators/motiongpt_wrapper.py) → canonical SMPL-22
→ artifact profile (g2_real_stress_profile) → stratified split (g2_real_stress_split)
```
상세: [docs/setup.md](../setup.md) (env), [AGENTS.md §2](../../AGENTS.md) (실행).
