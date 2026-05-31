# Results Draft — ArtifactRouter on MotionGPT (G2) Stage 2-G

> **INTERNAL DRAFT** (2026-06-01). 논문 results 골격 정리. **외부 공개 (논문·발표·README) 전 [AGENTS.md §3-12](../AGENTS.md) reproducibility-checklist 14+1 항목 + 사용자 확인 의무.**
>
> 본 문서는 Stage 2-G (G2 real stress 중심) pipeline 의 evidence 통합. framing 은 [current_research_position.md](current_research_position.md), metric 분류는 [metric_provenance.md](metric_provenance.md), action space 는 [action_space_provenance.md](action_space_provenance.md) 가 single source. 본 문서는 결과 인용만.

---

## 0. 핵심 메시지 (한 문장)

> Broad-support 로 학습한 Q policy 와 real physical gate 를 결합하면, MotionGPT (G2) 출력에서 측정된 artifact 를 줄이면서도 standard text-to-motion quality (FID/R-Precision) 와 physical constraint 를 보존한다.

영문 (단일 출처): "A Q policy trained on a broad action-effect dataset, combined with a real physical-constraint gate, refines MotionGPT outputs by reducing measured artifacts while preserving standard text-to-motion quality and physical validity."

---

## 1. Setup

### 1-1. Data (Stage 2-G v2, [g2_real_stress_split_v2](../evals/splits/g2_real_stress_split_v2.json))

- **G2 pool 600** (MotionGPT, HumanML3D test captions, motion-group stratified 8 그룹). group-aware percentile band: stress (top-20% artifact) / normal / g2_clean_like.
- splits: train_g2_real 338 (stress 37 + normal 178 + clean_like 123) / calib 71 / g2_stress_holdout 65 / g2_natural_holdout 99 / clean_noharm_holdout 100 / reserve 27.
- auxiliary: synthetic_aux_train 112 / synthetic_diag_holdout 48 (controlled diagnostic, appendix).
- **층화 동기** (사용자 directive): walking-dominant overfit 방지. jump/run/turn/dance stress 1→10+ 확보.

### 1-2. Action / Policy

- action = discrete tool {FootLock, BoneProjection, VelocitySmoothing} × bounded continuous intensity u∈[0,1] (action_space_provenance §5-2). transition = (state, tool, u, after_state, gate_result, utility).
- M0 (broad-support: Stage A clean+g2+synthetic+boundary, n=16,830) / M1 (G2 real only, 13,497) / M2 (G2+synthetic aux, 17,193) / M3 (M2 + isotonic calib).
- two-head: Q_utility (HGB regressor) + P_safe (HGB classifier). 추론: P_safe≥0.5 후보 중 argmax Q_utility → **real physical gate 재검증** (rollback+STOP).

### 1-3. Metric provenance (AGENTS.md §3-20)

- **Category A (standard, authoritative)**: FID / R-Precision / MM-Dist / Diversity (HumanML3D tm2t, Guo 2022 CVPR).
- **Category C (internal routing reward, NOT final quality)**: NetGain, artifact_total_score.
- **physical**: PhysicalGateV0 5-evaluator regression-based hard gate (real validation, NOT reward term).

---

## 2. Result 1 — Safety: real gate guarantees 0% physical violation

closed-loop real-gate-ON ([reports/2026-06-01.md §3-7](../reports/2026-06-01.md), [snapshot](../evals/snapshots/rl2_closed_loop_ablation_stage2_v1.json)):

| policy | g2_stress | g2_natural | clean | synthetic |
|---|---|---|---|---|
| **M0/M1/M2/M3 + gate** | **0%** | **0%** | **0%** | **0%** |
| random + gate | 0% | 0% | 0% | 4.2% |
| heuristic + gate | 0% | 0% | 0% | **39.6%** |

→ Q policy + real gate = **0% violation 전 holdout**. rule-based(heuristic)/random 은 synthetic 에서 violation (per-step gate 의 cumulative-drift 한계 영역). **physical validity 는 gate 가 보장하되, 좋은 정책은 gate 에 막히는 낭비가 적어야 함** (§3).

---

## 3. Result 2 — Final quality: standard metric preserved/improved (Δ vs original)

Step 6 part 2 ([reports/2026-06-01.md §8](../reports/2026-06-01.md), [snapshot](../evals/snapshots/standard_metric_closed_loop_v1.json)). Δ = vs original/noop (paired, same state set):

| holdout | M0 ΔFID | M0 ΔR@1 | M0 Δartifact | 해석 |
|---|---|---|---|---|
| **synthetic_diag** | **−48.5** (55.7→7.3) | **+0.178** | (proxy 모호¹) | high-severity 강한 복원 |
| **g2_stress** | **+0.003 (neutral)** | −0.007 | **−0.014 (감소)** | artifact↓ + FID 보존 |
| g2_natural | +0.004 (~0) | ~0 | ~0 | 최소 개입 |
| **clean_noharm** | **0.000** | **0.000** | **0.000** | 완전 no-harm |

¹ synthetic 에서 artifact proxy(C) 는 +0.030 (증가) 인데 FID(A) 는 −48.5 (개선) — proxy 한계 (FootFloating, 부록 Z). **standard metric authoritative** (§3-20). 이 불일치 = NetGain/proxy 단독 final claim 금지의 evidence.

→ **g2_stress (real-distribution 핵심)**: M0 가 artifact 줄이며 FID 보존. heuristic/oracle 은 artifact 더 줄이나 FID 악화 (heuristic ΔFID +0.040). **M0 의 보수성 = artifact-FID trade-off 의 안전 지점**.

---

## 4. Result 3 — Learned policy value: M0 > baselines (paired, snapshot ≥ 2)

per-state NetGain paired test (Category C internal reward; final quality 는 §3). [5단계 리포트](../evals/reports/2026-06-01_g2_learned_vs_baseline.md):

| 비교 | snapshot1 (v2) | snapshot2 (v3) | 재현 |
|---|---|---|---|
| M0 > random+gate (g2_stress) | +0.022, p=1.2e-6, d=0.47 | +0.028, p=1.2e-7, d=0.50 | ✓ |
| M0 > random+gate (g2_natural) | +0.017, p=1.6e-17, d=0.87 | +0.019, p=3.2e-15, d=0.72 | ✓ |
| M0 > heuristic (g2_natural) | p=2.3e-13, d=0.59 | p=2.2e-12, d=0.30 | ✓ |
| M0 > heuristic (synthetic) | p=8e-11, d=0.99 | p=2e-5, d=0.71 | ✓ |
| M0 vs heuristic (g2_stress) | ns (0.62) | ns (0.061) | ✓ (both ns) |
| dense_oracle > M0 (전 holdout) | 유의 (headroom) | 유의 | ✓ |

→ **M0 > random: 전 real holdout 유의** (learnable routing 이 무통제 baseline 보다 의미 있음 — [H-2026-205](../evals/hypotheses/H-2026-205.md) 방향). **M0 vs heuristic**: natural/synthetic 유의 우위, **g2_stress 는 NetGain 동률** (단 §3 의 FID 는 M0 우위).

---

## 5. Result 4 — Reproducibility (12/12)

snapshot1 (split v2, seed 20260531) vs snapshot2 (split v3, seed 20260615, 독립 partition, holdout overlap stress 46%/natural 21%): **12/12 핵심 비교 방향+유의성 일치** (§4 표). effect size 같은 크기대. M0>random, M0 vs heuristic 분기, oracle headroom 모두 robust.

→ **snapshot ≥ 2 충족** ([AGENTS.md §3-9](../AGENTS.md) 분포+paired+다중 snapshot 의무). single-trial 결론 아님.

---

## 6. Efficiency — candidate evals / rejection rate (정책 품질의 숨은 축)

좋은 정책 = 좋은 후보를 **적은 검증 비용으로** 제안 (사용자 criterion). [reports/2026-06-01.md §3](../reports/2026-06-01.md):

| holdout | M0 rejection% | M1 rejection% | heuristic rejection% |
|---|---|---|---|
| clean | **0.0%** | 34.8% | 100% |
| g2_natural | **0.4%** | 1.2% | 8.1% |
| g2_stress | **4.3%** | 6.9% | 18.9% |

→ M0 가 일관 낮은 rejection (safe 후보 직접 제안). M1 (G2-only) 은 clean 에서 34.8% (unsafe 후보 반복 제안, gate 가 막음) — broad-support 의 효율 우위.

---

## 7. Limitations & Claim Scope (정직)

### 7-1. 방어 가능한 claim (현재 evidence)

- ✅ **real G2 holdout 에서**: M0+gate 가 (a) physical 0% violation, (b) artifact 감소, (c) standard quality (FID/R-Prec) 보존/개선, (d) random 대비 유의 (2 snapshot 재현), (e) clean no-harm.
- ✅ M0 (broad-support) > M1 (G2-only specialization) on OOD holdout (clean/synthetic).

### 7-2. 아직 말하면 안 되는 것

- ❌ **generator-agnostic** — G1 (MDM/MLD diffusion) 미구축. G2 단독 evidence 로 generator-agnostic 단정 금지 ([H-2026-206](../evals/hypotheses/H-2026-206.md) 미검증, [AGENTS.md §6-10](../AGENTS.md)).
- ❌ **모든 모션 항상 개선** — g2_natural 은 최소 개입 (이미 좋은 motion), clean 은 무손상이 목표.
- ❌ **synthetic-only 최종 성능** — synthetic 은 controlled diagnostic (§3-17). synthetic FID −48.5 는 diagnostic finding, 최종 성능 sole evidence 아님.
- ❌ **dense oracle 수준 artifact 최대 감소** — M0 는 oracle 대비 headroom 존재 (g2_stress d≈0.43, 2 snapshot). 최적 아님.

### 7-3. 추가 limitation

- NetGain/artifact = Category C (internal). final quality claim 은 Category A (FID/R-Prec) + perceptual 로만.
- 재현 snapshot v2/v3 = **동일 600 pool 의 다른 partition** (완전 독립 pool 아님). 더 강한 재현 = 새 generation pool.
- perceptual validation 미완 (b1 sub-tier만, [reports/2026-05-28.md §12](../reports/2026-05-28.md) G-4 GIF). standard metric + perceptual 동반이 최종.
- closed-loop oracle = step-1 dense (단일 step ceiling). sequence oracle gap 별도.

---

## 8. Open items (사용자 결정 게이트)

| item | 상태 |
|---|---|
| H-2026-205 status (`active`→`supported`) | snapshot≥2 충족 + 12/12 재현. **사용자 승인 게이트** (§3-11) — Agent 단독 금지 |
| g2_stress Stage 3 (continuous argmax) | oracle headroom (d≈0.43) 회수, FID 보존 조건 |
| MDM/MLD (G1) 구축 | generator-agnostic 검증 (H-2026-206), heavy setup |
| perceptual b2/b3 | quality-validated evidence 보강 |
| 외부 공개 | §3-12 reproducibility-checklist 14+1 + 사용자 확인 |

---

## 9. Intent-Reconciliation (AGENTS.md §3-23)

- **Intent**: 사용자 추천 순서 (재현→**결과 섹션 초안**). snapshot 1+2 동조 확인 후 논문 results 골격 통합.
- **Acceptance criteria**: 4 result (safety/quality/policy-value/reproducibility) + efficiency + limitation + claim scope, single-source cross-link, 외부 공개 아닌 내부 draft 명시.
- **Actual outcome**: results draft 작성 (§0-8). 모든 수치 기존 snapshot/report cross-link. claim 범위 정직 (G1 미검증, oracle headroom, Category C 구분, single-pool 재현 명시). H-205 status 미전환 (게이트 준수).
- **Verdict**: **`aligned`** (결과 통합 + claim 범위 정직 + 게이트 준수). 외부 공개 시 §3-12 별도.
