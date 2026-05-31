# 5단계 비교 리포트 — G2 real holdout: learned policy (M0) vs baselines

> 생성: 2026-06-01 (snapshot 2 추가 갱신). 트랙: machine Compare (eval-compare 5단계).
> raw: snapshot1 [rl2_g2_paired_stats_v1.json](../snapshots/rl2_g2_paired_stats_v1.json) (split v2, seed 20260531) + snapshot2 [rl2_g2_paired_stats_v3.json](../snapshots/rl2_g2_paired_stats_v3.json) (split v3, seed 20260615, 독립 partition).
> **snapshot ≥ 2 충족** (v2 + v3, holdout overlap stress 46% / natural 21%). **12/12 비교 방향+유의성 재현 (§3-5)**. 단 가설 status 전환은 여전히 AGENTS.md §3-11 사용자 승인 게이트 — 본 리포트는 evidence 제시이며 status 전환 아님.

## 1. 제시한 가설

- [H-2026-205](../hypotheses/H-2026-205.md) (RQ3 — learnable routing: supervised/contextual-bandit selector 가 rule-based baseline 대비 net gain 의미 있게 개선).
- [H-2026-204](../hypotheses/H-2026-204.md) (RQ1+RQ2 — artifact-conditioned tool selection + closed-loop 가 fixed post-processing 대비 우위).

본 리포트의 비교: **M0 (broad-support learned Q policy) + real gate** vs **{random+gate, heuristic+gate (rule-based), STOP-only, M1/M2 (학습 데이터 ablation), dense_oracle_step1 (ceiling)}**.

## 2. 실험 평가 (사용 데이터)

- split: [g2_real_stress_split_v2](../splits/g2_real_stress_split_v2.json) (seed 20260531).
- holdout (train 에 미포함): g2_stress_holdout (n=65), g2_natural_holdout (n=99), clean_noharm_holdout (n=100), synthetic_diag_holdout (n=48, appendix).
- metric: **per-state NetGain** (Category C internal routing reward — final quality 는 Step 6 part 2 의 standard FID/R-Prec 가 authoritative). G2=Protocol B, clean/synthetic=Protocol A.
- 통계: Wilcoxon signed-rank (paired), Cohen's d (paired) + rank-biserial, bootstrap 95% CI (N=1000). closed-loop max_depth 3, real physical gate.

## 3. 실험적 근거 (M0 − baseline, paired)

### 3-1. g2_stress_holdout (n=65) — real generator stress

| vs baseline | M0−base Δ | Wilcoxon p | Cohen d | M0>base / base>M0 | boot CI95 |
|---|---|---|---|---|---|
| random+gate | **+0.0219** | **1.2e-06** | +0.465 | 56/9 | [+0.0074, +0.0232] |
| heuristic+gate | −0.0055 | 0.619 (ns) | −0.142 | 24/27 | [−0.0008, +0.0000] |
| STOP-only | +0.0084 | 0.052 (borderline) | +0.307 | 14/13 | [0, 0] |
| M1 (G2-only) | −0.0056 | 0.040 | −0.220 | 14/27 | [−0.0006, 0] |
| M2 | −0.0030 | 0.946 (ns) | −0.084 | 14/20 | [0, 0] |
| dense_oracle | −0.0165 | 1.4e-14 | −0.452 | 0/47 | [−0.0062, −0.0011] |

### 3-2. g2_natural_holdout (n=99) — over-correction 방지

| vs baseline | M0−base Δ | Wilcoxon p | Cohen d | M0>base | boot CI95 |
|---|---|---|---|---|---|
| random+gate | **+0.0165** | **1.6e-17** | **+0.871** | 95/2 | [+0.0085, +0.0139] |
| heuristic+gate | **+0.0055** | **2.3e-13** | **+0.587** | 73/9 | [+0.0036, +0.0051] |
| STOP-only | −0.0002 | 0.003 | −0.073 | 4/24 | [0, 0] |
| dense_oracle | −0.0018 | 1.6e-12 | −0.666 | 0/66 | [−0.0009, −0.0001] |

### 3-3. clean_noharm_holdout (n=100) — no-harm

모든 정책 NetGain ≈ 0 (clean correction = negligible magnitude). M0 vs STOP p=4.7e-10 이나 Δ=0.0000 — **practically zero** (M0 가 clean 에 미세 over-action, magnitude ~0). Step 6 part 2 의 ΔFID=0 과 일관 (실질 무손상).

### 3-4. synthetic_diag_holdout (n=48, appendix)

| vs baseline | M0−base Δ | Wilcoxon p | Cohen d | M0>base |
|---|---|---|---|---|
| heuristic+gate | **+0.1014** | **8.1e-11** | **+0.993** | 45/3 |
| STOP-only / M1 | **+0.0775** | **1.0e-06** | +0.759 | 30/7 |
| random+gate | +0.0030 | 0.988 (ns) | +0.019 | 23/25 |
| dense_oracle | −0.0889 | 4.1e-06 | −0.793 | 8/39 |

## 4. 가설 평가 (evidence — status 전환 아님)

- **M0 > random+gate: 모든 real holdout 에서 유의** (g2_stress p=1.2e-6 d=0.47, g2_natural p=1.6e-17 d=0.87, clean p=2e-3). → "learned routing 이 무통제 baseline 보다 의미 있는 결정" — **H-2026-205 방향 supports evidence**.
- **M0 vs heuristic+gate (rule-based): 동작별 분기**:
  - g2_natural: M0 유의 우위 (p=2.3e-13, d=0.59) — heuristic 이 natural 을 over-correct.
  - synthetic: M0 massive 우위 (p=8e-11, d=0.99) — heuristic 위험 (part 1: 39.6% violation).
  - **g2_stress: 유의차 없음 (p=0.619)** — NetGain 상 heuristic 경쟁적. 단 Step 6 part 2 에서 heuristic 은 FID 악화 (+0.040), M0 는 FID 보존. **NetGain 동률이나 standard quality 는 M0 우위**.
- **dense_oracle > M0: 모든 holdout 유의** — learned policy 와 ceiling 사이 **headroom 명확** (g2_stress d=0.45, synthetic d=0.79). M0 는 안전·효율적이나 최적 아님.
- **M1/M2 vs M0: g2 에선 미세차 (M1 g2_stress 약간 우위)** 이나 M1 은 clean/synthetic 에서 catastrophic (Step 5). → broad-support (M0) 의 우월성은 OOD holdout 전반 일관.

## 4-bis. 재현성 — snapshot 1 (v2) vs snapshot 2 (v3, 독립 split)

12 핵심 비교 (4 holdout × {random / heuristic / dense_oracle}) 의 방향 + 유의성 일치:

| holdout / vs | snap1 Δ (p, d) | snap2 Δ (p, d) | 일치 |
|---|---|---|---|
| g2_stress / random | +0.022 (1.2e-6, 0.47) | +0.028 (1.2e-7, 0.50) | ✓ |
| g2_stress / heuristic | −0.006 (0.62, ns) | −0.000 (0.061, ns) | ✓ (both ns) |
| g2_stress / oracle | −0.017 (1.4e-14) | −0.017 (5.7e-14) | ✓ (oracle ↑) |
| g2_natural / random | +0.017 (1.6e-17, 0.87) | +0.019 (3.2e-15, 0.72) | ✓ |
| g2_natural / heuristic | +0.006 (2.3e-13, 0.59) | +0.007 (2.2e-12, 0.30) | ✓ |
| g2_natural / oracle | −0.002 (1.6e-12) | −0.002 (5.2e-13) | ✓ |
| clean / random | +0.000 (2e-3) | +0.000 (8e-5) | ✓ (~0) |
| clean / heuristic·oracle | ~0 (4.7e-10) | ~0 (4.7e-10) | ✓ |
| synthetic / random | +0.003 (0.99, ns) | +0.003 (0.57, ns) | ✓ (both ns) |
| synthetic / heuristic | +0.101 (8e-11, 0.99) | +0.064 (2e-5, 0.71) | ✓ |
| synthetic / oracle | −0.089 (4e-6) | −0.124 (8e-10) | ✓ (oracle ↑) |

→ **12/12 일치**. M0 > random (전 real holdout), M0 vs heuristic (natural/synthetic 유의, stress ns), oracle headroom (전 holdout) — **두 독립 split 에서 robust**. effect size 도 같은 크기대 (random d 0.47↔0.50 stress, 0.87↔0.72 natural).

synthetic 의 M0 절대 NetGain 은 v2 +0.078 → v3 +0.041 로 변동 (holdout state 다름) — 단 **상대 비교 (M0 > heuristic/STOP, M0 ≈ random) 는 불변**.

## 5. 다음 스텝

- **snapshot ≥ 2 충족 + 12/12 재현** → H-2026-205 (learnable routing > rule-based) 의 supports 방향 evidence 가 **2 독립 snapshot 에서 확인**. status 전환 (`active` → `supported`) 은 **사용자 승인 게이트** (AGENTS.md §3-11) — Agent 단독 금지. 사용자 검토 권고.
- **g2_stress 의 oracle headroom (d 0.45↔0.41)** → M0 의 stress 보수성 개선 (Stage 3 continuous argmax) 의 정량 동기 (재현됨).
- **claim 범위** (현재 방어 가능, 2 snapshot): "broad-support learned policy + real gate 가 real G2 holdout 에서 random 대비 유의하게 (재현), rule-based 대비 동작별로 우위 (natural/synthetic 유의, stress NetGain 동률+FID 우위), physical 0% violation, standard quality 보존/개선." generator-agnostic (G1) 은 미검증.
