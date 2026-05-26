# Perceptual Pilot v2 — 사용자 작업 안내 (Step 8-C)

> 본 문서는 [`DESIGN.md`](DESIGN.md) 의 stratified design 의 사용자 작업 절차. AGENTS.md §3-17 quality-validated evidence (b1 internal sanity sub-tier).

---

## 1. 본 pilot 의 목적

**5-level RL-2 action space 의 perceptual 검증** (synthetic Case A) + **G2 STOP/weak correction 의 perceptual 검증**.

- **Group A (G2 STOP-best, n=7×2=14)**: G2 의 over-modification 검증.
- **Group B (G2 correction-best, n=3×2=6, pilot)**: G2 의 adaptive routing 의 lower-bound evidence.
- **Group C (Synthetic 5-level high-Δ enriched, n=25)**: RL-2 = 5-level 의 perceptual supports.
- **Group D (Negative controls, n=5)**: rater sanity check (correlation 분석 제외).

본 pilot 은 **1명 internal sanity check (b1 sub-tier)**. 정식 evidence (b3) 는 10-20명 user study 의 후속.

---

## 2. 사용자 작업 절차

### Step 1: 2-session 분할 평가 (권장)

총 50 pairs ~ 35-50 분. **2-session 분할 권장**:

| Session | pair 수 | 시간 | 포함 |
|---|---|---|---|
| **Session 1** | 23 | ~ 15-20 분 | Group A/B (G2) + NC1/3/5 (G2 NC) |
| **Session 2** | 27 | ~ 20-30 분 | Group C (synthetic) + NC2/4 (synthetic NC) |

각 session 사이 휴식 권장. session 1/2 순서는 자유.

### Step 2: GIF browse

[`reports/figures/2026-05-25/g2_general_gif_yup_fix/`](../../../reports/figures/2026-05-25/g2_general_gif_yup_fix/) (motion_001,004,007,008,010)
[`reports/figures/2026-05-26/perceptual_v2_gif_yup_fix/`](../../../reports/figures/2026-05-26/perceptual_v2_gif_yup_fix/) (motion_002,003,005,006,009 + synthetic)

### Step 3: pair_id 별 평가

각 row (`response_template_v2.csv`) 의 다음 column 채움:

| Field | 의무 | 값 |
|---|---|---|
| `user_choice` | 의무 | `A`, `B`, 또는 `tie` |
| `confidence` | 옵션 | 1-5 (1=very unsure, 5=very sure) |
| `comments` | 옵션 | 자유 텍스트 |

**평가 기준**: "어느 motion 이 더 자연스러운가?". GIF 의 gray vs orange 는 method_A vs method_B (manifest 에 매핑 hidden).

### Step 4: response_v2.csv 저장

작성 완료 후 [`response_v2.csv`](response_v2.csv).

### Step 5: Analysis 실행

```
python -m tools.perceptual_pilot_v2_framework analyze \
    --manifest evals/perceptual/pilot_v2/pairs_manifest_v2.json \
    --response evals/perceptual/pilot_v2/response_v2.csv \
    --output evals/snapshots/perceptual_pilot_v2.json
```

---

## 3. Group counts (manifest 의 분포)

| Group | Count |
|---|---|
| A_STOP_best | 14 |
| B_correction_best | 6 |
| C_synthetic_5level | 20 |
| C_positive_control | 5 |
| D_negative_control | 5 |
| **Total** | **50** |

---

## 4. 본 pilot 의 한계 (DESIGN.md §8 박제)

1. n=50, 1명 evaluator — **b1 sub-tier internal sanity** only.
2. Group C 는 high-Δ enriched subset (전체 representative 아님).
3. Group B 는 n=3 pilot (G2 natural n=50 보강 candidate 별도 등록).
4. Group D 는 correlation 분석에서 제외 (rater sanity only).

---

## 5. Cross-link

- Design: [`DESIGN.md`](DESIGN.md).
- v1 (참고): [`../pilot_v1/`](../pilot_v1/).
- Manifest: [`pairs_manifest_v2.json`](pairs_manifest_v2.json).
- Response template: [`response_template_v2.csv`](response_template_v2.csv).
- Framework: [`tools/perceptual_pilot_v2_framework.py`](../../../tools/perceptual_pilot_v2_framework.py).
