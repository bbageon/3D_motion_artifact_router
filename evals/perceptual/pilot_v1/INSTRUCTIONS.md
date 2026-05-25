# Perceptual Pilot v1 — 사용자 작업 안내 (2026-05-25)

> 본 문서는 사용자 9-step plan Step 5 (perceptual pilot response 수집) 의 정식 사용자 작업 안내. AGENTS.md §3-17 quality-validated evidence (b) perceptual rating sub-tier 의 첫 framework.

---

## 1. 본 pilot 의 목적

**NetGain validity 검증** — 본 프로젝트 의 NetGain proxy reward 가 사람 의 visual/perceptual quality 판단과 어느 정도 일치 하는지 정량 측정.

- **Branch A (NetGain validity 통과)**: agreement rate ≥ 60% → **RL-2 진입**.
- **Branch B (NetGain validity 부족)**: agreement rate < 60% → **NetGain reward 수정 우선** (FootFloating weight / jerk / contact estimator 등).

본 pilot 은 **1명 internal sanity check**. 정식 perceptual evidence (논문급) 는 3명 minimum, 10-20명 final 의 별도 user study.

---

## 2. 사용자 작업 절차

### Step 1: GIF browse

[`reports/figures/2026-05-25/g2_general_gif_yup_fix/`](../../../reports/figures/2026-05-25/g2_general_gif_yup_fix/) 의 **15 GIF** 를 확인. 각 GIF 는 두 motion 의 overlay (gray = method_A, orange = method_B).

15 GIF 목록 (5 sample × 3 comparison):
- motion_001 (walk normal): vs B2-small, vs B2-large, vs oracle.
- motion_004 (wave hand): vs B2-small, vs B2-large, vs oracle.
- motion_007 (stand up): vs B2-small, vs B2-large, vs oracle.
- motion_008 (turn left): vs B2-small, vs B2-large, vs oracle.
- motion_010 (kick forward): vs B2-small, vs B2-large, vs oracle.

### Step 2: Manifest + Template open

- **Manifest** ([`pairs_manifest_v1.json`](pairs_manifest_v1.json)) — 50 pairs 의 정의 (blind label A/B 의 method 매핑).
- **Response template** ([`response_template_v1.csv`](response_template_v1.csv)) — 50 pair 의 빈 row.

### Step 3: 50 pair 각각 평가

각 pair 의 CSV row 에 다음 채워 넣음:

| Field | 의무 / 옵션 | 값 |
|---|---|---|
| `user_choice` | **의무** | `A`, `B`, 또는 `tie` (한 자) |
| `confidence` | 옵션 | 1-5 (1=very unsure, 5=very sure) |
| `comments` | 옵션 | 자유 텍스트 (예: "B2-large 가 leg 의 dynamic 손실") |

**평가 기준** (사용자 directive): "어느 motion 이 더 자연스러운가?"
- A 가 더 자연스럽고 motion semantic 명확 → `A`.
- B 가 더 자연스러우면 → `B`.
- 차이 거의 없으면 → `tie`.

**Method label 완전 hidden**: 어느 GIF 가 어느 method 인지 모름 (manifest 의 `blind_label_A`, `blind_label_B` 는 사용자 작업 시 보지 않음 — analysis 후 자동 매핑).

### Step 4: response_v1.csv 저장

작성 완료 후 `evals/perceptual/pilot_v1/response_v1.csv` 로 저장.

### Step 5: Analysis 실행

```
python -m tools.perceptual_pilot_framework analyze \
    --manifest evals/perceptual/pilot_v1/pairs_manifest_v1.json \
    --response evals/perceptual/pilot_v1/response_v1.csv \
    --output evals/snapshots/perceptual_pilot_v1.json
```

결과: `evals/snapshots/perceptual_pilot_v1.json` 의 agreement rate, Spearman r, Kendall tau, pair_type 별 stats, disagreement list.

---

## 3. 5 Pair Type 의 expected pattern (사용자 directive 박제)

사용자 directive (2026-05-25) 의 expectation — analysis 의 reference:

| Pair Type | Expected Pattern | 함의 |
|---|---|---|
| **Original vs B2-large** | **높은 agreement** (Original 우월) — B2-large over-modification 명확 visible | B2-large 의 fine dynamics 손상 의 user-confirmation |
| **Original vs B2-small** | **tie 가 많아도 정상** — B2-small 의 minimal smoothing 이 subtle | NetGain -0.003 의 visual 의미 의 정량 |
| **B2-small vs B2-large** | **B2-small 선호** | family 내 best variant 의 정성 supports (부록 V 의 100% B2-small best) |
| **Original vs oracle** | **STOP-best 가 많으면 original 선호 또는 tie 많음** | 70% STOP-best (부록 CC) 의 user-confirmation |
| **B2-val-best vs oracle** | **가장 중요** — oracle 이 정말 더 낫다고 보이는지 | abstain-aware routing 의 직접 supports — H-2026-203 의 정식 evidence |

---

## 4. Analysis 결과 의 해석 가이드

analyze 결과 의 다음 항목 확인:

| 지표 | 해석 |
|---|---|
| `agreement.agreement_rate_excl_tie` | overall agreement (NetGain winner = human winner) — **success threshold = 0.60** |
| `spearman_netgain_diff_vs_human.r` | NetGain difference vs human winner 의 monotonic correlation — **expect positive r > 0** |
| `kendall_tau.tau` | non-parametric rank correlation |
| `per_pair_type` | 각 5 pair type 의 agree/disagree/tie count |
| `disagreement_list` | NetGain winner ≠ human winner sample — **manual inspection 의 trigger** |

---

## 5. 결과 의 분기점 (사용자 plan Step 7-8)

### Branch A: NetGain validity 통과
- Condition: agreement rate ≥ 60%, Spearman r > 0.
- → **Step 8: RL-2 (Q-learning / value iteration) 진입**.
- RL-2 의 reward = NetGain proxy. 결과 = visual/perceptual 로 검증.

### Branch B: NetGain validity 부족
- Condition: agreement rate < 60% 또는 Spearman r ≈ 0.
- → **Step 8: NetGain reward 수정 우선**.
- 수정 후보:
  - FootFloating weight 낮추기 (부록 Z 의 evaluator limitation 결합).
  - jerk / foot sliding 추가.
  - fidelity penalty 재조정.
  - contact estimator 개선 (Item 6).
  - correction magnitude penalty 복원.
- → RL-2 진입 보류.

---

## 6. 본 pilot 의 한계 (정직 박제)

1. **n=50 pairs small** — pilot scale.
2. **1명 evaluator (사용자 본인)** — inter-rater agreement 없음. 본 결과는 **internal sanity check** only.
3. **정식 perceptual evidence (논문급)**: 3명 minimum, 10-20명 final 의 별도 user study 필요.
4. **Single GIF view angle**: 일부 motion semantic (예: turn 의 측면 변화) 가 view 에서 부분 가려질 수 있음.
5. **Cognitive load**: 50 pair 의 시각 비교 ~ 30-60 분. **10-15 pair 단위 분할 권장**.

---

## 7. Cross-link

- Framework: [`tools/perceptual_pilot_framework.py`](../../../tools/perceptual_pilot_framework.py).
- Manifest: [`pairs_manifest_v1.json`](pairs_manifest_v1.json).
- Response template: [`response_template_v1.csv`](response_template_v1.csv).
- GIF source: [`reports/figures/2026-05-25/g2_general_gif_yup_fix/`](../../../reports/figures/2026-05-25/g2_general_gif_yup_fix/).
- Framework 박제: [`evals/reports/2026-05-19_h_2026_205_stage0.md`](../../reports/2026-05-19_h_2026_205_stage0.md) 부록 FF.
- Pair type expected pattern + evaluator scale: 부록 GG (본 commit 추가).
