# Perceptual Pilot v2 — Stratified Design (Step 8-A, v2-revised 2026-05-26)

> 사용자 directive (2026-05-26) 박제: **v2 전체를 5-level perceptual test 로 만들지 말고, 두 갈래로 분리.**
> - G2 영역 = STOP / weak correction perceptual 검증.
> - Synthetic 영역 = 5-level active correction perceptual 검증.
>
> 본 문서는 v2 의 **stratified design** 의 단일 출처. 실제 manifest 생성 (Step 8-C) 의 의무 reference.
>
> **2026-05-26 사용자 피드백 반영** (revision):
> - Group C 를 **high-delta enriched subset** 으로 명시 (전체 synthetic 의 대표성 아님).
> - `corrupted vs 5-level oracle` 을 **positive control** 로 격하 (5 pairs).
> - Group B 의 n=3 pilot 한계 명시 + G2 natural 보강 candidate 박제.
> - Group D negative controls 는 **correlation 분석에서 제외** (rater sanity check only).
> - 총 55 → **50 pairs**, 2-session 평가 권장.

---

## 1. 본 v2 의 목적 (v1 대비 정정)

### 1-1. v1 의 한계 (회고)

v1 ([`../pilot_v1/`](../pilot_v1/)) 은 50 pairs of G2 general 5 samples × 5 pair types — **G2 단일 도메인 의 STOP/no-harm 검증** 만. 5-level action space 의 정량 evidence (2026-05-26 부록 JJ) 가 perceptual 에서도 reproduce 되는지 검증 못함. 또한 STOP-best 와 correction-best 가 한 pair set 에 섞여 해석이 흐려짐.

### 1-2. v2 의 stratified design

사용자 directive 의 핵심: **synthetic 에서는 5-level 이 명확히 우월 (Case A, p=3.28e-10, d=+0.667), G2 에서는 거의 차이 없음 (+0.0004)**. 따라서:
- **G2 GIF 에서 xsmall vs small 같은 fine-grained 차이를 사람에게 묻는 건 효율 낮음** — 정량 evidence 가 없는 영역에서 perceptual label 낭비.
- **Synthetic 에서 5-level perceptual 검증** — 정량 Case A 가 사람 인지에서 reproduce 되는지.
- STOP-best 와 correction-best 를 별도 group 으로 분리 — 해석 명확.

### 1-3. v2 의 4 stratified groups (수정 후)

| Group | Domain | Pair Types | Sample Count | Pairs |
|---|---|---|---|---|
| **A. STOP-best** | G2 general (oracle_len=0) | 2 | 7 (motion_003,005-010) | 14 |
| **B. Correction-best** | G2 general (oracle_len>0) | 2 | 3 (motion_001,002,004) — pilot, 보강 candidate 있음 | 6 |
| **C. 5-level synthetic (high-Δ enriched)** | Synthetic top-10 high-Δ subset | 3 (10+10+5 split) | 10 | 25 |
| **D. Negative controls (correlation 제외)** | Mixed | 1 (identity) | 5 | 5 |
| **Total** | | | | **50** |

---

## 2. Group A — G2 STOP-best (7 samples × 2 pair types = 14 pairs)

### 2-1. Sample 선정

G2 general pilot v1 의 oracle_len=0 (STOP-best) sample 7개. 본 sample 의 oracle 권장 action = STOP — original motion 의 보존 이 정답 후보.

| trial_id | prompt (g2_general) | oracle | B2-small NetGain | B2-large NetGain |
|---|---|---|---|---|
| motion_003 | "A person slowly runs forward" | STOP | -0.0026 | -0.0701 |
| motion_005 | "A person waves both hands" | STOP | -0.0021 | -0.0163 |
| motion_006 | "A person sits down on a chair" | STOP | -0.0043 | -0.0337 |
| motion_007 | "A person stands up from chair" | STOP | -0.0022 | -0.0207 |
| motion_008 | "A person turns left" | STOP | -0.0040 | -0.0344 |
| motion_009 | "A person jumps in place" | STOP | -0.0052 | -0.0680 |
| motion_010 | "A person kicks forward" | STOP | -0.0006 | -0.0889 |

### 2-2. Pair Types

| Pair Type | Expected Pattern | 함의 |
|---|---|---|
| **Original vs B2-large** | **높은 agreement (Original 우월)** — B2-large 의 over-modification visible (NetGain mean -0.045) | family-large 가 G2 에 부적합 의 perceptual 검증 |
| **Original vs oracle (STOP)** | **tie 가 dominant** (oracle=identity, gray=orange 동일) — **strong evidence 아님, sanity check** | NetGain=0 의 sanity (STOP 의 의미 자체) |

**해석 주의** (사용자 피드백): `Original vs oracle(STOP)` 은 사실상 동일 motion 비교 → strong evidence 가 아니라 **sanity check**. tie 가 많이 나오면 정상 ; A/B 선호가 강하게 나오면 GIF rendering 또는 rater bias 의심.

### 2-3. 본 group 의 결과 의 해석

- Original vs B2-large agreement ≥ 70% → STOP-best decision 의 perceptual supports.
- Original vs oracle tie ≥ 60% → STOP 의 sanity 정상 (oracle=identity 이면 사용자가 "차이 없음" 으로 답해야).
- 만약 oracle (STOP) 도 over-modification 으로 보이면 → identity-display GIF 의 sanity 의심 (GIF rendering 자체 점검).

---

## 3. Group B — G2 Correction-best (3 samples × 2 pair types = 6 pairs)

### 3-1. Sample 선정 — pilot 한계 명시

G2 general pilot v1 의 oracle_len>0 sample 3개. 본 sample 의 oracle 권장 action = non-STOP — small/weak correction.

| trial_id | prompt (g2_general) | oracle length | oracle NetGain |
|---|---|---|---|
| motion_001 | "A person walks forward at a normal pace" | 1 | +0.0091 |
| motion_002 | "A person walks in a circle" | 1 | +0.0059 |
| motion_004 | "A person waves one hand" | 2 | +0.0052 |

**한계 박제** (사용자 피드백): **n=3 은 too small**. "G2 에서 adaptive routing correction 이 perceptual 하게 낫다" 를 단정 못함. 본 group 은 **pilot evidence** only — real-distribution routing evidence 의 정식 보강 candidate 는 다음 §3-3.

### 3-2. Pair Types

| Pair Type | Expected Pattern | 함의 |
|---|---|---|
| **Original vs B2-val-best** | **B2-val-best 선호 또는 tie** (val-best = 부록 V 의 100% B2-small) | weak correction 의 visual 유효성 |
| **B2-val-best vs oracle** | **oracle 선호** (sample-tuned vs family-best) | adaptive routing 의 직접 evidence |

### 3-3. 보강 candidate — G2 natural n=50 의 non-STOP top 5

향후 Group B 보강 시 G2 natural (사용 가능한 50 sample) 의 oracle non-STOP top 5 sample 후보:

| trial_id | oracle length | oracle NetGain | first action |
|---|---|---|---|
| motion_006 (g2-natural) | 2 | **+0.1733** | FootLock both_feet small5 |
| motion_007 (g2-natural) | 2 | +0.1211 | VelocitySmoothing full_body medium5 |
| motion_008 (g2-natural) | 1 | +0.0979 | FootLock both_feet small5 |
| motion_028 (g2-natural) | 1 | +0.0925 | FootLock both_feet medium5 |
| motion_013 (g2-natural) | 1 | +0.0162 | FootLock both_feet small5 |

본 5 sample 의 GIF 신규 생성 + Group B 보강 시 **n=3 → 8**, 본 group 의 evidence 가 pilot → preliminary supports 격상 가능. **단 본 commit 의 v2 manifest 는 일단 n=3 의 pilot only**, 보강은 사용자 추가 directive 시 진행.

### 3-4. 본 group 의 결과 의 해석

- B2-val-best vs oracle 의 agreement (oracle 우월) ≥ 50% → adaptive routing 의 supports (n=3 의 pilot limit 인정).
- agreement < 50% 또는 tie dominant → "G2 에서 어차피 차이 작음" — small effect size 의 perceptual lower bound (positive supports 의 ceiling).

---

## 4. Group C — Synthetic 5-level (10 high-Δ enriched samples × 3 pair types = 25 pairs)

**v2 의 핵심 신규 group**. 부록 JJ (2026-05-26) 의 Case A 정량 evidence 가 perceptual 에서도 reproduce 되는지 검증.

### 4-1. **Subset 의 enriched 성격 명시** (사용자 피드백 반영)

본 10 sample 은 **`high-delta enriched synthetic subset`** — Δ(NetGain_5 - NetGain_3) 의 top 10 으로 의도적 선택. **`not representative random synthetic sample`**.

| 의도 | 표현 |
|---|---|
| ✅ 적절 | "5-level 의 perceptual effect 가 보이는지 확인 (effect 가 큰 영역에서)" |
| ❌ 부적절 | "전체 synthetic distribution 에서 5-level 이 perceptually 우월" |

본 enriched subset 의 결과는 **5-level 의 perceptual visibility 의 sufficient condition test** (큰 효과에서 보이는지) — necessary condition (모든 효과에서 보이는지) 는 별도 random subset 의 후속 검증 의무.

### 4-2. Sample 선정 — top 10 high-Δ(5-3)

`compare_3level_5level_synthetic.json` + raw oracle snapshot 의 per-sample 비교 — Δ(NetGain_5 - NetGain_3) top 10:

| Rank | trial_id | 5-level NetGain | 3-level NetGain | Δ(5-3) |
|---|---|---|---|---|
| 1 | 014552 | 0.3328 | 0.1835 | **+0.1493** |
| 2 | M007995 | 0.2544 | 0.1855 | +0.0689 |
| 3 | 012798 | 0.3037 | 0.2382 | +0.0655 |
| 4 | M000741 | 0.2596 | 0.1963 | +0.0633 |
| 5 | 012631 | 0.2521 | 0.1937 | +0.0585 |
| 6 | M008358 | 0.2988 | 0.2544 | +0.0445 |
| 7 | 001885 | 0.2968 | 0.2584 | +0.0385 |
| 8 | 010382 | 0.2260 | 0.1918 | +0.0342 |
| 9 | 012943 | 0.2601 | 0.2295 | +0.0305 |
| 10 | M007140 | 0.2543 | 0.2269 | +0.0274 |

### 4-3. Pair Types (사용자 피드백 반영 — 10/10/5 분할)

| Pair Type | Sample 수 | Expected Pattern | 함의 |
|---|---|---|---|
| **3-level oracle vs 5-level oracle** | **10 (전체)** | **5-level 선호 ≥ 60%** (Case A 의 perceptual 검증) | **핵심** — RL-2 = 5-level 의 정식 supports |
| **B2-val-best vs 5-level oracle** | **10 (전체)** | **5-level 선호** (fixed smoothing < adaptive routing) | family-baseline 대비 active routing 의 직접 evidence |
| **corrupted (no correction) vs 5-level oracle** | **5 (top 5)** | **5-level 강하게 선호 ≥ 80%** | **positive control** — correction 자체 의 perceptual sanity (너무 쉬운 비교 — 정식 evidence 아님) |

**`corrupted vs 5-level oracle` 의 positive control 역할** (사용자 피드백): 본 pair 는 correction 이 적용된 motion vs corrupted motion 이라 너무 쉬운 비교 — agreement 가 높아도 "5-level 이 우월" 의 evidence 가 아니라 **"correction 자체가 눈에 보임"** 의 sanity. n=5 로 cognitive load 절감.

### 4-4. 본 group 의 결과 의 해석

- **3-level vs 5-level agreement (5-level 우월) ≥ 60%** → **RL-2 = 5-level 의 perceptual evidence (quality-validated, single-rater pilot)**.
- agreement < 50% → "정량 Δ +0.016 이 perceptual 임계 미만" — 5-level oracle 의 정량 우월이 high-Δ enriched 에서도 visible 하지 않음 → RL-2 진입 시 visual claim 보수적 표기.
- B2-val-best vs 5-level agreement (5-level 우월) ≥ 60% → fixed smoothing 대비 우월 (정량 + perceptual 일관).
- corrupted vs 5-level agreement ≥ 80% → correction 자체 의 sanity. < 80% 이면 evaluator/oracle 자체 의심 (positive control fail).

### 4-5. GIF 생성 의무 (Step 8-B)

- 10 sample × 2 comparison + 5 sample × 1 comparison = **25 GIF 생성 필요**.
- HumanML3D 의 raw motion → synthetic corruption injection (현재 oracle 의 corruption protocol 동일) → 3-level oracle / 5-level oracle / B2-val-best 적용 → Y-up overlay GIF (gray=base, orange=corrected).
- corruption protocol: oracle_sequence_multi_5level_v1 의 metadata 의 (foot_floating + global_jitter 등) 동일.

본 GIF generation tool = `tools/synthetic_5level_visualize_gif.py` (Step 8-B 신규).

---

## 5. Group D — Negative controls (5 pairs, correlation 제외)

### 5-1. 본 group 의 목적

**사용자 (rater) 의 응답 sanity check**. 같은 motion 의 identical 2 instance 를 보여줘 — tie 가 dominant 이어야 정상. 만약 사용자가 임의로 A/B 를 선호하면 → 의도적 bias 가능 (cognitive load 또는 fatigue 의 signal).

### 5-2. 분석 의 분리 (사용자 피드백)

**Group D 는 다음 분석에서 제외**:
- ❌ preference correlation 계산 (Spearman, Kendall)
- ❌ agreement rate 계산
- ❌ NetGain validity 검증

**Group D 가 사용되는 분석**:
- ✅ rater sanity check (tie 율)
- ✅ confidence calibration (tie 인데 confidence 5 면 over-confident)

### 5-3. Sample + Pair Type

5 pair 모두 **identical GIF**:

| Pair | A | B | Expected | 함의 |
|---|---|---|---|---|
| NC1 | motion_001 5-level oracle | motion_001 5-level oracle | **tie** | identical content sanity |
| NC2 | 014552 5-level oracle | 014552 5-level oracle | **tie** | synthetic identical sanity |
| NC3 | motion_003 original | motion_003 original | **tie** | G2 STOP identical sanity |
| NC4 | M007995 5-level oracle | M007995 5-level oracle | **tie** | synthetic sanity |
| NC5 | motion_010 B2-small | motion_010 B2-small | **tie** | G2 baseline sanity |

(blind label 은 A=B identical 이지만 사용자에게는 random shuffle 후 매핑.)

### 5-4. 본 group 의 결과 의 해석

- Negative control tie ≥ 4/5 → 사용자 응답 의 일관성 OK.
- tie < 4/5 → 사용자 응답 의 bias 신호 — 본 응답 데이터 의 reliability 의심, Group A/B/C 분석 시 caveat.

---

## 6. Blind Labeling + Randomization

### 6-1. Blind 의 보장

- Manifest 의 `blind_label_A`, `blind_label_B` field 는 method 매핑. 사용자 응답 의 CSV 에는 본 column 노출 되지 않음 (analyzer 가 매핑 수행).
- Pair 의 A/B 순서는 random seed 42 (numpy RNG) 로 50% 확률 swap.

### 6-2. Pair 순서 의 randomization

전체 50 pair 의 순서는 numpy RNG seed 42 의 shuffle. group A/B/C/D 가 섞임 — 사용자가 group pattern 을 학습 못 함.

### 6-3. 2-Session 평가 권장 (사용자 피드백)

50 pair 의 cognitive load ~ 35-50 분. **2-session 평가 권장**:

| Session | 포함 group | 예상 pair 수 | 시간 |
|---|---|---|---|
| **Session 1** | Group A (STOP-best, 14) + Group B (correction-best, 6) + Group D 의 G2-related NC (NC1, NC3, NC5 = 3) | **23 pairs** | ~ 15-20 분 |
| **Session 2** | Group C (synthetic 5-level, 25) + Group D 의 synthetic NC (NC2, NC4 = 2) | **27 pairs** | ~ 20-30 분 |
| **Total** | | **50 pairs** | ~ 35-50 분 |

각 session 사이 휴식 권장. Session 1/2 의 순서는 사용자 자유.

---

## 7. Success Criteria (v2 종합)

본 v2 의 결과 의 의사결정 분기 ([H-2026-203 secondary](../../hypotheses/H-2026-203.md), [H-2026-204 RQ1+RQ2](../../hypotheses/H-2026-204.md), [H-2026-205 RQ3](../../hypotheses/H-2026-205.md) 의 perceptual evidence sub-tier).

| Group | Success Threshold | 의미 |
|---|---|---|
| **A. STOP-best** | Original vs B2-large agreement ≥ 70% | G2 over-modification 의 perceptual supports |
| **A. STOP-best** | Original vs oracle tie ≥ 60% | STOP sanity (strong evidence 아님) |
| **B. Correction-best (pilot n=3)** | B2-val-best vs oracle agreement (oracle 우월) ≥ 50% | adaptive routing 의 lower-bound supports (pilot limit) |
| **C. 5-level synthetic (high-Δ enriched)** | **3-level vs 5-level agreement (5-level 우월) ≥ 60%** | **RL-2 = 5-level 의 perceptual supports (enriched, single-rater)** |
| **C. 5-level synthetic** | B2-val-best vs 5-level agreement (5-level 우월) ≥ 60% | fixed smoothing 대비 우월 (정량 + perceptual 일관) |
| **C. positive control** | corrupted vs 5-level agreement ≥ 80% | correction 자체 sanity (evidence 아닌 sanity) |
| **D. Negative controls** | tie ≥ 4/5 | rater sanity (correlation 분석 제외) |

### 7-1. 종합 분기

- **All Pass** (단 Group D 제외 의 분석 trustworthy): RL-2 (Step 9) 진입 의 추가 perceptual supports. Case A 의 quality-validated evidence 격상 (단 enriched subset + single-rater pilot caveat 유지).
- **Group C 만 Fail (5-level perceptual 미달)**: RL-2 진입 시 oracle ceiling 과 learned policy 평가 시 5-level 의 visual claim 보수적 표기.
- **Group A/B Fail**: G2 의 NetGain validity 의심 → NetGain reward 수정 우선.
- **Group D Fail**: rater 의 응답 데이터 자체 의심 — n=1 의 internal sanity 의 한계 인정, 정식 evidence 보류 (3명 이상 의 inter-rater agreement 의 후속 필요).

---

## 8. 본 v2 의 한계 (정직 박제 — v1 의 caveat 상속 + revision 추가)

1. **n=50 pairs, 1명 evaluator** — **internal sanity check (b1 sub-tier, AGENTS.md §3-17)**, 논문급 evidence 아님.
2. **정식 perceptual evidence (b3 sub-tier)** — 10-20명 의 별도 user study 의무.
3. **Synthetic GIF 의 corruption realism** — synthetic artifact (foot floating + global jitter) 가 G2 의 natural artifact 와 다를 수 있음 (AGENTS.md §3-17 caveat 일관).
4. **Group C 의 enriched subset 한계** (사용자 피드백) — top-10 high-Δ 의 의도적 선택 → "5-level perceptual visibility 의 sufficient condition test" only. 전체 synthetic distribution 의 representative claim 금지.
5. **Group B 의 small-n pilot 한계** — n=3 G2 general → "adaptive routing perceptual evidence 의 lower bound" only. G2 natural top-5 보강 시 n=8 로 격상 가능 (TBD, 별도 directive).
6. **Cognitive load** — 50 pair ~ 35-50 분. **2-session 평가 권장** (§6-3).
7. **GIF view angle** — 일부 motion semantic (turn 등) 의 측면 가려질 수 있음. multi-view 는 후속.
8. **Negative control 의 correlation 제외** (사용자 피드백) — preference correlation 계산에서 Group D 의 5 pair 는 의도적 제외, rater sanity check 만 사용.

---

## 9. Cross-link

### 9-1. Source data
- G2 general group (A/B): [`evals/snapshots/g2_general_pilot_v1.json`](../../snapshots/g2_general_pilot_v1.json).
- G2 natural 보강 candidate (B 향후): [`evals/snapshots/oracle_sequence_g2_natural_5level_v1.json`](../../snapshots/oracle_sequence_g2_natural_5level_v1.json).
- Synthetic group (3-level): [`evals/snapshots/oracle_sequence_multi_v2_n60.json`](../../snapshots/oracle_sequence_multi_v2_n60.json).
- Synthetic group (5-level): [`evals/snapshots/oracle_sequence_multi_5level_v1.json`](../../snapshots/oracle_sequence_multi_5level_v1.json).
- Comparison: [`evals/snapshots/compare_3level_5level_synthetic.json`](../../snapshots/compare_3level_5level_synthetic.json).

### 9-2. Existing GIF (Group A/B 의 G2 portion, 부분 cover)
- [`reports/figures/2026-05-25/g2_general_gif_yup_fix/`](../../../reports/figures/2026-05-25/g2_general_gif_yup_fix/) — motion_001, 004, 007, 008, 010 의 vs_b2_small/large/oracle GIF (15개).
- **누락 (Group A/B 보강)**: motion_002, 003, 005, 006, 009 의 vs_b2_large + vs_oracle (10 GIF 신규 생성 필요).

### 9-3. GIF 생성 의무 (Group C + Group D NC + Group A/B 누락)
- 신규 디렉토리: `reports/figures/2026-05-26/perceptual_v2_gif_yup_fix/`.
- Group C tool: `tools/synthetic_5level_visualize_gif.py` (Step 8-B 신규).
- Group A/B 보강 tool: 기존 `tools/g2_general_pilot_visualize_gif.py` 재활용.
- Group D NC tool: identical GIF 복사 (no separate generation).
- Y-up convention 의무 (AGENTS.md §3-19).

### 9-4. Framework
- 기존 v1: [`tools/perceptual_pilot_framework.py`](../../../tools/perceptual_pilot_framework.py).
- v2 의 stratified manifest 생성: 본 framework 의 `_build_manifest` 의 group-aware 분기 확장 (Step 8-C).
- v2 의 analyzer: Group D 의 correlation 제외 의무 (§5-2).

### 9-5. 박제 commit / journal
- v1: `aa636a8` (2026-05-25), `09bec06` (Step 6 framework).
- v2 design (본 commit): TBD, 2026-05-26 일지 후속 절.
- 부록 JJ (2026-05-26): 5-level oracle ablation evidence.

### 9-6. AGENTS.md 인용
- §3-17 (synthetic vs real evidence separation) — Group C 는 synthetic, Group A/B 는 G2 real-distribution.
- §3-18 (B2-family) — B2-val-best vs oracle pair.
- §3-19 (Y-up axis) — GIF 의무.
- §3-20 (Metric Citation Gate) — NetGain Category C, 본 perceptual = NetGain validity 검증.
- §3-21 (Action Space Provenance) — Group C 의 3-level vs 5-level 직접 비교.

---

## 10. Revision Log

| Date | Revision | Trigger |
|---|---|---|
| 2026-05-26 (v1) | initial draft, 55 pairs, 4 groups | Step 8-A user directive |
| 2026-05-26 (v2 — 본 문서) | encoding fix (BOM), Group C enriched 명시 + 25 pair 축소, Group B n=3 pilot 한계 + G2 natural 보강 candidate, Group D correlation 제외, 2-session 권장 | 사용자 6-point feedback |
