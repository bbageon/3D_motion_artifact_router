# General Prompts Pilot v1 — NetGain Validity 검증용 (2026-05-25)

> 본 문서는 사용자 directive (2026-05-25 9-step plan) Step 2: "일반 동작 prompt set 10개." NetGain validity 검증의 G2 pilot + perceptual pilot 의 기반 prompt.
>
> 기존 G2 batch (`external_assets/g2_generated_v1/`) 의 prompt 는 `external_assets/MotionGPT/demos/t2m.txt` 의 50 detailed prompts. 일반 동작 (walk, run, wave, sit/stand, jump, kick) 의 category 균등 coverage 부재 — perceptual pilot 의 정성 평가가 어려움.

## 1. 10 Prompt (general_prompts_pilot_v1.txt)

| # | Prompt | Category |
|---|---|---|
| 1 | A person walks forward at a normal pace. | locomotion / walk |
| 2 | A person walks in a circle. | locomotion / walk + turn |
| 3 | A person slowly runs forward. | locomotion / run |
| 4 | A person stands still and waves the right hand. | upper-body gesture |
| 5 | A person raises both arms above the head. | upper-body gesture |
| 6 | A person sits down on a chair. | posture transition |
| 7 | A person stands up from sitting. | posture transition |
| 8 | A person turns left in place. | rotation |
| 9 | A person jumps once and lands. | dynamic / jump |
| 10 | A person kicks forward with the right leg. | dynamic / kick |

## 2. Category 분포

| Category | Count |
|---|---|
| locomotion (walk + run) | 3 |
| upper-body gesture | 2 |
| posture transition (sit/stand) | 2 |
| rotation | 1 |
| dynamic (jump + kick) | 2 |

**Total: 10 prompts × 5 categories** — 균등 coverage.

## 3. 본 prompt 의 의도

- **G2 pilot generation 의 기반**: 본 prompt 로 MotionGPT motion 새 batch 생성 (`g2_general_pilot_v1/`).
- **NetGain validity 검증**: 일반 동작 의 NetGain rank 가 시각 / perceptual quality rank 와 일치 하는지 확인.
- **Perceptual pilot 의 기반**: 20-30 pairwise A/B 비교 의 motion source.
- **논문 figure 의 기반**: B2-family / RL-1 / oracle 의 visual 비교 GIF.

## 4. 본 prompt 의 한계

- **n=10 small** — large-scale generator output 평가 아님 (pilot).
- **English only, simple syntax** — MotionGPT 의 inference quality 가 t2m.txt 의 detailed prompt 대비 다를 수 있음 (별도 측정).
- **Hand-curated 5 category** — 균등 coverage 시도 단 motion semantics 의 표본 도메인 일부 만.

## 5. 본 prompt set 의 사용 plan (사용자 Steps 3-7)

| Step | Action |
|---|---|
| 3 | MotionGPT 로 10 motion 생성 → `external_assets/g2_general_pilot_v1/`. |
| 4 | evaluator + 보조 metric 측정 (jerk, foot sliding, MPJPE-to-original 등). |
| 4 | 방법 별 비교: original G2 / B2-small/medium/large/val-best / B5 / RL-1 best / sequence oracle. |
| 5 | GIF/MP4 확장 — 각 category 1-2 sample, original vs B2-family vs RL-1 vs oracle. |
| 6 | Perceptual pilot — 20-30 pairwise A/B. NetGain rank vs human preference Spearman correlation. |
| 7 | NetGain validity 판단 → RL-2 진입 분기. |
