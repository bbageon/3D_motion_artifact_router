# AR-078 - Routing Gate 비교 (held-out benefit prediction)

Status: in-progress (2026-07-13 착수 — 사용자 승인 "응 진행해줘")  
Epic: RL-Q  
Priority: 🔴  
Parent: AR-077 (skate 단일 gate 불충분: AUC 0.67, harmful-apply 36.7%) / AR-065 (pre-action state)

## Goal

개별 모션 단위로 "root correction 을 적용할지(APPLY) 말지(STOP)" 를 **보정 전 관측만으로**
결정할 수 있는가. gate 후보들을 같은 held-out 에서 비교해, richer state 가 단일 신호의
한계(harmful-apply 37%)를 실제로 낮추는지 확인한다. **본 결과로 AR-077 마감.**

## 사전등록 (2026-07-13 — 결과 보기 전 고정)

### 데이터 (전부 기존 박제물 — 새 계산 없음)

- state: `preaction_state_ar065_v1.csv` (2,700 — 전부 pre-action, leakage audit 완료)
- label: `routing_benefit_permotion_ar077_v1.csv` (2,699 — ΔMM, split) — (gen,sid,seed) join
- split: AR-077 과 동일 (sample_id 단독, seed 20260722 산출분의 split 열 그대로)
- δ: calibration VQ |ΔMM| median (AR-077 재현 — holdout 미사용). benefit = ΔMM<−δ, harm = ΔMM>+δ

### Gate 후보 (5)

| gate | score | 성격 |
|---|---|---|
| G1 generator-only | 1(MDM)/0(VQ) | 4차 피드백의 "유효한 baseline" |
| G2 skate-only | foot_skate | AR-077 재현 (AUC ~0.67) |
| G3 mismatch-only | root_gait_mismatch | counterfactual 단일 — 기전 feature 의 단독 성능 |
| G4 richer-state (no-gen) | LR(6 geometry feature + intent) | **순수 state-conditioned** |
| G5 richer-state (+gen) | LR(G4 + generator one-hot) | 전체 정보 |

LR(logistic regression) 은 **calibration rows 로만** 학습 (표준화 통계 포함). holdout 은 평가 전용.

### 지표 (holdout)

1. benefit-AUC + prompt-bootstrap CI (B=1000, multiplicity 보존) — **paired diff CI** (G4/G5 − G2).
2. **harmful-apply@적용률** (핵심 — selective prediction, Gangrade AISTATS 2021): 적용률 {10,20,30,33,40,50}% 에서 score 상위 k 를 APPLY 로 했을 때 (a) harmful-apply (실제 악화 비율), (b) non-beneficial, (c) benefit capture. 동률은 고정 seed jitter 로 순서 결정 (사전 고정 20260725).
3. 참조선: holdout 전체의 base benefit/harm rate.

### 판정 기준 (사전 고정)

- **개선 지지**: G4 또는 G5 가 G2 대비 (a) paired AUC diff CI 하한 > 0 **그리고** (b) 적용률 30% 에서 harmful-apply 를 낮춤 (bootstrap diff CI 로 확인).
- **한계 확정**: 개선 없음 → "현 state 로 per-motion 결정 불가 — generator-level rule 이 현재 한계" (유효한 결과).
- G3 단독이 G2 를 넘으면 informational 로 기록 (기전 feature 의 가치).
- 모든 수치 = **δ 조건부 exploratory** (AR-077 caveat 승계). "routing 작동" 표현은 개선 지지 + AR-073(다중 평가자) 전까지 금지.

## Claim Boundary

허용: "같은 held-out 에서 gate 별 benefit-AUC·harmful-apply@rate 비교 — richer state 의 개선 여부".
금지: MM-Dist proxy 를 지각 benefit 으로 / LR 을 학습된 정책 성능 일반 주장으로 / 단일 benchmark 밖 일반화 / δ 무관 확정 표현.

## Grounding (§3-22)

Gangrade et al. AISTATS 2021 (selective prediction) · Guo HumanML3D CVPR2022 (MM-Dist) · Safe Orchestration no-harm (position §0).
