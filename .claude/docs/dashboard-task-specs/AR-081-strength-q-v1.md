# AR-081 - Strength-Q v1: 한 번의 root 보정 강도 Q(s,u) (사전등록)

Status: in-progress (2026-07-13 착수 — 사용자 directive "정책 최적화 진행")  
Epic: RL-Q  
Priority: 🔴  
Parent: AR-078/079 (gate 한계) / AR-065 (state) / §5-3 action-space 등록

## Goal

"이 모션에 root 보정을 **얼마나 강하게(u)** 걸지 (u=0 = STOP 포함)"를 13차원 의미 상태로 학습.
**scope = one-shot** (한 번의 보정 — contextual Q; 이력 불요). closed-loop MDP 확장은 별도 task
(그때 history·budget·KDG 상태 추가 — 본 spec §5).

## 사전등록 (2026-07-13 — 결과 보기 전 고정. 외부 설계 피드백 채택: 13차원·additive·중복 제거·Markov 검정)

### 1. 상태 13차원 — 직관적 이름 (naming 단일 출처)

| # | 새 이름 (직관) | 구 이름 (AR-065 CSV) | 뜻 (한 줄) | 그룹 |
|---|---|---|---|---|
| 1 | `stepping_speed` | gait_implied_speed | 다리 움직임이 요구하는 몸통 속도 (벨트 속도) | A 필요도 |
| 2 | `body_speed_shortfall` | root_gait_mismatch | 몸통이 걸음보다 모자란 속도 (음수 = 과속) | A 필요도 |
| 3 | `foot_sliding` | foot_skate | 접지발 미끄러짐 (shortfall 과 독립인 잔여 증상) | A 필요도 |
| 4 | `ground_contact_ratio` | contact_fraction | 접지 프레임 비율 (관측 충분성) | B 신뢰도 |
| 5 | `contact_streak` | contact_run_mean / n_frames | 접지 지속 길이 (정규화 — 안정성) | B 신뢰도 |
| 6 | `path_straightness` | straightness | 이동 경로 직진도 | C 문맥 |
| 7 | `duration_sec` | n_frames / 20 | 모션 길이 (초) | C 문맥 |
| 8-10 | `wants_to_travel` / `wants_to_stay` / `intent_unclear` | locomotion_intent one-hot | prompt 의도 (이동/제자리/애매) | C 문맥 |
| 11-13 | `is_mdm` / `is_motiongpt` / `is_momask` | gen one-hot | generator prior (반응 가능한 외생변수) | D prior |

**제외 (중복·금지)**: `root_speed_actual` (= stepping_speed − shortfall 로 유도) · `mismatch_ratio` (중복 + 0 근처 폭주) · `path_length` (speed·duration 과 중복) · GT ratio (배포 시 불가) · 보정 후 값 (leakage) · sample id/seed/config hash (식별자). **E 불확실성 2차원** (support_distance·ensemble_disagreement) 은 v1 미구현 — 구현 시 "높으면 STOP" 용도로만 추가.

### 2. Action·라벨·데이터

- action: `u ∈ {0, 0.25, 0.5, 0.75, 1.0}` (u=0=STOP=무보정). §5-3 provenance 등록 완료.
- 라벨 (motion×u 단위): `ΔMM(u) = MM-Dist(corrected@u) − MM-Dist(original)` — **improvement utility = −ΔMM** (연속), **harm = ΔMM > +δ** (δ = calibration VQ |ΔMM@u=1| median — AR-077 protocol 승계, calibration 에서만).
- 데이터: 전체 pool 2,699 motions × 5 u (u=0·1.0 라벨은 기존 permotion CSV 재사용, u=0.25/0.5/0.75 신규 산출). split = AR-077/078 과 동일 sample_id 단독 split (seed 20260722 승계 — permotion CSV 의 split 열).
- §3-26: `transition_dataset_id=strength_q_ar081_v1` · u_grid · seed · mining_reason=none (natural pool).

### 3. 모델 — 그룹별 additive 구조 (해석 보존)

- `Q_improve(s,u) = f_A(A그룹,u) + f_B(B그룹,u) + f_C(C그룹,u) + f_D(D그룹,u)` — 그룹을 임의 단일 점수로 뭉개지 않음 (Neural Additive Models, Agarwal et al. NeurIPS 2021).
- `P_harm(s,u)` 별도 학습 (동일 additive 구조).
- **shortcut 검사**: `Q-state` (D 제외 10차원) vs `Q-state+gen` (13차원) 상시 병렬 평가 — generator ID 암기 여부.
- v1 구현: 그룹별 얕은 MLP 합 (또는 그룹별 GBM 합) — calibration 만 학습, holdout 평가 전용.

### 4. 판정 기준 (사전 고정, holdout)

- 정책: `u*(s) = argmax_u [Q_improve − λ_harm·P_harm]` (λ_harm 은 calibration 에서 harmful-rate ≤ 목표로 선택, 고정 후 holdout).
- **지지** = holdout 에서 u*(s) 정책이 (a) 고정 u=1.0 대비 실현 improvement(−ΔMM) 평균 유의 우위 (prompt-bootstrap CI) **그리고** (b) harmful rate 증가 없음 (diff CI 상한 ≤ 0 근방).
- **한계** = 미충족 (그 자체 유효 — "강도 축에도 현 state 불충분"의 특정).
- 전 수치 δ 조건부 exploratory (MM proxy — 지각 아님). "정책 작동" 표현은 지지 + 지각 검증 후.

### 5. Markov 축약 검정 (분석 단계 — v1 에 포함)

1. **행동 보존**: 제외 변수(예: path_length) 재투입 시 argmax u 가 유의하게 뒤집히나.
2. **reward 보존**: (s,u)→(improvement, harm) 예측이 제외 변수 재투입으로 유의 개선되나 — 되면 재포함.
3. **transition 보존**: one-shot 이라 v1 은 해당 없음 (closed-loop 확장 시 s_{t+1} 예측 잔차 검정 — Allen et al., NeurIPS 2021 Markov state abstraction).

closed-loop 확장 시 추가할 이력 상태 (사전 목록 고정): iteration_remaining · last_tool/last_u · same_pair_count · last_improvement · cumulative_induced_disp(과거 action 의 **관측된** 결과 — 입력 가능) · cumulative_fidelity_loss · current/best score · valid_action_mask(KDG). 미실행 후보 action 의 결과는 입력 금지; counterfactual 추정치는 provenance 명시 시 가능.

## Claim Boundary

허용: "one-shot root 보정의 강도 결정 Q(s,u) — holdout 에서 고정 u 대비 개선 여부". 금지: closed-loop 성능 / 지각 개선 (MM proxy) / RL-2 구계열(3/5-level·NetGain) 수치와 혼합 인용 (§6-15) / δ 무관 확정.

## Grounding (§3-22)

Agarwal et al. NeurIPS 2021 (Neural Additive Models — additive 해석) · Allen et al. NeurIPS 2021 (Markov state abstraction) · Gangrade AISTATS 2021 (abstain) · Guo HumanML3D CVPR2022 (MM-Dist).
