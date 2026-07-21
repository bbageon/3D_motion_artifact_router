# AR-085 - Strength-Q v2: over-correction 개방 (u ∈ [0,2]) 사전등록

Status: in-progress (2026-07-22 착수 — 사용자 directive "열어보자. 과보정이 아닐 수도 있어")  
Epic: RL-Q  
Priority: 🔴  
Parent: AR-081 (v1, u≤1) / strength 정식 정의 (action_space_provenance §5-3)

## 동기 (사용자 가설 — 결과 보기 전 등록)

u=1 은 "우리 접지 추정 기준 완전 강제"일 뿐 GT-최적이 아니다. 접지 추정이 낮게 잡히면
u=1 도 under-correction 일 수 있으므로, **u>1 이 일부 모션에서 실제로 더 나을 수 있다**
(= 과보정이 아니다). v1 의 u≤1 상한 가정을 제거하고 데이터로 검증한다.

## 사전등록 (2026-07-22 — 결과 보기 전 고정)

### Action space (§5-3 등록)

- u_grid = **{0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}** (u=0=STOP). U_MAX=2.0 (tool clip).
- u=0/.25/.5/.75/1.0 라벨 = AR-081 dataset 재사용. **신규 산출 = u ∈ {1.25, 1.5, 2.0}** (3 컬럼 × 2,699 motion, mgpt).
- 라벨: improvement(u) = MM-Dist(orig) − MM-Dist(corrected@u). δ = calibration VQ median (AR-077/081 승계).

### H-A: over-correction 효과 존재 검정 (핵심 — 사용자 가설)

- 측정: (i) grid 상 per-motion argmax-u 분포, (ii) mean/median improvement(u) 곡선.
- **지지** = locomotion holdout 에서 **argmax-u > 1 인 motion 비율 ≥ 15%** AND best-u improvement 가 u=1 improvement 대비 유의 우위 (prompt-bootstrap CI 하한 > 0).
- **기각** = u=1 이 사실상 ceiling (argmax>1 < 5% 이거나 best-u ≈ u=1). → "과보정 맞음, [0,1] 로 충분" 확정 (v1 정당화).
- 부분 = 그 사이 (일부 motion 만 u>1 이득).

### H-B: 정책 개선 검정 (v2 정책 vs v1 정책)

- 확장 grid Q(s,u) 재학습 (v1 과 동일 13차원 additive 구조·λ calibration 전용).
- **지지** = v2 정책이 v1 정책(u≤1) 대비 holdout mean improvement 유의 우위 AND harmful rate 증가 없음 (paired bootstrap).
- 아니면 한계 (u 개방이 정책 성능은 안 올림 — H-A 와 별개로 기록).

### 규율

- 전 수치 δ 조건부 exploratory (MM proxy). "정책 작동" 표현은 지각 검증(AR-082 계열) 후.
- v1 결과 무수정 (append-only) — v2 는 별도 snapshot. u_max 로 구분 인용 (§6-15).
- 기준 사후 조정 금지 (H-A/H-B 판정 분기 그대로).

## Claim Boundary

허용: "u∈[0,2] 개방 시 over-correction(u>1) 이 일부 motion 에서 이득인지 + 정책 개선 여부 (δ 조건부·MM proxy·one-shot)". 금지: 지각 개선 / GT 대비 최적성 주장(GT-free 라벨 아님·MM proxy) / closed-loop 일반화.

## Grounding (§3-22)

Agarwal NeurIPS 2021 (NAM) · Gangrade AISTATS 2021 (abstain) · Guo CVPR2022 (MM-Dist). u=1≠optimal 논리 = 우리 접지 추정 편향 (내부 pilot).
