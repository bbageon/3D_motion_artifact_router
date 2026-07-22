# AR-085 — Over-correction 개방 (u∈[0,2]): H-A 지지 (u=1은 ceiling 아님), H-B 한계

- 사전등록: [spec](../../.claude/docs/dashboard-task-specs/AR-085-strength-q-v2-open-u.md) (u_grid·H-A·H-B 기준 결과 보기 전 고정) · Raw: [result](../snapshots/strength_q_v2_result_ar085_v1.json) · [dataset](../snapshots/strength_q_v2_dataset_ar085_v1.csv) · 코드: [dataset](../../tools/strength_q_v2_dataset_ar085.py)/[analyze](../../tools/strength_q_v2_analyze_ar085.py)
- 사용자 directive 2026-07-22: "열어보자. 과보정이 아닐 수도 있어". **전 수치 δ(0.097) 조건부 exploratory·MM proxy·one-shot.**

## 1. H-A (over-correction 효과 존재) — **지지**: 사용자 가설 확인

| 지표 (holdout locomotion) | 값 |
|---|---|
| **argmax-u > 1 인 motion 비율** | **42.8%** (사전 임계 15% 크게 초과) |
| generator 별 | MDM **66.2%** / MotionGPT 33.8% / MoMask 28.4% |
| **best-u(=per-motion argmax) vs 고정 u=1 개선** | **+0.53 [0.46, 0.61]** (CI-clean) — ⚠️ oracle-u(label-peek) 상한, 정책 성능 아님 |

**improvement 곡선 (mean, holdout loco)** — 핵심 그림:

| u | 0.25 | 0.5 | 0.75 | **1.0** | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|
| mean imp | 0.122 | 0.230 | 0.294 | **0.312** | 0.279 | 0.198 | **−0.072** |

**해석 (사용자 가설에 대한 정확한 답)**:
- **u=1 은 ceiling 이 아니었다** — 개별 모션 최적이 u>1 로 널리 퍼짐(43%). u=1 의 "제약 완전 강제"가 **우리 접지 추정 편향으로 target 을 낮게 잡아 under-correction** 이었다는 정의(§5-3)가 실측 확인.
- **그러나 "과보정이 없다"는 아니다** — mean 곡선은 u≈1.0~1.25 에서 정점, **u=2.0 에서 음수(진짜 과보정)**. 즉 과보정의 경계는 u=1 이 아니라 **모션별로 ~1.25~1.5 근처**이고, u=2.0 은 대부분 과함.
- ⟹ 정확한 진술: **"u=1 은 상한이 아니라, 다수 모션(43%)에서 최적 strength 는 1~1.5 구간에 있다. 구 [0,1] grid 는 이 sweet spot 을 잘라내고 있었다. 단 u=2.0 은 진짜 과보정."**

## 2. H-B (v2 정책 > v1 정책) — **한계**

| 정책 (holdout) | mean imp | harmful |
|---|---|---|
| v2 (u≤2 grid) | 0.243 | 0.106 |
| v1 (u≤1 grid) | 0.220 | 0.118 |
| **v2 − v1** | **+0.024 [−0.006, +0.054]** | −0.012 [−0.033, +0.008] |

- 방향은 개선(improvement↑·harm↓)이나 **둘 다 CI 가 0 포함 = 비유의** → 사전 기준 미달, 판정 "한계".
- 원인: v2 정책이 u>1 을 27% 사용(u=1.25 11%·1.5 6%·2.0 10%)하나 **u=2.0(음수 구간)에 일부 오배정** → oracle-u 이득(+0.53)을 정책이 회수 못 함. AR-078/079 와 같은 **"효과 존재(H-A) ≠ 정책 회수(H-B)"의 순위-실현 격차** 재확인.

## 3. 종합 — 사용자 가설의 정확한 결론

**"과보정이 아닐 수도 있다"는 절반 맞다**:
1. ✅ **u=1 근처~1.5 는 과보정이 아니라 (다수 모션에서) 부족했던 보정의 연장** — 43% 가 u>1 에서 최적, MDM 은 66%. action space 개방은 과학적으로 옳았고, 구 상한 가정을 데이터로 반증.
2. ❌ **u=2.0 은 진짜 과보정** (mean imp 음수) — 무한정 강할수록 좋지는 않음.
3. ⚖️ **최적은 모션-의존적** — 그래서 정책이 필요하나, 현 state 로는 oracle-u 이득을 유의하게 회수 못 함(H-B 한계). 순위-실현 격차가 강도 축에서도 반복.

**AR-077 "over-correction 꼬리(ratio>1.5)" 재해석 (신중)**: GT-root-speed 대비 과속(ratio>1.5)이 MM-Dist(텍스트 정렬) 기준으로는 여전히 개선인 경우가 있음 → "GT 속도 초과 = 나쁨"이 항상 참은 아님. 단 두 지표(GT-speed vs MM-proxy)의 divergence 이지 "GT 초과가 좋다"의 증명은 아님 — 별도 검증 대상.

## Claim Boundary

허용: "u∈[0,2] 개방 시 holdout 43% 모션이 u>1 에서 최적(oracle-u), best-u 가 고정 u=1 대비 +0.53(상한); u=2.0 은 과보정; 확장-grid 정책은 v1 대비 비유의(+0.024). δ 조건부·MM proxy·one-shot."
금지: best-u(+0.53) 를 정책 성능으로 인용 (oracle-u 상한) / "정책 개선"(H-B 한계) / 지각·GT-최적 주장 / v1 결과와 u_max 무구분 인용.

## 다음 (별도)

- 정책이 u=2.0 오배정을 줄이도록 harm-aware 강화 (AR-079 계열) 또는 richer state.
- (선택) u>1 sweet spot 의 지각 검증 — best-u 보정본 A/B (AR-082 계열).

## Grounding (§3-22)

Agarwal NeurIPS 2021 (NAM) · Gangrade AISTATS 2021 · Guo CVPR2022 (MM-Dist).
