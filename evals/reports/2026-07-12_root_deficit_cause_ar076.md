# AR-076 — Root Deficit 원인 진단 (원본 MDM: 왜 몸이 안 나갔나)

- Raw: [root_deficit_cause_ar076_v1.json](../snapshots/root_deficit_cause_ar076_v1.json) · harness: [tools/root_deficit_cause_ar076.py](../../tools/root_deficit_cause_ar076.py)
- 표본: MDM locomotion 165 prompt (GT root speed > 0.01) × 3 seed + 동일 prompt HumanML3D GT. bootstrap CI + OLS. GPU 불필요.
- **관측 진단** — 재학습/ablation 아님 (§3-22 아래 claim boundary).

## 판정 요약

| 후보 | 예측 | 결과 | 판정 |
|---|---|---|---|
| **B. 평균 회귀 (mode collapse)** | 기울기<1 + 분산 축소 | **OLS 기울기 = 0.027 [0.019, 0.041]** (≈0!) · MDM std = GT std 의 **5.7%** · 절편 0.0084 | ✅ **지지 — 극단적** |
| C. 적분 오차 | 후반 감속 (누적) | 후반/전반 speed = **0.992 [0.983, 1.001]** (≈1.0) | ❌ **기각** (누적 편향 없음) |
| A. Root 채널 (gait 정상) | 발-골반 상대 gait 는 GT 수준 | **cadence = GT 의 1.15×** [1.04, 1.26] (보존) · stride 진폭은 noisy(3.3×, CI 넓음) | 🔶 **cadence 로 부분 지지** (leg timing 정상) |

## 지배 원인: B (root 채널의 mode collapse / 평균 회귀) — 극단적

핵심 수치: **MDM 의 root 전진 속도는 프롬프트가 무엇을 요구하든 거의 일정**하다.
- OLS 기울기 **0.027** — GT 가 요구하는 이동 속도와 **거의 무관** (기울기 1이면 완벽 추종, 0이면 완전 무시). 절편 0.0084 = MDM 이 사실상 출력하는 상수 속도.
- MDM root 속도 표준편차 = GT 의 **5.7%** — 300 prompt 가 요구하는 다양한 이동량(느린 걸음~빠른 달리기)을 **하나의 좁은 값으로 뭉갬**.

비유: **다양한 목적지를 주문받은 택시가 어디를 요청하든 항상 같은 거리만 가고 멈추는 것.** AR-071 의 "평균 42%" 를 정밀화하면 — 균일하게 42%가 아니라 **절대값이 상수라 요청이 클수록 비율이 더 낮은** 형태다.

이것은 x0-예측 diffusion 의 전형적 **regression-to-mean**: 여러 그럴듯한 이동량의 조건부 분포를 하나로 축소할 때 저주파·큰 변위 성분(=전체 이동)이 평균으로 붕괴. C 기각(시간 누적 아님, 처음부터 균일)과 A(leg cadence 보존)가 이를 보강 — **다리는 걷는 동작을 정상 타이밍으로 만드는데 root 출력만 collapse**.

## 왜 중요한가 (P5·VQ 대비)

> **문구 제한 (2026-07-12 피드백 채택)**: 본 결과의 정식 표현은 **"MDM 에서 root progression collapse 관측"** — "diffusion 패러다임 일반 현상" 은 MDM 단일 관측에서 과확장이므로 타 diffusion(MLD 등) 재현 전까지 주장하지 않는다.

1. **MDM 국소 관측 (일반화 유보)** — root 채널의 mode collapse 는 x0-예측 생성의 실패 유형과 정합하나, **본 측정은 MDM 단일**. 타 diffusion 재현은 future work.
2. **VQ(MotionGPT) root 가 깨끗한 이유 설명** — codebook 이 실데이터 root token 을 그대로 복사(quantize)하므로 평균 붕괴가 없음. AR-071/AR-063 의 "VQ 는 skate 희소" 와 한 그림.
3. **AR-072 처방이 왜 통했는지 설명** — 병인이 "root 값이 상수로 붕괴"라, 접지 제약으로 root 를 다시 풀면(GT-free) 다리가 함의하는 이동량이 복원됨. 처방과 병인이 정합.

## Claim boundary

허용: "원본 MDM root deficit 의 지배 신호는 root-속도 채널의 mode-collapse (평균 회귀) 로 진단됨 — 관측 기반. 적분 누적(C)은 기각, leg cadence(A)는 보존."
금지: 재학습/ablation 없는 "생성 알고리즘의 확정 인과" (관측 진단이지 개입 증명 아님) / MDM 외 diffusion·타 벤치마크 일반화 / A 의 stride 진폭(noisy)을 강한 근거로 사용.

## 남는 불확실성 (§3-22)

- 관측 진단 — 인과는 training-time ablation 필요 (범위 밖).
- "mode collapse" 는 현상학적 명명 (관측: 상수 출력·분산 붕괴); 근저 training 원인(loss·표현·guidance)은 추론.
- MDM 단일 diffusion — MDM/MLD 등 타 diffusion 재현은 AR-024/067 계열 future work.
- A 의 stride 진폭 3.3× 는 접지/보폭 검출 노이즈 가능 — cadence(1.15×, tight)만 강한 근거로.
