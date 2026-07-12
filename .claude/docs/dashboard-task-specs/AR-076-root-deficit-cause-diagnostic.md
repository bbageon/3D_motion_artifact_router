# AR-076 - Root Deficit Cause Diagnostic (원본 MDM: 왜 몸이 안 나갔나)

Status: in-progress (2026-07-12)  
Epic: Evidence  
Priority: 🔴  
Parent: AR-071 (현상: v_root=GT의 42%) / AR-072 (처방 성공) — 본 task 는 **원인** 규명

## Goal

AR-071 은 "root 가 부족하다"는 **현상**을, AR-072 는 "고치면 낫다"는 **처방**을 보였다.
본 task 는 **왜 원본 MDM 이 그렇게 생성됐는가** 를 원본 데이터(생성물 vs GT)만으로
3-way 판별한다. GPU 불필요.

## 후보 3종 (문헌 근거, §3-22 — 확정 아님, internal hypothesis)

| 후보 | 기전 | 근거 |
|---|---|---|
| **A. Root 채널** | HumanML3D 263-D 의 root velocity 채널만 부정확 재구성 (관절 pose 는 정상) | Guo CVPR2022 (표현 정의) |
| **B. 평균 회귀** | x0 예측 diffusion 이 이동량 분포를 평균으로 축소 (큰 변위일수록 심함) | Tevet MDM ICLR2023 (x0 예측) + regression-to-mean 일반현상 |
| **C. 적분 오차** | 프레임당 속도 하향 편향이 적분으로 누적 → 후반일수록 뒤처짐 | 속도→위치 적분 파이프라인 |

## 판별 검정 (각 후보의 상이한 예측)

- **A**: 발-골반 상대 stride length·cadence 가 GT 수준인가 (관절 gait 는 정상, root 만 문제).
  측정: 접지발의 골반-상대 stride 진폭·보폭 주기 (MDM vs GT).
- **B**: ratio = f(GT 속도) 회귀 기울기 < 1 (큰 이동일수록 더 축소) + 이동거리 **분산** GT 대비 축소.
  측정: per-prompt (v_root_MDM) vs (v_root_GT) 선형회귀 기울기·절편; std 비교.
- **C**: 부족의 시간 누적 — 전반부 vs 후반부 root drift, 프레임당 편향 부호 일관성.
  측정: 정규화 시간축 전/후반 speed ratio; per-frame (MDM−GT 방향 정렬) 부호.

3 후보 비배타 (동시 가능) — 각 예측의 지지/기각을 독립 보고.

## Success Criteria

- 각 후보 예측의 정량 판정 (지지/기각/부분) + 어느 것이 지배적인지.
- B 지지 시: MDM 국소 버그가 아니라 **diffusion 패러다임 수준 현상** → P5 강화 + VQ(root 깨끗) 대비 설명.

## Claim Boundary

허용: "원본 root deficit 의 지배 기전은 X 로 진단됨 (관측 기반)".
금지: 재학습/ablation 없이 "생성 알고리즘의 확정 원인" (관측 진단이지 개입 증명 아님) / 단일 generator(MDM)·단일 벤치마크 일반화.

## Grounding (§3-22)

Guo et al. HumanML3D CVPR2022 · Tevet et al. MDM ICLR2023 · (VQ 대비) Guo MoMask CVPR2024.
