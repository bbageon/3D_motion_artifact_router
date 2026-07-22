# AR-086 - u>1 이득의 물리–semantic 분해 (사전등록)

Status: in-progress (2026-07-22 착수 — 사용자 우선순위 "정책 복잡화 전에 분해 먼저")  
Epic: Evidence  
Priority: 🔴  
Parent: AR-085 (H-A: u>1 oracle headroom 존재, 단 원인 미분리)

## 질문

AR-085 의 "u>1 이 MM-Dist 를 개선한다"가 — (a) **물리적 under-correction 복구** (u=1 이 접지 추정 편향으로 root 를 덜 밀어, u>1 이 GT 에 근접) 인가, (b) **MM proxy 특성** (tm2t encoder 가 root 이동량 큰 모션을 GT 무관하게 선호) 인가.

## 측정 (holdout locomotion, GT root speed>0.01; motion3d — 물리는 GPU 불요)

각 motion × u_grid{0..2.0}:
- **root_ratio(u)** = root_speed(corrected@u) / GT_root_speed (GT 참조 진단).
- **foot_skate_world(u)** = 접지발 world 미끄러짐.
- **MM-Dist(u)** = AR-085 dataset 재사용 (신규 embed 없음).

## 사전등록 판정 기준 (결과 보기 전 고정)

per-motion **MM-argmax-u** 에서:

- **물리 복구 지지** IF: median root_ratio(@MM-argmax) ∈ **[0.85, 1.20]** (GT 근접) **AND** foot_skate(@MM-argmax) ≤ foot_skate(@u=1) (미끄러짐 비악화).
- **proxy 선호 지지** IF: median root_ratio(@MM-argmax) **> 1.35** (GT 크게 초과) **OR** foot_skate(@MM-argmax) 가 foot_skate(@u=1) 대비 유의 증가 (MM 은 개선하나 물리 악화 = proxy 가 이동량 선호).
- **혼합** = 그 사이 (둘 다 부분).

보조: root_ratio(u)·foot_skate(u)·MM(u) 3곡선 동반 제시; MM 개선(past u=1)이 root_ratio→1 수렴과 상관인지(물리) root_ratio 증가 자체와 상관인지(proxy) 검정.

## 함의

- **물리 복구** → u>1 은 진짜 이득, tool 의 접지 target 을 상향(추정 개선)하는 게 정공법 (grid 확장보다). AR-085 H-A 신규성 강화.
- **proxy 선호** → u>1 이득은 MM proxy artifact — 지각/GT-referenced 로만 확정 가능, MM 단독 확장 grid claim 하향.
- **혼합** → 부분 물리·부분 proxy, 지각 검증(AR-082 계열) 필수.

## Claim Boundary

허용: "MM-argmax-u 의 물리(root_ratio·foot_skate) 특성으로 u>1 이득의 물리 vs proxy 분해 (GT 참조 진단)". 금지: 지각 결론(A/B 몫) / GT_root_speed 를 유일 정답으로(텍스트가 GT 보다 큰 이동 요구 가능성 caveat).

## Grounding (§3-22)

Guo HumanML3D CVPR2022 (MM-Dist·tm2t) · AR-071/076 (GT-참조 root 진단 계보).
