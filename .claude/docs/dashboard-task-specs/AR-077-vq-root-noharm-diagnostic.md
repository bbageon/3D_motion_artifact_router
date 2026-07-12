# AR-077 - VQ Root State + No-Harm Diagnostic (MotionGPT / MoMask)

Status: in-progress (2026-07-12)  
Epic: Evidence  
Priority: 🔴  
Parent: AR-072 (MDM 성공) / AR-074 (GT-free STOP 신호) / AR-076 (MDM mode collapse)

## Goal (프레이밍 — 결과 보기 전 고정)

MDM 에서 성공한 root-aware correction 이 VQ(MotionGPT/MoMask)에도 **필요한가, 아니면
STOP 이 정답인가**. 목적은 "VQ 에서도 좋다" 가 아니라 **generator-conditioned routing 근거**:
같은 correction 이 generator/state 에 따라 apply/STOP 이 달라져야 함을 확인.

## 사전 등록 3-시나리오 (결과 보기 전 — 어떤 결과든 유효)

| 시나리오 | VQ root ratio | correction 효과 (Cat-A) | routing 함의 |
|---|---|---|---|
| **깔끔** | ≈1.0 (deficit 없음) | 없음/악화 (no-harm 위반) | generator-conditioned: MDM apply / VQ STOP — 명확 |
| **애매** | 0.7~0.85 (덜 부족) | 소폭 개선 or 무해 | **state-conditioned** (deficit 정도로 결정) — 더 강한 주장 |
| **놀람** | 낮음 (MDM 수준) | 개선 | 병인이 VQ 에도 — mode collapse 재현 |

**성공 기준 = "결정이 generator/state 에 따라 달라진다"** (VQ 악화만을 성공으로 미리 고정하지 않음 — HARKing 차단).

## 검정 (diagnostic 먼저, A/B 는 애매/놀람 시만)

1. **root ratio**: MotionGPT/MoMask v_root/GT vs GT (AR-071/076 harness 재사용). VQ deficit 있나.
2. **root correction 적용** (RootGaitConsistencyTool) 후 physical (fs·induced_disp·path_gain) + **Cat-A** (R-Prec/MM-Dist/FID, mgpt env — no-harm 판정).
3. **AR-074 GT-free STOP 신호** (path_gain AUC, induced_disp) 를 VQ 에 적용 → STOP 판정 재현되나.
4. **routing decision 표**: MDM=apply / VQ=? (apply/STOP/state-conditioned).
5. A/B: diagnostic 이 "애매/놀람" 이면 소규모 (깔끔하면 수치로 충분).

비교 공정성: VQ 도 **같은 필터**(GT locomotion + foot-skate 분포)로 표본 — generator 차이와 표본 차이 분리.

## Claim Boundary

허용: "VQ 는 root deficit X / correction 이 Cat-A 를 Y → routing 은 generator/state 조건부여야".
금지: no-harm(Cat-A 악화)의 지각 단정 없이 A/B claim / VQ 결과를 MDM 처럼 일반화 / 3-시나리오 중 하나를 사후 성공으로 재프레임.

## Grounding (§3-22)

Guo MoMask CVPR2024 (VQ) · Jiang MotionGPT NeurIPS2023 (VQ) · Tevet MDM ICLR2023 · Safe Orchestration no-harm (position §0).
