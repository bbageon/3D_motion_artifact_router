# AR-074 - Root-Aware Correction STOP Boundary

Status: in-progress (2026-07-12)  
Epic: Tool  
Priority: 🔴  
Parent: AR-072 (처방 성공) — 본 task 는 **어디서 STOP 해야 하나 (no-harm)**

## Goal

root-aware correction 은 강력하나 위험하다. 특히 **non-locomotion / in-place /
already-good / over-correction 꼬리** 에서 root 를 건드리면 harm. 본 task 는:
1. 어떤 motion state 에서 correction 이 harm 인지 (STOP 대상) 정량 식별.
2. **GT-free STOP signal** 도출 — 추론 시 GT 없이 apply/STOP 을 가르는 관측 신호.

이것이 확인되면 routing 주장이 강해진다: 같은 correction 이라도 state 에 따라
apply/STOP 이 달라져야 한다.

## 검정 (원본 MDM pool, GT 는 label 로만)

- **self-STOP 가설**: non-locomotion(GT root speed 낮음, in-place)에서 접지 제약이
  root 전진을 함의하지 않으므로 solve 가 **자동으로 near-no-op** 인가? (self-STOP)
  아니면 root 를 밀어버리는가? (harm) — 측정: 유도 root 변위 절대량.
- **already-good 가설**: baseline deficit 낮은 sample 은 correction 이득 없음/harm.
- **over-correction 꼬리** (ratio_after>1.5, pool 18.2%): 예측자 (constrained_frame_frac,
  motion 길이, path 비율) 로 사전 식별 가능한가?
- **GT-free STOP rule**: 위 관측 신호로 apply/STOP 이진 규칙 도출 + no-harm 검증.

## Success Criteria

- STOP 대상 state 정량 정의 + GT-free 판정 신호 (AUC/분리도).
- non-loco 에서 self-STOP 여부 결론 (harm 이면 명시적 gate 필요).

## Claim Boundary

허용: "root correction 은 state X 에서 harm → STOP 이 정답; GT-free 신호 Y 로 판정 가능".
금지: STOP rule 을 학습된 policy 로 포장 (본 task 는 관측 기반 rule) / A/B 없이 지각 harm 단정.

## Grounding (§3-22)

Safe Orchestration no-harm gate (current_research_position §0) · PhysDiff ICCV2023 (post-proc side effect).
