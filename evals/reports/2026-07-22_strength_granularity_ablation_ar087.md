# AR-087 — strength 세분화 필요한가: **binary {STOP, full} 충분** (연속 강도 불필요)

- 사용자 질문 "굳이 strength 조절해서 보정할 필요 있나"에 직접 답. 동일 additive Q 기계, grid 만 교체. Raw: [snapshot](../snapshots/strength_granularity_ablation_ar087_v1.json) · [tool](../../tools/strength_granularity_ablation_ar087.py). δ 조건부·MM proxy·one-shot.

## 1. 결과 (holdout)

| 정책 (u_grid) | mean improvement | **harmful** | u 분포 |
|---|---|---|---|
| **binary** {0, 1.0} | 0.2165 | **0.0919** (최저) | STOP 64% / full 36% |
| graded_v1 {0,.25,.5,.75,1} | 0.2196 | 0.1178 | STOP 52% / 중간 15% / full 33% |
| graded_v2 {0..2.0} | 0.2434 | 0.1059 | (over-correction 포함) |

**paired diff vs binary (bootstrap):**
- graded_v1 − binary: improvement **+0.003 [−0.012, +0.016]** (비유의) · harmful **+0.026 [+0.012, +0.040]** (graded 가 **유의하게 더 harm**)
- graded_v2 − binary: improvement +0.027 [+0.001, +0.052] (경계 유의) — 단 AR-086 에서 이 이득은 **MM proxy**(foot_skate 악화)로 판정됨, 물리 이득 아님.

## 2. 사전등록 판정: **binary 충분 — 연속 강도 조절 불필요**

- graded_v1(중간 강도 추가)은 binary 대비 **improvement 동률·harm 오히려 증가** → 세분화가 가치를 못 더함. 중간 강도(u=0.25/0.5/0.75)는 완전 이득 없이 harm 만 늘림.
- graded_v2 의 미미한 improvement 우위는 over-correction(proxy) 기여 → 물리적으로 무효.
- ⟹ **"얼마나 세게"(연속 강도)는 불필요. 필요한 결정은 "고칠까 말까"(binary, u∈{0,1.0}).**

## 3. 사용자 질문에 대한 정확한 답

**연속 강도 조절은 필요 없다 — 맞다. 단 "보정 자체가 불필요"는 아니다.**

- ✅ **결정 B (연속 강도, 0<u<1): 불필요** — binary {STOP, u=1.0} 가 improvement 동률·harm 우위. AR-081 의 "strength-Q 정책이 강도를 학습" 프레이밍은 **과잉설계**였고, 정책의 실제 이득은 강도 gradation 이 아니라 STOP 결정에서 나왔다.
- ✅ **결정 A (apply/STOP, u∈{0,1.0}): 핵심** — binary 정책이 64% STOP·36% full. "전부 u=1 적용"(harm 0.326, AR-081) vs "라우팅 STOP"(harm 0.092) 의 차이가 시스템 가치의 전부. generator/state-조건부(E10)가 여기 해당.

## 4. 연구 서사 단순화 (논문 관점 — 오히려 강해짐)

**"학습된 연속 강도 정책 Q(s,u)"** → **"generator/state-조건부 binary 보정 게이트 (u=1.0 또는 STOP)"**. Occam: 동일 증거를 더 단순한 claim 으로. AR-081/082 의 지각·정량 이득은 binary 게이트로 재귀속(정책의 STOP 결정이 이득원). AR-086(u=1.0 물리·semantic 최적) + AR-087(binary 충분)이 **강도 축 전체를 "u=1.0 고정 + STOP 게이트"로 붕괴** — 연속 action space(§5-2 reframe)는 결국 불필요.

## Claim Boundary

허용: "연속/graded strength 조절은 binary {STOP, u=1.0} 대비 improvement 동률·harm 비개선(오히려 증가) → 불필요; 가치는 apply/STOP 라우팅(결정 A). δ 조건부·MM proxy·one-shot." 금지: "보정 불필요"(STOP-vs-correct 는 핵심) / binary 게이트의 지각 재검 생략(AR-082 는 주로 u=1.0 을 검증했으므로 binary 와 정합하나, graded→binary 전환의 지각 동등성은 미검) / closed-loop 일반화.

## 다음

- P5 의 강도 정책 절(E12) 을 binary 게이트로 재서술.
- (선택) binary 게이트의 지각 재확인 — 이미 AR-082 가 대부분 u=1.0 검증이라 강한 재검 불요.

## Grounding (§3-22)

Agarwal NeurIPS 2021 (NAM) · Gangrade AISTATS 2021 (abstain=STOP) · Guo CVPR2022 (MM-Dist).
