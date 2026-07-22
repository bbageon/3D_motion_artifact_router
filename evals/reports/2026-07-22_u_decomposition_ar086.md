# AR-086 — u>1 이득 분해: **MM proxy 선호** (물리 복구 아님)

- 사전등록: [spec](../../.claude/docs/dashboard-task-specs/AR-086-u-physical-semantic-decomposition.md) (물리 vs proxy 판정 기준 결과 보기 전 고정) · Raw: [result](../snapshots/u_decomposition_ar086_v1.json) · 코드: [tools/u_decomposition_ar086.py](../../tools/u_decomposition_ar086.py)
- holdout locomotion n=783 (GT root speed>0.01). 물리 = tool 재적용(GPU 불요), MM = AR-085 dataset 재사용.

## 1. 3곡선 (mean, holdout loco) — 결정적 그림

| u | root_ratio (GT대비) | **foot_skate** | MM 개선 |
|---|---|---|---|
| 0.0 | 0.786 | 0.00846 | 0 |
| 0.5 | 0.800 | 0.00678 | +0.188 |
| 0.75 | 0.838 | 0.00627 | +0.242 |
| **1.0** | **0.896** | **0.00606 ← 최소** | **+0.251 ← 최대** |
| 1.25 | 0.976 | 0.00660 ↑ | +0.221 |
| 1.5 | 1.068 | 0.00758 ↑ | +0.157 |
| 2.0 | 1.278 | 0.01002 ↑ | −0.046 |

**핵심**: foot_skate(물리 artifact)와 MM 개선(semantic)이 **둘 다 정확히 u=1.0 에서 최적으로 일치**. u>1 로 가면 foot_skate 는 **악화**(0.00606→0.01002, 발이 반대로 미끄러짐=overshoot)하는데 일부 모션은 MM 만 더 좋아짐.

## 2. 사전등록 판정: **MM proxy 선호**

MM-argmax-u 에서:
- **foot_skate(@argmax) − foot_skate(@u=1) = +0.00182 [+0.0016, +0.002]** (CI 하한>0 = **유의 악화**), **64.5%** 모션이 자기 MM-argmax 에서 skate 악화.
- median root_ratio(@argmax, all)=0.937 (물리 범위) 이나 **foot_skate 악화가 proxy 기준 발동** → 등록 규칙대로 **proxy 선호** (physical_ok=False·proxy_ok=True).

⟹ **u>1 의 per-motion MM 이득(AR-085 H-A 의 43%)은 물리 복구가 아니라 tm2t encoder 가 GT 초과 이동을 선호하는 proxy 특성.** MM-Dist 를 더 낮추려고 발을 더 미끄러뜨리는 것.

## 3. 결론 — AR-085 재해석 (사용자 caveat 확정)

1. **u=1.0 이 물리·semantic 양축 공통 최적** — foot_skate 최소 = MM 최대 = u=1.0. "평균 고정 strength 최적점 = u=1.0"(사용자 리뷰)이 **물리적으로도 확증**.
2. **AR-085 의 "u>1 headroom" 은 대체로 proxy** — MM 단독으로는 u>1 이 이득처럼 보이나, 그 이득은 foot_skate 악화를 대가로 하는 MM proxy artifact. **MM 단독 확장-grid claim 하향.**
3. **잔여 물리 신호 (소량)**: u=1.0 의 root_ratio=0.896 (GT 대비 여전히 −10%). 즉 접지 제약 완전 만족(foot_skate 최소)에도 root 는 GT 보다 약간 덜 감 → **접지 제약(foot_skate)이 GT_root_speed 보다 나은 물리 anchor** (GT 는 "contact" 중에도 발이 조금 움직여 root 를 더 밀 수 있음). GT 도달(u≈1.25)은 foot_skate 악화를 대가로 하므로 "GT 초과가 좋다"가 아님.

## 4. AR-077 "over-correction 꼬리" 확정 (이전 재해석 여지 종결)

AR-085 report §4 에서 "ratio>1.5 가 MM 기준 개선일 수 있다"고 신중히 열어둔 재해석 → **AR-086 이 종결**: over-correction(u>1, ratio>1)은 foot_skate 를 실제로 악화시키는 **진짜 물리 과보정**이 맞고, MM 상 개선은 proxy. AR-077 의 "over-correction 꼬리 = 물리 harm" 라벨링이 옳았음.

## 5. 함의 (다음 방향)

- **정책 복잡화(harm-aware·확장 grid) 는 우선순위 하향** — u>1 이 물리적으로 이득이 아니므로, [0, ~1.25] 로 action 제한 + u=1 중심 운용이 물리적으로 정당. 사용자 판단(정책 복잡화 전 분해)이 정확히 이 결론을 막아줌.
- **접지 추정 개선 (tool target 상향)** 은 root_ratio 0.896→1.0 의 소량 물리 gap 을 foot_skate 악화 없이 회수하는 정공법 후보 (grid 확장보다) — 단 gap 이 −10% 로 작아 우선순위 낮음.
- MM proxy 의 over-translation 편향은 **평가 지표 자체의 한계** — 최종 품질은 지각(A/B)·foot_skate 병행 필수 (P5 규율 재확인).

## Claim Boundary

허용: "MM-argmax-u 에서 foot_skate 유의 악화 → u>1 MM 이득은 proxy 선호 (GT 참조 진단). u=1.0 이 물리·semantic 공통 최적." 금지: 지각 결론(A/B 몫) / GT_root_speed 유일 정답 단정 (foot_skate 가 더 나은 anchor 라는 본 결과와 정합) / proxy 판정을 tm2t 외 encoder 로 일반화.

## Grounding (§3-22)

Guo HumanML3D CVPR2022 (MM-Dist·tm2t 특성) · AR-071/076 (GT-참조 root 진단 계보).
