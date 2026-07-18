# AR-078 — Routing Gate 5종 비교 (held-out): 판정 "한계 확정", 단 유의미한 부분 신호

- 사전등록: [spec](../../.claude/docs/dashboard-task-specs/AR-078-routing-gate-compare.md) (gate 5종·지표·판정 기준 — 결과 보기 전 고정) · Raw: [routing_gate_compare_ar078_v1.json](../snapshots/routing_gate_compare_ar078_v1.json) · harness: [tools/routing_gate_compare_ar078.py](../../tools/routing_gate_compare_ar078.py)
- 데이터: 기존 박제물 2개 join (state 2,700 + label 2,699 → **2,699, unjoined 0**). holdout 1,350 (sample_id 단독 split, AR-077 동일). δ=0.0972 (calibration VQ median). **전 수치 δ 조건부 exploratory.**

## 1. 결과 (holdout, base: benefit 35.5% / harm 32.6%)

| gate | benefit-AUC [CI] | AUC diff vs G2 [CI] | harm@30% | capture@30% |
|---|---|---|---|---|
| G1 generator-only | 0.682 [0.625, 0.714] | −0.004 [−0.054, +0.051] | 0.296 | 0.499 |
| G2 skate-only | 0.674 [0.627, 0.715] | — | **0.360** | 0.451 |
| G3 mismatch-only | 0.625 [0.574, 0.667] | −0.051 [−0.107, +0.009] | **0.274** | 0.466 |
| **G4 richer (no-gen)** | **0.716** [0.676, 0.753] | **+0.042 [+0.005, +0.080]** ✓ | 0.328 | 0.474 |
| **G5 richer (+gen)** | **0.722** [0.684, 0.758] | **+0.048 [+0.010, +0.088]** ✓ | 0.291 | 0.511 |

harm@30% bootstrap diff vs G2: G4 −0.029 [−0.084, **+0.027**] ✗ · G5 −0.063 [−0.123, **+0.003**] ✗ (아슬아슬 미달)

## 2. 사전등록 판정: **"한계 확정"**

기준 = AUC diff CI 하한>0 **AND** harm@30 diff CI 상한<0 (둘 다). G4/G5 모두 **AUC 조건은 충족**했으나 **harm 조건 미충족** (G5 는 상한 +0.003 으로 겨우) → 등록된 분기대로:

> **현 state 로 per-motion 결정은 no-harm 기준을 아직 못 넘는다 — generator-level rule 이 현재의 정직한 한계.** 기준 사후 완화 없음 (G5 의 near-miss 를 "사실상 통과"로 재해석하지 않는다).

## 3. 판정 밖에서 배운 것 (exploratory — 다음 설계의 재료)

1. **Richer state 는 순위를 유의하게 개선한다** (AUC +0.042/+0.048, CI-clean; generator 없이도 G4 0.716). "state 가 신호를 더 담는다"는 방향 자체는 확인 — 부족한 건 **안전 축**.
2. **순위 능력(AUC)과 안전(harm) 은 다른 축이다**: G2(skate)는 AUC 는 되지만 harm@30 **0.360 = base(0.326)보다 나쁨** — skate 상위 모션은 이득도 크지만 망가질 위험도 함께 큼. 반대로 G3(mismatch)는 AUC 최저인데 **harm@30 최저 (0.274)** — 기전 feature 는 "안전한 apply" 를 고르는 신호. G5 가 both 를 절충 (harm 0.291 + capture 0.511 최고).
3. G1(generator-only) AUC 0.682 ≈ G2 — 피드백 진단대로 **generator-level rule 은 강한 baseline**.
4. 함의: 다음 시도는 "더 많은 feature"가 아니라 **harm 을 직접 최적화하는 목적함수** (benefit 분류가 아니라 harm-averse selective prediction — abstain 비용을 명시한 학습) 또는 **harm label 별도 예측기**. (future work — 등록 후 진행.)

## 4. 이 결과가 연구 서사에서 갖는 위치

- P5 의 "routing 문제가 남아 있다" 문구가 **정량 근거를 얻음**: richer state 로도 no-harm 기준의 per-motion gate 는 미완 — 무엇이 부족한지(안전 축)까지 특정됨.
- **AR-075 (orchestrator 통합) 는 보류 유지**: 사전등록 기준을 넘은 gate 가 없으므로 "검증된 gate 만 통합" 원칙에 따라 편입할 gate 가 아직 없다. 현행 문서화 가능한 안전 운용 = generator-level rule (MDM 만 apply) + 보수적 STOP.
- **AR-077 마감** (4차 피드백 합의: "AR-078 결과로 마감") — 진단 결론: MDM 개선/VQ 악화 (pool-scoped) + per-motion gate 미완 (본 결과로 확정).

## Claim Boundary

허용: "같은 held-out 에서 richer pre-action state 는 benefit 순위를 유의 개선(AUC +0.04~0.05)했으나, 사전등록된 no-harm 기준(harm@30% 유의 감소)은 충족하지 못했다 — per-motion routing gate 는 미완."
금지: "routing 작동" / G5 near-miss 의 통과 재해석 / MM-Dist proxy 를 지각 benefit 으로 / LR 결과의 정책 성능 일반화 / δ·단일 벤치마크 밖 확정.

## Grounding (§3-22)

Gangrade et al. AISTATS 2021 (selective prediction — abstain 원리) · Guo et al. HumanML3D CVPR 2022 (MM-Dist). harm-averse 학습 확장은 진행 시 별도 grounding 등록.
