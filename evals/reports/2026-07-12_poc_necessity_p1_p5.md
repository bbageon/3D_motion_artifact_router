# P5 — Routing Necessity Synthesis (P1~E11 evidence chain, 최종본)

- Board: AR-058-5 (초안 2026-07-12 → **최종화 2026-07-13**: E8~E11 반영) · Raw: [poc_necessity_p1_p5_v1.json](../snapshots/poc_necessity_p1_p5_v1.json)
- Figures: [4-panel (P1~E7)](../../reports/figures/2026-07-12/poc_necessity_p1_p5/p5_evidence_chain_4panel.png) · **[2-panel 결정 계층 (E10~E11)](../../reports/figures/2026-07-13/poc_necessity_p1_p5/p5_decision_layer_2panel.png)** · 생성: [synthesis](../../tools/p5_synthesis_ar058_5.py) / [decision](../../tools/p5_decision_figure_ar058_5.py) (frozen snapshot 로드만)
- Spec 제약 준수: v0.1.0 gate-fire 불인용 (v2만) · [확정 연구 가닥 §0-0](../../.claude/docs/governance/current_research_position.md) pair 구조 · 외부 피드백 1~4차 반영 (pool-scoped·δ 조건부·intervention evidence·harm 분해).

## 0. 주장 (최종 — 2026-07-13)

> **① Coordinate-level artifact score 를 직접 줄이는 보정은 지각 품질을 보장하지 않는다.**
> **② MDM 의 foot skating 은 말단 발 위치의 문제가 아니라 root progression 의 붕괴에서 발생하며, 그 기전을 겨냥한 보정은 물리·표준지표·지각을 함께 회복시킨다 (intervention evidence).**
> **③ 같은 보정이라도 generator/state 에 따라 효과가 반대다 — 본 pool 에서 MDM 은 개선, VQ 는 손상. 따라서 고정 적용은 불가하며 결정 계층(routing)이 필요하다.**
> **④ 다만 개별 모션 단위의 no-harm 결정은 미해결로 남는다 — richer pre-action state 가 순위(AUC)는 유의 개선하나 안전 기준(harmful-apply 유의 감소)은 아직 못 넘는다.**

(원래 spec 의 "P1-P4 가 state-conditioned routing 을 정당화한다" 위에, ①②가 기전 층을, ③④가 결정 층의 근거와 **정직한 한계**를 얹는다.)

## 1. Evidence chain (전 항목 frozen snapshot 박제)

| # | 단계 | 발견 | 핵심 수치 | tier / category |
|---|---|---|---|---|
| P1/P2 | 문제 실재 + 프로파일 | foot skating 은 실재하고 **generator-특이적** | MDM fs 0.0108 = VQ 의 ~2배 (CI 분리); v2 gate fire **23% vs 2%/1%** | real-dist / B |
| P3 | 품질 관계 | MDM foot-skate ↔ 나쁜 FID/R@1 (generator 수준); bone 축은 matched_dist 3/3 방향 일치 | — (상관) | real-dist / A-proxy 연결 |
| P4 | 고정 처방 실패 | fixed FootLock: artifact_total↓ **이면서** foot_skate↑ — **3/3 generator mixed** | MDM fs Δ+0.0145 | real-dist / B·C |
| E5 | 조건부 물리 성공 | coord cleanup: **MDM CI-clean 개선 vs VQ 는 STOP 이 정답** — 조건부 선택 실증 | MDM Δ−0.0013 (p<1e-6) vs VQ +0.0001~3 | real-dist / B |
| E6 | **물리 ≠ 지각** | 물리 개선(fs −48%, legCV 0)에도 **blind 선호 = 우연 — 사전등록 2회 재현** | v1 11/20 (슬로모) · v2 10/20 (실속도) | b1 (한계 명시) |
| E7 | **기전 진단** | MDM root 전진 = **GT 의 42%** (165/165 prompt 전부 부족) + 부족↔fs **ρ=+0.34 (p=8.5e-6)**; 방향 서명은 불확정 | ratio 0.419 [0.39, 0.45] | real-dist 진단 / B-계 |
| E8 | **기전 처방 성공** | root solve: 물리·Cat-A·지각 **삼중 회복** — anchoring(우연)과 정면 대비 | ratio 0.42→1.02 · FID 7.2→3.3 · **A/B 19/20** | b1 + A (intervention evidence) |
| E9 | 생성-기전 원인 | **MDM root progression collapse** — 요구 무관 상수 출력 | OLS 기울기 0.027, 분산 = GT 의 5.7% | 관측 진단 (MDM 단일) |
| E10 | **generator-조건부** | 같은 처방: MDM 개선 vs **VQ 손상** (pool-scoped) | VQ R@1 −0.034/−0.035·MM +0.09/+0.15 유의 | A (pool-scoped) |
| E11 | **결정 계층 미완** | richer state: 순위 유의 개선, **no-harm 기준 미달** | AUC 0.722 vs 0.674 (CI-clean) · harm@30% 감소 비유의 | δ 조건부 exploratory |

Cross-link: [P1](2026-06-27_poc_artifact_occurrence_ar058_1.md) · [P3](2026-06-27_poc_artifact_quality_link_ar058_3.md) · [P4 snapshot](../snapshots/p4_fixed_tool_effect_v1.json) · [E5](2026-07-07_coordinate_footskate_effect_ar061.md) · [E6-v1](2026-07-08_ab_preference_result_ar058_3h.md)/[v2](2026-07-12_ab_preference_result_v2_ar058_3i.md) · [E7](2026-07-12_root_gait_mismatch_ar071.md) · [E8](2026-07-12_ab_preference_result_v3_ar072.md) · [E9](2026-07-12_root_deficit_cause_ar076.md) · [E10](2026-07-12_vq_root_noharm_ar077.md) · [E11](2026-07-13_routing_gate_compare_ar078.md)

## 2. Synthesis logic (최종 — spec 5단계의 확장)

```text
1.  생성 모션에는 artifact 후보가 실재한다 (P1).
2.  이질적 generator 를 canonical 좌표에서 비교 가능하다 (P2).
3.  artifact 후보는 품질 저하와 연결된다 (P3, 상관).
4.  고정 보정은 mixed effect 를 낳는다 → 처방은 조건부여야 한다 (P4).
5.  조건부 보정은 물리 지표를 CI-clean 하게 개선하고, 안 통하는 곳(VQ)에선 STOP 이 정답이다 (E5).
6.  그러나 물리 지표 개선이 지각 선호로 이어지지 않는다 — 사전등록 blind 2회 (E6).
7.  병인 진단: 증상(미끄러짐)의 상류에 root/gait mismatch 가 있다 (E7).
8.  병인을 겨냥한 보정은 물리·표준지표·지각을 함께 회복시킨다 — blind 19/20 (E8, intervention evidence).
9.  그 병인의 정체는 MDM 의 root progression collapse — 요구와 무관한 상수 출력 (E9, 관측).
10. 같은 보정을 VQ 에 고정 적용하면 표준지표가 손상된다 → 고정 적용 불가 (E10, pool-scoped).
11. 개별 모션 단위 no-harm 결정은 미해결 — richer state 로 순위는 개선되나 안전 기준 미달 (E11).
∴ refinement 는 (a) 조건부여야 하고 (routing), (b) 증상 점수가 아니라 기전을 다뤄야 하며,
   (c) 지각/표준 지표 검증 없는 물리 proxy 최적화를 신뢰하면 안 되고,
   (d) per-motion no-harm gate 는 특정된 미해결 문제로 남는다 (순위≠안전 축 분리까지 진단).
```

## 3. One-page introduction summary (논문 도입부용, EN — 최종본)

> Post-hoc refinement of generated human motion is commonly evaluated by physical artifact metrics such as foot-skate scores. We present a pre-registered evidence chain questioning this practice. On a representative 300-prompt HumanML3D pool across three generators, foot skating is real and generator-specific (diffusion ≈ 2× VQ). Fixed post-processing exhibits mixed effects on all three generators, motivating conditional correction. Symptomatic, coordinate-level cleanup then achieves CI-clean physical improvement — yet in two pre-registered blinded A/B tests (single-rater pilot), corrected motions were preferred only at chance (11/20; 10/20). A mechanism diagnostic explains why: the diffusion generator's root progression collapses to a near-constant output regardless of the prompt (regression slope 0.027 against ground-truth speed; variance 5.7% of GT), advancing at ~42% of GT speed on all 165 locomotion prompts. Targeting this mechanism — re-solving the root trajectory from stance-foot contact constraints, GT-free — recovers physics, standard text-motion metrics (R@1 +0.047, MM-Dist −1.0, FID 7.2→3.3), and blinded preference (19/20, p<1e-4) simultaneously, in direct contrast to the symptomatic fixes. The same correction, however, degrades standard metrics when force-applied to VQ generators whose root progression is already consistent — on this pool, correction helps MDM and harms MotionGPT/MoMask, so a decision layer is required. We further show that per-motion no-harm gating remains open: a richer pre-action state significantly improves benefit ranking (held-out AUC 0.72 vs 0.67 single-signal, CI-clean) but does not yet meet a pre-registered harmful-apply reduction bar; ranking ability and safety emerge as distinct axes. We conclude that refinement should target generation mechanisms rather than artifact scores, must be generator/state-conditional, and that safe per-motion gating is a well-specified open problem. *Scope: single benchmark (HumanML3D), single-rater perceptual pilot (multi-rater in progress), MDM as the sole diffusion model; benefit labels are δ-conditional MM-Dist proxies, not perceptual measurements.*

## 4. Claim boundary (최종)

**허용**: 발생·측정가능성·품질 관계·고정 처방 부작용·물리≠지각(2회)·기전 진단·기전 처방의 삼중 회복(intervention evidence)·generator-조건부 효과(pool-scoped)·gate 한계의 정량 특정 — 이 사슬이 "mechanism-aware refinement + 결정 계층 + 미해결 gate 문제"를 지지한다.
**금지**: "routing 이 작동한다" (E11 미달 — richer state 순위 개선까지만) / ArtifactRouter 가 artifact 를 해결했다 / 모든 generator·데이터셋 일반화 (single benchmark; diffusion n=1) / root collapse 가 유일 원인 (ρ²≈0.11; 비이동 잔여 skating 별도 병인) / b1 을 확정 지각 근거로 (b2/b3 = AR-073 진행 중) / "Cat-A 확정" (pool-scoped — 독립 재현 필요) / δ 무관 benefit 수치 인용.

## 5. E8 — mechanism-aware correction 이 지각을 회복한다 (AR-072, ✅ 2026-07-12 완료)

병인 직접 처방 **contact-consistent root solve** (접지발 world-고정 제약에서 root 속도 역산; 다리 무수정, GT-free, world 평행이동 → local·bone·y 구조적 불변) 를 사전등록 guard 와 함께 실행한 결과 — **전 축이 같은 방향으로 개선**:

| 축 | 결과 |
|---|---|
| 기전 표적 | v_root/GT **0.42 → 1.02** (GT 없이 GT-일치 복원) |
| R-Precision @1 / MM-Dist / FID | 0.276→0.319 / 4.89→4.29 / **7.19→3.31** (전부 유의 개선) |
| **blind A/B (b1, locomotion 20쌍)** | **19/20, p<1e-4** |
| 대비 (동일 조건, 처방만 교체) | anchoring: v1 11/20·v2 10/20 (우연) vs **root 19/20 (지지)** |

⟹ **"증상(artifact score)을 고치면 지각이 안 따라오지만, 병인(root/gait mismatch)을 고치면 물리·표준지표·지각이 모두 회복된다."** E6(물리≠지각)의 안티테제가 아니라 그 해답 — refinement 는 기전을 다뤄야 한다는 P5 주장의 직접 실증. **표현 규율: intervention evidence** (개입이 경로·속도·리듬 동시 변경 → 완전 인과 아님). [결과](2026-07-12_ab_preference_result_v3_ar072.md) · [spec](../../.claude/docs/dashboard-task-specs/AR-072-root-aware-correction-design.md)

**E9 — 병인의 생성-기전 진단 (AR-076, 문구 제한)**: **MDM 에서 root progression collapse 관측** — root-속도 채널이 프롬프트 요구와 거의 무관하게 상수로 붕괴 (OLS 기울기 0.027, 분산 = GT 의 5.7%). 적분 누적(C) 기각, leg cadence(A) 보존 → root-특이적. **스코프 엄격 제한: MDM 단일 관측** — "diffusion 패러다임 일반 현상" 은 타 diffusion 재현(future work) 전까지 주장 금지. **관측 진단** (재학습 ablation 아님 — 확정 인과 금지). [AR-076](2026-07-12_root_deficit_cause_ar076.md)

## 5-1. E10 — 같은 처방이 VQ 를 손상시킨다: generator-조건부 (AR-077, 2026-07-13 마감)

VQ(MotionGPT/MoMask) 는 **평균 root deficit 증거 없음** (ratio 0.985 [0.92,1.06] / 1.001 [0.93,1.08] — E9 에서 가설이던 것을 실측 확인). 같은 root 보정을 고정 적용하면 (locomotion 165, prompt-bootstrap B=1000, R-Prec one-seed 수정 후):

| | MDM | MotionGPT | MoMask |
|---|---|---|---|
| R@1 Δ | **+0.047** ↑ | **−0.034** ↓ | **−0.035** ↓ |
| MM-Dist Δ | **−1.00** ↓(좋음) | **+0.087** ↑(악화) | **+0.150** ↑(악화) |
| FID Δ | **−7.3** ↓(좋음) | +0.20 (비유의) | +0.21 (비유의) |

⟹ **pool-scoped 진술**: "현재 HumanML3D representative pool 에서 root correction 은 MDM 의 평균 품질을 개선했지만, VQ 에는 고정 적용 시 평균적인 품질 손상을 일으켰다" (robustness 확인 — 1~4차 피드백의 통계 수정 후에도 방향 유지; 독립 재현은 미완). **고정 적용 불가 → 결정 계층 필요**의 직접 근거. [AR-077](2026-07-12_vq_root_noharm_ar077.md)

## 5-2. E11 — 결정 계층의 정직한 현재: per-motion gate 미완 (AR-078, 사전등록)

gate 5종을 같은 held-out(1,350)에서 비교 (benefit = ΔMM<−δ, δ=calibration VQ median — **전 수치 δ 조건부 exploratory**):

- **순위**: richer pre-action state 가 단일 신호를 **유의하게** 넘음 — AUC 0.722 vs skate 0.674 (paired diff +0.048 [+0.010, +0.088]).
- **안전**: 30% 적용률에서 harmful-apply 감소가 **비유의** (−0.063 [−0.123, +0.003]) → 사전등록 AND 기준 미달, 판정 **"한계 확정"**.
- **축 분리 발견**: 순위 능력(AUC)과 안전(harm)은 다른 축 — skate 는 순위엔 유효하나 harm 이 base 보다 몰리고(0.360>0.326), 기전 신호(mismatch)는 순위 최저·harm 최저(0.274). 다음 설계 = harm 직접 최적화(selective prediction).

⟹ **"routing 이 작동한다" 가 아니라 "routing 은 필요하며(E10), 그 gate 는 순위≠안전 축 분리까지 특정된 미해결 문제"** — 이것이 P5 의 마지막 정직한 문장이다. [AR-078](2026-07-13_routing_gate_compare_ar078.md) · [figure](../../reports/figures/2026-07-13/poc_necessity_p1_p5/p5_decision_layer_2panel.png)

**남은 경계조건 (미검, future work)**: b2/b3 다중 평가자 (AR-073 진행 중) · harm-averse gate (AR-079 후보) · over-correction 꼬리 (ratio>1.5 = pool 18.2%; "walk in place" 급소) · 비이동 잔여 skating 의 별도 병인 · orchestrator 통합 (AR-075 — 검증된 gate 생기면) · 타 데이터셋 (AR-067) · 타 diffusion (MLD 등) 재현.

## 6. Limitations (최종)

1. **지각 = b1** (단일 평가자; blind 사전등록 3회 — 2 fail + 1 support) — 외부 claim 은 3+ 평가자 (AR-073 pack 배포됨).
2. **단일 벤치마크** (HumanML3D 생태계) — KIT-ML 미검 (AR-067). **diffusion n=1** (MDM) — collapse 의 일반성 미검.
3. E7 상관 + E8 개입 = intervention evidence 까지 (완전 인과 아님 — 개입이 경로·속도·리듬 동시 변경).
4. E10 은 pool-scoped (같은 pool 재분석의 robustness — 독립 snapshot 재현 아님). E11 benefit 은 δ 조건부 MM-Dist proxy (지각 아님).
5. 보조 pair (penetration 희소 / BoneCV raw guard / jitter secondary / floating intent-오염) 는 §0-0 caveat 그대로 — 본 종합은 primary pair 사슬에 기초.
6. 물리 개선(Cat-B)의 기능적 가치 (시뮬레이션·retargeting) 는 미검증 — 지각과 별개 축.
