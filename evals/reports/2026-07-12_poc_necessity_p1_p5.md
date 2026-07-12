# P5 — Routing Necessity Synthesis (P1~P5 evidence chain, 개정 frame)

- Board: AR-058-5 · Raw: [poc_necessity_p1_p5_v1.json](../snapshots/poc_necessity_p1_p5_v1.json) · Figure: [4-panel](../../reports/figures/2026-07-12/poc_necessity_p1_p5/p5_evidence_chain_4panel.png) · 생성: [tools/p5_synthesis_ar058_5.py](../../tools/p5_synthesis_ar058_5.py) (frozen snapshot 로드만, 재계산 없음)
- Spec 제약 준수: v0.1.0 gate-fire 불인용 (v2만) · [확정 연구 가닥 §0-0](../../.claude/docs/governance/current_research_position.md) pair 구조 · A/B 결과로 어조 확정 (완료: fail×2 → 개정 frame)

## 0. 주장 (개정 frame — 2026-07-12 확정)

> **Coordinate-level artifact score 를 직접 줄이는 보정은 지각 품질을 보장하지 않는다.**
> **MDM 의 foot skating 은 말단 발 위치의 문제가 아니라 root trajectory 와 gait speed 의 mismatch 에서 발생할 수 있다.**
> **따라서 post-generation refinement 는 artifact score 최적화가 아니라 motion 기전(mechanism)을 고려해야 하며, "무엇을 언제 고치고 언제 멈출지"의 결정 계층이 필요하다.**

(원래 spec 의 주장 "P1-P4 가 state-conditioned routing 을 정당화한다"는 유지되며, 위 3문장이 그 위에 **더 깊은 층**으로 얹힌다 — routing 의 필요성은 "고정 처방의 실패"에서, 기전 고려의 필요성은 "조건부 처방의 지각 실패 + 병인 진단"에서 나온다.)

## 1. Evidence chain (전 항목 frozen snapshot 박제)

| # | 단계 | 발견 | 핵심 수치 | tier / category |
|---|---|---|---|---|
| P1/P2 | 문제 실재 + 프로파일 | foot skating 은 실재하고 **generator-특이적** | MDM fs 0.0108 = VQ 의 ~2배 (CI 분리); v2 gate fire **23% vs 2%/1%** | real-dist / B |
| P3 | 품질 관계 | MDM foot-skate ↔ 나쁜 FID/R@1 (generator 수준); bone 축은 matched_dist 3/3 방향 일치 | — (상관) | real-dist / A-proxy 연결 |
| P4 | 고정 처방 실패 | fixed FootLock: artifact_total↓ **이면서** foot_skate↑ — **3/3 generator mixed** | MDM fs Δ+0.0145 | real-dist / B·C |
| E5 | 조건부 물리 성공 | coord cleanup: **MDM CI-clean 개선 vs VQ 는 STOP 이 정답** — 조건부 선택 실증 | MDM Δ−0.0013 (p<1e-6) vs VQ +0.0001~3 | real-dist / B |
| E6 | **물리 ≠ 지각** | 물리 개선(fs −48%, legCV 0)에도 **blind 선호 = 우연 — 사전등록 2회 재현** | v1 11/20 (슬로모) · v2 10/20 (실속도) | b1 (한계 명시) |
| E7 | **기전 진단** | MDM root 전진 = **GT 의 42%** (165/165 prompt 전부 부족) + 부족↔fs **ρ=+0.34 (p=8.5e-6)**; 방향 서명은 불확정 | ratio 0.419 [0.39, 0.45] | real-dist 진단 / B-계 |

Cross-link: [P1](2026-06-27_poc_artifact_occurrence_ar058_1.md) · [P3](2026-06-27_poc_artifact_quality_link_ar058_3.md) · [P4 snapshot](../snapshots/p4_fixed_tool_effect_v1.json) · [E5](2026-07-07_coordinate_footskate_effect_ar061.md) · [E6-v1](2026-07-08_ab_preference_result_ar058_3h.md)/[v2](2026-07-12_ab_preference_result_v2_ar058_3i.md) · [E7](2026-07-12_root_gait_mismatch_ar071.md)

## 2. Synthesis logic (개정 — spec 5단계의 확장)

```text
1. 생성 모션에는 artifact 후보가 실재한다 (P1).
2. 이질적 generator 를 canonical 좌표에서 비교 가능하다 (P2).
3. artifact 후보는 품질 저하와 연결된다 (P3, 상관).
4. 고정 보정은 mixed effect 를 낳는다 → 처방은 조건부여야 한다 (P4).
5. 조건부 보정은 물리 지표를 CI-clean 하게 개선하고, 안 통하는 곳(VQ)에선 STOP 이 정답이다 (E5).
6. 그러나 물리 지표 개선이 지각 선호로 이어지지 않는다 — 사전등록 blind 2회 (E6).
7. 병인 진단: 증상(미끄러짐)의 상류에 root/gait mismatch 가 있다 (E7).
∴ refinement 는 (a) 조건부여야 하고 (routing), (b) 증상 점수가 아니라 기전을 다뤄야 하며
   (mechanism-aware), (c) 지각/표준 지표 검증 없는 물리 proxy 최적화를 신뢰하면 안 된다.
```

비유: 증상(기침) 점수를 낮추는 약(진해제)이 검사 수치는 좋게 만들지만 환자가 낫다고 느끼지 못했고(2회 임상), 부검적 진단으로 병인(기도 아래쪽 문제)을 찾은 것 — 다음 처방은 기침이 아니라 병인을 향해야 한다.

## 3. One-page introduction summary (논문 도입부용, EN)

> Post-hoc refinement of generated human motion is commonly evaluated by physical artifact metrics such as foot-skate scores. We present a pre-registered evidence chain questioning this practice. On a representative 300-prompt HumanML3D pool across three generators, foot skating is real and generator-specific (diffusion ≈ 2× VQ; repaired contact-gate fire 23% vs 1–2%). Fixed post-processing exhibits mixed effects on all three generators (target proxy improves while foot-skate worsens), motivating state-conditioned correction. A contact-aware cleanup tool then achieves CI-clean physical improvement exactly where headroom exists (diffusion) while the correct action elsewhere is abstention (STOP). However, in two pre-registered blinded A/B tests (single-rater pilot), corrected motions were preferred at chance level (11/20; 10/20) despite 20/20 physical improvement — physical gains did not translate into perceived quality. A mechanism diagnostic explains why the symptomatic fixes fall short: the diffusion generator advances its root at only ~42% of ground-truth speed for the same prompts (165/165 prompts), and this deficit correlates with foot skating (ρ=+0.34, p<1e-5). We conclude that (i) refinement must be state-conditioned (what/when/whether to correct), and (ii) optimizing coordinate-level artifact scores is insufficient — refinement should target generation mechanisms (e.g., root-trajectory/gait consistency), with perceptual and standard-metric validation as first-class evidence. *Scope: single benchmark (HumanML3D), single-rater perceptual pilot; multi-rater validation and cross-dataset replication are future work.*

## 4. Claim boundary

**허용**: 관측된 발생·공통좌표 측정가능성·품질 관계·고정 처방의 부작용·조건부 물리 개선·지각 불충분(2회)·기전 진단이 "artifact-aware **and mechanism-aware** refinement + 결정 계층"의 필요성을 동기화한다.
**금지**: ArtifactRouter 가 artifact 를 해결했다 / 지각 품질 향상을 달성했다 / 모든 generator·데이터셋에 일반화된다 / root mismatch 가 foot skating 의 **유일** 원인이다 (ρ²≈0.11 — 복합 원인) / b1 지각 결과를 확정 지각 근거로 인용한다 (외부 공개 시 b2/b3 필수, AR-023).

## 5. Future work — mechanism-aware correction 가능성 (AR-072, 게이트 대기)

병인을 직접 다루는 후보: **contact-consistent root solve** — 접지 frame 의 stance foot world 속도가 0 이 되도록 root 수평 변위를 재해석 (발을 root 에 맞추는 대증요법의 역방향; GT-free). 착수 시 사전등록: semantic guard (이동 거리 = prompt 의미 → R-Precision/MM-Dist equal-N 보존 의무) + 새 pair 의 새 blind A/B. 본 개입 실험이 E7 상관의 인과 확증을 겸한다. [spec](../../.claude/docs/dashboard-task-specs/AR-072-root-aware-correction-design.md)

## 6. Limitations

1. **지각 = b1** (단일 평가자, 사전등록 blind 2회) — 방향 결정용으로 충분하나 외부 claim 은 3+ 평가자 필요.
2. **단일 벤치마크** (HumanML3D 생태계 — 시험지·정답지·채점기·체크포인트) — KIT-ML 미검 (AR-067).
3. E7 은 상관+보편성 — 인과는 개입(AR-072)으로만.
4. 보조 pair (penetration 희소 / BoneCV raw guard / jitter secondary / floating intent-오염) 는 §0-0 caveat 그대로 — 본 종합은 primary pair 사슬에 기초.
5. 물리 개선(Cat-B)의 기능적 가치 (시뮬레이션·retargeting 등 기계적 사용처) 는 미검증 — 지각과 별개 축으로 남음.
