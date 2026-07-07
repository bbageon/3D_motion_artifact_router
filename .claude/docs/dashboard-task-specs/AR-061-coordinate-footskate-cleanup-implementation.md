# AR-061 - CoordinateFootSkateCleanupTool Prototype Implementation + Evaluation

Status: **completed** (2026-07-07, commit 196e360)  
Epic: Tool  
Priority: Urgent  
Parent: AR-060 (design freeze) · Gate: AR-062 audit verdict **PROCEED** 후 착수

## Goal

[AR-060 frozen spec](../design/coordinate_footskate_cleanup_tool_frozen_spec.md) 을 구현하고, representative pool 에서 {none / Y-only FootLock / coord u_grid / oracle-u} 4-arm 비교로 tool 효과를 측정한다. 성능 증명이 아니라 **"재설계된 메커니즘이 실제로 문제 축(접지 중 수평 미끄러짐)을 수정하는가 + guard 는 무엇을 잃는가"** 의 정직한 첫 snapshot.

## Why This Exists

AR-058-4(P4): Y-only FootLock 은 artifact↓ vs foot_skate↑ mixed (축 mismatch). AR-060 이 coordinate-aware 재설계를 frozen. AR-062 audit 이 "foot-skate 만 (건전 Cat-B metric) + (aligned mechanism) + (P4 필요성) 3박자 성립" 으로 본 구현을 승인 — 단 **gate-fire 지표 인용 금지** (guard = foot_skate_world/float_mag/raw BoneCV/FID).

## What Was Done

1. **구현**: [correction_tools/coordinate_footskate_cleanup_tool.py](../../../correction_tools/coordinate_footskate_cleanup_tool.py) — v2 contact/skate(0.05/0.035/0.025) segment 탐지(gap≤1 병합) → validity filter(`skate_frac≥0.2`; 깨끗한 접지 no-op, ambiguous report-only) → median X/Z anchor + ground Y → `u·smoothstep blend` → leg-chain propagation 1.0/0.5/0.2/0.1 (NOT IK). **trajectory(world) 전제** (`coord_space≠trajectory` → ValueError).
2. **Registry**: export 만 — `DEFAULT_CORRECTION_TOOLS` **의도적 미포함** (candidate-set freeze, §3-25); rule_based lookup 은 inert 항목 (decision-invariant).
3. **센서 (§4)**: unit 10/10 신규 + 전체 83/83 + integration smoke (registry 무변화 → 기존 pipeline 결정 불변).
4. **평가**: [tools/coordinate_footskate_effect_ar061.py](../../../tools/coordinate_footskate_effect_ar061.py) — representative-300, prompt 단위(seed 평균) n=300/gen, bootstrap CI + Wilcoxon.
5. **시각화**: before/after GIF 3쌍 (Y-up §3-19 첫 frame 검사 통과).

## Results (snapshot 1)

| foot_skate_world Δ | Y-only large | coord u=1.0 | oracle-u(+STOP) |
|---|---|---|---|
| **mdm** | +0.01452 (fi 0.7%) | **−0.00129 [−0.0015,−0.0011], fi 58.3%, p<1e-6 (≈−12%)** | −0.00130 |
| motiongpt | +0.00467 | +0.00011 | −0.00020 |
| momask | +0.00458 | +0.00030 | −0.00011 |

- **MDM**: u 단조 개선 + float_mag 동반 개선(−0.0079) — Y-only 의 float↓↔skate↑ trade 소멸.
- **VQ 2종**: 미세 악화, oracle-grid≈0, **STOP 포함 시에야 ≈0** → 올바른 action = STOP. 같은 tool 이 generator 별 개선/악화 = **조건부 routing 직접 증거** (P4 보완).
- ⚠️ **guard**: bone_cv_max Δ +0.045~+0.053 @u=1.0 (u 비례; u=0.25 는 +0.002) — 선형 propagation(NOT IK) 한계 실증 → **AR-064 신설**.
- artifact_total(C proxy) 소폭 ↑ — proxy 에 skate 항목 부재 (tool 성공 기준 부적합, §3-20 재확인).

## Claim Boundary

허용: "coordinate-aware cleanup 이 MDM 의 foot_skate 를 CI-clean 개선하며, VQ 에는 STOP 이 옳다 — 조건부 선택 필요성의 증거."
금지: snapshot 1 로 가설 status 전환 (§3-9 snapshot≥2) / FID(A) 미측정 상태의 외부 최종성능 claim / learned optimum 주장 (u_grid = fixed sweep).

## Evidence

- Commit: 196e360 · [snapshot](../../../evals/snapshots/coordinate_footskate_effect_ar061_v1.json) · [report](../../../evals/reports/2026-07-07_coordinate_footskate_effect_ar061.md) · [visual](../../../reports/figures/2026-07-07/ar061_coordinate_cleanup_before_after/index.md) · [일지](../../../reports/2026-07-07.md)

## Follow-ups

- AR-064: bone-preserving propagation (BoneCV guard 후속).
- registry default 편입 + routing 연결 = tool 효과 재현(독립 pool) 후 별도 결정.
