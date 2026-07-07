# AR-061 — CoordinateFootSkateCleanupTool representative-pool 비교 (snapshot 1)

- Raw: [evals/snapshots/coordinate_footskate_effect_ar061_v1.json](../snapshots/coordinate_footskate_effect_ar061_v1.json)
- Harness: [tools/coordinate_footskate_effect_ar061.py](../../tools/coordinate_footskate_effect_ar061.py) · Tool: [correction_tools/coordinate_footskate_cleanup_tool.py](../../correction_tools/coordinate_footskate_cleanup_tool.py) ([frozen spec](../../.claude/docs/design/coordinate_footskate_cleanup_tool_frozen_spec.md))
- Split: `protocol_rep_pool_seed20260608` (representative-300, 3 seeds/prompt, 3 generators) — **real-distribution** tier (§3-17)
- 통계 단위: prompt (seed 평균), n=300/generator, bootstrap CI 95% + paired Wilcoxon (per-prompt Δ vs 0)
- **claim boundary**: snapshot 1 (snapshot≥2 미충족 — §3-9 가설 status 근거 금지). FID(Category A) 미측정 (별도 층). AR-062 audit 준수 — gate-fire 지표 불인용, guard = foot_skate_world/float_mag/raw BoneCV.

## 1. Primary — foot_skate_world Δ (Category B, lower=better)

| generator | Y-only large (P4 재현) | coord u=0.25 | u=0.5 | u=0.75 | u=1.0 | oracle-u (grid) | oracle-u+STOP |
|---|---|---|---|---|---|---|---|
| **mdm** | **+0.01452** [+0.0129,+0.0161] fi=0.7% | −0.00016 | −0.00044 | −0.00084 | **−0.00129** [−0.0015,−0.0011] fi=58.3% p<1e-6 | −0.00130 | −0.00130 |
| motiongpt | +0.00467 [+0.0039,+0.0054] | +0.00018 | +0.00025 | +0.00022 | +0.00011 [−0.0000,+0.0002] p=0.0013 | −0.00004 [−0.0001,+0.0000] | −0.00020 |
| momask | +0.00458 [+0.0037,+0.0054] | +0.00021 | +0.00033 | +0.00035 | +0.00030 [+0.0002,+0.0004] | +0.00010 | −0.00011 |

(fi = frac_improved. AR-044 baseline fs: mdm 0.0108 / motiongpt 0.0056 / momask 0.0047 → mdm u=1.0 상대 감소 ≈ **−12%**.)

## 2. Guard metrics Δ

| generator·arm | float_mag (B) | artifact_total (C proxy) | **bone_cv_max (raw)** | corr. mag (m) | apply |
|---|---|---|---|---|---|
| mdm coord u=1.0 | **−0.00794** (개선) | +0.00569 | **+0.04727** ⚠️ | 0.059 | 0.36ms |
| mdm coord u=0.25 | −0.00088 | +0.00062 | +0.00216 | 0.015 | 0.39ms |
| mdm Y-only large | −0.01878 | −0.00814 | +0.00851 | 0.054 | 0.08ms |
| motiongpt coord u=1.0 | +0.00018 | +0.00371 | +0.04483 ⚠️ | 0.042 | 0.24ms |
| momask coord u=1.0 | −0.00040 | +0.00309 | +0.05275 ⚠️ | 0.044 | 0.25ms |

## 3. 해석

1. **MDM (diffusion): coordinate tool 이 Y-only 실패를 뒤집음.** foot_skate CI-clean 감소 + **u 단조** + float_mag 동반 개선 (Y-only 의 float↓↔skate↑ trade 없음). Y-only 는 동일 pool 에서 +0.0145 로 심각 악화 (fi 0.7% — 사실상 전 prompt 악화). 메커니즘 재설계(수평 anchor)가 정확히 문제 축을 수정했음을 뒷받침.
2. **VQ (MotionGPT/MoMask): 미세 악화, oracle+STOP ≈ 0.** 물리적으로 이미 깨끗한 generator 에는 고칠 skate 가 없어 (AR-044·F7) anchor/blend 경계 변위만 남음. per-prompt 최적 u 를 강제로 골라도 (oracle-grid) ≈0, **STOP 허용 시에야 ≈0** — VQ 의 올바른 action 은 STOP.
3. **→ 조건부 routing 증거 (P4 보완)**: 같은 fixed tool 이 MDM 개선·VQ 악화 — tool 을 고정 적용하는 어떤 protocol 도 전 generator 에서 안전하지 않음. "state/generator 조건부 (tool, u, STOP) 선택" 필요성의 직접 증거. (H-2026-206 generator-agnostic 관련 — 단 snapshot 1 이므로 가설 인용은 informational.)
4. **⚠️ BoneCV 상승 (u 비례)**: u=1.0 에서 Δmax-CV +0.045~0.053 — leg-chain 선형 propagation(1.0/0.5/0.2/0.1, **NOT IK**) 이 bone 길이를 왜곡. frozen spec §3-4 가 예고한 engineering-heuristic 한계 실증. u=0.25 에선 +0.002 로 미미 → **강도 선택이 fs↓ vs boneCV↑ trade 를 조절** (routing 의 u 축 근거). 후속: bone-preserving propagation (IK 또는 사후 bone re-projection) = AR-064.
5. artifact_total(C proxy) 은 coord arm 에서 소폭 ↑ — Layer-A 합산엔 foot-skate 항목이 없고 (AR-062 A-1: gate skate 는 vacuous) 임의 pose 변경을 벌점화하기 때문. proxy 를 tool 성공 기준으로 쓰면 안 됨 (§3-20 Category C).

## 4. Per-sample smoke (AR-058-3f high-v2 9건, 참고)

u=1.0 fixed: 6/9 개선 (−5~−22%), 3/9 악화 (+1~+22%) — per-sample 수준에서도 mixed → per-sample state 조건화 여지. 시각 before/after: [reports/figures/2026-07-07/ar061_coordinate_cleanup_before_after/](../../reports/figures/2026-07-07/ar061_coordinate_cleanup_before_after/index.md) (3건, Y-up §3-19 첫 frame 검사 통과).

## 5. 남는 불확실성 (§3-22 4항목 중)

- **snapshot 1** — 독립 pool 재현 전 (AR-024 계열). 가설 status 전환 근거 아님.
- **FID(A) 미측정** — 외부 최종 성능 claim 불가 (P4-FID 방식 별도 층 필요).
- **perceptual 미검증** — AR-058-3f 인간 응답 대기; foot_skate_world 감소가 지각 품질 향상인지는 quality-validated tier 필요.
- propagation weight·blend·v2 임계 = engineering heuristic (frozen spec §8).
- oracle-grid 는 **single-step, u_grid 내 후향 선택** (§3-16) — learned policy 성능 아님.
