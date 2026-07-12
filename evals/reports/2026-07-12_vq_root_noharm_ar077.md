# AR-077 — VQ Root State + No-Harm (generator-conditioned routing 근거)

> Status: **in-progress** (사용자 판단 — remediation 후 확정). 외부 피드백(2026-07-12) 6개 지적 전부 반영.

- Snapshots: [diagnostic](../snapshots/vq_root_noharm_ar077_v1.json) · Cat-A v2 [mdm](../snapshots/vq_root_catA_v2_ar077_mdm_v1.json)/[motiongpt](../snapshots/vq_root_catA_v2_ar077_motiongpt_v1.json)/[momask](../snapshots/vq_root_catA_v2_ar077_momask_v1.json) · [GT-free 분리](../snapshots/gtfree_stop_separation_ar077_v1.json)
- 표본: **locomotion 165 prompt** (GT root speed>0.01, 3 generator 동일 필터) × 3 seed. 통계 단위 = **prompt** (seed 평균), prompt-bootstrap B=300.

## 1. Root deficit — MDM 만 있음 (VQ 없음)

| generator | root ratio (v_root/GT) [CI] | 해석 |
|---|---|---|
| **MDM** | **0.419** [0.392, 0.451] | 뚜렷한 deficit — AR-076: root progression collapse |
| MotionGPT | **0.985** [0.918, 1.060] | 평균 deficit 증거 없음 (CI 가 1.0 포함) |
| MoMask | **1.001** [0.928, 1.083] | 평균 deficit 증거 없음 |

(사전등록 3-시나리오 중 **"깔끔"** — VQ 는 root deficit 없음. preflight n=4 의 ratio 2.13 은 소표본 노이즈였음, full n=165 로 정정.)

## 2. Cat-A no-harm (locomotion 165, prompt-bootstrap **B=1000**, FID) — 핵심 표 (v3, 통계 수정)

> v3 = 피드백 2차 통계 수정본 ([mdm](../snapshots/vq_root_catA_v3_ar077_mdm_v1.json)/[motiongpt](../snapshots/vq_root_catA_v3_ar077_motiongpt_v1.json)/[momask](../snapshots/vq_root_catA_v3_ar077_momask_v1.json)): orig/corr **동일 permutation(paired)** + **seed 미평균**(FID 분포 보존) + **엄격 3-seed equal-N**(P=165, skip 0) + **bootstrap 1000**. v2(버그본)는 superseded — 방향 불변, 수치 정합.

root-aware correction 을 각 generator 에 적용 후 표준 text-motion metric 변화 (Δ = corrected − original).
**R@1 = R-Precision 수정본**([catA_rprec_fix](../snapshots/catA_rprec_fix_ar077_mdm_v1.json), 피드백 4: one-seed-per-prompt — 같은 caption seed 오답처리 버그 제거), MM-Dist/FID = v3:

| generator | R@1 Δ [CI] (수정본) | MM-Dist Δ [CI] | FID Δ [CI] | 판정 |
|---|---|---|---|---|
| **MDM** | **+0.047** [+0.019,+0.068] ↑ | **−1.00** [−1.20,−0.81] ↓ | **−7.3** [−9.3,−5.5] ↓ | **개선** (3/3 유의) |
| MotionGPT | **−0.034** [−0.043,−0.001] ↓ | **+0.087** [+0.034,+0.140] ↑ | +0.20 [−0.05,+0.43] | **harm** (R@1·MM 유의) |
| MoMask | **−0.035** [−0.062,−0.014] ↓ | **+0.150** [+0.064,+0.250] ↑ | +0.21 [−0.12,+0.58] | **harm** (R@1·MM 유의) |

(↑ R@1 = 좋음, ↓ MM/FID = 좋음. R-Prec 수정 전후 방향·유의성 동일 — MDM 개선/VQ harm 확정.)

- **MDM: 세 지표 모두 유의 개선** — correction 이 semantic·naturalness 회복 (AR-072 보강, loco 한정 정합 통계).
- **VQ 둘: R@1·MM-Dist 유의 악화** (FID 는 CI 가 0 포함 — 비유의). deficit 없는 root 를 건드려 정렬 악화. **near-no-op 아님**: root 실제 이동(induced 0.37/0.41 m), foot_skate 이득 없이 semantic 악화 (분포 수준 harm).

## 3. Generator 분리 진단 (피드백 정정 — "GT-free routing 실증" 아님)

> ⚠️ **표현 정정 (2026-07-12 2차 피드백)**: 아래는 "완전한 GT-free routing gate 실증" 이 **아니다**. (a) 평가 대상(locomotion 165)을 **GT root speed 로 선정** — gate 입력(foot_skate)은 GT-free 이나 subset 정의는 GT 사용. (b) apply-gate 통과율 74.5% 는 **임계값을 MDM 25th pct 로 정한 순환** — 성능 증거 아님. (c) AUC 의 label 은 "MDM 출력인가" = **generator 분류**이지 "correction 이 이득인가" 아님. 정확한 진술: **"GT 로 정의한 locomotion subset 안에서, 추론 가능한 foot_skate 가 generator 를 구분했다(AUC 0.83~0.88)."** 실배포 GT-free gate + benefit 예측 = §6 (held-out).

| generator | foot_skate [CI] | (임계=MDM 25th pct) 통과율 | induced_disp | path_gain |
|---|---|---|---|---|
| MDM | 0.0125 [0.0116,0.0134] | 74.5% (순환 — 참고 안 함) | 1.20 | 2.64 |
| MotionGPT | 0.0065 | 21.2% | 0.37 | 1.02 |
| MoMask | 0.0053 | 10.3% | 0.41 | 0.86 |

**skate AUC(MDM vs VQ) = 0.834 / 0.879** (label="MDM인가"). 의미: foot_skate 분포가 MDM/VQ 를 잘 구분. **STOP 규칙 아님** — benefit 예측은 §6.

## 4. 무조건-적용 보조분석 (superseded, 참고)

전체 300 prompt(non-loco 포함)에 무조건 적용 시 VQ Cat-A 악화 — 단 이는 "VQ라서"와 "non-loco 오적용(AR-074: 0.85 m push)" 혼재. **generator 비교 근거 아님**, 무조건-적용 harm 의 보조 증거일 뿐. [motiongpt](../snapshots/vq_root_catA_ar077_motiongpt_v1.json)/[momask](../snapshots/vq_root_catA_ar077_momask_v1.json) (record_type=FULL300_auxiliary).

## 5. Routing decision + 방어 가능한 결론

| generator | deficit | correction Cat-A | skate (apply 후보) | **경향** |
|---|---|---|---|---|
| MDM | 큼 (0.42) | 개선 | 높음 | **APPLY 후보 많음** |
| MotionGPT | 없음(0.99) | 악화 | 낮음 (통과 21%) | **STOP 후보 많음** |
| MoMask | 없음(1.00) | 악화 | 낮음 (통과 10%) | **STOP 후보 많음** |

> **MDM 과 VQ generator 는 root-aware correction 의 필요성이 서로 다르며(MDM: deficit+이득 R@1↑·MM↓·FID↓ / VQ: deficit 증거 없음 + Cat-A 악화), 관측 가능한 foot-skate 신호가 그 차이를 포착한다(AUC 0.83~0.88). 다만 현재 신호는 generator 분리 진단이며, held-out motion 에서 correction benefit 을 직접 예측하는 routing gate 로서의 성능은 §6 (held-out)에서 별도 검증한다.**

VQ 통과율이 0%가 아니라 10~21% 이므로 **"VQ=STOP"이 아니라 "STOP 후보 다수"** — 최종 결정은 generator 이름보다 **motion state** 로 (state-conditioned routing 을 오히려 더 지지).

## 6. Held-out benefit prediction — 진짜 routing gate 검증 (피드백 3, 핵심)

이전 AUC(0.83~0.88)는 label="MDM인가"(**generator 분류**)였다. 진짜 routing gate 는
"이 motion 에 correction 적용 시 이득인가"를 예측해야 한다. label 을 **per-motion
ΔMM-Dist < −δ** (δ=VQ noise floor 0.116 — 부호만 아님, 2차 피드백 반영)로 두고,
**전체 pool(3 gen, GT 필터 없음)** + **sample_id 단독 split**(cross-gen leakage 차단) +
**multiplicity 보존 bootstrap** 으로 재측정 ([snapshot](../snapshots/routing_benefit_gate_ar077_v1.json), n=2699).

**δ-sensitivity (benefit/neutral/harm, %)** — 피드백 1 반영:

| δ | MDM | MotionGPT | MoMask |
|---|---|---|---|
| 0 (부호만, 과장) | 65 / 0 / 35 | 48 / 0 / 52 | 47 / 0 / 53 |
| **0.116 (noise floor)** | **57 / 16 / 28** | **22 / 50 / 29** | **21 / 50 / 29** |

⟹ **"VQ 47% benefit"은 부호만의 착시.** 실질(δ)로 보면 **VQ 는 절반이 neutral**(correction 무효과) + benefit 21% ≈ harm 29% = **평균 net-무효~약간 harm**. MDM 은 benefit 57% > harm 28% = **net 이득**.

**held-out routing gate (benefit label δ=0.097 = calibration VQ median, 누수 차단, n=1350):**

| | 값 |
|---|---|
| **benefit AUC** | **0.674** [0.631, 0.717] (proper bootstrap) — moderate (부호만 0.55 보다 높음). δ 누수 수정 전(0.684)과 사실상 동일 → 누수가 부풀린 것 아님 |
| precision / recall | 0.504 / 0.499 |
| **false-apply / false-stop** | **0.496** / 0.274 |

(benefit@δ: MDM 0.581 / MotionGPT 0.235 / MoMask 0.226 — δ=calibration VQ median.)

**결론 (2차 피드백 후 — δ 조건부·exploratory 명시):**
- ⚠️ **δ 조건부 caveat (3차 피드백)**: δ=0.097 은 **calibration VQ |ΔMM| median** (holdout 미사용 — 누수 차단). 단 VQ median 기반이라 **VQ neutral 비율이 구조적으로 ~50% 근처**. 따라서 "VQ 절반이 실제 무효" / "MDM 58%·VQ 23% benefit" / "AUC 0.67" 은 **확정 사실 아님 — δ 조건부 exploratory**. 정확: **"내부 δ 조건에서 MDM ~58%·VQ ~23% benefit 분류, pooled AUC ~0.67 [0.63,0.72]"**.
- **generator 수준**: (δ 조건부) 실질 benefit MDM ~58% vs VQ ~23% — 차이 뚜렷.
- **motion 수준**: foot_skate benefit-AUC ~0.67 (δ 조건부) — 부호만 0.55 보다 높으나 exploratory. δ 누수 수정 전후 동일(0.68→0.67).
- **실용 gate**: **false-apply ~50%** = APPLY 선택 표본의 절반이 **최소 benefit(δ) 기준 미충족** (neutral+harm 포함 — "절반이 품질 악화" 아님). no-harm gate 로 불충분.

⟹ **"routing 이 작동한다" 아님 → "richer-state routing 의 필요성은 생겼으나 성능은 미검증"**. 단일 skate gate 는 δ 조건부 moderate 신호이나 no-harm 용으로 불충분.

## Claim Boundary (피드백 준수)

- **VQ 구조가 원인이라는 인과 금지** — VQ(discrete token)가 문헌과 정합하나(MotionGPT NeurIPS2023, MoMask CVPR2024), 본 결과는 관측이지 인과 증명 아님.
- **"VQ = GT 와 동일" 금지 → "평균 root deficit 증거 없음"** (CI 가 1.0 포함이지 동일 증명 아님).
- VQ Cat-A harm 은 표준 metric 악화 — **지각 harm 단정은 A/B 필요** (시나리오 "깔끔"이라 사전등록상 VQ A/B 는 선택).
- MDM·단일 벤치마크(HumanML3D)·locomotion 한정. VQ 개별 sample 은 deficit 있을 수 있음(평균 진술).

## 7. 최종 방어 가능한 결론 (2차 피드백 후 — 표현 하향)

> **Root-aware correction 은 MDM 에서 분포 수준의 품질을 개선하지만(R@1↑·MM↓·FID↓, Cat-A v3), 개별 motion 의 적용 이득은 foot-skate 단일 feature 로 신뢰성 있게 예측할 수 없다(holdout AUC 0.55, false-apply 45%). 따라서 generator-level rule 은 유효한 baseline 이지만, per-motion no-harm routing 을 위해서는 richer pre-action state 가 필요하다.**

| 층위 | 결론 (δ 조건부 exploratory) | 판정 |
|---|---|---|
| Generator 평균 | benefit@δ MDM 58% vs VQ 23% | **명확** |
| 개별 motion | foot-skate 로 benefit 예측 AUC 0.67 [0.63,0.72] | **중간 (useless 아님)** |
| 실용 gate | false-apply 50% | **단일-feature 로 불충분** |
| Routing 필요성 (richer state 동기) | 성립 | ✅ |
| Learned routing 성능 (benefit 잘 예측) | **미성립** | ❌ |

**표현 규율**: richer state 가 false-apply 를 낮추기 전까지 **"routing 이 작동한다" 금지 → "moderate 신호 있으나 단일-feature gate 로 문제 남음"** (현 state 부족의 정확한 진단이지 연구 실패 아님).

## 남은 항목 (마감 전 — AR-077 in-progress 정당)

- **§6-1 재측정** (2차 피드백 4수정): benefit δ-3분류 / sample_id 단독 split / proper bootstrap CI / (Cat-A) R-Precision one-seed-per-prompt.
- **의사결정**: AR-075 에 **foot-skate 단일 gate 통합 금지**. 먼저 **AR-065 Effect-aware state 재구축** 검토 → generator-only / skate-only / richer-state gate 를 held-out 에서 benefit-AUC·false-apply 로 비교 (AR-078).
- (선택) VQ 소규모 A/B — 시나리오 "깔끔"이라 사전등록상 미필수.
