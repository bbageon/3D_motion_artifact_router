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

## 2. Cat-A no-harm (locomotion 한정, prompt-bootstrap, FID) — 핵심 표

root-aware correction 을 각 generator 에 적용 후 표준 text-motion metric 변화:

| generator | R@1 Δ [CI] | MM-Dist Δ [CI] | FID Δ [CI] | 판정 |
|---|---|---|---|---|
| **MDM** | **+0.048** [+0.009,+0.089] ↑ | **−1.01** [−1.22,−0.83] ↓ | **−7.2** [−9.3,−5.4] ↓ | **개선** (apply 옳음) |
| MotionGPT | −0.029 [−0.066,+0.003] | **+0.12** [+0.055,+0.178] ↑ | **+0.44** [+0.24,+0.70] ↑ | **harm** (2/3 유의 악화) |
| MoMask | **−0.041** [−0.089,−0.008] ↓ | **+0.14** [+0.051,+0.243] ↑ | +0.23 [−0.07,+0.54] | **harm** (2/3 유의 악화) |

(↑ R@1 = 좋음, ↓ MM/FID = 좋음. Δ = corrected − original.)

- **MDM: 세 지표 모두 유의 개선** — correction 이 semantic·naturalness 를 회복 (AR-072 보강, 이번엔 loco 한정 prompt-bootstrap 로 더 엄격).
- **VQ 둘: 각각 2/3 지표 유의 악화** — deficit 없는 root 를 건드려 정렬 악화. **near-no-op 아님**: root 는 실제 이동(induced_disp MGPT 0.37 / MoMask 0.41 m), foot_skate 이득 없이(≈0) semantic 만 악화.

## 3. GT-free STOP 분리 (피드백 5) — GT 없이 apply/STOP 갈림

추론 시 GT root ratio 는 관측 불가. **관측 가능 신호(foot_skate)** 만으로 분리되나:

| generator | foot_skate [CI] | apply-gate 통과율 (skate>MDM 25th pct) | induced_disp | path_gain |
|---|---|---|---|---|
| MDM | 0.0125 [0.0116,0.0134] | **74.5%** (apply) | 1.20 | 2.64 |
| MotionGPT | 0.0065 | 21.2% (STOP) | 0.37 | 1.02 |
| MoMask | 0.0053 | 10.3% (STOP) | 0.41 | 0.86 |

**skate AUC(MDM vs VQ) = 0.834 / 0.879** — GT 없이 skate 신호 하나로 MDM(apply)/VQ(STOP) 강 분리. induced_disp·path_gain 도 분리(MDM 1.20 vs VQ 0.4; 2.64 vs ~1.0). ⟹ AR-074 GT-free STOP rule 이 VQ 를 자동 STOP.

## 4. 무조건-적용 보조분석 (superseded, 참고)

전체 300 prompt(non-loco 포함)에 무조건 적용 시 VQ Cat-A 악화 — 단 이는 "VQ라서"와 "non-loco 오적용(AR-074: 0.85 m push)" 혼재. **generator 비교 근거 아님**, 무조건-적용 harm 의 보조 증거일 뿐. [motiongpt](../snapshots/vq_root_catA_ar077_motiongpt_v1.json)/[momask](../snapshots/vq_root_catA_ar077_momask_v1.json) (record_type=FULL300_auxiliary).

## 5. Routing decision + 방어 가능한 결론

| generator | deficit | correction Cat-A | GT-free skate | **decision** |
|---|---|---|---|---|
| MDM | 큼 (0.42) | 개선 | 높음(74.5% apply) | **APPLY** |
| MotionGPT | 없음(0.99) | 악화 | 낮음(21% ) | **STOP** |
| MoMask | 없음(1.00) | 악화 | 낮음(10%) | **STOP** |

> **MDM locomotion 에서는 root progression deficit 과 root-aware correction 의 이득(R@1↑·MM↓·FID↓)이 관측된 반면, MotionGPT·MoMask 에서는 평균 root deficit 및 foot-skate 개선이 관측되지 않았고, 같은 보정을 적용하면 표준 text-motion metric 이 악화됐다. 따라서 root correction 은 generator 에 고정 적용할 수 없으며, 관측 가능한 motion state(foot-skate, AUC 0.83~0.88)에 기반한 APPLY/STOP 결정이 필요하다.**

이 주장 = **결정 계층(apply/STOP)의 필요성**까지 (학습 RL routing 의 우월성 아님).

## Claim Boundary (피드백 준수)

- **VQ 구조가 원인이라는 인과 금지** — VQ(discrete token)가 문헌과 정합하나(MotionGPT NeurIPS2023, MoMask CVPR2024), 본 결과는 관측이지 인과 증명 아님.
- **"VQ = GT 와 동일" 금지 → "평균 root deficit 증거 없음"** (CI 가 1.0 포함이지 동일 증명 아님).
- VQ Cat-A harm 은 표준 metric 악화 — **지각 harm 단정은 A/B 필요** (시나리오 "깔끔"이라 사전등록상 VQ A/B 는 선택).
- MDM·단일 벤치마크(HumanML3D)·locomotion 한정. VQ 개별 sample 은 deficit 있을 수 있음(평균 진술).

## 남은 항목 (마감 전)

- (선택) VQ 소규모 A/B — 시나리오 "깔끔"이라 사전등록상 미필수.
- AR-075 에서 GT-free STOP gate 를 orchestrator 에 구현·편입.
