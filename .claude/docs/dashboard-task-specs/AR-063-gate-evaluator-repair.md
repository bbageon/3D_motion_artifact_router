# AR-063 - Gate Evaluator Repair (PhysicalGateV0 v0.2.0)

Status: **completed** (2026-07-07, commit 5afbae9)  
Epic: Tool  
Priority: Urgent  
Parent: AR-062 (artifact↔tool alignment audit)

## Goal

[AR-062 audit](../findings/artifact_tool_alignment_audit.md) 이 재현으로 확정한 gate evaluator 의 구조적 결함 (A-1 Skate vacuity / A-2 Penetrate vacuity / A-4 local 좌표 contact 실패 / A-6 floating 정의 난립) 을 수리하고, clean 재calibration + **첫 유효 유병률** 을 산출한다. routing/RL 의 state 가 이 evaluator 출력이므로, 방치 시 잘못된 state 로 학습하게 되는 것을 차단.

## What Was Done (정의 변경 — 구 기록과 비호환)

| 항목 | v0.1.0 (구) | v0.2.0 (신) |
|---|---|---|
| contact 추정 | 수평(xz)속도 ≤ 0.02 & height ≤ 0.10 | **수직(y)속도 ≤ 0.035** & height ≤ (Skate 0.05 / Float 0.10) — v2 계열 |
| ground 추정 | min-Y (전 joint) | **feet lower-height 10th percentile** ([base.py::estimate_ground_y](../../../evaluators/base.py) 공용 신설) |
| skate 임계 | 0.05 m/frame | **0.025 m/frame** (v2) |
| SEVERITY_VERSION | physical_gate 0.1.0 / foot_floating 1.2.0 | **0.2.0-2026-07-07 / 2.0.0-2026-07-07** |

- 센서 (§4 evaluator 변경 의무): unit **90/90** (수리 검증 신규 7건 — AR-062 vacuity case 가 이제 fire / clean no-fire / A-4 regression) + integration smoke (수리 후 loop 가 floating 실제 검출·보정 0.066→0.0; 수리 전은 즉시 rollback 0회).
- **문서 동시 갱신**: metric_provenance §3-5-1~3 · reproducibility-checklist §3 · AR-043/044 spec retroactive caveat 헤더 · audit doc A-1~A-6 상태 표기.
- fixture 교훈: 양발 **전-구간 균일 lift** 주입은 feet-percentile ground 와 "지면 상승"이 구분 불가 (self-referential estimator 내재 한계 — base.py 문서화) → 관련 unit fixture 를 부분 구간 주입으로 교정.

## Results

**Clean calibration v2** (HumanML3D n=493, world — [snapshot](../../../evals/snapshots/physical_gate_clean_calibration_v2.json)): Skate p99 **0.346** / Penetrate p99 **0.040** (구: 둘 다 0.0 = vacuous) / JerkSpike 0.1056 (정의 무변경 — 구 값 일치, 일관성 ✓) / BoneCV 1e-5 (**무판별 재확인** — 분포 특성, 재보정으로 해소 불가 → raw score 원칙 고정).

**유병률 재측정** (representative-300, trajectory primary — [snapshot](../../../evals/snapshots/gate_prevalence_remeasure_ar063_v1.json)):

| generator | Penetrate occur / fire | Float occur / fire | **Skate occur / score / fire** |
|---|---|---|---|
| motiongpt | 1.2% / 0.7% | 67.7% / 0.3% | 73.8% / 0.083 / **1.7%** |
| **mdm** | 1.8% / 0.7% | 61.4% / 1.1% | 71.0% / **0.185** / **23.1%** |
| momask | 0.6% / 0.2% | 66.6% / 0.3% | 68.0% / 0.077 / **1.3%** |

1. **Skate gate 첫 판별력**: MDM fire 23.1% vs VQ 1.3~1.7% (>10배) — Category-B foot_skate_world 서열·AR-061 과 **수렴** → AR-044 headline 강화.
2. **Penetration 실측 희소** (occur ≤1.8%) → 전용 penetration tool **저우선 확정** (유병률 근거 후 설계 원칙 이행).
3. **Float gate-fire ≤1.1%** (clean p99 0.70 도달 불가) → floating 판단은 raw score 로.
4. **좌표 효과 실증 (A-4)**: Skate fire local 21.3% vs trajectory 1.7% (motiongpt, 12배 과검출) → **foot-계 gate 는 trajectory 입력 의무**.

## Claim Boundary

허용: "gate evaluator 의 구조적 vacuity 를 수리했고, 수리된 측정이 독립 Category-B metric 과 수렴한다."
금지: v0.1.0 기록의 Skate/Penetrate 0% 를 "없음"의 증거로 인용 (retroactive caveat) / 유병률을 도구 효과로 해석 / gate proxy(Category C) 를 외부 최종 성능 근거로 사용 (§3-20).

## Evidence

- Commit: 5afbae9 · [report](../../../evals/reports/2026-07-07_gate_evaluator_repair_ar063.md) · [remeasure](../../../evals/snapshots/gate_prevalence_remeasure_ar063_v1.json) · [calibration v2](../../../evals/snapshots/physical_gate_clean_calibration_v2.json) · [일지](../../../reports/2026-07-07.md)

## Follow-ups

- **AR-065**: 기존 learned ranker (AR-040/052 계열) 는 v0.1.0 무신호 feature (Skate/Penetrate ≡ 0) 로 학습됨 → 수리된 state 로 재구축 검토. 기존 결과 해석 시 "skate/penetrate state 무신호" caveat 동반.
- `float_mag`(physical_metric, 구 정의) 는 historical 비교용 미변경 (A-6 부분 수리) — 신규 인용 시 v2 계열 우선.
