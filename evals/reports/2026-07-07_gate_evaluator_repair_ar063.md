# AR-063 — Gate evaluator 수리 (v0.2.0) + clean 재calibration + 유병률 재측정

- Raw: [gate_prevalence_remeasure_ar063_v1.json](../snapshots/gate_prevalence_remeasure_ar063_v1.json) · [physical_gate_clean_calibration_v2.json](../snapshots/physical_gate_clean_calibration_v2.json)
- 수리 대상: [evaluators/physical_gate.py](../../evaluators/physical_gate.py) (SEVERITY 0.1.0→**0.2.0**) · [evaluators/foot_floating_evaluator.py](../../evaluators/foot_floating_evaluator.py) (1.2.0→**2.0.0**) · 공용 [evaluators/base.py::estimate_ground_y](../../evaluators/base.py)
- 근거: [AR-062 audit A-1/A-2/A-4/A-6](../../.claude/docs/findings/artifact_tool_alignment_audit.md) (구조적 vacuity 재현)
- Evidence tier: clean calibration = HumanML3D GT (world) / 유병률 = representative-300 **real-distribution** (§3-17). 유병률은 기술 통계 (도구 효과 아님).

## 1. 수리 내용 (정의 변경 — 구 기록과 비호환, severity_version 구분 의무)

| 항목 | v0.1.0 (구) | v0.2.0 (신) |
|---|---|---|
| contact 추정 | 수평(xz)속도 ≤ 0.02 & height ≤ 0.10 | **수직(y)속도 ≤ 0.035** & height ≤ (Skate 0.05 / Float 0.10) — v2 계열 (AR-058-3b) |
| ground 추정 | min-Y (전 joint) | **feet lower-height 10th percentile** (coords_protocol 일치) |
| skate 임계 | 0.05 m/frame | **0.025 m/frame** (v2) |
| FootFloating (Layer A) | 수평속도 contact (v1.2.0) | 수직속도 contact (**v2.0.0**) |

센서 (§4 evaluator 변경 의무): py_compile ✓ · **unit 90/90** (수리 검증 신규 7건 포함: vacuity case fire 확인, clean no-fire, A-4 regression) ✓ · integration smoke ✓ — 수리 후 loop 가 floating 을 실제 검출·보정 (score 0.066→0.0, 4회 적용; 수리 전 동일 smoke 는 즉시 rollback·0회 적용). fixture 교정: 양발 **전-구간 균일 lift** 주입은 feet-percentile ground 와 지면 상승이 구분 불가(내재 한계, 문서화) → 부분 구간 주입으로 변경.

## 2. Clean calibration v2 (HumanML3D n=493, world)

| evaluator | p50 | p95 | **p99 (gate)** | 비고 |
|---|---|---|---|---|
| Penetrate | 0 | 0 | **0.0404** | 구 0.0 (vacuous) → 유의미 |
| Float | 0.078 | 0.375 | **0.7007** | clean 자체 float-band 비율 높음 → gate-fire 판별력 약함 (raw score 원칙) |
| Skate | 0.050 | 0.200 | **0.3455** | 구 0.0 (vacuous) → 유의미. clean 보행도 근접지 수평이동 일부 존재 |
| JerkSpike | 0.017 | 0.061 | 0.1056 | 정의 무변경 — 구 값과 동일 (일관성 확인 ✓) |
| BoneLengthCV | 0 | 0 | 1e-05 | 정의 무변경 — **무판별 재확인** (분포 특성, 재보정으로 해소 불가; raw 사용 원칙) |

## 3. 유병률 재측정 (representative-300, prompt 단위 n=300/gen)

**Trajectory (primary — AR-048/A-4):**

| generator | Penetrate occur / fire | Float occur / fire | **Skate occur / score / fire(>p99)** |
|---|---|---|---|
| motiongpt | 1.2% / 0.7% | 67.7% / 0.3% | 73.8% / 0.083 / **1.7%** |
| **mdm** | 1.8% / 0.7% | 61.4% / 1.1% | 71.0% / **0.185** / **23.1%** |
| momask | 0.6% / 0.2% | 66.6% / 0.3% | 68.0% / 0.077 / **1.3%** |

해석:
1. **Skate gate 가 처음으로 판별력을 가짐** — MDM fire 23.1% vs VQ 1.3~1.7% (>10배 분리). Category-B `foot_skate_world` 서열 (MDM 2x) 및 AR-061 (coord tool 이 MDM 만 개선) 과 **수렴** — 독립 정의 두 축이 같은 그림. AR-044 headline 은 약화가 아니라 **강화**.
2. **Penetration 은 실측 희소** (occur ≤1.8%, fire ≤0.7%) — 이제 "모른다"가 아니라 "적다"를 안다. 전용 penetration tool = **저우선 확정** (audit matrix 2-3 의 "유병률 근거 후 설계" 이행).
3. **Float gate-fire ≤1.1%** (occur 61~68% 인데) — clean p99 0.70 이 사실상 도달 불가: floating 판단은 gate-fire 가 아니라 **raw score/FootFloating** 로.
4. **좌표 효과 실증 (A-4)**: local 측정 시 Skate fire — motiongpt **21.3%** vs trajectory **1.7%** (12배 과검출), momask 18.9% vs 1.3%. MDM 은 26.1% vs 23.1% (진짜 skate 존재). Penetrate 도 local 에서 왜곡 (motiongpt 11.8% vs 1.2%). → **foot-계 gate 는 trajectory 입력 의무** 재확인.

## 4. Retroactive caveat 박제

- [AR-043 spec](../../.claude/docs/dashboard-task-specs/AR-043-dataset-issue-prevalence-audit.md)·[AR-044 spec](../../.claude/docs/dashboard-task-specs/AR-044-g2-real-problem-taxonomy-headroom.md) 헤더에 v0.1.0 Skate/Penetrate 컬럼 인용 금지 명시.
- [metric_provenance §3-5-1~3](../../.claude/docs/governance/metric_provenance.md)·[reproducibility-checklist §3](../../.claude/skills/reproducibility-checklist/SKILL.md) 정의 v2 로 갱신.
- 구 snapshot 수치는 **무수정** (§6-2) — severity_version 으로 구분.

## 5. 남는 불확실성 (§3-22)

- v2 임계 (0.05/0.035/0.025) = GMD 계열 관례 + internal proxy assumption — perceptual 검증 (AR-058-3f) 대기.
- Float 통일: `float_mag`(physical_metric, 구 정의) 는 historical 비교용 미변경 (A-6 부분 수리).
- **downstream 재학습 미실시**: effect-aware state (AR-037/040/052 계열) 는 v0.1.0 feature (Skate/Penetrate ≡ 0 = 무신호) 로 학습됨 — 수리된 state 로의 re-train 은 별도 task. 기존 learned 결과 해석 시 "skate/penetrate state 무신호" caveat.
- BoneCV gate 무판별 = 분포 특성 (해소 불가 확인) — raw score 원칙 고정.
