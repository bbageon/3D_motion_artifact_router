# Artifact ↔ Tool Definition Alignment Audit (AR-062)

Status: **completed** — 2026-07-07
Task spec: [AR-062](../dashboard-task-specs/AR-062-artifact-tool-definition-audit.md) (Parent AR-058/AR-060)
Evidence: [snapshot](../../../evals/snapshots/artifact_tool_alignment_audit_ar062_v1.json) (controlled diagnostic) · [audit script](../../../tools/artifact_tool_alignment_audit_ar062.py)
Evidence tier (§3-17): 본 문서의 vacuity 재현 = **controlled diagnostic** (정의/논리 재현, 성능 결론 아님). 유병률 인용은 기존 real-distribution snapshot cross-link.

> **Goal**: foot-skate 교훈 (artifact = 접지 중 **수평** 미끄러짐 vs old tool = **Y-only** 내리누르기 → 축 mismatch) 을 전 artifact 로 확장 점검. tool 을 더 만들기 전에 **각 artifact 정의 ↔ metric ↔ tool 메커니즘이 같은 변수를 다루는지** 감사한다. 정의가 어긋난 채 routing/RL 을 학습하면 잘못된 action space 를 배운다.

---

## 0. 요약 (한눈에)

| # | 발견 | 등급 |
|---|---|---|
| A-1 | **SkateEvaluator (gate) 는 내부 contact 추정 사용 시 구조적으로 fire 불가** (vacuous) — contact 조건(xz 속도 ≤ 0.02)과 skate 조건(xz 속도 > 0.05)의 교집합이 공집합 | 🔴 구조적 |
| A-2 | **PenetrateEvaluator (gate) 는 내부 ground(min-Y) 사용 시 구조적으로 fire 불가** — 어떤 joint 도 전-joint min-Y 아래에 있을 수 없음 | 🔴 구조적 |
| A-3 | **BoneLengthCV gate 는 무판별(no-discrimination)** — clean p99 = 5.5e-6 (HumanML3D bone 이 사실상 상수) → 생성 motion 100% fire | 🟠 보정 |
| A-4 | **수평속도 기반 contact 검출을 local(root-relative) 좌표에 적용** — 보행 중 지지발은 pelvis 기준으로 뒤로 이동(≈보행속도)하므로 contact(속도≤0.02) 판정 실패 → FootFloating 은 보행 구간에서 사실상 blind | 🟠 좌표 |
| A-5 | (기지, P4) **Y-only FootLock 은 skate 에 축 mismatch** — 전 frame 무조건 Y 하강 → swing 발이 지면 근처에서 수평 이동 = skate↑ | 🔴 확정(P4) |
| A-6 | **floating 의 contact/ground 정의가 4종 공존** (FootFloating v1.2 / Float gate / float_mag / v2) — 같은 artifact 의 유병률이 정의에 따라 41% vs 0% 로 상반 | 🟠 정합 |

**AR-061 verdict: PROCEED** (§7). CoordinateFootSkateCleanupTool 은 metric(foot_skate_world)과 **같은 변수·같은 좌표계** 를 수정하는 유일한 aligned mechanism. 단 평가에서 gate-fire 지표(Skate/Penetrate/BoneCV-fire) 인용 금지 — guard 는 foot_skate_world / float_mag / **raw BoneCV 값** / FID(equal-N) 로.

---

## 1. 구조적 vacuity 재현 (A-1, A-2) — controlled diagnostic

비유: **"시속 5km 이하로 달리는 차 중에서 시속 20km 초과인 차를 찾아라"** — 어떤 차도 두 조건을 동시에 만족할 수 없다. SkateEvaluator 의 내부 contact 추정이 정확히 이 구조다.

### A-1. SkateEvaluator

```
contact (내부 추정) = xz_vel ≤ 0.02  (physical_gate.py:52, _estimate_contact :94)
sliding             = xz_vel > 0.05 & contact  (physical_gate.py:246)
→ sliding ⊆ {xz_vel ≤ 0.02} ∩ {xz_vel > 0.05} = ∅
```

재현 ([snapshot](../../../evals/snapshots/artifact_tool_alignment_audit_ar062_v1.json)): 두 발이 ground 에 붙은 채 0.06 m/frame 수평 slide 하는 synthetic (frame 전체가 skate) 에서 —
- SkateEvaluator (내부 contact): **0 report** (아무것도 검출 못함)
- SkateEvaluator (외부 contact_labels 강제): 2 report, score **1.0/1.0**
- v2 정의(수직속도 기반 contact, AR-058-3b): skate **30/30** 정상 검출

**전이 영향 (retroactive caveat)**:
- `physical_gate_clean_calibration_v1.json` 의 Skate clean p99 = 0.0 → **vacuous-by-construction** (calibration 이 `ev.evaluate(motion)` 만 호출 — labels 미전달, physical_gate_clean_calibration.py:86).
- `representative_pool_measure_v1.json` 의 `Skate_gate_fire = 0.0` (3 generator 전부) → **"generator 가 skate 하지 않는다"의 증거가 아님**. 실제 skate 는 foot_skate_world(Category B, 건전) 가 잡았고 (MDM 이 VQ 2배, CI 분리) AR-058-3f 인간 검증 pack 의 high_v2 9/9 도 v2 로 검출된 것.
- AR-043/AR-044 의 Skate_gate 컬럼 인용 금지.

### A-2. PenetrateEvaluator

```
ground (내부 추정) = min-Y (전 joint)  (physical_gate.py:75-77)
penetrate          = joint_y < ground − 0.02  (physical_gate.py:124)
→ 어떤 joint_y 도 전-joint min 아래일 수 없음 = ∅
```

재현: RIGHT_FOOT 이 10 frame 동안 −8cm 관통하는 synthetic 에서 — 내부 ground: **0 report** / `ground_y=0.0` 전달: score **0.333** (10/30) 정상 검출.

**전이 영향**: clean p99 = 0.0 vacuous; AR-043/044 의 "penetration ~0%" 는 관통 없음의 증거로 인용 금지 (estimate_ground 주입 후 재측정 필요).

### 영향받지 않는 결론

- **foot_skate_world** (trajectory + estimate_ground 10th pct, Category B) 는 건전 → AR-044 headline (MDM foot-skate headroom 2x) **유지**.
- float_mag / FID / R-Precision 기반 결론 유지.
- [motiongpt_implications F7](motiongpt_implications.md) "물리적으로 깨끗" 의 근거 중 gate-fire 부분은 vacuity caveat 동반 — 단 상대 서열(MotionGPT 가 foot_skate_world 최저)은 Category-B 로 성립 유지.

---

## 2. Artifact ↔ Tool matrix (8 질문 × 6 그룹)

각 행: 정의 → metric (caveat) → tool → 메커니즘 일치 → 알려진 실패 → guard → 증거 상태 → next.

### 2-1. Foot skating / contact drift

| 항목 | 내용 |
|---|---|
| 정의 | 접지(발이 바닥에 붙은) 중 발이 **수평(X/Z)** 으로 미끄러짐 |
| Metric | ✅ `foot_skate_world` (GMD/EDGE, trajectory + estimate_ground, Cat B) · ✅ v2 mask (AR-058-3b, 수직속도 contact) · ❌ SkateEvaluator gate (A-1 vacuous) · ⚠️ `foot_skate` (physical_metric_g2_stress — min-Y ground: 관통 존재 시 ground 가 내려가 h 과대 → skate 과소평가) |
| Metric caveat | v2/foot_skate_world 는 후보 metric — 인간 검증(AR-058-3f)은 응답 대기 (perceptual 확정 아님) |
| Tool | 기존 FootLockTool = **mismatch** (A-5, P4 확정) → [CoordinateFootSkateCleanupTool frozen spec](../design/coordinate_footskate_cleanup_tool_frozen_spec.md) (AR-060) |
| 메커니즘 일치 | ✅ **aligned** — metric 의 변수(접지 중 수평 변위)를 같은 좌표계(trajectory)에서 직접 수정 (v2 contact segment 의 X/Z 를 anchor 로) |
| 알려진 실패 | Y-only: artifact↓ vs foot_skate↑ mixed (p4_fixed_tool_effect_v1, 3/3 generator) |
| Guard | float_mag·artifact_total(C)·raw BoneCV·FID(equal-N)·correction magnitude |
| 증거 상태 | partial (설계 frozen, 구현/평가 미실시) |
| **Next** | **AR-061 진행** (구현+평가) + SkateEvaluator 수리는 별도(AR-063) |

### 2-2. Floating / missing contact

| 항목 | 내용 |
|---|---|
| 정의 | 딛고 있어야 할 발이 공중에 떠 있음 |
| Metric | ⚠️ FootFloatingEvaluator v1.2.0 (local + min-Y ground + **수평속도 contact** → A-4: 보행 중 blind, 정지 자세 중심 검출) · ⚠️ Float gate (clean p99 = **0.798** → 사실상 fire 불가·무력) · ✅ float_mag (Cat B, 경향 지표) |
| Metric caveat | 유병률이 정의 의존 (A-6): FootFloating_fire 40.7% (MotionGPT) vs Float_gate_fire 0% — 같은 artifact 에 상반 결론 |
| Tool | FootLockTool (Y 하강) |
| 메커니즘 일치 | ⚠️ **partial** — 변수(발 높이)는 일치하나 (a) 전 frame 무조건 적용 → swing 발까지 끌어내려 skate 생성, (b) contact-intended segment 조건화 부재. routed 사용(evaluator frames 를 frame_range 로) 시 부분 완화 |
| 알려진 실패 | float↓ → skate↑ trade (P4) |
| Guard | foot_skate_world·FID |
| 증거 상태 | partial |
| **Next** | FootLock 에 contact-segment 조건화 재설계 (AR-061 결과 반영 후; coordinate tool 의 Y 성분과 통합 검토) + floating 정의 통일(AR-063 에 포함) |

### 2-3. Ground penetration

| 항목 | 내용 |
|---|---|
| 정의 | 발/발목이 지면 아래로 파고듦 |
| Metric | ❌ PenetrateEvaluator (A-2 vacuous — 내부 min-Y ground) |
| Metric caveat | **현재 유병률 자체를 모름** (측정기가 구조적으로 0) |
| Tool | **없음** (missing). FootLock 은 ground 방향 양방향 보간이라 관통 시 위로 끌어올리긴 하나 동일한 전-frame 문제 |
| 메커니즘 일치 | n/a — metric 부터 수리 필요 |
| 알려진 실패 | — |
| Guard | (수리 후) float_mag·foot_skate_world |
| 증거 상태 | missing |
| **Next** | AR-063: estimate_ground 주입 + 유병률 재측정 → 유병률이 실재할 때만 clamp+blend tool 설계 (유병률 없이 tool 먼저 만들지 않음) |

### 2-4. Bone length CV

| 항목 | 내용 |
|---|---|
| 정의 | 시간에 따라 bone 길이가 변함 (skeleton 비강체성) |
| Metric | ✅ BoneLengthEvaluator (Layer A) + BoneLengthCVEvaluator raw 값 · ❌ BoneCV **gate-fire** (A-3: clean p99=5.5e-6 → 생성물 100% fire, 무판별 — dataset_issue_prevalence_audit.py:179 에 기지) |
| Metric caveat | AR-058-2a: BoneCV 는 export 좌표 단계부터 존재(protocol 검증); perceptual artifact 확정은 보류 |
| Tool | BoneProjectionTool |
| 메커니즘 일치 | ✅ **aligned** — bone 길이를 reference 로 직접 투영 (같은 변수). 단 side effect: child+descendant 통째 이동 → 발 위치 변경 → skate/float 유발 가능 (§6-12 cross-evaluator 기록 의무) |
| 알려진 실패 | 전 split 100% fire 로 q_proxy 지배 위험 (기지) |
| Guard | foot_skate_world·float_mag·FID |
| 증거 상태 | partial |
| **Next** | AR-063 에 gate threshold 재보정 포함 (raw CV 값 사용 원칙; fire 임계는 판별력 있는 분위수로 재설계) |

### 2-5. Temporal jitter / jerk

| 항목 | 내용 |
|---|---|
| 정의 | frame 간 가속도/저크 스파이크 (시간적 매끄러움 결손) |
| Metric | ✅ VelocityJitterEvaluator·JerkSpikeEvaluator (accel p95) — 단 local 측정이라 **root(전신 이동) jitter 는 미측정** |
| Metric caveat | engineering/proxy — perceptual·표준 지표 연결 전까지 secondary (task spec 지침과 일치) |
| Tool | VelocitySmoothingTool (gaussian) |
| 메커니즘 일치 | ✅ aligned (가속도를 직접 축소) — 단 **joint 별 독립 smoothing 이 bone 길이 왜곡** (BoneCV↑; B2 100% BoneCV 위반 이력) + contact 뭉갬 (skate/float 영향) |
| 알려진 실패 | "아무 smoothing 보상 함정" (기지) |
| Guard | raw BoneCV·foot_skate_world·FID |
| 증거 상태 | completed (P4/B2 계열에서 특성 파악) |
| **Next** | 유지 (secondary 위상 고정). 우선 구현 대상 아님 |

### 2-6. Semantic mismatch / intent conflict

| 항목 | 내용 |
|---|---|
| 정의 | 생성 motion 이 프롬프트 의도와 다름 |
| Metric | R-Precision·MM-Dist (Cat A, 외부) — [F8](motiongpt_implications.md): MotionGPT 위반은 이 축에 집중 |
| Tool | **없음/미래** (기하 tool 범위 밖 — F9) |
| 메커니즘 일치 | n/a — 좌표 보정 대상 아님 |
| **Next** | hold (AR-059 계열 미래 작업; 본 audit 범위 밖 확정) |

---

## 3. AR-061 verdict + 우선순위 함의

1. **AR-061 = PROCEED.** foot-skate 는 유일하게 (건전한 Cat-B metric) + (aligned mechanism 설계 frozen) + (P4 로 필요성 확정) 3박자가 갖춰진 그룹. audit 이 추가하는 제약: 평가에서 **gate-fire 지표 인용 금지** (Skate/Penetrate vacuous, BoneCV 무판별) — guard 는 foot_skate_world/float_mag/raw BoneCV/FID 로.
2. **AR-063 신설 (gate evaluator 수리)**: SkateEvaluator contact 를 v2(수직속도) 방식으로, PenetrateEvaluator/FootFloating 에 estimate_ground 주입, Float/BoneCV gate threshold 재보정, AR-043/044 gate 컬럼 retroactive caveat 박제. **routing/RL 의 state 가 이 evaluator 들에서 나오므로 A-1~A-4 방치 시 잘못된 state 로 학습** — AR-061 다음 우선.
3. **AR-058-5 (P5 종합) 는 본 audit 반영 전 광범위 routing claim 금지** (task spec Success Criteria) — P5 에서 gate-fire 컬럼을 근거로 쓰지 않도록 명시.

## 4. Claim boundary

허용: "artifact-tool 정합을 감사해 유효한 baseline tool 과 재설계 필요 tool 을 식별했다."
금지: "본 audit 이 tool 의 품질 개선을 증명한다" / "per-tool 실증 평가를 대체한다".

## 5. Research grounding (§3-22)

1. **판단/권고**: 위 matrix·verdict.
2. **근거 논문**: GMD (Karunratanakul et al., ICCV 2023) — foot-skate metric (근접지 높이 + 수평 skid); PhysDiff (Yuan et al., ICCV 2023) — 생성 motion 의 float/slide/penetration 실재 + naive post-proc side effect; HuMoR (Rempe et al., ICCV 2021) — bone 길이/신체 일관성 = plausibility 제약.
3. **적용 범위**: evaluators/(gate 5종 + Layer A 3종)·correction_tools/(3종+신규 1종)·routing state 설계.
4. **남는 불확실성**: temporal smoothness = engineering/proxy (perceptual 연결 전); v2 임계 상수(0.05/0.035/0.025) = GMD 계열 관례 + internal proxy assumption; floating 통일 정의는 미확정 (AR-063 에서); 본 audit 의 유병률 재측정은 미실시 (수리 후 측정).
