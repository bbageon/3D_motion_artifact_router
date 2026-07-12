# AR-074 — Root-Aware Correction STOP Boundary (어디서 멈춰야 하나 + GT-free 판정)

- Raw: [root_stop_boundary_ar074_v1.json](../snapshots/root_stop_boundary_ar074_v1.json) · harness: [tools/root_stop_boundary_ar074.py](../../tools/root_stop_boundary_ar074.py)
- 표본: MDM 300 prompt (locomotion 165 / non-locomotion 135, GT root speed ≷ 0.01) × 3 seed. GPU 불필요.

## 핵심 결과

### 1. self-STOP 가설 = **기각** (명시적 gate 필요)

non-locomotion(제자리·정지 요구) 에서 tool 이 자동 no-op 되기를 기대했으나 —

| | 유도 root 변위 (mean) | 보정 후 root 경로 (절대) |
|---|---|---|
| locomotion | 1.205 | (전진이 정답) |
| **non-locomotion** | **0.770** (loco 의 64%) | **0.85 m** |

**정지해야 할 motion 을 평균 0.85 m 밀어버린다.** 즉 root correction 을 무조건 적용하면
"제자리 걸음 → 걸어가 버림" harm 이 실현된다. self-STOP 없음 → **명시적 STOP gate 필요**.

비유: 러닝머신 벨트 속도에 기계를 맞추는 도구인데, **트레드밀 위 제자리 걷기**(다리는 움직이나
전진 의도 없음)에 적용하면 발의 왕복 흔들림을 "전진하라"로 오해석해 사람을 밀어낸다.

### 2. GT-free STOP 신호 = **존재** (추론 시 GT 불요)

over-correction(ratio_after>1.5, loco 30/165) 을 관측 신호로 사전 식별:

| GT-free 신호 | AUC | 판정 |
|---|---|---|
| **path_gain** (보정 root 경로 / 원본 root 경로) | **0.793** | ✅ 사용 가능 |
| induced_disp (유도 변위) | 0.738 | ✅ 사용 가능 |
| constrained_frame_frac | 0.218 | ❌ 무용 |

`induced_disp ↔ fs 개선` 상관 = +0.634 (보정이 클수록 고칠 skate 많음 — apply 가치 proxy).

## 함의 — routing 주장 강화

**같은 correction 이라도 state 에 따라 apply/STOP 이 달라져야 한다**가 정량 실증:
- root correction 은 강력(AR-072: loco 19/20)하나 **무조건 적용 시 harm**(non-loco 0.85 m push).
- 판정에 GT 불요 — **path_gain(AUC 0.79) + fs magnitude** 의 관측 신호로 gate 가능.
- ⟹ 결정 계층(Safe Orchestration §0)이 필수. "보정하지 않을 때(STOP)를 아는 것"이 시스템의 절반이라는 프레임의 직접 근거.

## GT-free STOP rule 후보 (관측 기반 — 학습 policy 아님)

```
apply root correction IF:
  (a) 고칠 skate 존재     — foot_skate_world 높음 (apply 가치)
  AND (b) over-correction 아님 — path_gain 정상 범위 (AUC 0.79 신호)
STOP otherwise (특히 non-locomotion: 유도 변위가 크면 harm)
```

## Claim Boundary

허용: "root correction 은 non-locomotion 에서 기하적 harm(0.85 m 변위) → STOP 이 정답; GT-free 신호(path_gain AUC 0.79)로 판정 가능 — routing 근거".
금지: non-loco harm 의 **지각** 단정 (기하 변위 측정 — 강한 prior 이나 A/B 미실시) / STOP rule 을 학습 policy 로 포장 (관측 rule) / MDM 외 일반화.

## 남는 불확실성 / 후속

- non-loco harm 은 기하(변위) 측정 — 지각 확정은 소규모 A/B (필요 시).
- GT-free gate 의 **구현·threshold 고정·orchestrator 편입** = AR-075 (root tool = KDG 최상위 + STOP gate).
- 동일 GT-free 신호를 **VQ(AR-077)** 에 적용 → VQ STOP 판정 재사용 가능.
