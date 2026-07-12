# AR-071 - Root Translation Deficit Diagnostic (MDM Foot-Skate 기전 가설)

Status: backlog (등록 — 사용자 관찰 2026-07-12)  
Epic: Evidence  
Priority: 🔴 (v2 판정 해석에 직결)  
Parent: AR-058-3i / 사용자 관찰: "걷는 것에 비해 전진 거리가 짧아 보이고, 그로 인해 foot skating 이 많이 발생한 것으로 보인다"

## Hypothesis (internal, 검정 대상 — 사용자 확정 문구 2026-07-12)

> **"foot skating 은 root trajectory 와 gait speed mismatch 의 결과일 수 있음."**

기전: root(골반) 전진량이 다리 보행 주기가 함의하는 속도보다 부족 (root translation
deficit). 다리는 v_gait 로 걷는 pose 를 만드는데 root 는 v_root < v_gait 로만 전진 →
접지발의 world 속도 = v_root − v_gait < 0 → **접지 중 발이 뒤로 미끄러짐** (문워크 방향).

## 확정 실행 순서 (사용자 승인 2026-07-12 — 판정 전 고정)

1. **AR-058-3i v2 A/B 판정 먼저** (본 진단 결과를 먼저 보면 기대 형성으로 판정 오염 —
   진단은 판정 후에만 실행·보고).
2. **AR-071 진단 실행** (아래 검정).
3. **지지 시**: root-aware correction 을 **새 task 로 등록** (새 pair = 새 사전등록 +
   사용자 게이트 — §3-11).
4. **지지 시**: 기존 FootLock / coordinate cleanup 을 **"대증요법(symptomatic
   treatment)"으로 재해석** — 증상(미끄러짐)을 국소 억제하나 병인(root deficit)을
   고치지 않음. P5/position 문서에 재해석 반영.

## 검정 (싼 진단, MDM representative 300)

1. `v_root` = 보행 구간 골반 수평 속도.
2. `v_gait_implied` = 접지 frame 의 |발−골반 상대 수평속도| (스탠스의 "벨트 속도").
3. `mismatch = v_gait_implied − v_root` (양수 = root 부족).
4. **방향 검정**: 접지 중 발 world 속도의 heading 방향 성분 부호 — 가설 예측 = **음수(뒤로)**.
5. `mismatch` ↔ `foot_skate_world` Spearman 상관 (prompt 단위 n=300).

## Implications (검정 결과에 따라)

- 가설 지지 시: **올바른 보정은 발 고정(anchoring)이 아니라 root 전진 rescale** 일 수 있음 — 발을 부족한 root 에 고정하면 "제자리걸음/러닝머신" 인상 (v1/v2 선호 실패의 기전 후보). root-space tool = **새 pair 후보** (KDG 도 root 우선 — §3-3 과 정합). 단 root rescale 은 semantic 위험(이동 거리 = 의미) — text 와의 정합 검토 필요.
- 기각 시: skate 는 국소 접지 불안정 — 현 pair 유지.

## 규율

- **v2 판정과 독립**: v2 는 combo 처방의 지각 가치를 사전등록대로 판정 (본 진단이 v2 를 대체·연기하지 않음). v2 실패 + 본 가설 지지 조합이면 → "현 pair Stop + 새 기전 발견" 으로 정직 기록 (goalpost 이동 아님 — 새 pair 는 새 사전등록 + 사용자 게이트).
- Grounding: GMD ICCV2023 / EDGE CVPR2023 (contact 일관성·trajectory 유도 동기) — "root-pose 불일치" 가설 자체는 internal hypothesis 표기.
