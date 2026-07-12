# AR-072 - Root-Aware Correction Design (병인 직접 처방 — 새 pair 후보)

Status: backlog — **착수 = 사용자 게이트** (새 pair = 새 사전등록, §3-11)  
Epic: Tool  
Priority: 🔴 (후보)  
Parent: AR-071 (부분 지지: MDM v_root = GT 의 42%, 부족↔fs ρ=+0.34) / 사전 합의 순서 ③

## Goal

foot skating 의 병인 후보(root 전진 부족)를 **직접** 다루는 보정 설계 — 발을 root 에 맞추는 대증요법(anchoring, b1 2회 실패)의 반대 방향: **root 를 발에 맞춘다**.

## Candidate Mechanisms (설계 시 비교)

1. **Contact-consistent root solve (권장 후보)**: 접지 frame 에서 stance foot 의 world 속도가 0 이 되도록 root 수평 변위를 frame 별로 재해석 — 관측된 접지가 곧 제약이므로 **GT 불필요**, 이동 거리를 임의로 늘리는 게 아니라 "다리가 이미 함의하는 만큼" 전진시킴. (비유: 러닝머신 벨트 속도에 기계 전진 속도를 맞춤.)
2. Root trajectory rescale (전역 배율): 단순하나 prompt 의미(거리) 왜곡 위험 큼.
3. Hybrid: 1 을 기본, 비접지 구간은 smooth interpolation.

## Guards (사전 고정)

- **Semantic (Cat-A)**: 이동 거리는 prompt 의미의 일부 — R-Precision/MM-Dist equal-N 비교 의무. 악화 시 기각.
- foot_skate_world(개선 기대) · float_mag · legCV · accel — 전 축 before/after (§6-12).
- KDG §3-3 정합: root-first — 본 tool 은 KDG 최상위 (기존 말단 tool 과의 순서 규칙 명시).

## 검정 경로 (착수 시 사전등록할 것)

물리 (pool) → **지각 A/B 는 새 pair 의 새 판정** (anchoring 계열의 "마지막 재검정 소진"과 별개 — 단 동일 규율: 사전 기준, blind, 1회 원칙). AR-071 의 인과 확증(개입 실험)을 겸함: root-solve 후 fs 가 기계적으로 감소하는 것은 당연하므로, **지각 개선 + Cat-A 보존**이 실질 판정 기준.

## Claim Boundary

금지: 등록만으로 "병인 해결" 주장 / GT 참조 보정 (GT 는 진단에만 — 보정은 GT-free 여야 배포 가능) / anchoring 실패의 재시도로 포장 (다른 병인, 다른 pair).
