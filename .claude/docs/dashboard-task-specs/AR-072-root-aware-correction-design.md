# AR-072 - Root-Aware Correction Design (병인 직접 처방 — 새 pair 후보)

Status: **in-progress** (사용자 게이트 통과 2026-07-12 "응 진행해줘")  
Epic: Tool  
Priority: 🔴  
Parent: AR-071 (부분 지지: MDM v_root = GT 의 42%, 부족↔fs ρ=+0.34) / 사전 합의 순서 ③

## ① 사전등록 (2026-07-12 — 결과 보기 전 고정)

**알고리즘 (RootGaitConsistencyTool)** — 다리(pose) 무수정, root 수평 궤적만 재구성:

```
1. contact = v2 정의 (height≤0.05 & |vy|≤0.035; ground=feet 10th pct)
   — 수평 root 이동에 불변 (height·vy 는 수평 이동과 무관) → 재검출 루프 없음
2. stance 제약: 접지발의 world 속도 = 0
   ⇔ root_vel_target[t] = −mean_{접지발 f}( d/dt(foot_xz − pelvis_xz)[f][t] )
3. 무접지 frame: 이웃 정의값 선형 보간 (전 구간 무접지 시 원본 유지)
4. smoothing: 해 velocity 에 gaussian σ=2 frame (노이즈→root jitter 방지)
5. blend: v_new = (1−u)·v_orig + u·v_target — primary u=1.0, u_grid {0.5,1.0} 참고
6. 적분→새 pelvis 경로 (시작점 고정), offset 을 전 joint xz 에 동일 적용
   → local(root-relative) 표현 = 구조적으로 완전 불변 (bone·pose·height 보존)
```

**Guards (사전 고정 — 기각 조건 포함)**:

| Guard | 기준 | 판정 |
|---|---|---|
| **Semantic (Cat-A, 핵심)** | R-Precision top-1/2/3 + MM-Dist, equal-N (corrected vs original, 동일 caption, tm2t evaluator) | **R@1 하락이 CI 밖이거나 MM-Dist 유의 악화 → 처방 기각** (물리가 아무리 좋아도) |
| 기전 표적 | v_root/GT ratio 가 0.42 → **1.0 방향으로 이동** | 이동 없으면 solve 결함 |
| fs (Cat-B) | 감소 (정의상 기대 — 성공 기준 아님, 보고만) | — |
| 불변성 | local·bone·float 완전 불변 (구조적) — assert 검증 | 위반 = 구현 버그 |
| smoothness | root accel p95 악화 감시 | 유의 악화 시 σ 재설계 (사전등록 재개정 필요) |

**검정 순서**: ② 구현+unit → ③ 물리(즉시)+Cat-A(mgpt env) → 통과 시 ④ 새 blind A/B (신규 20 prompt, v1/v2 와 비중복, 기준 ≥15/≤12/13-14 동일 구조, **1회 원칙**).

### ①-보강 (2026-07-12, 외부 피드백 채택 — Cat-A 결과 도착 전 amendment)

1. **FID guard 추가**: equal-N — FID(corrected, GT-ref) 가 FID(original, GT-ref) 대비 유의 악화 시 기각 (분포 수준 축; R-Prec/MM-Dist 는 짝 수준 축 — 상보).
2. **표현 규율**: 성공 시 "인과 증명" 금지 — **"기전 가설의 intervention evidence"** 까지만 (개입이 경로·속도·리듬을 동시에 바꾸므로 단일 요인 인과 아님).
3. **④ A/B 는 locomotion subset 한정** (GT 평균 root 속도 > 0.01 m/frame): **"walk in place" 류가 본 처방의 급소** — 다리는 걷되 root 정지가 정답인 prompt 에서 solve 가 root 를 밀면 의미 파괴. + over-correction 꼬리 (ratio_after ≫ 1) 점검 보고.
4. **contact 오추정 한계 명시**: confidence 가중 미구현 (현 완충: 접지 양끝 frame 만 제약 + 보간 + σ=2 smoothing + contact 판정의 수평-solve 불변성). guard 실패 시에만 구현하는 조건부 후속.
5. anchoring 과의 지각 3-arm 은 하지 않음 — pairwise protocol 유지, anchoring 대비는 수치 표 (동일 pool 기존 snapshot).

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
