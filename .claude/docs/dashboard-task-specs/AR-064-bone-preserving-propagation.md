# AR-064 - Bone-Preserving Propagation for Coordinate Cleanup

Status: backlog  
Epic: Tool  
Priority: 🟠  
Parent: AR-061 (guard 결과 후속)

## Goal

CoordinateFootSkateCleanupTool 의 leg-chain **선형 propagation** (foot/ankle/knee/hip = 1.0/0.5/0.2/0.1, NOT IK) 이 유발하는 bone 길이 왜곡을 제거하거나 유의미하게 줄인다.

## Why This Exists

AR-061 guard 실측: **bone_cv_max Δ +0.045~+0.053 @u=1.0** (u 비례; u=0.25 는 +0.002) — 발만 크게 옮기고 상위 관절은 비율로만 따라가면 다리 bone 이 늘어난다. frozen spec §3-4 가 예고한 engineering-heuristic 한계의 실증.

## Candidate Approaches (구현 시 비교)

1. **사후 bone re-projection**: cleanup 후 해당 다리 chain 에 BoneProjectionTool 식 length 투영 (기존 tool 재사용, KDG ordering 준수 필요 — §3-3).
2. **간이 2-bone IK**: hip 고정, knee/ankle 각도로 발 위치 도달 (bone 길이 보존이 구조적).
3. propagation weight 재조정 (완화책 — 근본 해결 아님).

## Success Criteria

- foot_skate_world 개선 (AR-061 수준 유지) **하면서** bone_cv_max Δ 가 u=1.0 에서도 무시 가능 수준 (사전 임계는 구현 시 clean 분포 기준으로 등록).
- 비교 protocol = AR-061 과 동일 (representative pool, prompt 단위, guard 전항).

## Claim Boundary

허용: "propagation 방식 교체가 skate 개선-bone 왜곡 trade 를 완화한다" (측정 후).
금지: IK 도입만으로 품질 향상 주장 (perceptual/FID 검증 전).

## Note

u=0.25 저강도에서는 bone 왜곡이 미미(+0.002)하므로, 본 task 전까지는 저강도 운용이 안전한 회피책.
