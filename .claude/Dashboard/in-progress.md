# 🟢 In Progress

> 현재 진행 중인 작업. **한 번에 1개** 권장. [index](README.md) · 인접 상태: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| AR-024 | Evidence | 독립 pool 재현 — **생성 완료 (2,700, fail 0) + 1차 재현 성공(62b55d0)**: root 붕괴 회귀가 독립 pool 에서 재현 (MDM slope 0.0272→**0.0283**·분산비 5.7→**6.0%**·ratio 0.419→**0.418**; VQ 추종 유지). **잔여: 신 pool Cat-A 재현** (root 보정 → MDM 개선/VQ 악화, mgpt ~1-2h) + (선택) 정책 신pool 평가 | 🔴 | 07-19 | 4f8da09·62b55d0 · replpool snapshots · [일지](../../reports/2026-07-19.md) |
| AR-065 | RL-Q | **① Pre-action state 재구축 — 산출 완료(4b15b9b)**: 2,700 motions [CSV](../../evals/snapshots/preaction_state_ar065_v1.csv) (leakage 없는 6 feature; counterfactual mismatch 가 GT-free 로 MDM 0.0100 vs VQ ≈0 분리). 잔여: 구 v0.1.0 feature 분포 비교 + AR-040/052 re-train 판단 | 🟠 | 07-13 | [spec](../docs/dashboard-task-specs/AR-065-effect-aware-state-rebuild.md). AR-078 에서 state 활용됨 (AUC 유의 개선 확인) |

---

운영: 작업 시작 시 [ready.md](ready.md) 에서 항목을 본 파일로 이동(pull). 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
