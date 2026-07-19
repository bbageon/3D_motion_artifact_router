# 🟢 In Progress

> 현재 진행 중인 작업. **한 번에 1개** 권장. [index](README.md) · 인접 상태: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| AR-024 | Evidence | **독립 pool 재현 착수(4f8da09)** — 새 base-seed 20260719, 동일 300 prompt bank × 3 gen × 3 seed 생성 (docker 서비스 경유, detach 실행 중). 완료 후 핵심 결과 재측정 (Cat-A pool-scoped → 재현) | 🔴 | 07-19 | 논문 요건 ③. log: external_assets/protocol_rep_pool_seed20260719/generation.log |
| AR-065 | RL-Q | **① Pre-action state 재구축 — 산출 완료(4b15b9b)**: 2,700 motions [CSV](../../evals/snapshots/preaction_state_ar065_v1.csv) (leakage 없는 6 feature; counterfactual mismatch 가 GT-free 로 MDM 0.0100 vs VQ ≈0 분리). 잔여: 구 v0.1.0 feature 분포 비교 + AR-040/052 re-train 판단 | 🟠 | 07-13 | [spec](../docs/dashboard-task-specs/AR-065-effect-aware-state-rebuild.md). AR-078 에서 state 활용됨 (AUC 유의 개선 확인) |

---

운영: 작업 시작 시 [ready.md](ready.md) 에서 항목을 본 파일로 이동(pull). 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
