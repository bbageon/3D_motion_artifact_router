# 🟢 In Progress

> 현재 진행 중인 작업. **한 번에 1개** 권장. [index](README.md) · 인접 상태: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| AR-082 | Perceptual | **u\*(s) 정책 지각 A/B — pack 완성, 사용자 판정 대기** (20쌍: MDM 12/MGPT 5/MoMask 3, u\*=1.0×15·0.75×2·0.5×3). [index](../../reports/figures/2026-07-19/ar082_policy_ab_pack/index.md) → 채팅으로 A/B 20개. ⚠️ answer_key 열람 금지 | 🔴 | 07-19 | 기준 ≥15 지지("정책 작동" 유보 해제)/≤12 기각(1회)/13-14 b2. [spec](../docs/dashboard-task-specs/AR-082-policy-perceptual-ab.md) · [prereg](../../evals/snapshots/ab_policy_pack_ar082_v1.json) |
| AR-065 | RL-Q | **① Pre-action state 재구축 — 산출 완료(4b15b9b)**: 2,700 motions [CSV](../../evals/snapshots/preaction_state_ar065_v1.csv) (leakage 없는 6 feature; counterfactual mismatch 가 GT-free 로 MDM 0.0100 vs VQ ≈0 분리). 잔여: 구 v0.1.0 feature 분포 비교 + AR-040/052 re-train 판단 | 🟠 | 07-13 | [spec](../docs/dashboard-task-specs/AR-065-effect-aware-state-rebuild.md). AR-078 에서 state 활용됨 (AUC 유의 개선 확인) |

---

운영: 작업 시작 시 [ready.md](ready.md) 에서 항목을 본 파일로 이동(pull). 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
