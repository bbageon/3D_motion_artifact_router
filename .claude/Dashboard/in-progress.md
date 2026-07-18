# 🟢 In Progress

> 현재 진행 중인 작업. **한 번에 1개** 권장. [index](README.md) · 인접 상태: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| AR-073 | Perceptual | **AR-072 다중 평가자 pack 완성 — 평가자 모집·판정 대기(사용자, 병렬)**. rater2/3 배포본 (익명 item_NN, 순서 셔플). 추가 rater = `--raters rater4 …` | 🔴 | 07-12 | [rater2](../../reports/figures/2026-07-12/ar073_multirater_pack/rater2/index.md) · [rater3](../../reports/figures/2026-07-12/ar073_multirater_pack/rater3/index.md). 응답 후 Fleiss κ + aggregate |
| AR-065 | RL-Q | **① Pre-action state 재구축 — 산출 완료(4b15b9b)**: 2,700 motions [CSV](../../evals/snapshots/preaction_state_ar065_v1.csv) (leakage 없는 6 feature; counterfactual mismatch 가 GT-free 로 MDM 0.0100 vs VQ ≈0 분리). 잔여: 구 v0.1.0 feature 분포 비교 + AR-040/052 re-train 판단 | 🟠 | 07-13 | [spec](../docs/dashboard-task-specs/AR-065-effect-aware-state-rebuild.md). AR-078 에서 state 활용됨 (AUC 유의 개선 확인) |

---

운영: 작업 시작 시 [ready.md](ready.md) 에서 항목을 본 파일로 이동(pull). 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
