# 🟢 In Progress

> 현재 진행 중인 작업. **한 번에 1개** 권장. [index](README.md) · 인접 상태: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| AR-081 | RL-Q | **Strength-Q v1 (사전등록 완료)** — 한 번의 root 보정 강도 Q(s,u): 13차원 직관 이름 상태 (stepping_speed·body_speed_shortfall·foot_sliding + 신뢰도 2 + 문맥 5 + generator 3) × u∈{0,.25,.5,.75,1} · 그룹 additive 구조 · harm head · Q±gen shortcut 검사 · Markov 축약 검정. one-shot scope (closed-loop 는 별도) | 🔴 | 07-13 | [spec](../docs/dashboard-task-specs/AR-081-strength-q-v1.md) · §5-3 provenance 등록. 1단계 = u-grid 라벨 생성 중 |
| AR-065 | RL-Q | **① Pre-action state 재구축 — 산출 완료(4b15b9b)**: 2,700 motions [CSV](../../evals/snapshots/preaction_state_ar065_v1.csv) (leakage 없는 6 feature; counterfactual mismatch 가 GT-free 로 MDM 0.0100 vs VQ ≈0 분리). 잔여: 구 v0.1.0 feature 분포 비교 + AR-040/052 re-train 판단 | 🟠 | 07-13 | [spec](../docs/dashboard-task-specs/AR-065-effect-aware-state-rebuild.md). AR-078 에서 state 활용됨 (AUC 유의 개선 확인) |

---

운영: 작업 시작 시 [ready.md](ready.md) 에서 항목을 본 파일로 이동(pull). 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
