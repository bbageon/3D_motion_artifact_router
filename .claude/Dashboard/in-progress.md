# 🟢 In Progress

> 현재 진행 중인 작업. **한 번에 1개** 권장. [index](README.md) · 인접 상태: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| AR-085 | RL-Q | **Strength-Q v2 — over-correction 개방 (u∈[0,2]) 사전등록** — 사용자 가설 "과보정 아닐 수도". strength 정식 정의 등록(§5-3) + tool clip 개방(U_MAX=2.0, u≤1 불변) + unit u>1 선형성 검증(97/97). H-A(u>1 이득 존재?)·H-B(정책 개선?) 사전 고정 | 🔴 | 07-22 | 사용자 directive. [spec](../docs/dashboard-task-specs/AR-085-strength-q-v2-open-u.md). 1단계 = u∈{1.25,1.5,2.0} 라벨 생성 |
| AR-065 | RL-Q | **① Pre-action state 재구축 — 산출 완료(4b15b9b)**: 2,700 motions [CSV](../../evals/snapshots/preaction_state_ar065_v1.csv) (leakage 없는 6 feature; counterfactual mismatch 가 GT-free 로 MDM 0.0100 vs VQ ≈0 분리). 잔여: 구 v0.1.0 feature 분포 비교 + AR-040/052 re-train 판단 | 🟠 | 07-13 | [spec](../docs/dashboard-task-specs/AR-065-effect-aware-state-rebuild.md). AR-078 에서 state 활용됨 (AUC 유의 개선 확인) |

---

운영: 작업 시작 시 [ready.md](ready.md) 에서 항목을 본 파일로 이동(pull). 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
