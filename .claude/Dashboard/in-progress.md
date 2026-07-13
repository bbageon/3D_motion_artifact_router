# 🟢 In Progress

> 현재 진행 중인 작업. **한 번에 1개** 권장. [index](README.md) · 인접 상태: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| AR-073 | Perceptual | **AR-072 다중 평가자 pack 완성 — 평가자 모집·판정 대기(사용자, 병렬)**. rater2/3 배포본 (익명 item_NN, 순서 셔플). 추가 rater = `--raters rater4 …` | 🔴 | 07-12 | [rater2](../../reports/figures/2026-07-12/ar073_multirater_pack/rater2/index.md) · [rater3](../../reports/figures/2026-07-12/ar073_multirater_pack/rater3/index.md). 응답 후 Fleiss κ + aggregate |
| AR-077 | Evidence | VQ root no-harm — **Cat-A (pool-scoped) 강지지**: 본 pool 에서 MDM 평균 개선(R@1+0.047·MM−1.0·FID−7.3) vs VQ 고정적용 시 평균 손상 (robustness 확인 — 독립 재현 아님). per-motion gate 미완: benefit-AUC ~0.67(δ 조건부), non-beneficial apply ~50%. **마감 = AR-078 결과 후** (진단 작업 — AR-075 구현까지 안 묶음) | 🔴 | 07-12 | [report](../../evals/reports/2026-07-12_vq_root_noharm_ar077.md) · Cat-A v3+R-Prec fix · [spec](../docs/dashboard-task-specs/AR-077-vq-root-noharm-diagnostic.md) |
| AR-065 | RL-Q | **① Pre-action state 재구축 착수** — generator ID·foot_skate·root–gait 속도 불일치(counterfactual)·접촉 지속성·trajectory 형태·locomotion intent (+support/uncertainty 후보). 보정 후 Cat-A·path 변화는 **label/최종 guard 로만** (after-action leakage 금지) | 🟠 | 07-13 | 4차 피드백: P5 최종화보다 우선. [spec](../docs/dashboard-task-specs/AR-065-effect-aware-state-rebuild.md). AR-078 재료 |

---

운영: 작업 시작 시 [ready.md](ready.md) 에서 항목을 본 파일로 이동(pull). 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
