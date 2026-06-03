# Policy-Validation Traceability — field spec & claim rules

> [AGENTS.md §3-25/§3-26](../../AGENTS.md) (Policy-Validation Traceability Gate / Action-Effect Coverage) 의 **상세 field spec**. AGENTS.md 는 invariant(1줄)만, 본 docs 가 field 목록·claim rule·예시 (참조용). 사용자 directive 2026-06 박제.

## 0. 원리

ArtifactRouter 의 궁극 목표는 motion quality 향상이므로, 하네스는 policy 의 역할(ranker / direct executor / risk estimator / oracle teacher)을 하나로 제한하지 않는다. 다만 learned policy / Q surface / risk head / heuristic / oracle / physical gate 를 사용한 결과를 인용할 때 **무엇이 품질 향상에 기여했는지 분리 가능**해야 한다. (AGENTS.md §3-24 의 "configuration 과 claim 일치" 강제와 일관.)

## 1. Policy-Validation Traceability (AGENTS.md §3-25) 의무 field

raw record + report 에 기록:

| field | 값 |
|---|---|
| `selection_mode` | `direct_policy` / `ranked_candidates` / `risk_filtered` / `gated_execution` / `oracle` / `heuristic` / `diagnostic_no_gate` 중 하나 또는 조합 |
| `candidate_trace` | 후보별 `(tool, u 또는 strength, predicted_utility, predicted_risk 또는 p_safe, rank, gate_result, utility_after_apply)`. top-k 방식이면 최소 top-k 후보 전체 |
| `gate_recheck` | physical gate 를 실제 적용 후 재검증했는지 (bool) |
| `policy_contribution_baseline` | policy contribution 주장 시 비교 baseline (`random_safe` / `heuristic_safe` / `STOP_only` / `dense_oracle` 등) |
| `policy_model_id`, `action_space_type`, `u_grid_train`, `u_grid_eval`, `gate_result`, `executed_action`, `rollback_reason` | (AGENTS.md §3-6 평가 기록 의무의 policy 부분) |

## 2. Claim rule

- `gate_recheck=false` 결과 → **no-gate diagnostic 으로만 인용** (`diagnostic_no_gate`). physical safety / final quality claim 의 sole evidence 금지.
- learned policy 가 motion quality 를 높였다고 주장 → gate 포함 동일 조건에서 random/heuristic/STOP baseline 대비 utility 또는 Category A/B/C metric 개선 제시. 개선 없거나 baseline 없으면 **`policy contribution not isolated`** 표기.
- physical gate 결과의 gate-failed 후보 → executed action 이 아니라 **rejected candidate** 로 기록.
- **§6-14**: learned `P_safe`/`Q_safe`/`risk_head`/classifier score 를 real gate 검증 없이 physical safety evidence 로 인용 금지. gate-free 는 `diagnostic_no_gate` + unsafe-as-safe/false-safe rate 동반 보고.
- **§6-16**: gate 붙여서 좋아졌다는 사실만으로 learned policy 우월 주장 금지. baseline 비교 없으면 `gate-validated result; policy contribution not isolated`.

## 3. Action-Effect Coverage / Hard-Example Provenance (AGENTS.md §3-26)

continuous-u / dense-grid / Q-surface 학습의 단위 = motion sample 이 아니라 `(state, tool, u, after_state, gate_result, utility)` transition. transition dataset / hard-example mining 생성 시 기록:

- **transition dataset**: `transition_dataset_id`, `state_source_distribution`, `state_count`, `transition_count`, `tool_set`, `u_grid`, `horizon`, `seed`.
- **hard-mined**: `mining_reason`, `mining_policy_snapshot`, `target_tool`, `target_u_range`, `boundary_type`, `high_utility_unsafe`, `adjacent_safe_pair`, `source_distribution`.
- hard-mined transition 은 natural/random transition 과 **분리 보고**. hard mining improvement 를 real generator natural performance 로 일반화 시 [AGENTS.md §3-17](../../AGENTS.md) evidence tier 명시.

## 4. Action space evidence (AGENTS.md §6-15)

3-level / 5-level / dense-grid proxy / bounded continuous-u 결과를 `action_space_type` + `u_grid` + `tool_u_mapper_version` + evaluator/gate config 없이 같은 evidence 로 묶어 인용 금지. action space 다른 결과는 **oracle ceiling / learned policy / continuous surface** 중 어느 claim 인지 분리. 단일 출처: [docs/action_space_provenance.md](../../docs/action_space_provenance.md).

## 5. 위반 시 effect

위 field 누락 / claim rule 위반 = [AGENTS.md §6-5 silent invalidation](../../AGENTS.md) 동급. policy contribution / gate contribution 분리 근거로 인용 무효.
