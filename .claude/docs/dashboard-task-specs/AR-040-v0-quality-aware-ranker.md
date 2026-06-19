# AR-040 — v0 Global Quality-Aware Ranker

## 목적

AR-051에서 확인한 fixed tool trade-off를 바탕으로, v0 ranker의 학습 label이 될 `q_proxy`를 정의하고, v0 state + action-effect transition + M1 dense effect oracle을 만든다.

## 쉬운 정의

`q_proxy`는 "이 보정을 적용했을 때 전체 모션 품질이 좋아졌는가?"를 대략적으로 숫자로 만든 내부 점수다.

보상:

- artifact proxy가 줄어듦
- world foot-skate가 줄어듦
- acceleration이 부드러워짐
- FootFloating fire가 줄어듦

감점:

- 원본 모션에서 너무 많이 멀어짐
- correction magnitude가 큼
- physical gate risk가 생김
- artifact는 줄었지만 foot-skate가 악화되는 AR-051형 failure

FID/R-Precision/MM-Dist는 최종 validation 지표이므로 v0 `q_proxy`에 직접 넣지 않는다.

## Claim Boundary

이 작업은 learned policy 성능을 증명하지 않는다. 목표는 다음 단계의 학습이 가능한 label과 oracle을 만드는 것이다.

- v0 action = `(tool, u)`
- target-aware claim 없음
- continuous policy claim 없음
- `q_proxy`는 internal proxy assumption

## 입력

- Representative pool: `external_assets/protocol_rep_pool_seed20260608`
- Prompt bank: `evals/prompts/protocol_rep_test_300_seed20260608.json`
- AR-037 state schema: `tools/effect_aware_state.py`
- AR-051 tool application protocol: `tools/representative_refinement_effect.py`

## 출력

- `evals/snapshots/effect_aware_v0_oracle_v1.json`
- `reports/2026-06-19.md` AR-040 section
- Dashboard done row

## 성공 조건

1. `q_proxy` formula와 component weight가 snapshot metadata에 기록된다.
2. v0 state vector는 before-action observable feature만 사용한다.
3. transition row는 `(state_id, tool, u, before, after, effect, q_proxy)`를 포함한다.
4. M1 dense oracle은 prompt 단위 seed-mean으로 best action을 고른다.
5. STOP/noop이 action set에 포함된다.
6. generator별 oracle action 분포와 mean q_proxy gain이 보고된다.
7. AR-051에서 본 failure, 특히 artifact 개선 + foot-skate 악화가 `q_proxy`에서 낮게 평가되는지 확인한다.

## 근거

- AR-037: state schema와 before-action observable 원칙.
- AR-043: q_proxy가 weak/no-discrimination evaluator에 지배되지 않아야 함.
- AR-051: fixed FootLock은 artifact proxy를 줄이면서 foot-skate/FID를 악화할 수 있음.

## 다음 작업

AR-040이 끝난 뒤:

- v0 learned ranker 학습
- v0 learned ranker vs M1 oracle gap
- AR-041 target-aware extension
