# AR-052 — v0 Learned Ranker Train/Eval

## 목적

AR-040에서 만든 `q_proxy_v0.1`과 M1 dense effect oracle을 학습기가 얼마나 근사할 수 있는지 검증한다.

쉽게 말하면:

> 모든 tool을 실제로 다 적용해보고 고르는 M1 oracle을, v0 state만 본 learned ranker가 얼마나 따라잡는가?

## Claim Boundary

이 작업은 v0 learned policy의 첫 일반화 테스트다.

- target-aware claim 없음
- final motion quality claim 없음
- q_proxy 기준의 offline ranking 성능만 평가
- final Category-A/FID 검증은 별도 작업

## 입력

- AR-040 compact snapshot: `evals/snapshots/effect_aware_v0_oracle_v1.json`
- 필요 시 raw transition 재생성:
  `python tools/effect_aware_v0_oracle.py --raw-output evals/raw/effect_aware_v0_oracle_v1_full.json`

## Split 원칙

통계 단위는 prompt다.

- 같은 prompt의 3 seed transition이 train/test에 동시에 들어가면 leakage.
- generator별 split과 pooled split을 둘 다 보고한다.
- train/test는 prompt ID 기준으로 나눈다.

## 모델 후보

- RandomForestRegressor / HistGradientBoostingRegressor for `q_proxy`
- optional pairwise/ranking loss는 다음 단계

## 평가 지표

- q_proxy R2 / MAE
- safe top-1 match: learned best action이 M1 best action과 같은가
- top-k oracle inclusion
- regret: M1 best q_proxy - learned-selected q_proxy
- false-improve: learned가 STOP보다 낫다고 골랐지만 실제 q_proxy <= 0
- STOP calibration: q_proxy threshold별 act-rate / regret trade-off

## 성공 조건

1. prompt-level split이 snapshot metadata에 기록된다.
2. learned ranker가 random action보다 낮은 regret을 보인다.
3. false-improve rate가 보고된다.
4. v0의 한계, 특히 target/semantic 미관측으로 인한 failure가 정직하게 보고된다.

## 다음 작업

AR-052가 충분히 작동하면 AR-041 target-aware ranker로 넘어간다. 작동하지 않으면 v0 state/q_proxy calibration부터 수정한다.
