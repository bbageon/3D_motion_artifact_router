# AR-043 — Dataset Predefined Issue Prevalence Audit

> ⚠️ **Retroactive caveat (2026-07-07, AR-062/AR-063)**: 본 audit 의 **Skate/Penetrate gate 컬럼 (v0.1.0)** 은 구조적 vacuity (내부 contact/ground 추정으로는 정의상 fire 불가 — [audit A-1/A-2](../findings/artifact_tool_alignment_audit.md)) 로 "없음"의 증거가 아님. 유효한 유병률 = 수리(v0.2.0) 후 [gate_prevalence_remeasure_ar063_v1](../../../evals/snapshots/gate_prevalence_remeasure_ar063_v1.json). foot_skate_world 등 Category-B 결론은 영향 없음.

## 한 줄 목표

AR-040 의 `q_proxy` / v0 ranker 실험 전에, 현재 학습·평가 데이터셋 안에 ArtifactRouter 가 사전에 정의한 문제가 실제로 존재하는지 정량 확인한다.

## 배경

Effect-Aware Ranker 를 학습하기 전에 먼저 확인해야 할 전제는 다음이다.

```text
우리가 고치려는 artifact / physical issue 가 데이터셋에 실제로 존재하는가?
```

이 전제가 약하면 q_proxy, oracle, learned ranker 를 설계해도 실험이 헛돌 수 있다.

## 확인할 issue family

| family | evaluator |
|---|---|
| artifact proxy | `FootFloatingEvaluator`, `BoneLengthEvaluator`, `VelocityJitterEvaluator` |
| physical gate | `PenetrateEvaluator`, `FloatEvaluator`, `SkateEvaluator`, `JerkSpikeEvaluator`, `BoneLengthCVEvaluator` |

## 확인할 dataset / split

우선순위는 real-distribution evidence 를 먼저 본다.

| tier | 대상 |
|---|---|
| real generator | G2 pool / g2_stress_holdout / g2_natural_holdout |
| no-harm reference | clean_noharm_holdout |
| controlled diagnostic | synthetic_aux / synthetic_diag |

## Audit 질문

1. 각 split 에서 issue score 분포가 어떤가?
2. issue 가 거의 없는 split 과 많은 split 이 구분되는가?
3. `g2_stress` 는 실제로 artifact-rich 인가?
4. `g2_natural` / `clean_noharm` 은 no-harm 평가에 적절한가?
5. 어떤 evaluator 가 신뢰 가능한 signal 을 주고, 어떤 evaluator 는 약한가?

## 성공 조건

1. 각 dataset/split 별 evaluator score summary 를 만든다.
2. predefined issue 의 prevalence 를 보고한다.
3. `stress` / `natural` / `clean` 구분이 evaluator 기준으로 성립하는지 확인한다.
4. 신뢰 약한 evaluator 는 limitation 으로 표시한다.
5. AR-040 진행 여부를 결정한다.

## Claim Boundary

이 작업은 정책 성능을 증명하지 않는다. 데이터셋에 연구 문제가 실제로 존재하는지 확인하는 전제 검증이다.
