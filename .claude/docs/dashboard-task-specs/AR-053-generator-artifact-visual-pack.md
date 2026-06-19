# AR-053 — Generator Artifact Visual Pack

## 목적

AR-044에서 정량 확인한 generator별 artifact가 실제로 눈에 보이는지 확인하기 위한 정성 시각 자료를 만든다.

## 쉽게 말하면

표 하나로만 "MotionGPT는 발이 뜬다", "MDM은 발이 미끄러진다"라고 말하면 설득력이 약하다.  
그래서 대표 사례를 GIF로 뽑아, 문제를 사람이 직접 확인할 수 있게 만든다.

## 대상

| Generator | Artifact | 선택 기준 |
|---|---|---|
| MotionGPT | FootFloating | FootFloatingEvaluator score 상위 |
| MDM | world foot-skate | `foot_skate_world` 상위 |
| MoMask | FootFloating | FootFloatingEvaluator score 상위 |

## 출력

- `reports/figures/2026-06-19/ar053_generator_artifacts/*.gif`
- `reports/figures/2026-06-19/ar053_generator_artifacts/manifest.json`
- `evals/snapshots/generator_artifact_visual_pack_v1.json`

## Claim Boundary

이 작업은 성능 평가가 아니다.

- generator가 나쁘다는 주장 아님
- refinement 효과 주장 아님
- representative pool에서 관찰된 artifact의 정성 확인용 자료

## 성공 조건

1. 3개 generator 각각 대표 artifact GIF가 생성된다.
2. 각 GIF의 sample_id, prompt, seed, metric score가 manifest에 기록된다.
3. Y-up convention을 지킨다.
4. GIF가 nonblank인지 간단히 확인한다.
5. AR-044의 정량 taxonomy와 연결된다.
