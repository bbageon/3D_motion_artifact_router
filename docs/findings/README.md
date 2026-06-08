# docs/findings/ — 발견한 사실·시사점 (index)

> 본 폴더 = 실험으로 **확립된 발견(finding)과 그 연구적 시사점** 의 단일 출처.
> raw 측정 = `evals/snapshots/`, 일자 일지 = `reports/`, 본 폴더 = 그것을 **합성한 결론·함의**.
> 모든 finding 은 evidence cross-link + evidence tier(§3-17) + metric category(§3-20) + 남는 불확실성을 동반한다 (§3-22 4항목).

| 문서 | 주제 | 한 줄 결론 |
|---|---|---|
| [motiongpt_implications.md](motiongpt_implications.md) (F0~F10) | **MotionGPT(G2)에 대한 본 연구의 시사점** | MotionGPT(VQ)는 물리적으로 깨끗(정의 문제 거의 없음, F7) → real refinement headroom 작고 대표 결과 FID NEUTRAL + no-harm(F0); 위반은 의미 축에 집중(F8) → 우리 기하 tool 범위 밖(F9, 메커니즘 F10) → **MotionGPT 단독 필요성 약함, 필요성은 diffusion 에서** |

## 관련

- 데이터셋 카드: [../dataset/motiongpt_g2_pool.md](../dataset/motiongpt_g2_pool.md)
- generator 일반 실패 유형 (문헌): [../generator_failure_mode_survey.md](../generator_failure_mode_survey.md)
- metric 정의·category: [../metric_provenance.md](../metric_provenance.md)
- 연구 position: [../current_research_position.md](../current_research_position.md)
