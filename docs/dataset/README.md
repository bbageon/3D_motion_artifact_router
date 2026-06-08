# docs/dataset/ — 데이터셋 카드 (index)

> 본 폴더 = ArtifactRouter 가 사용하는 **데이터셋의 provenance·구성·통계·caveat 단일 출처**.
> 실험 결과의 "무엇을 어디서 어떻게 만든 데이터로 측정했는가" 를 한 곳에 정리한다 (AGENTS.md §3-6/§3-10/§3-17 일관).

| 카드 | 대상 | 한 줄 |
|---|---|---|
| [motiongpt_g2_pool.md](motiongpt_g2_pool.md) | **G2 = MotionGPT 생성 pool** | 본 연구의 primary real-distribution test bed. 600개 실제 MotionGPT 출력 + stress/natural 층화 + clean/synthetic companion |

## 용어

- **G2** — MotionGPT (Jiang et al., NeurIPS 2023) 로 생성한 motion. 본 프로젝트의 active real generator (G1=MDM 미구축, 사용자 별도 확보 예정).
- **real-distribution evidence** (§3-17) — generator 의 natural 출력 (corruption 아님). controlled diagnostic(synthetic) 과 분리 인용.
- **stress / natural** — G2 출력을 artifact 프로파일로 층화한 split. stress=흠집 많은 실제 출력, natural=대표적 출력. **둘 다 실제 MotionGPT 출력** (stress 는 selection 이지 corruption 아님).

관련: 발견·시사점은 [../findings/](../findings/README.md), generator 일반 실패 유형은 [../generator_failure_mode_survey.md](../generator_failure_mode_survey.md), metric 정의는 [../metric_provenance.md](../metric_provenance.md).
