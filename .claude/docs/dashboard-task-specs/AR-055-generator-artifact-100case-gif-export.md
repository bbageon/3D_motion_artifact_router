# AR-055 — Generator Artifact 100-Case GIF Export

## Goal

AR-054에서 수집한 generator별 artifact 예시 100개를 모두 GIF로 시각화한다.

## Scope

- 입력:
  - `reports/figures/2026-06-19/ar054_generator_artifact_100cases/artifact_100_manifest_motiongpt.json`
  - `reports/figures/2026-06-19/ar054_generator_artifact_100cases/artifact_100_manifest_mdm.json`
  - `reports/figures/2026-06-19/ar054_generator_artifact_100cases/artifact_100_manifest_momask.json`
- 출력:
  - `reports/figures/2026-06-21/ar055_generator_artifact_100case_gifs/`
- 단위: one GIF per selected artifact case.

## Claim Boundary

정성 시각화 자료이다. generator ranking, prevalence estimate, refinement 성능 claim으로 쓰지 않는다.

## Visualization Protocol

- trajectory/world-space motion 사용.
- Y-up renderer.
- robust `ground_y` floor plane 표시.
- foot joints highlight.
- foot trace 표시.
- compact GIF profile 사용: `frame_stride=4`, `max_frames=36`, `fps=8`.

