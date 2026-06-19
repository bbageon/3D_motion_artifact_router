# AR-054 — Generator Artifact 100-Case Library

## Goal

대표 300 prompt × 3 seed × 3 generator pool에서 generator별 artifact 예시 100개를 수집하고, MotionGPT/MDM/MoMask별로 정리한다.

## Scope

- 입력: `external_assets/protocol_rep_pool_seed20260608`
- 출력:
  - `reports/figures/2026-06-19/ar054_generator_artifact_100cases/`
  - `evals/snapshots/generator_artifact_100case_library_v1.json`
- 단위: one generated motion instance = `(generator, sample_id, seed)`

## Claim Boundary

이 작업은 qualitative artifact library 구축이다. generator ranking, refinement 성능, 최종 quality improvement claim은 하지 않는다.

## Selection Rules

1. 기존 representative pool 안에서만 수집한다.
2. generator별 100개를 만든다.
3. motion group quota를 우선하여 walking-only 편향을 피한다.
4. generator별 artifact signature를 반영한다.
5. 같은 prompt에서 seed만 다른 샘플은 가능한 한 중복 선택하지 않는다.
6. MotionGPT length-control failure는 별도 bucket으로 기록한다.

## Expected Outputs

- `artifact_100_manifest_motiongpt.json`
- `artifact_100_manifest_mdm.json`
- `artifact_100_manifest_momask.json`
- `summary.md`
- `generator_artifact_100case_library_v1.json`

