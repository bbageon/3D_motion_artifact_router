# AR-058-5 - P5 Routing Necessity Synthesis

Status: ready  
Epic: Evidence  
Priority: Urgent  
Parent: AR-058

> **2026-07-08 제약 2건**: (1) [AR-062 audit](../findings/artifact_tool_alignment_audit.md) — v0.1.0 gate-fire 컬럼 인용 금지 (v2 수치만). (2) 주장 구조는 [확정 연구 가닥 §0-0](../governance/current_research_position.md) 의 **1 primary + 4 보조 pair** 를 따른다 — primary(foot skating) 만 완주 사슬로 주장하고, 보조/보류 pair 는 측정된 caveat 와 함께 표기. perceptual A/B (primary 한정) 결과에 따라 어조 확정.

## Claim

P1-P4 together justify artifact-aware, state-conditioned routing.

## Why This Step Exists

이 단계는 새로운 실험을 많이 추가하는 단계가 아니라, P1-P4 결과를 논문 도입부 논리로 묶는 단계다. 최종 산출물은 evidence chain, 4-panel figure, claim boundary다.

## Input Evidence

| Step | Input |
|---|---|
| AR-058-1 | Artifact occurrence |
| AR-058-2 | Common coordinate measurement |
| AR-058-3 | Artifact-quality relation |
| AR-058-4 | Fixed tool insufficiency |

## Synthesis Logic

```text
1. Generated motions have artifact candidates.
2. Heterogeneous generators can be compared in canonical coordinate space.
3. Artifact candidates can degrade quality under intent-aware interpretation.
4. Fixed correction has mixed effects and side effects.
5. Therefore, refinement should be state-conditioned rather than fixed.
```

## Outputs

1. `evals/reports/<date>_poc_necessity_p1_p5.md`
2. `evals/snapshots/poc_necessity_p1_p5_v1.json`
3. `reports/figures/<date>/poc_necessity_p1_p5/`
4. 4-panel figure:
   - P1 artifact prevalence
   - P2 generator-specific artifact profile
   - P3 artifact-quality relation
   - P4 fixed tool mixed effect
5. One-page introduction evidence summary.

## Success Criteria

- The report can support the paper introduction claim without relying on RL policy performance.
- Claim boundaries are explicit: PoC necessity, not final policy superiority.

## Claim Boundary

가능한 주장:

> The observed artifact occurrence, shared coordinate measurability, quality relation, and fixed-tool side effects motivate artifact-aware state-conditioned routing.

금지 주장:

> ArtifactRouter already solves all generator artifacts or guarantees large quality gains across all generators.

