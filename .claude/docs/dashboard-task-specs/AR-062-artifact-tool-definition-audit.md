# AR-062 - Artifact-Tool Definition Audit

Status: completed (2026-07-07)  
Epic: Tool  
Priority: Urgent  
Parent: AR-058 / AR-060

> **결과**: [artifact_tool_alignment_audit.md](../findings/artifact_tool_alignment_audit.md) (matrix + A-1~A-6 발견) · [snapshot](../../../evals/snapshots/artifact_tool_alignment_audit_ar062_v1.json) (vacuity controlled-diagnostic 재현). **Verdict: AR-061 PROCEED** (foot-skate 만 metric·tool 정합 성립); gate evaluator 수리(Skate/Penetrate vacuity + BoneCV 무판별 + floating 정의 통일) = **AR-063 신설**; AR-058-5 는 gate-fire 컬럼 근거 사용 금지.

## Goal

Before implementing more correction tools, audit whether each artifact definition is matched with an appropriate correction-tool mechanism.

This follows the foot-skate lesson:

```text
artifact = horizontal foot sliding during contact
old tool = Y-only foot lowering
result = artifact/tool axis mismatch
```

The goal is to avoid repeating this mismatch for BoneCV, floating, penetration, jitter, and other artifact categories.

## Why This Exists

P3 and P4 are not only evidence collection steps. They are also refining:

1. what each artifact actually means;
2. whether the current evaluator measures that artifact;
3. whether the proposed tool corrects the same failure mode;
4. which side effects must be guarded.

If artifact and tool definitions are wrong, routing or RL will learn the wrong action space.

## Scope

Audit these artifact groups:

| Artifact | Current status |
|---|---|
| Foot skating / contact drift | v2 coordinate definition exists; current Y-only FootLock mismatched; AR-060 proposes coordinate-aware cleanup |
| Floating / missing contact | measured by FootFloating/float_mag; current FootLock partly aligned but can create skating |
| Ground penetration | candidate metric exists but current dedicated correction tool is missing |
| BoneLengthCV | evaluator and BoneProjection exist; protocol verified, perceptual claim still cautious |
| Temporal jitter / jerk | evaluator and VelocitySmoothing exist; should be secondary temporal smoothness artifact, not primary physical artifact |
| Semantic mismatch / intent conflict | final validation/future verifier, not coordinate correction target yet |

## Audit Questions

For each artifact:

1. **Definition**: What is the artifact in plain language?
2. **Coordinate metric**: Which evaluator/metric measures it?
3. **Metric caveat**: Is it a candidate metric or a perceptual artifact?
4. **Tool mapping**: Which current/future tool is supposed to correct it?
5. **Mechanism match**: Does the tool modify the same variable the artifact is about?
6. **Side effects**: Which metrics can get worse?
7. **Evidence status**: completed / partial / missing.
8. **Next action**: implement, redesign, or hold.

## Expected Output

Create an artifact-tool matrix:

```text
artifact
→ definition
→ metric
→ tool
→ mechanism match
→ known failure
→ guard metrics
→ next task
```

This matrix should decide whether AR-061 is still the right next implementation target or whether another artifact/tool mismatch is more urgent.

## Claim Boundary

Allowed:

> The project has audited artifact-tool alignment and identified which correction tools are valid baselines versus which need redesign.

Forbidden:

> The audit proves a tool improves motion quality.

> The audit replaces per-tool empirical evaluation.

## Research Grounding

Use metric/tool grounding only where appropriate:

- GMD ICCV 2023: foot-skate metric grounding.
- PhysDiff ICCV 2023: generated motions can show floating, foot sliding, and penetration; physical constraints matter.
- HuMoR ICCV 2021 / related physical plausibility work: bone length and body consistency are important plausibility constraints.
- Temporal smoothness metrics are engineering/proxy unless tied to perceptual or standard-quality validation.

If a mapping is an engineering heuristic, label it as such.

## Success Criteria

- All current artifact groups have a reviewed artifact-tool row.
- Foot skating remains mapped to AR-061 only if the coordinate-aware cleanup mechanism is judged aligned.
- Any weak or mismatched tool is renamed/reclassified before implementation.
- AR-058-5 synthesis is blocked from making broad routing claims until this audit is reflected.
