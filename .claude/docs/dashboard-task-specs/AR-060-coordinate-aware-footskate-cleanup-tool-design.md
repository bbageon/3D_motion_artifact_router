# AR-060 - Coordinate-Aware FootSkateCleanup Tool Design

Status: design-frozen (2026-07-05) — implementation = AR-061  
Epic: Tool  
Priority: Urgent  
Parent: AR-058-4

> **Design freeze 완료**: [frozen implementation spec](../design/coordinate_footskate_cleanup_tool_frozen_spec.md) (interface·algorithm·segment/anchor/blend·guard metric·claim boundary pin). 일지 [reports/2026-07-05b.md](../../../reports/2026-07-05b.md) "Design Freeze" 절. 프로토타입 구현·평가는 AR-061.

## Goal

Design a coordinate-based foot-skate correction tool that is stronger than the current Y-only `FootLockTool` but still does not depend on VisRAG or learned policy.

The immediate goal is **tool design**, not final performance proof.

## Why This Exists

AR-058-4 showed that fixed Y-only FootLock has mixed effects:

- `artifact_total` improves;
- `foot_skate` worsens;
- the effect depends on generator and strength.

Therefore the current tool is not a true foot-skate cleanup tool. It mostly pulls feet down to the ground. Foot skating requires contact-aware X/Z stabilization.

## Current Tool Limitation

Current `FootLockTool`:

```text
new_y = current_y + factor * (ground_y - current_y)
```

It modifies only Y height.

It does not:

- detect valid contact segments;
- decide which foot should be locked;
- stabilize X/Z sliding;
- preserve leg-chain plausibility with IK;
- blend contact enter/exit frames;
- distinguish foot skating from floating or normal stepping.

## Proposed Tool

Working name:

```text
CoordinateFootSkateCleanupTool
```

or:

```text
ContactAwareFootLockTool
```

## Core Idea

Use coordinates to find contact-like foot segments, then lock only those segments.

Pipeline:

```text
generated motion
↓
estimate ground_y
↓
left/right foot contact candidate detection
↓
contiguous contact segment extraction
↓
segment validity filtering
↓
per-segment anchor point selection
↓
X/Z/Y foot locking inside selected segment
↓
ankle/knee/hip leg-chain propagation or IK approximation
↓
entry/exit blending
↓
re-evaluate foot_skate, floating, fidelity, BoneCV
```

## Coordinate Contact Candidate

Start from the v2 foot-skate definition:

```text
contact_t =
  foot_y - ground_y <= 0.05m
  and |vertical_velocity| <= 0.035m/frame

skate_t =
  contact_t
  and horizontal_displacement_xz >= 0.025m/frame
```

Then group consecutive `contact_t` frames into contact segments.

## Segment Validity Filters

A segment is eligible only if:

- segment length >= `min_contact_len`;
- contact ratio inside segment is high enough;
- mean foot height is near ground;
- segment has enough foot-skate signal to justify correction;
- segment is not marked as obvious floating-dominant by later perceptual review, when available.

Initial coordinate-only version does not use text/VisRAG. It should report ambiguous segments rather than forcing correction.

## Anchor Point

For each selected segment, choose a foot anchor in world X/Z.

Options to compare:

1. first contact frame foot X/Z;
2. median X/Z over contact segment;
3. lowest-velocity frame X/Z;
4. weighted average by lower foot height.

Default design: median X/Z, because it is less sensitive to noisy first frames.

Y anchor:

```text
anchor_y = ground_y
```

## Correction

For each frame in the contact segment:

```text
target_foot_xyz =
  (anchor_x, ground_y, anchor_z)

new_foot_xyz =
  current_foot_xyz + u * (target_foot_xyz - current_foot_xyz)
```

where:

```text
u ∈ [0,1]
```

This differs from current FootLock:

| Current FootLock | CoordinateFootSkateCleanup |
|---|---|
| Y only | X/Z/Y |
| all frames | selected contact segments |
| both feet always | selected foot only |
| no segment anchor | segment anchor |
| no blend | entry/exit blend |

## Leg-Chain Propagation

Foot displacement should not be applied only to the foot joint.

Initial deterministic approximation:

```text
foot:   1.0 * delta
ankle:  0.5 * delta
knee:   0.2 * delta
hip:    0.1 * delta
```

This is not full IK. It is a coordinate correction baseline. If it creates pose artifacts, a true IK solver becomes a later task.

## Blending

Apply a temporal window around segment boundaries:

```text
blend_weight = smoothstep(0→1) at enter
blend_weight = smoothstep(1→0) at exit
```

This avoids sudden jumps when the foot becomes locked/unlocked.

## Strength `u`

Use bounded continuous strength:

```text
u ∈ [0,1]
```

For first implementation/evaluation:

```text
u_grid = {0.25, 0.5, 0.75, 1.0}
```

Do not claim a learned optimum yet.

## Evaluation Plan

Primary:

- `foot_skate_world` should decrease on selected foot-skate segments.

Guard metrics:

- `float_mag` must not increase materially;
- `artifact_total` should not worsen;
- `BoneLengthCV` must not spike;
- fidelity/correction magnitude must be reported;
- visual before/after samples must be generated.

Comparison:

1. no correction;
2. current Y-only `FootLockTool`;
3. coordinate-aware cleanup with `u_grid`;
4. optional oracle-best `u` per sample as ceiling.

## Success Criteria

Design-level success:

- Tool interface and correction algorithm are specified.
- Input/output, segment selection, anchor selection, blending, and guard metrics are fixed.
- Claim boundary is clear.

Implementation-level future success:

- average `foot_skate_world` decreases without systematic `float_mag`/BoneCV/FID degradation;
- at least one generator, especially MDM, shows clear improvement over Y-only FootLock;
- prompt-level mixed effects are reported honestly.

## Claim Boundary

Allowed:

> A coordinate-aware foot-skate cleanup tool is a necessary next baseline after Y-only FootLock failed.

Forbidden:

> This design proves foot skating is solved.

> This design replaces human/perceptual validation.

> This design handles text intent or VisRAG verification.

## Research Grounding

- WACV 2020 Footskate Reducer: contact foot points should have near-zero velocity during ground contact.
- GMD ICCV 2023: foot-skating ratio uses near-ground foot height and horizontal skid thresholds.
- PhysDiff ICCV 2023: physical correction must consider constraints and dynamics; naive post-processing can create side effects.

## Next Outputs

1. Implementation spec review.
2. `CoordinateFootSkateCleanupTool` prototype.
3. Small smoke on AR-058-3f high-v2 examples.
4. Representative-pool comparison against current Y-only FootLock.
