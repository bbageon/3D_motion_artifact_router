# CoordinateFootSkateCleanupTool — Frozen Implementation Spec (AR-060)

Status: **frozen (design)** — 2026-07-05
Parent task: [AR-060](../dashboard-task-specs/AR-060-coordinate-aware-footskate-cleanup-tool-design.md) (Parent AR-058-4)
Deliverable: AR-060 Next Output #1 (implementation spec review). Prototype/구현 = follow-up (AR-061).

> 본 문서 = **구현 전 설계 고정**. AR-058-4 에서 Y-only `FootLockTool` 이 `foot_skate_world` 를 악화시킨(float↓→skate↑ trade-off, 3/3 generator mixed) 결과의 후속. 목표는 성능 증명이 아니라 **interface·correction algorithm·segment/anchor/blend·guard metric·claim boundary 를 pin** 하는 것 (AR-060 Success Criteria design-level). 구현은 본 spec 을 벗어나지 않는다.

---

## 0. 왜 존재하나 (한 줄 비유)

현재 Y-only `FootLockTool` 은 **"미끄러지는 발을 아래로 눌러 바닥에 붙이는"** 도구다. 그런데 발을 바닥에 더 붙일수록(Y↓) 접지 판정 frame 이 늘어나고, 그 접지 중 발이 **수평(X/Z)** 으로는 여전히 움직이므로 `foot_skate` 는 오히려 커진다 — **"바닥에 발을 대고 스케이트를 지치는"** 꼴. 본 tool 은 접지 구간에서 발의 **수평 위치까지 한 점에 고정(anchor)** 해 실제로 미끄러짐을 없앤다. 비유: Y-only = 발을 바닥에 **누르기만**, coordinate-aware = 발을 바닥의 **한 지점에 압정으로 박기**. (비유의 한계: 압정처럼 완전 고정하면 leg-chain 이 부러져 보일 수 있어 `u`(강도)·blend·chain propagation 으로 부분 고정한다.)

---

## 1. 인터페이스 (base.py 준수 — AGENTS.md §3-2)

`correction_tools/coordinate_footskate_cleanup_tool.py`:

```python
class CoordinateFootSkateCleanupTool(CorrectionTool):
    name = "CoordinateFootSkateCleanupTool"

    def __init__(
        self,
        contact_h: float = 0.05,          # 접지 높이 임계 (m, ground 위)
        contact_vy: float = 0.035,         # 접지 수직속도 임계 (m/frame)
        skate_dxz: float = 0.025,          # skate 수평변위 임계 (m/frame)
        min_contact_len: int = 3,          # 유효 segment 최소 frame 수
        min_contact_ratio: float = 0.6,    # segment 내 contact 비율 하한
        min_skate_frac: float = 0.2,       # segment 내 skate 비율 하한 (correction 정당화)
        blend_frames: int = 3,             # enter/exit smoothstep window (frame)
        ground_percentile: float = 10.0,   # estimate_ground percentile (coords_protocol 일치)
    ) -> None: ...

    def apply(
        self, motion, target_part, target_joints, frame_range,
        strength="medium", metadata=None, **kwargs,
    ) -> tuple[np.ndarray, CorrectionReport]: ...
```

`apply()` 시그니처는 base 와 **동일**(변경 없음). 기존 caller/orchestrator 계약 보존.

### 1-1. 인자 재해석 (기존 인터페이스 유지하면서 내부 탐지 추가)

| 인자 | 본 tool 에서의 의미 |
|---|---|
| `motion` | **`motion_trajectory` (world, root 포함)** 여야 함. §1-2 참조. `[T,22,3]`. |
| `target_part` | 후보 발 제한: `left_foot` / `right_foot` / `both_feet`. `both_feet` = tool 이 segment 별로 skating foot 자동 선택. |
| `target_joints` | 명시되면 우선 (예: `["LEFT_FOOT"]`). |
| `frame_range` | tool 이 **동작 허용된 window** `[start,end]`. 이 안에서만 contact segment 탐지·보정. 전체 = `[0,T-1]` → 전 구간 자동. |
| `strength` | token → `u` fallback (§4). |
| `metadata` | `ground_y`(float, 없으면 estimate_ground), `continuous_u`(∈[0,1], 있으면 우선), `coord_space`(기본 `"trajectory"`). |

### 1-2. 좌표계 전제 (중요 — Y-only 와의 결정적 차이)

Y-only FootLock 은 Y 만 건드려 좌표계에 비교적 둔감했다. 본 tool 은 **수평(X/Z) anchor** 를 쓰므로 반드시 **trajectory(world) 표현** 에서 동작해야 한다 — `motion_local`(PELVIS=원점) 은 root 이동이 제거돼 X/Z anchor 가 무의미하다. (비유: 지도 위에서 발자국을 고정하려면 "세계 좌표"가 필요하다. 각 사람 기준 상대좌표에서는 "발이 몸 대비 어디" 만 알고 "땅 위 어디"는 모른다.)

- 전제: `metadata.coord_space != "trajectory"` 이면 `apply()` 는 `ValueError` (silent 오적용 방지).
- ground = `metadata.ground_y` 우선, 없으면 `estimate_ground(motion, ground_percentile)` (coords_protocol, lower-foot-height 10th pct — NOT min-Y).

---

## 2. 파이프라인 (pin)

```
motion_trajectory (world) + frame_range
  ↓ ground_y = metadata.ground_y or estimate_ground(motion)
  ↓ per foot ∈ eligible(target_part): contact_t, skate_t  (v2 정의, §3)
  ↓ contact_t 를 contiguous segment 로 grouping (gap ≤ 1 frame 허용)
  ↓ frame_range 로 clip
  ↓ segment validity filter (§3-1) → {correct, ambiguous, skip}
  ↓ per correct-segment: anchor = (median_x, ground_y, median_z)  (§3-2)
  ↓ per frame in segment: X/Z/Y lock with u·blend_w  (§3-3)
  ↓ leg-chain propagation foot/ankle/knee/hip = 1.0/0.5/0.2/0.1 · delta  (§3-4)
  ↓ CorrectionReport (segment trace + correction_magnitude)
```

---

## 3. 알고리즘 (pin)

### 3-1. Contact / skate 판정 + segment (v2, AR-058-3f/g 와 동일 상수)

발 `f`, `fy = motion[:,f,1] - ground_y`:
```
vy_t      = |diff(fy)|                          # 수직속도 (마지막 frame pad=0)
dxz_t     = ||diff(motion[:,f,[0,2]])||          # 수평변위 (마지막 frame pad=마지막값)
contact_t = (fy <= contact_h) & (vy_t <= contact_vy)
skate_t   = contact_t & (dxz_t >= skate_dxz)
```
`contact_t` 를 연속 구간으로 묶되 **1-frame gap 은 같은 segment** 로 이어 붙인다(짧은 노이즈 흔들림 흡수). 각 segment 를 `frame_range` 로 clip.

**Segment validity filter** — segment 를 `correct` 로 인정하는 조건 (전부 충족):
- `len >= min_contact_len` (3)
- `mean(contact_t in seg) >= min_contact_ratio` (0.6)
- `mean(fy in seg) <= contact_h` (평균 발높이 접지 근처)
- `skate_frac = mean(skate_t in seg) >= min_skate_frac` (0.2) → **미끄러짐 신호가 있어야 보정** (깨끗한 정상 접지는 건드리지 않음)

분류:
- 위 전부 충족 → `correct`
- contact 는 유효하나 `skate_frac ∈ (0, min_skate_frac)` (애매) → **`ambiguous` — 보정 안 함, report 만** (AR-060 spec: "report ambiguous rather than forcing correction")
- 그 외 → `skip`

### 3-2. Anchor (pin: **median X/Z**)

correct-segment 의 contact frame 집합에 대해:
```
anchor_x = median(motion[contact_frames, f, 0])
anchor_z = median(motion[contact_frames, f, 2])
anchor_y = ground_y
```
median 선택 이유: 첫 frame(옵션 1)·최저속 frame(옵션 3) 대비 **노이즈에 둔감**(AR-060 spec default). 옵션 1/3/4 는 향후 ablation 후보(§7 미해결).

### 3-3. Correction (X/Z/Y, u·blend)

segment 내 frame `t`, 발 현재 `p_t = motion[t,f,:]`:
```
target_t   = [anchor_x, ground_y, anchor_z]
blend_w(t) = smoothstep on enter/exit over blend_frames (interior=1.0)   # §3-5
delta_t    = u * blend_w(t) * (target_t - p_t)                            # 3-vector
motion[t,f,:] += delta_t
```
- `u ∈ [0,1]` (§4). `blend_w` 로 경계에서 급격한 점프 방지.
- Y-only 와의 차이 요약:

| 항목 | Y-only FootLock | CoordinateFootSkateCleanup |
|---|---|---|
| 축 | Y only | **X/Z/Y** |
| 대상 frame | frame_range 전체 | **탐지된 skate contact segment 만** |
| 대상 발 | 항상 both (or part) | **segment 별 skating foot** |
| anchor | 없음 (ground_y 만) | **segment median X/Z** |
| blend | 없음 | **enter/exit smoothstep** |

### 3-4. Leg-chain propagation (deterministic, **NOT IK** — engineering heuristic)

발만 옮기면 다리가 늘어난다. 같은 다리 chain 에 delta 를 감쇠 전파:
```
foot:  1.0 * delta_t   (이미 적용)
ankle: 0.5 * delta_t
knee:  0.2 * delta_t
hip:   0.1 * delta_t
```
Left foot → `LEFT_FOOT/LEFT_ANKLE(7)/LEFT_KNEE(4)/LEFT_HIP(1)`, Right → `RIGHT_FOOT/RIGHT_ANKLE(8)/RIGHT_KNEE(5)/RIGHT_HIP(2)`. **이건 true IK 가 아니라 좌표 근사**(§7-0-D engineering heuristic). pose artifact(BoneLengthCV spike) 유발 시 → 진짜 IK solver 는 별도 후속 task. guard metric 으로 감시(§5).

### 3-5. Blend (smoothstep)

`w = 3s^2 - 2s^3`, `s = clip(offset/blend_frames, 0, 1)`. enter: segment 시작 후 `blend_frames` 동안 0→1. exit: segment 끝 전 `blend_frames` 동안 1→0. 나머지 interior = 1.0. segment 가 `2*blend_frames` 보다 짧으면 삼각형(peak<1) 로 clamp.

---

## 4. Strength `u` (bounded continuous)

`u ∈ [0,1]`. 우선순위: `metadata.continuous_u` → 없으면 token mapping.

```
STRENGTH_U = {                       # 3-level (기존 FootLock 패턴과 정렬)
    "small": 0.25, "medium": 0.5, "large": 0.75,
    "xsmall": 0.2, "small5": 0.4, "medium5": 0.6, "large5": 0.8, "xlarge": 1.0,   # 5-level
}
```
평가용 grid: `u_grid = {0.25, 0.5, 0.75, 1.0}`. **learned optimum 주장 금지**(§6). `action_space_type` 기록 = `bounded_continuous_u`, mapper version 명시 (AGENTS.md §3-21).

---

## 5. Report / guard metric 계약

`CorrectionReport`:
- `modified_joints`: correct-segment 에서 실제 수정된 joint 이름 union.
- `correction_magnitude`: 보정된 발 frame 의 mean `||delta_t||` (m).
- `metadata`: `{ground_y, coord_space, u, u_source(continuous|token), n_correct, n_ambiguous, n_skip, segments:[{foot,start,end,skate_frac,anchor_xz}]}`.

**Guard metric** (tool 이 아니라 평가 harness 가 before/after 로 계산 — AR-061):
| metric | 방향 | Category |
|---|---|---|
| `foot_skate_world` (GMD/EDGE) | **감소 (primary)** | B (physical) |
| `float_mag` | 유의미 증가 금지 | B |
| `artifact_total` | 악화 금지 | **C** (internal proxy) |
| `BoneLengthCV` | spike 금지 | B |
| correction_magnitude / MPJPE | 보고 | B |
| FID (equal-N) | 보고, 악화 감시 | **A** |

Category 표기 = metric_provenance.md 준수. `artifact_total`/NetGain 은 C 라 **외부 최종성능 근거 단독 금지**(AGENTS.md §3-20).

---

## 6. 평가/비교 protocol (AR-061 에서 실행)

Representative pool (`protocol_rep_pool_seed20260608`) 에서 generator(MDM/MoMask/MotionGPT)별:
1. no correction (baseline)
2. Y-only `FootLockTool` (both_feet, ground_y=estimate_ground) — **AR-058-4 재현 조건**
3. `CoordinateFootSkateCleanupTool`, `u_grid={0.25,0.5,0.75,1.0}`
4. (선택) oracle-best `u` per sample = ceiling

통계: paired Wilcoxon signed-rank + bootstrap CI, **n≥20 / snapshot≥2** (AGENTS.md §3-9, §6-7). 3-tier evidence 표기(§3-17): representative = real-distribution. prompt-level mixed effect **정직 보고**(어떤 prompt 는 개선, 어떤 건 악화). 결과는 `evals/reports/` + `reports/<date>.md` cross-link.

---

## 7. Claim boundary (AR-060 spec)

허용:
> Y-only FootLock 실패 후 **coordinate-aware cleanup 이 필요한 다음 baseline** 이다.

금지:
> foot skating 이 해결됐다 / 본 설계가 perceptual 검증을 대체한다 / text intent·VisRAG 를 처리한다.

---

## 8. Research grounding (AGENTS.md §3-22)

1. **판단/권고**: 접지 중 발의 **near-zero horizontal velocity** 원칙으로 X/Z anchor lock 을 설계하고, naive post-processing 의 부작용을 guard metric + 정직한 mixed-effect 보고로 방어.
2. **근거 논문**:
   - **GMD — Karunratanakul et al., ICCV 2023** (Guided Motion Diffusion): foot-skating ratio = near-ground foot height + horizontal skid threshold → 본 tool 의 `contact_h`/`skate_dxz` metric 정의 근거 (2020+ peer-reviewed top-tier). ✓
   - **PhysDiff — Yuan et al., ICCV 2023**: 물리 보정은 제약/동역학을 고려해야 하며 **naive post-processing 은 side effect** 유발 → guard metric 의무 + mixed-effect 정직 보고 근거 (2020+ peer-reviewed). ✓
   - **Footskate cleanup — Kovar, Schreiner, Gleicher, SCA 2002** (배경 only): 접지 발의 zero-velocity lock 원칙의 고전 출처. **2020 이전 → 배경 설명 한정, 새 평가 기준 단독 근거 사용 금지**(§7-0-A.4).
3. **ArtifactRouter 적용 범위**: `correction_tools/` 의 coordinate 기반 foot-skate cleanup tool 1종 설계. orchestrator·evaluator 인터페이스 변경 없음.
4. **남는 불확실성 (§7-0-D)**:
   - leg-chain propagation weight (1.0/0.5/0.2/0.1), `blend_frames=3`, gap≤1 merge, filter 임계(0.6/0.2) = **engineering heuristic** (논문 근거 아님). 외부 claim·hypothesis 전환 단독 근거 금지.
   - anchor=median 은 default 선택 — 옵션 1(first)/3(min-vel)/4(weighted) ablation 미실시.
   - AR-060 원 spec 이 인용한 "WACV 2020 Footskate Reducer" 는 **본 실행에서 출처 검증 불가 → 근거에서 제외**(오인용 방지, §3-22). 위 3개만 근거로 사용.

---

## 9. 미해결/후속 (AR-061 이후)

- 프로토타입 구현 + `tests/unit/correction_tools/` round-trip·property(target skate 단조 감소) test.
- orchestrator scoring 에 tool 등록 (AGENTS.md §4 `correction_tools/<name>_tool.py` → base+구현체+scoring 동시 갱신 + integration smoke).
- AR-058-3f high-v2 예시 smoke → representative-pool 비교(§6).
- anchor/propagation ablation, 필요 시 true IK.
