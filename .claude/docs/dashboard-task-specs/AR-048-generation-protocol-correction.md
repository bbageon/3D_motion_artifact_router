# AR-048 — Generation Protocol Correction (큰 pool 확장 전 freeze)

## 한 줄 목표

AR-022(MDM)/AR-045(MoMask) 의 큰 pool 로 확장하기 **전에**, generation·비교 protocol 을 표준화·고정한다. 현재 n=50 smoke 의 결함을 교정해, 이후 3-generator pool 이 공정·재현·물리측정-타당하게 만들어지도록 한다.

## 표현 통일 원칙

세 generator의 모션을 한 좌표 표현으로 억지 변환하지 않는다. 각 generator가 만든 **동일한 모션 한 개**를 아래 두 표현으로 함께 보존한다.

| 저장 필드 | 쉬운 의미 | 생성 방법 | 주 사용처 |
|---|---|---|---|
| `motion_trajectory` | 사람이 공간에서 어디로 이동했는지 포함한 모션 | generator 복원 결과에서 PELVIS 이동을 제거하기 **전** 저장 | foot skate, contact, root path, ground 관계 |
| `motion_local` | 사람이 제자리에서 어떤 자세를 취했는지 보는 모션 | `motion_trajectory - motion_trajectory[:, PELVIS:PELVIS+1, :]` | canonical tool/policy 입력, bone, local pose, joint dynamics |

두 표현은 모두 `[T,22,3]`, fps=20, 동일 SMPL-22 joint order, 동일 axis convention, meter 단위를 사용한다. AGENTS.md §3-1의 canonical motion은 `motion_local`이며, `motion_trajectory`는 trajectory-sensitive 평가를 위한 보조 표현이다.

generator별 변환은 다음으로 고정한다.

1. **MotionGPT**: 현재 root trajectory가 남은 복원 결과를 `motion_trajectory`로 저장하고, 여기서 `motion_local`을 파생한다.
2. **MDM/MoMask**: 복원 직후 수행하던 root 제거를 저장 이전에 하지 않는다. 먼저 `motion_trajectory`를 저장하고, 같은 식으로 `motion_local`을 파생한다.
3. 이미 root를 제거한 `motion_local`에서 trajectory를 추정하거나 재구성하지 않는다. 삭제된 이동 정보는 복원할 수 없기 때문이다.

이 단계는 motion을 smoothing하거나 보정하는 전처리가 아니다. generator 간 비교 전에 **동일한 joint/axis/unit/length 계약으로 표현과 provenance를 맞추는 adapter protocol**이다.

## 배경 (왜 지금 freeze 하나)

n=50 preliminary smoke ([generator_prevalence_smoke_{mdm,momask}_v1](../../../evals/snapshots/)) 는 **방향 탐색**이며 **generator 우열을 확정하지 않는다** (n 작음·protocol 미교정). 다음 결함이 공정 비교·물리 측정 타당성을 저해한다:

- `n_frames=40` 고정인데 GT/G2 는 가변(52~196) → artifact 과소측정 + 길이 불일치(불공정 비교).
- **root-relative 만 저장** ([_mdm_inference.py:151](../../../generators/_mdm_inference.py) `motion - motion[:,0:1,:]`) → foot skate/penetrate 가 전역 trajectory 없이 왜곡 (AR-029 caveat, ground=minY degenerate).
- 단일 seed → diffusion(MDM)의 확률성 미반영.
- 미러 중복·동일 문장 중복·부분(segment) 캡션·GT 부재 가능성 미점검.

## 고정할 protocol 항목

### A. 교정 prompt bank
1. **HumanML3D test split 사용** — 공식 `test.txt`.
2. **full-motion caption만 선택** — `texts/<id>.txt` 줄 `caption#pos#start#end` 의 frame-range 가 **전체 모션**(관례상 `0.0 0.0`)인 캡션만. 부분(segment) 캡션 제외.
3. **원본·미러 중복 제거** — `M` prefix(좌우 반전 증강)와 원본 중 하나만.
4. **GT 파일 존재 확인** — `external_assets/HumanML3D/new_joints/<id>.npy` 실재. 없으면 제외.
5. **동일 문장 중복 제거** — 캡션 텍스트 정규화 후 중복 prompt 제거.
6. **GT 길이 필터 + target_length 규칙** (GT 길이를 그대로 쓰지 않음):
   - **eligibility**: `40 ≤ GT_length < 200` 인 prompt 만.
   - **target_length = min(4 × floor(GT_length / 4), 196)** (4의 배수, 최대 196 frame).
7. **기록**: `sample_id`, `prompt`, `gt_length`, `target_length`, `gt_path`.

### B. 좌표계 출력 보강 (생성 시점)
8. **`motion_trajectory`** — root(PELVIS) 이동을 유지한 trajectory-preserving 표현. generator 복원 결과를 root 제거 **전**에 저장한다.
9. **`motion_local`** — `motion_trajectory`에서 frame별 PELVIS 위치를 빼서 파생한 root-relative canonical 표현. policy/tool의 기본 motion interface는 이를 유지한다.
10. **평가 분리**: **foot skate·contact·root 이동·ground 관계 = `motion_trajectory`** 에서 / **local pose·bone artifact·joint dynamics = `motion_local`** 에서 측정한다. evaluator record에 사용한 `motion_representation`을 기록한다.
11. **ground 기준 = min-Y 금지** — 별도 기준 확정 (예: trajectory-preserving 의 contact-window 기반 robust 추정). spec 구현에서 단일 출처로 고정·문서화.

### C. 통계·seed 규약
12. **통계 단위 = prompt** (n=prompt 수). **seed 는 반복 생성**(generator 확률성 평균용, §5 N≥3), 독립 표본 단위 아님 → seed 평균 후 prompt 단위 집계.

### D. 비교 규약
13. **같은 prompt끼리 paired** — cross-generator 비교는 동일 prompt(=동일 GT, 동일 target_length) index 기준 (§6-10). generator 별 출력은 **서로 다른 모션**.

## 소규모 protocol 검증

**30 prompts × 3 seeds × 3 generators = 270 모션.** 동일 prompt·target_length 사용, generator별 출력은 상이.

## 7개 성공 조건 (검사)

1. **prompt/GT 정렬** — 각 모션이 올바른 prompt·GT 에 매핑.
2. **실제 생성 길이 일치** — 생성 frame 수 = target_length.
3. **generator별 seed 3개 존재** — 각 (generator, prompt) 에 seed 3.
4. **trajectory/local 표현 모두 존재·정합** — `motion_trajectory`와 `motion_local`이 모두 저장되고, `motion_local == motion_trajectory - pelvis`가 tolerance 내에서 성립.
5. **generator별 디렉터리 분리** — `generator_id` 별 분리 (§3-5).
6. **metadata 및 checkpoint hash 존재** — sample meta + `generator_class_hash`.
7. **동일 prompt 기준 paired 비교 가능** — prompt index 정렬 키로 3-way 정렬.

## AR-048 freeze

7개 성공 조건 통과 시 **freeze snapshot + 재현 명령** 기록 → AR-048 Done. **본 단계에서 generator 성능 우열을 결론내리지 않는다.**

## 이후 (AR-022 → AR-045 재개, freeze 후)

MDM 큰 pool + profile → MoMask 큰 pool + profile → 같은 protocol 로 MotionGPT 포함 **3-way 비교** → 실제 artifact prevalence + tool headroom 확인.

## Claim Boundary

policy/generator 성능 증명 아님. 확장 전 generation·비교 protocol 표준화·고정. 이 변환 자체를 quality improvement 또는 refinement로 주장하지 않으며, 소규모 결과로 generator 우열 결론 금지.

## 근거

- **HumanML3D** (Guo et al., CVPR 2022): caption/split/GT joints, 미러 증강, frame-range 注釈 (A).
- **§5 generator 비결정성** → N≥3 (C).
- **§3-1 canonical root-relative** + **AR-029 caveat** (foot_skate ground/trajectory 민감 → world-space 필요) (B).
- **§6-10 paired cross-generator** (D).
- **PhysDiff** (Yuan ICCV 2023): physical plausibility 는 world-space contact 기반이 타당 (B-10·11).
