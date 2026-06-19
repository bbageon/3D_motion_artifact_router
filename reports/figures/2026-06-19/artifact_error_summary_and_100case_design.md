# Generator Artifact Error Summary and 100-Case Design

작성일: 2026-06-19  
대상 폴더: `reports/figures/2026-06-19`  
기준 시각화 팩: `reports/figures/2026-06-19/ar053_generator_artifacts/manifest.json`

## 1. 목적과 claim boundary

이 문서는 생성 모션 프레임워크별로 어떤 artifact가 눈에 보이는지 정리하고, generator별 artifact 예시 100개를 어떻게 구성할지 사전에 고정하기 위한 설계 문서다.

중요한 제한:

- 현재 3개 GIF는 정성적 오류 설명용이다.
- 이 3개만으로 generator 성능 순위나 refinement 성능을 주장하지 않는다.
- 100개 예시 세트도 최종 성능 지표가 아니라, 문제 유형을 보여주는 qualitative evidence library다.
- 최종 성능 주장은 별도 metric, paired test, Category-A 지표, perceptual 평가와 함께 해야 한다.

## 2. 현재 확인된 대표 오류

| generator | 대표 오류 | 현재 예시 prompt | 현재 GIF |
|---|---|---|---|
| MotionGPT | FootFloating | `a person takes a small hop forward.` | `ar053_generator_artifacts/motiongpt_footfloating_005803_seed20386608.gif` |
| MDM | World FootSkate | `a person walks quickly forward, moving at a slight angle to the right` | `ar053_generator_artifacts/mdm_world_footskate_012495_seed20519610.gif` |
| MoMask | FootFloating | `a person holds their right arm out on something to support them while sticking their right leg up to balance.` | `ar053_generator_artifacts/momask_footfloating_010843_seed20493610.gif` |

### 2.1 FootFloating

의미:

발이 땅에 닿아야 자연스러운 구간에서 발이 떠 있거나, 착지와 접촉이 불안정하게 보이는 오류다.

눈으로 보는 포인트:

- 점프 후 착지했는데 발이 바닥에 제대로 붙지 않는다.
- 한 발로 버티는 동작에서 지지 발이 공중에 떠 있는 것처럼 보인다.
- 발 접촉 타이밍과 몸의 무게 이동이 맞지 않는다.

주로 잘 보이는 동작:

- jump, hop, landing
- one-leg balance
- kick, lunge, squat
- sit/stand transition

현재 대표 사례:

- MotionGPT: 작은 hop forward 동작에서 footfloating score가 높음.
- MoMask: 한 발을 들고 균형을 잡는 support pose에서 footfloating score가 높음.

### 2.2 World FootSkate

의미:

발이 땅에 닿아 고정되어야 하는데, world-space 기준으로 바닥 위를 미끄러지듯 움직이는 오류다.

눈으로 보는 포인트:

- 걷기나 방향 전환 중 접지 발이 바닥 위에서 밀린다.
- 몸은 앞으로 이동하지만 발 접촉이 물리적으로 설득력 있게 고정되지 않는다.
- local pose만 보면 티가 덜 나도, trajectory를 보존한 world-space에서는 미끄러짐이 드러난다.

주로 잘 보이는 동작:

- walking forward
- fast walking
- circle walking
- turn, pivot
- run, jog

현재 대표 사례:

- MDM: 빠르게 앞으로 걷는 동작에서 foot_skate_world가 크게 나타남.

### 2.3 Length-Control Failure

의미:

요청한 길이와 실제 생성 길이가 크게 다른 오류다. 이건 관절 위치 artifact라기보다 생성 프로토콜 및 temporal completeness 문제다.

눈으로 보는 포인트:

- 같은 prompt인데 seed에 따라 길이가 크게 달라진다.
- 어떤 경우에는 극단적으로 짧은 모션이 나온다.
- 동작이 완성되기 전에 끝나거나, 평가 길이와 맞지 않아 비교가 어려워진다.

현재 상태:

- AR-050에서 MotionGPT의 length-control failure가 확인됨.
- AR-053의 3개 GIF 중 핵심 시각 오류로 넣지는 않았지만, MotionGPT artifact library에는 별도 유형으로 포함한다.

### 2.4 BoneLengthCV

의미:

frame이 바뀌면서 같은 bone의 길이가 흔들리는 물리적 오류다.

주의:

- BoneLengthCV는 연구적으로 중요하지만, 단일 GIF만으로 일반 사용자가 바로 인지하기 어려울 수 있다.
- 따라서 100개 시각 예시에서는 주 artifact로 쓰기보다, metadata와 appendix 분석에 넣는 것이 안전하다.

## 3. 100개 artifact 예시 구성 원칙

100개 예시는 각 generator마다 따로 만든다.

단위:

- 1개 예시 = `generator + sample_id + seed`로 정해지는 하나의 motion instance.
- 같은 prompt에서 seed만 다른 샘플은 원칙적으로 중복 선택하지 않는다.
- 단, length-control failure처럼 seed 자체가 오류의 원인인 경우에는 같은 prompt의 여러 seed를 별도 묶음으로 보여줄 수 있다.

선택 pool:

- 우선 `protocol_rep_pool_seed20260608`의 representative 300 prompt × 3 seed를 사용한다.
- generator별 최대 후보는 900개다.
- 새 prompt를 먼저 만들지 않고, 이미 고정된 representative pool 안에서 artifact 예시를 뽑는다.

선택 방식:

1. generator별 모든 후보에 artifact metric을 붙인다.
2. motion group을 붙인다.
3. generator별 대표 artifact profile에 따라 artifact bucket을 나눈다.
4. motion group quota를 맞추며 severity 높은 샘플을 우선 선택한다.
5. 부족한 group은 shortage로 기록하고, 가까운 동작군에서 보충한다.
6. 최종 manifest에 선택 이유, metric, prompt, seed, 길이 정보를 모두 남긴다.

## 4. 동작군 구성

각 generator의 100개는 아래 동작군을 기준으로 구성한다. 목표는 walking만 과하게 보이는 것을 막고, foot contact, landing, balance, transition artifact를 골고루 보여주는 것이다.

| motion group | n per generator | 포함할 동작 예시 | 주로 볼 artifact |
|---|---:|---|---|
| walk_locomotion | 16 | normal walk, fast walk, backward walk, march | foot-skate, contact drift |
| run_jog | 10 | run forward, jog, sprint-like motion | foot-skate, floating, jitter |
| turn_path | 12 | turn left/right, circle, pivot, curved path | foot-skate, trajectory/contact mismatch |
| jump_hop | 12 | jump, hop, landing, small leap | footfloating, landing failure |
| kick_squat_lunge | 12 | kick, squat, lunge, crouch | floating, penetration-like contact, balance issue |
| dance_rhythmic | 8 | dance, sway, spin, rhythmic arm/leg motion | jitter, rhythm loss, contact instability |
| sit_stand_transition | 10 | sit down, stand up, bend, kneel | transition instability, contact mismatch |
| upper_body_low_loco | 10 | reach, raise arm, throw, balance/support pose | global pose oddity, support-foot floating |
| other_unclassified | 10 | climb, stagger, broadjump, mixed free-form prompt | classifier 밖 visible artifact |
| total | 100 |  |  |

## 5. Generator별 artifact bucket 설계

동작군 quota는 위 표를 공통으로 유지하고, 그 안에서 generator별 artifact bucket을 다르게 둔다. 이유는 generator마다 실제로 드러난 오류 profile이 다르기 때문이다.

### 5.1 MotionGPT 100개

현재 관찰:

- FootFloating이 많이 보임.
- length-control failure가 별도 문제로 확인됨.
- 일부 짧은 생성은 artifact라기보다 생성 실패 또는 temporal completeness failure로 분리해야 함.

구성:

| artifact bucket | n | 선택 기준 |
|---|---:|---|
| FootFloating | 45 | FootFloating score 상위, 특히 jump/balance/transition |
| Length-control failure | 20 | length ratio가 크게 벗어난 샘플, 극단적으로 짧은 샘플 포함 |
| High artifact_total / jitter-like | 15 | artifact_total 상위, accel 또는 temporal instability 동반 |
| FootSkate / trajectory contact issue | 10 | foot_skate_world 상위 MotionGPT 사례 |
| Mixed / borderline visible | 10 | 여러 지표가 중간 이상이고 눈으로 확인 가능한 사례 |
| total | 100 |  |

### 5.2 MDM 100개

현재 관찰:

- world-space foot-skate가 가장 뚜렷한 signature.
- diffusion 계열 특성상 trajectory와 접지 안정성이 핵심 시각 포인트가 될 가능성이 큼.

구성:

| artifact bucket | n | 선택 기준 |
|---|---:|---|
| World FootSkate | 55 | foot_skate_world 상위, walk/run/turn 중심 |
| Trajectory/contact drift | 15 | 접지 구간에서 root trajectory와 foot contact가 어긋나는 사례 |
| FootFloating | 10 | FootFloating score 상위 MDM 사례 |
| Jitter / smoothness oddity | 10 | accel 또는 temporal instability 상위 |
| Mixed / borderline visible | 10 | 여러 오류가 약하게 섞인 사례 |
| total | 100 |  |

### 5.3 MoMask 100개

현재 관찰:

- 전체적으로 상대적으로 깨끗하지만 FootFloating 사례가 존재.
- balance나 complex pose에서 contact 문제가 잘 보일 수 있음.

구성:

| artifact bucket | n | 선택 기준 |
|---|---:|---|
| FootFloating / contact instability | 45 | FootFloating score 상위, balance/jump/kick 중심 |
| Complex-pose artifact | 20 | fine-grained body-part prompt 또는 support pose에서 visible issue |
| Rare FootSkate | 10 | MoMask 내 foot_skate_world 상위 |
| Temporal / smoothness oddity | 10 | accel 또는 local instability 상위 |
| Mixed / borderline visible | 15 | 오류가 강하지 않지만 시각 비교에 유용한 사례 |
| total | 100 |  |

## 6. 산출물 설계

최종적으로 아래 산출물을 만든다.

| 산출물 | 역할 |
|---|---|
| `artifact_100_manifest_motiongpt.json` | MotionGPT 100개 예시의 sample_id, seed, prompt, metric, 선택 이유 |
| `artifact_100_manifest_mdm.json` | MDM 100개 예시 manifest |
| `artifact_100_manifest_momask.json` | MoMask 100개 예시 manifest |
| `artifact_100_summary.md` | generator별 오류 분포와 대표 사례 요약 |
| preview GIF pack | 전체 100개를 바로 만들기 전, generator별 24개 preview |
| optional full GIF pack | 최종 발표 또는 appendix용 100개 전체 GIF |

preview GIF 권장 구성:

- generator별 24개
- 8개 motion group × 각 group top 3개
- 먼저 preview로 눈검증한 뒤, 문제가 없으면 100개 전체 GIF 생성

## 7. 연구적으로 조심할 점

이 100개 세트는 artifact 예시 라이브러리다. 따라서 다음 표현은 피한다.

- "이 generator가 가장 나쁘다"
- "100개 예시에서 많이 보였으므로 전체 분포에서도 많다"
- "시각 예시만으로 refinement가 성능을 개선했다"

대신 다음처럼 쓴다.

- "각 generator에서 관찰된 artifact 유형을 정성적으로 보여준다."
- "MDM은 world-space foot-skate 사례를 보여주기 적합하다."
- "MotionGPT와 MoMask는 FootFloating/contact instability 사례를 보여주기 적합하다."
- "정량적 prevalence와 품질 개선 여부는 별도 metric과 paired test로 평가한다."

## 8. 다음 구현 순서

1. `tools/generator_artifact_100case_select.py` 작성
2. representative pool의 metric snapshot을 입력으로 generator별 100개 manifest 생성
3. motion group quota와 artifact bucket quota 충족 여부 검증
4. generator별 24개 preview GIF 생성
5. preview 확인 후 100개 full GIF 생성 여부 결정
