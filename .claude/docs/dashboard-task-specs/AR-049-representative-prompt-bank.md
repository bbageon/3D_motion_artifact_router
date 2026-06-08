# AR-049 — Representative Prompt Bank + Complexity Annotation

## 한 줄 목표

AR-048 적격 조건을 통과한 HumanML3D test 후보에서 **고정 seed 무작위 추출한 representative 300 prompts**를 primary dataset으로 고정한다. 같은 300개에 complexity annotation을 추가하여 전체 분포 성능과 조건별 성능을 한 데이터셋에서 함께 평가한다.

## 설계 변경 이유

초기안은 `soft 50 / hard 300`을 별도로 구성하고 hard subset을 primary evidence로 사용하는 방식이었다. 그러나 hardness top-300의 82.7%가 180--199 frame에 집중되어 사실상 long-motion challenge set이 되었고, HumanML3D test 분포의 대표성을 잃을 위험이 확인되었다.

따라서 다음 원칙으로 변경한다.

1. primary 표본은 complexity score로 선별하지 않는다.
2. 길이 quota도 적용하지 않는다.
3. complexity는 표본 구성 기준이 아니라 **조건부 분석용 annotation**으로만 사용한다.
4. 기존 hard top-300 bank는 삭제하지 않되, 생성하지 않은 `complexity_challenge_candidate`로 보존한다.

## Primary Dataset

| 항목 | 고정 규약 |
|---|---|
| source | HumanML3D 공식 test split |
| eligibility | AR-048: full-motion caption, non-M mirror dedup, GT 존재, 문장 dedup, `40 <= GT_length < 200` |
| sampling | 적격 후보 전체에서 고정 seed 단순 무작위 추출 |
| prompt 수 | 300 |
| generator | MotionGPT / MDM / MoMask 동일 prompt |
| 반복 생성 | prompt당 generator별 3 seed |
| 통계 단위 | prompt; seed는 prompt 내부 반복 측정 |
| protocol | AR-048 trajectory/local 이중 표현, common-length semantic floor, 동일 provenance |

이 데이터셋은 엄밀히 말해 HumanML3D test 전체가 아니라 **AR-048 적격 조건부 test 분포의 representative sample**이다.

## Complexity Annotation

각 prompt에 generator 결과와 무관한 다음 원시 값과 표준화 값을 기록한다.

| 축 | 정의 | 역할 |
|---|---|---|
| compositional | POS 동사 수 + 사전 정의 순차/동시 표지 | 복합·순차 동작 분석 |
| fine-grained | 신체 부위어 + 좌우 지시 수 | 국소·세밀 지시 분석 |
| long | GT frame length | 긴 동작 분석 |
| rare | 적격 caption corpus의 content-word rarity | 저빈도 언어 조건 탐색 |

`overall_complexity`는 분석 편의를 위한 내부 composite label이며, 검증된 절대 난이도 척도로 주장하지 않는다. 원시 4축 결과를 항상 함께 보고한다.

## 사전 고정 조건부 분석

generator 결과를 보기 전에 다음 분석을 고정한다.

1. representative 300 전체: primary 성능.
2. 각 complexity 축 상위 25%: 해당 조건에서의 성능과 failure mode.
3. overall complexity 하위 25%: low-complexity no-harm.
4. overall complexity 상위 25%: 복합 조건 robustness.
5. prompt가 여러 축에 동시에 포함되는 것을 허용하고 membership을 기록한다.

사분위 분석은 표본을 새로 모집하는 것이 아니라 동일 representative 300 내부의 descriptive subgroup analysis다. 전체 성능보다 우선하는 primary claim으로 사용하지 않는다.

## 기존 Hard Bank 처리

`protocol_hard_test_300_seed20260608`은 다음 상태로 보존한다.

- 이름/등급: `complexity_challenge_candidate`
- 현재 상태: prompt bank만 생성, generator motion 미생성
- primary evidence 사용 금지
- representative 결과에서 특정 complexity 축의 실패가 확인될 때 appendix robustness 실험 후보
- 길이 편향과 rarity 오타 민감성을 limitation metadata에 기록

## 산출물 / 성공 조건

1. representative prompt bank 300개와 고정 seed가 기록된다.
2. AR-048 eligibility와 중복 제거 검사가 통과한다.
3. 각 prompt에 4축 raw/z-score와 사분위 membership이 기록된다.
4. generator 결과를 사용하지 않고 selection/annotation이 완료된다.
5. 동일 prompt index로 3 generator x 3 seed 생성이 가능하다.
6. 기존 hard bank가 primary와 혼합되지 않고 challenge candidate로 명시된다.

## 평가 규약

- Category A: representative 300 전체에서 FID/R-Precision/MM-Dist를 동일 N으로 비교하고 paired/bootstrap CI를 동반한다.
- Category B/C: prompt 단위 physical/artifact 변화와 no-harm을 함께 보고한다.
- complexity subgroup: 조건별 결과로만 보고하며 전체 분포 성능을 대체하지 않는다.
- n=300은 HumanML3D full test보다 작으므로 절대 FID 단독 주장을 금지한다.

## Claim Boundary

primary claim은 `AR-048 적격 HumanML3D test representative-300` 범위다. complexity annotation은 generator 난이도의 정답 label이 아니라 a-priori linguistic/GT descriptor다. 기존 hard bank 결과로 HumanML3D 전체 성능을 주장하지 않는다.

## 근거

- HumanML3D (Guo et al., CVPR 2022): 공식 test caption/GT와 Category-A 평가 기반.
- Effectively Unbiased FID (Chong & Forsyth, CVPR 2020): 작은 표본의 FID 편향, 동일 N과 불확실성 보고 필요.
- AGENTS.md §3-9/§3-17/§3-20: 다중 trial, evidence tier, metric citation gate.
