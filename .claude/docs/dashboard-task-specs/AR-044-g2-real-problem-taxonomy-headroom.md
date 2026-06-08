# AR-044 — G2 Real Problem Taxonomy + Visual/Tool Headroom Audit

## 한 줄 목표

`q_proxy` / RL ranker 설계 전에, MotionGPT가 실제 생성한 G2 real motion에서 어떤 문제가 언제, 얼마나, 눈에 보일 정도로 발생하는지와 현재 refinement tool이 그 문제를 실제로 고칠 headroom이 있는지 확인한다.

## 배경

사용자 결정:

> 학습부터 하지 말고, 실제 데이터 기반으로 문제점을 먼저 분석해서 알고리즘을 세팅한다.

따라서 AR-040의 `q_proxy`는 사전에 임의로 정하지 않는다. 먼저 G2 real output에서 관찰되는 failure mode와 tool effect를 확인한 뒤 정의한다.

## 입력 데이터

| source | 역할 |
|---|---|
| MotionGPT G2 real pool | 주 분석 대상 |
| g2_stress_holdout | 문제 많은 G2 motion 분석 |
| g2_natural_holdout | 과보정 위험 분석 |
| train_g2_real / calib_g2_real | 분포 확인용, 학습 성능 claim에는 사용 금지 |

Synthetic corruption은 본 작업의 핵심 근거가 아니다. 필요하면 appendix diagnostic으로만 비교한다.

## 분석 질문

1. G2 real output에서 사전에 정의한 artifact/physical issue가 실제로 보이는가?
2. 어떤 motion group에서 어떤 문제가 많이 발생하는가?
3. evaluator가 잡은 issue가 시각적으로도 납득 가능한가?
4. 현재 tool이 실제 G2 issue를 고칠 수 있는가?
5. tool 적용 후 artifact는 줄지만 standard quality나 fidelity가 망가지는 사례가 있는가?
6. AR-040 `q_proxy`가 어떤 항을 반드시 포함해야 하는가?

## 산출물

| 산출물 | 설명 |
|---|---|
| issue taxonomy table | issue type x motion group x severity |
| visual pack | 보정 전/후 GIF 또는 MP4, Y-up 확인 |
| tool headroom table | issue별 best tool/u와 실패 사례 |
| failure gallery | artifact 감소하지만 quality 훼손되는 사례 |
| q_proxy implication memo | AR-040에서 reward에 넣어야 할 항목 |

## 성공 조건

1. 최소 4개 motion group 이상에서 대표 issue 사례를 확보한다.
2. `g2_stress`와 `g2_natural`의 차이를 수치와 시각 사례로 설명한다.
3. tool별로 "고칠 수 있는 문제"와 "망치는 문제"를 분리한다.
4. AR-040 `q_proxy`의 후보 항목을 데이터 기반으로 제안한다.
5. 분석 결과가 `reports/<YYYY-MM-DD>.md`와 snapshot에 기록된다.

## Claim Boundary

이 작업은 policy 성능을 증명하지 않는다. G2 real output에 존재하는 문제 구조와 tool effect headroom을 확인하여, 이후 `q_proxy`와 policy 설계를 데이터 기반으로 만들기 위한 전제 분석이다.

## 근거

- MotionGPT, NeurIPS 2023: 현재 G2 generator source.
- HumanML3D, CVPR 2022: text-to-motion caption/evaluation benchmark.
- PhysDiff, ICCV 2023: generated motion의 physical plausibility/contact 문제 motivation.
- HuMoR, ICCV 2021: body/contact/bone consistency 기반 human motion plausibility motivation.
