# AR-022 — MDM Generator Acquisition

## 한 줄 목표

MotionGPT 외 추가 generator 1번으로 MDM(Motion Diffusion Model)을 확보하여 diffusion 계열 text-to-motion output pool을 만든다.

## 선택 이유

| 기준 | 판단 |
|---|---|
| generator family | diffusion |
| benchmark | HumanML3D / KIT text-to-motion |
| 연구 근거 | ICLR 2023 |
| MotionGPT와의 차이 | MotionGPT는 motion-language/token 계열, MDM은 denoising diffusion 계열 |
| ArtifactRouter 적용 가치 | generator family가 달라 failure mode 비교에 적합 |

## 작업 범위

1. 공식 repo / checkpoint / dependency 확보 가능성 확인.
2. 별도 conda env 또는 isolated runner 설계.
3. HumanML3D caption prompt로 generation smoke.
4. output을 canonical SMPL 22 `[T,22,3]`, fps=20, root-relative 형식으로 변환.
5. `generator_id=MDM` metadata와 seed/checkpoint hash 기록.
6. 작은 pool 우선 생성: n=50 smoke → n=200 또는 n=300 확장.

## 산출물

| 산출물 | 설명 |
|---|---|
| setup note | 설치/체크포인트/known issue |
| generator wrapper | `generators/mdm_wrapper.py` 또는 equivalent |
| smoke pool | canonicalized MDM output |
| profile snapshot | evaluator issue prevalence |

## 성공 조건

1. 같은 prompt bank에서 MDM output을 50개 이상 생성한다.
2. canonical format round-trip이 통과한다.
3. evaluator profile이 MotionGPT와 분리된 `generator_id`로 기록된다.
4. setup이 재현 가능한 명령어로 문서화된다.

## Claim Boundary

이 작업은 generator-agnostic claim을 증명하지 않는다. MDM output을 ArtifactRouter 분석 대상으로 확보하는 setup 단계다.

## 근거

- Tevet et al., "Human Motion Diffusion Model", ICLR 2023.
- HumanML3D, CVPR 2022 benchmark와 호환되는 text-to-motion setting.
