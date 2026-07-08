# AR-067 - KIT-ML Dataset Generalization Check

Status: backlog (등록만 — 착수 시점 미정, 사용자 directive 2026-07-08 "나중에 할꺼니까 backlog 로 넣기만")  
Epic: Evidence  
Priority: ⚪  
Parent: 없음 (단일 벤치마크 한계 해소)

## Goal

현재 모든 결론이 **HumanML3D 단일 생태계** (시험지=test 캡션, 정답지=GT, 채점기=Guo evaluator, 수험생=HumanML3D 학습 체크포인트 3종) 위에 있다 — 핵심 결론이 **두 번째 벤치마크(KIT-ML)에서도 방향이 재현되는지** 확인해 데이터셋-일반화 caveat 를 해소한다.

## Why This Exists

- 논문 표준 관행 = HumanML3D + KIT-ML **두 벤치마크 병용** (MDM ICLR 2023, T2M-GPT CVPR 2023, MotionGPT NeurIPS 2023, MoMask CVPR 2024 모두 양쪽 보고).
- 우리의 generator-일반화(§6-10, 3 generator)는 확보됐지만 **데이터셋-일반화는 미검증** — 유병률 프로파일(MDM foot-skate 우세 등)·tool 효과가 HumanML3D 특성(캡션 스타일·모션 분포)에 얹혀 있을 가능성.

## Scope (착수 시 단계)

1. **자산 확보**: KIT-ML 데이터 + 각 generator 의 **KIT 학습 체크포인트** (MDM/T2M-GPT/MotionGPT/MoMask 는 논문에서 KIT 결과를 보고하므로 공개 체크포인트 존재 확인) + Guo evaluator **KIT 버전** (R-Precision/FID 용).
2. **골격 처리 (최대 난관)**: KIT = MMM 21-joint 형식 → canonical SMPL-22 retargeting 또는 KIT 전용 처리 경로. `skeleton_normalizer/` 변경 = **§4 round-trip 재검증 + 02-sensor §3-3 전체 게이트 격상** 대상.
3. **KIT prompt bank**: test split 캡션에서 무작위 표본 + annotation (AR-049 설계 재사용 — 선택 편향 회피 동일 원칙).
4. **핵심 재측정 (요약 재현)**: 유병률 (AR-063 스타일, trajectory) → fixed tool mixed effect (P4) → coord tool 효과 (AR-061) 의 **방향 재현 여부**.

## Success Criteria

- HumanML3D 결론들의 방향이 KIT-ML 에서 재현되는지 보고 — **재현/불일치 모두 유효한 결과** (불일치 시 데이터셋 의존성 자체가 발견, §3-13 negative result 보존).

## Claim Boundary

허용: (측정 후) "핵심 결론이 두 벤치마크에서 방향 일치/불일치".
금지: 등록·자산 확보만으로 일반화 주장 / retargeting 검증 전 KIT 수치 인용 (§3-1 canonical 게이트).

## Research Grounding (§3-22)

- **MDM** (Tevet et al., ICLR 2023) · **MoMask** (Guo et al., CVPR 2024) · **MotionGPT** (Jiang et al., NeurIPS 2023) — 두 벤치마크 병용이 분야 표준임의 근거.
- **HumanML3D** (Guo et al., CVPR 2022) — evaluator protocol 원본 (KIT 버전 evaluator 포함).
- KIT-ML (Plappert et al. 2016) 은 2020 이전 — **데이터 자산 인용(배경)으로만** 사용 (§7-0-A.4).

## Note

- 비용이 큰 작업 (retargeting + 체크포인트 4종 + evaluator + pool 재생성) — 착수는 HumanML3D 위 핵심 evidence chain (P5 등) 마무리 후 판단.
- 외부 공개 시 그때까지는 "단일 벤치마크(HumanML3D) 범위" 명시 의무.
