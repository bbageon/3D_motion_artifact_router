# AR-058-3h - A/B Preference Perceptual Test (Primary Pair 한정)

Status: ready (사용자 go 대기)  
Epic: Perceptual  
Priority: 🔴  
Parent: AR-058-3e/3f (human validation 계보) / [확정 연구 가닥 §0-0](../governance/current_research_position.md)

## Goal

사용자의 관찰 ("human이 봤을 때 artifact가 명확한 문제로 받아들이기 어렵다") 을 **결정적 검정**으로 전환한다. 질문을 절대평가("이게 문제인가")에서 **A/B 강제선택("어느 쪽이 더 자연스러운가")** 으로 교체 — 안경 검안 방식. 대상은 확정 연구 가닥의 **primary pair (foot skating ↔ coord cleanup) 만**.

## Method

1. **자극**: MDM worst-tail (foot_skate_world 상위) 15~20 sample — 원본 vs coord cleanup 보정본(u는 BoneCV guard 안전 범위) 을 **같은 구간·같은 카메라**로 나란히 (AR-061 시각화 방식 재사용, 이미 3쌍 존재).
2. **형식**: 좌우 무작위 배치, 강제선택("발을 보세요 — 어느 쪽이 더 자연스럽습니까?") + 확신도. 평가자당 ~10분.
3. **사전 등록 판정 기준** (결과 보기 전 고정): 보정본 선호가 이항검정으로 우연(50%)을 유의하게 못 넘으면 → **primary pair 의 perceptual claim 폐기 + §10 Go/Stop 정식 소집**. 넘으면 → quality-validated tier 진입 (b2, 평가자 수에 따라).

## Why Before P5

P5 의 어조 ("지각 검증된 개선" vs "물리 지표상 개선(지각 미검)" vs Go/Stop) 를 본 결과가 결정. 실패 시나리오도 유효한 결과 (§3-13).

## Claim Boundary

허용: (통과 시) "primary pair 의 보정이 인간 선호로 검증됨 (n=평가자 수 명시)".
금지: 보조 pair 로 일반화 / 단일 평가자 결과의 통계 주장 (b1 tier 로 표기).
