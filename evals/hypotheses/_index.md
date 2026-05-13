# Hypothesis Index — motion-artifact-router

본 디렉토리는 [`.claude/skills/hypothesis-registry/SKILL.md`](../../.claude/skills/hypothesis-registry/SKILL.md) 에 따라 등록된 ArtifactRouter 연구 가설의 인덱스이다.

## 활성 가설

| h_id | 한 줄 요약 | 등록 | domain | track_scope |
|---|---|---|---|---|
| [H-2026-200](H-2026-200.md) | Artifact-conditioned tool selection + closed-loop refinement 가 fixed post-processing 보다 NetGain 우위 | 2026-05-13 | orchestration | G1, G2, G3 |
| [H-2026-201](H-2026-201.md) | Learnable Routing — supervised/contextual-bandit selector 가 rule-based baseline 대비 net gain 의미 있게 개선 | 2026-05-13 | routing-learning | G1, G2, G3 |
| [H-2026-202](H-2026-202.md) | 학습된 selector 가 새로운 generator output 에 대해서도 generator-agnostic 일반화 가능 | 2026-05-13 | generator-transfer | G1, G2, G3 |
| [H-2026-203](H-2026-203.md) | (secondary) High-quality generator output 에 대해 No-harm 운영 — 분포·semantic·fidelity 훼손 없음 | 2026-05-13 | no-harm | G1 |

## 종결 가설

(없음)

## 승계 그래프

```
H-2026-200 (active) — ArtifactRouter Main RQ1+RQ2
H-2026-201 (active) — Learnable Routing (depends on H-2026-200)
H-2026-202 (active) — Generator transfer (depends on H-2026-201)
H-2026-203 (active, secondary) — No-harm on high-quality
```

## 참고 — 이전 저장소 가설

본 저장소는 별도 프로젝트이며 이전 저장소 [`3D-Motion-Trajectory-prediction`](../../../3D-Motion-Trajectory-prediction/) 의 가설 (H-2026-001~004, H-2026-101, H-2026-102) 과 직접 연결되지 않는다.

이전 저장소의 H-2026-101 (delta vs absolute drift) 과 H-2026-102 (LLM 숫자 token 직접 학습 baseline) 의 sanity check 결과는 본 저장소의 G3 (legacy artifact-rich generator) 사용 근거이지 본 가설의 supersede 관계가 아니다.

## 유지보수

- 새 가설은 [`hypothesis-registry SKILL §3`](../../.claude/skills/hypothesis-registry/SKILL.md) 절차로 등록 후 본 표에 한 줄 추가.
- status 전환·supersede 는 [`hypothesis-registry SKILL §4`](../../.claude/skills/hypothesis-registry/SKILL.md) 사용자 승인 게이트 통과 후 본 표 갱신.
- 본 인덱스는 가설 본문을 보관하지 않는다 — 각 `<h_id>.md` 파일이 단일 출처.
