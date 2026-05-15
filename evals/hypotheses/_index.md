# Hypothesis Index — motion-artifact-router

본 디렉토리는 [`.claude/skills/hypothesis-registry/SKILL.md`](../../.claude/skills/hypothesis-registry/SKILL.md) 에 따라 등록된 ArtifactRouter 연구 가설의 인덱스이다.

본 저장소는 motion refinement framework (ArtifactRouter) 의 **독립 프로젝트** 이며, 이전 저장소 [`3D-Motion-Trajectory-prediction`](../../../3D-Motion-Trajectory-prediction/) 의 LLM-based motion generation 실험 (이전 저장소 가설 H-2026-101·102) 의 **후속 연구가 아니다**. 본 저장소의 가설은 이전 저장소의 가설들과 supersede 관계가 없다.

## 활성 가설

| h_id | 한 줄 요약 | 등록 | domain | track_scope |
|---|---|---|---|---|
| [H-2026-203](H-2026-203.md) | (secondary) High-quality generator output 에 대해 No-harm 운영 — 분포·semantic·fidelity 훼손 없음 | 2026-05-13 | no-harm | G1 |
| [H-2026-204](H-2026-204.md) | (H-2026-200 supersede) Artifact-conditioned tool selection + closed-loop refinement 가 fixed post-processing 보다 NetGain 우위 | 2026-05-15 | orchestration | G1, G2 |
| [H-2026-205](H-2026-205.md) | (H-2026-201 supersede) Learnable Routing — supervised/contextual-bandit (RL-style) selector 가 rule-based 대비 net gain 개선 | 2026-05-15 | routing-learning | G1, G2 |
| [H-2026-206](H-2026-206.md) | (H-2026-202 supersede) 학습된 selector 의 generator-agnostic 일반화 (G1↔G2 bidirectional) | 2026-05-15 | generator-transfer | G1, G2 |

## 종결 가설

| h_id | status | 종결 일 | 후속 | 사유 |
|---|---|---|---|---|
| [H-2026-200](H-2026-200.md) | superseded | 2026-05-15 | [H-2026-204](H-2026-204.md) | 초기 등록 시 `track_scope: [G1, G2, G3]` 로 기재했으나 G3 (이전 저장소 LoRA) 는 본 프로젝트 scope 외 — misregistration 정정 |
| [H-2026-201](H-2026-201.md) | superseded | 2026-05-15 | [H-2026-205](H-2026-205.md) | 부모 가설 supersede 의 자식 사유 상속, RL refinement 가 핵심 contribution 임을 명시 |
| [H-2026-202](H-2026-202.md) | superseded | 2026-05-15 | [H-2026-206](H-2026-206.md) | 동일 misregistration 정정, transfer pair 를 G1↔G2 bidirectional 로 한정 |

## 승계 그래프

```
H-2026-200 (superseded 2026-05-15) ──→ H-2026-204 (active) — ArtifactRouter Main RQ1+RQ2
                                          │
H-2026-201 (superseded 2026-05-15) ──→ H-2026-205 (active) — Learnable Routing (RL-style)
                                          │
H-2026-202 (superseded 2026-05-15) ──→ H-2026-206 (active) — Generator transfer (G1↔G2)

H-2026-203 (active, secondary) — No-harm on high-quality
```

supersede 의 공통 사유: **본 프로젝트는 독립 프로젝트이며 이전 저장소의 LLM motion generation 후속이 아님**. 초기 등록 시 generator scope 에 G3 (이전 저장소 LoRA 자산) 를 inadvertent inclusion 한 misregistration 을 정정한 결과.

## 참고 — 이전 저장소 가설

이전 저장소 [`../../3D-Motion-Trajectory-prediction/evals/hypotheses/`](../../../3D-Motion-Trajectory-prediction/evals/hypotheses/) 의 가설 (H-2026-001~004, H-2026-101, H-2026-102) 은 **별개의 연구** 이며, 본 저장소와 supersede 관계가 없다. 본 저장소가 이전 저장소에서 import 하는 것은 public dataset (HumanML3D) 과 시각화 utility (plot_3d_motion.py) 등 **연구 가설과 무관한 자산** 에 한정한다.

## 유지보수

- 새 가설은 [`hypothesis-registry SKILL §3`](../../.claude/skills/hypothesis-registry/SKILL.md) 절차로 등록 후 본 표에 한 줄 추가.
- status 전환·supersede 는 [`hypothesis-registry SKILL §4`](../../.claude/skills/hypothesis-registry/SKILL.md) 사용자 승인 게이트 통과 후 본 표 갱신.
- 본 인덱스는 가설 본문을 보관하지 않는다 — 각 `<h_id>.md` 파일이 단일 출처.
