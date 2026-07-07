# .claude/docs/findings/ — 발견한 사실·시사점 (index)

> 본 폴더 = 실험으로 **확립된 발견(finding)과 그 연구적 시사점** 의 단일 출처.
> raw 측정 = `evals/snapshots/`, 일자 일지 = `reports/`, 본 폴더 = 그것을 **합성한 결론·함의**.
> 모든 finding 은 evidence cross-link + evidence tier(§3-17) + metric category(§3-20) + 남는 불확실성을 동반한다 (§3-22 4항목).

| 문서 | 주제 | 한 줄 결론 |
|---|---|---|
| [artifact_tool_alignment_audit.md](artifact_tool_alignment_audit.md) (A-1~A-6) | **artifact↔tool 정의 정합 감사 (AR-062)** | Skate/Penetrate gate 는 내부 추정 사용 시 **구조적으로 fire 불가**(vacuous, 재현됨) + BoneCV gate 무판별(100% fire) + FootFloating 은 local 좌표에서 보행 중 blind → **gate-fire 지표 인용 금지**, guard 는 foot_skate_world/float_mag/raw BoneCV/FID 로; foot-skate 만 metric·tool 정합 성립 → **AR-061 PROCEED**, gate 수리 = AR-063 |
| [motiongpt_implications.md](motiongpt_implications.md) (F0~F10) | **MotionGPT(G2)에 대한 본 연구의 시사점** | MotionGPT(VQ)는 물리적으로 깨끗(정의 문제 거의 없음, F7 — 단 gate-fire 부분은 [audit A-1/A-2](artifact_tool_alignment_audit.md) vacuity caveat, 상대 서열은 Cat-B 로 유지) → real refinement headroom 작고 대표 결과 FID NEUTRAL + no-harm(F0); 위반은 의미 축에 집중(F8) → 우리 기하 tool 범위 밖(F9, 메커니즘 F10) → **MotionGPT 단독 필요성 약함, 필요성은 diffusion 에서** |

## 관련

- 데이터셋 카드: [../dataset/motiongpt_g2_pool.md](../dataset/motiongpt_g2_pool.md)
- generator 일반 실패 유형 (문헌): [../generator/generator_failure_mode_survey.md](../generator/generator_failure_mode_survey.md)
- metric 정의·category: [../governance/metric_provenance.md](../governance/metric_provenance.md)
- 연구 position: [../governance/current_research_position.md](../governance/current_research_position.md)
