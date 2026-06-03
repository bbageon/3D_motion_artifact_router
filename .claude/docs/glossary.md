# Glossary — ArtifactRouter 용어·약어 사전

> 본 프로젝트 전용 용어·약어의 **단일 출처**. 채팅·일지·리포트에서 용어 첫 등장 시 본 사전의 풀이를 인용한다 ([phase 01-instructions §2-5-1](../rules/phase/01-instructions.md) 고급 용어 풀이 의무). 정식 정의가 별도 문서에 있으면 그 문서가 canonical (본 사전은 요약 + 링크).

## 1. 프로젝트 정체성

| 용어 | 풀이 |
|---|---|
| **ArtifactRouter** | 외부 motion generator output 위에서 **artifact state → correction action 매핑**을 routing 문제로 정식화한 generator-agnostic, tool-extensible decision system. 새 generator·새 단일 calibrator를 만들지 않음. |
| **Safe Orchestration** | 본 프로젝트 정식 framing. RL objective = `maximize artifact_improvement s.t. physical_validity + no_harm`. NetGain은 internal routing reward일 뿐 최종 quality 아님. 단일 출처: [docs/current_research_position.md §0](../../docs/current_research_position.md). |

## 2. Generator tier

| 용어 | 풀이 |
|---|---|
| **G1** | high-quality diffusion-based generator (MDM/MLD). **현재 미구축** (계획만). |
| **G2** | token-based generator = 공식 MotionGPT (NeurIPS 2023). 본 프로젝트의 실 generator evidence 출처. conda env `mgpt`. |
| **G3** | legacy artifact-rich generator (Gemma LoRA). **본 프로젝트 generator tier 아님** (AGENTS.md §3-5, vestigial archive). |

## 3. 컴포넌트

| 용어 | 풀이 |
|---|---|
| **Evaluator** | motion → artifact report (score/severity). artifact evaluator 3 (FootFloating/BoneLength/VelocityJitter) + physical gate evaluator 5. |
| **Correction Tool** | motion 국소 보정. **FootLock** (foot를 ground로 interpolate), **BoneProjection** (bone length를 ref로 projection), **VelocitySmoothing** (frame축 gaussian smoothing). 공통 interface `apply(...)`. |
| **Orchestrator** | evaluator report + tool history → tool decision. rule-based / supervised / contextual-bandit. |
| **KDG** | Kinematic Dependency Graph (골격 의존성 그래프). tool ordering·conflict 규칙. root/상위 node 먼저, hard-constraint → soft 순. AGENTS.md §3-3. |
| **Refinement loop** | closed-loop: generate → normalize → evaluate → correct → re-evaluate → STOP. Score 비감소 의무. |
| **Skeleton Normalizer** | 외부 generator output → canonical SMPL 22-joint `[T,22,3]`, fps=20, root-relative. |

## 4. Physical Gate

| 용어 | 풀이 |
|---|---|
| **Physical Constraint Gate** | Safe Orchestration의 hard gate. accept / repair / rollback / STOP 결정. **NetGain weight가 아닌 별도 hard gate** (PhysDiff ICCV 2023 motivation). |
| **PhysicalGateV0** | 5 evaluator: **Penetrate** (ground 관통) / **Float** (떠있음) / **Skate** (foot sliding) / **JerkSpike** (가속도 급변) / **BoneLengthCV** (bone length 변동계수). |
| **regression-based threshold** | gate 임계 = `max(before+eps, clean_p99)`. absolute clean_p99 단독은 FP-heavy → 금지 (current_research_position §3-6-3-1). |
| **clean p95/p99** | HumanML3D clean motion 분포의 percentile. gate 임계 calibration 출처. |
| **gate_recheck** | 추론 시 후보 선택 후 **실제 gate를 한 번 더 적용**해 재검증. learned P_safe는 신뢰 못 하므로 필수 (AGENTS.md §6-14). |

## 5. RL / Policy

| 용어 | 풀이 |
|---|---|
| **RL-0/-1/-2** | RL stage. RL-0=oracle, RL-1=imitation, RL-2=safe routing policy. |
| **action space** | discrete tool × intensity. grid type: `discrete_3level` (small/medium/large) / `discrete_5level` / `dense_grid_proxy` / `bounded_continuous_u`. 단일 출처: [docs/action_space_provenance.md](../../docs/action_space_provenance.md). |
| **u (intensity)** | bounded continuous intervention intensity `u ∈ [0,1]`. tool mapper: FootLock/BoneProj factor=u, VelocitySmoothing sigma=2.0·u. |
| **Q_safe(s, tool, u)** | bounded continuous action-effect surface. policy(s)→action class 대신 utility surface 학습. action_space_provenance §5-2. |
| **action-effect transition** | `(state, tool, u, after_state, gate_result, utility)`. Q_safe 학습의 원재료 (motion sample 아님, AGENTS.md §3-26). |
| **two-head** | Q_utility (utility 예측, regressor) + P_safe (gate pass 확률, classifier). |
| **P_safe** | learned safety head. **신뢰할 safety model 아님** — Q proposes, real gate validates ([[project-psafe-safety-framing]] 메모리). |
| **selection_mode** | `direct_policy` / `ranked_candidates` / `risk_filtered` / `gated_execution` / `oracle` / `heuristic` / `diagnostic_no_gate`. AGENTS.md §3-25. |
| **STOP / abstain** | 아무 correction도 안 하는 action (u=0 identity). no-harm regime의 정답. |
| **M0/M1/M2/M3** | Q policy 학습 데이터 ablation. **M0**=Stage A broad mix (clean+g2+synthetic+boundary) / **M1**=G2 real only / **M2**=G2+synthetic aux / **M3**=M2+isotonic calib. M0가 OOD generalization 우월. |
| **dense oracle / sequence oracle / single-step oracle** | oracle ceiling. single-step=후보별 개별 적용 best / sequence=tool 시퀀스 best path / dense=fine u-grid의 best safe (real gate). |
| **hard-example mining** | Q/P_safe가 가장 틀린 경계면 (high-utility-but-gate-fail, boundary-adjacent, STOP-vs-weak)을 직접 수집. AGENTS.md §3-26. |

## 6. Metric (3 Category — [docs/metric_provenance.md](../../docs/metric_provenance.md) 단일 출처)

| 용어 | 풀이 |
|---|---|
| **Category A (standard)** | top-tier 논문 동일 정의. 외부 공개 최종 성능 근거 **가능**. FID/R-Precision/MM-Dist/Diversity (HumanML3D, Guo 2022 CVPR), MPJPE. |
| **Category B (variant)** | top-tier 변형. variant 명시 의무. foot skating, bone length variation, jerk. |
| **Category C (proxy)** | 본 프로젝트 자체 정의 = internal routing reward / diagnostic only. 외부 공개 최종 성능 근거 **금지**. **NetGain**, artifact_total_score, safe_utility. |
| **NetGain** | `ArtifactReduction − α·FidelityLoss − β·CorrectionMag − γ·ToolCost` (α=5.0, β=γ=0, calibrated_protocol_a_v1). **Category C**. |
| **FidelityLoss** | Protocol A (clean GT 기준 MPJPE 차) / B (generator output 기준, simplified, variant) / C (distributional FID/FGD). |
| **FID** | Fréchet Inception Distance (motion VAE latent). 낮을수록 자연 분포 인접. |
| **R-Precision** | text-motion retrieval top-K accuracy. text semantic 보존 측정. |
| **MM-Dist** | matched motion-text latent distance. |
| **MPJPE** | Mean Per-Joint Position Error — 관절별 위치 오차 평균. |

## 7. Evidence tier (AGENTS.md §3-17)

| 용어 | 풀이 |
|---|---|
| **controlled diagnostic** | synthetic corruption. tool/evaluator/oracle headroom 진단용. **최종 성능 sole evidence 금지**. |
| **real-distribution** | G1/G2 natural output. 가설 최종 성능 evidence. |
| **quality-validated** | perceptual + visual. sub-tier: **b1** (1명 internal) / **b2** (3명+ inter-rater) / **b3** (10-20명+ 논문급). |
| **stress / normal / g2_clean_like** | G2 pool의 artifact severity band (group-wise percentile). stress=top-20% artifact. |

## 8. 데이터 split (Stage 2-G)

| 용어 | 풀이 |
|---|---|
| **train_g2_real / calib / g2_stress_holdout / g2_natural_holdout / clean_noharm_holdout** | G2 real 중심 split. holdout = train 미포함 일반화 테스트. |
| **synthetic_aux_train / synthetic_diag_holdout** | synthetic = auxiliary / appendix (중심 아님). |
| **snapshot** | 동일 파이프라인의 독립 seed/split 반복. 회귀·재현 판정은 snapshot≥2 (AGENTS.md §3-9). |

## 9. ID 체계

| 용어 | 풀이 |
|---|---|
| **H-id** (예: H-2026-205) | 사전 등록 가설. append-only. status 전환은 사용자 승인 게이트. [evals/hypotheses/](../../evals/hypotheses/) + [hypotheses-summary.md](hypotheses-summary.md). |
| **W-id** (예: W-2026-001) | 우회·간접 해결 ledger. append-only. [evals/workarounds/](../../evals/workarounds/). |
| **AR-id** (예: AR-020) | 작업 board issue. [.claude/Dashboard/PROJECT_BOARD.md](../Dashboard/PROJECT_BOARD.md). |

## 10. 통계

| 용어 | 풀이 |
|---|---|
| **paired Wilcoxon signed-rank** | 짝지은 두 분포의 중앙값 차 유의성 (비모수). M0 vs baseline per-state NetGain. |
| **Cohen's d (paired)** | paired 차의 effect size (평균/표준편차). |
| **rank-biserial** | M0 우위 / baseline 우위 비율 기반 effect. |
| **bootstrap 95% CI** | N=1000 resampling으로 median 차의 신뢰구간. |
