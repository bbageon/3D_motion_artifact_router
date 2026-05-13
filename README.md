# ArtifactRouter

> **ArtifactRouter — Artifact-Conditioned Tool Selection for Generator-Agnostic Human Motion Refinement**

본 저장소는 [docs/motion_research_strategy_summary.md](docs/motion_research_strategy_summary.md) 의 새 명세에 따라 진행되는 **generator-agnostic, tool-extensible motion refinement harness** 연구 프로젝트이다.

## 핵심 메시지

본 연구는 새로운 motion generator를 개발하지 않는다. 기존/미래 motion generator (MotionGPT, T2M-GPT, MDM, MLD 등)의 출력 skeleton motion을 입력으로 받아:

1. **Evaluator tool registry**가 artifact를 평가하고,
2. **Orchestrator**가 artifact 상태에 따라 적절한 **correction tool**을 선택·조합하며,
3. **Re-evaluation loop**가 종료/반복을 결정해

refined motion + quality report + tool call trace를 생성하는 **closed-loop refinement framework**를 제안한다.

## 핵심 가설 (사전 등록)

- **H-2026-200**: Artifact-conditioned tool selection이 fixed post-processing보다 효과적이며 closed-loop refinement가 single-step보다 trade-off (artifact reduction vs motion fidelity) 우수.
- **H-2026-201**: Artifact state → correction action 매핑은 학습 가능한 routing 문제로 정식화 가능. supervised/contextual-bandit selector가 rule-based baseline 대비 net gain을 의미 있게 개선.
- **H-2026-202**: 학습된 selector가 새로운 generator output에 대해서도 generator-agnostic 일반화 가능.
- **H-2026-203** (secondary): High-quality motion (SOTA generator output)에 대해 No-harm 운영 특성 — 분포·의미·fidelity 훼손 없음.

## 본 저장소의 위치

본 저장소는 [3D-Motion-Trajectory-prediction](../3D-Motion-Trajectory-prediction/) (이전 LLM raw-token motion prediction sanity check 저장소)의 후속이지만 **독립적인 연구 프로젝트**다. 이전 저장소는 freeze 상태로 다음 자산을 본 저장소에 공급한다:

- **HumanML3D 데이터셋** (GT 분포, synthetic injection base) — `external_assets/HumanML3D`
- **정규화된 motion JSON** — `external_assets/processed_noaug`
- **학습된 LoRA adapter** (G3 generator로 활용) — `external_assets/local_lora_g3`
- **시각화·parser 도구** — `external_assets/code/`
- **하네스 연구 템플릿** — `docs/harness-research-template/` (이전 저장소에서 복사)

이전 저장소의 sanity check 결과 (단일 sample LLM raw-token prediction의 token explosion + generalization 부재 + format collapse)는 본 연구의 supporting evidence + G3 generator 공급원으로 활용된다.

## 디렉토리 구조

```
motion-artifact-router/
├── README.md                       # 본 파일
├── AGENTS.md                       # ArtifactRouter 도메인 명세 (하네스 §1-7)
├── CLAUDE.md                       # @AGENTS.md 인용
├── docs/
│   ├── motion_research_strategy_summary.md   # 연구 명세
│   ├── harness-research-template/             # 하네스 4계층 원본 (복사본)
│   └── design_decisions.md                    # 설계 결정 기록
├── external_assets/                # 이전 저장소 자산
├── generators/                     # §6.1 Base Motion Generator wrappers
├── skeleton_normalizer/            # §6.1 Skeleton Normalizer
├── evaluators/                     # §6.2 Evaluator Tool Registry
├── correction_tools/               # §6.3 Correction Tool Registry
├── orchestrator/                   # §6.4 Orchestrator (rule_based / bandit / KDG)
├── refinement_loop/                # §6.5 Closed-loop Refinement
├── tools/                          # 시각화·실험 도구
├── evals/                          # 평가 파이프라인 (hypotheses + raw + reports)
├── .claude/                        # 하네스 적용본 (rules/phase + skills)
├── reports/                        # 연구일지
└── experiments/                    # MVP feasibility study (4주 일정)
```

## 시작하기

```
# 환경 설정
conda create -n motion-router python=3.10 -y
conda activate motion-router
pip install -r requirements.txt

# 이전 저장소 자산 import (Windows symlink, Developer Mode 활성화 필요)
mklink /D external_assets\processed_noaug ..\..\3D-Motion-Trajectory-prediction\processed_noaug
mklink /D external_assets\local_lora_g3 ..\..\3D-Motion-Trajectory-prediction\lora_3d_delta_h2026004_singlesample_002989_stage1_seed42
mklink /D external_assets\HumanML3D ..\..\3D-Motion-Trajectory-prediction\data\HumanML3D
```

## 명세 (단일 출처)

연구 방향·메서드·실험 설계의 단일 출처는 [docs/motion_research_strategy_summary.md](docs/motion_research_strategy_summary.md) 이다. AGENTS.md는 본 명세를 하네스 4계층 ([01-instructions.md](docs/harness-research-template/01-instructions.md) ~ [04-evaluation.md](docs/harness-research-template/04-evaluation.md)) 형식으로 적응한 결과이다.

## License

TBD
