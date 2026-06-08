# Generator Failure-Mode Survey — "생성기가 잘 못 만드는 모션"

> 목적: ArtifactRouter 의 **real refinement headroom** 이 어디에 있는지 정하기 위해, text-to-motion / motion generator 가 **잘 만들지 못하는 모션 유형** 을 기존 연구로 조사한다.
> 동기: 2026-06-03 발견 — G2(MotionGPT)·HumanML3D test 에서 g2_natural FID 0.846 ≈ clean 0.717 → MotionGPT 가 이미 거의 GT 수준 → real FID headroom 이 작아 tool 이 standard metric 을 못 올림. "어떤 모션에서 generator 가 실패하는가" 를 알아야 headroom 있는 regime 을 고를 수 있다.
> 근거 규약: AGENTS.md §3-22 (2020+ peer-reviewed top-tier). load-bearing claim 은 peer-reviewed venue 로만, preprint/survey 는 보조.

---

## 0. 한 줄 결론

생성기 실패는 **두 층** 으로 갈린다:

- **(A) 저수준 물리 artifact** (foot sliding / floating / penetration / jitter / bone 불일치) — **ArtifactRouter tool 이 고칠 수 있는 층**. 단 **FID 는 이 층에 둔감** (distributional 지표라 per-frame local 실패를 거의 못 잡음).
- **(B) 고수준 의미·구성 실패** (rare/unseen action, fine-grained/body-part, compositional, long, trajectory) — **FID·R-Precision 을 크게 움직이는 층** 이지만 **post-hoc local tool 로는 못 고침** (생성 자체의 문제).

→ ArtifactRouter 의 가치는 **(A) 층** 에 있고, (A) 의 headroom 은 **token/VQ 생성기(MotionGPT)보다 diffusion 생성기(MDM/MLD)에서 크다** (아래 §3). 평가 지표도 **FID 단독이 아니라 physical plausibility metric(Skate/Float/Penetrate/PFC)** 이어야 (A) 의 개선이 보인다.

---

## 1. (A) 저수준 물리 plausibility artifact — ArtifactRouter 가 고치는 층

> 정의: 한 frame / 짧은 구간의 운동학·접촉 위반. "동작이 무엇인가"(semantic) 는 맞지만 "물리적으로 그럴듯한가"가 틀림.

| artifact | 설명 | ArtifactRouter tool |
|---|---|---|
| **Foot sliding (Skate)** | 지지발이 접지 중인데 수평으로 미끄러짐 | `FootLockTool` ✅ |
| **Floating (Float)** | 발이 바닥 위로 떠 있음 | `FootLockTool` / `BoneProjectionTool` ✅ |
| **Ground penetration (Penetrate)** | 발/신체가 바닥 아래로 관통 | (전용 projection tool 없음 — **gap**) ⚠️ |
| **High-freq jitter / jerk** | 프레임 간 고주파 떨림, 가속도 spike | `VelocitySmoothingTool` ✅ |
| **Bone length 불일치** | 골격 길이가 프레임마다 변동 (BoneLengthCV) | `BoneProjectionTool` ✅ |
| **Self-collision / interpenetration** | 신체 부위 간 물리적으로 불가능한 교차 (underexplored) | (전용 tool 없음 — **gap**) ❌ |
| **Unnatural rotation / leaning** | 부자연스러운 회전·뒤로 기울어짐 | (부분적) ⚠️ |

**근거 (peer-reviewed):**
- **PhysDiff (Yuan et al., ICCV 2023)**: "existing motion diffusion models often generate physically-implausible motions with pronounced artifacts such as **floating, foot sliding, and ground penetration**." root cause = diffusion 이 데이터 분포의 통계적 fitting 에만 의존하고 **명시적 물리 제약이 없음**.
- **MDM (Tevet et al., ICLR 2023)**: foot sliding 을 인지하고 **foot-contact geometric loss** 를 명시적으로 추가 (즉 base diffusion 은 foot contact 가 깨진다는 것을 자인).
- **HumanML3D (Guo et al., CVPR 2022)**: foot skating ratio 를 표준 평가 항목으로 포함 (생성 motion 의 foot contact 위반이 흔한 평가 대상).
- **POMP (Ji et al., CVPR 2025)**: physics-consistent 생성을 위해 phase manifold 도입 — 동일 artifact family(skate/float/penetrate)를 표적.

**보조 (preprint, 단독 근거 금지):** Morph (arXiv 2411.14951, motion-free physics optimization), survey 2505.09379 — "high-frequency jitter, foot sliding, anatomical inconsistencies", self-collision underexplored.

### 1-1. 왜 FID 가 이 층에 둔감한가 (proxy-standard misalignment 의 기원)

FID/FGD 는 **분포 거리** (tm2t latent 의 mean/cov 거리). 한 clip 의 발이 2cm 미끄러지는 local 위반은 latent 통계에 거의 영향 없음 → **g2_natural 이 local artifact 를 가져도 FID 는 clean 과 비슷**. 이것이 본 프로젝트의 핵심 관찰 (artifact proxy ↓ 인데 FID neutral)을 설명. **따라서 (A) 층 개선의 standard 증거는 FID 가 아니라 physical metric.**

→ **Skate / Float / Penetrate / PFC(Physical Foot Contact)** 는 PhysDiff·Morph·POMP 에서 **표준 물리 plausibility 지표** 로 사용 → 본 프로젝트에서 **Category B (standard-variant)** 로 인용 가능 (Category C proxy 보다 상위 증거). AR-029(standard foot skating metric) 의 정당성.

---

## 2. (B) 고수준 의미·구성 실패 — ArtifactRouter 범위 밖

> 정의: "어떤 동작인가" 자체가 틀리거나 누락. 생성 모델의 표현력·일반화 한계. **post-hoc local correction 으로 못 고침** (foot/bone/smoothing 이 semantic 을 바꾸지 못함).

| 실패 유형 | 설명 |
|---|---|
| **Rare / unseen action (OOD)** | 학습에 없던 동작 → 생성 실패 또는 가장 가까운 학습 동작으로 붕괴 |
| **Fine-grained / body-part-specific** | "왼손만 흔들며 걷기" 같은 세부 부위 지정 → coarse 학습 텍스트라 매핑 학습 안 됨 |
| **Compositional / sequential** | "걷다가 앉고 손 흔들기" 다중 동작 조합 → 전이 부자연·동작 누락 |
| **Long sequence** | 긴 시퀀스에서 품질 저하·discontinuity·drift |
| **Fast-changing root / trajectory / timing** | 빠른 root 변화·정밀 궤적·정확한 타이밍 미반영 |
| **Fine-grained intensity/magnitude** | 동작 강도(세기) 제어 — semantic 과 magnitude 가 텍스트에 얽혀 분리 안 됨 |

**근거:**
- **Controllability 5-aspect taxonomy** (survey *Text-driven Motion Generation: Overview, Challenges and Directions*, 2505.09379 — survey/preprint, overview 용도): (1) 다양한 명령 응답 부족, (2) pose 초기화 제한, (3) long-term 취약, (4) unseen 처리 부족, (5) body-part 세부 제어 부족.
- **Fine-grained 매핑 실패**: "Motion Generation from Fine-grained Textual Descriptions" (arXiv 2403.13518, preprint) — coarse 텍스트로 학습한 모델은 fine-grained 단어→motion primitive 매핑을 학습 못 해 unseen description 에서 실패.
- **Rare text / unseen + complex combination**: MOST (arXiv 2507.06590, preprint), T2MBench OOD benchmark (arXiv 2602.13751, preprint) — in-distribution 평가만으로는 일반화 평가 불가, 대부분 모델이 OOD fine-grained accuracy 에서 취약.

> ⚠️ 본 §2 의 근거는 대부분 preprint — §3-22 상 **단독 load-bearing 금지**. 본 문서에서는 "ArtifactRouter 범위 밖" 을 정하는 **scope 판단의 보조 근거** 로만 사용하고, peer-reviewed 확인 시 venue 갱신.

---

## 3. 생성기 paradigm 별 (A) 층 artifact 경향 — 왜 MotionGPT 는 headroom 이 작았나

| paradigm | 대표 | (A) 물리 artifact 경향 | 근거 |
|---|---|---|---|
| **Diffusion** (continuous) | MDM, MLD, MotionDiffuse | **많음** — 명시적 물리 제약 없이 통계 fitting → skate/float/penetrate/jitter | PhysDiff ICCV2023, MDM ICLR2023 |
| **Token / VQ** (discrete codebook) | MotionGPT, T2M-GPT, MoMask | **상대적으로 적음** — clean motion 으로 학습한 codebook 으로 양자화 → 물리적으로 그럴듯한 token 으로 snap | MoMask CVPR2024, MotionGPT NeurIPS2023 (codebook 구조) |

**해석 (본 프로젝트 적용):**
- MotionGPT(G2)는 **VQ codebook** 구조라 출력이 학습 분포의 그럴듯한 token 으로 snap → **(A) 층 물리 artifact 가 적음** → ArtifactRouter 가 고칠 게 적음 → real FID headroom 작음. **이것이 g2_natural FID≈clean 의 구조적 원인.**
- **Diffusion(MDM/MLD)은 (A) 층 artifact 가 구조적으로 많음** (PhysDiff 의 전제) → **ArtifactRouter 의 real headroom 이 큼**. → **AR-022(MDM 확보)가 정확한 다음 단계**.

> 주의: 이 표는 paradigm 의 **경향** 이며, 본 프로젝트에서 G1(MDM)·G2(MotionGPT)·MoMask 각각 **physical metric 으로 직접 측정** 해 확인해야 함 (§3-17 real-distribution evidence). 표 자체를 결론으로 인용 금지.

---

## 4. ArtifactRouter 에의 함의 (actionable)

1. **Headroom 은 (A) 층 + diffusion 생성기에 있다** → **AR-022(MDM/MLD), AR-045(MoMask)** 로 multi-generator stress pool 확보. MotionGPT 단독은 headroom 빈약.
2. **Standard 증거는 FID 가 아니라 physical metric** (Skate/Float/Penetrate/PFC, Category B) → **AR-029** 우선순위. FID 는 (B) 층 지표라 ArtifactRouter 개선에 둔감 → FID 단독으로 "효과 없음" 결론 내리면 잘못된 negative.
3. **(B) 층은 ArtifactRouter scope 밖** → contribution framing 에서 "semantic quality 향상" 을 주장하지 않음. "물리 plausibility 의 safe 향상 (no-harm)" 으로 정직하게 한정.
4. **Tool gap 식별**: penetration 전용 projection tool 없음 ⚠️, self-collision tool 없음 ❌ → diffusion 도입 후 penetration prevalence 가 높으면 tool 확보 후보 (새 가설 게이트).

---

## 5. Sources

Peer-reviewed (load-bearing 가능):
- PhysDiff: Physics-Guided Human Motion Diffusion Model — Yuan et al., **ICCV 2023**. <https://arxiv.org/pdf/2212.02500>
- Human Motion Diffusion Model (MDM) — Tevet et al., **ICLR 2023**. <https://openreview.net/pdf?id=SJ1kSyO2jwu>
- HumanML3D — Guo et al., **CVPR 2022**.
- MoMask — Guo et al., **CVPR 2024**.
- MotionGPT — Jiang et al., **NeurIPS 2023**.
- POMP: Physics-consistent Motion Generative Model through Phase Manifolds — Ji et al., **CVPR 2025**. <https://openaccess.thecvf.com/content/CVPR2025/papers/Ji_POMP_Physics-consistent_Motion_Generative_Model_through_Phase_Manifolds_CVPR_2025_paper.pdf>

Preprint / survey (보조, 단독 근거 금지):
- Text-driven Motion Generation: Overview, Challenges and Directions — <https://arxiv.org/html/2505.09379v1>
- Motion Generation from Fine-grained Textual Descriptions — <https://arxiv.org/abs/2403.13518>
- Morph: Motion-free Physics Optimization Framework — <https://arxiv.org/html/2411.14951>
- MOST: Motion Diffusion Model for Rare Text — <https://arxiv.org/pdf/2507.06590>
- T2MBench: Benchmark for OOD Text-to-Motion — <https://arxiv.org/pdf/2602.13751>
