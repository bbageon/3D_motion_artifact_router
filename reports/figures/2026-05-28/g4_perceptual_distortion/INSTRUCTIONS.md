# Step G-4 Perceptual Distortion Validation — 사용자 작업 안내

## 질문 (사용자 directive 2026-05-28)
1. **Panel 2 (B2-artifact-best, over-smoothing)** 가 Panel 1 (Original) 대비 distortion
   (다리 길이 변형 / 자세 부자연 / 리듬 손실) 으로 보이는가?
2. **Panel 3 (safe oracle, gate-aware)** 가 Panel 1 (Original) 의 quality 를 보존하는가?

## GIF (top-6 b2_artifact_best bone CV distortion samples)
reports/figures/2026-05-28/g4_perceptual_distortion/*_3panel.gif

각 GIF 3-panel: **Original (gray) | B2-artifact-best (orange) | safe oracle (blue)**.

## 평가 (각 sample 별)
| 질문 | 응답 |
|---|---|
| B2 (panel 2) distortion 보이는가? | yes / no / subtle |
| safe oracle (panel 3) original 보존하는가? | yes / no |

## 해석 분기
- B2 distortion 다수 visible + safe oracle 보존 → **standard metric (FID/bone CV) 악화 가 perceptual 로 confirm** (quality-validated evidence).
- B2 distortion 안 보임 → metric 악화 가 perceptual threshold 미만 (caveat: G2 low/moderate severity).

본 평가 = b1 sub-tier (1명 internal sanity, AGENTS.md §3-17). 정식 evidence 는 3명+ (b2/b3).
