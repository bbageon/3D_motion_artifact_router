# AR-049 — Hard-tier Prompt Bank + Tiered Sample Size

## 한 줄 목표

실사용까지 가려면 generator 가 실제로 어려워하는 **hard-tier**(compositional/fine-grained/long/rare) 모션에서 평가해야 한다. soft-tier 와 hard-tier 를 **표본 크기를 차등**해 확보·평가한다. AR-048 frozen protocol 재사용(prompt bank 만 교체).

## 동기

현재 bank 는 단순 동작(walk/jog/wave) 위주 → generator 가 깨끗 → refinement headroom 작음(F7/F8). generator 의 artifact·headroom 은 **hard 동작**에 있다(survey §2). 실사용 evidence 는 hard-tier 에서 나온다.

## Tiered 표본 크기 (사용자 결정 2026-06-08, §3-22 검토 반영)

| tier | 역할 | prompt n | 허용 지표 |
|---|---|---|---|
| **soft-tier** | **diagnostic only** (low-headroom + no-harm 확인) | **50** | **per-sample(Category B/C) 전용** — foot skate/float/penetrate·no-harm Δ. **FID/R-Prec(Category A) 인용 금지** |
| **hard-tier** | **primary evidence** (refinement 필요성·효과) | **300** | full Category-A(FID/R-Prec, paired Δ+CI) + per-sample. per-gen 동일, 3 seed (§5·§6-10) |

### 설계 (A) 확정 규약 (사용자 결정 2026-06-08)

- **soft-tier 지표 = per-sample(Category B/C) 전용** (n=50 에서 신뢰 가능, §3-9 n≥20). soft 에서 **FID/R-Precision 계산·인용 금지** (n=50 은 R-Prec batch 부족·FID bias 로 unreliable).
- **tier 비교(soft vs hard) = per-sample 물리 지표로만** (N 달라도 robust). **FID 로 soft↔hard 비교 금지** (Chong&Forsyth: bias 가 N 마다 달라 unequal-N FID 비교 무효).
- soft 가 보여야 할 것("generator 깨끗 + no-harm")은 per-sample 물리(foot skate≈0)·no-harm Δ 로 충분 → FID 불필요.

### §3-22 검토 — 왜 soft 50 (diagnostic) / hard 300 (primary) 인가

1. **R-Precision = 32개 배치 내 랭킹** (Guo et al., **CVPR 2022** HumanML3D protocol): 생성 모션을 32 후보 중에서 retrieve → top-1/2/3.
   - n=50 → batch ~1.5개 → **R-Precision 매우 불안정**. n=300 → ~9 batch → 안정. ⇒ **R-Prec claim 은 n≫32 (≥300) 필요**.
2. **FID 는 표본수 의존 편향** (Chong & Forsyth, **CVPR 2020** "Effectively Unbiased FID"): FID 는 N 증가 시 감소하고 **bias 계수가 model 마다 다름** → 작은 N 에서 model 간 FID 비교는 신뢰 불가. "수천 미만은 unreliable".
   - n=50 FID = noisy·biased → **diagnostic only**. n=300 도 field ideal(full test set ~4384) 미만이나, **equal-N paired Δ + CI** 로 상대 비교는 workable. 절대 FID 단독 인용 금지, 필요 시 FID_∞ extrapolation.
3. **§3-9**: 성능 결론 trial≥20 + snapshot≥2. soft 50 은 diagnostic 이라 OK, hard 300 은 primary claim 가능 규모.

**결론**:
- **soft-tier 50** = "generator 가 깨끗하고 고칠 게 적다 + no-harm" 의 **diagnostic 확인**. **Category-A(FID/R-Prec) claim 으로 인용 금지** (§3-20·§3-17 일관).
- **hard-tier 300** = **primary evidence**. 단 (a) per-generator 동일 N + **paired Δ + CI** (절대 FID 단독 금지), (b) 3 seed 평균(§5), (c) R-Precision ~9 batch 확보.

## 고도화 방법 (complexity 기준) — **추후 결정** (사용자)

hard 선별 기준 후보 (확정 아님): compositional(다중 verb·then/while/and) · fine-grained(body-part 명시) · long(GT 길이 상위) · rare(저빈도 동작). **in-distribution HumanML3D 한정**(GT 유지 → FID/R-Prec 비교 가능). OOD custom prompt 는 GT 없어 제외.

## 산출물 / 성공 조건 (개요)

- hard-tier prompt bank (300, AR-048 builder + complexity 필터) + AR-048 protocol 로 3-generator 생성.
- soft(50) diagnostic / hard(300) primary 분리 보고.
- 표본 크기 근거·tier 라벨 기록 (본 spec).

## Claim Boundary

soft-tier 결과로 성능/no-harm 외 claim 금지. hard-tier 도 300 이 field ideal 미만 → paired Δ + CI 기반, 절대값 단독 금지.

## 근거

- HumanML3D protocol (Guo et al., CVPR 2022) — R-Precision batch-32.
- Effectively Unbiased FID (Chong & Forsyth, CVPR 2020) — FID 표본수 편향.
- survey §2 (generator failure modes), AGENTS §3-9/§3-17/§3-20.
