# AR-071 — Root/Gait Mismatch 진단 결과 (부분 지지: 핵심 주장 확인)

- 가설 (사용자, 2026-07-12): **"foot skating 은 root trajectory 와 gait speed mismatch 의 결과일 수 있음"**
- Raw: [root_gait_mismatch_ar071_v1.json](../snapshots/root_gait_mismatch_ar071_v1.json) · harness: [tools/root_gait_mismatch_ar071.py](../../tools/root_gait_mismatch_ar071.py)
- 표본: MDM representative 300 prompts × 3 seeds + 같은 prompt 의 HumanML3D GT (누락 0)
- 설계 노트: deficit 은 **fs 정의와 독립인 골반 속도만** 사용 (GT 기준) — spec 초안의 접지 상대속도 기반 mismatch 는 fs 와 수학적 얽힘(tautology 위험)으로 참고 지표 강등. verdict 는 **CI 기반 규칙** (commit 전 로직 수정 기록: 초기 코드가 점추정>0.5 만 봐 T1 과판정 → 정정. 원 통계치 무변경).

## 결과

| 검정 | 예측 | 결과 | 판정 |
|---|---|---|---|
| **T2 — 전진 부족 실재** (GT 대비) | ratio < 1 | **v_root(MDM)/v_root(GT) = 0.419 [0.392, 0.447]**, median 0.369, **165/165 prompt (100%) < 1** | ✅ **지지 (압도적)** — MDM 은 같은 prompt 의 실측 대비 **절반 이하 속도로만 전진** |
| **T3 — 부족이 skate 를 설명** | rho > 0 | **Spearman ρ = +0.339, p = 8.5×10⁻⁶** (n=165) | ✅ **지지** — 부족이 클수록 foot_skate 큼 |
| T1 — 미끄럼 방향 서명 (뒤로/문워크) | backward 우세 | backward frac 0.538 **[0.497, 0.578]** (CI 가 0.5 포함), mean signed −0.0004, Wilcoxon p=0.92 | 🔶 **불확정** — 약한 뒤 쏠림이나 우연과 구분 불가 |

**Overall: 부분 지지.** 사용자 관찰의 핵심 ("전진 거리가 짧다" + "그로 인해 skate 발생") 은 T2·T3 로 확인. frame 단위 방향 서명(T1)은 미확정 — 해석 후보: (a) per-frame heading 이 회전·방향전환 보행에서 노이즈, (b) root 부족이 균일한 후진 drag 가 아니라 **양방향 수평 jitter** 로 발현될 수 있음 (부족 + 불안정의 혼합).

## v2 A/B 와의 정합 (탐색 수준)

v2 에서 "fs 를 가장 많이 고친 5쌍 전부 원본 선호" — root 가 GT 의 42%로만 전진하는 모션에서 발을 강하게 고정할수록 **보폭은 큰데 몸이 안 나가는** 러닝머신 인상이 강해진다는 기전과 정합. anchoring 계열의 지각 실패(v1 11/20, v2 10/20)의 **원인 후보**로서 일관.

## 한계 (§3-22)

- **상관 ≠ 인과** — 확증은 개입 실험(root-rescale 후 fs·지각 변화)으로만.
- GT 대비 속도비: prompt 가 "천천히" 를 요구하면 낮은 ratio 가 정답일 수 있음 — 단 GT 가 **같은 prompt 의 실측**이라 의미 요구는 상당 부분 통제됨 (그래도 100%/0.42 는 스타일 차이로 설명하기 어려운 크기).
- T1 heading 정의(순간 골반 속도)의 노이즈 — 직선 보행 부분집합 재검은 후속.

## 사전 합의 순서의 ③·④ 이행

- **③ AR-072 등록** (backlog): root-aware correction 설계 — **새 pair, 새 사전등록 + §3-11 사용자 게이트**.
- **④ 대증요법 재해석**: 기존 FootLock/coordinate cleanup/combo = **증상(미끄러짐)을 국소 억제하나 병인(root 전진 부족)을 건드리지 않는 대증요법** — [current_research_position §0-0](../../.claude/docs/governance/current_research_position.md) 에 반영. 이들의 물리 지표 개선(Category B)은 유효하나 지각 가치는 b1 2회 불성립.
