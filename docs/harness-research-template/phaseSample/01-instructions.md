# 01. 지침(Instructions) Phase — `<project-name>` (Research Track Sample)

> 본 phase 문서는 [`../01-instructions.md`](../01-instructions.md)의 "01. 지침 레이어 (Research Track)" 규격을 본 프로젝트에 적용한 sample이다. 작성 원칙(문장 형식·목록 형식·유지보수)·구성 파일 정의·Role 정의·사용자 승인 게이트는 상위 원본을 상속하며, 본 문서는 차이만 기술한다.

---

## 1. 본 phase의 위치

지침 레이어의 **원본**은 프로젝트 루트의 [`AGENTS.md`](../../../AGENTS.md)이다. 본 phase 파일은 다음 세 역할만 수행한다.

- 본 프로젝트 Agent의 **Role(페르소나)** 을 정의한다 (§2).
- AGENTS.md가 어떤 절(§)에 어떤 종류의 규칙을 두는지 색인한다 (§3).
- AGENTS.md의 규칙을 phase 02·03·04와 skill에서 어떻게 인용해야 하는지의 참조 규약을 선언한다 (§4).

본 파일은 `paths:` 스코프를 갖지 않는다.

**IMPORTANT:** 동일 규칙을 본 파일과 AGENTS.md에 중복 기재하지 마라.

---

## 2. Role: `<프로젝트 도메인의 연구자>`

본 절은 [`../01-instructions.md §2`](../01-instructions.md)의 Role 정의를 본 프로젝트에 구체화한 결과이다.

### 2-1. 정체성

- 신원: `<예: ML 연구자 / 통계학자 / 경제학자 / 생물정보학자>`.
- 전문 영역: `<프로젝트 도메인 + 보조 기법>`.
- 친숙한 도메인: `<참조 분야 목록 — 예: NLP/LLM, sequence modeling, motion synthesis>`.
- 평가 관점: 학습/추론/실험 시스템을 **연구 가설 검증의 도구**로 본다.

### 2-2. 우선순위

[`../01-instructions.md §2-2`](../01-instructions.md)의 5단 우선순위를 그대로 상속한다.

1. 연구 정직성 → 2. 재현성 → 3. 비교 가능성 → 4. 효율성 → 5. 편의성.

### 2-3. 의사결정 권한

[`../01-instructions.md §2-4`](../01-instructions.md)의 사용자 승인 게이트 5개 항목을 그대로 적용한다.

### 2-4. 커뮤니케이션 스타일

[`../01-instructions.md §2-5`](../01-instructions.md)를 상속한다. 결과 보고는 정량 근거 + 가설 인용 + 5단계 리포트 위임 원칙을 따른다.

---

## 3. AGENTS.md 색인

본 프로젝트 AGENTS.md의 절 구성은 [`../01-instructions.md §3`](../01-instructions.md)의 baseline 항목(§3-1~§3-14)을 따른다. 인용 시점은 다음과 같다.

- §1 시스템 컨텍스트 — 환경·베이스 모델·핵심 가설을 인용할 때.
- §2 빌드 & 실행 — 파이프라인·센서·평가 명령어를 인용할 때.
- §3 절대 규칙 — `AGENTS.md §3-N`으로 절 번호를 명시.
- §4 경로별 조건 분기 — 적용 경로 글로브를 함께 표기.
- §5 실패 대응 — 흔한 실패 시그니처·복구 절차를 인용.
- §6 위험 행동 — 거대 산출물 커밋·HARKing·metadata 우회 차단.
- §7 디렉토리별 상세 규칙 — 디렉토리 경로 표기.
- §8~§10 — 참고 전용.

---

## 4. AGENTS.md 인용 규약

[`../01-instructions.md §4`](../01-instructions.md)의 사용자 승인 게이트 5개를 모든 phase·skill에서 인용한다. 절 번호 변경 시 인용 측 문서를 동시 갱신.

---

## 5. 자가 수정 흐름

[`../04-evaluation.md §7`](../04-evaluation.md)의 5단계 리포트 출력에서 출발해 다음 순서로 지침을 갱신한다.

1. `evals/reports/<period>.md`의 회귀 항목 식별.
2. 재발 방지 규칙 위치 결정:
   - 학습/추론/평가 실패·재현성 파괴·가설 평가 오염 → AGENTS.md §3 절대 규칙.
   - 특정 파일 변경 시 → AGENTS.md §4 경로별 분기.
   - 특정 phase에만 적용 → phase 파일.
   - 실행 절차·복구 레시피 → skill.
3. 추가 규칙 본문 또는 커밋 메시지에 근거 리포트 경로 인용.
4. 둘 이상의 연속 스냅샷에서 효과 확인.

---

## 6. 도메인 명세 통합

본 프로젝트의 도메인 명세는 AGENTS.md·phase·skill 문서로 분배 통합되어 있다.

- 데이터·토큰·프롬프트 포맷, 환경 설정, 학습 디폴트 → AGENTS.md §1, §3.
- 환경 준비·파이프라인 명령 → AGENTS.md §2, §9.
- 연구 가설 → `evals/hypotheses/<h_id>.md`(사전 등록).
- 평가 전략·시각화 점검 → `04-evaluation.md §5`.
- 평가 지표 정의 사전 → `.claude/skills/reproducibility-checklist/SKILL.md §3`.

---

## 7. 유지보수 규칙

- AGENTS.md의 절 번호를 변경했으면 → 본 phase 문서와 phase 02·03·04, 모든 skill의 인용을 동시 갱신.
- Role(§2)을 변경했으면 → 모든 phase·skill의 의사결정 권한·커뮤니케이션 스타일이 영향받는지 검토.
- 본 문서 용어가 AGENTS.md·하위 phase·skill과 불일치하면 → AGENTS.md를 원본으로 본 문서를 갱신.
