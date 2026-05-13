# Workaround Ledger Index — motion-artifact-router

본 디렉토리는 [`.claude/skills/workaround-tracking/SKILL.md`](../../.claude/skills/workaround-tracking/SKILL.md) 에 따라 등록된 우회·간접 해결 항목의 인덱스이다.

## Active (open)

(없음 — 본 저장소 시작 시점.)

## Resolved

(없음)

## Accepted-permanent

(없음)

## 등급 정의 요약

- **critical** — 결과 의미를 바꾸거나 외부 재현 거의 불가능. 본격 학습/외부 공개 진입 전 반드시 해소 또는 supersede 로 명시.
- **material** — 환경·도구 의존성으로 결과 시간/방식이 달라짐 (수치 동일). 진입 전 검증 + 일지 명시 의무.
- **low** — 시각화·디버그 또는 사전 등록된 임시 정책 변경. 가설 변경 이력 기록 후 진행 가능.

## 참고 — 이전 저장소 우회

이전 저장소 [`3D-Motion-Trajectory-prediction`](../../../3D-Motion-Trajectory-prediction/) 의 [`evals/workarounds/_index.md`](../../../3D-Motion-Trajectory-prediction/evals/workarounds/_index.md) 에 W-2026-001~010 등록되어 있다. 본 저장소는 별도 프로젝트이므로 직접 상속하지 않으나 다음 항목은 본 저장소에서도 영향 받을 가능성 있음:

- **이전 W-2026-010** (NF4 inter-session 비결정성): 본 저장소가 G3 generator 로 이전 LoRA 사용 시 동일 이슈 발생 가능. 본 저장소에서 발견 시 새 W-id 로 등록.

## 유지보수

- 새 항목은 [`workaround-tracking SKILL §4`](../../.claude/skills/workaround-tracking/SKILL.md) 절차로 등록 후 본 표 추가.
- status 전환 (`open` → `resolved` / `accepted-permanent`) 은 사용자 승인 게이트 (critical 필수, material 권장) 거친 후 본 표 갱신.
- 본 인덱스는 ledger 본문을 보관하지 않는다 — 각 `W-YYYY-NNN.md` 가 단일 출처.
