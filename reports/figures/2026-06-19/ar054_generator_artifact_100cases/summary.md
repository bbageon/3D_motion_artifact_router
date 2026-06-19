# AR-054 Generator Artifact 100-Case Library

작성일: 2026-06-19

이 문서는 representative pool에서 수집한 generator별 artifact 예시 100개를 요약한다.
이 세트는 정성적 오류 라이브러리이며, generator 성능 순위나 refinement 성능 claim이 아니다.

## Generator별 요약

| MGF | n | 주요 bucket | 주요 동작군 | 비고 |
|---|---:|---|---|---|
| MotionGPT | 100 | footfloating:45, length_control_failure:23, high_artifact_total:14 | walk_locomotion:17, turn_path:14, kick_squat_lunge:12 | FootFloating + length-control failure 중심 |
| MDM | 100 | world_footskate:55, trajectory_contact_drift:16, footfloating:11 | walk_locomotion:19, turn_path:12, kick_squat_lunge:12 | world-space foot-skate 중심 |
| MoMask | 100 | footfloating:45, complex_pose_contact:20, mixed_visible:15 | walk_locomotion:18, jump_hop:12, turn_path:12 | FootFloating/contact instability 중심 |

## 산출물

- `artifact_100_manifest_motiongpt.json`
- `artifact_100_manifest_mdm.json`
- `artifact_100_manifest_momask.json`
- `manifest.json`

## Preview 후보

각 generator별 상위 24개는 GIF preview 후보로 사용할 수 있다. 전체 100개 GIF는 파일 수와 용량이 크므로, 먼저 preview 검증 후 생성한다.

## 동작군 shortage

대표 pool에는 dance prompt가 unique 기준 5개뿐이다. seed 중복을 피했기 때문에 `dance_rhythmic`은 목표 8개를 채우지 않고 5개로 보고한다.

### MotionGPT
- #001 `footfloating` / `walk_locomotion` / 010315 seed 20483609 — the person is walking forward with the cake.
- #002 `footfloating` / `walk_locomotion` / 005537 seed 20380609 — a figure walks upstairs without a handrail.
- #003 `footfloating` / `walk_locomotion` / 005163 seed 20373608 — a person is strolling around
- #004 `footfloating` / `walk_locomotion` / 004601 seed 20353610 — someone is climbing a ladder,  they walk up 3 steps and then back down.
- #005 `footfloating` / `kick_squat_lunge` / 012492 seed 20518609 — a person stands with legs shoulder-width apart, slightly bent at the knees, arms outstretched at shoulder height, lowers left arm for several seconds, then brings with arm back to shoulder height.
- #006 `footfloating` / `turn_path` / 011402 seed 20503609 — a person slowly walks in a 3/4 circle.
- #007 `footfloating` / `jump_hop` / 006236 seed 20397609 — a figure jumps a few times counter-clockwise, then quickly turns clockwise with one big twisting jump, then stumbles backward upon landing.
- #008 `footfloating` / `walk_locomotion` / 008069 seed 20432608 — the person grabs something in front of them, walks away, walks back, returns it and picks it back up again and begins to walk away again.
- #009 `footfloating` / `kick_squat_lunge` / 010557 seed 20488609 — person puts hands on head then chest then knees then toes
- #010 `footfloating` / `turn_path` / 010116 seed 20474609 — a man energetically backsteps, then turns to his right and walks forward, then backsteps again, keeping a spring in his step.

### MDM
- #001 `world_footskate` / `walk_locomotion` / 012495 seed 20519610 — a person walks quickly forward, moving at a slight angle to the right
- #002 `world_footskate` / `walk_locomotion` / 003224 seed 20319610 — person walks seven steps from centre back to slightly right of centre
- #003 `world_footskate` / `walk_locomotion` / 001723 seed 20290609 — a person taking a huge diagonal step
- #004 `world_footskate` / `walk_locomotion` / 003539 seed 20326609 — a person is pushed by their left arm while walking forward.
- #005 `world_footskate` / `walk_locomotion` / 000304 seed 20268608 — a person is walking in a steady forward motion.
- #006 `world_footskate` / `turn_path` / 004350 seed 20348610 — a person walks forward and turns and walks backwards.
- #007 `world_footskate` / `kick_squat_lunge` / 005519 seed 20378610 — the guy lunges forward just a bit and the video ends.
- #008 `world_footskate` / `jump_hop` / 013222 seed 20536608 — a person hops forward then turns around and hops back past where they started.
- #009 `world_footskate` / `walk_locomotion` / 014506 seed 20557609 — figure walks forward, raises foot to step up and over, uses other foot to drop down, walks forward and stops.
- #010 `world_footskate` / `turn_path` / 011743 seed 20506610 — turning body from side to side.

### MoMask
- #001 `footfloating` / `walk_locomotion` / 005537 seed 20380609 — a figure walks upstairs without a handrail.
- #002 `footfloating` / `walk_locomotion` / 004601 seed 20353610 — someone is climbing a ladder,  they walk up 3 steps and then back down.
- #003 `footfloating` / `walk_locomotion` / 004823 seed 20361608 — a person walks down stairs holding on to something on the right.
- #004 `footfloating` / `walk_locomotion` / 002357 seed 20300608 — a headless line figure takes four steps forward, down a ramp, toward the viewer.
- #005 `footfloating` / `jump_hop` / 008349 seed 20441609 — a person stands with his two feet apart and light hops in that position 14 times.
- #006 `footfloating` / `walk_locomotion` / 006814 seed 20412610 — a person walks forward and then up stairs
- #007 `footfloating` / `turn_path` / 000021 seed 20260609 — person is walking normally in a circle
- #008 `footfloating` / `kick_squat_lunge` / 012668 seed 20522608 — a man is standing still, puts his hands together and behind his back, then proceeds to lunge his hands forwards, as if he is fishing, then brings them back in
- #009 `footfloating` / `jump_hop` / 008157 seed 20437609 — a person turns to their left while leaping forward.
- #010 `footfloating` / `walk_locomotion` / 013174 seed 20533609 — a person walks forward and picks up and moves a heavy object.
