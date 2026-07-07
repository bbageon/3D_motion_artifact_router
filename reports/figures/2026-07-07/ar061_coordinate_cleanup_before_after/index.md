# AR-061 coordinate cleanup before/after (top skate segment)

> 같은 frame window·같은 발·같은 ground 로 original vs corrected(u=1.0).
> before: 접지 중 발이 수평으로 미끄러짐(빨강 trail 이 길게 끌림) / after: trail 이 anchor 근처에 응집.

| Case | Before | After | 첫 frame (Y-up 검사) |
|---|---|---|---|
| motiongpt_high_v2_01_013130_seed20532609 | [before](motiongpt_high_v2_01_013130_seed20532609__before.gif) | [after u=1.0](motiongpt_high_v2_01_013130_seed20532609__after_u100.gif) | [b](motiongpt_high_v2_01_013130_seed20532609__before_f0.png) [a](motiongpt_high_v2_01_013130_seed20532609__after_f0.png) |
| mdm_high_v2_02_012495_seed20519610 | [before](mdm_high_v2_02_012495_seed20519610__before.gif) | [after u=1.0](mdm_high_v2_02_012495_seed20519610__after_u100.gif) | [b](mdm_high_v2_02_012495_seed20519610__before_f0.png) [a](mdm_high_v2_02_012495_seed20519610__after_f0.png) |
| momask_high_v2_01_003255_seed20320610 | [before](momask_high_v2_01_003255_seed20320610__before.gif) | [after u=1.0](momask_high_v2_01_003255_seed20320610__after_u100.gif) | [b](momask_high_v2_01_003255_seed20320610__before_f0.png) [a](momask_high_v2_01_003255_seed20320610__after_f0.png) |
