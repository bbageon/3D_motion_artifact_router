# AR-058-3f Rater Index

This is the blinded rater-facing index. It intentionally hides tier, generator, and metric values.

## Rating Task

For each case, open the PNG and fill the response sheet.

Response sheet: [response_sheet_template.csv](response_sheet_template.csv)

| Case | Prompt | PNG |
|---|---|---|
| motiongpt_high_v2_01_013130_seed20532609 | a person runs forward with one leg crossing in front of the other repetitively ... | [motiongpt_high_v2_01_013130_seed20532609.png](motiongpt_high_v2_01_013130_seed20532609.png) |
| motiongpt_high_v2_02_009924_seed20468610 | the  person is running over a vault. | [motiongpt_high_v2_02_009924_seed20468610.png](motiongpt_high_v2_02_009924_seed20468610.png) |
| motiongpt_high_v2_03_001617_seed20289609 | a person walking one way and then increasing pace running back | [motiongpt_high_v2_03_001617_seed20289609.png](motiongpt_high_v2_03_001617_seed20289609.png) |
| motiongpt_low_control_01_004975_seed20367608 | a man shrugs his shoulders then takes a step back before waving with his right ... | [motiongpt_low_control_01_004975_seed20367608.png](motiongpt_low_control_01_004975_seed20367608.png) |
| motiongpt_low_control_02_005932_seed20388610 | a person stretches their hips, then arms, then bends forwards and steps forwards. | [motiongpt_low_control_02_005932_seed20388610.png](motiongpt_low_control_02_005932_seed20388610.png) |
| motiongpt_low_control_03_003321_seed20321608 | a man stands with arms on the side, leans to the right and moves one step to th... | [motiongpt_low_control_03_003321_seed20321608.png](motiongpt_low_control_03_003321_seed20321608.png) |
| mdm_high_v2_01_006185_seed20396609 | the person is running around in a circle. | [mdm_high_v2_01_006185_seed20396609.png](mdm_high_v2_01_006185_seed20396609.png) |
| mdm_high_v2_02_012495_seed20519610 | a person walks quickly forward, moving at a slight angle to the right | [mdm_high_v2_02_012495_seed20519610.png](mdm_high_v2_02_012495_seed20519610.png) |
| mdm_high_v2_03_001723_seed20290609 | a person taking a huge diagonal step | [mdm_high_v2_03_001723_seed20290609.png](mdm_high_v2_03_001723_seed20290609.png) |
| mdm_low_control_01_000759_seed20277608 | a person walks unbalanced as if they are on a tight rope. | [mdm_low_control_01_000759_seed20277608.png](mdm_low_control_01_000759_seed20277608.png) |
| mdm_low_control_02_009351_seed20456609 | a person squats to almost parallel then jumps to the horizontally to the left. | [mdm_low_control_02_009351_seed20456609.png](mdm_low_control_02_009351_seed20456609.png) |
| mdm_low_control_03_014506_seed20557610 | figure walks forward, raises foot to step up and over, uses other foot to drop ... | [mdm_low_control_03_014506_seed20557610.png](mdm_low_control_03_014506_seed20557610.png) |
| momask_high_v2_01_003255_seed20320610 | a person walking straight forward. | [momask_high_v2_01_003255_seed20320610.png](momask_high_v2_01_003255_seed20320610.png) |
| momask_high_v2_02_000304_seed20268610 | a person is walking in a steady forward motion. | [momask_high_v2_02_000304_seed20268610.png](momask_high_v2_02_000304_seed20268610.png) |
| momask_high_v2_03_014326_seed20553608 | person lunges forward with left foot first repeatedly | [momask_high_v2_03_014326_seed20553608.png](momask_high_v2_03_014326_seed20553608.png) |
| momask_low_control_01_010142_seed20476608 | man turn right hand inward and outward the reach to the left with both hands br... | [momask_low_control_01_010142_seed20476608.png](momask_low_control_01_010142_seed20476608.png) |
| momask_low_control_02_004037_seed20337608 | takes a step forward while right hand goes above the left hand in a circle moti... | [momask_low_control_02_004037_seed20337608.png](momask_low_control_02_004037_seed20337608.png) |
| momask_low_control_03_006014_seed20390610 | the man take sideways steps to the right. | [momask_low_control_03_006014_seed20390610.png](momask_low_control_03_006014_seed20390610.png) |

## Animated GIF (움직임 — 미끄러짐 판단 **우선**)

> foot-skate 는 시간적 artifact 라 정지 PNG(경로 plot)보다 **GIF(재생)** 로 판단하는 게 직관적입니다.
> 각 GIF 는 **측정된 접지 구간만** 재생합니다. **발이 바닥에 붙은 채 수평으로 미끄러지면 foot-skate**, 발이 들렸다 다른 위치에 놓이면 정상 step 입니다.
> (빨강 title 구간 = 접지+수평이동 측정 구간 / 초록 title = control 대조.)

| Case | GIF (움직임) | PNG (경로 plot, 보조) |
|---|---|---|
| motiongpt_high_v2_01_013130_seed20532609 | [gif](gif/motiongpt_high_v2_01_013130_seed20532609.gif) | [png](motiongpt_high_v2_01_013130_seed20532609.png) |
| motiongpt_high_v2_02_009924_seed20468610 | [gif](gif/motiongpt_high_v2_02_009924_seed20468610.gif) | [png](motiongpt_high_v2_02_009924_seed20468610.png) |
| motiongpt_high_v2_03_001617_seed20289609 | [gif](gif/motiongpt_high_v2_03_001617_seed20289609.gif) | [png](motiongpt_high_v2_03_001617_seed20289609.png) |
| motiongpt_low_control_01_004975_seed20367608 | [gif](gif/motiongpt_low_control_01_004975_seed20367608.gif) | [png](motiongpt_low_control_01_004975_seed20367608.png) |
| motiongpt_low_control_02_005932_seed20388610 | [gif](gif/motiongpt_low_control_02_005932_seed20388610.gif) | [png](motiongpt_low_control_02_005932_seed20388610.png) |
| motiongpt_low_control_03_003321_seed20321608 | [gif](gif/motiongpt_low_control_03_003321_seed20321608.gif) | [png](motiongpt_low_control_03_003321_seed20321608.png) |
| mdm_high_v2_01_006185_seed20396609 | [gif](gif/mdm_high_v2_01_006185_seed20396609.gif) | [png](mdm_high_v2_01_006185_seed20396609.png) |
| mdm_high_v2_02_012495_seed20519610 | [gif](gif/mdm_high_v2_02_012495_seed20519610.gif) | [png](mdm_high_v2_02_012495_seed20519610.png) |
| mdm_high_v2_03_001723_seed20290609 | [gif](gif/mdm_high_v2_03_001723_seed20290609.gif) | [png](mdm_high_v2_03_001723_seed20290609.png) |
| mdm_low_control_01_000759_seed20277608 | [gif](gif/mdm_low_control_01_000759_seed20277608.gif) | [png](mdm_low_control_01_000759_seed20277608.png) |
| mdm_low_control_02_009351_seed20456609 | [gif](gif/mdm_low_control_02_009351_seed20456609.gif) | [png](mdm_low_control_02_009351_seed20456609.png) |
| mdm_low_control_03_014506_seed20557610 | [gif](gif/mdm_low_control_03_014506_seed20557610.gif) | [png](mdm_low_control_03_014506_seed20557610.png) |
| momask_high_v2_01_003255_seed20320610 | [gif](gif/momask_high_v2_01_003255_seed20320610.gif) | [png](momask_high_v2_01_003255_seed20320610.png) |
| momask_high_v2_02_000304_seed20268610 | [gif](gif/momask_high_v2_02_000304_seed20268610.gif) | [png](momask_high_v2_02_000304_seed20268610.png) |
| momask_high_v2_03_014326_seed20553608 | [gif](gif/momask_high_v2_03_014326_seed20553608.gif) | [png](momask_high_v2_03_014326_seed20553608.png) |
| momask_low_control_01_010142_seed20476608 | [gif](gif/momask_low_control_01_010142_seed20476608.gif) | [png](momask_low_control_01_010142_seed20476608.png) |
| momask_low_control_02_004037_seed20337608 | [gif](gif/momask_low_control_02_004037_seed20337608.gif) | [png](momask_low_control_02_004037_seed20337608.png) |
| momask_low_control_03_006014_seed20390610 | [gif](gif/momask_low_control_03_006014_seed20390610.gif) | [png](momask_low_control_03_006014_seed20390610.png) |
