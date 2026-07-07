"""AR-061 visual: coordinate cleanup before/after GIF (top skate segment, per generator 1건).

frozen spec §5 "visual before/after samples must be generated" 이행.
AR-058-3f 의 검증된 renderer (`footskate_segment_gif_ar058_3f.render_gif`, §3-19 Y-up
vertical_axis="y") 를 재사용해 **같은 frame window·같은 발·같은 ground** 로
original vs corrected(u=1.0) 를 나란히 렌더링. 첫 frame PNG 도 함께 저장 (§3-19
첫 frame inspection 용).

CLI (motion3d env):
    python tools/coordinate_footskate_visual_ar061.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from correction_tools import CoordinateFootSkateCleanupTool
from tools.coords_protocol import estimate_ground
from tools.footskate_segment_gif_ar058_3f import _top_skate_segment, render_gif

PACK = REPO_ROOT / "reports" / "figures" / "2026-07-02" / "ar058_3f_footskate_human_feedback_pack"
POOL = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608"
OUT = REPO_ROOT / "reports" / "figures" / "2026-07-07" / "ar061_coordinate_cleanup_before_after"

#: smoke (AR-061) 에서 foot_skate_world 개선이 확인된 case — generator 별 1건.
CASES = (
    "motiongpt_high_v2_01_013130_seed20532609",   # -22.4%
    "mdm_high_v2_02_012495_seed20519610",         # -9.9%
    "momask_high_v2_01_003255_seed20320610",      # -16.5%
)


def first_frame_png(gif_path: Path, png_path: Path) -> None:
    with Image.open(gif_path) as im:
        im.seek(0)
        im.convert("RGB").save(png_path)


def main() -> None:
    rows = {r["case_id"]: r for r in csv.DictReader(open(PACK / "case_key.csv", encoding="utf-8"))}
    tool = CoordinateFootSkateCleanupTool()
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# AR-061 coordinate cleanup before/after (top skate segment)",
             "",
             "> 같은 frame window·같은 발·같은 ground 로 original vs corrected(u=1.0).",
             "> before: 접지 중 발이 수평으로 미끄러짐(빨강 trail 이 길게 끌림) / after: trail 이 anchor 근처에 응집.",
             "",
             "| Case | Before | After | 첫 frame (Y-up 검사) |",
             "|---|---|---|---|"]
    for cid in CASES:
        r = rows[cid]
        traj = np.load(POOL / r["generator"] / f"{r['sample_id']}__seed{r['seed']}.trajectory.npy").astype(np.float64)
        T = traj.shape[0]
        ground = estimate_ground(traj)
        seg = _top_skate_segment(traj, ground)
        if seg is None:
            print(f"[SKIP] {cid}: no skate segment")
            continue
        foot, s, e, nsk = seg
        corr, rep = tool.apply(traj, target_part="both_feet", target_joints=[],
                               frame_range=(0, T - 1), strength="medium",
                               metadata={"continuous_u": 1.0, "ground_y": ground,
                                         "coord_space": "trajectory"})
        gb = OUT / f"{cid}__before.gif"
        ga = OUT / f"{cid}__after_u100.gif"
        render_gif(traj, foot, s, e, ground, gb, is_control=False)
        render_gif(corr, foot, s, e, ground, ga, is_control=False)
        first_frame_png(gb, OUT / f"{cid}__before_f0.png")
        first_frame_png(ga, OUT / f"{cid}__after_f0.png")
        lines.append(f"| {cid} | [before]({gb.name}) | [after u=1.0]({ga.name}) | "
                     f"[b]({cid}__before_f0.png) [a]({cid}__after_f0.png) |")
        print(f"[OK] {cid} seg=[{s},{e}] n_skate={nsk} n_correct={rep.metadata['n_correct']}")
    (OUT / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {OUT / 'index.md'}")


if __name__ == "__main__":
    main()
