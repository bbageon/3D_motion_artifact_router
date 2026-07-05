"""AR-058-3f GIF: 측정된 foot-skate segment 애니메이션 (사람 지각 검증용).

정지 PNG(경로 plot)는 좌표 판정과 맞지만 "사람 눈에도 미끄러져 보이나" 판단엔 비직관적이다.
본 도구는 각 case 의 **top skate segment**(접지 중 발이 수평 이동한 구간)만 재생하는 GIF 를 만든다:
skeleton 애니메이션(Y-up, §3-19) + skating foot 의 궤적 trail + ground plane. body 는 움직이는데
발은 낮게(접지) 유지되며 수평으로 미끄러지는 것을 눈으로 보게 한다.

v2 contact/skate 정의는 3b/3g 와 동일 (contact_h=0.05, contact_vy=0.035, skate_dxz=0.025).

CLI (motion3d env):
    python tools/footskate_segment_gif_ar058_3f.py            # 18 case 전부
    python tools/footskate_segment_gif_ar058_3f.py --only motiongpt_high_v2_01_013130_seed20532609
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from skeleton_normalizer.canonical_smpl_22 import T2M_KINEMATIC_CHAIN
from tools.coords_protocol import LEFT_FOOT, RIGHT_FOOT, estimate_ground

CONTACT_H, CONTACT_VY, SKATE_DXZ = 0.05, 0.035, 0.025
PACK = REPO_ROOT / "reports" / "figures" / "2026-07-02" / "ar058_3f_footskate_human_feedback_pack"
POOL = REPO_ROOT / "external_assets" / "protocol_rep_pool_seed20260608"


def _flags(motion, foot, ground):
    fy = motion[:, foot, 1] - ground
    fxz = motion[:, foot, :][:, [0, 2]]
    disp = np.linalg.norm(np.diff(fxz, axis=0), axis=1)
    disp = np.concatenate([disp, [disp[-1] if len(disp) else 0.0]])
    vy = np.abs(np.concatenate([np.diff(fy), [0.0]]))
    contact = (fy <= CONTACT_H) & (vy <= CONTACT_VY)
    skate = contact & (disp >= SKATE_DXZ)
    return contact, skate


def _segments(mask):
    out, s = [], None
    for i, f in enumerate(mask.astype(bool)):
        if f and s is None:
            s = i
        elif not f and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out


def _top_skate_segment(motion, ground):
    """가장 skate frame 많은 (contiguous) contact segment 를 반환. (foot, start, end, n_skate)."""
    best = None
    for foot in (LEFT_FOOT, RIGHT_FOOT):
        contact, skate = _flags(motion, foot, ground)
        for (s, e) in _segments(contact):
            nsk = int(skate[s:e + 1].sum())
            if nsk > 0 and (best is None or nsk > best[3]):
                best = (foot, s, e, nsk)
    return best


def _top_contact_segment(motion, ground):
    """skate 없을 때 fallback: 가장 긴 contact segment (정상 접지, 미끄러짐 없음 = control)."""
    best = None
    for foot in (LEFT_FOOT, RIGHT_FOOT):
        contact, _ = _flags(motion, foot, ground)
        for (s, e) in _segments(contact):
            if best is None or (e - s) > (best[2] - best[1]):
                best = (foot, s, e)
    if best is None:
        T = motion.shape[0]
        return (RIGHT_FOOT, 0, min(T - 1, 40))
    foot, s, e = best
    return (foot, s, min(e, s + 50))  # cap 길이 (control GIF 크기 절감)


def render_gif(motion, foot, start, end, ground, out_path, pad=5, fps=8, is_control=False):
    T = motion.shape[0]
    s, e = max(0, start - pad), min(T - 1, end + pad)
    # fixed axis limits (전체 skeleton + 이동 범위) so the slide is visible against fixed axes.
    seg = motion[s:e + 1]
    xmin, xmax = seg[:, :, 0].min(), seg[:, :, 0].max()
    ymin, ymax = seg[:, :, 1].min(), seg[:, :, 1].max()
    zmin, zmax = seg[:, :, 2].min(), seg[:, :, 2].max()
    mx = max(xmax - xmin, zmax - zmin, ymax - ymin) * 0.55 + 0.1
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    fcolor = "#d62728" if foot == LEFT_FOOT else "#1f77b4"

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    def draw(t):
        ax.clear()
        ax.view_init(elev=12, azim=-70, vertical_axis="y")  # §3-19 Y-up
        ax.set_xlim(cx - mx, cx + mx); ax.set_ylim(cy - mx, cy + mx); ax.set_zlim(cz - mx, cz + mx)
        # skeleton
        for chain in T2M_KINEMATIC_CHAIN:
            for a, b in zip(chain[:-1], chain[1:]):
                ax.plot([motion[t, a, 0], motion[t, b, 0]], [motion[t, a, 1], motion[t, b, 1]],
                        [motion[t, a, 2], motion[t, b, 2]], color="#333", lw=1.6)
        # ground plane (Y = ground)
        gx = np.array([cx - mx, cx + mx]); gz = np.array([cz - mx, cz + mx])
        GX, GZ = np.meshgrid(gx, gz)
        ax.plot_surface(GX, np.full_like(GX, ground), GZ, alpha=0.12, color="gray")
        # skating foot + trail (start..t)
        in_seg = start <= t <= end
        ax.scatter([motion[t, foot, 0]], [motion[t, foot, 1]], [motion[t, foot, 2]],
                   color=fcolor, s=90, edgecolors="k", zorder=5)
        tr = motion[start:min(t, end) + 1, foot] if t >= start else motion[start:start + 1, foot]
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=fcolor, lw=3, alpha=0.9)
        if is_control:
            lab = "control: contact (should NOT slide)" if in_seg else ""
            col = "#080"
        else:
            lab = "<< CONTACT + horizontal slide (skate)" if in_seg else ""
            col = "#c00" if in_seg else "#666"
        ax.set_title(f"frame {t}  {lab}", fontsize=11, color=(col if in_seg else "#666"))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    anim = FuncAnimation(fig, draw, frames=range(s, e + 1), interval=1000 / fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return {"seg_frames": [int(s), int(e)], "skate_seg": [int(start), int(end)],
            "foot": "left" if foot == LEFT_FOOT else "right"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="single case_id")
    ap.add_argument("--out-dir", type=Path, default=PACK / "gif")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(PACK / "case_key.csv", encoding="utf-8")))
    if args.only:
        rows = [r for r in rows if r["case_id"] == args.only]
    made = []
    for r in rows:
        gen, sid, seed, cid = r["generator"], r["sample_id"], r["seed"], r["case_id"]
        traj_p = POOL / gen / f"{sid}__seed{seed}.trajectory.npy"
        if not traj_p.exists():
            print(f"[SKIP] {cid}: no motion {traj_p.name}"); continue
        motion = np.load(str(traj_p)).astype(np.float64)
        ground = estimate_ground(motion)
        seg = _top_skate_segment(motion, ground)
        if seg is not None:
            foot, start, end, nsk = seg
            info = render_gif(motion, foot, start, end, ground, args.out_dir / f"{cid}.gif", is_control=False)
            print(f"[OK skate] {cid}  foot={info['foot']} skate_seg={info['skate_seg']} n_skate={nsk}")
        else:
            foot, start, end = _top_contact_segment(motion, ground)
            info = render_gif(motion, foot, start, end, ground, args.out_dir / f"{cid}.gif", is_control=True)
            nsk = 0
            print(f"[OK control] {cid}  foot={info['foot']} contact_seg=[{start},{end}] (no skate)")
        made.append((cid, info, nsk))
    print(f"\n[DONE] {len(made)} GIF -> {args.out_dir}")


if __name__ == "__main__":
    main()
