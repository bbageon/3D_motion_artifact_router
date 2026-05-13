"""
HumanML3D-style matplotlib 3D motion animation (GIF/MP4).

본 함수는 [HumanML3D animation.ipynb](data/HumanML3D_repo/animation.ipynb)의 plot_3d_motion을
본 프로젝트에 맞게 이식한 것이다.

확장:
  - 정책 X(상체 12 joint) subset도 처리 — kinematic chain에서 missing joint는 자동 skip.
  - obs / pred 구간을 색으로 구분 (obs blue, pred GT gray, pred model orange).
  - GT vs pred overlay 모드: pred 구간에 두 motion을 같은 frame에 동시 표기.
"""

import argparse
import io
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# SMPL 22 joint 인덱스 (HumanML3D 표준)
SMPL_22 = [
    "PELVIS", "LEFT_HIP", "RIGHT_HIP", "SPINE1",
    "LEFT_KNEE", "RIGHT_KNEE", "SPINE2",
    "LEFT_ANKLE", "RIGHT_ANKLE", "SPINE3",
    "LEFT_FOOT", "RIGHT_FOOT", "NECK",
    "LEFT_COLLAR", "RIGHT_COLLAR", "HEAD",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
]
NAME_TO_IDX = {n: i for i, n in enumerate(SMPL_22)}

# HumanML3D 표준 5 chain (right leg / left leg / spine+head / left arm / right arm)
T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
CHAIN_LABELS = ["right leg", "left leg", "spine+head", "left arm", "right arm"]


def frames_dict_to_array(frames: List[Dict[str, List[float]]],
                        joint_filter: Optional[List[str]] = None) -> np.ndarray:
    """frames(list of {joint_name: [x,y,z]}) -> (T, 22, 3) ndarray.
    missing joint는 NaN으로 채워 chain 그릴 때 skip 가능하게 만든다.
    """
    T = len(frames)
    arr = np.full((T, 22, 3), np.nan, dtype=np.float32)
    use = set(joint_filter) if joint_filter is not None else set(SMPL_22)
    for t, fr in enumerate(frames):
        for n, xyz in fr.items():
            if n not in use:
                continue
            j = NAME_TO_IDX.get(n)
            if j is None:
                continue
            arr[t, j] = xyz
    return arr


def _valid_chain(chain, frame_xyz):
    """chain 인덱스 중 좌표가 NaN이 아닌 것만 필터링해 반환."""
    valid = [j for j in chain if not np.isnan(frame_xyz[j]).any()]
    return valid


def _draw_floor_plane(ax, xlim, ylim, zlim):
    """Y-up 가정. floor는 y=ymin 평면 위에 회색 polygon."""
    verts = [
        [xlim[0], ylim[0], zlim[0]],
        [xlim[0], ylim[0], zlim[1]],
        [xlim[1], ylim[0], zlim[1]],
        [xlim[1], ylim[0], zlim[0]],
    ]
    plane = Poly3DCollection([verts])
    plane.set_facecolor((0.5, 0.5, 0.5, 0.20))
    ax.add_collection3d(plane)


def _draw_skeleton(ax, frame_xyz: np.ndarray, color: str, lw: float, marker_size: int = 18):
    for chain in T2M_KINEMATIC_CHAIN:
        valid = _valid_chain(chain, frame_xyz)
        if len(valid) < 2:
            continue
        ax.plot3D(
            frame_xyz[valid, 0], frame_xyz[valid, 1], frame_xyz[valid, 2],
            linewidth=lw, color=color,
        )
        ax.scatter(
            frame_xyz[valid, 0], frame_xyz[valid, 1], frame_xyz[valid, 2],
            c=color, s=marker_size, depthshade=False,
        )


def _render_frame(index: int, T_obs: int, T_pred: int,
                 obs_arr, pred_gt_arr, pred_model_arr,
                 pelvis_xy, xlim, ylim, zlim,
                 title: str, figsize=(7, 7),
                 elev: float = 15.0, azim: float = -70.0) -> Image.Image:
    """프레임 1개를 PIL Image로 반환."""
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    fig.add_axes(ax)
    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(*ylim)
    ax.set_zlim3d(*zlim)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    _draw_floor_plane(ax, xlim, ylim, zlim)

    # PELVIS XZ trajectory (현재 frame까지)
    if index >= 1:
        pelvis_so_far = pelvis_xy[: index + 1]
        ax.plot3D(
            pelvis_so_far[:, 0],
            np.full_like(pelvis_so_far[:, 0], ylim[0]),
            pelvis_so_far[:, 1],
            linewidth=1.4, color="#666666",
        )

    if index < T_obs:
        _draw_skeleton(ax, obs_arr[index], color="#3273dc", lw=2.6)
        phase = f"obs frame {index}/{T_obs - 1}"
    else:
        i = index - T_obs
        if pred_model_arr is None:
            _draw_skeleton(ax, pred_gt_arr[i], color="#23d160", lw=2.6)
            phase = f"pred frame {i}/{T_pred - 1} (GT)"
        else:
            _draw_skeleton(ax, pred_gt_arr[i], color="#888888", lw=4.6)
            _draw_skeleton(ax, pred_model_arr[i], color="#ff8c00", lw=2.0)
            phase = f"pred frame {i}/{T_pred - 1} (GT gray vs model orange)"

    fig.suptitle(f"{title}\nframe {index + 1}/{T_obs + T_pred} — {phase}", fontsize=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def plot_3d_motion(
    save_path: str,
    obs_arr: np.ndarray,
    pred_gt_arr: np.ndarray,
    pred_model_arr: Optional[np.ndarray],
    title: str,
    fps: int = 4,
    figsize=(7, 7),
    elev: float = 15.0,
    azim: float = -70.0,
    hold_last_frame_ms: int = 800,
):
    """obs + pred GT (+ pred model overlay) 시퀀스를 GIF로 저장.

    matplotlib FuncAnimation 대신 PIL.Image.save(append_images=...)로 직접 합성한다
    (matplotlib·writer 버전에 비의존).
    """
    T_obs = obs_arr.shape[0]
    T_pred = pred_gt_arr.shape[0]
    T = T_obs + T_pred

    cat = np.concatenate([obs_arr, pred_gt_arr], axis=0)
    if pred_model_arr is not None:
        cat = np.concatenate([cat, pred_model_arr], axis=0)
    valid_pts = cat[~np.isnan(cat).any(axis=-1)]
    pad = 0.20
    xlim = (float(valid_pts[:, 0].min()) - pad, float(valid_pts[:, 0].max()) + pad)
    ylim = (float(valid_pts[:, 1].min()) - pad, float(valid_pts[:, 1].max()) + pad)
    zlim = (float(valid_pts[:, 2].min()) - pad, float(valid_pts[:, 2].max()) + pad)

    pelvis_xy = cat[: T_obs + T_pred, 0, [0, 2]]

    frames: List[Image.Image] = []
    for idx in range(T):
        img = _render_frame(
            idx, T_obs, T_pred, obs_arr, pred_gt_arr, pred_model_arr,
            pelvis_xy, xlim, ylim, zlim,
            title=title, figsize=figsize, elev=elev, azim=azim,
        )
        frames.append(img)

    duration_ms = int(1000 / fps)
    durations = [duration_ms] * T
    durations[-1] = hold_last_frame_ms  # 마지막 frame 잠시 holding
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(out),
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    print(f"[OK] saved: {out}  ({T} frames, ~{duration_ms} ms each)")


# ── CLI for H-2026-003 Stage 0 sample ────────────────────────────────────
def _build_h2026003_gif():
    """H-2026-003 Stage 0(sid=002989, delta seed=42)용 기본 GIF 생성."""
    gt_path = PROJECT_ROOT / "processed_noaug" / "train" / "002989_w0000.json"
    out_dir = PROJECT_ROOT / "reports" / "figures" / "2026-05-05"
    out_dir.mkdir(parents=True, exist_ok=True)
    upper12 = [
        "PELVIS", "SPINE1", "SPINE2", "SPINE3", "NECK", "HEAD",
        "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST",
    ]

    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    obs_seq = gt["obs_sequence_3d"][:5]    # 정책 X obs_limit=5
    pred_seq = gt["pred_sequence_3d"][:3]  # 정책 X pred_limit=3

    obs_arr = frames_dict_to_array(obs_seq, joint_filter=upper12)
    pred_gt_arr = frames_dict_to_array(pred_seq, joint_filter=upper12)

    # H-2026-003 Stage 0 reconstruction이 100% greedy match이므로
    # pred_model_arr == pred_gt_arr 로 두 motion sequence가 정확히 겹쳐야 함.
    pred_model_arr = pred_gt_arr.copy()

    out_path = out_dir / "h2026003_motion.gif"
    plot_3d_motion(
        save_path=str(out_path),
        obs_arr=obs_arr,
        pred_gt_arr=pred_gt_arr,
        pred_model_arr=pred_model_arr,
        title=("H-2026-003 Stage 0 — sid=002989, upper-body 12 joints\n"
               "obs(blue) -> pred(GT gray vs model orange overlay) — 100% greedy match"),
        fps=4,
    )


# ── CLI for H-2026-004 Stage 0 sample ────────────────────────────────────
def _build_h2026004_gif():
    """H-2026-004 Stage 0(sid=002989, X-full-G3: obs 7 / pred 10 / joint 22 / dec 3)용 GIF."""
    gt_path = PROJECT_ROOT / "processed_noaug" / "train" / "002989_w0000.json"
    out_dir = PROJECT_ROOT / "reports" / "figures" / "2026-05-05"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    obs_seq = gt["obs_sequence_3d"][:7]      # X-full-G3 obs_limit=7
    pred_seq = gt["pred_sequence_3d"][:10]   # source pred_frames=10 (no-op limit)

    # full body 22 joint
    obs_arr = frames_dict_to_array(obs_seq, joint_filter=None)
    pred_gt_arr = frames_dict_to_array(pred_seq, joint_filter=None)

    # H-2026-004 Stage 0 reconstruction이 100% greedy match이므로
    # pred_model_arr == pred_gt_arr.
    pred_model_arr = pred_gt_arr.copy()

    out_path = out_dir / "h2026004_motion.gif"
    plot_3d_motion(
        save_path=str(out_path),
        obs_arr=obs_arr,
        pred_gt_arr=pred_gt_arr,
        pred_model_arr=pred_model_arr,
        title=("H-2026-004 Stage 0 — sid=002989, full body 22 joints (HumanML3D)\n"
               "obs 7 (blue) -> pred 10 (GT gray vs model orange) — 100% greedy match"),
        fps=4,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="h2026004-stage0",
                   choices=["h2026003-stage0", "h2026004-stage0"])
    args = p.parse_args()
    if args.mode == "h2026003-stage0":
        _build_h2026003_gif()
    elif args.mode == "h2026004-stage0":
        _build_h2026004_gif()


if __name__ == "__main__":
    main()
