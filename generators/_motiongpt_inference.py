"""MotionGPT 내부 inference helper script — mgpt conda env 안에서만 실행된다.

본 스크립트는 [`motiongpt_wrapper.py`](motiongpt_wrapper.py) 의 subprocess 호출 대상이며
mgpt env 의존성 (pytorch_lightning, omegaconf, mGPT 패키지 등) 을 import 한다.
motion-router env 에서는 본 파일을 직접 실행하지 마라.

흐름:
  1. MotionGPT clone 경로를 sys.path 에 prepend + cwd 변경.
  2. mGPT.config.parse_args(phase='demo') 로 config 로드 (sys.argv 모킹).
  3. cfg 의 checkpoint·seed 를 CLI 인자로 override.
  4. build_data + build_model + checkpoint state_dict 로드.
  5. batch = {'length': [n_frames], 'text': [prompt]} 로 t2m 추론.
  6. outputs['joints'][0] (= [T, 22, 3]) 을 --output 경로에 npy 로 저장.
  7. metadata (생성 길이·prompt·seed 등) 을 .json 으로 함께 저장.

본 스크립트는 MotionGPT 의 데모 인터페이스 (configs/config_h3d_stage3.yaml + Test Checkpoint
+ batch = {'length', 'text'} → model(batch, task='t2m')) 를 그대로 호출한다 — README §Demo
와 demo.py:119~ 의 흐름을 참조.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MotionGPT inside-mgpt-env inference helper (DO NOT invoke from motion-router env)"
    )
    parser.add_argument("--motiongpt-root", required=True, type=Path,
                        help="Path to OpenMotionLab/MotionGPT clone (external_assets/MotionGPT)")
    parser.add_argument("--cfg", default="configs/config_h3d_stage3.yaml",
                        help="Config path relative to motiongpt-root")
    parser.add_argument("--ckpt", required=True, type=Path,
                        help="Pretrained checkpoint (.ckpt) path")
    parser.add_argument("--prompt", required=True,
                        help="Text-to-motion prompt")
    parser.add_argument("--n-frames", type=int, default=40,
                        help="Target frame count hint (MotionGPT may ignore — actual length recorded in metadata)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path,
                        help="Output .npy path. metadata is saved next to it as .json.")
    args = parser.parse_args()

    motiongpt_root = args.motiongpt_root.resolve()
    if not motiongpt_root.exists():
        raise FileNotFoundError(f"MotionGPT clone not found: {motiongpt_root}")

    # Inject MotionGPT 모듈 경로 + cwd 변경 (cfg 의 상대경로 처리를 위해)
    sys.path.insert(0, str(motiongpt_root))
    os.chdir(motiongpt_root)

    import torch  # noqa: E402
    import pytorch_lightning as pl  # noqa: E402

    # parse_args 는 sys.argv 를 읽으므로 모킹
    sys.argv = ["_motiongpt_inference", "--cfg", args.cfg]
    from mGPT.config import parse_args as mgpt_parse_args  # noqa: E402
    cfg = mgpt_parse_args(phase="demo")

    # CLI 인자로 override
    cfg.TEST.CHECKPOINTS = str(args.ckpt)
    cfg.SEED_VALUE = args.seed

    pl.seed_everything(args.seed)

    if cfg.ACCELERATOR == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    from mGPT.data.build_data import build_data  # noqa: E402
    from mGPT.models.build_model import build_model  # noqa: E402

    datamodule = build_data(cfg)
    model = build_model(cfg, datamodule)

    state_dict = torch.load(str(args.ckpt), map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device).eval()

    batch = {
        "length": [args.n_frames],
        "text": [args.prompt],
    }
    with torch.no_grad():
        outputs = model(batch, task="t2m")

    joints_obj = outputs["joints"]
    # MotionGPT 의 출력은 list of tensors 또는 batched tensor — 첫 번째 sample 만 추출
    if isinstance(joints_obj, list):
        joints = joints_obj[0]
    else:
        joints = joints_obj[0] if joints_obj.dim() == 4 else joints_obj

    if hasattr(joints, "cpu"):
        joints = joints.detach().cpu().numpy()
    joints = np.asarray(joints, dtype=np.float64)

    # joints shape 검증 (canonical [T, 22, 3])
    if joints.ndim != 3 or joints.shape[1] != 22 or joints.shape[2] != 3:
        raise ValueError(
            f"Unexpected joints shape {joints.shape}; expected [T, 22, 3]. "
            "MotionGPT 출력 포맷이 변경되었을 가능성 — wrapper 검토 필요."
        )

    # 'length' field 에서 실제 생성 길이 추출
    length_field = outputs.get("length")
    if length_field is not None:
        try:
            length_generated = int(length_field[0])
        except (TypeError, IndexError):
            length_generated = int(joints.shape[0])
    else:
        length_generated = int(joints.shape[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.output), joints)

    metadata = {
        "length_generated": length_generated,
        "n_frames_requested": args.n_frames,
        "prompt": args.prompt,
        "seed": args.seed,
        "cfg": args.cfg,
        "ckpt": str(args.ckpt),
        "task": "t2m",
        "joints_shape": list(joints.shape),
    }
    metadata_path = args.output.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[OK] joints {joints.shape} (length_generated={length_generated}) -> {args.output}")
    print(f"[OK] metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
