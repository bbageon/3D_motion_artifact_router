"""
processed/{split}/*.json -> 3D delta JSONL 변환기

입력:
  02_humanml3d_to_pose_json.py 가 생성한 JSON 파일들
  각 파일: {"obs_sequence_3d": [...], "pred_sequence_3d": [...], ...}

출력:
  finetune_3d_delta_train.jsonl
  finetune_3d_delta_val.jsonl
  finetune_3d_delta_test.jsonl

JSONL 레코드 포맷:
  {
    "prompt": "Observed motion deltas: PELVIS:(...),... | ...",
    "completion": "Next motion deltas: PELVIS:(...),... | ...",
    "sample_id": "000000_w0000",
    "task": "trajectory_delta_3d"
  }

숫자 토큰 포맷 (2D와 동일):
  -[NUM][INT]{int:03d}[SEP][DEC]{dec:05d}[ENDNUM]

3D 프레임 포맷 (2D 확장):
  JOINT:(tok_x,tok_y,tok_z),(tok_x,tok_y,tok_z),...
"""

import argparse
import glob
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEC_DIGITS = 5
DEC_SCALE = 10 ** DEC_DIGITS

SCHEMA_VERSION = "v1"
TOKEN_FORMAT_VERSION = "v1"
PROMPT_TEMPLATE_VERSION = "v1"

# 상체 12관절 subset (H-2026-003 데이터 정책 X). 본 subset은 sanity check 한정.
UPPER_BODY_12 = [
    "PELVIS", "SPINE1", "SPINE2", "SPINE3", "NECK", "HEAD",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
]


def _git_metadata() -> Dict[str, object]:
    """현재 작업 디렉토리의 git commit·dirty 여부를 수집한다(없으면 unknown)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": None}


def _build_meta(track: str, split: str, source_dir: Path, sample_id: Optional[str]) -> Dict:
    """data-versioning §2-2의 첫 줄 _meta 레코드.

    학습기·추론기·평가기는 JSONL 첫 줄에 `_meta` 키가 있으면 그 줄을 데이터에서 제외한다.
    """
    git = _git_metadata()
    return {
        "_meta": {
            "schema_version": SCHEMA_VERSION,
            "track": track,
            "split": split,
            "source_dir": str(source_dir),
            "sample_id_filter": sample_id,
            "token_format_version": TOKEN_FORMAT_VERSION,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "decimal_digits": DEC_DIGITS,
            "converter_file": Path(__file__).name,
            "git_commit": git["git_commit"],
            "git_dirty": git["git_dirty"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    }

# ----------------------------
# 1️⃣ 시스템 지시문
# ----------------------------
prompt_system = (
    "You are a motion prediction assistant that forecasts future motion deltas "
    "based on the observed motion sequence."
)

# {0}: 관찰 프레임 수, {1}: 예측 프레임 수, {2}: 데이터 본문
prompt_template = (
    "Forecast the next {1:d} motion deltas (dx, dy, dz) for all observed joints using the given {0:d} observed frames.\n"
    "Each coordinate is tokenized as [NUM][INT]000[SEP][DEC]00000[ENDNUM].\n"
    "Return a valid JSON object with predicted deltas for each joint in the same format.\n"
    "{2:s}"
)


def num_to_tokens(x: float, decimal_digits: int = DEC_DIGITS) -> str:
    """
    float -> 구조화 숫자 토큰 문자열 (2D 프로젝트와 동일)
    예: -0.01234 -> -[NUM][INT]000[SEP][DEC]01234[ENDNUM]

    decimal_digits를 override하면 정책 X(=3) 같은 임시 축소 가능.
    """
    dec_scale = 10 ** decimal_digits
    sign = "-" if x < 0 else ""
    x = abs(x)
    int_part = int(x)
    dec_part = int(round((x - int_part) * dec_scale))
    if dec_part >= dec_scale:
        int_part += 1
        dec_part -= dec_scale
    return f"{sign}[NUM][INT]{int_part:03d}[SEP][DEC]{dec_part:0{decimal_digits}d}[ENDNUM]"


def format_pose_sequence_deltas_3d(
    seq: List[Dict],
    max_frames: int = 20,
    joint_filter: Optional[List[str]] = None,
    decimal_digits: int = DEC_DIGITS,
) -> str:
    """
    pose_sequence_3d (프레임 리스트) -> 3D delta 토큰 문자열
    seq[i]: {"JOINT_NAME": [x, y, z], ...}
    반환: "JOINT:(tok_x,tok_y,tok_z),... | JOINT:..."

    joint_filter가 주어지면 해당 joints만 사용한다(원본 순서 보존).
    decimal_digits로 토큰 자릿수를 줄일 수 있다(정책 X).
    """
    if len(seq) < 2:
        return ""

    joints_list = list(seq[0].keys())
    if joint_filter is not None:
        filter_set = set(joint_filter)
        joints_list = [j for j in joints_list if j in filter_set]
    parts = []

    for joint in joints_list:
        deltas = []
        for i in range(1, len(seq)):
            if joint not in seq[i] or joint not in seq[i - 1]:
                continue
            x1, y1, z1 = seq[i - 1][joint]
            x2, y2, z2 = seq[i][joint]
            dx = round(x2 - x1, decimal_digits)
            dy = round(y2 - y1, decimal_digits)
            dz = round(z2 - z1, decimal_digits)
            deltas.append(
                f"({num_to_tokens(dx, decimal_digits)},"
                f"{num_to_tokens(dy, decimal_digits)},"
                f"{num_to_tokens(dz, decimal_digits)})"
            )
        if deltas:
            parts.append(f"{joint}:{','.join(deltas[:max_frames])}")

    return " | ".join(parts)


def convert_file(
    json_path: Path,
    obs_limit: Optional[int] = None,
    pred_limit: Optional[int] = None,
    joint_filter: Optional[List[str]] = None,
    decimal_digits: int = DEC_DIGITS,
) -> Tuple[str, str, str, str]:
    """
    JSON 파일 1개 -> (system, prompt, completion, sample_id)
    변환 실패 시 빈 문자열 반환

    obs_limit / pred_limit가 주어지면 앞에서부터 N 프레임만 사용한다(정책 X용).
    joint_filter / decimal_digits는 format_pose_sequence_deltas_3d에 전달된다.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    obs_seq = data.get("obs_sequence_3d", [])
    pred_seq = data.get("pred_sequence_3d", [])
    sample_id = data.get("sample_id", json_path.stem)

    if obs_limit is not None and obs_limit > 0:
        obs_seq = obs_seq[:obs_limit]
    if pred_limit is not None and pred_limit > 0:
        pred_seq = pred_seq[:pred_limit]

    if len(obs_seq) < 2 or len(pred_seq) < 2:
        return "", "", "", sample_id

    obs_text = format_pose_sequence_deltas_3d(
        obs_seq, joint_filter=joint_filter, decimal_digits=decimal_digits)
    pred_text = format_pose_sequence_deltas_3d(
        pred_seq, joint_filter=joint_filter, decimal_digits=decimal_digits)

    if not obs_text or not pred_text:
        return "", "", "", sample_id

    obs_frames = len(obs_seq) - 1
    pred_frames = len(pred_seq) - 1
    body = f"Observed motion deltas: {obs_text}"
    prompt = prompt_template.format(obs_frames, pred_frames, body)
    completion = f"Next motion deltas: {pred_text}"
    return prompt_system, prompt, completion, sample_id


def convert_split(
    input_dir: Path,
    output_path: Path,
    split_name: str,
    sample_id_filter: Optional[str] = None,
    max_windows: Optional[int] = None,
    obs_limit: Optional[int] = None,
    pred_limit: Optional[int] = None,
    joint_filter: Optional[List[str]] = None,
    decimal_digits: int = DEC_DIGITS,
) -> int:
    """split 디렉토리의 JSON 파일들을 JSONL로 변환.

    sample_id_filter가 주어지면 sample_id가 일치하는(또는 prefix 일치) 파일만 처리한다.
    max_windows가 주어지면 처음 N개만 사용한다(단일 sample sanity check용).
    """
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] {input_dir} 에 JSON 파일이 없습니다.")
        return 0

    if sample_id_filter:
        # sample_id_filter는 sample_id의 prefix 또는 정확 일치(예: "000000" 또는 "000000_w0000")
        json_files = [
            p for p in json_files
            if p.stem == sample_id_filter or p.stem.startswith(f"{sample_id_filter}_w")
        ]
        if not json_files:
            print(f"[WARN] sample_id_filter='{sample_id_filter}'에 일치하는 JSON 없음")
            return 0
        print(f"[INFO] sample_id_filter 적용: {len(json_files)}개 윈도")

    if max_windows is not None and max_windows > 0:
        json_files = json_files[:max_windows]
        print(f"[INFO] max_windows={max_windows} 적용 후 {len(json_files)}개 윈도")

    saved = 0
    skipped = 0

    meta = _build_meta(
        track="delta",
        split=split_name,
        source_dir=input_dir,
        sample_id=sample_id_filter,
    )
    # 정책 정보를 _meta에 포함
    meta["_meta"]["data_policy"] = {
        "obs_limit": obs_limit,
        "pred_limit": pred_limit,
        "joint_filter": joint_filter,
        "decimal_digits": decimal_digits,
    }

    with open(output_path, "w", encoding="utf-8") as fout:
        # data-versioning §2-2: 첫 줄에 _meta 레코드
        fout.write(json.dumps(meta, ensure_ascii=False) + "\n")

        for json_path in json_files:
            try:
                system, prompt, completion, sample_id = convert_file(
                    json_path,
                    obs_limit=obs_limit,
                    pred_limit=pred_limit,
                    joint_filter=joint_filter,
                    decimal_digits=decimal_digits,
                )
            except (ValueError, TypeError) as e:
                skipped += 1
                continue
            if not prompt or not completion:
                skipped += 1
                continue
            record = {
                "system": system,
                "prompt": prompt,
                "completion": completion,
                "sample_id": sample_id,
                "task": "trajectory_delta_3d",
                "split": split_name,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1

    print(f"[{split_name}] saved={saved}, skipped={skipped} -> {output_path}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D delta JSONL 생성기 (data-versioning §2-2 _meta 동봉)"
    )
    parser.add_argument(
        "--input-dir", default="./processed",
        help="02_humanml3d_to_pose_json.py 출력 폴더 (train/val/test 하위 폴더 포함)"
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="JSONL 저장 경로 (기본: 현재 디렉토리)"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="처리할 split 목록"
    )
    parser.add_argument(
        "--max-obs-frames", type=int, default=20,
        help="관측 구간 최대 프레임 수 (기본 20)"
    )
    parser.add_argument(
        "--sample-id", default=None,
        help="단일 sample 모드 — 지정 sample_id(prefix 또는 정확 일치)만 변환. "
             "H-2026-003 sanity check용.",
    )
    parser.add_argument(
        "--max-windows", type=int, default=None,
        help="sample-id 모드에서 처음 N개 윈도만 사용 (Stage 0=1, Stage 1=None).",
    )
    parser.add_argument(
        "--output-name", default=None,
        help="출력 파일명을 직접 지정. 미지정 시 finetune_3d_delta_<split>.jsonl. "
             "단일 sample 모드 권장 명명: finetune_3d_delta_singlesample_<id>_stage<n>.jsonl",
    )
    parser.add_argument(
        "--obs-limit", type=int, default=None,
        help="obs frames 앞에서부터 N 프레임만 사용 (정책 X: 5)",
    )
    parser.add_argument(
        "--pred-limit", type=int, default=None,
        help="pred frames 앞에서부터 N 프레임만 사용 (정책 X: 3)",
    )
    parser.add_argument(
        "--joint-subset", default=None,
        choices=[None, "upper-body-12", "all"],
        help="관절 subset. upper-body-12: 상체 12관절(정책 X). all 또는 미지정: 전체 22관절.",
    )
    parser.add_argument(
        "--decimal-digits", type=int, default=DEC_DIGITS,
        help=f"DEC_DIGITS override (기본 {DEC_DIGITS}, 정책 X: 3)",
    )
    args = parser.parse_args()

    if args.joint_subset == "upper-body-12":
        joint_filter = UPPER_BODY_12
    else:
        joint_filter = None

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in args.splits:
        split_dir = input_dir / split
        if not split_dir.exists():
            print(f"[WARN] {split_dir} 없음, 건너뜀")
            continue
        if args.output_name:
            out_path = output_dir / args.output_name
        else:
            out_path = output_dir / f"finetune_3d_delta_{split}.jsonl"
        saved = convert_split(
            split_dir,
            out_path,
            split,
            sample_id_filter=args.sample_id,
            max_windows=args.max_windows,
            obs_limit=args.obs_limit,
            pred_limit=args.pred_limit,
            joint_filter=joint_filter,
            decimal_digits=args.decimal_digits,
        )
        total += saved

    print(f"[INFO] 전체 저장: {total}개 샘플 (+ 첫 줄 _meta 1개)")
    print("[INFO] 다음 단계: python 05_finetune_lora_3d.py --train-jsonl finetune_3d_delta_train.jsonl")


if __name__ == "__main__":
    main()
