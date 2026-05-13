"""
3D delta 프롬프트 빌더 (핵심 모듈)

2D delta/make_prompt_from_pose.py를 3D로 확장한 버전.
- 입력: pose_sequence_3d (각 프레임에 22개 SMPL joint의 [x,y,z])
- 출력: 학습/추론용 delta 토큰 프롬프트 문자열

숫자 토큰 포맷 (2D와 동일):
  -[NUM][INT]{int:03d}[SEP][DEC]{dec:05d}[ENDNUM]

3D 프레임 토큰 포맷 (2D 확장):
  JOINT:(tok_x,tok_y,tok_z),(tok_x,tok_y,tok_z),...

프롬프트 형식:
  Observed motion deltas: PELVIS:(tok,tok,tok),... | LEFT_HIP:(...) | ...
  Next motion deltas:
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEC_DIGITS = 5
DEC_SCALE = 10 ** DEC_DIGITS

MARKER_OBS = "Observed motion deltas:"
MARKER_NEXT = "Next motion deltas:"

PROMPT_SYSTEM = (
    "You are a 3D motion prediction assistant that extrapolates future joint movements "
    "based on observed delta (dx, dy, dz) coordinate sequences. "
    "IMPORTANT: Provide EXACTLY the requested number of frames. Do not provide more or less."
)

PROMPT_TEMPLATE = (
    "Forecast the next {pred_len:d} (x, y, z) coordinate deltas for all observed joints "
    "using the given {obs_len:d} observed delta frames.\n"
    "Each delta value must follow this token format: -[NUM][INT]000[SEP][DEC]00000[ENDNUM] "
    "(minus sign optional).\n"
    "Return one line in this exact format:\n"
    "Next motion deltas: JOINT:(tok,tok,tok),(tok,tok,tok),... | JOINT:(tok,tok,tok),...\n"
    "### Observed Delta Sequences ###\n"
    "{obs_text}"
)


def num_to_tokens(x: float) -> str:
    """float -> 구조화 숫자 토큰 (2D 프로젝트와 동일)"""
    sign = "-" if x < 0 else ""
    x = abs(x)
    int_part = int(x)
    dec_part = int(round((x - int_part) * DEC_SCALE))
    if dec_part >= DEC_SCALE:
        int_part += 1
        dec_part -= DEC_SCALE
    return f"{sign}[NUM][INT]{int_part:03d}[SEP][DEC]{dec_part:0{DEC_DIGITS}d}[ENDNUM]"


def decode_token_number(token_str: str) -> float:
    """구조화 숫자 토큰 -> float (2D 프로젝트와 동일)"""
    pattern = r"(-?)\[NUM\]\[INT\](\d{3})\[SEP\]\[DEC\](\d{5})\[ENDNUM\]"
    m = re.search(pattern, token_str)
    if not m:
        raise ValueError(f"Invalid token format: {token_str!r}")
    sign = -1 if m.group(1) == "-" else 1
    int_part = int(m.group(2))
    dec_part = int(m.group(3))
    return round(sign * (int_part + dec_part / DEC_SCALE), DEC_DIGITS + 1)


def format_pose_sequence_deltas_3d(
    seq: List[Dict],
    max_frames: int = 20,
) -> str:
    """
    pose_sequence_3d -> 3D delta 토큰 문자열
    seq[i]: {"JOINT_NAME": [x, y, z], ...}
    반환: "JOINT:(tok_x,tok_y,tok_z),... | JOINT:..."
    """
    if len(seq) < 2:
        return ""

    joints_list = list(seq[0].keys())
    parts = []

    for joint in joints_list:
        deltas = []
        for i in range(1, len(seq)):
            if joint not in seq[i] or joint not in seq[i - 1]:
                continue
            x1, y1, z1 = seq[i - 1][joint]
            x2, y2, z2 = seq[i][joint]
            dx = round(x2 - x1, DEC_DIGITS)
            dy = round(y2 - y1, DEC_DIGITS)
            dz = round(z2 - z1, DEC_DIGITS)
            deltas.append(
                f"({num_to_tokens(dx)},{num_to_tokens(dy)},{num_to_tokens(dz)})"
            )
        if deltas:
            parts.append(f"{joint}:{','.join(deltas[:max_frames])}")

    return " | ".join(parts)


def parse_deltas_3d(
    text: str,
    marker: str,
) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    모델 출력 텍스트 -> joint별 (dx, dy, dz) 리스트 딕셔너리

    2D parse_deltas 의 3D 확장 버전.
    토큰 패턴: (-?[NUM][INT]ddd[SEP][DEC]ddddd[ENDNUM])가 3개
    """
    start = text.find(marker)
    if start != -1:
        text = text[start + len(marker):]
    text = text.strip()
    if not text:
        return {}

    tok_pat = r"-?\[NUM\]\[INT\]\d{3}\[SEP\]\[DEC\]\d{5}\[ENDNUM\]"
    frame_pat = re.compile(
        rf"\(\s*({tok_pat})\s*,\s*({tok_pat})\s*,\s*({tok_pat})\s*\)"
    )
    # joint name regex: SPINE1·SPINE2·SPINE3 등 숫자 포함 joint 호환 (이전 [A-Z][A-Z_]*:은 누락 → 2026-05-06 H-2026-004 Stage 1에서 발견·수정)
    joint_pat = re.compile(r"([A-Z][A-Z0-9_]*):")

    result: Dict[str, List[Tuple[float, float, float]]] = {}
    matches = list(joint_pat.finditer(text))
    for i, jm in enumerate(matches):
        joint = jm.group(1)
        seg_start = jm.end()
        seg_end = matches[i + 1].start() if (i + 1) < len(matches) else len(text)
        seg = text[seg_start:seg_end]
        frames = []
        for x_tok, y_tok, z_tok in frame_pat.findall(seg):
            try:
                frames.append((
                    decode_token_number(x_tok),
                    decode_token_number(y_tok),
                    decode_token_number(z_tok),
                ))
            except ValueError:
                continue
        if frames:
            result[joint] = frames
    return result


def accumulate_3d(
    base_xyz: Tuple[float, float, float],
    deltas: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    """
    기준 좌표 + delta 시퀀스 -> 절대 좌표 시퀀스
    반환 리스트 첫 번째 원소는 기준 프레임 포함
    """
    x, y, z = base_xyz
    coords = [(x, y, z)]
    for dx, dy, dz in deltas:
        x += dx
        y += dy
        z += dz
        coords.append((x, y, z))
    return coords


def build_absolute_positions_3d(
    baseline_frame: Dict,
    deltas_by_joint: Dict[str, List[Tuple[float, float, float]]],
) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    기준 프레임 + joint별 delta -> joint별 절대 좌표 시퀀스
    """
    return {
        j: accumulate_3d(tuple(baseline_frame[j]), seq)
        for j, seq in deltas_by_joint.items()
        if j in baseline_frame
    }


def create_prompt_from_json(
    path_or_data,
    pred_len: int = 10,
    max_obs: int = 20,
    with_system: bool = True,
    prompt_style: str = "train",
) -> str:
    """
    JSON 파일 경로 또는 데이터 객체 -> 3D delta 프롬프트 문자열

    prompt_style:
      "train": "Observed motion deltas: ..." (학습/JSONL 포맷)
      "instruct": PROMPT_TEMPLATE 래핑 포함 (추론용)
    """
    if isinstance(path_or_data, (str, Path)):
        with open(path_or_data, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = path_or_data

    obs_seq = data.get("obs_sequence_3d", [])
    if not obs_seq:
        return f"{MARKER_OBS} "

    obs_text = format_pose_sequence_deltas_3d(obs_seq, max_frames=max_obs)

    if prompt_style == "train":
        return f"{MARKER_OBS} {obs_text}"

    obs_len = max_obs
    base_prompt = PROMPT_TEMPLATE.format(pred_len=pred_len, obs_len=obs_len, obs_text=obs_text)
    if with_system:
        return f"{PROMPT_SYSTEM}\n\n{base_prompt}"
    return base_prompt


def build_instruct_prompt_from_observed(
    observed_prompt: str,
    pred_len: int = 10,
    with_system: bool = True,
) -> str:
    """
    학습용 train-style 프롬프트 -> instruct-style 프롬프트 변환

    입력: "Observed motion deltas: JOINT:(tok,tok,tok),... | ..."
    """
    text = (observed_prompt or "").strip()
    prefix = MARKER_OBS
    obs_text = text[len(prefix):].strip() if text.startswith(prefix) else text

    obs_len = 0
    if obs_text:
        first_seg = obs_text.split(" | ", 1)[0]
        if ":" in first_seg:
            first_seg = first_seg.split(":", 1)[1]
        tok_pat = r"-?\[NUM\]\[INT\]\d{3}\[SEP\]\[DEC\]\d{5}\[ENDNUM\]"
        frame_pat = re.compile(
            rf"\(\s*{tok_pat}\s*,\s*{tok_pat}\s*,\s*{tok_pat}\s*\)"
        )
        obs_len = len(frame_pat.findall(first_seg))

    base_prompt = PROMPT_TEMPLATE.format(pred_len=pred_len, obs_len=obs_len, obs_text=obs_text)
    if with_system:
        return f"{PROMPT_SYSTEM}\n\n{base_prompt}"
    return base_prompt
