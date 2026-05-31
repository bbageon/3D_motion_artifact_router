"""Step 2-G v2 (사용자 directive 2026-05-31): motion-group-stratified prompt selector.

사용자 directive 박제:
> "HumanML3D test captions 을 motion-group 분류 → 부족한 그룹 중심으로 G2 추가 생성. 기존
>  300 + 신규 300 = 총 600 목표. 비율 (walking 25% / running 10% / turning 10% / jumping
>  10% / kick-squat-lunge 15% / dance 10% / sit-stand 10% / upper-body 10%). 정책이
>  walking 중심 artifact 에만 학습 되는 위험 회피."

본 도구는:
  (1) 기존 G2 pool (300) 의 prompt 를 classifier 로 분류.
  (2) HumanML3D test split 의 4,384 caption 를 동일 classifier 로 분류.
  (3) target 600 × user_ratio 와 기존 의 gap 계산.
  (4) 기존 pool 의 prompt 와 disjoint 한 신규 300 prompt 를 group-stratified 로 선별.
  (5) prompt file (g2_generate_batch.py 의 입력) + per-prompt metadata 동시 출력.

CLI:
    python -m tools.g2_balanced_prompt_selector \
        --output-prompts evals/prompts/humanml3d_balanced300_seed20260531.txt \
        --output-meta evals/prompts/humanml3d_balanced300_seed20260531_meta.json

근거 (AGENTS.md §3-22): action/state coverage stratification — locomotion overfit 위험은 학습
데이터의 동작 분포에서 결정. HumanML3D (Guo 2022 CVPR) text-motion 다양성 보장.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HML3D = REPO_ROOT / "external_assets" / "HumanML3D"
DEFAULT_EXISTING_POOL = REPO_ROOT / "external_assets" / "g2_generated_hml3d_test300_seed20260527"

# Motion-group keyword classifier (사용자 8 그룹 + other).
# 우선순위: 더 구체적인 그룹 (jump/run/turn/dance/kick) 먼저, walking 은 나중에 (walk 키워드가 너무 흔하므로).
GROUPS_ORDER = [
    ("jump_hop", ["jump", "jumps", "jumping", "hop", "hops", "hopping", "leap", "leaps", "leaping", "bounce", "bounces", "bouncing"]),
    ("run_jog", ["run", "runs", "running", "jog", "jogs", "jogging", "sprint", "sprinting", "dash", "dashes"]),
    ("dance_rhythmic", ["dance", "dances", "dancing", "sway", "sways", "swaying", "wiggle", "wiggles", "wiggling", "clap", "claps", "clapping", "twirl", "twirls", "twirling"]),
    ("kick_squat_lunge", ["kick", "kicks", "kicking", "squat", "squats", "squatting", "lunge", "lunges", "lunging", "crouch", "crouches", "crouching", "knee", "knees"]),
    ("turn_path", ["turn", "turns", "turning", "pivot", "pivots", "pivoting", "circle", "clockwise", "counterclockwise", "spin", "spins", "spinning", "rotate", "rotates"]),
    ("sit_stand_transition", ["sit down", "sits down", "sitting down", "stand up", "stands up", "standing up", "bend", "bends", "bending", "kneel", "kneels", "kneeling", "lie down", "lies down", "lay down"]),
    ("walk_locomotion", ["walk", "walks", "walking", "step", "steps", "stepping", "stroll", "strolls", "strolling", "pace", "paces", "march", "marches", "marching"]),
    ("upper_body_low_loco", ["wave", "waves", "waving", "raise", "raises", "raising", "reach", "reaches", "reaching", "salute", "salutes", "saluting", "stretch", "stretches", "stretching", "throw", "throws", "throwing", "catch", "catches", "catching", "punch", "punches", "punching"]),
]


def classify(prompt: str) -> str:
    """우선순위 기반 motion-group 분류. 첫 매치 그룹 반환."""
    p = prompt.lower()
    for g, kws in GROUPS_ORDER:
        for kw in kws:
            # multi-word phrase 는 그대로 검색; 단어 는 word-boundary.
            if " " in kw:
                if kw in p:
                    return g
            elif re.search(r"\b" + re.escape(kw) + r"\b", p):
                return g
    return "other"


def _load_existing_prompts(pool_dir: Path) -> tuple[set[str], Counter]:
    prompts = set()
    groups = Counter()
    for p in sorted(pool_dir.glob("motion_*.json")):
        meta = json.load(open(p, encoding="utf-8"))
        prompt = meta.get("prompt", "").strip()
        if not prompt:
            continue
        prompts.add(prompt)
        groups[classify(prompt)] += 1
    return prompts, groups


def _load_humanml3d_pool(hml3d_root: Path, split: str, exclude_prompts: set[str]) -> list[dict]:
    """HumanML3D split 의 (sample_id, caption) 목록, exclude prompt 와 disjoint."""
    ids_path = hml3d_root / f"{split}.txt"
    with open(ids_path, encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    rows = []
    for sid in ids:
        text_path = hml3d_root / "texts" / f"{sid}.txt"
        if not text_path.exists():
            continue
        with open(text_path, encoding="utf-8") as fh:
            for line in fh:
                cap = line.split("#", 1)[0].strip()
                if cap and cap not in exclude_prompts:
                    rows.append({"sample_id": sid, "prompt": cap, "group": classify(cap)})
                    break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml3d-root", type=Path, default=DEFAULT_HML3D)
    parser.add_argument("--existing-pool", type=Path, default=DEFAULT_EXISTING_POOL)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--n-new", type=int, default=300)
    parser.add_argument("--target-total", type=int, default=600,
                        help="기존 + 신규 의 총 목표 pool 크기")
    parser.add_argument("--seed", type=int, default=20260531)
    # 사용자 target ratio (8 group).
    parser.add_argument("--target-ratios", type=str,
                        default="walk_locomotion:0.25,run_jog:0.10,turn_path:0.10,jump_hop:0.10,kick_squat_lunge:0.15,dance_rhythmic:0.10,sit_stand_transition:0.10,upper_body_low_loco:0.10")
    parser.add_argument("--output-prompts", type=Path,
                        default=REPO_ROOT / "evals" / "prompts" / "humanml3d_balanced300_seed20260531.txt")
    parser.add_argument("--output-meta", type=Path,
                        default=REPO_ROOT / "evals" / "prompts" / "humanml3d_balanced300_seed20260531_meta.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Existing pool 분석.
    existing_prompts, existing_groups = _load_existing_prompts(args.existing_pool)
    print(f"[INFO] existing pool: {len(existing_prompts)} prompts")
    for g, _ in GROUPS_ORDER:
        n = existing_groups.get(g, 0)
        print(f"   existing {g:<26} {n:3d}  ({n/max(len(existing_prompts),1)*100:5.1f}%)")
    print(f"   existing {'other':<26} {existing_groups.get('other', 0):3d}")

    # Target counts (600 × ratio).
    targets = {}
    for spec in args.target_ratios.split(","):
        g, r = spec.split(":")
        targets[g.strip()] = int(round(args.target_total * float(r)))
    # Gap (target - existing), positive only.
    gap_raw = {g: max(0, targets[g] - existing_groups.get(g, 0)) for g in targets}
    gap_sum = sum(gap_raw.values())
    print(f"\n[INFO] target_total={args.target_total}, sum_gap_raw={gap_sum}")
    # Scale gaps to fit n_new budget.
    n_new = args.n_new
    if gap_sum > n_new:
        scaled_gap = {g: int(round(gap_raw[g] * n_new / gap_sum)) for g in gap_raw}
    else:
        scaled_gap = dict(gap_raw)
    # Adjust for rounding (sum to exactly n_new).
    diff = n_new - sum(scaled_gap.values())
    if diff != 0:
        # Add/subtract on largest-gap groups deterministically.
        ordered = sorted(gap_raw.keys(), key=lambda g: -gap_raw[g])
        for g in ordered:
            if diff == 0:
                break
            scaled_gap[g] += 1 if diff > 0 else -1
            diff -= 1 if diff > 0 else -1
    print(f"\n  group        target  existing  raw_gap  scaled_gap")
    for g, _ in GROUPS_ORDER:
        print(f"   {g:<22} {targets[g]:<4} {existing_groups.get(g,0):<4} {gap_raw[g]:<4} {scaled_gap[g]}")

    # HumanML3D test pool (disjoint from existing prompts).
    pool = _load_humanml3d_pool(args.humanml3d_root, args.split, existing_prompts)
    pool_by_group = defaultdict(list)
    for row in pool:
        pool_by_group[row["group"]].append(row)
    print(f"\n[INFO] HumanML3D {args.split} pool (disjoint): {len(pool)} prompts")
    for g, _ in GROUPS_ORDER:
        print(f"   available {g:<24} {len(pool_by_group.get(g, [])):4d}")
    print(f"   available {'other':<24} {len(pool_by_group.get('other', [])):4d}")

    # Select per group.
    selected = []
    shortage = {}
    for g, _ in GROUPS_ORDER:
        need = scaled_gap.get(g, 0)
        available = pool_by_group.get(g, [])
        if len(available) < need:
            shortage[g] = need - len(available)
            need = len(available)
        if need <= 0:
            continue
        rng.shuffle(available)
        picked = available[:need]
        # 같은 HumanML3D sample_id 가 두 번 뽑히지 않게 (실제로는 첫 caption만 사용해 disjoint 보장됨).
        selected.extend([{**p, "selection_group": g} for p in picked])

    # Shortage 가 있으면 다른 그룹 (other 우선) 에서 보충.
    target_n = sum(scaled_gap.values())
    if len(selected) < target_n:
        deficit = target_n - len(selected)
        used_ids = {s["sample_id"] for s in selected}
        # 'other' 그룹부터, 그 다음 가장 큰 available group.
        backup = []
        for g in ["other"] + sorted([g for g, _ in GROUPS_ORDER], key=lambda g: -len(pool_by_group.get(g, []))):
            backup.extend([p for p in pool_by_group.get(g, []) if p["sample_id"] not in used_ids])
        rng.shuffle(backup)
        for p in backup[:deficit]:
            selected.append({**p, "selection_group": "backup_other_or_overflow"})

    selected = selected[:n_new]
    print(f"\n[INFO] selected: {len(selected)} prompts")
    sel_groups = Counter(s["group"] for s in selected)
    for g, _ in GROUPS_ORDER:
        print(f"   selected {g:<24} {sel_groups.get(g, 0):3d}")
    print(f"   selected {'other':<24} {sel_groups.get('other', 0):3d}")
    if shortage:
        print(f"\n[WARN] shortage 발생 (HumanML3D pool 부족): {shortage}")

    # Final post-merge distribution check.
    print(f"\n[INFO] projected merged (existing + new) distribution vs target:")
    merged = Counter()
    merged.update(existing_groups)
    merged.update(sel_groups)
    for g, _ in GROUPS_ORDER:
        n = merged.get(g, 0); pct = n / (len(existing_prompts) + len(selected)) * 100
        tgt = targets[g]
        print(f"   {g:<26} merged {n:3d} ({pct:5.1f}%)  vs target {tgt} ({tgt/args.target_total*100:5.1f}%)")

    # Write prompt file (g2_generate_batch.py 호환 — 한 prompt 당 한 줄).
    args.output_prompts.parent.mkdir(parents=True, exist_ok=True)
    args.output_prompts.write_text("\n".join(s["prompt"] for s in selected) + "\n", encoding="utf-8")
    # Write metadata.
    meta = {
        "schema_version": "1.0.0", "record_type": "g2_balanced_prompt_meta",
        "task_id": "humanml3d_balanced300_seed20260531",
        "humanml3d_split": args.split, "seed": args.seed,
        "n_new": len(selected), "target_total": args.target_total,
        "target_ratios": {g: targets[g]/args.target_total for g, _ in GROUPS_ORDER},
        "existing_pool": str(args.existing_pool.relative_to(REPO_ROOT)),
        "existing_group_counts": dict(existing_groups),
        "scaled_gap": scaled_gap,
        "shortage": shortage,
        "merged_projected_counts": dict(merged),
        "prompts": [
            {"line_index": i, "sample_id": s["sample_id"], "prompt": s["prompt"],
             "group": s["group"], "selection_group": s["selection_group"]}
            for i, s in enumerate(selected)
        ],
    }
    args.output_meta.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] prompts -> {args.output_prompts}")
    print(f"[OK] meta    -> {args.output_meta}")


if __name__ == "__main__":
    main()
