"""AR-048: corrected cross-generator pool generation (protocol freeze validation).

교정 prompt bank 의 각 prompt 를 3 generator × N seed 로 생성, 각 모션을 trajectory +
local 이중 표현으로 저장한다. generator 별 디렉터리 분리(§3-5), GT 기반 target_length,
prompt당 multi-seed(§5). seed = base_seed + prompt_idx*1000 + seed_idx (결정적).

본 도구는 quality 보정이 아니라 표현·provenance adapter (AR-048 spec).

CLI (motion3d env, docker 서비스 가동 필요):
    python tools/generate_protocol_pool.py \
        --prompt-bank evals/prompts/protocol_v1_test_30_seed20260607.json \
        --generators mdm momask motiongpt --n-seeds 3 --base-seed 20260607 \
        --output-root external_assets/protocol_v1_pool_seed20260607
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from generators._http_client import generate_dual_via_http, service_url
from tools.coords_protocol import estimate_ground, local_matches_trajectory


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-bank", type=Path,
                    default=REPO_ROOT / "evals" / "prompts" / "protocol_v1_test_30_seed20260607.json")
    ap.add_argument("--generators", nargs="+", default=["mdm", "momask", "motiongpt"])
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=20260607)
    ap.add_argument("--output-root", type=Path,
                    default=REPO_ROOT / "external_assets" / "protocol_v1_pool_seed20260607")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    args.prompt_bank = args.prompt_bank.resolve()
    args.output_root = args.output_root.resolve()

    bank = json.load(open(args.prompt_bank, encoding="utf-8"))
    rows = bank["rows"]
    args.output_root.mkdir(parents=True, exist_ok=True)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    overall = {"created_at": _now(), "board_id": "AR-048", "record_type": "protocol_pool_generation",
               "prompt_bank": _rel(args.prompt_bank), "n_prompts": len(rows),
               "generators": args.generators, "n_seeds": args.n_seeds, "base_seed": args.base_seed,
               "seed_rule": "base_seed + prompt_idx*1000 + seed_idx",
               "length_rule": "request target_length; truncate to target if generator returns longer",
               "generators_summary": {}}

    for gen in args.generators:
        gen_dir = args.output_root / gen
        gen_dir.mkdir(parents=True, exist_ok=True)
        base = service_url(gen)
        n_ok = n_fail = 0
        samples = []
        for p_idx, row in enumerate(rows):
            prompt = row["prompt"]
            target = int(row["target_length"])
            for s_idx in range(args.n_seeds):
                seed = args.base_seed + p_idx * 1000 + s_idx
                stem = f"{row['sample_id']}__seed{seed}"
                traj_p = gen_dir / f"{stem}.trajectory.npy"
                local_p = gen_dir / f"{stem}.local.npy"
                meta_p = gen_dir / f"{stem}.json"
                if args.skip_existing and traj_p.exists() and local_p.exists() and meta_p.exists():
                    samples.append(json.load(open(meta_p, encoding="utf-8")))
                    n_ok += 1
                    continue
                try:
                    traj, local, meta = generate_dual_via_http(base, prompt, target, seed)
                except Exception as exc:
                    print(f"  [FAIL {gen} {stem}] {exc}", file=sys.stderr)
                    n_fail += 1
                    continue
                if traj is None:
                    raise RuntimeError(f"{gen} returned no motion_trajectory — docker 재빌드 필요 (dual 미지원)")
                actual = int(traj.shape[0])
                final = min(actual, target)
                traj, local = traj[:final], local[:final]
                np.save(traj_p, traj.astype(np.float32))
                np.save(local_p, local.astype(np.float32))
                sm = {
                    "sample_id": row["sample_id"], "prompt": prompt, "prompt_idx": p_idx,
                    "seed": seed, "seed_idx": s_idx, "generator": gen,
                    "generator_id": meta.get("generator_id"),
                    "target_length": target, "gt_length": row["gt_length"],
                    "actual_generated": actual, "final_length": final,
                    "length_matches_target": bool(final == target),
                    "ground_y": estimate_ground(traj),
                    "local_matches_trajectory": local_matches_trajectory(local, traj),
                    "compute_backend": meta.get("compute_backend"),
                    "elapsed_sec": meta.get("elapsed_sec"),
                    "trajectory_npy": _rel(traj_p),
                    "local_npy": _rel(local_p),
                }
                json.dump(sm, open(meta_p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
                samples.append(sm)
                n_ok += 1
            print(f"[{gen}] prompt {p_idx+1}/{len(rows)} done (ok={n_ok} fail={n_fail})", flush=True)
        overall["generators_summary"][gen] = {"n_ok": n_ok, "n_fail": n_fail, "dir": _rel(gen_dir)}

    summary_p = args.output_root / "_pool_summary.json"
    json.dump(overall, open(summary_p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[OK] pool summary: {summary_p}")
    for gen, s in overall["generators_summary"].items():
        print(f"   {gen:<10} ok={s['n_ok']} fail={s['n_fail']}")


if __name__ == "__main__":
    main()
