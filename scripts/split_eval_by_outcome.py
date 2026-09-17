#!/usr/bin/env python3
"""Re-aggregate an eval run split by each episode's success/failure label.

The per-dataset summary mixes successful and failed episodes together, which
hides the case we actually care about: whether the model predicts failure
rollouts as well as successful ones. The validation sets carry a `success`
flag per episode in meta/episodes.jsonl, so this joins that onto the scored
metrics and reports the two groups separately.

    python3 scripts/split_eval_by_outcome.py <eval_dir> [<eval_dir> ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATASETS = {
    "teleop_success": "datasets/g1_hf_pick_trocar_teleop_success_val",
    "rollouts_30k": "datasets/g1_hf_pick_trocar_rollouts_30k_val",
    "rollouts_10k": "datasets/g1_hf_pick_trocar_rollouts_10k_val",
}
ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dirs", type=Path, nargs="+")
    return parser.parse_args()


def success_map(dataset_path: Path) -> dict[int, bool]:
    path = dataset_path / "meta" / "episodes.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "success" in record:
            out[int(record["episode_index"])] = bool(record["success"])
    return out


def collect(eval_dir: Path) -> list[dict]:
    rows = []
    for name, rel in DATASETS.items():
        labels = success_map(ROOT / rel)
        for metrics_file in sorted((eval_dir / name).glob("episode_*/metrics.json")):
            for record in json.loads(metrics_file.read_text()):
                index = int(record["episode_index"])
                if index not in labels:
                    continue
                rows.append(
                    {
                        "dataset": name,
                        "episode": index,
                        "success": labels[index],
                        "tf_psnr": record["teacher_forced"]["psnr"],
                        "tf_mse": record["teacher_forced"]["mse"],
                        "cl_psnr": record["closed_loop"]["psnr"],
                        "cl_mse": record["closed_loop"]["mse"],
                    }
                )
    return rows


def show(title: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  {title:<26} (none)")
        return
    tf = np.array([r["tf_psnr"] for r in rows])
    cl = np.array([r["cl_psnr"] for r in rows])
    tf_mse = np.array([r["tf_mse"] for r in rows])
    cl_mse = np.array([r["cl_mse"] for r in rows])
    print(
        f"  {title:<26} n={len(rows):<4} "
        f"TF {tf.mean():6.3f}+/-{tf.std() / np.sqrt(len(tf)):.3f}  "
        f"CL {cl.mean():6.3f}+/-{cl.std() / np.sqrt(len(cl)):.3f}  "
        f"TF-MSE {tf_mse.mean():8.1f}  CL-MSE {cl_mse.mean():8.1f}"
    )


def main() -> None:
    args = parse_args()
    for eval_dir in args.eval_dirs:
        rows = collect(eval_dir)
        if not rows:
            print(f"=== {eval_dir.name}: no labelled episodes found ===\n")
            continue
        print(f"=== {eval_dir.name} ===")
        succeeded = [r for r in rows if r["success"]]
        failed = [r for r in rows if not r["success"]]
        show("all", rows)
        show("success episodes", succeeded)
        show("failure episodes", failed)
        if succeeded and failed:
            gap_tf = np.mean([r["tf_psnr"] for r in succeeded]) - np.mean(
                [r["tf_psnr"] for r in failed]
            )
            gap_cl = np.mean([r["cl_psnr"] for r in succeeded]) - np.mean(
                [r["cl_psnr"] for r in failed]
            )
            print(f"  gap (success - failure):   TF {gap_tf:+.3f} dB, CL {gap_cl:+.3f} dB")

        print("\n  per dataset, split by outcome:")
        for name in DATASETS:
            subset = [r for r in rows if r["dataset"] == name]
            for label, flag in (("success", True), ("failure", False)):
                show(f"{name} / {label}", [r for r in subset if r["success"] is flag])
        print()


if __name__ == "__main__":
    main()
