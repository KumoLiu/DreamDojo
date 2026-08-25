#!/usr/bin/env python3
"""Select the checkpoint with the lowest mean closed-loop MAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    checkpoint_root = Path(
        "outputs/train/dreamdojo/pick_trocar_self_forcing_97f/"
        "g1_pick_trocar_iter2500_warmup_stratified/checkpoints"
    )
    for candidate in args.candidates:
        records = json.loads((candidate / "metrics.json").read_text())
        mean_mae = sum(record["closed_loop"]["mae"] for record in records) / len(records)
        iteration = int(candidate.name.removeprefix("warmup_iter"))
        summaries.append(
            {
                "evaluation_dir": str(candidate),
                "checkpoint": str(
                    (checkpoint_root / f"iter_{iteration:09d}").resolve()
                ),
                "mean_mae": mean_mae,
            }
        )

    selected = min(summaries, key=lambda item: item["mean_mae"])
    payload = {"selected": selected, "candidates": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(selected["checkpoint"])


if __name__ == "__main__":
    main()
