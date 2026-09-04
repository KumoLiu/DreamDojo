#!/usr/bin/env python3
"""Summarize success/failure separation across real-mix checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def nested_mean(records: list[dict], keys: tuple[str, ...]) -> float:
    values = []
    for record in records:
        value = record
        for key in keys:
            value = value[key]
        values.append(value)
    return float(np.mean(values))


def optional_nested_mean(records: list[dict], keys: tuple[str, ...]) -> float | None:
    try:
        return nested_mean(records, keys)
    except KeyError:
        return None


def summarize(path: Path) -> dict:
    records = json.loads((path / "rollout/metrics.json").read_text())
    success = [record for record in records if record["label"] == "success"]
    failure = [record for record in records if record["label"] == "fail"]
    return {
        "checkpoint": path.name,
        "success_closed_final": nested_mean(
            success, ("reward_probability", "closed", "final")
        ),
        "failure_closed_final": nested_mean(
            failure, ("reward_probability", "closed", "final")
        ),
        "success_closed_psnr": nested_mean(success, ("closed_reconstruction", "psnr")),
        "failure_closed_psnr": nested_mean(failure, ("closed_reconstruction", "psnr")),
        "success_zero_final": optional_nested_mean(
            success, ("reward_probability", "zero", "final")
        ),
        "success_shuffled_final": optional_nested_mean(
            success, ("reward_probability", "shuffled", "final")
        ),
        "failure_action_sensitivity": optional_nested_mean(
            failure, ("action_sensitivity_mae",)
        ),
        "failure_shuffle_sensitivity": optional_nested_mean(
            failure, ("shuffle_sensitivity_mae",)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_root", type=Path)
    root = parser.parse_args().evaluation_root
    summaries = [
        summarize(path)
        for path in sorted(root.glob("iter_*"))
        if (path / "rollout/metrics.json").exists()
    ]
    (root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
