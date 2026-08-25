#!/usr/bin/env python3
"""Validate manifest coverage and generated Teacher Generation artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ARTIFACTS = {
    "actions": ".json",
    "images": ".png",
    "latents": ".pt",
    "videos": ".mp4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-samples", type=int, default=10_000)
    parser.add_argument("--expected-episodes", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.root / "sample_manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    expected_indices = set(range(args.expected_samples))

    if len(rows) != args.expected_samples:
        raise RuntimeError(f"Manifest has {len(rows)} rows, expected {args.expected_samples}")
    output_indices = {int(row["output_index"]) for row in rows}
    if output_indices != expected_indices:
        raise RuntimeError("Manifest output indices are incomplete")
    if len({int(row["dataset_index"]) for row in rows}) != len(rows):
        raise RuntimeError("Manifest dataset indices are not unique")

    episode_counts = Counter(int(row["episode_index"]) for row in rows)
    if len(episode_counts) != args.expected_episodes:
        raise RuntimeError(
            f"Manifest covers {len(episode_counts)} episodes, expected {args.expected_episodes}"
        )

    artifact_counts: dict[str, int] = {}
    for subdir, suffix in ARTIFACTS.items():
        found = {int(path.stem) for path in (args.root / subdir).glob(f"*{suffix}")}
        missing = expected_indices - found
        extra = found - expected_indices
        if missing or extra:
            raise RuntimeError(
                f"{subdir}: missing={len(missing)}, extra={len(extra)}, "
                f"first_missing={min(missing) if missing else None}"
            )
        artifact_counts[subdir] = len(found)

    print(
        json.dumps(
            {
                "root": str(args.root),
                "manifest_rows": len(rows),
                "episodes": len(episode_counts),
                "samples_per_episode": sorted(set(episode_counts.values())),
                "artifact_counts": artifact_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
