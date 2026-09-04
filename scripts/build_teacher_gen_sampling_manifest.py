#!/usr/bin/env python3
"""Build a balanced Teacher Generation sampling manifest for pick-trocar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


LEFT_ARM = slice(15, 22)
LEFT_HAND = slice(22, 29)
RAW_WINDOW_SPAN = 26  # 14 source samples at stride 2: indices 0, 2, ..., 26.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("datasets/g1_pick_trocar_rollout_all_260_headcam_train"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-samples", type=int, default=10_000)
    parser.add_argument("--keyframe-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2500)
    return parser.parse_args()


def load_episodes(dataset_path: Path) -> list[dict]:
    path = dataset_path / "meta" / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def parquet_path(dataset_path: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return dataset_path / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def action_scores(dataset_path: Path, episode: dict) -> np.ndarray:
    path = parquet_path(dataset_path, int(episode["episode_index"]))
    actions = np.asarray(pq.read_table(path, columns=["action"])["action"].to_pylist(), dtype=np.float32)
    if len(actions) != int(episode["length"]):
        raise ValueError(f"Length mismatch for {path}: {len(actions)} != {episode['length']}")

    changes = np.zeros(len(actions), dtype=np.float32)
    if len(actions) > 1:
        delta = np.diff(actions, axis=0)
        arm = np.linalg.norm(delta[:, LEFT_ARM], axis=1)
        hand = np.linalg.norm(delta[:, LEFT_HAND], axis=1)
        changes[1:] = 0.35 * arm + 0.65 * hand

    num_windows = len(actions) - RAW_WINDOW_SPAN
    if num_windows <= 0:
        raise ValueError(f"Episode {episode['episode_index']} is too short")

    # Score a local interval rather than one frame so windows include approach,
    # contact, and lift context around abrupt left-hand changes.
    scores = np.empty(num_windows, dtype=np.float32)
    for base_index in range(num_windows):
        local = changes[base_index : base_index + RAW_WINDOW_SPAN + 1]
        scores[base_index] = float(local.max() + 0.25 * local.mean())
    return scores


def allocate_counts(num_episodes: int, total_samples: int) -> list[int]:
    quotient, remainder = divmod(total_samples, num_episodes)
    return [quotient + (position < remainder) for position in range(num_episodes)]


def choose_windows(
    scores: np.ndarray,
    count: int,
    keyframe_fraction: float,
    rng: np.random.Generator,
) -> list[tuple[int, str]]:
    keyframe_count = round(count * keyframe_fraction)
    uniform_count = count - keyframe_count
    candidates = np.arange(len(scores), dtype=np.int64)

    uniform = np.unique(np.rint(np.linspace(0, len(scores) - 1, uniform_count)).astype(np.int64))
    if len(uniform) < uniform_count:
        remaining = np.setdiff1d(candidates, uniform, assume_unique=True)
        uniform = np.concatenate([uniform, remaining[: uniform_count - len(uniform)]])

    remaining = np.setdiff1d(candidates, uniform, assume_unique=True)
    if keyframe_count > len(remaining):
        raise ValueError(f"Requested {count} unique windows from only {len(scores)} candidates")

    remaining_scores = scores[remaining].astype(np.float64)
    scale = float(np.quantile(remaining_scores, 0.9))
    if scale <= 0:
        probabilities = np.full(len(remaining), 1.0 / len(remaining))
    else:
        normalized = np.clip(remaining_scores / scale, 0.0, 3.0)
        probabilities = 0.05 + normalized**2
        probabilities /= probabilities.sum()
    keyframes = rng.choice(remaining, size=keyframe_count, replace=False, p=probabilities)

    selected = [(int(index), "uniform") for index in uniform]
    selected.extend((int(index), "left_action_weighted") for index in keyframes)
    return sorted(selected)


def main() -> None:
    args = parse_args()
    if not 0 <= args.keyframe_fraction <= 1:
        raise ValueError("--keyframe-fraction must be in [0, 1]")

    episodes = load_episodes(args.dataset_path)
    counts = allocate_counts(len(episodes), args.total_samples)
    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    global_offset = 0

    for episode_position, (episode, count) in enumerate(zip(episodes, counts, strict=True)):
        scores = action_scores(args.dataset_path, episode)
        selected = choose_windows(scores, count, args.keyframe_fraction, rng)
        for base_index, sampling_type in selected:
            records.append(
                {
                    "output_index": -1,
                    "dataset_index": global_offset + base_index,
                    "episode_position": episode_position,
                    "episode_index": int(episode["episode_index"]),
                    "base_index": base_index,
                    "sampling_type": sampling_type,
                    "left_action_score": float(scores[base_index]),
                }
            )
        global_offset += len(scores)

    rng.shuffle(records)
    for output_index, record in enumerate(records):
        record["output_index"] = output_index

    if len(records) != args.total_samples:
        raise RuntimeError(f"Expected {args.total_samples} records, got {len(records)}")
    if len({record["dataset_index"] for record in records}) != len(records):
        raise RuntimeError("Manifest contains duplicate dataset indices")
    if len({record["episode_index"] for record in records}) != len(episodes):
        raise RuntimeError("Manifest does not cover every training episode")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    sampling_counts = {
        name: sum(record["sampling_type"] == name for record in records)
        for name in ("uniform", "left_action_weighted")
    }
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(records),
                "episodes": len(episodes),
                "sampling_counts": sampling_counts,
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
