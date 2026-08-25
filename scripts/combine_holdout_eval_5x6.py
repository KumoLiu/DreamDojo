#!/usr/bin/env python3
"""Reproduce the original local 5x6 success/failure holdout layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapy
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--block-width", type=int, default=1440)
    parser.add_argument("--block-height", type=int, default=360)
    parser.add_argument("--header-height", type=int, default=44)
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def load_cases(dataset_path: Path) -> list[dict]:
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes = [
        json.loads(line)
        for line in episodes_path.read_text().splitlines()
        if line
    ]
    if len(episodes) != 10:
        raise ValueError(f"Expected 10 eval episodes, found {len(episodes)}")
    original_ids = [int(episode["episode_index"]) for episode in episodes]
    expected_ids = [0, 19, 21, 27, 53, 55, 127, 154, 198, 236]
    if original_ids != expected_ids:
        raise ValueError(
            f"Unexpected eval episode IDs: {original_ids}; "
            f"expected {expected_ids}"
        )
    return episodes


def load_comparisons(input_dir: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for position in range(10):
        filename = f"episode_{position:03d}_seed_0_comparison.mp4"
        matches = list(
            input_dir.glob(f"shard_*/episode_{position:03d}/{filename}")
        )
        matches.extend(input_dir.glob(f"episode_{position:03d}/{filename}"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one full-eval video for case {position}, "
                f"found {matches}"
            )
        paths[position] = matches[0]
    return paths


def pad_video(video: np.ndarray, frame_count: int) -> np.ndarray:
    if len(video) >= frame_count:
        return video[:frame_count]
    padding = np.repeat(video[-1:], frame_count - len(video), axis=0)
    return np.concatenate([video, padding], axis=0)


def prepare(
    path: Path,
    title: str,
    color: tuple[int, int, int],
    block_width: int,
    block_height: int,
    header_height: int,
) -> np.ndarray:
    video = mediapy.read_video(str(path))
    frames = []
    for frame in video:
        resized = cv2.resize(
            frame,
            (block_width, block_height),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.zeros(
            (block_height + header_height, block_width, 3),
            dtype=np.uint8,
        )
        canvas[:header_height] = color
        canvas[header_height:] = resized
        cv2.putText(
            canvas,
            title,
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(canvas)
    return np.stack(frames)


def main() -> None:
    args = parse_args()
    cases = load_cases(args.dataset_path)
    paths = load_comparisons(args.input_dir)
    success_positions = [i for i, case in enumerate(cases) if case["success"]]
    failure_positions = [i for i, case in enumerate(cases) if not case["success"]]
    expected_success = [2, 4, 6, 7, 8]
    expected_failure = [0, 1, 3, 5, 9]
    if success_positions != expected_success or failure_positions != expected_failure:
        raise ValueError(
            f"Unexpected split: success={success_positions}, "
            f"failure={failure_positions}"
        )

    rows = []
    for success_pos, failure_pos in zip(
        success_positions,
        failure_positions,
        strict=True,
    ):
        success = prepare(
            paths[success_pos],
            f"SUCCESS | CASE {success_pos:02d}",
            (40, 125, 40),
            args.block_width,
            args.block_height,
            args.header_height,
        )
        failure = prepare(
            paths[failure_pos],
            f"FAILURE | CASE {failure_pos:02d}",
            (45, 45, 170),
            args.block_width,
            args.block_height,
            args.header_height,
        )
        frame_count = max(len(success), len(failure))
        rows.append(
            np.concatenate(
                [
                    pad_video(success, frame_count),
                    pad_video(failure, frame_count),
                ],
                axis=2,
            )
        )

    frame_count = max(map(len, rows))
    combined = np.concatenate(
        [pad_video(row, frame_count) for row in rows],
        axis=1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(args.output), combined, fps=args.fps)
    print(f"Wrote {args.output} with shape {combined.shape}")


if __name__ == "__main__":
    main()
