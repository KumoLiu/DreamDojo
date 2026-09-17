#!/usr/bin/env python3
"""Decode episode videos into a flat JPEG cache for classifier training.

Training samples frames in random order across episodes, which is the access
pattern H.264 is worst at: every seek costs a decode from the preceding
keyframe. Decoding each episode once, sequentially, and storing frames
individually turns that into a cheap read.

Frames are stored slightly larger than the training crop so that random
cropping has room to jitter.

    .venv/bin/python -m scripts.milestone.extract_frames
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2

from scripts.milestone.labels import (
    DATASET_ROOT,
    ROLLOUT_DATASETS,
    TELEOP_DATASETS,
    read_episodes,
)

CACHE_ROOT = Path("/localhome/local-yunl/DreamDojo/datasets/milestone_cache")
STORE_SIZE = (360, 270)  # width, height; training crops 320x240 out of this
JPEG_QUALITY = 92


def episode_dir(dataset: str, episode_index: int) -> Path:
    return CACHE_ROOT / dataset / f"episode_{episode_index:06d}"


def frame_path(dataset: str, episode_index: int, frame: int) -> Path:
    return episode_dir(dataset, episode_index) / f"{frame:06d}.jpg"


def _extract(task: tuple[str, int, int, str]) -> tuple[str, int, int, str]:
    dataset, episode_index, expected, video = task
    out = episode_dir(dataset, episode_index)
    existing = len(list(out.glob("*.jpg"))) if out.exists() else 0
    if existing == expected:
        return dataset, episode_index, existing, "cached"
    out.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(video)
    written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if written >= expected:
            # The container can hold trailing frames the parquet does not index.
            break
        resized = cv2.resize(frame, STORE_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(
            str(out / f"{written:06d}.jpg"),
            resized,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        written += 1
    capture.release()

    note = "ok" if written == expected else f"MISMATCH expected {expected}"
    return dataset, episode_index, written, note


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=list(TELEOP_DATASETS + ROLLOUT_DATASETS))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    tasks = []
    for dataset in args.datasets:
        for ep in read_episodes(dataset):
            video = ep.video_path
            if not video.exists():
                print(f"missing video: {video}")
                continue
            tasks.append((dataset, ep.episode_index, ep.length, str(video)))

    print(f"{len(tasks)} episodes to cache into {CACHE_ROOT}")
    problems, total = [], 0
    with Pool(args.workers) as pool:
        for i, (dataset, index, written, note) in enumerate(
            pool.imap_unordered(_extract, tasks), 1
        ):
            total += written
            if note not in ("ok", "cached"):
                problems.append(f"{dataset} ep{index}: wrote {written}, {note}")
            if i % 100 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} episodes, {total} frames")

    print(f"\n{total} frames cached")
    if problems:
        print(f"{len(problems)} episodes disagreed with their declared length:")
        for p in problems:
            print(" ", p)
    else:
        print("every episode matched its declared frame count")


if __name__ == "__main__":
    main()
