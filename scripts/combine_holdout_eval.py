#!/usr/bin/env python3
"""Combine ten holdout cases into a 10-row by 3-column comparison video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapy
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cell-width", type=int, default=320)
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def resize_video(video: np.ndarray, width: int) -> np.ndarray:
    height = round(video.shape[1] * width / video.shape[2])
    return np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in video]
    )


def main() -> None:
    args = parse_args()
    rows = []
    gt_paths = sorted(args.input_dir.rglob("case_*_gt.mp4"))
    if len(gt_paths) != 10:
        raise ValueError(f"Expected 10 GT videos, found {len(gt_paths)}")

    for gt_path in gt_paths:
        prefix = gt_path.name.removesuffix("_gt.mp4")
        teacher_path = gt_path.parent / f"{prefix}_teacher_forced.mp4"
        closed_path = gt_path.parent / f"{prefix}_closed_loop.mp4"
        videos = [
            resize_video(mediapy.read_video(str(path)), args.cell_width)
            for path in (gt_path, teacher_path, closed_path)
        ]
        frame_count = min(len(video) for video in videos)
        rows.append(np.concatenate([video[:frame_count] for video in videos], axis=2))

    frame_count = min(len(row) for row in rows)
    combined = np.concatenate([row[:frame_count] for row in rows], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(args.output), combined, fps=args.fps)
    print(f"Wrote {args.output} with shape {combined.shape}")


if __name__ == "__main__":
    main()
