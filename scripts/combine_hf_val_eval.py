#!/usr/bin/env python3
"""Tile per-episode validation comparisons into a few overview videos.

Each source clip is already a 3-panel strip (real GT | teacher forced | closed
loop). This lays those strips out in a grid so a whole validation set can be
skimmed at once, splitting into multiple pages so no single file gets so large
or so densely packed that it stops being readable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapy
import numpy as np

DATASET_COLORS = {
    "teleop_success": (40, 110, 40),
    "rollouts_30k": (120, 70, 30),
    "rollouts_10k": (110, 40, 110),
}
DEFAULT_COLOR = (70, 70, 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", type=Path, help="Eval run output root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <eval_dir>/overview.",
    )
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=1200)
    parser.add_argument("--cell-height", type=int, default=300)
    parser.add_argument("--header-height", type=int, default=30)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["teleop_success", "rollouts_30k", "rollouts_10k"],
    )
    return parser.parse_args()


def collect_clips(eval_dir: Path, dataset: str) -> list[dict]:
    """Pair each episode's comparison clip with its metrics, if present."""
    clips = []
    for episode_dir in sorted((eval_dir / dataset).glob("episode_*")):
        videos = sorted(episode_dir.glob("*_comparison.mp4"))
        if not videos:
            continue
        metrics_path = episode_dir / "metrics.json"
        records = (
            json.loads(metrics_path.read_text()) if metrics_path.exists() else []
        )
        by_episode = {int(r["episode_index"]): r for r in records}
        for video in videos:
            episode_index = int(video.name.split("_")[1])
            clips.append(
                {
                    "path": video,
                    "dataset": dataset,
                    "episode": episode_index,
                    "record": by_episode.get(episode_index),
                }
            )
    return clips


def label_for(clip: dict) -> str:
    text = f"{clip['dataset']} | ep {clip['episode']:03d}"
    record = clip["record"]
    if record:
        text += (
            f" | TF PSNR {record['teacher_forced']['psnr']:.1f}"
            f" | CL PSNR {record['closed_loop']['psnr']:.1f}"
        )
    return text


def pad_to(video: np.ndarray, frame_count: int) -> np.ndarray:
    """Freeze on the last frame so short clips do not truncate a whole page."""
    if len(video) >= frame_count:
        return video[:frame_count]
    padding = np.repeat(video[-1:], frame_count - len(video), axis=0)
    return np.concatenate([video, padding], axis=0)


def render_cell(clip: dict, args: argparse.Namespace) -> np.ndarray:
    video = mediapy.read_video(str(clip["path"]))
    color = DATASET_COLORS.get(clip["dataset"], DEFAULT_COLOR)
    text = label_for(clip)
    frames = []
    for frame in video:
        resized = cv2.resize(
            np.asarray(frame),
            (args.cell_width, args.cell_height),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.zeros(
            (args.cell_height + args.header_height, args.cell_width, 3),
            dtype=np.uint8,
        )
        canvas[: args.header_height] = color
        canvas[args.header_height :] = resized
        cv2.putText(
            canvas,
            text,
            (12, args.header_height - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        frames.append(canvas)
    return np.stack(frames)


def build_page(clips: list[dict], args: argparse.Namespace) -> np.ndarray:
    cells = [render_cell(clip, args) for clip in clips]
    frame_count = max(len(cell) for cell in cells)
    cells = [pad_to(cell, frame_count) for cell in cells]
    blank = np.zeros_like(cells[0])

    rows = []
    for start in range(0, len(cells), args.cols):
        row = cells[start : start + args.cols]
        row += [blank] * (args.cols - len(row))
        rows.append(np.concatenate(row, axis=2))
    return np.concatenate(rows, axis=1)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.eval_dir / "overview"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_page = args.cols * args.rows

    written = []
    for dataset in args.datasets:
        clips = collect_clips(args.eval_dir, dataset)
        if not clips:
            print(f"{dataset}: no comparison videos found, skipping")
            continue
        pages = [clips[i : i + per_page] for i in range(0, len(clips), per_page)]
        for page_index, page_clips in enumerate(pages, start=1):
            combined = build_page(page_clips, args)
            suffix = f"_page{page_index:02d}" if len(pages) > 1 else ""
            output = output_dir / f"{dataset}{suffix}.mp4"
            mediapy.write_video(str(output), combined, fps=args.fps)
            size_mb = output.stat().st_size / 1e6
            print(
                f"Wrote {output.name}: {len(page_clips)} episodes, "
                f"{combined.shape[0]} frames, "
                f"{combined.shape[2]}x{combined.shape[1]}, {size_mb:.1f} MB"
            )
            written.append(output)

    if not written:
        raise SystemExit(f"No comparison videos found under {args.eval_dir}")
    print(f"\n{len(written)} overview video(s) in {output_dir}")


if __name__ == "__main__":
    main()
