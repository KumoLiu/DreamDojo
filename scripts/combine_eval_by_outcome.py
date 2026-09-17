#!/usr/bin/env python3
"""Tile per-episode eval comparisons into pages grouped by success/failure.

Each source clip is a 3-panel strip (real GT | teacher forced | closed loop) at
1920x480. This lays ten of them out as five rows of two, banners the successful
rollouts green on the left and the failed ones blue on the right, and writes one
file per page.

The two rollout validation sets are merged into a single group because the split
between them is an artefact of how they were collected, not something to compare
across; the label that matters is whether the underlying rollout succeeded, which
only the rollout sets carry.

Ordering is deterministic: episodes sort by (dataset, episode index) before being
paired, and rows are then grouped onto pages by clip length. Both keys depend
only on which episodes were scored, which is the same set for every model, so
cell (page, row, column) refers to the same episode across models. That is what
makes it possible to open two models' page 1 and compare a cell against the same
cell.

Grouping by length matters because a page runs as long as its longest clip and
short clips freeze on their last frame. Episode lengths here span 25 to 241
frames against a median of 61, so mixing them freely would leave most of a page
static for most of its duration.

    python3 scripts/combine_eval_by_outcome.py <eval_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapy
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Group name -> eval subdirectories to merge, and the val set each maps to for
# its success labels.
GROUPS = {
    "rollouts": {
        "rollouts_30k": "datasets/g1_hf_pick_trocar_rollouts_30k_val",
        "rollouts_10k": "datasets/g1_hf_pick_trocar_rollouts_10k_val",
    },
    "teleop_success": {
        "teleop_success": "datasets/g1_hf_pick_trocar_teleop_success_val",
    },
}

SUCCESS_COLOR = (40, 125, 40)
FAILURE_COLOR = (45, 45, 170)
UNLABELLED_COLOR = (70, 70, 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", type=Path, help="Eval run output root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <eval_dir>/by_outcome.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(GROUPS),
        choices=list(GROUPS),
    )
    parser.add_argument("--rows-per-page", type=int, default=5)
    # Only used for single-outcome sets, where the column count is free. Three
    # strips side by side is 4800px wide, about 2.1:1 overall, which still reads
    # as a widescreen frame; four would be too wide to take in at once.
    parser.add_argument("--max-cols", type=int, default=3)
    # 1920x480 native downscaled to 1600x400 keeps the 4:1 strip undistorted and
    # each of the three panels at 533x400, which is where the burnt-in panel
    # captions stay legible.
    parser.add_argument("--block-width", type=int, default=1600)
    parser.add_argument("--block-height", type=int, default=400)
    parser.add_argument("--header-height", type=int, default=48)
    parser.add_argument(
        "--gutter",
        type=int,
        default=12,
        help="Pixels of separator between cases. 0 disables it.",
    )
    parser.add_argument(
        "--gutter-level",
        type=int,
        default=255,
        help="Grey level of the separator; 255 is white, 0 black.",
    )
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def success_map(dataset_rel: str) -> dict[int, bool]:
    path = ROOT / dataset_rel / "meta" / "episodes.jsonl"
    if not path.exists():
        return {}
    labels = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "success" in record:
            labels[int(record["episode_index"])] = bool(record["success"])
    return labels


def collect(eval_dir: Path, members: dict[str, str]) -> list[dict]:
    clips = []
    for dataset, dataset_rel in members.items():
        labels = success_map(dataset_rel)
        for episode_dir in sorted((eval_dir / dataset).glob("episode_*")):
            videos = sorted(episode_dir.glob("*_comparison.mp4"))
            if not videos:
                continue
            metrics_path = episode_dir / "metrics.json"
            records = (
                json.loads(metrics_path.read_text())
                if metrics_path.exists()
                else []
            )
            by_episode = {int(r["episode_index"]): r for r in records}
            for video in videos:
                episode = int(video.name.split("_")[1])
                clips.append(
                    {
                        "path": video,
                        "dataset": dataset,
                        "episode": episode,
                        "success": labels.get(episode),
                        "record": by_episode.get(episode),
                        "frames": frame_count(video),
                    }
                )
    clips.sort(key=lambda c: (c["dataset"], c["episode"]))
    return clips


def frame_count(path: Path) -> int:
    """Read the length from the container rather than decoding the clip."""
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def banner_for(clip: dict) -> tuple[str, tuple[int, int, int]]:
    if clip["success"] is True:
        state, color = "SUCCESS", SUCCESS_COLOR
    elif clip["success"] is False:
        state, color = "FAILURE", FAILURE_COLOR
    else:
        state, color = "UNLABELLED", UNLABELLED_COLOR
    text = f"{state} | {clip['dataset']} ep {clip['episode']:03d}"
    record = clip["record"]
    if record:
        text += (
            f" | TF {record['teacher_forced']['psnr']:.1f} dB"
            f" | CL {record['closed_loop']['psnr']:.1f} dB"
        )
    return text, color


def pad_to(video: np.ndarray, frame_count: int) -> np.ndarray:
    """Freeze on the last frame so a short clip does not truncate the page."""
    if len(video) >= frame_count:
        return video[:frame_count]
    padding = np.repeat(video[-1:], frame_count - len(video), axis=0)
    return np.concatenate([video, padding], axis=0)


def render_cell(clip: dict, args: argparse.Namespace) -> np.ndarray:
    video = mediapy.read_video(str(clip["path"]))
    text, color = banner_for(clip)
    height = args.block_height + args.header_height
    frames = []
    for frame in video:
        resized = cv2.resize(
            np.asarray(frame),
            (args.block_width, args.block_height),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.zeros((height, args.block_width, 3), dtype=np.uint8)
        canvas[: args.header_height] = color
        canvas[args.header_height :] = resized
        cv2.putText(
            canvas,
            text,
            (18, args.header_height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            args.font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(canvas)
    return np.stack(frames)


def pair_rows(clips: list[dict]) -> list[tuple[dict, dict | None]]:
    """Put successes in the left column and failures in the right.

    Once one outcome runs out the remaining clips fill both columns rather than
    leaving half the page blank; every cell still carries its own banner, so the
    outcome is never read from position alone.
    """
    successes = [c for c in clips if c["success"] is True]
    others = [c for c in clips if c["success"] is not True]
    rows: list[tuple[dict, dict | None]] = []
    while successes or others:
        left = successes.pop(0) if successes else others.pop(0)
        if others:
            right = others.pop(0)
        elif successes:
            right = successes.pop(0)
        else:
            right = None
        rows.append((left, right))
    # Two competing goals: keep the outcome readable from column position, and
    # keep each page's clips similar in length so short ones do not sit frozen
    # while a long one plays out. Sorting by length only within the rows that
    # pair one outcome against the other satisfies both, because the paired rows
    # stay ahead of the leftovers and so fill whole pages before any mixed page
    # appears.
    rows.sort(key=lambda row: (not is_paired(row), row_frames(row)))
    return rows


def is_paired(row: tuple[dict, dict | None]) -> bool:
    left, right = row
    return right is not None and left["success"] is not right["success"]


def row_frames(row: tuple[dict, dict | None]) -> int:
    left, right = row
    return max(left["frames"], right["frames"] if right else 0)


def plan_pages(
    clips: list[dict],
    args: argparse.Namespace,
) -> list[tuple[list[dict | None], int]]:
    """Split clips into pages of (cells, column count).

    A validation set with both outcomes keeps the two-column layout, because
    there the column carries meaning. A set with a single outcome does not, so
    the column count is free to vary and is chosen to fill every cell: 25
    teleop episodes become 15 in three columns then 10 in two, rather than
    leaving a blank cell in a fixed two-column grid.
    """
    successes = [c for c in clips if c["success"] is True]
    failures = [c for c in clips if c["success"] is not True]
    pages: list[tuple[list[dict | None], int]] = []

    if successes and failures:
        rows = pair_rows(clips)
        for start in range(0, len(rows), args.rows_per_page):
            cells: list[dict | None] = []
            for left, right in rows[start : start + args.rows_per_page]:
                cells += [left, right]
            pages.append((cells, 2))
        return pages

    remaining = sorted(clips, key=lambda c: c["frames"])
    while remaining:
        cols = fewest_blanks(len(remaining), args)
        take = min(len(remaining), cols * args.rows_per_page)
        cells = list(remaining[:take])
        cells += [None] * (-len(cells) % cols)
        pages.append((cells, cols))
        remaining = remaining[take:]
    return pages


def fewest_blanks(count: int, args: argparse.Namespace) -> int:
    """Pick the column count that wastes the fewest cells on this page."""

    def blanks(cols: int) -> tuple[int, int]:
        take = min(count, cols * args.rows_per_page)
        return -take % cols, -cols

    return min(range(1, args.max_cols + 1), key=blanks)


def join(parts: list[np.ndarray], gutter: np.ndarray | None, axis: int):
    if gutter is None:
        return np.concatenate(parts, axis=axis)
    spaced: list[np.ndarray] = []
    for index, part in enumerate(parts):
        if index:
            spaced.append(gutter)
        spaced.append(part)
    return np.concatenate(spaced, axis=axis)


def build_page(
    cells: list[dict | None],
    cols: int,
    args: argparse.Namespace,
) -> np.ndarray:
    rendered = [render_cell(c, args) if c is not None else None for c in cells]
    frame_count = max(len(r) for r in rendered if r is not None)
    blank = np.zeros_like(
        next(pad_to(r, frame_count) for r in rendered if r is not None)
    )
    rendered = [
        pad_to(r, frame_count) if r is not None else blank for r in rendered
    ]

    # The three panels inside a case are butted together by the validator, so
    # without a gutter between cases the closed-loop panel of one case abuts the
    # ground-truth panel of the next and the case boundary reads the same as a
    # panel boundary.
    height, width = rendered[0].shape[1:3]
    column_gutter = row_gutter = None
    if args.gutter > 0:
        column_gutter = np.full(
            (frame_count, height, args.gutter, 3), args.gutter_level, np.uint8
        )
    rows = [
        join(rendered[start : start + cols], column_gutter, axis=2)
        for start in range(0, len(rendered), cols)
    ]
    if args.gutter > 0:
        row_gutter = np.full(
            (frame_count, args.gutter, rows[0].shape[2], 3),
            args.gutter_level,
            np.uint8,
        )
    return join(rows, row_gutter, axis=1)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.eval_dir / "by_outcome"
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for group in args.groups:
        clips = collect(args.eval_dir, GROUPS[group])
        if not clips:
            print(f"{group}: no comparison videos found, skipping")
            continue
        pages = plan_pages(clips, args)
        n_success = sum(1 for c in clips if c["success"] is True)
        print(
            f"{group}: {len(clips)} episodes "
            f"({n_success} success, {len(clips) - n_success} failure/unlabelled)"
            f" -> {len(pages)} page(s)"
        )
        for page_index, (cells, cols) in enumerate(pages, start=1):
            combined = build_page(cells, cols, args)
            suffix = f"_page{page_index:02d}" if len(pages) > 1 else ""
            output = output_dir / f"{group}{suffix}.mp4"
            mediapy.write_video(str(output), combined, fps=args.fps)
            size_mb = output.stat().st_size / 1e6
            filled = [c for c in cells if c is not None]
            lengths = [c["frames"] for c in filled]
            blanks = len(cells) - len(filled)
            print(
                f"  {output.name}: {len(filled)} cases in "
                f"{len(cells) // cols}x{cols}, {combined.shape[0]} frames, "
                f"{combined.shape[2]}x{combined.shape[1]}, {size_mb:.1f} MB, "
                f"clip lengths {min(lengths)}-{max(lengths)}"
                + (f", {blanks} blank cell(s)" if blanks else "")
            )
            written.append(output)

    if not written:
        raise SystemExit(f"No comparison videos found under {args.eval_dir}")
    print(f"\n{len(written)} video(s) in {output_dir}")


if __name__ == "__main__":
    main()
