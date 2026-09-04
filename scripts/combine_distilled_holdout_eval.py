#!/usr/bin/env python3
"""Combine teacher, Warmup, and Self-Forcing held-out evaluations."""

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
    parser.add_argument("--cell-width", type=int, default=256)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--teacher-dir", default="teacher_iter2500")
    parser.add_argument("--warmup-dir", default="warmup_iter20000")
    parser.add_argument("--self-forcing-dir", default="self_forcing_iter3000")
    parser.add_argument("--teacher-label", default="TEACHER 2500")
    parser.add_argument("--warmup-label", default="WARMUP 20000")
    parser.add_argument("--self-forcing-label", default="SELF-FORCING 3000")
    return parser.parse_args()


def read_video(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(mediapy.read_video(str(path)))


def resize_video(video: np.ndarray, width: int) -> np.ndarray:
    height = round(video.shape[1] * width / video.shape[2])
    return np.stack(
        [
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            for frame in video
        ]
    )


def pad_video(video: np.ndarray, frame_count: int) -> np.ndarray:
    if len(video) >= frame_count:
        return video[:frame_count]
    return np.concatenate(
        [video, np.repeat(video[-1:], frame_count - len(video), axis=0)],
        axis=0,
    )


def split_labeled_columns(video: np.ndarray, columns: int) -> list[np.ndarray]:
    if video.shape[2] % columns:
        raise ValueError(
            f"Video width {video.shape[2]} is not divisible by {columns}"
        )
    return list(np.split(video, columns, axis=2))


def add_case_label(video: np.ndarray, label: str) -> np.ndarray:
    output = video.copy()
    for frame in output:
        cv2.putText(
            frame,
            label,
            (12, frame.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def set_model_label(video: np.ndarray, label: str) -> np.ndarray:
    output = video.copy()
    for frame in output:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(
            frame,
            label,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def load_cases(dataset_path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (dataset_path / "meta" / "episodes.jsonl").read_text().splitlines()
        if line
    ]
    if len(rows) != 10:
        raise ValueError(f"Expected 10 held-out episodes, found {len(rows)}")
    return rows


def load_case(
    input_dir: Path,
    position: int,
    case: dict,
    cell_width: int,
    teacher_dir: str,
    warmup_dir: str,
    self_forcing_dir: str,
    teacher_label: str,
    warmup_label: str,
    self_forcing_label: str,
) -> np.ndarray:
    teacher = read_video(
        input_dir / teacher_dir / f"episode_{position:03d}_seed_0_comparison.mp4"
    )
    warmup = read_video(
        input_dir / warmup_dir / f"episode_{position:03d}_comparison.mp4"
    )
    self_forcing = read_video(
        input_dir
        / self_forcing_dir
        / f"episode_{position:03d}_comparison.mp4"
    )

    teacher_columns = split_labeled_columns(teacher, 3)
    warmup_columns = split_labeled_columns(warmup, 2)
    self_forcing_columns = split_labeled_columns(self_forcing, 2)
    columns = [
        teacher_columns[0],
        teacher_columns[2],
        warmup_columns[1],
        self_forcing_columns[1],
    ]
    frame_count = min(map(len, columns))
    columns = [resize_video(column[:frame_count], cell_width) for column in columns]
    columns = [
        set_model_label(column, label)
        for column, label in zip(
            columns,
            ("REAL GT", teacher_label, warmup_label, self_forcing_label),
            strict=True,
        )
    ]
    outcome = "SUCCESS" if case["success"] else "FAILURE"
    label = (
        f"{outcome} | CASE {position:02d} | "
        f"ORIG {int(case['episode_index']):03d}"
    )
    columns[0] = add_case_label(columns[0], label)
    return np.concatenate(columns, axis=2)


def aggregate_metrics(input_dir: Path, cases: list[dict], model_dirs: dict[str, str]) -> dict:
    output: dict[str, dict] = {}
    for label, dirname in model_dirs.items():
        records = json.loads((input_dir / dirname / "metrics.json").read_text())
        by_episode = {int(record["episode_index"]): record for record in records}
        if set(by_episode) != set(range(10)):
            raise ValueError(f"Incomplete metrics for {label}: {sorted(by_episode)}")
        model_summary: dict[str, dict[str, float]] = {}
        for split, positions in {
            "all": list(range(10)),
            "success": [i for i, case in enumerate(cases) if case["success"]],
            "failure": [i for i, case in enumerate(cases) if not case["success"]],
        }.items():
            model_summary[split] = {
                metric: float(
                    np.mean(
                        [
                            by_episode[position]["closed_loop"][metric]
                            for position in positions
                        ]
                    )
                )
                for metric in ("mae", "mse", "psnr")
            }
        output[label] = model_summary
    return output


def main() -> None:
    args = parse_args()
    cases = load_cases(args.dataset_path)
    success = [i for i, case in enumerate(cases) if case["success"]]
    failure = [i for i, case in enumerate(cases) if not case["success"]]
    if len(success) != 5 or len(failure) != 5:
        raise ValueError("Expected five success and five failure cases")

    rows = [
        np.concatenate(
            [
                load_case(
                    args.input_dir,
                    success_pos,
                    cases[success_pos],
                    args.cell_width,
                    args.teacher_dir,
                    args.warmup_dir,
                    args.self_forcing_dir,
                    args.teacher_label,
                    args.warmup_label,
                    args.self_forcing_label,
                ),
                load_case(
                    args.input_dir,
                    failure_pos,
                    cases[failure_pos],
                    args.cell_width,
                    args.teacher_dir,
                    args.warmup_dir,
                    args.self_forcing_dir,
                    args.teacher_label,
                    args.warmup_label,
                    args.self_forcing_label,
                ),
            ],
            axis=2,
        )
        for success_pos, failure_pos in zip(success, failure, strict=True)
    ]
    frame_count = max(map(len, rows))
    combined = np.concatenate(
        [pad_video(row, frame_count) for row in rows],
        axis=1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(args.output), combined, fps=args.fps)

    summary = aggregate_metrics(
        args.input_dir,
        cases,
        {
            "teacher": args.teacher_dir,
            "warmup_selected": args.warmup_dir,
            "self_forcing_97f": args.self_forcing_dir,
        },
    )
    summary_path = args.output.with_suffix(".metrics.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output} with shape {combined.shape}")


if __name__ == "__main__":
    main()
