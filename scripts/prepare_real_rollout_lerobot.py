#!/usr/bin/env python3
"""Build leakage-free LeRobot views for DreamDojo real-data fine-tuning."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.adapt_g1_dex3_to_dreamdojo import pad_28_to_43


TASK = (
    "Pick all instruments from the left treatment disk and place them onto "
    "the right Mayo stand in strict order."
)
FPS = 30
VAL_EPISODE_IDS = frozenset({1, 8, 9, 10, 13, 15, 27})
MIN_WINDOW_FRAMES = 27


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-root",
        type=Path,
        default=Path("/localhome/local-yunl/real_rollouts"),
    )
    parser.add_argument(
        "--teleop-root",
        type=Path,
        default=Path(
            "/localhome/local-yunl/DreamDojo/datasets/g1_pick_trocar_200_headcam"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/localhome/local-yunl/DreamDojo/datasets"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json_lines(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def prepare_destination(path: Path, force: bool) -> None:
    if path.exists() or path.is_symlink():
        if not force:
            raise FileExistsError(f"{path} exists; pass --force to replace it")
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    (path / "meta").mkdir(parents=True)
    (path / "data/chunk-000").mkdir(parents=True)
    (path / "videos/chunk-000/observation.images.cam_head").mkdir(parents=True)


def numeric_stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def scalar_stats(values: np.ndarray) -> dict:
    return {
        "min": [values.min().item()],
        "max": [values.max().item()],
        "mean": [values.mean().item()],
        "std": [values.std().item()],
        "count": [len(values)],
    }


def episode_stats(
    episode_index: int,
    global_start: int,
    states: np.ndarray,
    actions: np.ndarray,
) -> dict:
    length = len(states)
    frame_index = np.arange(length, dtype=np.int64)
    timestamps = frame_index.astype(np.float32) / FPS
    global_indices = np.arange(global_start, global_start + length, dtype=np.int64)
    return {
        "episode_index": episode_index,
        "stats": {
            "observation.state": numeric_stats(states),
            "action": numeric_stats(actions),
            "timestamp": scalar_stats(timestamps),
            "frame_index": scalar_stats(frame_index),
            "episode_index": scalar_stats(
                np.full(length, episode_index, dtype=np.int64)
            ),
            "index": scalar_stats(global_indices),
            "task_index": scalar_stats(np.zeros(length, dtype=np.int64)),
        },
    }


def write_parquet(
    path: Path,
    episode_index: int,
    global_start: int,
    states: np.ndarray,
    actions: np.ndarray,
) -> None:
    length = len(states)
    frame_index = np.arange(length, dtype=np.int64)
    table = pa.Table.from_arrays(
        [
            pa.FixedSizeListArray.from_arrays(
                pa.array(states.reshape(-1), type=pa.float32()), 43
            ),
            pa.FixedSizeListArray.from_arrays(
                pa.array(actions.reshape(-1), type=pa.float32()), 43
            ),
            pa.array(frame_index.astype(np.float32) / FPS, type=pa.float32()),
            pa.array(frame_index, type=pa.int64()),
            pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()),
            pa.array(
                np.arange(global_start, global_start + length, dtype=np.int64),
                type=pa.int64(),
            ),
            pa.array(np.zeros(length, dtype=np.int64), type=pa.int64()),
        ],
        names=[
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ],
    )
    pq.write_table(table, path)


def transcode_video(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            "scale=640:480",
            "-r",
            str(FPS),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(destination),
        ],
        check=True,
    )


def video_frame_count(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def build_real_info(template: dict, num_episodes: int, num_frames: int) -> dict:
    info = json.loads(json.dumps(template))
    info["robot_type"] = "Unitree_G1_real_rollout_headcam"
    info["total_episodes"] = num_episodes
    info["total_frames"] = num_frames
    info["total_videos"] = num_episodes
    info["total_chunks"] = 1
    info["splits"] = {"full": f"0:{num_episodes}"}
    info["features"] = {
        key: value
        for key, value in info["features"].items()
        if key != "observation.images.cam_room"
    }
    return info


def write_real_dataset(
    output: Path,
    episodes: list[Path],
    template_root: Path,
    *,
    force: bool,
    contact_only: bool = False,
) -> None:
    prepare_destination(output, force)
    template_info = json.loads((template_root / "meta/info.json").read_text())
    modality = json.loads((template_root / "meta/modality.json").read_text())
    modality["video"]["ego_view"]["original_key"] = "observation.images.cam_head"
    (output / "meta/modality.json").write_text(json.dumps(modality, indent=4) + "\n")
    write_json_lines(output / "meta/tasks.jsonl", [{"task_index": 0, "task": TASK}])

    episode_records = []
    stats_records = []
    all_states = []
    all_actions = []
    global_index = 0
    for output_index, source in enumerate(episodes):
        metadata = json.loads((source / "meta.json").read_text())
        states = pad_28_to_43(np.load(source / "states.npy"))
        actions = pad_28_to_43(np.load(source / "actions.npy"))
        source_count, source_fps, _, _ = video_frame_count(source / "color_0.mp4")
        length = min(
            metadata["num_frames"],
            len(states),
            len(actions),
            source_count,
        )
        if length < MIN_WINDOW_FRAMES:
            raise ValueError(f"{source} has only {length} usable frames")
        states = states[:length]
        actions = actions[:length]
        if abs(source_fps - FPS) > 0.01:
            raise ValueError(f"{source} has unexpected fps {source_fps}")

        parquet_path = output / f"data/chunk-000/episode_{output_index:06d}.parquet"
        video_path = output / (
            "videos/chunk-000/observation.images.cam_head/"
            f"episode_{output_index:06d}.mp4"
        )
        write_parquet(
            parquet_path,
            output_index,
            global_index,
            states,
            actions,
        )
        transcode_video(source / "color_0.mp4", video_path)
        transcoded_count, transcoded_fps, width, height = video_frame_count(video_path)
        if (
            transcoded_count != length
            or abs(transcoded_fps - FPS) > 0.01
            or (width, height) != (640, 480)
        ):
            raise ValueError(
                f"Bad transcoded video {video_path}: "
                f"{transcoded_count} frames, {transcoded_fps} fps, "
                f"{width}x{height}; expected {length}, 30, 640x480"
            )

        record = {
            "episode_index": output_index,
            "tasks": [metadata.get("task", TASK)],
            "length": length,
            "source_session": source.parent.name,
            "source_episode_id": metadata["episode_id"],
            "success": bool(metadata["success"]),
        }
        if contact_only:
            record["sample_start"] = length // 3
            record["sample_end"] = max(length // 3 + 1, 2 * length // 3)
        episode_records.append(record)
        stats_records.append(episode_stats(output_index, global_index, states, actions))
        all_states.append(states)
        all_actions.append(actions)
        global_index += length

    write_json_lines(output / "meta/episodes.jsonl", episode_records)
    write_json_lines(output / "meta/episodes_stats.jsonl", stats_records)
    info = build_real_info(template_info, len(episodes), global_index)
    (output / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")
    dataset_stats = {
        "observation.state": numeric_stats(np.concatenate(all_states)),
        "action": numeric_stats(np.concatenate(all_actions)),
    }
    (output / "meta/stats.json").write_text(json.dumps(dataset_stats, indent=4) + "\n")


def build_teleop_view(
    source: Path,
    output: Path,
    episode_ids: set[int],
    *,
    force: bool,
) -> None:
    if output.exists() or output.is_symlink():
        if not force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
        if output.is_symlink():
            output.unlink()
        else:
            shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "meta").mkdir()
    (output / "data").symlink_to((source / "data").resolve())
    (output / "videos").symlink_to((source / "videos").resolve())

    episodes = [
        record
        for record in json_lines(source / "meta/episodes.jsonl")
        if record["episode_index"] in episode_ids
    ]
    episode_stats_by_id = {
        record["episode_index"]: record
        for record in json_lines(source / "meta/episodes_stats.jsonl")
    }
    write_json_lines(output / "meta/episodes.jsonl", episodes)
    write_json_lines(
        output / "meta/episodes_stats.jsonl",
        [episode_stats_by_id[record["episode_index"]] for record in episodes],
    )
    shutil.copy2(source / "meta/tasks.jsonl", output / "meta/tasks.jsonl")
    shutil.copy2(source / "meta/modality.json", output / "meta/modality.json")
    shutil.copy2(source / "meta/stats.json", output / "meta/stats.json")
    info = json.loads((source / "meta/info.json").read_text())
    info["total_episodes"] = len(episodes)
    info["total_frames"] = sum(record["length"] for record in episodes)
    info["total_videos"] = len(episodes) * 2
    info["splits"] = {"full": f"0:{len(episodes)}"}
    (output / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")


def discover_real_episodes(root: Path) -> list[Path]:
    episodes = sorted(root.glob("*/episode_*"))
    required = {"meta.json", "states.npy", "actions.npy", "color_0.mp4"}
    missing = [
        str(path)
        for path in episodes
        if not required.issubset(item.name for item in path.iterdir())
    ]
    if missing:
        raise ValueError(f"Incomplete episodes: {missing}")
    return episodes


def validate_split(train: list[Path], val: list[Path]) -> None:
    def ids(paths: list[Path]) -> set[int]:
        return {
            json.loads((path / "meta.json").read_text())["episode_id"] for path in paths
        }

    overlap = ids(train) & ids(val)
    if overlap:
        raise ValueError(f"Train/validation episode ID overlap: {overlap}")


def main() -> None:
    args = parse_args()
    real_episodes = discover_real_episodes(args.real_root)
    train = []
    val = []
    for episode in real_episodes:
        metadata = json.loads((episode / "meta.json").read_text())
        destination = val if metadata["episode_id"] in VAL_EPISODE_IDS else train
        destination.append(episode)
    validate_split(train, val)
    train_failures = [
        path
        for path in train
        if not json.loads((path / "meta.json").read_text())["success"]
    ]

    teleop_train = args.output_root / "g1_pick_trocar_200_headcam_train"
    teleop_val = args.output_root / "g1_pick_trocar_200_headcam_val"
    real_train = args.output_root / "g1_pick_trocar_real_train_headcam"
    real_val = args.output_root / "g1_pick_trocar_real_val_headcam"
    real_contact = args.output_root / "g1_pick_trocar_real_train_fail_contact_headcam"
    build_teleop_view(args.teleop_root, teleop_train, set(range(190)), force=args.force)
    build_teleop_view(
        args.teleop_root,
        teleop_val,
        set(range(190, 200)),
        force=args.force,
    )
    write_real_dataset(real_train, train, args.teleop_root, force=args.force)
    write_real_dataset(real_val, val, args.teleop_root, force=args.force)
    write_real_dataset(
        real_contact,
        train_failures,
        args.teleop_root,
        force=args.force,
        contact_only=True,
    )

    manifest = {
        "validation_source_episode_ids": sorted(VAL_EPISODE_IDS),
        "teleop_train_episode_ids": list(range(190)),
        "teleop_val_episode_ids": list(range(190, 200)),
        "real_train": [str(path) for path in train],
        "real_val": [str(path) for path in val],
        "real_train_fail_contact": [str(path) for path in train_failures],
    }
    manifest_path = args.output_root / "g1_pick_trocar_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Prepared teleop train/val: 190/10; rollout train/val: "
        f"{len(train)}/{len(val)}; contact failures: {len(train_failures)}"
    )
    print(f"Split manifest: {manifest_path}")


if __name__ == "__main__":
    main()
