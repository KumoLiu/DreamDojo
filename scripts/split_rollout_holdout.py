#!/usr/bin/env python3
"""Create leakage-free train/eval views of the 260-episode rollout dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_EVAL_SUCCESS = (21, 53, 127, 154, 198)
DEFAULT_EVAL_FAIL = (0, 19, 27, 55, 236)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def numeric_stats(values: np.ndarray) -> dict:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def recompute_stats(source: Path, info: dict, episode_ids: set[int]) -> dict:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    chunk_size = info["chunks_size"]
    pattern = info["data_path"]
    for episode_id in sorted(episode_ids):
        parquet_path = source / pattern.format(
            episode_chunk=episode_id // chunk_size,
            episode_index=episode_id,
        )
        table = pq.read_table(parquet_path, columns=["observation.state", "action"])
        states.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float64))
        actions.append(np.asarray(table["action"].to_pylist(), dtype=np.float64))
    return {
        "observation.state": numeric_stats(np.concatenate(states)),
        "action": numeric_stats(np.concatenate(actions)),
    }


def build_view(
    source: Path,
    output: Path,
    episode_ids: set[int],
    episodes_by_id: dict[int, dict],
    episode_stats_by_id: dict[int, dict],
    source_info: dict,
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

    episodes = [episodes_by_id[index] for index in sorted(episode_ids)]
    write_jsonl(output / "meta/episodes.jsonl", episodes)
    write_jsonl(
        output / "meta/episodes_stats.jsonl",
        [episode_stats_by_id[index] for index in sorted(episode_ids)],
    )
    for filename in ("tasks.jsonl", "modality.json"):
        shutil.copy2(source / "meta" / filename, output / "meta" / filename)

    info = json.loads(json.dumps(source_info))
    videos_per_episode = source_info["total_videos"] / source_info["total_episodes"]
    info["total_episodes"] = len(episodes)
    info["total_frames"] = sum(episode["length"] for episode in episodes)
    info["total_videos"] = round(len(episodes) * videos_per_episode)
    info["splits"] = {"full": f"0:{len(episodes)}"}
    (output / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")

    stats = recompute_stats(source, source_info, episode_ids)
    (output / "meta/stats.json").write_text(json.dumps(stats, indent=4) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/g1_pick_trocar_rollout_all_260_headcam"),
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("datasets/g1_pick_trocar_rollout_all_260_headcam_train"),
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=Path("datasets/g1_pick_trocar_rollout_all_260_headcam_eval_5s5f"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/g1_pick_trocar_rollout_all_260_headcam_split_manifest.json"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    episodes_by_id = {record["episode_index"]: record for record in episodes}
    episode_stats_by_id = {
        record["episode_index"]: record
        for record in read_jsonl(source / "meta/episodes_stats.jsonl")
    }
    all_ids = set(episodes_by_id)
    eval_success = set(DEFAULT_EVAL_SUCCESS)
    eval_fail = set(DEFAULT_EVAL_FAIL)
    eval_ids = eval_success | eval_fail
    train_ids = all_ids - eval_ids

    if len(all_ids) != 260 or len(train_ids) != 250:
        raise ValueError(f"Expected 260 source and 250 train episodes, got {len(all_ids)} and {len(train_ids)}")
    if any(not episodes_by_id[index].get("success") for index in eval_success):
        raise ValueError("An eval-success episode is not labeled successful")
    if any(episodes_by_id[index].get("success") for index in eval_fail):
        raise ValueError("An eval-fail episode is not labeled failed")

    source_info = json.loads((source / "meta/info.json").read_text())
    build_view(
        source,
        args.train_output,
        train_ids,
        episodes_by_id,
        episode_stats_by_id,
        source_info,
        force=args.force,
    )
    build_view(
        source,
        args.eval_output,
        eval_ids,
        episodes_by_id,
        episode_stats_by_id,
        source_info,
        force=args.force,
    )

    manifest = {
        "source": str(source),
        "eval_success": sorted(eval_success),
        "eval_fail": sorted(eval_fail),
        "eval_episode_indices": sorted(eval_ids),
        "train_episode_indices": sorted(train_ids),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created {args.train_output} with {len(train_ids)} episodes")
    print(f"Created {args.eval_output} with {len(eval_ids)} episodes")
    print(f"Wrote split manifest to {args.manifest}")


if __name__ == "__main__":
    main()
