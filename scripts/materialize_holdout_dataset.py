#!/usr/bin/env python3
"""Materialize and reindex a LeRobot holdout view into a standalone dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def resolve_episode_path(
    root: Path,
    pattern: str,
    episode_id: int,
    chunk_size: int,
    *,
    video_key: str | None = None,
) -> Path:
    values: dict[str, int | str] = {
        "episode_chunk": episode_id // chunk_size,
        "episode_index": episode_id,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return root / pattern.format(**values)


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"Required parquet column is missing: {name}")
    return table.set_column(index, name, values)


def rewrite_episode_stats(
    source_record: dict,
    new_episode_id: int,
    global_start: int,
    length: int,
) -> dict:
    record = json.loads(json.dumps(source_record))
    record["source_episode_index"] = int(record["episode_index"])
    record["episode_index"] = new_episode_id
    stats = record.get("stats", {})

    if "episode_index" in stats:
        stats["episode_index"].update(
            min=[float(new_episode_id)],
            max=[float(new_episode_id)],
            mean=[float(new_episode_id)],
            std=[0.0],
            count=[length],
        )
    if "index" in stats:
        stats["index"].update(
            min=[float(global_start)],
            max=[float(global_start + length - 1)],
            mean=[float(global_start + (length - 1) / 2)],
            count=[length],
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "datasets/g1_pick_trocar_rollout_all_260_headcam_eval_5s5f"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/g1_pick_trocar_rollout_holdout_10_materialized"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        if not args.force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
        if output.is_symlink():
            output.unlink()
        else:
            shutil.rmtree(output)

    info = json.loads((source / "meta/info.json").read_text())
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    episode_stats = read_jsonl(source / "meta/episodes_stats.jsonl")
    if len(episodes) != 10 or len(episode_stats) != 10:
        raise ValueError(
            "Expected exactly 10 episode and episode-stat records; "
            f"found {len(episodes)} and {len(episode_stats)}"
        )

    stats_by_id = {int(row["episode_index"]): row for row in episode_stats}
    chunk_size = int(info["chunks_size"])
    data_pattern = str(info["data_path"])
    video_pattern = str(info["video_path"])
    video_keys = [
        key for key, feature in info["features"].items()
        if feature.get("dtype") == "video"
    ]
    if not video_keys:
        raise ValueError("No video features found in meta/info.json")

    (output / "meta").mkdir(parents=True)
    (output / "data/chunk-000").mkdir(parents=True)
    for video_key in video_keys:
        (output / f"videos/chunk-000/{video_key}").mkdir(parents=True)

    rewritten_episodes: list[dict] = []
    rewritten_episode_stats: list[dict] = []
    mapping: list[dict] = []
    global_start = 0

    for new_id, source_episode in enumerate(episodes):
        source_id = int(source_episode["episode_index"])
        length = int(source_episode["length"])
        source_parquet = resolve_episode_path(
            source, data_pattern, source_id, chunk_size
        )
        table = pq.read_table(source_parquet)
        if table.num_rows != length:
            raise ValueError(
                f"Episode {source_id}: metadata length={length}, "
                f"parquet rows={table.num_rows}"
            )

        table = replace_column(
            table,
            "episode_index",
            pa.array([new_id] * length, type=table["episode_index"].type),
        )
        table = replace_column(
            table,
            "index",
            pa.array(
                range(global_start, global_start + length),
                type=table["index"].type,
            ),
        )
        destination_parquet = resolve_episode_path(
            output, data_pattern, new_id, chunk_size
        )
        pq.write_table(table, destination_parquet)

        for video_key in video_keys:
            source_video = resolve_episode_path(
                source,
                video_pattern,
                source_id,
                chunk_size,
                video_key=video_key,
            )
            destination_video = resolve_episode_path(
                output,
                video_pattern,
                new_id,
                chunk_size,
                video_key=video_key,
            )
            if not source_video.is_file():
                raise FileNotFoundError(source_video)
            shutil.copy2(source_video, destination_video)

        episode = json.loads(json.dumps(source_episode))
        episode["source_episode_index"] = source_id
        episode["episode_index"] = new_id
        rewritten_episodes.append(episode)
        rewritten_episode_stats.append(
            rewrite_episode_stats(
                stats_by_id[source_id], new_id, global_start, length
            )
        )
        mapping.append(
            {
                "episode_index": new_id,
                "source_episode_index": source_id,
                "success": episode.get("success"),
                "length": length,
            }
        )
        global_start += length

    for filename in ("tasks.jsonl", "modality.json", "stats.json"):
        shutil.copy2(source / "meta" / filename, output / "meta" / filename)
    write_jsonl(output / "meta/episodes.jsonl", rewritten_episodes)
    write_jsonl(
        output / "meta/episodes_stats.jsonl", rewritten_episode_stats
    )

    output_info = json.loads(json.dumps(info))
    output_info["total_episodes"] = 10
    output_info["total_frames"] = global_start
    output_info["total_videos"] = 10 * len(video_keys)
    output_info["total_chunks"] = 1
    output_info["splits"] = {"full": "0:10"}
    (output / "meta/info.json").write_text(
        json.dumps(output_info, indent=4) + "\n"
    )
    (output / "meta/source_episode_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n"
    )

    if any(path.is_symlink() for path in output.rglob("*")):
        raise RuntimeError("Materialized output unexpectedly contains symlinks")
    print(f"Created standalone dataset: {output}")
    print(f"Episodes: 10, frames: {global_start}, videos: {10 * len(video_keys)}")
    print("Mapping:")
    for row in mapping:
        print(
            f"  {row['episode_index']:02d} <- "
            f"source {row['source_episode_index']:03d} "
            f"success={row['success']} length={row['length']}"
        )


if __name__ == "__main__":
    main()
