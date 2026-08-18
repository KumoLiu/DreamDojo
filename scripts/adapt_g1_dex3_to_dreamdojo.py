#!/usr/bin/env python3
"""Adapt a G1-Dex3 LeRobot dataset to DreamDojo's official G1 layout.

This is shared by assemble-trocar and pick-trocar datasets. It:
- remaps 28-D upper-body state/action into the official 43-D G1 layout,
  padding the unused leg and waist dimensions with zeros;
- maps a configurable camera key to video.ego_view in modality.json;
- updates metadata, train/test split declarations, and feature names;
- recomputes state/action statistics; and
- symlinks videos instead of duplicating them.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


G1_DIM = 43
# Official G1 layout slices
LEFT_ARM = slice(15, 22)
LEFT_HAND = slice(22, 29)
RIGHT_ARM = slice(29, 36)
RIGHT_HAND = slice(36, 43)
# Source (user) layout: left_arm, right_arm, left_hand, right_hand
SRC_LEFT_ARM = slice(0, 7)
SRC_RIGHT_ARM = slice(7, 14)
SRC_LEFT_HAND = slice(14, 21)
SRC_RIGHT_HAND = slice(21, 28)


def pad_28_to_43(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    assert x.shape[-1] == 28, f"expected 28-D, got {x.shape}"
    out = np.zeros(x.shape[:-1] + (G1_DIM,), dtype=np.float32)
    out[..., LEFT_ARM] = x[..., SRC_LEFT_ARM]
    out[..., LEFT_HAND] = x[..., SRC_LEFT_HAND]
    out[..., RIGHT_ARM] = x[..., SRC_RIGHT_ARM]
    out[..., RIGHT_HAND] = x[..., SRC_RIGHT_HAND]
    return out


def convert_parquet(src: Path, dst: Path) -> None:
    table = pq.read_table(src)
    n = table.num_rows
    state = np.stack([np.asarray(table.column("observation.state")[i].as_py(), dtype=np.float32) for i in range(n)])
    action = np.stack([np.asarray(table.column("action")[i].as_py(), dtype=np.float32) for i in range(n)])
    state43 = pad_28_to_43(state)
    action43 = pad_28_to_43(action)

    state_arr = pa.FixedSizeListArray.from_arrays(pa.array(state43.reshape(-1), type=pa.float32()), G1_DIM)
    action_arr = pa.FixedSizeListArray.from_arrays(pa.array(action43.reshape(-1), type=pa.float32()), G1_DIM)

    arrays = []
    names = []
    for name in table.column_names:
        if name == "observation.state":
            arrays.append(state_arr)
        elif name == "action":
            arrays.append(action_arr)
        else:
            arrays.append(table.column(name))
        names.append(name)

    out = pa.Table.from_arrays(arrays, names=names)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dst)


def build_modality(camera_key: str) -> dict:
    # Minimal G1 modality used by DreamDojo groot_configs (legs/waist/arms/hands + ego_view)
    return {
        "state": {
            "left_leg": {"start": 0, "end": 6},
            "right_leg": {"start": 6, "end": 12},
            "waist": {"start": 12, "end": 15},
            "left_arm": {"start": 15, "end": 22},
            "left_hand": {"start": 22, "end": 29},
            "right_arm": {"start": 29, "end": 36},
            "right_hand": {"start": 36, "end": 43},
        },
        "action": {
            "left_leg": {"start": 0, "end": 6},
            "right_leg": {"start": 6, "end": 12},
            "waist": {"start": 12, "end": 15},
            "left_arm": {"start": 15, "end": 22},
            "left_hand": {"start": 22, "end": 29},
            "right_arm": {"start": 29, "end": 36},
            "right_hand": {"start": 36, "end": 43},
        },
        "video": {
            "ego_view": {"original_key": camera_key},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def update_info(info: dict, camera_key: str) -> dict:
    info = json.loads(json.dumps(info))  # deep copy
    camera_name = camera_key.rsplit(".", 1)[-1]
    info["robot_type"] = f"Unitree_G1_adapted_from_Dex3_{camera_name}"
    # Keep source camera features; modality.json selects which one is ego_view.
    # Update state/action shapes to 43
    g1_names = (
        [f"leg_L_{i}" for i in range(6)]
        + [f"leg_R_{i}" for i in range(6)]
        + [f"waist_{i}" for i in range(3)]
        + [
            "kLeftShoulderPitch",
            "kLeftShoulderRoll",
            "kLeftShoulderYaw",
            "kLeftElbow",
            "kLeftWristRoll",
            "kLeftWristPitch",
            "kLeftWristYaw",
            "kLeftHandThumb0",
            "kLeftHandThumb1",
            "kLeftHandThumb2",
            "kLeftHandMiddle0",
            "kLeftHandMiddle1",
            "kLeftHandIndex0",
            "kLeftHandIndex1",
            "kRightShoulderPitch",
            "kRightShoulderRoll",
            "kRightShoulderYaw",
            "kRightElbow",
            "kRightWristRoll",
            "kRightWristPitch",
            "kRightWristYaw",
            "kRightHandThumb0",
            "kRightHandThumb1",
            "kRightHandThumb2",
            "kRightHandIndex0",
            "kRightHandIndex1",
            "kRightHandMiddle0",
            "kRightHandMiddle1",
        ]
    )
    for key in ("observation.state", "action"):
        info["features"][key]["shape"] = [G1_DIM]
        info["features"][key]["names"] = [g1_names]
    # Prefer train/test split so DreamDojo eval --data-split test works
    n = info["total_episodes"]
    n_test = max(1, int(round(n * 0.05)))
    n_train = n - n_test
    info["splits"] = {"train": f"0:{n_train}", "test": f"{n_train}:{n}"}
    return info


def recompute_stats(data_dir: Path) -> dict:
    states = []
    actions = []
    for pq_path in sorted(data_dir.glob("chunk-*/episode_*.parquet")):
        table = pq.read_table(pq_path, columns=["observation.state", "action"])
        for i in range(table.num_rows):
            states.append(table.column("observation.state")[i].as_py())
            actions.append(table.column("action")[i].as_py())
    state = np.asarray(states, dtype=np.float64)
    action = np.asarray(actions, dtype=np.float64)

    def pack(arr: np.ndarray) -> dict:
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }

    return {"observation.state": pack(state), "action": pack(action)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt a G1-Dex3 LeRobot dataset to DreamDojo's G1 schema."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path.home() / "assemble_trocar_sim_box_v3_60",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/localhome/local-yunl/DreamDojo/datasets/g1_assemble_trocar_sim_box_v3_60"),
    )
    parser.add_argument(
        "--camera-key",
        default="observation.images.cam_room",
        help=(
            "LeRobot video feature to expose as DreamDojo video.ego_view "
            "(e.g. observation.images.cam_room or observation.images.cam_head)."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    camera_key: str = args.camera_key

    source_info = json.loads((src / "meta" / "info.json").read_text())
    camera_feature = source_info.get("features", {}).get(camera_key)
    if camera_feature is None or camera_feature.get("dtype") != "video":
        raise SystemExit(f"Camera key is not a video feature in source info.json: {camera_key}")
    camera_dirs = list((src / "videos").glob(f"chunk-*/{camera_key}"))
    if not camera_dirs:
        raise SystemExit(f"No videos found for camera key: {camera_key}")

    if dst.exists():
        if not args.force:
            raise SystemExit(f"Destination exists: {dst} (pass --force to overwrite)")
        shutil.rmtree(dst)

    dst.mkdir(parents=True)
    (dst / "meta").mkdir()
    (dst / "data").mkdir()
    (dst / "videos").mkdir()

    # Copy / rewrite meta
    info = update_info(source_info, camera_key)
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    (dst / "meta" / "modality.json").write_text(
        json.dumps(build_modality(camera_key), indent=4) + "\n"
    )
    for name in ("episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"):
        shutil.copy2(src / "meta" / name, dst / "meta" / name)

    # Convert parquets
    src_parquets = sorted((src / "data").glob("chunk-*/episode_*.parquet"))
    print(f"Converting {len(src_parquets)} parquet files...")
    for i, sp in enumerate(src_parquets):
        rel = sp.relative_to(src / "data")
        convert_parquet(sp, dst / "data" / rel)
        if (i + 1) % 10 == 0 or i + 1 == len(src_parquets):
            print(f"  {i + 1}/{len(src_parquets)}")

    # Symlink videos (keep original keys so modality original_key works)
    for cam_dir in sorted((src / "videos").glob("chunk-*/*")):
        if not cam_dir.is_dir():
            continue
        rel = cam_dir.relative_to(src / "videos")
        target = dst / "videos" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(cam_dir.resolve())
        print(f"symlink videos/{rel} -> {cam_dir}")

    # Stats from converted data (DreamDojo may still prefer shared_meta/G1_stats for norm)
    stats = recompute_stats(dst / "data")
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4) + "\n")
    print(f"Done. Adapted dataset at {dst}")


if __name__ == "__main__":
    main()
