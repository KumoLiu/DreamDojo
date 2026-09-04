"""Shared utilities for DreamDojo LeRobot inference and validation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from loguru import logger

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from groot_dreams.dataloader import MultiVideoActionDataset


CONFIG_FILE = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
CHUNK_SIZE = 12
TIMESTEP_INTERVAL = 2


def largest_chunk_aligned_frames(
    base_index: int,
    episode_len: int,
    *,
    chunk_size: int = CHUNK_SIZE,
    timestep_interval: int = TIMESTEP_INTERVAL,
) -> int:
    """Return the largest valid frame count aligned to complete action chunks."""
    max_n = (episode_len - base_index - 1) // timestep_interval
    num_frames = 1 + chunk_size * ((max_n - 1) // chunk_size)
    if num_frames < 1 + chunk_size:
        raise RuntimeError(f"Not enough frames from base_index={base_index}")
    return num_frames


def episode_lengths(dataset_path: str | Path) -> dict[int, int]:
    """Read episode lengths from LeRobot metadata."""
    path = Path(dataset_path) / "meta" / "episodes.jsonl"
    lengths: dict[int, int] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        lengths[int(row["episode_index"])] = int(row["length"])
    return lengths


def ego_view_source_key(dataset_path: str | Path) -> str:
    """Return the LeRobot video feature mapped to DreamDojo's ego_view."""
    modality_path = Path(dataset_path) / "meta" / "modality.json"
    modality = json.loads(modality_path.read_text())
    try:
        return str(modality["video"]["ego_view"]["original_key"])
    except KeyError as error:
        raise ValueError(
            f"Missing video.ego_view.original_key in {modality_path}"
        ) from error


def load_lerobot_episode(
    dataset_path: str,
    episode_index: int,
    num_frames: int,
) -> dict:
    """Load one full-prefix LeRobot episode sample starting at base index zero."""
    dataset = MultiVideoActionDataset(
        num_frames=num_frames,
        dataset_path=dataset_path,
        data_split="full",
        single_base_index=True,
        restrict_len=None,
        deterministic_uniform_sampling=False,
    )
    inner = dataset.datasets[0].lerobot_dataset
    trajectory_id, base_index = inner.all_steps[episode_index]
    if int(trajectory_id) != episode_index or int(base_index) != 0:
        raise RuntimeError(
            f"Expected episode {episode_index} base 0; got "
            f"{trajectory_id}, {base_index}"
        )
    data = dataset[episode_index]
    logger.info(
        f"Loaded episode={episode_index}, video={tuple(data['video'].shape)}, "
        f"action={tuple(data['action'].shape)}"
    )
    return data


def generated_to_uint8(generated: torch.Tensor) -> np.ndarray:
    """Convert a generated BCTHW tensor in [-1, 1] to uint8 THWC."""
    return (
        (torch.clamp((generated[0] + 1.0) / 2.0, 0, 1) * 255)
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )


def _condition_tensor(condition: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(condition, np.ndarray):
        if condition.ndim != 3:
            raise ValueError(f"Expected HWC condition, got {condition.shape}")
        return torch.from_numpy(condition.copy()).permute(2, 0, 1).unsqueeze(0)

    if condition.ndim == 3:
        return condition.unsqueeze(0)
    if condition.ndim == 4 and condition.shape[0] == 1:
        return condition
    raise ValueError(f"Expected CHW or 1CHW condition, got {tuple(condition.shape)}")


def generate_chunk(
    pipeline: Video2WorldInference,
    condition: np.ndarray | torch.Tensor,
    actions: torch.Tensor,
    lam_video: torch.Tensor,
    *,
    seed: int,
    num_inference_steps: int | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """Generate one action-conditioned chunk from a single condition frame."""
    image = _condition_tensor(condition)
    video_input = torch.cat(
        [image, torch.zeros_like(image).repeat(chunk_size, 1, 1, 1)],
        dim=0,
    )
    video_input = (
        video_input.to(torch.uint8).unsqueeze(0).permute(0, 2, 1, 3, 4)
    )
    kwargs = {}
    if num_inference_steps is not None:
        kwargs["num_steps"] = num_inference_steps
    generated = pipeline.generate_vid2world(
        prompt="",
        input_path=video_input,
        action=actions.float(),
        guidance=0,
        num_video_frames=chunk_size + 1,
        num_latent_conditional_frames=1,
        resolution="480,640",
        seed=seed,
        lam_video=lam_video,
        **kwargs,
    )
    return generated_to_uint8(generated)


def generate_closed_loop(
    pipeline: Video2WorldInference,
    data: dict,
    num_frames: int,
) -> np.ndarray:
    """Generate a full closed-loop rollout using GT actions.

    This preserves the seed and stitching behavior of the original long-form
    inference script.
    """
    actions = data["action"][: num_frames - 1]
    lam_video = data["lam_video"]
    current: np.ndarray | torch.Tensor = data["video"].transpose(0, 1)[:1]
    chunks: list[np.ndarray] = []

    for chunk_index, start in enumerate(range(0, len(actions), CHUNK_SIZE)):
        action_chunk = actions[start : start + CHUNK_SIZE]
        if action_chunk.shape[0] != CHUNK_SIZE:
            raise RuntimeError(f"Incomplete action chunk at {start}")
        generated = generate_chunk(
            pipeline,
            current,
            action_chunk,
            lam_video[start * 2 : (start + CHUNK_SIZE) * 2],
            seed=start,
        )
        current = generated[-1]
        chunks.append(generated)
        logger.info(f"chunk {chunk_index + 1}/{len(actions) // CHUNK_SIZE}")

    stitched = [chunks[0]] + [chunk[:CHUNK_SIZE] for chunk in chunks[1:]]
    return np.concatenate(stitched, axis=0)


def add_label(
    video: np.ndarray,
    label: str,
    color: tuple[int, int, int] = (255, 255, 255),
    *,
    bar_height: int = 48,
    font_scale: float = 0.7,
) -> np.ndarray:
    """Add a translucent label bar to every frame."""
    labeled = video.copy()
    text_y = min(32, bar_height - 10)
    for frame in labeled:
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (0, 0), (frame.shape[1], bar_height), (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(
            frame,
            label,
            (12, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA,
        )
    return labeled
