"""Shared utilities for DreamDojo LeRobot inference and validation."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

CONFIG_FILE = (
    "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
)
CHUNK_SIZE = 12


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
    video_input = video_input.to(torch.uint8).unsqueeze(0).permute(0, 2, 1, 3, 4)
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
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_height), (0, 0, 0), -1)
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
