#!/usr/bin/env python3
"""Deployed v2 classifier as a causal, three-stage reward at native 30 Hz.

Each confirmed milestone pays once, in order; later drops do not revoke rewards.
Pick/hand require 13 positives in 15 ticks, place requires 2 in 2, all at 0.8.
Thresholds were calibrated on reviewed rollouts including failures, not on
successful teleop alone. See docs/TROCAR_PROJECT.md for the experiment summary.

This CLI reads every input frame once. RLinf separately duplicates 15 fps WM
frames into 30 Hz ticks and handles batched histories and KIR initialization.

    .venv/bin/python -m scripts.classifier.reward --video clip.mp4
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.classifier.dataset import (
    CROP_SIZE,
    FRAME_OFFSETS,
    STORE_SIZE,
    _normalise,
)
from scripts.classifier.infer import OUTPUT_ROOT, load_checkpoint
from scripts.classifier.labels import HEAD_NAMES, NUM_HEADS
from scripts.classifier.model import MilestoneNet

# Order: picked, handed, placed. Shorter place confirmation accommodates clips
# ending immediately after placement. Calibrate with actual reward preprocessing.
CONFIRM_WINDOW = (15, 15, 2)
CONFIRM_COUNT = (13, 13, 2)
THRESHOLDS = (0.8, 0.8, 0.8)


@dataclass
class Step:
    reward: float
    stage: int
    progress: float
    probs: np.ndarray
    newly_reached: tuple[str, ...]


class MilestoneReward:
    def __init__(
        self,
        checkpoint: Path = OUTPUT_ROOT / "v2/best.pt",
        *,
        thresholds: tuple[float, ...] = THRESHOLDS,
        confirm_window: tuple[int, ...] = CONFIRM_WINDOW,
        confirm_count: tuple[int, ...] = CONFIRM_COUNT,
        device: str = "cuda",
    ) -> None:
        state = load_checkpoint(checkpoint)
        self.model = MilestoneNet(pretrained=False)
        self.model.load_state_dict(state["model"])
        self.model.to(device).eval()
        self.device = device

        self.thresholds = np.asarray(thresholds, dtype=np.float32)
        self.confirm_window = confirm_window
        self.confirm_count = confirm_count
        self.reset()

    def reset(self) -> None:
        self.frames: deque[np.ndarray] = deque(maxlen=max(FRAME_OFFSETS) + 1)
        self.recent = [deque(maxlen=w) for w in self.confirm_window]
        self.stage = 0

    @staticmethod
    def _prepare(frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.shape[:2][::-1] != STORE_SIZE:
            frame_bgr = cv2.resize(frame_bgr, STORE_SIZE, interpolation=cv2.INTER_AREA)
        x0 = (STORE_SIZE[0] - CROP_SIZE[0]) // 2
        y0 = (STORE_SIZE[1] - CROP_SIZE[1]) // 2
        return frame_bgr[y0 : y0 + CROP_SIZE[1], x0 : x0 + CROP_SIZE[0]]

    @torch.no_grad()
    def step(self, frame_bgr: np.ndarray) -> Step:
        """Feed the newest camera frame and get the reward for this timestep."""
        self.frames.append(self._prepare(frame_bgr))
        history = list(self.frames)[::-1]  # index 0 is the newest
        stack = [history[min(o, len(history) - 1)] for o in FRAME_OFFSETS]

        batch = _normalise(stack).unsqueeze(0).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(batch)
        probs = torch.sigmoid(logits.float())[0].cpu().numpy()

        above = probs >= self.thresholds
        for k in range(NUM_HEADS):
            self.recent[k].append(bool(above[k]))

        before = self.stage
        # Advance one milestone at a time and only in order, so a head firing
        # out of sequence cannot skip the ones before it.
        while (
            self.stage < NUM_HEADS
            and sum(self.recent[self.stage]) >= self.confirm_count[self.stage]
        ):
            self.stage += 1

        return Step(
            reward=float(self.stage - before),
            stage=self.stage,
            progress=float(probs.sum()),
            probs=probs,
            newly_reached=tuple(HEAD_NAMES[before : self.stage]),
        )

    def potential(self) -> float:
        """Shaping potential for a potential-based bonus.

        Used as gamma * phi(s') - phi(s), a dense term built from this leaves
        the optimal policy unchanged, which no other dense shaping guarantees.
        """
        return float(self.stage)


def score_video(path: Path, reward: MilestoneReward) -> dict:
    """Run the reward over a video, e.g. a world-model rollout."""
    reward.reset()
    capture = cv2.VideoCapture(str(path))
    total, frames, first = 0.0, 0, {}
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        step = reward.step(frame)
        for name in step.newly_reached:
            first[name] = frames
        total += step.reward
        frames += 1
    capture.release()
    return {
        "video": str(path),
        "frames": frames,
        "total_reward": total,
        "final_stage": reward.stage,
        "first_reached": first,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT_ROOT / "v2/best.pt")
    args = parser.parse_args()

    reward = MilestoneReward(args.checkpoint)
    print(
        "thresholds: "
        + ", ".join(f"{n} {t:.3f}" for n, t in zip(HEAD_NAMES, reward.thresholds))
    )
    result = score_video(args.video, reward)
    print(
        f"\n{result['frames']} frames, total reward {result['total_reward']:.0f}, "
        f"final stage {result['final_stage']}/{NUM_HEADS}"
    )
    for name in HEAD_NAMES:
        frame = result["first_reached"].get(name)
        print(f"  {name:8s} {'frame ' + str(frame) if frame is not None else 'never'}")


if __name__ == "__main__":
    main()
