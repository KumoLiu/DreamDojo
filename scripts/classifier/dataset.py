"""Frame caching and frame-stack datasets for the three-stage classifier.

A single head-camera frame cannot resolve the handover: mid-transfer both hands
are wrapped around the trocar and whether the grip has actually changed is only
visible in how the scene is moving. Every sample therefore carries a short
causal history, stacked along channels so the backbone stays an ordinary
ResNet. Only past frames are used, so the same model runs online as a reward.

Augmentation deliberately excludes horizontal flips. Two of the three
milestones are defined by *which hand* holds the trocar, and a mirrored frame
carries the opposite label.

Decode each episode once into a JPEG cache for fast random training access.
Cache frames are slightly larger than the training crop to allow crop jitter:

    .venv/bin/python -m scripts.classifier.dataset
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.classifier.labels import (
    NUM_HEADS,
    ROLLOUT_DATASETS,
    TELEOP_DATASETS,
    Episode,
    ExternalLabel,
    cumulative_targets,
    loss_weights,
    read_episodes,
)

CACHE_ROOT = Path("/localhome/local-yunl/DreamDojo/datasets/milestone_cache")
STORE_SIZE = (360, 270)  # width, height; training crops 320x240 out of this
JPEG_QUALITY = 92

# Offsets into the past, in frames at the native 30 fps: now, 133 ms, 267 ms
# and 533 ms ago.
FRAME_OFFSETS = (0, 4, 8, 16)
NUM_STACK = len(FRAME_OFFSETS)

CROP_SIZE = (320, 240)  # width, height
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Sample:
    dataset: str
    episode_index: int
    frame: int


def episode_dir(dataset: str, episode_index: int) -> Path:
    return CACHE_ROOT / dataset / f"episode_{episode_index:06d}"


def frame_path(dataset: str, episode_index: int, frame: int) -> Path:
    return episode_dir(dataset, episode_index) / f"{frame:06d}.jpg"


def _extract(task: tuple[str, int, int, str]) -> tuple[str, int, int, str]:
    dataset, episode_index, expected, video = task
    out = episode_dir(dataset, episode_index)
    existing = len(list(out.glob("*.jpg"))) if out.exists() else 0
    if existing == expected:
        return dataset, episode_index, existing, "cached"
    out.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(video)
    written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if written >= expected:
            # The container can hold trailing frames the parquet does not index.
            break
        resized = cv2.resize(frame, STORE_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(
            str(out / f"{written:06d}.jpg"),
            resized,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        written += 1
    capture.release()

    note = "ok" if written == expected else f"MISMATCH expected {expected}"
    return dataset, episode_index, written, note


def _read(dataset: str, episode_index: int, frame: int) -> np.ndarray:
    image = cv2.imread(str(frame_path(dataset, episode_index, frame)))
    if image is None:
        raise FileNotFoundError(frame_path(dataset, episode_index, frame))
    return image


def _normalise(stack: list[np.ndarray]) -> torch.Tensor:
    """BGR uint8 crops -> (NUM_STACK*3, H, W) normalised float tensor."""
    arr = np.stack(stack).astype(np.float32) / 255.0
    arr = arr[..., ::-1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(0, 3, 1, 2).reshape(-1, arr.shape[1], arr.shape[2])
    return torch.from_numpy(np.ascontiguousarray(arr))


class MilestoneFrames(Dataset):
    def __init__(
        self,
        episodes: list[Episode],
        *,
        train: bool,
        labels: dict[tuple[str, int], ExternalLabel] | None = None,
        margin_mask: bool = True,
    ) -> None:
        self.train = train
        self.margin_mask = margin_mask
        self.episodes = episodes
        self.samples: list[Sample] = []
        self.targets: dict[tuple[str, int], np.ndarray] = {}
        self.weights: dict[tuple[str, int], np.ndarray] = {}

        for ep in episodes:
            key = (ep.dataset, ep.episode_index)
            external = labels.get(key) if labels else None
            if external is not None:
                milestones, length = external.milestones, external.length
                supervise = np.asarray(external.supervise, dtype=np.float32)
                drop = external.drop
            elif ep.milestones is not None:
                milestones, length = ep.milestones, ep.length
                supervise = np.ones(NUM_HEADS, dtype=np.float32)
                # Every teleop demonstration succeeds, so none of them drop.
                drop = None
            else:
                continue

            weights = (
                loss_weights(milestones, length, drop=drop)
                if margin_mask
                else np.ones((length, NUM_HEADS), dtype=np.float32)
            )
            self.targets[key] = cumulative_targets(milestones, length, drop=drop)
            self.weights[key] = weights * supervise[None, :]
            self.samples.extend(
                Sample(ep.dataset, ep.episode_index, t) for t in range(length)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_params(self) -> tuple[int, int]:
        max_x = STORE_SIZE[0] - CROP_SIZE[0]
        max_y = STORE_SIZE[1] - CROP_SIZE[1]
        if not self.train:
            return max_x // 2, max_y // 2
        return np.random.randint(max_x + 1), np.random.randint(max_y + 1)

    def __getitem__(self, index: int):
        s = self.samples[index]
        x0, y0 = self._crop_params()
        # One jitter draw for the whole stack: the frames show one scene a few
        # hundred milliseconds apart, and perturbing them independently would
        # inject motion that is not there.
        if self.train:
            gain = np.float32(np.random.uniform(0.8, 1.25))
            bias = np.float32(np.random.uniform(-20, 20))
        else:
            gain, bias = np.float32(1.0), np.float32(0.0)

        crops = []
        for offset in FRAME_OFFSETS:
            frame = max(0, s.frame - offset)
            image = _read(s.dataset, s.episode_index, frame)
            crop = image[y0 : y0 + CROP_SIZE[1], x0 : x0 + CROP_SIZE[0]]
            if self.train:
                crop = np.clip(crop.astype(np.float32) * gain + bias, 0, 255)
            crops.append(crop)

        key = (s.dataset, s.episode_index)
        return (
            _normalise(crops),
            torch.from_numpy(self.targets[key][s.frame]),
            torch.from_numpy(self.weights[key][s.frame]),
        )

    def positive_rates(self) -> np.ndarray:
        stacked = np.concatenate([self.targets[k] for k in self.targets])
        return stacked.mean(0)


class EpisodeFrames(Dataset):
    """Every frame of one episode in order, for inference."""

    def __init__(self, dataset: str, episode_index: int, length: int) -> None:
        self.dataset = dataset
        self.episode_index = episode_index
        self.length = length
        max_x = STORE_SIZE[0] - CROP_SIZE[0]
        max_y = STORE_SIZE[1] - CROP_SIZE[1]
        self.origin = (max_x // 2, max_y // 2)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, t: int) -> torch.Tensor:
        x0, y0 = self.origin
        crops = []
        for offset in FRAME_OFFSETS:
            image = _read(self.dataset, self.episode_index, max(0, t - offset))
            crops.append(image[y0 : y0 + CROP_SIZE[1], x0 : x0 + CROP_SIZE[0]])
        return _normalise(crops)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache video frames for classifier training."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(TELEOP_DATASETS + ROLLOUT_DATASETS)
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    tasks = []
    for dataset in args.datasets:
        for ep in read_episodes(dataset):
            video = ep.video_path
            if not video.exists():
                print(f"missing video: {video}")
                continue
            tasks.append((dataset, ep.episode_index, ep.length, str(video)))

    print(f"{len(tasks)} episodes to cache into {CACHE_ROOT}")
    problems, total = [], 0
    with Pool(args.workers) as pool:
        for i, (dataset, index, written, note) in enumerate(
            pool.imap_unordered(_extract, tasks), 1
        ):
            total += written
            if note not in ("ok", "cached"):
                problems.append(f"{dataset} ep{index}: wrote {written}, {note}")
            if i % 100 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} episodes, {total} frames")

    print(f"\n{total} frames cached")
    if problems:
        print(f"{len(problems)} episodes disagreed with their declared length:")
        for p in problems:
            print(" ", p)
    else:
        print("every episode matched its declared frame count")


if __name__ == "__main__":
    main()
