#!/usr/bin/env python3
"""Choose the reward's thresholds and confirmation windows from reviewed data.

The classifier reports an operating point at 99% precision, and for this task it
is unusable: it is measured on the teleop split, every episode of which succeeds,
so nothing there constrains the false positive rate. It lands on 0.036 for
`picked` and an unreachable 1.0 for `placed`.

The reviewed rollout validation episodes do contain failures, which is exactly
what a reward has to survive. This scores the ratchet rule against them over a
grid and reports where it gets every episode right, so the defaults in
`reward.py` can be read off a measurement instead of guessed.

Two numbers are worth watching besides the count. One is how wide the winning
region is: a single winning cell would mean the setting was fitted to 30
episodes, while a broad plateau means it follows from something real. The other
is the delay, since the reward arrives that many frames after the event and at
15 fps a 24-frame window costs over a second.

Probabilities come from `MilestoneReward`'s own preprocessing, reading the
episode videos rather than the JPEG frame cache that training and `infer.py`
use. The two are not interchangeable: at quality 92 the cache is close enough
for training and still moves the odd probability enough to flip a milestone
whose window barely fits inside the episode, which is how a setting calibrated
on the cache scored 30 of 30 there and 29 of 30 in the reward itself.

    .venv/bin/python -m scripts.classifier.calibrate_reward
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.classifier.infer import OUTPUT_ROOT
from scripts.classifier.labels import DATASET_ROOT, HEAD_NAMES, NUM_HEADS
from scripts.classifier.reward import MilestoneReward

NEVER = {"never", "none", "-", ""}


def cache_path(checkpoint: Path) -> Path:
    """One cache per checkpoint, so two models can be compared on one grid.

    A single shared file silently answered for whichever model ran last, which
    is the one mistake that would make a calibration look like a model result.
    """
    return OUTPUT_ROOT / f"reward_probs_val_{checkpoint.parent.name}.npz"


def video_path(dataset: str, episode_index: int) -> Path:
    return (
        DATASET_ROOT
        / dataset
        / "videos/chunk-000/observation.images.cam_head"
        / f"episode_{episode_index:06d}.mp4"
    )


@torch.no_grad()
def episode_probs(reward: MilestoneReward, path: Path) -> np.ndarray:
    """Per-frame probabilities exactly as the reward sees them."""
    reward.reset()
    capture = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        out.append(reward.step(frame).probs)
    capture.release()
    return np.stack(out)


def load(sheet: Path, checkpoint: Path, refresh: bool):
    rows = list(csv.DictReader(open(sheet)))
    truths = [
        [None if str(r[h]).strip().lower() in NEVER else int(r[h]) for h in HEAD_NAMES]
        for r in rows
    ]
    keys = [f"{r['dataset']}|{int(r['episode_index'])}" for r in rows]

    cache = cache_path(checkpoint)
    if cache.exists() and not refresh:
        stored = np.load(cache)
        if all(k in stored for k in keys):
            return [(stored[k], t) for k, t in zip(keys, truths)]

    print(f"running {checkpoint.parent.name} over {len(rows)} episode videos")
    reward = MilestoneReward(checkpoint)
    probs = {
        k: episode_probs(reward, video_path(r["dataset"], int(r["episode_index"])))
        for k, r in zip(keys, rows)
    }
    np.savez_compressed(cache, **probs)
    return [(probs[k], t) for k, t in zip(keys, truths)]


def ratchet(
    probs: np.ndarray,
    thresholds: tuple[float, ...],
    window: tuple[int, ...],
    count: tuple[int, ...],
) -> tuple[int, list[int | None]]:
    """Replay `MilestoneReward`'s decision rule over precomputed probabilities."""
    recent = [deque(maxlen=w) for w in window]
    stage, first = 0, [None] * NUM_HEADS
    for t in range(len(probs)):
        for k in range(NUM_HEADS):
            recent[k].append(probs[t, k] >= thresholds[k])
        while stage < NUM_HEADS and sum(recent[stage]) >= count[stage]:
            first[stage] = t
            stage += 1
    return stage, first


def score(episodes, thresholds, window, count):
    exact, delays = 0, [[] for _ in HEAD_NAMES]
    granted_false, missed_true = 0, 0
    for probs, truth in episodes:
        stage, first = ratchet(probs, thresholds, window, count)
        reached = sum(t is not None for t in truth)
        exact += stage == reached
        for k, t in enumerate(truth):
            if t is None and first[k] is not None:
                granted_false += 1
            elif t is not None and first[k] is None:
                missed_true += 1
            elif t is not None:
                delays[k].append(first[k] - t)
    median = [int(np.median(d)) if d else -1 for d in delays]
    return exact, median, granted_false, missed_true


def margin(episodes, thresholds, window, count) -> float:
    """How many extra positive frames the nearest wrong episode would need.

    A setting that gets everything right because one negative episode fell two
    votes short is not the same as one where it fell eight short, and the
    difference is invisible in the pass/fail count. This reports the smallest
    such slack over the episodes that must be rejected, in votes.
    """
    worst = float("inf")
    for probs, truth in episodes:
        for k, t in enumerate(truth):
            if t is not None:
                continue
            above = probs[:, k] >= thresholds[k]
            # Largest number of positives the head ever accumulated inside one
            # window; the rule fires when this reaches count[k].
            if len(above) < window[k]:
                peak = int(above.sum())
            else:
                sums = np.convolve(
                    above.astype(int), np.ones(window[k], int), mode="valid"
                )
                peak = int(sums.max())
            worst = min(worst, count[k] - peak)
    return worst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sheet", type=Path, default=OUTPUT_ROOT / "review_v1_val/review_sheet.csv"
    )
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT_ROOT / "v2/best.pt")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run the model instead of reusing cached probabilities.",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.8, 0.9, 0.95])
    args = parser.parse_args()

    episodes = load(args.sheet, args.checkpoint, args.refresh)
    n = len(episodes)
    print(f"{n} reviewed episodes from {args.sheet.parent.name}\n")

    # A grasp and a handover persist to the end of the episode; a placement is
    # followed by almost nothing, so the two are searched on separate grids.
    grasp_grid = [(7, 5), (10, 8), (12, 10), (15, 13), (18, 16), (20, 18), (24, 22)]
    place_grid = [(2, 2), (3, 3), (5, 4), (7, 5), (10, 8)]

    print(
        f"{'thr':>5} {'pick/hand':>10} {'placed':>8} {'exact':>7} "
        f"{'false':>6} {'missed':>7} {'margin':>7} "
        f"{'delay picked/handed/placed':>28}"
    )
    winners = []
    for thr in args.thresholds:
        for gw, gc in grasp_grid:
            for pw, pc in place_grid:
                window, count = (gw, gw, pw), (gc, gc, pc)
                thresholds = (thr, thr, thr)
                exact, delay, false, missed = score(episodes, thresholds, window, count)
                if exact != n:
                    continue
                slack = margin(episodes, thresholds, window, count)
                winners.append((slack, -sum(delay), thr, window, count, delay))
                print(
                    f"{thr:>5} {f'{gw}/{gc}':>10} {f'{pw}/{pc}':>8} "
                    f"{exact:>4}/{n} {false:>6} {missed:>7} {slack:>7} "
                    f"{str(delay):>28}"
                )

    if not winners:
        print("no setting labels every episode correctly")
        return

    print(
        f"\n{len(winners)} of "
        f"{len(args.thresholds) * len(grasp_grid) * len(place_grid)} settings "
        f"get all {n} right."
    )
    # Among those, take the one the nearest wrong episode is furthest from
    # tripping, and break ties towards the one that pays out soonest.
    slack, _, thr, window, count, delay = max(winners)
    print("\nsuggested defaults, the widest margin in that region:")
    print(f"  THRESHOLDS     = {tuple(round(thr, 2) for _ in HEAD_NAMES)}")
    print(f"  CONFIRM_WINDOW = {window}")
    print(f"  CONFIRM_COUNT  = {count}")
    print(
        f"\nthe nearest episode that must be rejected is {slack} votes short "
        f"of being granted"
    )
    print(
        f"reward arrives this many frames after the event "
        f"({', '.join(HEAD_NAMES)}): {delay}"
    )


if __name__ == "__main__":
    main()
