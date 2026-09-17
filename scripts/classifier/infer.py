#!/usr/bin/env python3
"""Predict probabilities, decode ordered stages, and auto-label whole episodes.

The teleop demonstrations are all successful, so a classifier trained only on
them has never seen a grasp that slips or a handover that drops the trocar --
exactly the states a reinforcement learning policy spends its early training
in. Labelling the rollout episodes, most of which fail, is what closes that
gap. This produces those labels automatically and ranks them so that a human
only has to check the ones the model is unsure about.

Offline decoding uses the entire sequence and can constrain the final state
using a recorded success label. It is not the causal online reward in reward.py.

    .venv/bin/python -m scripts.classifier.infer --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.classifier.dataset import EpisodeFrames
from scripts.classifier.labels import (
    HEAD_NAMES,
    NUM_HEADS,
    ROLLOUT_DATASETS,
    Episode,
    pad_milestones,
    read_episodes,
    to_json,
)
from scripts.classifier.model import MilestoneNet

OUTPUT_ROOT = Path("/localhome/local-yunl/DreamDojo/outputs/milestone")

NUM_STATES = NUM_HEADS + 1
EPS = 1e-6
AMBIGUOUS = (0.1, 0.9)


@dataclass
class Decoded:
    milestones: tuple[int, ...]
    states: np.ndarray
    mean_loglik: float
    reached: int
    ambiguous_frames: tuple[int, ...] = field(default=())
    constraint_conflict: bool = False

    @property
    def complete(self) -> bool:
        return self.reached == NUM_HEADS

    def reached_frames(self) -> tuple[int, ...]:
        """Only the milestones that actually fired, without the sentinels."""
        return self.milestones[: self.reached]


def _emissions(probs: np.ndarray) -> np.ndarray:
    """(T, NUM_STATES) log-likelihood of each frame under each phase."""
    p = np.clip(probs, EPS, 1 - EPS)
    pos, neg = np.log(p), np.log1p(-p)
    emission = np.zeros((len(p), NUM_STATES), dtype=np.float64)
    for state in range(NUM_STATES):
        # In state s the first s milestones have happened and the rest have not.
        emission[:, state] = pos[:, :state].sum(1) + neg[:, state:].sum(1)
    return emission


def _viterbi(
    emission: np.ndarray,
    allowed_final: set[int],
    advance_penalty: float,
) -> tuple[np.ndarray, float]:
    T = len(emission)
    score = np.full((T, NUM_STATES), -np.inf)
    back = np.zeros((T, NUM_STATES), dtype=np.int8)
    score[0, 0] = emission[0, 0]
    for t in range(1, T):
        for s in range(NUM_STATES):
            stay = score[t - 1, s]
            advance = score[t - 1, s - 1] - advance_penalty if s else -np.inf
            if advance > stay:
                score[t, s], back[t, s] = advance + emission[t, s], 1
            else:
                score[t, s], back[t, s] = stay + emission[t, s], 0

    final = [s for s in allowed_final if np.isfinite(score[-1, s])]
    if not final:
        final = [int(np.argmax(score[-1]))]
    end = max(final, key=lambda s: score[-1, s])

    states = np.zeros(T, dtype=np.int8)
    s = end
    for t in range(T - 1, 0, -1):
        states[t] = s
        s -= back[t, s]
    states[0] = s
    return states, float(score[-1, end])


def decode(
    probs: np.ndarray,
    *,
    success: bool | None = None,
    advance_penalty: float = 0.0,
) -> Decoded:
    """Decode (T, NUM_HEADS) probabilities into ordered milestone frames."""
    emission = _emissions(probs)
    unconstrained, _ = _viterbi(emission, set(range(NUM_STATES)), advance_penalty)

    if success is None:
        allowed = set(range(NUM_STATES))
    elif success:
        allowed = {NUM_HEADS}
    else:
        allowed = set(range(NUM_HEADS))
    states, total = _viterbi(emission, allowed, advance_penalty)

    reached = int(states[-1])
    transitions = [int(np.argmax(states >= k + 1)) for k in range(reached)]
    milestones = pad_milestones(tuple(transitions), len(probs))

    ambiguous = tuple(
        int(((probs[:, k] > AMBIGUOUS[0]) & (probs[:, k] < AMBIGUOUS[1])).sum())
        for k in range(NUM_HEADS)
    )
    return Decoded(
        milestones=milestones,
        states=states,
        mean_loglik=total / len(probs),
        reached=reached,
        ambiguous_frames=ambiguous,
        constraint_conflict=bool(int(unconstrained[-1]) != reached),
    )


def review_priority(d: Decoded) -> float:
    """Higher means the automatic label is less trustworthy.

    A conflict with the recorded outcome dominates everything else: the model
    and the ground truth disagree about what happened, so one of them is wrong.
    Below that, episodes are ranked by how poorly the decoded path explains the
    probabilities and by how long the model dithered around each transition.
    """
    dither = sum(d.ambiguous_frames) / max(1, len(d.states))
    return (100.0 if d.constraint_conflict else 0.0) - d.mean_loglik + 3.0 * dither


def load_checkpoint(checkpoint: Path) -> dict:
    # These files are written by train.py in this repository and carry the run
    # arguments alongside the weights, which the weights-only reader rejects.
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def load_model(checkpoint: Path, device: str = "cuda") -> MilestoneNet:
    model = MilestoneNet(pretrained=False)
    model.load_state_dict(load_checkpoint(checkpoint)["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict_probs(
    model: MilestoneNet,
    episodes: list[Episode],
    *,
    device: str = "cuda",
    batch_size: int = 256,
    workers: int = 12,
) -> dict[tuple[str, int], np.ndarray]:
    out = {}
    for ep in episodes:
        loader = DataLoader(
            EpisodeFrames(ep.dataset, ep.episode_index, ep.length),
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
        )
        chunks = []
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch.to(device, non_blocking=True))
            chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
        out[(ep.dataset, ep.episode_index)] = np.concatenate(chunks)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT_ROOT / "v2/best.pt")
    parser.add_argument("--datasets", nargs="+", default=list(ROLLOUT_DATASETS))
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "auto_labels.json")
    parser.add_argument("--advance-penalty", type=float, default=0.0)
    parser.add_argument(
        "--ignore-outcome",
        action="store_true",
        help="Decode without forcing success episodes to reach the plate.",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    episodes = [ep for d in args.datasets for ep in read_episodes(d)]
    print(f"labelling {len(episodes)} episodes from {len(args.datasets)} datasets")

    probs = predict_probs(model, episodes)
    records, conflicts = [], 0
    for ep in episodes:
        p = probs[(ep.dataset, ep.episode_index)]
        d = decode(
            p,
            success=None if args.ignore_outcome else ep.success,
            advance_penalty=args.advance_penalty,
        )
        conflicts += d.constraint_conflict
        records.append(
            {
                "dataset": ep.dataset,
                "episode_index": ep.episode_index,
                "length": ep.length,
                "success": ep.success,
                "milestones": to_json(d.milestones, ep.length),
                "reached": d.reached,
                "mean_loglik": round(d.mean_loglik, 4),
                "ambiguous_frames": list(d.ambiguous_frames),
                "constraint_conflict": d.constraint_conflict,
                "review_priority": round(review_priority(d), 4),
            }
        )

    records.sort(key=lambda r: -r["review_priority"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=1))

    probs_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        probs_path,
        **{f"{d}|{i}": v for (d, i), v in probs.items()},
    )

    complete = sum(r["reached"] == len(HEAD_NAMES) for r in records)
    print(f"\n{complete}/{len(records)} episodes reached all three milestones")
    print(f"{conflicts} conflicted with the recorded success label")
    for k, name in enumerate(HEAD_NAMES):
        reached = [r for r in records if r["reached"] > k]
        print(f"  {name:8s} reached in {len(reached):4d} episodes")
    print(f"\nlabels -> {args.output}")
    print(f"probabilities -> {probs_path}")
    print("\nhighest review priority:")
    for r in records[:10]:
        print(
            f"  {r['dataset'].split('trocar_')[-1]:18s} ep{r['episode_index']:4d} "
            f"success={str(r['success']):5s} reached={r['reached']} "
            f"conflict={str(r['constraint_conflict']):5s} "
            f"priority={r['review_priority']:.2f}"
        )


if __name__ == "__main__":
    main()
