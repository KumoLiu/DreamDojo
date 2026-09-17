#!/usr/bin/env python3
"""Run the milestone classifier over whole episodes and auto-label rollouts.

The teleop demonstrations are all successful, so a classifier trained only on
them has never seen a grasp that slips or a handover that drops the trocar --
exactly the states a reinforcement learning policy spends its early training
in. Labelling the rollout episodes, most of which fail, is what closes that
gap. This produces those labels automatically and ranks them so that a human
only has to check the ones the model is unsure about.

    .venv/bin/python -m scripts.milestone.infer --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.milestone.dataset import EpisodeFrames
from scripts.milestone.decode import decode, review_priority
from scripts.milestone.labels import (
    HEAD_NAMES,
    ROLLOUT_DATASETS,
    Episode,
    read_episodes,
    to_json,
)
from scripts.milestone.model import MilestoneNet

OUTPUT_ROOT = Path("/localhome/local-yunl/DreamDojo/outputs/milestone")


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
    parser.add_argument("--checkpoint", type=Path,
                        default=OUTPUT_ROOT / "v2/best.pt")
    parser.add_argument("--datasets", nargs="+", default=list(ROLLOUT_DATASETS))
    parser.add_argument("--output", type=Path,
                        default=OUTPUT_ROOT / "auto_labels.json")
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
        records.append({
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
        })

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
        print(f"  {r['dataset'].split('trocar_')[-1]:18s} ep{r['episode_index']:4d} "
              f"success={str(r['success']):5s} reached={r['reached']} "
              f"conflict={str(r['constraint_conflict']):5s} "
              f"priority={r['review_priority']:.2f}")


if __name__ == "__main__":
    main()
