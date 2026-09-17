#!/usr/bin/env python3
"""Train the milestone classifier.

Loss is plain masked BCE with no positive-class reweighting even though
`placed` is only 7% of frames. Upweighting the rare class trades precision for
recall, and for a reward function that is the wrong direction: paying out for a
placement that did not happen teaches the policy something false, while paying
out a few frames late costs almost nothing. The imbalance is handled after
training instead, by picking each head's threshold on the validation
precision-recall curve.

Validation reports frame-level average precision, but the number to watch is
the milestone frame error: how far the decoded transition sits from the one a
human marked.

    .venv/bin/python -m scripts.classifier.train --name v2_reproduction
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.classifier.dataset import MilestoneFrames
from scripts.classifier.infer import OUTPUT_ROOT, decode, predict_probs
from scripts.classifier.labels import (
    HEAD_NAMES,
    NUM_HEADS,
    TELEOP_DATASETS,
    load_external_labels,
    read_episodes,
)
from scripts.classifier.model import MilestoneNet

TRAIN_DATASET, VAL_DATASET = TELEOP_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", required=True, help="New run name; never overwrite v2."
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--extra-labels",
        type=Path,
        default=OUTPUT_ROOT / "reviewed_labels_v3.json",
        help="Final v2 training labels; v3 in this filename is the label revision.",
    )
    parser.add_argument("--include-rollout-val", action="store_true")
    return parser.parse_args()


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    hits = labels[order]
    if hits.sum() == 0:
        return float("nan")
    cum = np.cumsum(hits)
    precision = cum / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def threshold_for_precision(
    scores: np.ndarray, labels: np.ndarray, target: float
) -> tuple[float, float]:
    """Lowest threshold whose precision still meets the target, and its recall."""
    order = np.argsort(-scores)
    hits = labels[order]
    cum = np.cumsum(hits)
    precision = cum / np.arange(1, len(hits) + 1)
    recall = cum / max(1, hits.sum())
    ok = np.flatnonzero(precision >= target)
    if len(ok) == 0:
        return 1.0, 0.0
    i = ok[-1]
    return float(scores[order][i]), float(recall[i])


@torch.no_grad()
def evaluate(model: MilestoneNet, episodes, device: str) -> dict:
    model.eval()
    probs = predict_probs(model, episodes, device=device)
    scores = np.concatenate([probs[(e.dataset, e.episode_index)] for e in episodes])
    truth = np.concatenate(
        [
            (np.arange(e.length)[:, None] >= np.asarray(e.milestones)[None, :])
            for e in episodes
        ]
    ).astype(np.float32)

    report = {"per_head": {}}
    for k, name in enumerate(HEAD_NAMES):
        thresh, recall = threshold_for_precision(scores[:, k], truth[:, k], 0.99)
        report["per_head"][name] = {
            "ap": round(average_precision(scores[:, k], truth[:, k]), 4),
            "threshold_p99": round(thresh, 4),
            "recall_at_p99": round(recall, 4),
        }

    errors = [[] for _ in range(NUM_HEADS)]
    missed = 0
    for e in episodes:
        d = decode(probs[(e.dataset, e.episode_index)], success=True)
        if not d.complete:
            missed += 1
            continue
        for k in range(NUM_HEADS):
            errors[k].append(abs(d.milestones[k] - e.milestones[k]))

    report["frame_error"] = {
        name: {
            "median": float(np.median(errs)) if errs else float("nan"),
            "mean": round(float(np.mean(errs)), 2) if errs else float("nan"),
            "within_5": round(float(np.mean(np.asarray(errs) <= 5)), 3)
            if errs
            else 0.0,
            "within_10": round(float(np.mean(np.asarray(errs) <= 10)), 3)
            if errs
            else 0.0,
        }
        for name, errs in zip(HEAD_NAMES, errors)
    }
    report["episodes_without_full_chain"] = missed
    report["mean_ap"] = round(
        float(np.nanmean([v["ap"] for v in report["per_head"].values()])), 4
    )
    report["median_frame_error"] = round(
        float(np.mean([report["frame_error"][n]["median"] for n in HEAD_NAMES])), 2
    )
    return report


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    out_dir = OUTPUT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=False)

    train_episodes = [e for e in read_episodes(TRAIN_DATASET) if e.labelled]
    val_episodes = [e for e in read_episodes(VAL_DATASET) if e.labelled]
    extra = None
    if args.extra_labels:
        extra = load_external_labels(args.extra_labels)
        if not args.include_rollout_val:
            # Held out so that the rollout distribution, which is the one the
            # teleop validation split cannot speak for, stays measurable.
            extra = {k: v for k, v in extra.items() if not k[0].endswith("_val")}
        for dataset in sorted({d for d, _ in extra}):
            train_episodes += [
                e
                for e in read_episodes(dataset)
                if (e.dataset, e.episode_index) in extra
            ]
        print(f"folding in {len(extra)} rollout episodes")
        by_source = Counter(v.provenance for v in extra.values())
        for tag, n in by_source.most_common():
            print(f"    {n:4d}  {tag}")

    train_set = MilestoneFrames(train_episodes, train=True, labels=extra)
    print(f"train {len(train_set)} frames from {len(train_episodes)} episodes")
    print(
        f"val   {sum(e.length for e in val_episodes)} frames "
        f"from {len(val_episodes)} episodes"
    )
    print(
        "positive rates: "
        + ", ".join(
            f"{n} {r:.3f}" for n, r in zip(HEAD_NAMES, train_set.positive_rates())
        )
    )

    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    model = MilestoneNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * len(loader),
        pct_start=0.1,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    history, best = [], -1.0
    for epoch in range(args.epochs):
        model.train()
        started, running, seen = time.time(), 0.0, 0
        for images, targets, weights in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
            loss_per = criterion(logits.float(), targets) * weights
            loss = loss_per.sum() / weights.sum().clamp(min=1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            running += loss.item() * len(images)
            seen += len(images)

        report = evaluate(model, val_episodes, device)
        report["epoch"] = epoch
        report["train_loss"] = round(running / seen, 5)
        report["seconds"] = round(time.time() - started, 1)
        history.append(report)
        print(
            f"\nepoch {epoch}  loss {report['train_loss']:.4f}  "
            f"mAP {report['mean_ap']:.4f}  "
            f"median frame error {report['median_frame_error']:.1f}  "
            f"({report['seconds']:.0f}s)"
        )
        for name in HEAD_NAMES:
            h, f = report["per_head"][name], report["frame_error"][name]
            print(
                f"    {name:8s} AP {h['ap']:.4f}  recall@P99 {h['recall_at_p99']:.3f}"
                f"  frame err med {f['median']:.0f} within10 {f['within_10']:.2f}"
            )

        if report["mean_ap"] > best:
            best = report["mean_ap"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": {k: str(v) for k, v in vars(args).items()},
                    "report": report,
                },
                out_dir / "best.pt",
            )
            print(f"    saved best (mAP {best:.4f})")

    (out_dir / "history.json").write_text(json.dumps(history, indent=1, default=str))
    print(f"\nbest mAP {best:.4f}; checkpoint {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
