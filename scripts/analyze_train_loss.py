#!/usr/bin/env python3
"""Summarise a DreamDojo training loss curve and test whether it has flattened.

Per-iteration diffusion loss is dominated by the randomly sampled timestep, so
the raw trace is far too noisy to eyeball a plateau. This bins the trace, then
compares the slope of the tail against the run's own noise level: if the total
drift predicted over the tail window is small next to the scatter of the bin
means, the curve is flat to within what this logging can resolve.

    python3 scripts/analyze_train_loss.py <console.log> [<console.log> ...]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# Two callbacks log loss. on_training_step_end fires only for the first 100
# iterations ("Iteration 7: Hit counter: 7/100 | Loss: ..."); every_n_impl
# carries the rest of the run at logging_iter granularity ("1500 : iter_speed
# 5.4 seconds per iteration | Loss: ..."). Only the latter covers all 3000
# iterations, so prefer it and fall back to the former for very short runs.
EVERY_N_RE = re.compile(r"\]\s*(\d+)\s*:\s*iter_speed\b.*?Loss:\s*([0-9.]+)")
STEP_END_RE = re.compile(r"Iteration (\d+):.*?Loss:\s*([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--bin-size", type=int, default=250)
    parser.add_argument(
        "--tail",
        type=int,
        default=1000,
        help="Iterations at the end to test for a plateau.",
    )
    return parser.parse_args()


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text(errors="ignore")
    for pattern in (EVERY_N_RE, STEP_END_RE):
        points = [
            (int(m.group(1)), float(m.group(2)))
            for line in text.splitlines()
            if (m := pattern.search(line))
        ]
        if points:
            points.sort()
            iters, losses = zip(*points)
            return np.asarray(iters), np.asarray(losses)
    raise SystemExit(f"No loss lines found in {path}")


def binned(iters: np.ndarray, losses: np.ndarray, size: int):
    edges = np.arange(0, iters.max() + size, size)
    centers, means, sems = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (iters > lo) & (iters <= hi)
        if mask.sum() < 2:
            continue
        chunk = losses[mask]
        centers.append((lo + hi) / 2)
        means.append(chunk.mean())
        sems.append(chunk.std(ddof=1) / np.sqrt(mask.sum()))
    return np.asarray(centers), np.asarray(means), np.asarray(sems)


def main() -> None:
    args = parse_args()
    for path in args.logs:
        iters, losses = load(path)
        centers, means, sems = binned(iters, losses, args.bin_size)

        print(f"=== {path.parent.name} ===")
        print(f"  iterations logged: {len(iters)} (max {iters.max()})")
        print(f"  raw loss: mean {losses.mean():.4f}, std {losses.std():.4f}")
        print(f"\n  mean loss per {args.bin_size}-iteration bin:")
        for center, mean, sem in zip(centers, means, sems):
            bar = "#" * int(round(mean / max(means) * 40))
            print(f"    iter {int(center):>5}  {mean:.4f} +/- {sem:.4f}  {bar}")

        tail_mask = iters > iters.max() - args.tail
        tail_iters, tail_losses = iters[tail_mask], losses[tail_mask]
        slope, intercept = np.polyfit(tail_iters, tail_losses, 1)
        drift = slope * (tail_iters.max() - tail_iters.min())

        # Compare the fitted drift against the scatter of the tail's bin means:
        # a real trend should move the mean by more than the bins wander.
        tail_bins = means[centers > iters.max() - args.tail]
        scatter = tail_bins.std(ddof=1) if len(tail_bins) > 1 else float("nan")

        first, last = means[0], means[-1]
        level = tail_losses.mean()
        # Report drift relative to the loss level. Comparing it only against the
        # scatter can flag a 1% trend as "still trending" purely because that
        # run's tail happened to be quiet, which says nothing useful.
        drift_pct = 100 * drift / level
        print(f"\n  first bin {first:.4f} -> last bin {last:.4f} "
              f"({100 * (last - first) / first:+.1f}%)")
        print(f"  last {args.tail} iters: slope {slope:+.2e}/iter, "
              f"drift {drift:+.4f} ({drift_pct:+.1f}% of {level:.4f})")
        print(f"  scatter of tail bin means: {scatter:.4f} "
              f"({100 * scatter / level:.1f}%)")
        if abs(drift_pct) < 2.0:
            print(f"  => tail moves {abs(drift_pct):.1f}% over {args.tail} "
                  "iters: plateaued")
        elif abs(drift) < scatter:
            print("  => drift is real-sized but under the bin-to-bin scatter: "
                  "inconclusive, needs more logging")
        else:
            print(f"  => tail still moving {drift_pct:+.1f}%: not yet flat")
        print()


if __name__ == "__main__":
    main()
