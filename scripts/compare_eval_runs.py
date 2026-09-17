#!/usr/bin/env python3
"""Compare evaluation runs episode by episode against a baseline run.

Aggregate means hide whether a gap is real: with 55 episodes whose closed-loop
MSE spans an order of magnitude, a few percent difference in the mean can come
entirely from one episode landing differently. Every run here scored the same
episode list, so the comparison can be paired, which removes between-episode
variance and is what makes a small mean difference interpretable.

    python3 scripts/compare_eval_runs.py <baseline_dir> <other_dir> [...] \
        --metrics closed_loop.mse closed_loop.mse_late_over_early
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DATASETS = ("teleop_success", "rollouts_30k", "rollouts_10k")

# mse_late_over_early is reported with both of its components, never alone. A
# model that starts worse inflates the denominator and so scores a flattering
# ratio while being worse at every point in the rollout: clip25 and lr1e-5 both
# "improved" the ratio by 10-22% here while mse_early and mse_late were each
# 13-111% worse. The ratio only means less compounding when the two parts move
# together, so they have to be on screen to read it.
DEFAULT_METRICS = (
    "closed_loop.mse",
    "closed_loop.psnr",
    "closed_loop.moving_region_mse",
    "closed_loop.motion_weighted_psnr",
    "closed_loop.mse_early",
    "closed_loop.mse_late",
    "closed_loop.mse_late_over_early",
    "closed_loop.mse_slope_per_frame",
)

# Metrics where a larger number is the better result.
# lpips is a distance, so lower is better even though "ssim" next to it is not.
HIGHER_IS_BETTER = ("psnr", "ssim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", type=Path, nargs="+", help="Baseline first.")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument(
        "--dataset",
        choices=(*DATASETS, "all"),
        default="all",
        help="Restrict to one validation set.",
    )
    parser.add_argument(
        "--filename",
        default="metrics.json",
        help="metrics_extended.json for runs rescored with perceptual metrics.",
    )
    return parser.parse_args()


def add_drift_of(record: dict, source: str, label: str) -> None:
    """Derive early/late/ratio for a per-frame series, in place.

    SSIM and LPIPS arrive only as a per-frame series and an episode mean, but a
    mean cannot show whether a perceptual failure builds up over the rollout,
    which is the thing morphing actually looks like. Split in quarters to match
    how mse_early and mse_late are defined so the two are read the same way.
    """
    for phase in ("teacher_forced", "closed_loop"):
        series = (record.get(phase) or {}).get(source)
        if not series:
            continue
        quarter = max(1, len(series) // 4)
        early = sum(series[:quarter]) / quarter
        late = sum(series[-quarter:]) / quarter
        record[phase][f"{label}_early"] = early
        record[phase][f"{label}_late"] = late
        if early != 0:
            record[phase][f"{label}_late_over_early"] = late / early


def load_run(run_dir: Path, dataset: str, filename: str) -> dict[str, dict]:
    """Map "<dataset>/<episode>" to that episode's metric dict."""
    wanted = DATASETS if dataset == "all" else (dataset,)
    episodes: dict[str, dict] = {}
    for name in wanted:
        for path in sorted((run_dir / name).glob(f"episode_*/{filename}")):
            payload = json.loads(path.read_text())
            # metrics.json holds one record per seed; metrics_extended.json,
            # written by rescore_eval_videos.py, holds the record directly.
            if isinstance(payload, list):
                if not payload:
                    continue
                payload = payload[0]
            add_drift_of(payload, "per_frame_lpips", "lpips")
            add_drift_of(payload, "per_frame_ssim", "ssim")
            episodes[f"{name}/{path.parent.name}"] = payload
    return episodes


def pluck(record: dict, dotted: str) -> float | None:
    node = record
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, (int, float)) else None


def wilcoxon_p(diffs: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value via a normal approximation.

    Closed-loop error is heavily right-skewed, so a paired t-test would be
    driven by the worst few episodes; the signed-rank test only uses the
    ordering of the absolute differences. The normal approximation is adequate
    at these sample sizes and avoids a SciPy dependency.
    """
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n < 6:
        return None

    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        shared = (i + j) / 2 + 1  # average rank, 1-based, for the tied group
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1

    w_plus = sum(r for d, r in zip(nonzero, ranks) if d > 0)
    mean = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sd == 0:
        return None
    z = (w_plus - mean) / sd
    return math.erfc(abs(z) / math.sqrt(2))


def main() -> None:
    args = parse_args()
    baseline_dir, *others = args.dirs

    baseline = load_run(baseline_dir, args.dataset, args.filename)
    print(f"baseline: {baseline_dir.name}  ({len(baseline)} episodes)")
    print(f"dataset:  {args.dataset}\n")

    for other_dir in others:
        other = load_run(other_dir, args.dataset, args.filename)
        shared = sorted(set(baseline) & set(other))
        print(f"=== {other_dir.name} vs baseline ({len(shared)} paired episodes) ===")
        if len(shared) != len(baseline) or len(shared) != len(other):
            print(
                f"  note: baseline has {len(baseline)}, this run has {len(other)};"
                f" comparing the {len(shared)} in both"
            )

        header = f"  {'metric':34s} {'baseline':>10s} {'this run':>10s} {'delta':>9s} {'wins':>7s} {'p':>8s}"
        print(header)
        for metric in args.metrics:
            pairs = [
                (pluck(baseline[k], metric), pluck(other[k], metric)) for k in shared
            ]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if not pairs:
                print(f"  {metric:34s} {'not present':>10s}")
                continue

            base_vals = [a for a, _ in pairs]
            this_vals = [b for _, b in pairs]
            diffs = [b - a for a, b in pairs]
            base_mean = sum(base_vals) / len(base_vals)
            this_mean = sum(this_vals) / len(this_vals)

            higher_better = any(k in metric for k in HIGHER_IS_BETTER)
            wins = sum(1 for d in diffs if (d > 0) == higher_better and d != 0)
            pct = 100 * (this_mean - base_mean) / base_mean if base_mean else float("nan")
            p = wilcoxon_p(diffs)
            p_text = "     n/a" if p is None else f"{p:8.4f}"
            better = "+" if (this_mean > base_mean) == higher_better else "-"
            print(
                f"  {metric:34s} {base_mean:10.3f} {this_mean:10.3f}"
                f" {pct:+8.1f}% {wins:3d}/{len(pairs):<3d} {p_text} {better}"
            )
        print()

    print("delta is this run minus baseline, as a percentage of baseline.")
    print("wins counts episodes where this run is better; p is paired Wilcoxon.")
    print("trailing + means this run is better on the mean, - means worse.")


if __name__ == "__main__":
    main()
