#!/usr/bin/env python3
"""Plot several DreamDojo training loss curves on shared axes.

Every run here was chained across 4-hour Slurm jobs, so its trace is split over
several stdout files and `debug.log` in the run directory holds only the last
segment. This stitches the segments back together by iteration number, keeping
the later occurrence when a job re-ran iterations it had already logged after
resuming from a checkpoint.

Per-iteration diffusion loss is dominated by the randomly sampled timestep, so
the raw trace is unreadable; each curve is a rolling median over a window given
in iterations, which is robust to the occasional very high-timestep step.

    python3 scripts/plot_train_loss.py --logs-dir /tmp/cluster_logs2 \
        --panel 'Learning rate|1e-4=dd_lora_r32_lr1e-4_long9k,2e-4=dd_lora_r32_scratch_lr2e-4_18k' \
        --out /tmp/loss.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Two callbacks log loss. on_training_step_end covers only the first 100
# iterations after a (re)start; every_n_impl carries the rest at logging_iter
# granularity. Both are read so the segment boundaries stay continuous.
PATTERNS = (
    re.compile(r"\]\s*(\d+)\s*:\s*iter_speed\b.*?Loss:\s*([0-9.]+)"),
    re.compile(r"Iteration (\d+):.*?Loss:\s*([0-9.]+)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument(
        "--panel",
        action="append",
        required=True,
        metavar="'Title|label=run[@batch],...'",
        help="Repeat for each subplot. @batch sets samples per step for --x samples.",
    )
    parser.add_argument("--extra-log", action="append", default=[],
                        help="run=path for a trace not in --logs-dir.")
    parser.add_argument("--x", choices=("iters", "samples"), default="iters")
    parser.add_argument("--window", type=int, default=1000,
                        help="Rolling median width, in iterations.")
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None,
                        help="The runs separate by ~10%, so the opening drop "
                             "from 0.08 has to be clipped to see anything.")
    parser.add_argument("--note", default=None,
                        help="Caption under the figure, for footnoted labels.")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Stitch log segments into one iteration-ordered trace."""
    by_iter: dict[int, float] = {}
    for path in sorted(paths):
        text = path.read_text(errors="ignore")
        for line in text.splitlines():
            for pattern in PATTERNS:
                if m := pattern.search(line):
                    by_iter[int(m.group(1))] = float(m.group(2))
                    break
    if not by_iter:
        raise SystemExit(f"No loss lines in {[str(p) for p in paths]}")
    iters = np.asarray(sorted(by_iter))
    return iters, np.asarray([by_iter[i] for i in iters])


def rolling_median(iters: np.ndarray, losses: np.ndarray, window: int):
    """Median over a window measured in iterations, not in samples.

    Runs log at different densities, so a fixed number of points would smooth
    them by different amounts and make the curves look artificially different.
    """
    out = np.empty_like(losses)
    for i, it in enumerate(iters):
        mask = (iters > it - window / 2) & (iters <= it + window / 2)
        out[i] = np.median(losses[mask])
    return out


def parse_panel(spec: str) -> tuple[str, str, list[tuple[str, str, int]]]:
    """Parse 'Title[|x=samples]|label=run[@batch],...'.

    The x axis is per panel because comparing batch sizes needs both views: at
    equal iterations the larger batch has seen more data, and at equal data it
    has taken fewer steps.
    """
    title, *middle, series = spec.split("|")
    axis = ""
    for chunk in middle:
        if chunk.startswith("x="):
            axis = chunk[2:]
    parsed = []
    for item in series.split(","):
        label, _, run = item.partition("=")
        run, _, batch = run.partition("@")
        parsed.append((label.strip(), run.strip(), int(batch) if batch else 32))
    return title.strip(), axis, parsed


def main() -> None:
    args = parse_args()
    extra = dict(item.split("=", 1) for item in args.extra_log)
    panels = [parse_panel(spec) for spec in args.panel]

    fig, axes = plt.subplots(
        1, len(panels), figsize=(6.2 * len(panels), 4.8), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, (title, axis, series) in zip(axes, panels):
        axis = axis or args.x
        for label, run, batch in series:
            paths = sorted(args.logs_dir.glob(f"{run}-*.out"))
            if run in extra:
                paths += [Path(extra[run])]
            if not paths:
                raise SystemExit(f"No logs for {run} in {args.logs_dir}")
            iters, losses = load(paths)
            smooth = rolling_median(iters, losses, args.window)
            x = iters * batch if axis == "samples" else iters
            (line,) = ax.plot(x, smooth, lw=1.8, label=label)
            ax.scatter(x, losses, s=1.5, alpha=0.10, color=line.get_color(),
                       linewidths=0)
            print(f"{label:22s} {run:34s} "
                  f"iters {iters.min()}-{iters.max()}, "
                  f"final smoothed {smooth[-1]:.4f}")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("samples consumed" if axis == "samples" else "iteration")
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=9, framealpha=0.9)
        if args.ymax or args.ymin:
            ax.set_ylim(bottom=args.ymin, top=args.ymax)

    axes[0].set_ylabel(f"training loss (rolling median, {args.window} iters)")
    fig.tight_layout()
    if args.note:
        fig.subplots_adjust(bottom=0.20)
        fig.text(0.01, 0.02, args.note, fontsize=8.5, va="bottom", wrap=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
