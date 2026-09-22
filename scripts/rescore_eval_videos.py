#!/usr/bin/env python3
"""Score finished eval runs with the motion/drift/perceptual metrics.

The per-episode outputs keep only the comparison video, not the raw predicted
frames, so this recovers the panels from that video: it is a horizontal concat
of [real GT | teacher forced | closed loop | (zero action)], each panel 640 wide
with a 48-row label bar across the top that has to be cropped before scoring.

That means the frames have been through H.264, so the absolute numbers here are
not identical to what the evaluator would have written from the raw arrays. The
compression is identical for every run, so comparisons between runs hold; the
printed `psnr vs metrics.json` column quantifies the offset so it stays visible
rather than being quietly assumed away.

    python3 scripts/rescore_eval_videos.py outputs/eval/<run>
    python3 scripts/rescore_eval_videos.py outputs/eval/<lora> outputs/eval/<full>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapy
import numpy as np

from scripts.video_metrics import evaluate

PANEL_WIDTH = 640
# add_label() paints a translucent bar over the top of every panel; scoring it
# would add error that has nothing to do with the model.
LABEL_HEIGHT = 48
PANEL_NAMES = ["ground_truth", "teacher_forced", "closed_loop", "zero_action"]
DATASETS = ["teleop_success", "rollouts_30k", "rollouts_10k"]
HEADLINE = [
    "psnr",
    "moving_region_psnr",
    "motion_weighted_psnr",
    "ssim",
    "lpips",
    # Absolute drift, which tracks overall accuracy too.
    "mse_slope_per_frame",
    # Relative drift, which is what isolates compounding error.
    "mse_late_over_early",
    "psnr_drop_db",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | cuda | cuda:N. Only affects SSIM/LPIPS.",
    )
    parser.add_argument(
        "--no-perceptual",
        action="store_true",
        help="Skip SSIM and LPIPS (much faster).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rescore episodes that already have metrics_extended.json.",
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            torch.zeros(8, device="cuda")
            return "cuda"
    except Exception:
        pass
    return "cpu"


def split_panels(video: np.ndarray) -> dict[str, np.ndarray]:
    height, width = video.shape[1:3]
    if width % PANEL_WIDTH:
        raise ValueError(f"Unexpected comparison width {width}, not a multiple of {PANEL_WIDTH}")
    count = width // PANEL_WIDTH
    if count > len(PANEL_NAMES):
        raise ValueError(f"{count} panels exceeds the known layout {PANEL_NAMES}")
    cropped = video[:, LABEL_HEIGHT:height, :, :]
    return {
        PANEL_NAMES[i]: cropped[:, :, i * PANEL_WIDTH : (i + 1) * PANEL_WIDTH, :]
        for i in range(count)
    }


def score_episode(video_path: Path, device: str, perceptual: bool) -> dict:
    panels = split_panels(np.asarray(mediapy.read_video(str(video_path))))
    gt = panels["ground_truth"]
    scored = {}
    for name in ("teacher_forced", "closed_loop", "zero_action"):
        if name in panels:
            scored[name] = evaluate(
                gt, panels[name], device=device, with_perceptual=perceptual
            )
    return scored


def original_psnr(episode_dir: Path, episode_index: int) -> dict[str, float]:
    path = episode_dir / "metrics.json"
    if not path.exists():
        return {}
    for record in json.loads(path.read_text()):
        if int(record["episode_index"]) == episode_index:
            return {
                "teacher_forced": record["teacher_forced"]["psnr"],
                "closed_loop": record["closed_loop"]["psnr"],
            }
    return {}


def process(eval_dir: Path, args: argparse.Namespace, device: str) -> list[dict]:
    rows = []
    for dataset in args.datasets:
        for episode_dir in sorted((eval_dir / dataset).glob("episode_*")):
            for video in sorted(episode_dir.glob("*_comparison.mp4")):
                episode_index = int(video.name.split("_")[1])
                out_path = episode_dir / "metrics_extended.json"
                if out_path.exists() and not args.overwrite:
                    scored = json.loads(out_path.read_text())
                else:
                    scored = score_episode(video, device, not args.no_perceptual)
                    out_path.write_text(json.dumps(scored, indent=2) + "\n")
                reference = original_psnr(episode_dir, episode_index)
                for rollout, metrics in scored.items():
                    rows.append(
                        {
                            "dataset": dataset,
                            "episode": episode_index,
                            "rollout": rollout,
                            "original_psnr": reference.get(rollout),
                            **metrics,
                        }
                    )
                print(
                    f"  {dataset}/episode_{episode_index:03d}: "
                    + ", ".join(
                        f"{r} PSNR {m['psnr']:.2f}" for r, m in scored.items()
                    )
                )
    return rows


def summarise(rows: list[dict], rollout: str) -> dict[str, float]:
    subset = [r for r in rows if r["rollout"] == rollout]
    if not subset:
        return {}
    out = {"episodes": len(subset)}
    for key in HEADLINE:
        values = [r[key] for r in subset if key in r and r[key] is not None]
        values = [v for v in values if not np.isnan(v)]
        if values:
            out[key] = float(np.mean(values))
    offsets = [
        r["psnr"] - r["original_psnr"]
        for r in subset
        if r.get("original_psnr") is not None
    ]
    if offsets:
        out["psnr_vs_metrics_json"] = float(np.mean(offsets))
    return out


def short_name(name: str) -> str:
    """Eval directory names are long and mostly shared; keep the variant part."""
    for token in ("lora", "full"):
        if f"_{token}_" in name:
            return token
    return name[:14]


def print_table(title: str, summaries: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {title} ===")
    keys = [
        k
        for k in HEADLINE + ["psnr_vs_metrics_json"]
        if any(k in s for s in summaries.values())
    ]
    width = max(len(k) for k in keys) + 2
    print(f"{'metric':<{width}}" + "".join(f"{short_name(n):>14}" for n in summaries))
    print(
        f"{'episodes':<{width}}"
        + "".join(f"{s.get('episodes', 0):>14}" for s in summaries.values())
    )
    for key in keys:
        line = f"{key:<{width}}"
        for summary in summaries.values():
            value = summary.get(key)
            line += f"{value:>14.4f}" if value is not None else f"{'-':>14}"
        print(line)


def print_paired(title: str, left: list[dict], right: list[dict], names: tuple[str, str]) -> None:
    """Paired per-episode comparison.

    Both runs score the same validation episodes, so the difference should be
    taken per episode and not between the two means: episode-to-episode spread
    is far larger than the gap between runs, and comparing means throws away
    the pairing that cancels it out.
    """
    index = {(r["dataset"], r["episode"]): r for r in right}
    pairs = [(l, index[(l["dataset"], l["episode"])]) for l in left if (l["dataset"], l["episode"]) in index]
    if not pairs:
        return

    print(f"\n=== {title}: paired {names[0]} - {names[1]} over {len(pairs)} episodes ===")
    print(f"{'metric':<24}{'mean diff':>12}{'std err':>10}{'t':>8}{f'{names[0]} wins':>12}")
    for key in HEADLINE:
        deltas = np.array(
            [
                a[key] - b[key]
                for a, b in pairs
                if key in a and key in b and not (np.isnan(a[key]) or np.isnan(b[key]))
            ]
        )
        if len(deltas) < 2:
            continue
        sem = float(deltas.std(ddof=1) / np.sqrt(len(deltas)))
        mean = float(deltas.mean())
        # Lower is better for these, so a "win" is a negative difference.
        lower_better = key in {"lpips", "mse_slope_per_frame", "psnr_drop_db"}
        wins = int((deltas < 0).sum() if lower_better else (deltas > 0).sum())
        t = mean / sem if sem > 0 else float("nan")
        print(
            f"{key:<24}{mean:>+12.4f}{sem:>10.4f}{t:>8.2f}"
            f"{f'{wins}/{len(deltas)}':>12}"
        )
    print(
        "  |t| > 2 is the rough threshold for a difference that is not just "
        "episode noise."
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device for SSIM/LPIPS: {device}")

    per_dir = {}
    for eval_dir in args.eval_dirs:
        if not eval_dir.is_dir():
            raise SystemExit(f"Not a directory: {eval_dir}")
        print(f"\n### {eval_dir.name}")
        rows = process(eval_dir, args, device)
        if not rows:
            print("  no comparison videos found")
            continue
        per_dir[eval_dir.name] = rows
        (eval_dir / "metrics_extended_summary.json").write_text(
            json.dumps(
                {
                    rollout: summarise(rows, rollout)
                    for rollout in ("teacher_forced", "closed_loop", "zero_action")
                },
                indent=2,
            )
            + "\n"
        )

    if not per_dir:
        raise SystemExit("Nothing scored")

    names = list(per_dir)
    for rollout in ("teacher_forced", "closed_loop"):
        summaries = {
            name: summarise(rows, rollout)
            for name, rows in per_dir.items()
            if summarise(rows, rollout)
        }
        if summaries:
            print_table(rollout, summaries)
        if len(names) == 2:
            print_paired(
                rollout,
                [r for r in per_dir[names[0]] if r["rollout"] == rollout],
                [r for r in per_dir[names[1]] if r["rollout"] == rollout],
                (short_name(names[0]), short_name(names[1])),
            )

    print(
        "\nHigher is better: psnr, moving_region_psnr, motion_weighted_psnr, ssim."
        "\nLower is better: lpips, mse_slope_per_frame, psnr_drop_db."
        "\npsnr_vs_metrics_json is the H.264 offset against the original run."
    )


if __name__ == "__main__":
    main()
