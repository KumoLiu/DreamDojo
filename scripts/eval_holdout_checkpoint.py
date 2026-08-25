#!/usr/bin/env python3
"""Evaluate one checkpoint on selected balanced rollout holdout cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapy
import numpy as np
import torch

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from scripts.inference_utils import CONFIG_FILE, add_label
from scripts.validate_pick_trocar_world_model import (
    generate_rollout,
    image_metrics,
    load_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("datasets/g1_pick_trocar_rollout_all_260_headcam_eval_5s5f"),
    )
    parser.add_argument("--episode-indices", nargs="+", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument(
        "--full-episode",
        action="store_true",
        help="Use every complete 12-frame action chunk available in each episode.",
    )
    parser.add_argument("--save-fps", type=int, default=15)
    parser.add_argument("--num-inference-steps", type=int, default=35)
    parser.add_argument(
        "--experiment",
        default="dreamdojo_2b_480_640_g1_pick_trocar_headcam_rollout_holdout_5s5f_ema_both",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = [
        json.loads(line)
        for line in (args.dataset_path / "meta/episodes.jsonl").read_text().splitlines()
    ]
    datasets = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Video2WorldInference(
        experiment_name=args.experiment,
        ckpt_path=str(args.checkpoint.resolve()),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=CONFIG_FILE,
    )
    results = []
    try:
        for position in args.episode_indices:
            metadata = episodes[position]
            original_id = int(metadata["episode_index"])
            outcome = "SUCCESS" if metadata["success"] else "FAIL"
            num_chunks = args.num_chunks
            if args.full_episode:
                num_chunks = (int(metadata["length"]) - 1) // 12
            if num_chunks <= 0:
                raise ValueError(f"Case {position} has no complete action chunks")
            num_frames = 1 + 12 * num_chunks
            if num_frames not in datasets:
                datasets[num_frames] = load_samples(
                    str(args.dataset_path),
                    num_frames,
                )
            dataset = datasets[num_frames]
            sample = dataset[position]
            gt = sample["video"].permute(1, 2, 3, 0).cpu().numpy()
            actions = sample["action"][: num_frames - 1]
            lam_video = sample["lam_video"][: (num_frames - 1) * 2]
            teacher = generate_rollout(
                pipeline,
                gt,
                actions,
                lam_video,
                seed=original_id * 1000,
                num_inference_steps=args.num_inference_steps,
                teacher_forcing=True,
                zero_actions=False,
            )
            closed = generate_rollout(
                pipeline,
                gt,
                actions,
                lam_video,
                seed=original_id * 1000,
                num_inference_steps=args.num_inference_steps,
                teacher_forcing=False,
                zero_actions=False,
            )

            prefix = f"case_{position:02d}_orig_{original_id:03d}"
            mediapy.write_video(
                str(args.output_dir / f"{prefix}_gt.mp4"),
                add_label(gt, f"CASE {position:02d} | ORIG {original_id:03d} | {outcome} | REAL GT"),
                fps=args.save_fps,
            )
            mediapy.write_video(
                str(args.output_dir / f"{prefix}_teacher_forced.mp4"),
                add_label(
                    teacher,
                    f"CASE {position:02d} | {outcome} | {args.checkpoint_label} | TEACHER FORCED",
                ),
                fps=args.save_fps,
            )
            mediapy.write_video(
                str(args.output_dir / f"{prefix}_closed_loop.mp4"),
                add_label(
                    closed,
                    f"CASE {position:02d} | {outcome} | {args.checkpoint_label} | CLOSED LOOP",
                ),
                fps=args.save_fps,
            )
            record = {
                "position": position,
                "original_episode_index": original_id,
                "success": bool(metadata["success"]),
                "checkpoint": args.checkpoint_label,
                "num_chunks": num_chunks,
                "num_frames": num_frames,
                "teacher_forced": image_metrics(gt, teacher),
                "closed_loop": image_metrics(gt, closed),
            }
            results.append(record)
            print(json.dumps(record))
    finally:
        pipeline.cleanup()

    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    torch.enable_grad(False)
    main()
