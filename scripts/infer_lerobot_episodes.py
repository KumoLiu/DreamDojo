#!/usr/bin/env python3
"""Infer one or more action-conditioned LeRobot episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import mediapy
import numpy as np
import torch
from loguru import logger

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from scripts.inference_utils import (
    CONFIG_FILE,
    add_label,
    ego_view_source_key,
    episode_lengths,
    generate_closed_loop,
    largest_chunk_aligned_frames,
    load_lerobot_episode,
)


ROOT = Path("/localhome/local-yunl/DreamDojo")
SAVE_FPS = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="datasets/g1_pick_tube_300")
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="+",
        default=[297],
        help="One or more episode indices to infer",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/dreamdojo/pick_trocar/g1_pick_trocar_2b/checkpoints/iter_000002000/model_ema_bf16.pt",
    )
    parser.add_argument("--experiment", default="dreamdojo_2b_480_640_g1_pick_trocar")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/pick_trocar_2b_iter2000_ep297_pick_tube",
    )
    parser.add_argument("--pred-label", default="PRED | pick_trocar 2B @2k")
    args = parser.parse_args()

    torch.enable_grad(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lengths = episode_lengths(args.dataset_path)
    camera_key = ego_view_source_key(args.dataset_path)
    camera_label = camera_key.rsplit(".", 1)[-1]

    pipeline = Video2WorldInference(
        experiment_name=args.experiment,
        ckpt_path=str(args.checkpoint),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=CONFIG_FILE,
    )
    try:
        for episode_index in args.episode_indices:
            if episode_index not in lengths:
                raise KeyError(f"Episode {episode_index} missing from {args.dataset_path}")
            raw_len = lengths[episode_index]
            num_frames = largest_chunk_aligned_frames(0, raw_len)
            logger.info(
                f"episode={episode_index} raw_len={raw_len} num_frames={num_frames} "
                f"(~{num_frames / SAVE_FPS:.1f}s @ {SAVE_FPS}fps)"
            )
            data = load_lerobot_episode(
                args.dataset_path, episode_index, num_frames
            )
            gt = data["video"].permute(1, 2, 3, 0).cpu().numpy()
            pred = generate_closed_loop(pipeline, data, num_frames)

            prefix = f"episode_{episode_index:06d}"
            mediapy.write_video(str(args.output_dir / f"{prefix}_gt.mp4"), gt, fps=SAVE_FPS)
            mediapy.write_video(str(args.output_dir / f"{prefix}_pred.mp4"), pred, fps=SAVE_FPS)
            merged = np.concatenate(
                [
                    add_label(
                        gt,
                        f"GT | ep{episode_index} {camera_label}",
                        (255, 255, 255),
                    ),
                    add_label(pred, args.pred_label, (80, 255, 80)),
                ],
                axis=2,
            )
            out = args.output_dir / f"{prefix}_gt_pred_merged.mp4"
            mediapy.write_video(str(out), merged, fps=SAVE_FPS)
            print(f"DONE: {out}")
    finally:
        pipeline.cleanup()
        del pipeline
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
