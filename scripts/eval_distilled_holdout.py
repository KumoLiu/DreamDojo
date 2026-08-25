#!/usr/bin/env python3
"""Evaluate a causal distilled model on held-out pick-trocar episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapy
import numpy as np
import torch

from cosmos_predict2._src.predict2.interactive.inference.action_video2world import (
    ActionStreamingInference,
)
from groot_dreams.dataloader import MultiVideoActionDataset
from scripts.inference_utils import add_label


ROOT = Path("/localhome/local-yunl/DreamDojo")
CONFIG_FILE = (
    "cosmos_predict2/_src/predict2/interactive/configs/config_distill.py"
)
EXPERIMENT = "pick_trocar_iter2500_self_forcing_no_s3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default="datasets/g1_pick_trocar_rollout_all_260_headcam_eval_5s5f",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-indices", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-fps", type=int, default=15)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument(
        "--net-max-frames",
        type=int,
        default=None,
        help="Override causal net max_frames (Warmup checkpoints use 128).",
    )
    return parser.parse_args()


def image_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    frame_count = min(len(reference), len(prediction))
    difference = (
        reference[1:frame_count].astype(np.float32)
        - prediction[1:frame_count].astype(np.float32)
    )
    mse = float(np.mean(np.square(difference)))
    return {
        "mse": mse,
        "psnr": float(20 * np.log10(255.0 / np.sqrt(max(mse, 1e-12)))),
        "mae": float(np.mean(np.abs(difference))),
    }


def load_dataset(path: str, num_frames: int) -> MultiVideoActionDataset:
    return MultiVideoActionDataset(
        num_frames=num_frames,
        dataset_path=path,
        data_split="full",
        single_base_index=True,
        restrict_len=None,
        deterministic_uniform_sampling=False,
    )


def generated_to_uint8(generated: torch.Tensor) -> np.ndarray:
    return (
        (generated[0].clamp(-1, 1) + 1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )


def main() -> None:
    args = parse_args()
    if args.num_frames < 13 or (args.num_frames - 1) % 12:
        raise ValueError("--num-frames must equal 1 + 12 * num_chunks")

    dataset = load_dataset(args.dataset_path, args.num_frames)
    if len(dataset) != 10:
        raise ValueError(f"Expected 10 held-out episodes, found {len(dataset)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.expanduser().resolve()
    inference = ActionStreamingInference(
        config_path=CONFIG_FILE,
        experiment_name=args.experiment,
        ckpt_path=str(checkpoint),
        s3_credential_path="",
        cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
        context_parallel_size=1,
        enable_fsdp=False,
        torch_compile=False,
        experiment_opts=(
            [f"model.config.net.max_frames={args.net_max_frames}"]
            if args.net_max_frames is not None
            else None
        ),
    )

    records: list[dict] = []
    try:
        for episode_index in args.episode_indices:
            sample = dataset[episode_index]
            gt = sample["video"].permute(1, 2, 3, 0).cpu().numpy()
            actions = sample["action"][: args.num_frames - 1].cpu().numpy()
            generated = inference.generate_action_streaming(
                video_path=gt,
                actions_np=actions,
                resolution_hw=(gt.shape[1], gt.shape[2]),
                num_steps=args.num_steps,
                seed=args.seed + episode_index * 1000,
                start_frame_idx=0,
                max_frames=args.num_frames,
            )
            prediction = generated_to_uint8(generated)
            frame_count = min(len(gt), len(prediction))
            gt_aligned = gt[:frame_count]
            prediction = prediction[:frame_count]

            prefix = f"episode_{episode_index:03d}"
            mediapy.write_video(
                str(args.output_dir / f"{prefix}_prediction.mp4"),
                prediction,
                fps=args.save_fps,
            )
            comparison = np.concatenate(
                [
                    add_label(gt_aligned, "REAL GT"),
                    add_label(prediction, "CAUSAL CLOSED LOOP"),
                ],
                axis=2,
            )
            mediapy.write_video(
                str(args.output_dir / f"{prefix}_comparison.mp4"),
                comparison,
                fps=args.save_fps,
            )
            record = {
                "episode_index": episode_index,
                "num_reference_frames": int(len(gt)),
                "num_prediction_frames": int(len(prediction)),
                "closed_loop": image_metrics(gt_aligned, prediction),
            }
            records.append(record)
            print(json.dumps(record, indent=2), flush=True)
    finally:
        inference.cleanup()

    (args.output_dir / "metrics.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(f"Evaluation outputs: {args.output_dir}")


if __name__ == "__main__":
    torch.enable_grad(False)
    main()
