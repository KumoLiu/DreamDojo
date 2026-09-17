#!/usr/bin/env python3
"""Validate pick-trocar dynamics with real actions and counterfactual controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapy
import numpy as np
import torch

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from groot_dreams.dataloader import MultiVideoActionDataset, VideoActionDataset
from scripts.inference_utils import (
    CHUNK_SIZE,
    CONFIG_FILE,
    add_label,
    generate_chunk,
)
from scripts.video_metrics import evaluate


ROOT = Path("/localhome/local-yunl/DreamDojo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default="datasets/g1_hf_pick_trocar_teleop_success_val",
    )
    parser.add_argument("--episode-indices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--num-chunks", type=int, default=3)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent
        / (
            "models/dreamdojo/lora_r32_scratch_lr3e-4_18k/"
            "checkpoints/iter_000018000/"
            "model_ema_bf16.pt"
        ),
    )
    parser.add_argument(
        "--experiment",
        default="dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/pick_trocar_world_model_validation",
    )
    parser.add_argument("--save-fps", type=int, default=15)
    parser.add_argument("--num-inference-steps", type=int, default=35)
    parser.add_argument(
        "--run-zero-action",
        action="store_true",
        help="Also generate and display a zero-action closed-loop rollout.",
    )
    parser.add_argument(
        "--perceptual-metrics",
        action="store_true",
        help="Also compute SSIM and LPIPS (needs torchmetrics weights).",
    )
    return parser.parse_args()


def load_samples(dataset_path: str, num_frames: int):
    dataset = MultiVideoActionDataset(
        num_frames=num_frames,
        dataset_path=dataset_path,
        data_split="full",
        single_base_index=True,
        restrict_len=None,
        deterministic_uniform_sampling=False,
    )
    if not all(
        isinstance(subset, VideoActionDataset) for subset in dataset.datasets
    ):
        loaded_types = [type(subset).__name__ for subset in dataset.datasets]
        raise RuntimeError(
            "Pick-trocar evaluation requires VideoActionDataset, "
            f"but loaded {loaded_types} from {dataset_path!r}. "
            "Ensure the container dataset path contains the embodiment name."
        )
    return dataset


def generate_rollout(
    pipeline: Video2WorldInference,
    gt: np.ndarray,
    actions: torch.Tensor,
    lam_video: torch.Tensor,
    *,
    seed: int,
    num_inference_steps: int,
    teacher_forcing: bool,
    zero_actions: bool,
) -> np.ndarray:
    output = [gt[0]]
    current = gt[0]
    for chunk_index, start in enumerate(range(0, len(actions), CHUNK_SIZE)):
        if teacher_forcing:
            current = gt[start]
        action_chunk = actions[start : start + CHUNK_SIZE]
        if zero_actions:
            action_chunk = torch.zeros_like(action_chunk)
        generated = generate_chunk(
            pipeline,
            current,
            action_chunk,
            lam_video[start * 2 : (start + CHUNK_SIZE) * 2],
            seed=seed + chunk_index,
            num_inference_steps=num_inference_steps,
        )[-CHUNK_SIZE:]
        output.extend(generated)
        current = generated[-1]
    return np.stack(output)


def image_metrics(
    reference: np.ndarray, prediction: np.ndarray, *, perceptual: bool = False
) -> dict[str, float]:
    """Frame metrics plus the motion-weighted and per-step breakdown.

    Scored here on the raw arrays rather than recovered from the comparison
    video, so these carry no compression bias. SSIM and LPIPS are off by
    default: they need torchmetrics weights, and the diffusion pipeline is
    already resident on the GPU at this point.
    """
    return evaluate(reference, prediction, device="cpu", with_perceptual=perceptual)


def main() -> None:
    args = parse_args()
    if args.num_chunks <= 0 or args.num_seeds <= 0:
        raise ValueError("--num-chunks and --num-seeds must be positive")
    num_frames = 1 + CHUNK_SIZE * args.num_chunks
    dataset = load_samples(args.dataset_path, num_frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Video2WorldInference(
        experiment_name=args.experiment,
        ckpt_path=str(args.checkpoint.expanduser().resolve()),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=CONFIG_FILE,
    )
    results = []
    try:
        for episode_index in args.episode_indices:
            sample = dataset[episode_index]
            gt = sample["video"].permute(1, 2, 3, 0).cpu().numpy()
            actions = sample["action"][: num_frames - 1]
            lam_video = sample["lam_video"][: (num_frames - 1) * 2]
            for seed_index in range(args.num_seeds):
                base_seed = episode_index * 1000 + seed_index * 100
                teacher = generate_rollout(
                    pipeline,
                    gt,
                    actions,
                    lam_video,
                    seed=base_seed,
                    num_inference_steps=args.num_inference_steps,
                    teacher_forcing=True,
                    zero_actions=False,
                )
                closed = generate_rollout(
                    pipeline,
                    gt,
                    actions,
                    lam_video,
                    seed=base_seed,
                    num_inference_steps=args.num_inference_steps,
                    teacher_forcing=False,
                    zero_actions=False,
                )
                comparison_columns = [
                    add_label(gt, "REAL GT"),
                    add_label(teacher, "GT ACTION | TEACHER FORCED"),
                    add_label(closed, "GT ACTION | CLOSED LOOP"),
                ]
                zero = None
                if args.run_zero_action:
                    zero = generate_rollout(
                        pipeline,
                        gt,
                        actions,
                        lam_video,
                        seed=base_seed,
                        num_inference_steps=args.num_inference_steps,
                        teacher_forcing=False,
                        zero_actions=True,
                    )
                    comparison_columns.append(
                        add_label(zero, "ZERO ACTION | CLOSED LOOP")
                    )
                comparison = np.concatenate(comparison_columns, axis=2)
                prefix = f"episode_{episode_index:03d}_seed_{seed_index}"
                mediapy.write_video(
                    str(args.output_dir / f"{prefix}_comparison.mp4"),
                    comparison,
                    fps=args.save_fps,
                )
                record = {
                    "episode_index": episode_index,
                    "seed_index": seed_index,
                    "teacher_forced": image_metrics(
                        gt, teacher, perceptual=args.perceptual_metrics
                    ),
                    "closed_loop": image_metrics(
                        gt, closed, perceptual=args.perceptual_metrics
                    ),
                }
                if zero is not None:
                    record["action_sensitivity_mae"] = float(
                        np.mean(
                            np.abs(
                                closed[1:].astype(np.float32)
                                - zero[1:].astype(np.float32)
                            )
                        )
                    )
                results.append(record)
                # One line per rollout: the record now carries per-frame arrays,
                # and dumping those turns the shard logs into thousands of
                # numbers per episode.
                for rollout in ("teacher_forced", "closed_loop"):
                    metrics = record[rollout]
                    print(
                        f"episode {episode_index:03d} seed {seed_index} {rollout}: "
                        f"psnr {metrics['psnr']:.2f} | "
                        f"moving-region psnr {metrics['moving_region_psnr']:.2f} | "
                        f"mse slope {metrics['mse_slope_per_frame']:+.3f}/frame | "
                        f"psnr drop {metrics['psnr_drop_db']:+.2f} dB"
                    )
    finally:
        pipeline.cleanup()

    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(f"Validation outputs: {args.output_dir}")


if __name__ == "__main__":
    torch.enable_grad(False)
    main()
