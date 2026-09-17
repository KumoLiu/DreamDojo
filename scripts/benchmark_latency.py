#!/usr/bin/env python3
"""Measure world-model generation latency per action chunk.

The evaluation harness reports quality, never speed, and its wall clock mixes in
dataset decoding, metric computation and video writing across 8 shards, so a
per-chunk number cannot be read off it better than to a factor of two. This
times the one call that matters, `generate_chunk`, on real actions from a
validation episode.

One chunk is CHUNK_SIZE new frames conditioned on a single frame, which at the
15 fps the model is trained and evaluated at is CHUNK_SIZE/15 seconds of video.
The ratio of latency to that is the number to quote for closed-loop use, since a
closed-loop rollout cannot start chunk k+1 until chunk k is finished.

    .venv/bin/python scripts/benchmark_latency.py --steps 35 20 10 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from scripts.inference_utils import CHUNK_SIZE, CONFIG_FILE, generate_chunk

ROOT = Path("/localhome/local-yunl/DreamDojo")
FPS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent
        / "models/dreamdojo/lora_r32_scratch_lr3e-4_18k/checkpoints/iter_000018000/model_ema_bf16.pt",
        help="Any checkpoint of the same architecture: latency is set by the "
        "network and the LoRA rank, not by the values of the weights.",
    )
    parser.add_argument(
        "--experiment",
        default="dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora",
    )
    parser.add_argument(
        "--dataset-path",
        default="datasets/g1_hf_pick_trocar_teleop_success_val",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--steps", nargs="+", type=int, default=[35, 20, 10, 5])
    parser.add_argument("--reps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from scripts.validate_pick_trocar_world_model import load_samples

    num_frames = 1 + CHUNK_SIZE * 2
    sample = load_samples(args.dataset_path, num_frames)[args.episode_index]
    condition = sample["video"].transpose(0, 1)[:1]
    actions = sample["action"][:CHUNK_SIZE]
    lam_video = sample["lam_video"][: CHUNK_SIZE * 2]

    pipeline = Video2WorldInference(
        experiment_name=args.experiment,
        ckpt_path=str(args.checkpoint.expanduser().resolve()),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=CONFIG_FILE,
    )

    print(f"\ndevice: {torch.cuda.get_device_name(0)}")
    print(
        f"chunk:  {CHUNK_SIZE} frames = {CHUNK_SIZE / FPS:.2f} s of video "
        f"at {FPS} fps\n"
    )
    print(
        f"{'steps':>6} {'latency':>12} {'per step':>10} {'per frame':>10} "
        f"{'vs realtime':>12}"
    )

    for steps in args.steps:
        # The first call at a new step count pays for allocator growth and any
        # lazily built sampler state, which is not part of steady-state latency.
        generate_chunk(
            pipeline, condition, actions, lam_video, seed=0, num_inference_steps=steps
        )
        torch.cuda.reset_peak_memory_stats()
        times = []
        for rep in range(args.reps):
            start = time.perf_counter()
            generate_chunk(
                pipeline,
                condition,
                actions,
                lam_video,
                seed=rep + 1,
                num_inference_steps=steps,
            )
            times.append(time.perf_counter() - start)
        mean = statistics.mean(times)
        spread = statistics.stdev(times) if len(times) > 1 else 0.0
        print(
            f"{steps:>6} {mean:>8.2f} s "
            f"{'+/- %.2f' % spread if spread else '':>3} "
            f"{mean / steps * 1000:>7.0f} ms "
            f"{mean / CHUNK_SIZE * 1000:>7.0f} ms "
            f"{mean / (CHUNK_SIZE / FPS):>9.1f}x"
        )

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\npeak GPU memory during timing: {peak:.1f} GB")


if __name__ == "__main__":
    main()
