#!/usr/bin/env python3
"""Validate DreamDojo on labeled real success and failure rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapy
import numpy as np
import piq
import torch

from cosmos_predict2._src.predict2.inference.video2world import (
    Video2WorldInference,
)


ROOT = Path("/localhome/local-yunl/DreamDojo")
RLINF_ROOT = Path("/localhome/local-yunl/RLinf")
CONFIG_FILE = (
    "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
)
CHUNK_SIZE = 12
RAW_ACTION_HORIZON = 25
RAW_STEPS_PER_CHUNK = 24

sys.path.insert(0, str(RLINF_ROOT))
from rlinf.envs.world_model.dreamdojo_action import (  # noqa: E402
    G1DreamDojoActionBridge,
)
from rlinf.envs.world_model.success_classifier_reward import (  # noqa: E402
    SuccessClassifierReward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=35)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        default="dreamdojo_2b_480_640_g1_pick_trocar_headcam",
    )
    parser.add_argument(
        "--action-statistics",
        type=Path,
        default=ROOT / "shared_meta/G1_stats.json",
    )
    parser.add_argument(
        "--reward-checkpoint",
        type=Path,
        default=RLINF_ROOT
        / "success_classifier/dreamdojo_progress_run/checkpoints/best.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-fps", type=int, default=15)
    parser.add_argument(
        "--failure-extra-chunks",
        type=int,
        default=0,
        help=(
            "For failed episodes, extrapolate this many chunks beyond the "
            "recording while holding the final action."
        ),
    )
    parser.add_argument(
        "--run-zero-action",
        action="store_true",
        help="Also generate a closed-loop rollout with zero action conditioning.",
    )
    parser.add_argument(
        "--run-shuffled-action",
        action="store_true",
        help="Also generate a closed-loop rollout with temporally shuffled actions.",
    )
    return parser.parse_args()


def read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[:2] != (480, 640):
                frame = cv2.resize(frame, (640, 480))
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames)


def load_episode(path: Path) -> dict:
    metadata = json.loads((path / "meta.json").read_text())
    video = read_video(path / "color_0.mp4")
    actions = np.load(path / "actions.npy")
    length = min(len(video), len(actions), metadata["num_frames"])
    return {
        "metadata": metadata,
        "video": video[:length],
        "actions": actions[:length],
    }


def generate_chunk(
    pipeline: Video2WorldInference,
    condition: np.ndarray,
    action_condition: torch.Tensor,
    seed: int,
    num_steps: int,
) -> np.ndarray:
    image = torch.from_numpy(condition.copy()).permute(2, 0, 1).unsqueeze(0)
    input_video = torch.cat(
        [image, torch.zeros_like(image).repeat(CHUNK_SIZE, 1, 1, 1)]
    )
    generated = pipeline.generate_vid2world(
        prompt="",
        input_path=input_video.unsqueeze(0).permute(0, 2, 1, 3, 4),
        action=action_condition.float(),
        guidance=0,
        num_video_frames=CHUNK_SIZE + 1,
        num_latent_conditional_frames=1,
        resolution="480,640",
        seed=seed,
        num_steps=num_steps,
        lam_video=None,
    )
    return (
        (torch.clamp((generated[0] + 1) / 2, 0, 1) * 255)
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()[-CHUNK_SIZE:]
    )


def generate_rollout(
    pipeline: Video2WorldInference,
    bridge: G1DreamDojoActionBridge,
    video: np.ndarray,
    actions: np.ndarray,
    num_chunks: int,
    *,
    seed: int,
    num_steps: int,
    teacher_forcing: bool,
    action_mode: str,
) -> np.ndarray:
    output = [video[0]]
    current = video[0]
    for chunk_index in range(num_chunks):
        raw_start = chunk_index * RAW_STEPS_PER_CHUNK
        if teacher_forcing and raw_start < len(video):
            current = video[raw_start]
        action_window_array = actions[raw_start : raw_start + RAW_ACTION_HORIZON]
        if not len(action_window_array):
            action_window_array = actions[-1:]
        if len(action_window_array) < RAW_ACTION_HORIZON:
            padding = np.repeat(
                action_window_array[-1:],
                RAW_ACTION_HORIZON - len(action_window_array),
                axis=0,
            )
            action_window_array = np.concatenate([action_window_array, padding], axis=0)
        action_window = torch.from_numpy(action_window_array)
        if action_mode == "shuffle":
            action_window = torch.roll(action_window, RAW_ACTION_HORIZON // 2, dims=-2)
        action_condition = bridge.encode_30hz_actions(action_window)
        if action_mode == "zero":
            action_condition = torch.zeros_like(action_condition)
        generated = generate_chunk(
            pipeline,
            current,
            action_condition,
            seed + chunk_index,
            num_steps,
        )
        observed_steps = max(0, (len(video) - 1 - raw_start) // 2)
        valid_steps = min(CHUNK_SIZE, observed_steps) if observed_steps else CHUNK_SIZE
        output.extend(generated[:valid_steps])
        current = generated[-1]
    return np.stack(output)


def sampled_ground_truth(video: np.ndarray, num_chunks: int) -> np.ndarray:
    output = [video[0]]
    for chunk_index in range(num_chunks):
        raw_start = chunk_index * RAW_STEPS_PER_CHUNK
        observed_steps = max(0, (len(video) - 1 - raw_start) // 2)
        valid_steps = min(CHUNK_SIZE, observed_steps)
        if valid_steps:
            indices = raw_start + np.arange(1, valid_steps + 1) * 2
            output.extend(video[indices])
        else:
            output.extend(np.repeat(video[-1:], CHUNK_SIZE, axis=0))
    return np.stack(output)


def reconstruction_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict:
    reference = torch.from_numpy(reference).permute(0, 3, 1, 2).float() / 255
    prediction = torch.from_numpy(prediction).permute(0, 3, 1, 2).float() / 255
    lpips = piq.LPIPS()
    lpips_values = [
        lpips(prediction[start : start + 4], reference[start : start + 4])
        for start in range(0, len(reference), 4)
    ]
    return {
        "psnr": float(piq.psnr(prediction, reference).mean()),
        "ssim": float(piq.ssim(prediction, reference).mean()),
        "lpips": float(torch.stack(lpips_values).mean()),
    }


def reward_metrics(model: SuccessClassifierReward, video: np.ndarray) -> dict:
    probabilities = model.predict_probabilities(
        torch.from_numpy(video).unsqueeze(0)
    ).squeeze(0)
    return {
        "mean": float(probabilities.mean()),
        "final": float(probabilities[-1]),
        "max": float(probabilities.max()),
    }


def label_video(video: np.ndarray, text: str) -> np.ndarray:
    output = video.copy()
    for frame in output:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
    return output


def main() -> None:
    args = parse_args()
    if args.failure_extra_chunks < 0:
        raise ValueError("--failure-extra-chunks must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bridge = G1DreamDojoActionBridge(args.action_statistics)
    reward_model = SuccessClassifierReward(args.reward_checkpoint, num_envs=1).eval()
    pipeline = Video2WorldInference(
        experiment_name=args.experiment,
        ckpt_path=str(args.checkpoint.resolve()),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=CONFIG_FILE,
    )
    records = []
    try:
        for episode_dir in args.episode_dirs:
            episode = load_episode(episode_dir)
            result = episode["metadata"]["result"]
            available_steps = (len(episode["video"]) - 1) // 2
            available_chunks = (available_steps + CHUNK_SIZE - 1) // CHUNK_SIZE
            observed_chunks = min(args.max_chunks, available_chunks)
            extra_chunks = args.failure_extra_chunks if result == "fail" else 0
            num_chunks = observed_chunks + extra_chunks
            gt = sampled_ground_truth(episode["video"], num_chunks)
            observed_length = min(len(gt), available_steps + 1)
            outputs = {}
            experiments = [
                ("teacher", True, "actual"),
                ("closed", False, "actual"),
            ]
            if args.run_zero_action:
                experiments.append(("zero", False, "zero"))
            if args.run_shuffled_action:
                experiments.append(("shuffled", False, "shuffle"))
            for name, teacher, mode in experiments:
                outputs[name] = generate_rollout(
                    pipeline,
                    bridge,
                    episode["video"],
                    episode["actions"],
                    num_chunks,
                    seed=args.seed,
                    num_steps=args.num_inference_steps,
                    teacher_forcing=teacher,
                    action_mode=mode,
                )
            comparison_columns = [
                label_video(
                    gt,
                    f"REAL GT | {result}" + (" | END HELD" if extra_chunks else ""),
                ),
                label_video(outputs["teacher"], "GT ACTION | TEACHER"),
                label_video(outputs["closed"], "GT ACTION | CLOSED"),
            ]
            if "zero" in outputs:
                comparison_columns.append(
                    label_video(outputs["zero"], "ZERO ACTION | CLOSED")
                )
            if "shuffled" in outputs:
                comparison_columns.append(
                    label_video(outputs["shuffled"], "SHUFFLED ACTION | CLOSED")
                )
            comparison = np.concatenate(comparison_columns, axis=2)
            name = f"{episode_dir.parent.name}_{episode_dir.name}"
            mediapy.write_video(
                str(args.output_dir / f"{name}.mp4"),
                comparison,
                fps=args.save_fps,
            )
            record = {
                "episode": str(episode_dir),
                "label": result,
                "num_chunks": num_chunks,
                "observed_chunks": observed_chunks,
                "extra_chunks": extra_chunks,
                "teacher_reconstruction": reconstruction_metrics(
                    gt[:observed_length],
                    outputs["teacher"][:observed_length],
                ),
                "closed_reconstruction": reconstruction_metrics(
                    gt[:observed_length],
                    outputs["closed"][:observed_length],
                ),
                "reward_probability": {
                    "real": reward_metrics(reward_model, gt[:observed_length]),
                    **{
                        key: reward_metrics(reward_model, value)
                        for key, value in outputs.items()
                    },
                },
            }
            closed_float = outputs["closed"].astype(np.float32)
            if "zero" in outputs:
                record["action_sensitivity_mae"] = float(
                    np.abs(closed_float - outputs["zero"]).mean()
                )
            if "shuffled" in outputs:
                record["shuffle_sensitivity_mae"] = float(
                    np.abs(closed_float - outputs["shuffled"]).mean()
                )
            records.append(record)
            print(json.dumps(record, indent=2))
    finally:
        pipeline.cleanup()
    (args.output_dir / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    torch.enable_grad(False)
    main()
