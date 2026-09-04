#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <checkpoint-directory> <output-directory>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINT_DIR="$(realpath "$1")"
OUTPUT_DIR="$2"
source "$ROOT/env_local.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

counterfactual_args=()
if [[ "${RUN_ZERO_ACTION:-0}" == "1" ]]; then
  counterfactual_args+=(--run-zero-action)
fi
if [[ "${RUN_SHUFFLED_ACTION:-0}" == "1" ]]; then
  counterfactual_args+=(--run-shuffled-action)
fi

python "$ROOT/scripts/validate_real_rollouts.py" \
  --checkpoint "$CHECKPOINT_DIR/model_ema_bf16.pt" \
  --episode-dirs \
    /localhome/local-yunl/real_rollouts/20260817_180132/episode_0013_fail \
    /localhome/local-yunl/real_rollouts/20260817_182307/episode_0013_success \
    /localhome/local-yunl/real_rollouts/20260817_180132/episode_0015_success \
    /localhome/local-yunl/real_rollouts/20260817_182307/episode_0015_fail \
  --max-chunks 100 \
  --failure-extra-chunks "${FAILURE_EXTRA_CHUNKS:-0}" \
  "${counterfactual_args[@]}" \
  --output-dir "$OUTPUT_DIR/rollout"

python "$ROOT/scripts/validate_pick_trocar_world_model.py" \
  --checkpoint "$CHECKPOINT_DIR/model_ema_bf16.pt" \
  --dataset-path \
    /localhome/local-yunl/DreamDojo/datasets/g1_pick_trocar_200_headcam_val \
  --episode-indices 0 1 \
  --num-chunks 3 \
  --output-dir "$OUTPUT_DIR/teleop"
