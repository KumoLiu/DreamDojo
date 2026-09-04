#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
source "$ROOT/env_local.sh"

CHECKPOINTS_DIR=${1:-checkpoints/DreamDojo/2B_G1_post-train}
EXPERIMENT=${2:-dreamdojo_2b_480_640_g1}
SAVE_DIR=${3:-results/zeroshot_2b_g1_assemble_trocar}
DATASET_PATH=${4:-datasets/g1_assemble_trocar_sim_box_v3_60}
NUM_SAMPLES=${5:-8}
NUM_FRAMES=${6:-49}
DATA_SPLIT=${7:-test}
CHECKPOINT_INTERVAL=${8:-5000}

python examples/action_conditioned.py \
  -o "outputs/action_conditioned/$(basename "$SAVE_DIR")" \
  --checkpoints-dir "$CHECKPOINTS_DIR" \
  --experiment "$EXPERIMENT" \
  --save-dir "$SAVE_DIR" \
  --num-frames "$NUM_FRAMES" \
  --num-samples "$NUM_SAMPLES" \
  --dataset-path "$DATASET_PATH" \
  --data-split "$DATA_SPLIT" \
  --deterministic-uniform-sampling \
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
