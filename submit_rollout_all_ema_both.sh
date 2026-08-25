#!/usr/bin/env bash
set -euo pipefail

# Run this script from the DreamDojo repository on the Slurm cluster.
USER_BASE="${USER_BASE:-/lustre/fsw/portfolios/healthcareeng/users/yunl}"
REPO_ROOT="${REPO_ROOT:-${USER_BASE}/code/DreamDojo}"
DATASET_PATH="${DATASET_PATH:-${USER_BASE}/datasets/g1_pick_trocar_rollout_all_260_headcam}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-${USER_BASE}/datasets/g1_pick_trocar_200_headcam_val}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${USER_BASE}/checkpoints/g1_pick_trocar_headcam_2b_iter_000010000_ema_both}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${USER_BASE}/outputs/train}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_g1_pick_trocar_headcam_rollout_all_ema_both}"
NUM_GPUS="${NUM_GPUS:-8}"

for required_path in \
    "${REPO_ROOT}/dreamdojo_train.slurm" \
    "${REPO_ROOT}/configs/2b_480_640_g1_pick_trocar_headcam_rollout_all_ema_both.yaml" \
    "${DATASET_PATH}/meta/info.json" \
    "${VAL_DATASET_PATH}/meta/info.json" \
    "${CHECKPOINT_PATH}/model/.metadata"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required path: ${required_path}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_ROOT}"

echo "Submitting ${EXPERIMENT}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Train data: ${DATASET_PATH}"
echo "Output:     ${OUTPUT_ROOT}"

sbatch \
    --gpus="${NUM_GPUS}" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},DATASET_PATH=${DATASET_PATH},VAL_DATASET_PATH=${VAL_DATASET_PATH},CHECKPOINT_PATH=${CHECKPOINT_PATH},OUTPUT_ROOT=${OUTPUT_ROOT},EXPERIMENT=${EXPERIMENT},NUM_GPUS=${NUM_GPUS}" \
    "${REPO_ROOT}/dreamdojo_train.slurm"
