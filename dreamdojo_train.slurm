#!/bin/bash
#SBATCH -A healthcareeng_isaac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=4:00:00
#SBATCH --job-name=dd_g1_holdout_official
#SBATCH -o /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/dreamdojo/%x-%j.out
#SBATCH -e /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/dreamdojo/%x-%j.err
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"DreamDojo model, text encoder, and distributed checkpoint loading can take several minutes"}}'

set -euo pipefail

PROJECT_BASE="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac"
USER_BASE="/lustre/fsw/portfolios/healthcareeng/users/yunl"

# Host paths. Any value can still be overridden through sbatch --export.
REPO_ROOT="${REPO_ROOT:-${USER_BASE}/code/DreamDojo}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${USER_BASE}/docker/dreamdojo-cu128-v2.sqsh}"
DATASET_PATH="${DATASET_PATH:-${USER_BASE}/datasets/g1_pick_trocar_rollout_all_260_headcam_train}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-${USER_BASE}/datasets/g1_pick_trocar_rollout_all_260_headcam_eval_5s5f}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${USER_BASE}/checkpoints}"
CHECKPOINT_PATH="${CHECKPOINT_ROOT}/DreamDojo/2B_G1_post-train/iter_000050000"
OUTPUT_ROOT="${OUTPUT_ROOT:-${USER_BASE}/outputs/train}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${USER_BASE}/cache/huggingface}"

EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_g1_pick_trocar_headcam_rollout_holdout_5s5f_official}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-8}}"
MASTER_PORT="${MASTER_PORT:-12343}"

export EXPERIMENT NUM_GPUS MASTER_PORT
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=8

for required_path in \
    "${CONTAINER_IMAGE}" \
    "${REPO_ROOT}/configs/2b_480_640_g1_pick_trocar_headcam_rollout_holdout_5s5f_official.yaml" \
    "${DATASET_PATH}/meta/info.json" \
    "${VAL_DATASET_PATH}/meta/info.json"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required path: ${required_path}" >&2
        exit 1
    fi
done

mkdir -p "${CHECKPOINT_ROOT}" "${OUTPUT_ROOT}" "${HF_CACHE_ROOT}"

echo "=== DreamDojo Slurm Training ==="
echo "Repo:       ${REPO_ROOT}"
echo "Container:  ${CONTAINER_IMAGE}"
echo "Experiment: ${EXPERIMENT}"
echo "Train data: ${DATASET_PATH}"
echo "Val data:   ${VAL_DATASET_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output:     ${OUTPUT_ROOT}"
echo "GPUs:       ${NUM_GPUS}"
echo "================================"

# Keep the environment baked into the image, but mount the current configs so
# newly added experiments do not require rebuilding the image.
MOUNTS="${REPO_ROOT}/configs:/workspace/configs"
MOUNTS="${MOUNTS},${DATASET_PATH}:/data/train"
MOUNTS="${MOUNTS},${VAL_DATASET_PATH}:/data/val"
MOUNTS="${MOUNTS},${CHECKPOINT_ROOT}:/checkpoints"
MOUNTS="${MOUNTS},${OUTPUT_ROOT}:/outputs"
MOUNTS="${MOUNTS},${HF_CACHE_ROOT}:/root/.cache/huggingface"

srun --ntasks=1 --gpus="${NUM_GPUS}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
        export HF_HOME=/root/.cache/huggingface
        export IMAGINAIRE_OUTPUT_ROOT=/outputs
        CHECKPOINT=/checkpoints/DreamDojo/2B_G1_post-train/iter_000050000

        if ! hf auth whoami; then
            echo "Hugging Face authentication is required for gated model nvidia/Cosmos-Predict2.5-2B." >&2
            echo "Store a read token at ${HF_HOME}/token and resubmit." >&2
            exit 1
        fi

        # Build the complete config once before spawning 8 ranks. This catches
        # missing dependencies/auth and populates shared HF cache only once.
        python -c "
import h5py
import pytorch3d
import torchcodec
from cosmos_predict2._src.predict2.action.configs.action_conditioned.config import make_config
make_config()
print(\"DreamDojo training config preflight passed\")
"

        if [[ ! -f "${CHECKPOINT}/model/.metadata" ]]; then
            echo "Downloading official nvidia/DreamDojo G1 post-train checkpoint..."
            hf download nvidia/DreamDojo \
                --include "2B_G1_post-train/iter_000050000/model/**" \
                --local-dir /checkpoints/DreamDojo
        fi

        if [[ ! -f "${CHECKPOINT}/model/.metadata" ]]; then
            echo "Official checkpoint download is incomplete: ${CHECKPOINT}" >&2
            exit 1
        fi

        NPROC="${NUM_GPUS}" MASTER_PORT="${MASTER_PORT}" \
            /workspace/launch_local.sh "${EXPERIMENT}" \
            "checkpoint.load_path=${CHECKPOINT}" \
            "checkpoint.load_training_state=false" \
            "checkpoint.strict_resume=false" \
            "dataloader_train.dataset.dataset_path=[/data/train]" \
            "dataloader_val.dataset.dataset_path=[/data/val]"
    '
