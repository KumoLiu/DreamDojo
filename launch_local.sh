#!/usr/bin/env bash
set -euo pipefail

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
source "$ROOT/env_local.sh"

export TORCH_DIST_INIT_BARRIER=1
export LD_PRELOAD=""
export FI_EFA_USE_DEVICE_RDMA=1
export RDMAV_FORK_SAFE=1

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiment_name> [hydra overrides ...]" >&2
  echo "Optional: set DATA_PATH to override train/val dataset paths." >&2
  exit 1
fi
config_name=$1
shift

echo "Running on $NNODES nodes with $NPROC processes per node. This node rank is $NODE_RANK."

overrides=(
  "experiment=$config_name"
  "job.wandb_mode=disabled"
  "~dataloader_train.dataloaders"
)

# Preserve dataset paths and mixing weights from the experiment YAML by
# default. DATA_PATH is an explicit opt-in override for single-dataset runs.
if [[ -n "${DATA_PATH:-}" ]]; then
  overrides+=(
    "dataloader_train.dataset.dataset_path=[${DATA_PATH}]"
    "dataloader_val.dataset.dataset_path=[${DATA_PATH}]"
  )
fi

# Allow additional Hydra overrides at the command line.
overrides+=("$@")

torchrun --nnodes="$NNODES" --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" --master_addr "$MASTER_ADDR" \
  --node_rank="$NODE_RANK" -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  "${overrides[@]}"
