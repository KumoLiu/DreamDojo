#!/usr/bin/env bash
# Post-train DreamDojo 2B G1 from official post-train weights using HF teleop + rollout datasets.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Only the final rank32 / LR3e-4 / 18k recipe is retained locally.
VARIANT="${1:-lora}"
case "$VARIANT" in
  lora)
    EXPERIMENT="dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora"
    RUN_NAME="hf_teleop_rollout_posttrain_lora"
    JOB_PATH="hf_teleop_rollout_posttrain_lora/lora_r32_scratch_lr3e-4_18k"
    ;;
  *)
    echo "Usage: $0 [lora] (other experiment configs have been retired)" >&2
    exit 1
    ;;
esac

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/${RUN_NAME}_${RUN_TS}"
mkdir -p "$LOG_DIR"

NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-12341}"

DATASETS_DIR="/localhome/local-yunl/DreamDojo/datasets"
TRAIN_DATASETS=(
  "$DATASETS_DIR/g1_hf_pick_trocar_teleop_success_train"
  "$DATASETS_DIR/g1_hf_pick_trocar_rollouts_30k_train"
  "$DATASETS_DIR/g1_hf_pick_trocar_rollouts_10k_train"
)
VAL_DATASETS=(
  "$DATASETS_DIR/g1_hf_pick_trocar_teleop_success_val"
  "$DATASETS_DIR/g1_hf_pick_trocar_rollouts_30k_val"
  "$DATASETS_DIR/g1_hf_pick_trocar_rollouts_10k_val"
)
INIT_CKPT="/localhome/local-yunl/DreamDojo/checkpoints/DreamDojo/2B_G1_post-train/iter_000050000/"

# A partially-killed run leaves ranks wedged in an NCCL collective, still holding
# GPU memory at 0% utilization. Those stragglers make the next run OOM, so refuse
# to start until they are gone.
if stale_pids=$(pgrep -f "scripts\.train|torchrun.*scripts\.train") && [[ -n "$stale_pids" ]]; then
  echo "ERROR: training processes are still running:" >&2
  echo "$stale_pids" | tr '\n' ' ' >&2
  echo >&2
  echo "Kill them first: kill -9 \$(pgrep -f 'scripts\.train|torchrun')" >&2
  exit 1
fi

{
  echo "============================================================"
  echo "DreamDojo G1 2B HF Post-Training Run"
  echo "Started: $(date -Is)"
  echo "============================================================"
  echo
  echo "[Step 1] Environment"
  echo "  ROOT=$ROOT"
  echo "  VARIANT=$VARIANT"
  echo "  LOG_DIR=$LOG_DIR"
  echo "  EXPERIMENT=$EXPERIMENT"
  echo "  NPROC=$NPROC"
  echo "  MASTER_PORT=$MASTER_PORT"
  echo "  IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$ROOT/outputs/train}"
  echo
  echo "[Step 2] Init checkpoint"
  echo "  load_path=$INIT_CKPT"
  ls -la "$INIT_CKPT/model" 2>/dev/null | head -5 || true
  echo
  echo "[Step 3] Adapted datasets (28-D Dex3 -> 43-D official G1 layout)"
  echo "  -- train (mixing weights 0.34 / 0.33 / 0.33) --"
  for ds in "${TRAIN_DATASETS[@]}" "__VAL__" "${VAL_DATASETS[@]}"; do
    if [[ "$ds" == "__VAL__" ]]; then
      echo "  -- val --"
      continue
    fi
    echo "  $(basename "$ds")"
    python3 - <<PY
import json
from pathlib import Path
info = json.loads(Path("$ds/meta/info.json").read_text())
print(f"    episodes={info['total_episodes']}, frames={info['total_frames']}, robot={info['robot_type']}")
PY
  done
  echo
  echo "[Step 4] GPU status"
  nvidia-smi -L || true
  nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader || true
  echo
  echo "[Step 5] Training mode"
  rg -n "use_lora|lora_rank|lora_alpha|lora_target_modules|  lr:" \
    "configs/${EXPERIMENT#dreamdojo_}.yaml" 2>/dev/null | sed 's/^/  /' || true
  echo
  echo "[Step 6] Launch command"
  echo "  NPROC=$NPROC ./launch_local.sh $EXPERIMENT"
  echo
} | tee "$LOG_DIR/00_setup.log"

source "$ROOT/env_local.sh"

export NPROC
export MASTER_PORT

LAUNCH_LOG="$LOG_DIR/01_training.log"
SETUP_LOG="$LOG_DIR/00_setup.log"

echo "Training stdout/stderr -> $LAUNCH_LOG" | tee -a "$SETUP_LOG"

set +e
./launch_local.sh "$EXPERIMENT" 2>&1 | tee "$LAUNCH_LOG"
EXIT_CODE=${PIPESTATUS[0]}
set -e

{
  echo
  echo "============================================================"
  echo "Finished: $(date -Is)"
  echo "Exit code: $EXIT_CODE"
  echo "Setup log: $SETUP_LOG"
  echo "Training log: $LAUNCH_LOG"
  echo "Run artifacts: $IMAGINAIRE_OUTPUT_ROOT/dreamdojo/$JOB_PATH/"
  echo "============================================================"
} | tee -a "$SETUP_LOG" "$LAUNCH_LOG"

exit "$EXIT_CODE"
