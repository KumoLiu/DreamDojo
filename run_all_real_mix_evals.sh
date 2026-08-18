#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINT_ROOT="$ROOT/outputs/train/dreamdojo/pick_trocar_headcam_real_mix/g1_pick_trocar_headcam_2b_real_mix/checkpoints"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/real_mix_checkpoint_eval}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$OUTPUT_ROOT"

for checkpoint in "$CHECKPOINT_ROOT"/iter_*; do
  name="$(basename "$checkpoint")"
  if [[ -f "$OUTPUT_ROOT/$name/rollout/metrics.json" \
    && -f "$OUTPUT_ROOT/$name/teleop/metrics.json" ]]; then
    echo "Skipping completed evaluation $name"
    continue
  fi
  if [[ ! -f "$checkpoint/model_ema_bf16.pt" ]]; then
    echo "Exporting EMA checkpoint for $name"
    source "$ROOT/env_local.sh"
    python "$ROOT/scripts/convert_distcp_to_pt.py" \
      "$checkpoint/model" "$checkpoint"
  fi
  echo "Evaluating $name"
  "$ROOT/run_real_mix_eval.sh" "$checkpoint" "$OUTPUT_ROOT/$name"
done

source "$ROOT/env_local.sh"
python "$ROOT/scripts/summarize_real_mix_evals.py" "$OUTPUT_ROOT"
