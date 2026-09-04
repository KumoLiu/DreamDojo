#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/env_local.sh"

OUTPUT_ROOT="${1:?Usage: $0 OUTPUT_ROOT [ITERATION]}"
ITERATION="${2:-2500}"
GPU_A="${GPU_A:-5}"
GPU_B="${GPU_B:-6}"
ITER_PADDED="$(printf "%09d" "$ITERATION")"

RUN_ROOT="${RUN_ROOT:-outputs/train/dreamdojo/pick_trocar_headcam_rollout_holdout_5s5f_ema_both/g1_pick_trocar_headcam_2b_rollout_holdout_5s5f_ema_both}"
CHECKPOINT="${RUN_ROOT}/checkpoints/iter_${ITER_PADDED}/model_ema_bf16.pt"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_g1_pick_trocar_headcam_rollout_holdout_5s5f_ema_both}"
DATASET="${DATASET:-datasets/g1_pick_trocar_rollout_all_260_headcam_eval_5s5f}"
ITER_OUTPUT="${OUTPUT_ROOT}/iter_${ITERATION}"

if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2
    exit 1
fi
test -f "$CHECKPOINT"
mkdir -p "$ITER_OUTPUT" "${OUTPUT_ROOT}/logs"

run_cases() {
    local gpu="$1"
    shift
    export CUDA_VISIBLE_DEVICES="$gpu"
    for spec in "$@"; do
        local episode="${spec%%:*}"
        local chunks="${spec##*:}"
        python scripts/validate_pick_trocar_world_model.py \
            --checkpoint "$CHECKPOINT" \
            --experiment "$EXPERIMENT" \
            --dataset-path "$DATASET" \
            --episode-indices "$episode" \
            --num-chunks "$chunks" \
            --num-inference-steps 35 \
            --save-fps 15 \
            --output-dir "${ITER_OUTPUT}/episode_$(printf "%03d" "$episode")"
    done
}

run_cases "$GPU_A" 9:20 3:17 2:13 4:11 1:8 \
    > "${OUTPUT_ROOT}/logs/gpu${GPU_A}.log" 2>&1 &
PID_A=$!
run_cases "$GPU_B" 0:16 6:17 7:17 8:15 5:10 \
    > "${OUTPUT_ROOT}/logs/gpu${GPU_B}.log" 2>&1 &
PID_B=$!
wait "$PID_A"
wait "$PID_B"

python scripts/combine_holdout_eval_5x6.py \
    "$ITER_OUTPUT" \
    "${ITER_OUTPUT}/combined_5x6_success_failure_iter_${ITERATION}_full.mp4" \
    --dataset-path "$DATASET" \
    --fps 15

python - "$ITER_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
for position in range(10):
    path = root / f"episode_{position:03d}" / "metrics.json"
    episode_records = json.loads(path.read_text())
    if len(episode_records) != 1:
        raise RuntimeError(f"Expected one metric record in {path}")
    records.append(episode_records[0])

summary = {
    mode: {
        metric: sum(record[mode][metric] for record in records) / len(records)
        for metric in ("mse", "psnr", "mae")
    }
    for mode in ("teacher_forced", "closed_loop")
}
(root / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "LOCAL_HOLDOUT_FULL_EVAL_COMPLETE: ${ITER_OUTPUT}"
