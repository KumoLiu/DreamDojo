#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/env_local.sh"

EVAL_ROOT="results/pick_trocar_distilled_holdout_97f"
TEACHER_CHECKPOINT="outputs/train/dreamdojo/pick_trocar_headcam_rollout_holdout_5s5f_ema_both/g1_pick_trocar_headcam_2b_rollout_holdout_5s5f_ema_both/checkpoints/iter_000003000/model_ema_bf16.pt"
DATASET="datasets/g1_pick_trocar_rollout_all_260_headcam_eval_5s5f"
EXPERIMENT="dreamdojo_2b_480_640_g1_pick_trocar_headcam_rollout_holdout_5s5f_ema_both"

mkdir -p "${EVAL_ROOT}/teacher_97f_gpu2" "${EVAL_ROOT}/teacher_97f_gpu3"
(
    CUDA_VISIBLE_DEVICES=2 python scripts/validate_pick_trocar_world_model.py \
        --checkpoint "$TEACHER_CHECKPOINT" \
        --experiment "$EXPERIMENT" \
        --dataset-path "$DATASET" \
        --episode-indices 0 2 4 6 8 \
        --num-chunks 8 \
        --num-inference-steps 35 \
        --save-fps 15 \
        --output-dir "${EVAL_ROOT}/teacher_97f_gpu2"
) > results/g1_pick_trocar_teacher_97f_gpu2_eval.log 2>&1 &
PID2=$!
(
    CUDA_VISIBLE_DEVICES=3 python scripts/validate_pick_trocar_world_model.py \
        --checkpoint "$TEACHER_CHECKPOINT" \
        --experiment "$EXPERIMENT" \
        --dataset-path "$DATASET" \
        --episode-indices 1 3 5 7 9 \
        --num-chunks 8 \
        --num-inference-steps 35 \
        --save-fps 15 \
        --output-dir "${EVAL_ROOT}/teacher_97f_gpu3"
) > results/g1_pick_trocar_teacher_97f_gpu3_eval.log 2>&1 &
PID3=$!
wait "$PID2"
wait "$PID3"

python - <<'PY'
import json
import shutil
from pathlib import Path

root = Path("results/pick_trocar_distilled_holdout_97f")
output = root / "teacher_iter3000_97f"
output.mkdir(parents=True, exist_ok=True)
records = []
for split in (root / "teacher_97f_gpu2", root / "teacher_97f_gpu3"):
    records.extend(json.loads((split / "metrics.json").read_text()))
    for video in split.glob("*_comparison.mp4"):
        shutil.copy2(video, output / video.name)
records.sort(key=lambda record: (record["episode_index"], record["seed_index"]))
if [record["episode_index"] for record in records] != list(range(10)):
    raise RuntimeError("Teacher 97-frame evaluation is incomplete")
(output / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
print(f"Merged {len(records)} teacher evaluations into {output}")
PY

python scripts/combine_distilled_holdout_eval.py \
    "$EVAL_ROOT" \
    "${EVAL_ROOT}/heldout_teacher_warmup_self_forcing_97f_5x8.mp4" \
    --dataset-path "$DATASET" \
    --teacher-dir teacher_iter3000_97f \
    --warmup-dir warmup_selected_97f \
    --self-forcing-dir self_forcing_97f \
    --teacher-label "TEACHER 3000" \
    --warmup-label "WARMUP 3000" \
    --self-forcing-label "SELF-FORCING 97F" \
    --fps 15

echo "PICK_TROCAR_97F_EVALUATION_COMPLETE"
