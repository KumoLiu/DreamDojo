#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/env_local.sh"

TEACHER_OUTPUT="datasets/g1_pick_trocar_iter2500_teacher_gen_10k_stratified"
TEACHER_MANIFEST="${TEACHER_OUTPUT}/sample_manifest.jsonl"
TRAIN_DATASET="datasets/g1_pick_trocar_rollout_all_260_headcam_train"
TEACHER_CHECKPOINT="outputs/train/dreamdojo/pick_trocar_headcam_rollout_holdout_5s5f_ema_both/g1_pick_trocar_headcam_2b_rollout_holdout_5s5f_ema_both/checkpoints/iter_000003000/model"
WARMUP_ROOT="outputs/train/dreamdojo/pick_trocar_self_forcing_97f/g1_pick_trocar_iter2500_warmup_stratified"
SELF_FORCING_ROOT="outputs/train/dreamdojo/pick_trocar_self_forcing_97f/g1_pick_trocar_iter2500_self_forcing_97f"
EVAL_ROOT="results/pick_trocar_distilled_holdout_97f"
DATASET="datasets/g1_pick_trocar_rollout_all_260_headcam_eval_5s5f"

export IMAGINAIRE_OUTPUT_ROOT="$ROOT/outputs/train"
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=8

if [[ ! -f "$TEACHER_MANIFEST" ]]; then
    python scripts/build_teacher_gen_sampling_manifest.py \
        --dataset-path "$TRAIN_DATASET" \
        --output "$TEACHER_MANIFEST"
fi

if [[ ! -f "${TEACHER_OUTPUT}/actions/9999.json" ]]; then
    echo "Generating 10k stratified Teacher targets..."
    CUDA_VISIBLE_DEVICES=2,3,5,6 torchrun --nnodes=1 --nproc_per_node=4 \
        --master_port=12358 --master_addr=localhost --node_rank=0 \
        -m cosmos_predict2._src.predict2.action.inference.inference_gr00t_warmup -- \
        --experiment=dreamdojo_2b_480_640_g1_pick_trocar_headcam_rollout_holdout_5s5f_ema_both \
        --ckpt_path="$TEACHER_CHECKPOINT" \
        --dataset_path="$TRAIN_DATASET" \
        --sample_manifest="$TEACHER_MANIFEST" \
        --save_root="$TEACHER_OUTPUT" \
        --guidance=0 --chunk_size=12 --start=0 --end=10000 \
        --query_steps=0,9,18,27,34 \
        2>&1 | tee results/g1_pick_trocar_iter2500_teacher_gen_10k_stratified.log
fi

python scripts/validate_teacher_gen_dataset.py "$TEACHER_OUTPUT"

echo "Starting stratified Warmup (5000 iterations)..."
CUDA_VISIBLE_DEVICES=2,3,5,6 torchrun --nnodes=1 --nproc_per_node=4 \
    --master_port=12359 --master_addr=localhost --node_rank=0 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/interactive/configs/config_warmup.py -- \
    experiment=pick_trocar_iter2500_warmup_stratified_no_s3 \
    job.wandb_mode=disabled \
    2>&1 | tee results/g1_pick_trocar_iter2500_warmup_stratified.log

test -f "${WARMUP_ROOT}/checkpoints/iter_000003000/model/.metadata"
test -f "${WARMUP_ROOT}/checkpoints/iter_000005000/model/.metadata"

mkdir -p "$EVAL_ROOT"
echo "Evaluating Warmup iterations 3000 and 5000..."
(
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_distilled_holdout.py \
        --checkpoint "${WARMUP_ROOT}/checkpoints/iter_000003000" \
        --output-dir "${EVAL_ROOT}/warmup_iter3000" \
        --episode-indices 0 1 2 3 4 5 6 7 8 9 \
        --num-frames 49 --net-max-frames 128 \
        > results/g1_pick_trocar_warmup_stratified_iter3000_eval.log 2>&1
) &
EVAL_3000_PID=$!
(
    CUDA_VISIBLE_DEVICES=3 python scripts/eval_distilled_holdout.py \
        --checkpoint "${WARMUP_ROOT}/checkpoints/iter_000005000" \
        --output-dir "${EVAL_ROOT}/warmup_iter5000" \
        --episode-indices 0 1 2 3 4 5 6 7 8 9 \
        --num-frames 49 --net-max-frames 128 \
        > results/g1_pick_trocar_warmup_stratified_iter5000_eval.log 2>&1
) &
EVAL_5000_PID=$!
wait "$EVAL_3000_PID"
wait "$EVAL_5000_PID"

SELECTED_FILE="${EVAL_ROOT}/selected_warmup.json"
SELECTED_WARMUP="$(
    python scripts/select_distilled_checkpoint.py \
        "${EVAL_ROOT}/warmup_iter3000" \
        "${EVAL_ROOT}/warmup_iter5000" \
        --output "$SELECTED_FILE"
)"
echo "Selected Warmup checkpoint: $SELECTED_WARMUP"

echo "Running one-iteration 97-frame memory smoke test..."
CUDA_VISIBLE_DEVICES=2,3,5,6 torchrun --nnodes=1 --nproc_per_node=4 \
    --master_port=12360 --master_addr=localhost --node_rank=0 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py -- \
    experiment=pick_trocar_iter2500_self_forcing_97f_no_s3 \
    checkpoint.load_path="$SELECTED_WARMUP" \
    checkpoint.save_iter=100000 \
    trainer.max_iter=1 \
    job.name=g1_pick_trocar_iter2500_self_forcing_97f_smoke \
    job.wandb_mode=disabled \
    2>&1 | tee results/g1_pick_trocar_iter2500_self_forcing_97f_smoke.log

echo "Starting 97-frame Self-Forcing (3000 iterations)..."
CUDA_VISIBLE_DEVICES=2,3,5,6 torchrun --nnodes=1 --nproc_per_node=4 \
    --master_port=12361 --master_addr=localhost --node_rank=0 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py -- \
    experiment=pick_trocar_iter2500_self_forcing_97f_no_s3 \
    checkpoint.load_path="$SELECTED_WARMUP" \
    job.wandb_mode=disabled \
    2>&1 | tee results/g1_pick_trocar_iter2500_self_forcing_97f.log

test -f "${SELF_FORCING_ROOT}/checkpoints/iter_000003000/model/.metadata"

echo "Evaluating selected Warmup and 97-frame Self-Forcing at 97 frames..."
(
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_distilled_holdout.py \
        --checkpoint "$SELECTED_WARMUP" \
        --output-dir "${EVAL_ROOT}/warmup_selected_97f" \
        --episode-indices 0 1 2 3 4 5 6 7 8 9 \
        --num-frames 97 --net-max-frames 128 \
        > results/g1_pick_trocar_warmup_selected_97f_eval.log 2>&1
) &
EVAL_WARMUP_PID=$!
(
    CUDA_VISIBLE_DEVICES=3 python scripts/eval_distilled_holdout.py \
        --checkpoint "${SELF_FORCING_ROOT}/checkpoints/iter_000003000" \
        --output-dir "${EVAL_ROOT}/self_forcing_97f" \
        --episode-indices 0 1 2 3 4 5 6 7 8 9 \
        --num-frames 97 \
        > results/g1_pick_trocar_self_forcing_97f_eval.log 2>&1
) &
EVAL_SELF_FORCING_PID=$!
wait "$EVAL_WARMUP_PID"
wait "$EVAL_SELF_FORCING_PID"

bash scripts/finish_pick_trocar_97f_evaluation.sh

echo "PICK_TROCAR_97F_PIPELINE_COMPLETE"
