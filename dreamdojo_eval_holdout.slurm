#!/bin/bash
#SBATCH -A healthcareeng_isaac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --time=4:00:00
#SBATCH --job-name=dd_g1_eval_10
#SBATCH -o /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/dreamdojo/%x-%j.out
#SBATCH -e /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/dreamdojo/%x-%j.err
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"DreamDojo checkpoint conversion and two inference pipelines can take several minutes to initialize"}}'

set -euo pipefail

USER_BASE="/lustre/fsw/portfolios/healthcareeng/users/yunl"
REPO_ROOT="${REPO_ROOT:-${USER_BASE}/code/DreamDojo}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${USER_BASE}/docker/dreamdojo-cu128-v2.sqsh}"
DATASET_PATH="${DATASET_PATH:-${USER_BASE}/datasets/g1_pick_trocar_rollout_all_260_headcam_eval_5s5f}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-${USER_BASE}/outputs/train/dreamdojo/pick_trocar_headcam_rollout_holdout_5s5f_official/g1_pick_trocar_headcam_2b_rollout_holdout_5s5f_official}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${USER_BASE}/outputs/eval/holdout_5s5f_official_full_local_style}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${USER_BASE}/cache/huggingface}"

ITERATION="${ITERATION:-3000}"
ITER_PADDED="$(printf "%09d" "${ITERATION}")"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_g1_pick_trocar_headcam_rollout_holdout_5s5f_official}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-35}"
SAVE_FPS="${SAVE_FPS:-15}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-90}"

DCP_MODEL_PATH="${TRAIN_RUN_ROOT}/checkpoints/iter_${ITER_PADDED}/model"
RUN_OUTPUT="${OUTPUT_ROOT}/iter_${ITERATION}"

for required_path in \
    "${CONTAINER_IMAGE}" \
    "${REPO_ROOT}/configs/2b_480_640_g1_pick_trocar_headcam_rollout_holdout_5s5f_official.yaml" \
    "${REPO_ROOT}/scripts/validate_pick_trocar_world_model.py" \
    "${REPO_ROOT}/scripts/combine_holdout_eval_5x6.py" \
    "${REPO_ROOT}/scripts/convert_distcp_to_pt.py" \
    "${DATASET_PATH}/meta/info.json" \
    "${DCP_MODEL_PATH}/.metadata"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required path: ${required_path}" >&2
        exit 1
    fi
done

mkdir -p "${RUN_OUTPUT}" "${HF_CACHE_ROOT}"

echo "=== DreamDojo 10-case Holdout Evaluation ==="
echo "Container:  ${CONTAINER_IMAGE}"
echo "Experiment: ${EXPERIMENT}"
echo "Checkpoint: ${DCP_MODEL_PATH}"
echo "Dataset:    ${DATASET_PATH}"
echo "Output:     ${RUN_OUTPUT}"
echo "Evaluator:  scripts/validate_pick_trocar_world_model.py"
echo "Length:     complete per-case action sequence"
echo "Layout:     5 rows x 6 columns (success | failure)"
echo "================================================"

MOUNTS="${REPO_ROOT}/configs:/workspace/configs"
MOUNTS="${MOUNTS},${REPO_ROOT}/scripts:/workspace/scripts"
MOUNTS="${MOUNTS},${DATASET_PATH}:/data/eval"
MOUNTS="${MOUNTS},${TRAIN_RUN_ROOT}:/training_run"
MOUNTS="${MOUNTS},${RUN_OUTPUT}:/eval_output"
MOUNTS="${MOUNTS},${HF_CACHE_ROOT}:/root/.cache/huggingface"

export EXPERIMENT ITERATION ITER_PADDED NUM_INFERENCE_STEPS SAVE_FPS
export STARTUP_STAGGER_SECONDS

srun --ntasks=1 --gpus=2 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
        set -euo pipefail
        export HF_HOME=/root/.cache/huggingface
        export PYTHONPATH="/workspace:${PYTHONPATH:-}"

        if ! hf auth whoami; then
            echo "Hugging Face authentication is required." >&2
            exit 1
        fi

        DCP_MODEL="/training_run/checkpoints/iter_${ITER_PADDED}/model"
        EXPORT_DIR="/eval_output/checkpoint_export"
        CHECKPOINT="${EXPORT_DIR}/model_ema_bf16.pt"

        if [[ ! -f "${CHECKPOINT}" ]]; then
            echo "Converting DCP checkpoint to EMA bf16..."
            mkdir -p "${EXPORT_DIR}"
            python /workspace/scripts/convert_distcp_to_pt.py \
                "${DCP_MODEL}" \
                "${EXPORT_DIR}"
        fi

        if [[ ! -f "${CHECKPOINT}" ]]; then
            echo "Missing converted checkpoint: ${CHECKPOINT}" >&2
            exit 1
        fi
        rm -f "${EXPORT_DIR}/model.pt" "${EXPORT_DIR}/model_ema_fp32.pt"

        mkdir -p /eval_output/logs

        (
            set -e
            export CUDA_VISIBLE_DEVICES=0
            for spec in 9:20 3:17 2:13 4:11 1:8; do
                position="${spec%%:*}"
                chunks="${spec##*:}"
                output_dir="/eval_output/shard_0/episode_$(printf "%03d" "${position}")"
                python /workspace/scripts/validate_pick_trocar_world_model.py \
                    --checkpoint "${CHECKPOINT}" \
                    --experiment "${EXPERIMENT}" \
                    --dataset-path /data/eval \
                    --episode-indices "${position}" \
                    --num-chunks "${chunks}" \
                    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
                    --save-fps "${SAVE_FPS}" \
                    --output-dir "${output_dir}"
            done
        ) > /eval_output/logs/gpu0.log 2>&1 &
        PID0=$!

        # Keep the evaluation identical while avoiding two simultaneous
        # checkpoint/text-encoder loading peaks on the cluster node.
        sleep "${STARTUP_STAGGER_SECONDS}"

        (
            set -e
            export CUDA_VISIBLE_DEVICES=1
            for spec in 0:16 6:17 7:17 8:15 5:10; do
                position="${spec%%:*}"
                chunks="${spec##*:}"
                output_dir="/eval_output/shard_1/episode_$(printf "%03d" "${position}")"
                python /workspace/scripts/validate_pick_trocar_world_model.py \
                    --checkpoint "${CHECKPOINT}" \
                    --experiment "${EXPERIMENT}" \
                    --dataset-path /data/eval \
                    --episode-indices "${position}" \
                    --num-chunks "${chunks}" \
                    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
                    --save-fps "${SAVE_FPS}" \
                    --output-dir "${output_dir}"
            done
        ) > /eval_output/logs/gpu1.log 2>&1 &
        PID1=$!

        wait "${PID0}"
        wait "${PID1}"

        python -m scripts.combine_holdout_eval_5x6 \
            /eval_output \
            "/eval_output/combined_5x6_success_failure_iter_${ITERATION}.mp4" \
            --dataset-path /data/eval \
            --fps "${SAVE_FPS}"

        echo "Evaluation complete: /eval_output/combined_5x6_success_failure_iter_${ITERATION}.mp4"
    '
