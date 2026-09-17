#!/usr/bin/env bash
# Evaluate one sweep run on the held-out validation sets and build the overview
# videos. Pairs with dreamdojo_sweep_train.slurm: pass the same RUN_NAME.
#
#   sbatch --export=ALL,RUN_NAME=lora_r32_scratch_lr3e-4_18k,\
#EXPERIMENT=dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora \
#          dreamdojo_sweep_eval.slurm
#SBATCH -A healthcareeng_isaac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=2:00:00
#SBATCH --job-name=dd_sweep_eval
#SBATCH -o /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/dreamdojo/%x-%j.out
#SBATCH -e /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/dreamdojo/%x-%j.err
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"DreamDojo checkpoint conversion and evaluation"}}'

set -euo pipefail

USER_BASE="${USER_BASE:-/lustre/fsw/portfolios/healthcareeng/users/yunl}"
REPO_ROOT="${REPO_ROOT:-${USER_BASE}/code/DreamDojo}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${USER_BASE}/docker/dreamdojo-cu128-v2.sqsh}"
DATASET_ROOT="${DATASET_ROOT:-${USER_BASE}/datasets}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${USER_BASE}/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${USER_BASE}/outputs/train}"
EVAL_ROOT="${EVAL_ROOT:-${USER_BASE}/outputs/eval}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${USER_BASE}/cache/huggingface}"

if [[ -z "${EXPERIMENT:-}" || -z "${RUN_NAME:-}" ]]; then
    echo "EXPERIMENT and RUN_NAME are both required" >&2
    exit 1
fi
ITERATION="${ITERATION:-18000}"
HORIZON="${HORIZON:-auto}"
# The output directory keys off RUN_NAME, ITERATION and HORIZON, none of which
# capture the protocol the model is scored under. Evaluating one run twice with
# different clip lengths would otherwise merge both sets of per-episode results
# into one directory. EVAL_TAG separates them.
EVAL_TAG="${EVAL_TAG:-}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-8}}"

# dreamdojo_sweep_train.slurm sets job.name=RUN_NAME, and job.group comes from
# the experiment YAML, so the run directory is <group>/<RUN_NAME>.
CONFIG_NAME="${CONFIG_NAME:-${EXPERIMENT#dreamdojo_}}"
JOB_GROUP="${JOB_GROUP:-$(
    python3 -c "
import sys, yaml
c = yaml.safe_load(open('${REPO_ROOT}/configs/${CONFIG_NAME}.yaml'))
print((c.get('job') or {}).get('group', ''))
"
)}"
if [[ -z "${JOB_GROUP}" ]]; then
    echo "Could not read job.group from configs/${CONFIG_NAME}.yaml" >&2
    exit 1
fi

mkdir -p "${EVAL_ROOT}" "${HF_CACHE_ROOT}"

echo "=== DreamDojo Sweep Eval ==="
echo "Run name:   ${RUN_NAME}"
echo "Experiment: ${EXPERIMENT}"
echo "Job group:  ${JOB_GROUP}"
echo "Iteration:  ${ITERATION}"
echo "Horizon:    ${HORIZON}"
echo "Eval tag:   ${EVAL_TAG:-<none>}"
echo "==========================="

MOUNTS="${REPO_ROOT}/configs:/workspace/configs"
MOUNTS="${MOUNTS},${REPO_ROOT}/cosmos_predict2:/workspace/cosmos_predict2"
MOUNTS="${MOUNTS},${REPO_ROOT}/groot_dreams:/workspace/groot_dreams"
MOUNTS="${MOUNTS},${REPO_ROOT}/shared_meta:/workspace/shared_meta"
MOUNTS="${MOUNTS},${REPO_ROOT}/scripts:/workspace/scripts"
MOUNTS="${MOUNTS},${REPO_ROOT}/launch_local.sh:/workspace/launch_local.sh"
# env_local.sh and .venv come from the container image; mounting the
# workstation copy would point it at a .venv that does not exist there.
MOUNTS="${MOUNTS},${DATASET_ROOT}:/workspace/datasets"
# The model loads the latent action model from the relative path
# checkpoints/DreamDojo/LAM_400k.ckpt, resolved against the container workdir.
MOUNTS="${MOUNTS},${CHECKPOINT_ROOT}:/workspace/checkpoints"
MOUNTS="${MOUNTS},${OUTPUT_ROOT}:/outputs"
MOUNTS="${MOUNTS},${EVAL_ROOT}:/eval"
MOUNTS="${MOUNTS},${HF_CACHE_ROOT}:/root/.cache/huggingface"

srun --ntasks=1 --gpus="${NUM_GPUS}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    --export=ALL \
    bash -lc "
        set -euo pipefail
        export HF_HOME=/root/.cache/huggingface
        export PYTHONPATH=/workspace:\${PYTHONPATH:-}
        ITER_PADDED=\$(printf '%09d' '${ITERATION}')
        CKPT_DIR=/outputs/dreamdojo/${JOB_GROUP}/${RUN_NAME}/checkpoints/iter_\${ITER_PADDED}

        if [[ ! -d \"\${CKPT_DIR}/model\" ]]; then
            echo \"Missing checkpoint: \${CKPT_DIR}/model\" >&2
            exit 1
        fi

        # A missing LAM checkpoint only prints a warning and leaves the latent
        # action model randomly initialised, which would corrupt the last 32
        # action dims and quietly depress every metric.
        if [[ ! -f /workspace/checkpoints/DreamDojo/LAM_400k.ckpt ]]; then
            echo 'Missing LAM checkpoint: /workspace/checkpoints/DreamDojo/LAM_400k.ckpt' >&2
            exit 1
        fi

        if [[ ! -f \"\${CKPT_DIR}/model_ema_bf16.pt\" ]]; then
            python scripts/convert_distcp_to_pt.py \"\${CKPT_DIR}/model\" \"\${CKPT_DIR}\"
        fi

        VARIANT=custom \
        RUN_ROOT=/outputs/dreamdojo/${JOB_GROUP}/${RUN_NAME} \
        EXPERIMENT='${EXPERIMENT}' \
        ITERATION='${ITERATION}' \
        HORIZON='${HORIZON}' \
        NUM_GPUS='${NUM_GPUS}' \
        OUT_ROOT=/eval/${RUN_NAME}_iter${ITERATION}_h${HORIZON}${EVAL_TAG:+_${EVAL_TAG}} \
            bash scripts/run_hf_posttrain_val_eval.sh
    "
