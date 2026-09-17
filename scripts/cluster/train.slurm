#!/usr/bin/env bash
# Generic DreamDojo training job for comparison sweeps.
#
# This template does not hardcode a fixed set of datasets.
# It mounts the whole dataset root at /data and rewrites
# every dataset_path in the experiment YAML to its /data equivalent, so any
# experiment with any number of datasets works without editing this file.
#
# Submit via scripts/cluster/submit.py, or directly:
#   sbatch --export=ALL,EXPERIMENT=dreamdojo_...,RUN_NAME=my_run \
#          scripts/cluster/train.slurm
#SBATCH -A healthcareeng_isaac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=4:00:00
#SBATCH --job-name=dd_sweep
#SBATCH -o /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/dreamdojo/%x-%j.out
#SBATCH -e /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/dreamdojo/%x-%j.err
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"DreamDojo model and distributed checkpoint loading"}}'

set -euo pipefail

USER_BASE="${USER_BASE:-/lustre/fsw/portfolios/healthcareeng/users/yunl}"
REPO_ROOT="${REPO_ROOT:-${USER_BASE}/code/DreamDojo}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${USER_BASE}/docker/dreamdojo-cu128-v2.sqsh}"
DATASET_ROOT="${DATASET_ROOT:-${USER_BASE}/datasets}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${USER_BASE}/checkpoints}"
# Under /workspace rather than /checkpoints: the model loads the latent action
# model from the relative path checkpoints/DreamDojo/LAM_400k.ckpt, so the
# checkpoint tree has to sit at the container workdir. Keeping a single mount
# point avoids binding the same host directory twice.
CONTAINER_CHECKPOINT_PATH="${CONTAINER_CHECKPOINT_PATH:-/workspace/checkpoints/DreamDojo/2B_G1_post-train/iter_000050000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${USER_BASE}/outputs/train}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${USER_BASE}/cache/huggingface}"

if [[ -z "${EXPERIMENT:-}" ]]; then
    echo "EXPERIMENT is required (e.g. dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora)" >&2
    exit 1
fi
# configs/<stem>.yaml registers as experiment dreamdojo_<stem>.
CONFIG_NAME="${CONFIG_NAME:-${EXPERIMENT#dreamdojo_}}"
# Distinguishes output dirs when a sweep runs the same experiment with
# different overrides; without it every run would collide on job.name.
RUN_NAME="${RUN_NAME:-${CONFIG_NAME}}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-8}}"
MASTER_PORT="${MASTER_PORT:-$((12000 + RANDOM % 2000))}"
MAX_ITER="${MAX_ITER:-18000}"
LOAD_TRAINING_STATE="${LOAD_TRAINING_STATE:-false}"
# Extra Hydra overrides. OVERRIDES_FILE is what scripts/cluster/submit.py uses,
# because sbatch --export splits on commas and mangles spaces; OVERRIDES stays
# available for one-off manual submissions.
OVERRIDES="${OVERRIDES:-}"
if [[ -n "${OVERRIDES_FILE:-}" ]]; then
    if [[ ! -f "${OVERRIDES_FILE}" ]]; then
        echo "OVERRIDES_FILE does not exist: ${OVERRIDES_FILE}" >&2
        exit 1
    fi
    OVERRIDES="${OVERRIDES} $(tr '\n' ' ' < "${OVERRIDES_FILE}")"
fi

CONFIG_PATH="${REPO_ROOT}/configs/${CONFIG_NAME}.yaml"

for required_path in "${CONTAINER_IMAGE}" "${CONFIG_PATH}" "${DATASET_ROOT}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required path: ${required_path}" >&2
        exit 1
    fi
done

mkdir -p "${CHECKPOINT_ROOT}" "${OUTPUT_ROOT}" "${HF_CACHE_ROOT}"

# A long run is a chain of 4h jobs that resubmit themselves, each resuming from
# the run's own latest_checkpoint.txt. If a link somehow starts after the run is
# already done, it must exit rather than train past the target.
JOB_GROUP="$(python3 -c "
import yaml
config = yaml.safe_load(open('${CONFIG_PATH}'))
print((config.get('job') or {}).get('group', ''))
")"
if [[ -z "${JOB_GROUP}" ]]; then
    echo "Could not read job.group from ${CONFIG_PATH}" >&2
    exit 1
fi
RUN_DIR="${OUTPUT_ROOT}/dreamdojo/${JOB_GROUP}/${RUN_NAME}"
LATEST_FILE="${RUN_DIR}/checkpoints/latest_checkpoint.txt"
if [[ -f "${LATEST_FILE}" ]]; then
    # File holds a directory name like iter_000003000.
    CURRENT_ITER=$(( 10#$(tr -dc '0-9' < "${LATEST_FILE}") ))
    echo "Resuming from iteration ${CURRENT_ITER} (target ${MAX_ITER})"
    if (( CURRENT_ITER >= MAX_ITER )); then
        echo "Run already reached ${MAX_ITER} iterations; nothing to do."
        exit 0
    fi
else
    echo "No checkpoint in ${RUN_DIR}; starting from ${CONTAINER_CHECKPOINT_PATH}"
fi

export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=8

# Runs longer than the 4h wall clock continue as a self-resubmitting chain.
# SLURM copies this script into its spool dir, so $0 cannot resolve the repo;
# source the library from the canonical checkout instead.
source "${REPO_ROOT}/scripts/lib/chain.sh"

# Training is killed here rather than by SLURM, leaving ~6 min of the 4h slot
# for container teardown and the resubmit itself. A kill costs at most
# checkpoint.save_iter iterations, because the next link resumes from the last
# completed save recorded in checkpoints/latest_checkpoint.txt.
CHAIN_TIMEOUT="${CHAIN_TIMEOUT:-3.9h}"

echo "=== DreamDojo Sweep Training ==="
echo "Chain:      ${CHAIN_ID:-<none>} run ${RUN_COUNT}/${MAX_RUNS} (timeout ${CHAIN_TIMEOUT})"
echo "Run name:   ${RUN_NAME}"
echo "Experiment: ${EXPERIMENT}"
echo "Config:     ${CONFIG_PATH}"
echo "GPUs:       ${NUM_GPUS}"
echo "Max iter:   ${MAX_ITER}"
echo "Overrides:  ${OVERRIDES:-<none>}"
echo "Job ID:     ${SLURM_JOB_ID:-<none>}"
echo "================================"

MOUNTS="${REPO_ROOT}/configs:/workspace/configs"
MOUNTS="${MOUNTS},${REPO_ROOT}/cosmos_predict2:/workspace/cosmos_predict2"
MOUNTS="${MOUNTS},${REPO_ROOT}/groot_dreams:/workspace/groot_dreams"
MOUNTS="${MOUNTS},${REPO_ROOT}/shared_meta:/workspace/shared_meta"
MOUNTS="${MOUNTS},${REPO_ROOT}/scripts:/workspace/scripts"
MOUNTS="${MOUNTS},${REPO_ROOT}/external:/workspace/external"
MOUNTS="${MOUNTS},${DATASET_ROOT}:/data"
MOUNTS="${MOUNTS},${CHECKPOINT_ROOT}:/workspace/checkpoints"
MOUNTS="${MOUNTS},${OUTPUT_ROOT}:/outputs"
MOUNTS="${MOUNTS},${HF_CACHE_ROOT}:/root/.cache/huggingface"

# A timeout here is the expected way every link but the last one ends, so the
# exit code is captured rather than allowed to abort the script under `set -e`.
set +e
timeout "${CHAIN_TIMEOUT}" \
srun --ntasks=1 --gpus="${NUM_GPUS}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    --export=ALL \
    bash -lc "
        set -euo pipefail
        export HF_HOME=/root/.cache/huggingface
        export IMAGINAIRE_OUTPUT_ROOT=/outputs
        export PYTHONPATH=/workspace:\${PYTHONPATH:-}
        CHECKPOINT='${CONTAINER_CHECKPOINT_PATH}'

        if ! hf auth whoami; then
            echo 'Hugging Face authentication is required.' >&2
            exit 1
        fi

        # Fail before model construction; the mounted LAM source also checks
        # for missing/incompatible weights during strict checkpoint loading.
        if [[ ! -f /workspace/checkpoints/DreamDojo/LAM_400k.ckpt ]]; then
            echo 'Missing LAM checkpoint: /workspace/checkpoints/DreamDojo/LAM_400k.ckpt' >&2
            echo 'Refusing to train without the pretrained latent action model.' >&2
            exit 1
        fi

        python -c \"
import h5py
import pytorch3d
import torchcodec
from cosmos_predict2._src.predict2.action.configs.action_conditioned.config import make_config
make_config()
print('DreamDojo training config preflight passed')
\"

        # The YAML holds workstation-absolute dataset paths. Rewrite each one to
        # its /data equivalent and fail loudly if a dataset was never synced,
        # rather than letting the trainer die minutes later.
        DATASET_OVERRIDES=\$(python - <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path('/workspace/configs/${CONFIG_NAME}.yaml').read_text())
overrides, missing = [], []
for section in ('dataloader_train', 'dataloader_val'):
    paths = (config.get(section) or {}).get('dataset', {}).get('dataset_path')
    if not paths:
        continue
    if isinstance(paths, str):
        paths = [paths]
    remapped = []
    for path in paths:
        name = Path(path).name
        container_path = Path('/data') / name
        if not (container_path / 'meta' / 'info.json').exists():
            missing.append(str(container_path))
        remapped.append(str(container_path))
    overrides.append(f\"{section}.dataset.dataset_path=[{','.join(remapped)}]\")

if missing:
    sys.stderr.write('Datasets missing under /data:\n')
    for path in missing:
        sys.stderr.write(f'  {path}\n')
    sys.stderr.write('Sync them with scripts/cluster/sync.sh data first.\n')
    raise SystemExit(1)

print(' '.join(overrides))
PY
)
        echo \"Dataset overrides: \${DATASET_OVERRIDES}\"

        if [[ \"\${CHECKPOINT}\" == /workspace/checkpoints/* && ! -f \"\${CHECKPOINT}/model/.metadata\" ]]; then
            hf download nvidia/DreamDojo \
                --include '2B_G1_post-train/iter_000050000/model/**' \
                --local-dir /workspace/checkpoints/DreamDojo
        fi

        if [[ ! -f \"\${CHECKPOINT}/model/.metadata\" ]]; then
            echo \"Official checkpoint download is incomplete: \${CHECKPOINT}\" >&2
            exit 1
        fi

        NPROC='${NUM_GPUS}' MASTER_PORT='${MASTER_PORT}' \
            bash /workspace/scripts/train_wm.sh '${EXPERIMENT}' \
            checkpoint.load_path=\${CHECKPOINT} \
            checkpoint.load_training_state=${LOAD_TRAINING_STATE} \
            checkpoint.strict_resume=false \
            trainer.max_iter=${MAX_ITER} \
            job.name='${RUN_NAME}' \
            \${DATASET_OVERRIDES} \
            ${OVERRIDES}
    "
EXIT_CODE=$?
set -e

echo "=== srun exited with ${EXIT_CODE} (124 = hit CHAIN_TIMEOUT) ==="
if [[ -f "${LATEST_FILE}" ]]; then
    echo "Latest checkpoint now: $(cat "${LATEST_FILE}")"
fi

chain_resubmit "${EXIT_CODE}" \
    --job-name="${SLURM_JOB_NAME:-dd_${RUN_NAME}}" \
    --gpus="${NUM_GPUS}" \
    -- \
    EXPERIMENT="${EXPERIMENT}" \
    CONFIG_NAME="${CONFIG_NAME}" \
    RUN_NAME="${RUN_NAME}" \
    MAX_ITER="${MAX_ITER}" \
    NUM_GPUS="${NUM_GPUS}" \
    LOAD_TRAINING_STATE="${LOAD_TRAINING_STATE}" \
    ${OVERRIDES_FILE:+OVERRIDES_FILE="${OVERRIDES_FILE}"}
