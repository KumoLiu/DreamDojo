#!/usr/bin/env bash
set -euo pipefail

# Override these variables when your username, data-copier node, or portfolio
# path differs:
#   REMOTE_HOST=user@host REMOTE_BASE=/scratch/... bash sync_to_slurm.sh
REMOTE_HOST="${REMOTE_HOST:-yunl@nb-hel-cs-001-dc-01}"
REMOTE_BASE="${REMOTE_BASE:-/lustre/fsw/portfolios/healthcareeng/users/yunl}"
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Set SYNC_CODE=1 to also copy the DreamDojo source tree. Large/generated
# directories and the local virtual environment are excluded.
SYNC_CODE="${SYNC_CODE:-0}"

DATASETS=(
    # "g1_pick_trocar_rollout_all_260_headcam_train"
    # "g1_pick_trocar_rollout_all_260_headcam_eval_5s5f"
)

CHECKPOINTS=(
    # "g1_pick_trocar_headcam_2b_iter_000010000_ema_both"
)

echo "Remote: ${REMOTE_HOST}:${REMOTE_BASE}"
echo "Creating destination directories..."
ssh "${REMOTE_HOST}" \
    "mkdir -p '${REMOTE_BASE}/datasets' '${REMOTE_BASE}/checkpoints' '${REMOTE_BASE}/code'"

for dataset in "${DATASETS[@]}"; do
    source_path="${LOCAL_ROOT}/datasets/${dataset}"
    destination="${REMOTE_HOST}:${REMOTE_BASE}/datasets/${dataset}/"

    if [[ ! -f "${source_path}/meta/info.json" ]]; then
        echo "Missing LeRobot dataset: ${source_path}" >&2
        exit 1
    fi

    echo
    echo "Syncing dataset: ${dataset}"
    # -L is required because the prepared DreamDojo datasets contain video
    # symlinks that point to local source datasets.
    rsync -avhL \
        --partial \
        --info=progress2 \
        "${source_path}/" \
        "${destination}"
done

for checkpoint in "${CHECKPOINTS[@]}"; do
    source_path="${LOCAL_ROOT}/migrated_checkpoints/${checkpoint}"
    destination="${REMOTE_HOST}:${REMOTE_BASE}/checkpoints/${checkpoint}/"

    if [[ ! -f "${source_path}/model/.metadata" ]]; then
        echo "Missing DCP checkpoint: ${source_path}" >&2
        exit 1
    fi

    echo
    echo "Syncing checkpoint: ${checkpoint}"
    rsync -avhL \
        --partial \
        --info=progress2 \
        "${source_path}/" \
        "${destination}"
done

if [[ "${SYNC_CODE}" == "1" ]]; then
    echo
    echo "Syncing DreamDojo source..."
    rsync -avh \
        --partial \
        --info=progress2 \
        --exclude="/.venv/" \
        --exclude="/datasets/" \
        --exclude="/checkpoints/" \
        --exclude="/migrated_checkpoints/" \
        --exclude="/docker/" \
        --exclude="/outputs/" \
        --exclude="/results/" \
        --exclude="__pycache__/" \
        "${LOCAL_ROOT}/" \
        "${REMOTE_HOST}:${REMOTE_BASE}/code/DreamDojo/"
fi

echo
echo "Verifying remote datasets and checkpoints..."
for dataset in "${DATASETS[@]}"; do
    ssh "${REMOTE_HOST}" "
        test -f '${REMOTE_BASE}/datasets/${dataset}/meta/info.json'
        du -sh '${REMOTE_BASE}/datasets/${dataset}'
    "
done

for checkpoint in "${CHECKPOINTS[@]}"; do
    ssh "${REMOTE_HOST}" "
        test -f '${REMOTE_BASE}/checkpoints/${checkpoint}/model/.metadata'
        du -sh '${REMOTE_BASE}/checkpoints/${checkpoint}'
    "
done

echo "Sync completed."
