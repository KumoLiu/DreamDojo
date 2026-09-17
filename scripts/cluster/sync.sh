#!/usr/bin/env bash
# Code and the six final adapted datasets are separate, explicit sync actions.
set -euo pipefail
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REMOTE_HOST="${REMOTE_HOST:-nvidia-cluster}"
REMOTE_BASE="${REMOTE_BASE:-/lustre/fsw/portfolios/healthcareeng/users/yunl}"
action="${1:---help}"
case "$action" in
    code|data|all) ;;
    --help|-h)
        echo 'Usage: [DRY_RUN=1] bash scripts/cluster/sync.sh {code|data|all}'
        echo 'Overrides: LOCAL_ROOT, REMOTE_HOST, REMOTE_BASE'
        echo 'No remote deletions; no checkpoint transfer. DRY_RUN=1 makes no connection.'
        exit 0 ;;
    *) echo "Unknown action: $action" >&2; exit 2 ;;
esac

run() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "$@"; printf '\n'
    else
        "$@"
    fi
}

if [[ "$action" == code || "$action" == all ]]; then
    printf -v code_dir_q '%q' "${REMOTE_BASE}/code/DreamDojo"
    run ssh "$REMOTE_HOST" "mkdir -p $code_dir_q"
    run rsync -avh --protect-args --partial --info=progress2 \
        --exclude='/.git/' --exclude='/.venv/' --exclude='/datasets/' \
        --exclude='/checkpoints/' --exclude='/migrated_checkpoints/' \
        --exclude='*.sqsh' --exclude='/outputs/' --exclude='/results/' \
        --exclude='/logs/' --exclude='/train/' --exclude='/.cursor/' \
        --exclude='__pycache__/' --exclude='/.pytest_cache/' --exclude='/.ruff_cache/' \
        "$LOCAL_ROOT/" "${REMOTE_HOST}:${REMOTE_BASE}/code/DreamDojo/"
fi

if [[ "$action" == data || "$action" == all ]]; then
    datasets=(
        g1_hf_pick_trocar_teleop_success_train
        g1_hf_pick_trocar_rollouts_30k_train
        g1_hf_pick_trocar_rollouts_10k_train
        g1_hf_pick_trocar_teleop_success_val
        g1_hf_pick_trocar_rollouts_30k_val
        g1_hf_pick_trocar_rollouts_10k_val
    )
    for dataset in "${datasets[@]}"; do
        [[ -f "$LOCAL_ROOT/datasets/$dataset/meta/info.json" ]] || {
            echo "Missing dataset: $LOCAL_ROOT/datasets/$dataset" >&2; exit 1;
        }
    done
    printf -v data_dir_q '%q' "${REMOTE_BASE}/datasets"
    run ssh "$REMOTE_HOST" "mkdir -p $data_dir_q"
    for dataset in "${datasets[@]}"; do
        # Adapted datasets use local video symlinks; copy their actual contents.
        run rsync -avhL --protect-args --partial --info=progress2 \
            "$LOCAL_ROOT/datasets/$dataset/" \
            "${REMOTE_HOST}:${REMOTE_BASE}/datasets/$dataset/"
    done
fi
