#!/bin/bash
# Reusable Slurm job-chaining library.
#
# Source this near the top of a .slurm script to get automatic chain tracking
# via the first job's SLURM_JOB_ID (CHAIN_ID). Each chain gets its own
# checkpoint namespace so concurrent runs of the same EXP_NAME cannot collide.
#
# Provides:
#   CHAIN_ID              — first job's SLURM_JOB_ID, stable across the chain
#   RUN_COUNT / MAX_RUNS  — progress tracking
#   chain_should_resume   — returns 0 (true) on run 2+
#   chain_resubmit        — conditionally resubmits $0 with CHAIN_ID preserved
#
# Usage in a .slurm script:
#
#   source "$(dirname "$0")/lib/chain.sh"
#
#   # ... do work ...
#   EXIT_CODE=$?
#
#   chain_resubmit "$EXIT_CODE" \
#       --job-name="${EXP_NAME}" --gpus="${NUM_GPUS}" \
#       -- \
#       EXP_NAME=${EXP_NAME} NUM_GPUS=${NUM_GPUS}
#
# DreamDojo note: the imaginaire checkpointer tracks completed saves with a
# checkpoints/latest_checkpoint.txt file rather than a `latest` directory, so
# pass no argument to chain_should_resume here -- scripts/cluster/train.slurm
# reads that file directly to decide whether the run is already finished.

# Stable chain identifier: first job's SLURM_JOB_ID, forwarded on resubmit.
CHAIN_ID="${CHAIN_ID:-${SLURM_JOB_ID}}"
RUN_COUNT="${RUN_COUNT:-1}"
MAX_RUNS="${MAX_RUNS:-6}"

# chain_should_resume [checkpoint_dir]
#
# Returns 0 (resume) iff this is run 2+ AND, if a checkpoint_dir is given,
# a `latest` symlink/dir exists inside it pointing at a complete checkpoint.
# The training entrypoint updates `latest` only after a checkpoint save fully
# finishes, so this guards against:
#   (a) run-1 crashing before the first SAVE_STEPS (no checkpoint at all), and
#   (b) run-1 being killed mid-save (a partial `checkpoint-<step>` exists but
#       `latest` was never repointed at it).
chain_should_resume() {
    [[ $RUN_COUNT -gt 1 ]] || return 1
    if [[ $# -gt 0 ]]; then
        [[ -d "$1/latest" ]] || return 1
    fi
    return 0
}

# chain_resubmit EXIT_CODE [sbatch-flags...] -- [KEY=VAL exports...]
#
# Resubmits the calling script ($0) when EXIT_CODE is 124 (timeout from the
# `timeout` wrapper) and RUN_COUNT < MAX_RUNS. CHAIN_ID and RUN_COUNT are
# forwarded automatically; callers only list experiment-specific variables.
chain_resubmit() {
    local exit_code="$1"; shift

    if [[ $exit_code == 124 ]] && [[ $RUN_COUNT -lt $MAX_RUNS ]]; then
        local next_run=$((RUN_COUNT + 1))
        echo "=== Timeout reached. Resubmitting as run ${next_run}/${MAX_RUNS} ==="

        local sbatch_args=()
        local exports="ALL,CHAIN_ID=${CHAIN_ID},RUN_COUNT=${next_run},MAX_RUNS=${MAX_RUNS}"

        while [[ $# -gt 0 ]]; do
            if [[ "$1" == "--" ]]; then
                shift; break
            fi
            sbatch_args+=("$1"); shift
        done
        for kv in "$@"; do
            exports="${exports},${kv}"
        done

        sbatch --dependency=afterany:${SLURM_JOB_ID} \
               --export="${exports}" \
               "${sbatch_args[@]}" \
               "$0"
    elif [[ $exit_code == 0 ]]; then
        echo "=== Training completed successfully ==="
    else
        echo "=== Training failed with exit code ${exit_code} ==="
    fi
}
