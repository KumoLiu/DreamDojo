#!/usr/bin/env bash
# Report the state of a chained training job on the cluster.
#
# Pulls the four things that actually tell you whether a chain is healthy: the
# queue (is the next link waiting?), the job's own state and exit code, the tail
# of the SLURM log (did the container start, did it hit the timeout, did it
# resubmit), and the run's checkpoint/iteration progress.
#
#   bash scripts/check_cluster_job.sh 1006384
#   bash scripts/check_cluster_job.sh            # newest dd_ job in the queue
#   WATCH=1 bash scripts/check_cluster_job.sh     # re-check every 5 min
set -uo pipefail

REMOTE_HOST="${REMOTE_HOST:-nvidia-cluster}"
REMOTE_BASE="${REMOTE_BASE:-/lustre/fsw/portfolios/healthcareeng/users/yunl}"
LOG_DIR="${LOG_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/dreamdojo}"

JOB_ID="${1:-${JOB_ID:-}}"
RUN_NAME="${RUN_NAME:-lora_r32_lr1e-4_long9k}"
JOB_GROUP="${JOB_GROUP:-hf_teleop_rollout_posttrain_lora}"
MAX_ITER="${MAX_ITER:-9000}"
INTERVAL="${INTERVAL:-300}"

while :; do
    ssh "${REMOTE_HOST}" bash -s -- \
        "${JOB_ID}" "${RUN_NAME}" "${JOB_GROUP}" "${MAX_ITER}" \
        "${REMOTE_BASE}" "${LOG_DIR}" <<'REMOTE'
set -uo pipefail
JOB_ID="$1"; RUN_NAME="$2"; JOB_GROUP="$3"; MAX_ITER="$4"
REMOTE_BASE="$5"; LOG_DIR="$6"
RUN_DIR="${REMOTE_BASE}/outputs/train/dreamdojo/${JOB_GROUP}/${RUN_NAME}"

echo "########## $(date -Is) ##########"

echo
echo "=== queue ==="
squeue -u "$USER" -o '%.10i %.30j %.9T %.11M %.11l %.19V %R'

# Without an explicit id, take the most recently submitted dd_ job so the rest
# of the report follows the chain as it advances.
if [[ -z "$JOB_ID" ]]; then
    JOB_ID=$(squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /^dd_/ {print $1}' | tail -1)
    echo "(no job id given; using ${JOB_ID:-none found})"
fi

echo
echo "=== job ${JOB_ID} ==="
if [[ -n "$JOB_ID" ]]; then
    sacct -j "$JOB_ID" --format=JobID%15,JobName%30,State%14,Elapsed%10,Timelimit%10,ExitCode%8,NodeList%14 2>/dev/null ||
        scontrol show job "$JOB_ID" 2>&1 | head -20
fi

echo
echo "=== checkpoint progress ==="
LATEST="${RUN_DIR}/checkpoints/latest_checkpoint.txt"
if [[ -f "$LATEST" ]]; then
    ITER=$(( 10#$(tr -dc '0-9' < "$LATEST") ))
    echo "  latest_checkpoint.txt : $(cat "$LATEST")  -> iteration ${ITER} / ${MAX_ITER}"
    echo "  saved checkpoints     : $(ls -1 "${RUN_DIR}/checkpoints" | grep -c '^iter_')"
    ls -1t "${RUN_DIR}/checkpoints" | grep '^iter_' | head -5 | sed 's/^/      /'
    df -h "${RUN_DIR}" 2>/dev/null | tail -1 | sed 's/^/  disk: /'
else
    echo "  no latest_checkpoint.txt yet at ${LATEST}"
fi

echo
echo "=== training progress (last 5 logged iterations) ==="
DEBUG="${RUN_DIR}/debug.log"
if [[ -f "$DEBUG" ]]; then
    grep -oE '\] [0-9]+ : iter_speed [0-9.]+ seconds per iteration.*Loss: [0-9.]+' "$DEBUG" |
        tail -5 | sed 's/^/  /'
    # Rough ETA from the mean speed over the last 50 logged points.
    grep -oE 'iter_speed [0-9.]+' "$DEBUG" | tail -50 | awk -v it="${MAX_ITER}" '
        {sum += $2; n++}
        END {if (n) printf "  mean %.2f s/iter over last %d points\n", sum/n, n}'
else
    echo "  no debug.log yet (container still starting, or run dir differs)"
fi

echo
echo "=== chain / error signals in slurm logs ==="
for suffix in out err; do
    f=$(ls -t "${LOG_DIR}"/*-"${JOB_ID}"."${suffix}" 2>/dev/null | head -1)
    [[ -z "$f" ]] && continue
    echo "  --- $(basename "$f") ---"
    grep -nE 'Resubmitting|Timeout reached|completed successfully|failed with exit|srun exited|Resuming from iteration|nothing to do|Traceback|CUDA|error:|Error:|slurmstepd' "$f" |
        tail -12 | sed 's/^/    /'
    echo "  --- tail ---"
    tail -12 "$f" | sed 's/^/    /'
done
REMOTE

    [[ -z "${WATCH:-}" ]] && break
    echo
    echo "sleeping ${INTERVAL}s (WATCH=1); ctrl-c to stop"
    sleep "${INTERVAL}"
done
