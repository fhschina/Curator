#!/usr/bin/env bash
#SBATCH --job-name=dedup-h100
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=256G
#SBATCH --time=01:00:00

set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${V07_REPO_DIR:?Set the absolute repository path}"
: "${PYTHON_BIN:?Set the Python executable in the existing environment}"
: "${V07_RUN_ROOT:?Set the prepared run directory}"
MODE="${1:-smoke}"
case "$MODE" in
    smoke) run_args=(--smoke-only) ;;
    full) run_args=() ;;
    *) printf 'Usage: sbatch run.slurm [smoke|full]\n' >&2; exit 2 ;;
esac
# Frozen physical GPU indices require the same whole-node allocation on resume.
if [[ "${SLURM_GPUS_ON_NODE:-}" != 8 || "${SLURM_JOB_NUM_NODES:-}" != 1 ]]; then
    printf 'This example requires one whole node with eight allocated GPUs.\n' >&2
    exit 2
fi
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/tmp/dedup-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
cd "$V07_REPO_DIR"
test -f "$V07_RUN_ROOT/manifest.json"
RUNTIME_ROOT=$("$PYTHON_BIN" -c 'import json, sys; print(json.load(open(sys.argv[1]))["local_backend"]["runtime_root"])' "$V07_RUN_ROOT/manifest.json")

archive_runtime_logs() {
    local run_exit=$?
    local archive="$V07_RUN_ROOT/recovery/slurm-logs/$SLURM_JOB_ID"
    local ray_session="$RUNTIME_ROOT/ray/session_latest"
    mkdir -p "$archive" || true
    shopt -s nullglob
    for log in "$ray_session"/nemo_curator_dynamo_*/*.log "$ray_session"/logs/runtime_env_setup*.log; do
        cp "$log" "$archive/" || true
    done
    exit "$run_exit"
}
# Preserve worker logs while node-local storage is still accessible.
trap archive_runtime_logs EXIT

printf 'Job: %s\nNode: %s\nMode: %s\nRun: %s\n' "$SLURM_JOB_ID" "$(hostname)" "$MODE" "$V07_RUN_ROOT"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
# A detached launch would let the batch script release its allocation too early.
srun --ntasks=1 "$PYTHON_BIN" -m eval.dedup run \
    --root "$V07_RUN_ROOT" \
    --session "$V07_RUN_ROOT/recovery/sessions/slurm-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}-$MODE" \
    "${run_args[@]}"
"$PYTHON_BIN" -m eval.dedup status --root "$V07_RUN_ROOT"
if [[ "$MODE" == full ]]; then
    "$PYTHON_BIN" -m eval.dedup audit --root "$V07_RUN_ROOT"
fi
