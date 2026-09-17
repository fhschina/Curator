#!/usr/bin/env bash
#SBATCH --job-name=dedup-prepare
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
umask 077
: "${V07_REPO_DIR:?Set the absolute repository path}"
: "${PYTHON_BIN:?Set the Python executable in the existing environment}"
: "${V07_RUN_ROOT:?Set a new run directory}"
: "${V07_RUNNER_CONFIG:?Set the generated runner config.yaml path}"
: "${CURATOR_V07_SOURCE_RUN:?Set the frozen input bundle path}"
: "${CURATOR_V07_LOCAL_MODEL_PATH:?Set the pinned FP8 checkpoint path}"
: "${CURATOR_V07_LOCAL_TOOLS_DIR:?Set the etcd and nats-server directory}"
case "${1:-full}" in
    full) prepare_args=() ;;
    smoke) prepare_args=(--smoke-only) ;;
    *) printf 'Usage: sbatch prepare.slurm [full|smoke]\n' >&2; exit 2 ;;
esac
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
cd "$V07_REPO_DIR"
"$PYTHON_BIN" -m eval.dedup prepare \
    --root "$V07_RUN_ROOT" \
    --backend local \
    --local-devices "${V07_LOCAL_DEVICES:-0}" \
    --local-runner-config "$V07_RUNNER_CONFIG" \
    "${prepare_args[@]}"
