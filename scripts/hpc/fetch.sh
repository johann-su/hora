#!/bin/bash
# Pull training results back from the cluster. RUNS LOCALLY.
#
#   ./scripts/hpc/fetch.sh                 # everything under outputs/
#   ./scripts/hpc/fetch.sh hora_v1         # just outputs/AllegroHandHora/hora_v1
#   ./scripts/hpc/fetch.sh --cache         # the generated grasp caches instead
#   ./scripts/hpc/fetch.sh --logs          # SLURM job logs
#
# Never deletes on either side: a partially finished run on the cluster should
# not wipe local checkpoints, and vice versa.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/cluster.env"

RSYNC_ARGS=(-rlptzh)
if [ -t 1 ]; then RSYNC_ARGS+=(--info=progress2); else RSYNC_ARGS+=(--info=stats1); fi

case "${1:-}" in
    --cache)
        echo "[INFO] fetching grasp caches -> ${REPO_ROOT}/cache/"
        mkdir -p "${REPO_ROOT}/cache"
        rsync "${RSYNC_ARGS[@]}" "${CLUSTER_LOGIN}:${CLUSTER_HORA_DIR}/cache/" "${REPO_ROOT}/cache/"
        ;;
    --logs)
        echo "[INFO] fetching SLURM logs -> ${REPO_ROOT}/outputs/slurm-logs/"
        mkdir -p "${REPO_ROOT}/outputs/slurm-logs"
        rsync "${RSYNC_ARGS[@]}" "${CLUSTER_LOGIN}:${CLUSTER_WS}/slurm-logs/" "${REPO_ROOT}/outputs/slurm-logs/"
        ;;
    '')
        echo "[INFO] fetching all outputs -> ${REPO_ROOT}/outputs/"
        mkdir -p "${REPO_ROOT}/outputs"
        rsync "${RSYNC_ARGS[@]}" "${CLUSTER_LOGIN}:${CLUSTER_HORA_DIR}/outputs/" "${REPO_ROOT}/outputs/"
        ;;
    *)
        run="$1"
        echo "[INFO] fetching run '${run}' -> ${REPO_ROOT}/outputs/AllegroHandHora/${run}/"
        mkdir -p "${REPO_ROOT}/outputs/AllegroHandHora/${run}"
        rsync "${RSYNC_ARGS[@]}" \
            "${CLUSTER_LOGIN}:${CLUSTER_HORA_DIR}/outputs/AllegroHandHora/${run}/" \
            "${REPO_ROOT}/outputs/AllegroHandHora/${run}/"
        ;;
esac
echo "[INFO] done"
