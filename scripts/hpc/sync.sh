#!/bin/bash
# Push the hora working tree to the cluster workspace. RUNS LOCALLY.
#
# Deliberately rsync of the working tree rather than a git clone on the cluster:
# assets/usd/ and cache/*.npy are gitignored but are exactly what training needs,
# and they are small (~55 MB together). It also means uncommitted local changes
# are what actually runs, which is usually what you want while iterating.
#
#   ./scripts/hpc/sync.sh                 # code + assets + grasp cache
#   ./scripts/hpc/sync.sh --with-outputs  # also push outputs/ (stage-1 ckpts for stage 2)
#   ./scripts/hpc/sync.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/cluster.env"

# A live progress bar is only useful on a terminal; under sbatch/CI it produces
# tens of thousands of carriage-return lines in the log.
RSYNC_ARGS=(-rlptzh)
if [ -t 1 ]; then RSYNC_ARGS+=(--info=progress2); else RSYNC_ARGS+=(--info=stats1); fi
WITH_OUTPUTS=0
for arg in "$@"; do
    case "$arg" in
        --with-outputs) WITH_OUTPUTS=1 ;;
        --dry-run)      RSYNC_ARGS+=(--dry-run -v) ;;
        --delete)       RSYNC_ARGS+=(--delete) ;;
        *) echo "unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ros2_allegro is the hardware-deployment side and never runs on the cluster.
# build/, install/, log/ are colcon output belonging to that same side.
#
# .git IS synced (~100 MB, and only once -- rsync sends deltas after that):
# train.py stamps each run with `git_hash()` and writes `git diff HEAD` to
# outputs/<run>/gitdiff.patch. Both shell out to git and raise, killing the run,
# if the tree is not a repository. .devcontainer/ is tracked, so it is synced too
# rather than showing up as a spurious deletion in that patch.
EXCLUDES=(
    --exclude='*.pyc' --exclude=__pycache__/
    --exclude=ros2_allegro/
    --exclude=build/ --exclude=install/ --exclude=log/
)
if [ "${WITH_OUTPUTS}" -eq 0 ]; then
    # outputs/ is written by the job on the cluster; pushing it risks clobbering
    # a run's checkpoints with stale local ones. Use fetch.sh to bring them back.
    EXCLUDES+=(--exclude=outputs/)
fi

echo "[INFO] ${REPO_ROOT}/ -> ${CLUSTER_LOGIN}:${CLUSTER_HORA_DIR}/"
ssh "${CLUSTER_LOGIN}" "mkdir -p '${CLUSTER_HORA_DIR}'"
rsync "${RSYNC_ARGS[@]}" "${EXCLUDES[@]}" \
    "${REPO_ROOT}/" "${CLUSTER_LOGIN}:${CLUSTER_HORA_DIR}/"

# Warn about the two gitignored inputs whose absence only surfaces minutes into
# a job, after Isaac Sim has already started.
if [ ! -d "${REPO_ROOT}/assets/usd" ]; then
    echo "[WARN] assets/usd/ missing locally -- run ./scripts/convert_assets.sh first,"
    echo "[WARN] or the job will fail with 'converted USD not found'."
fi
if ! compgen -G "${REPO_ROOT}/cache/*.npy" > /dev/null; then
    echo "[WARN] cache/*.npy missing locally -- stage 1 needs a grasp cache per scale."
    echo "[WARN] Generate on the cluster with: scripts/hpc/submit.sh gen_grasp <scale>"
fi
echo "[INFO] done"
