#!/bin/bash
# Run a command inside the IsaacLab container. RUNS ON THE CLUSTER (compute node).
#
# Used by submit.sh as the job body, and directly for interactive debugging:
#   ./scripts/hpc/run_in_container.sh python train.py task=AllegroHandHora headless=True ...
#   ./scripts/hpc/run_in_container.sh bash          # poke around inside
#
# `python` as the first word is rewritten to Isaac Sim's own interpreter, which
# is the only one in the image that can import isaacsim/isaaclab.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster.env"

[ -f "${CLUSTER_SIF}" ] || { echo "[ERROR] no image at ${CLUSTER_SIF} -- run scripts/hpc/build_image.sh" >&2; exit 1; }
[ $# -ge 1 ] || { echo "usage: $0 <command...>" >&2; exit 1; }

# --- caches --------------------------------------------------------------------
# Isaac Sim's shader/kit caches are many small files written continuously. On
# Lustre that is both slow and rude to other users, so the live copies sit on
# node-local disk and are only synced back at the end. A warm cache is the
# difference between ~5 min and <1 min of startup.
#
# `home/` is bound as the container's /root, which is where most of them live
# (~/.cache/ov, ~/.cache/nvidia/GLCache, ~/.nv/ComputeCache, ~/.local/share/ov).
# Binding /root as a single unit rather than one bind per cache -- the way
# IsaacLab's run_singularity.sh does it -- avoids any question of whether an
# individual bind under /root is shadowed by the home mount itself.
NODE_SCRATCH="${NODE_SCRATCH:-/tmp/${USER}-hora-${SLURM_JOB_ID:-interactive}}"
NODE_CACHE="${NODE_SCRATCH}/isaac-sim-cache"
mkdir -p "${NODE_CACHE}"/{kit,kit-logs,home,tmp}

if [ -d "${CLUSTER_CACHE_DIR}" ]; then
    echo "[INFO] seeding node-local cache from ${CLUSTER_CACHE_DIR}"
    rsync -a "${CLUSTER_CACHE_DIR}/" "${NODE_CACHE}/" || true
fi

# --- image staging --------------------------------------------------------------
# Isaac Sim opens thousands of files across its extensions at startup. Reading
# those from the SIF over Lustre is noticeably slower than one bulk copy of the
# image to local disk, so stage by default. STAGE_SIF=0 to read from Lustre.
SIF_TO_RUN="${CLUSTER_SIF}"
if [ "${STAGE_SIF:-1}" = "1" ]; then
    SIF_TO_RUN="${NODE_SCRATCH}/$(basename "${CLUSTER_SIF}")"
    if [ ! -f "${SIF_TO_RUN}" ]; then
        echo "[INFO] staging image to node-local disk ($(du -h "${CLUSTER_SIF}" | cut -f1))"
        cp "${CLUSTER_SIF}" "${SIF_TO_RUN}"
    fi
fi

# --- bind mounts -----------------------------------------------------------------
# The image is read-only, so every path Isaac Sim writes to must be a bind. The
# list mirrors the named volumes in IsaacLab's docker-compose.yaml. Binding
# /root wholesale (via --home) covers the ~/.cache/* and ~/.local/* ones in one
# go; kit/cache and kit/logs live outside $HOME and need their own binds.
BINDS=(
    -B "${CLUSTER_HORA_DIR}:${CONTAINER_HORA}"
    -B "${NODE_CACHE}/kit:${ISAACSIM_ROOT}/kit/cache"
    -B "${NODE_CACHE}/kit-logs:${ISAACSIM_ROOT}/kit/logs"
    -B "${NODE_CACHE}/tmp:/tmp"
)
# `if`, not `[ -d ... ] && ...`: under `set -e` a standalone && list whose test
# fails takes the whole script down with it.
if [ -d "${CLUSTER_PYDEPS_DIR}" ]; then
    BINDS+=(-B "${CLUSTER_PYDEPS_DIR}:${CONTAINER_PYDEPS}")
else
    echo "[WARN] ${CLUSTER_PYDEPS_DIR} missing -- run scripts/hpc/setup_deps.sh, or"
    echo "[WARN] training will fail on 'ModuleNotFoundError: termcolor'."
fi

# --containall keeps the host environment and filesystem out, so runs are
# reproducible and do not depend on the submitting shell. --nv maps the driver
# in over the placeholder /bin/nvidia-* files the image ships for this purpose.
APPTAINER_ARGS=(
    --nv --containall
    --home "${NODE_CACHE}/home:/root"
    --pwd "${CONTAINER_HORA}"
    --env "PYTHONPATH=${CONTAINER_PYDEPS}"
    # shim/python first, so scripts/train_s1.sh and friends resolve `python` to
    # Isaac Sim's interpreter (see shim/python for why the image's alias fails).
    --env "PATH=${CONTAINER_HORA}/scripts/hpc/shim:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    --env "ISAACLAB_PATH=/workspace/isaaclab"
    --env "OMNI_KIT_ACCEPT_EULA=YES"
    --env "ACCEPT_EULA=Y"
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
)

# Isaac Sim insists on writing a few paths that are not worth binding
# individually (kit registry scratch, /var/tmp). A small tmpfs overlay absorbs
# them; nothing that must survive the job goes there.
APPTAINER_ARGS+=(--writable-tmpfs)

# `python foo.py` -> `/isaac-sim/python.sh foo.py`
if [ "$1" = "python" ] || [ "$1" = "python3" ]; then
    shift
    set -- "${CONTAINER_PYTHON}" "$@"
fi

echo "[INFO] image   : ${SIF_TO_RUN}"
echo "[INFO] workdir : ${CONTAINER_HORA} (= ${CLUSTER_HORA_DIR})"
echo "[INFO] command : $*"

set +e
apptainer exec "${APPTAINER_ARGS[@]}" "${BINDS[@]}" "${SIF_TO_RUN}" "$@"
rc=$?
set -e

# --- persist caches ---------------------------------------------------------------
# Best-effort: a failed sync must not turn a finished training run into a failed
# job. home/ *is* synced -- the shader and Omniverse caches that make the next
# start fast live inside it. Only genuinely throwaway paths are excluded.
echo "[INFO] syncing Isaac Sim caches back to ${CLUSTER_CACHE_DIR}"
mkdir -p "${CLUSTER_CACHE_DIR}"
rsync -a \
    --exclude=tmp/ \
    --exclude=kit-logs/ \
    --exclude='home/.nvidia-omniverse/logs/' \
    --exclude='home/.cache/pip/' \
    "${NODE_CACHE}/" "${CLUSTER_CACHE_DIR}/" || echo "[WARN] cache sync-back failed (ignored)"

echo "[INFO] exit code ${rc}"
exit ${rc}
