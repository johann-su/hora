#!/bin/bash
# Install hora's extra pip dependencies. RUNS ON THE CLUSTER. One-time.
#
# The image is a read-only SIF and cannot be modified on the cluster (no root,
# no fakeroot -- see build_image.sh), so anything missing goes into a directory
# that run_in_container.sh bind-mounts and puts on PYTHONPATH instead.
#
# Almost everything hora needs is already in the image: hydra-core, omegaconf,
# tensorboard and torch all arrive with Isaac Sim / IsaacLab. Mirrors the note in
# environment_isaaclab.yaml -- termcolor is the one genuine gap.
#
# Installed with Isaac Sim's own interpreter so the wheels match its Python
# version and ABI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster.env"

PACKAGES=(termcolor)

[ -f "${CLUSTER_SIF}" ] || { echo "[ERROR] no image at ${CLUSTER_SIF} -- run scripts/hpc/build_image.sh --submit first" >&2; exit 1; }

mkdir -p "${CLUSTER_PYDEPS_DIR}"
# --containall gives no home, and pip wants a writable one even with
# --no-cache-dir (for its temporary build directories).
SCRATCH_HOME=$(mktemp -d)
trap 'rm -rf "${SCRATCH_HOME}"' EXIT

# Bind the target in directly rather than going through run_in_container.sh:
# this runs before the deps exist, and needs no GPU or cache setup.
run_in_image() {
    apptainer exec \
        --containall --writable-tmpfs \
        --home "${SCRATCH_HOME}:/root" \
        -B "${CLUSTER_PYDEPS_DIR}:${CONTAINER_PYDEPS}" \
        --env "PIP_DISABLE_PIP_VERSION_CHECK=1" \
        --env "PYTHONPATH=${CONTAINER_PYDEPS}" \
        "${CLUSTER_SIF}" "$@"
}

echo "[INFO] installing ${PACKAGES[*]} -> ${CLUSTER_PYDEPS_DIR}"
run_in_image "${CONTAINER_PYTHON}" -m pip install --no-cache-dir \
    --target "${CONTAINER_PYDEPS}" --upgrade "${PACKAGES[@]}"

echo "[INFO] verifying hora's imports resolve inside the container:"
# Versions come from package metadata, not module.__version__ -- termcolor 3.x
# no longer defines that attribute. Double quotes only, and %-formatting rather
# than f-strings: the whole snippet is a single-quoted shell argument, and the
# container's Python 3.11 rejects backslashes inside f-string expressions.
run_in_image "${CONTAINER_PYTHON}" -c '
import importlib, importlib.metadata as md
dists = {"termcolor": "termcolor", "hydra": "hydra-core", "omegaconf": "omegaconf",
         "tensorboard": "tensorboard", "torch": "torch"}
for mod, dist in dists.items():
    importlib.import_module(mod)
    print("%-12s %s" % (mod, md.version(dist)))
import torch
print("%-12s %s" % ("cuda", torch.version.cuda))
'
echo "[INFO] done"
