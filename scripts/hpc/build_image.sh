#!/bin/bash
# Create the Apptainer image for training. RUNS ON THE CLUSTER.
#
#   ./scripts/hpc/build_image.sh --submit    # as a batch job (recommended)
#   ./scripts/hpc/build_image.sh             # here and now (needs a big allocation)
#
# --- Why not IsaacLab's `docker/cluster/cluster_interface.sh push` -------------
#
# That flow runs `docker compose build` on your workstation (Dockerfile.base:
# FROM nvcr.io/nvidia/isaac-sim, COPY the IsaacLab tree in, apt-get, then
# `isaaclab.sh --install`), converts the result to a SIF with apptainer, tars it
# and scp's ~13 GB to the cluster. It needs docker AND apptainer installed
# locally plus tens of GB of free disk, and every image change repeats the upload.
#
# None of that is necessary here: NVIDIA publishes the *result* of that
# Dockerfile as nvcr.io/nvidia/isaac-lab, anonymously pullable, so we pull it
# straight onto the cluster.
#
# --- Doesn't ZIH forbid building containers on the cluster? -------------------
#
# The rule is about builds that need root, i.e. that run a %post section. This
# is a pure format conversion (OCI layers -> squashfs), which is unprivileged
# and explicitly blessed by the compendium. A def-file build really does not
# work here; verified on c114:
#
#   INFO:    User not listed in /etc/subuid, trying root-mapped namespace
#   FATAL:   exec /.singularity.d/libs/fakeroot failed
#
# That is why nothing hora-specific is baked in: the image is used exactly as
# NVIDIA publishes it, and hora's code, assets and extra pip packages are
# bind-mounted at runtime (run_in_container.sh, setup_deps.sh).
#
# If you ever do need a modified image, build it with docker where you have
# root, and import the result -- still no build on the cluster:
#   docker save my-isaac-lab:latest -o my-isaac-lab.tar
#   rsync -P my-isaac-lab.tar <login>:$CLUSTER_SIF_DIR/
#   apptainer build my-isaac-lab.sif docker-archive://my-isaac-lab.tar
#
# --- Resources ----------------------------------------------------------------
#
# Unpacking ~20 GB of OCI layers and running mksquashfs over them needs real
# memory. Do NOT run this in a small interactive allocation: an `srun ... --mem=13G`
# shell OOM-kills the build, and apptainer does not fail loudly when that
# happens -- it writes a SIF with an *empty* filesystem partition and still
# prints "Build complete". Symptom of that failure:
#
#   $ apptainer inspect isaac-lab-2.3.1.sif
#   FATAL: ... failed to read SIF partition at offset 40960: EOF
#   $ ls -la isaac-lab-2.3.1.sif        # ~38 KB instead of ~9 GB
#
# verify_image() below turns that silent corruption into a hard error. Note that
# `ssh <compute node>` is adopted into that node's running job cgroup, so a
# build started over ssh inherits the interactive job's memory limit too.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster.env"

BUILD_CPUS="${BUILD_CPUS:-8}"
BUILD_MEM="${BUILD_MEM:-64G}"
BUILD_TIME="${BUILD_TIME:-02:00:00}"

# --- batch mode ----------------------------------------------------------------
if [ "${1:-}" = "--submit" ]; then
    LOG_DIR="${CLUSTER_WS}/slurm-logs"
    mkdir -p "${LOG_DIR}"
    job_script=$(mktemp)
    cat > "${job_script}" <<EOT
#!/bin/bash
#SBATCH --job-name=hora-build-image
#SBATCH --account=${HORA_ACCOUNT}
#SBATCH --partition=${HORA_PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${BUILD_CPUS}
#SBATCH --mem=${BUILD_MEM}
#SBATCH --time=${BUILD_TIME}
#SBATCH --output=${LOG_DIR}/%x-%j.out
#SBATCH --error=${LOG_DIR}/%x-%j.out
bash "${SCRIPT_DIR}/build_image.sh"
EOT
    echo "[INFO] submitting image build (${BUILD_CPUS} cpu, ${BUILD_MEM}, ${BUILD_TIME})"
    echo "[INFO] logs: ${LOG_DIR}/hora-build-image-<jobid>.out"
    sbatch "${job_script}"
    rm -f "${job_script}"
    exit 0
fi

# --- direct mode ---------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "[ERROR] Not in a SLURM allocation. Pulling ~8 GB and building a squashfs" >&2
    echo "[ERROR] on a login node is antisocial and will likely be killed." >&2
    echo "[ERROR] Use: $0 --submit" >&2
    exit 1
fi

# Catch the small-allocation case up front rather than 20 minutes in.
mem_max=$(cat "/sys/fs/cgroup$(awk -F: '/^0::/{print $3}' /proc/self/cgroup)/memory.max" 2>/dev/null || echo max)
if [ "${mem_max}" != "max" ] && [ "${mem_max}" -lt $((32 * 1024 * 1024 * 1024)) ]; then
    echo "[ERROR] This allocation caps memory at $((mem_max / 1024 / 1024 / 1024)) GB; the build needs >= 32 GB." >&2
    echo "[ERROR] It would be OOM-killed and produce a silently empty image. Use: $0 --submit" >&2
    exit 1
fi

# Blob cache and build scratch both want fast local disk with tens of GB free.
# Compute nodes have ~800 GB on /tmp; the workspace is Lustre, which is far
# slower for the many small files an OCI extract produces.
SCRATCH="${APPTAINER_SCRATCH:-/tmp/${USER}-apptainer}"
export APPTAINER_CACHEDIR="${SCRATCH}/cache"
export APPTAINER_TMPDIR="${SCRATCH}/tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${CLUSTER_SIF_DIR}"

verify_image() {
    local sif="$1"
    local size
    size=$(stat -c %s "${sif}")
    if [ "${size}" -lt $((1024 * 1024 * 1024)) ]; then
        echo "[ERROR] ${sif} is only $((size / 1024)) KB -- the build was killed (see header)." >&2
        return 1
    fi
    apptainer inspect "${sif}" > /dev/null || { echo "[ERROR] ${sif} is not a readable image." >&2; return 1; }
    apptainer exec "${sif}" test -x /isaac-sim/python.sh \
        || { echo "[ERROR] ${sif} has no /isaac-sim/python.sh -- wrong image?" >&2; return 1; }
}

# Build to node-local disk, verify, and only then publish to the workspace, so a
# failed build can never leave a corrupt image where jobs will pick it up.
STAGED="${SCRATCH}/$(basename "${CLUSTER_SIF}")"
echo "[INFO] image  : ${ISAACLAB_IMAGE}"
echo "[INFO] staging: ${STAGED}"
echo "[INFO] target : ${CLUSTER_SIF}"
echo "[INFO] This takes ~15-25 min (pull ~8 GB, extract, mksquashfs)."

apptainer build --force "${STAGED}" "docker://${ISAACLAB_IMAGE}"
verify_image "${STAGED}"

echo "[INFO] verified, publishing to workspace ($(du -h "${STAGED}" | cut -f1))"
mkdir -p "${CLUSTER_SIF_DIR}"
cp "${STAGED}" "${CLUSTER_SIF}.tmp"
mv "${CLUSTER_SIF}.tmp" "${CLUSTER_SIF}"
verify_image "${CLUSTER_SIF}"

echo "[INFO] Done: ${CLUSTER_SIF} ($(du -h "${CLUSTER_SIF}" | cut -f1))"
echo "[INFO] Scratch left at ${SCRATCH}; it is node-local and safe to delete."
