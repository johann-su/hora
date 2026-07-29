#!/bin/bash
# Submit a hora training job to SLURM. RUNS ON THE CLUSTER (login node).
#
#   ./scripts/hpc/submit.sh gen_grasp 0.8
#   ./scripts/hpc/submit.sh train_s1  0 hora_v1 [extra hydra overrides]
#   ./scripts/hpc/submit.sh train_s2  0 hora_v1 [extra hydra overrides]
#   ./scripts/hpc/submit.sh raw       scripts/eval_s1.sh 0 hora_v1
#
# The workload scripts under scripts/ are reused verbatim -- they remain the one
# place the hydra overrides are defined. GPU id is always 0: SLURM hands the job
# its own GPU and CUDA_VISIBLE_DEVICES is allocation-local.
#
# Resources default to one GPU's fair share of a Capella node (4x H100, 64 cores,
# 773 GB). Override per submission:
#   HORA_TIME=23:00:00 HORA_MEM=240G ./scripts/hpc/submit.sh train_s1 0 hora_v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster.env"

[ $# -ge 1 ] || { sed -n '2,17p' "$0" | sed 's/^# \?//'; exit 1; }

workload=$1; shift
case "${workload}" in
    gen_grasp)
        [ $# -ge 1 ] || { echo "usage: $0 gen_grasp <scale> [extra]" >&2; exit 1; }
        scale=$1; shift
        job_name="hora-grasp-${scale}"
        job_cmd=(bash scripts/gen_grasp.sh 0 "${scale}" "$@")
        ;;
    train_s1|train_s2)
        [ $# -ge 2 ] || { echo "usage: $0 ${workload} <seed> <output_name> [extra]" >&2; exit 1; }
        seed=$1; name=$2; shift 2
        job_name="hora-${workload}-${name}"
        job_cmd=(bash "scripts/${workload}.sh" 0 "${seed}" "${name}" "$@")
        ;;
    raw)
        [ $# -ge 1 ] || { echo "usage: $0 raw <command...>" >&2; exit 1; }
        job_name="hora-raw"
        job_cmd=("$@")
        ;;
    *)
        echo "unknown workload '${workload}' (expected gen_grasp|train_s1|train_s2|raw)" >&2
        exit 1
        ;;
esac

[ -f "${CLUSTER_SIF}" ] || { echo "[ERROR] no image at ${CLUSTER_SIF} -- run scripts/hpc/build_image.sh" >&2; exit 1; }
[ -d "${CLUSTER_HORA_DIR}" ] || { echo "[ERROR] no code at ${CLUSTER_HORA_DIR} -- run scripts/hpc/sync.sh from your workstation" >&2; exit 1; }

LOG_DIR="${CLUSTER_WS}/slurm-logs"
mkdir -p "${LOG_DIR}"

# printf %q so hydra overrides containing '=' or spaces survive the round trip
# through the heredoc into the job script.
printf -v job_cmd_str '%q ' "${job_cmd[@]}"

job_script=$(mktemp)
cat > "${job_script}" <<EOT
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${HORA_ACCOUNT}
#SBATCH --partition=${HORA_PARTITION}
#SBATCH --gres=gpu:${HORA_GPUS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${HORA_CPUS}
#SBATCH --mem=${HORA_MEM}
#SBATCH --time=${HORA_TIME}
#SBATCH --output=${LOG_DIR}/%x-%j.out
#SBATCH --error=${LOG_DIR}/%x-%j.out

set -euo pipefail
echo "[job] \$(date -Is) on \$(hostname), job \${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash "${CLUSTER_HORA_DIR}/scripts/hpc/run_in_container.sh" ${job_cmd_str}
EOT

echo "[INFO] workload : ${workload}"
echo "[INFO] command  : ${job_cmd[*]}"
echo "[INFO] resources: ${HORA_GPUS} gpu, ${HORA_CPUS} cpu, ${HORA_MEM}, ${HORA_TIME}"
echo "[INFO] logs     : ${LOG_DIR}/${job_name}-<jobid>.out"
sbatch "${job_script}"
rm -f "${job_script}"
