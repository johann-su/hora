#!/bin/bash
# Generate a grasp-pose cache for one object scale.
#
# CACHE can be some existing output folder, does not matter
# episodeLen=50 to save time
# see the object, check whether it's simple tennis ball or fancy balls
# mass should be about 50g
#
# Contact sensing runs on GPU under IsaacLab, so the old "pipeline needs to be cpu to get
# the pairwise contact" and "no custom PD because bug in CPU mode" constraints are gone.
#
# numEnvs is 2048 rather than the original 20000: grasp generation cannot use the
# replicated fast path (filtered contacts need replicate_physics=False), and 20000
# heterogeneous envs will not fit on a 12 GB card. Raise it if you have the memory.
# Expect roughly 45-60 minutes per scale to reach 50k poses.
#
# Run once per entry in task.env.randomization.randomizeScaleList:
#   for s in 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86; do ./scripts/gen_grasp.sh 0 $s; done
# One process per scale is required -- a SimulationContext cannot be rebuilt in-process.
if [ $# -lt 2 ]; then
    echo "usage: $0 <gpus> <scale>"
    echo "  e.g.  $0 0 0.8"
    echo
    echo "All 2 arguments are required. In particular an empty <gpus> expands to"
    echo "CUDA_VISIBLE_DEVICES= , which hides every GPU -- Isaac Sim then fails with"
    echo "'no CUDA-capable device is detected', which looks like a broken driver but is not."
    exit 1
fi


GPUS=$1
SCALE=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

CUDA_VISIBLE_DEVICES=${GPUS} \
python gen_grasp.py task=AllegroHandGrasp headless=True \
task.env.numEnvs=2048 test=True \
task.env.controller.controlFrequencyInv=8 task.env.episodeLength=50 \
task.env.controller.torque_control=False task.env.genGrasps=True task.env.baseObjScale="${SCALE}" \
task.env.object.type=simple_tennis_ball \
task.env.randomization.randomizeMass=True task.env.randomization.randomizeMassLower=0.05 task.env.randomization.randomizeMassUpper=0.051 \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=False \
train.ppo.priv_info=True \
${EXTRA_ARGS}
