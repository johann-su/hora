#!/bin/bash
# CACHE can be some existing output folder, does not matter
# numEnvs=20000, headless=True, episodeLen=50 to save time
# see the object, check whether it's simple tennis ball or fancy balls
# pipeline need to be cpu to get the pairwise contact
# no custom PD because bug in CPU mode
# mass should be about 50g
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
CUDA_VISIBLE_DEVICES=${GPUS} \
python gen_grasp.py task=AllegroHandGrasp headless=True pipeline=cpu \
task.env.numEnvs=20000 test=True \
task.env.controller.controlFrequencyInv=8 task.env.episodeLength=50 \
task.env.controller.torque_control=False task.env.genGrasps=True task.env.baseObjScale="${SCALE}" \
task.env.object.type=simple_tennis_ball \
task.env.randomization.randomizeMass=True task.env.randomization.randomizeMassLower=0.05 task.env.randomization.randomizeMassUpper=0.051 \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=False \
train.ppo.priv_info=True