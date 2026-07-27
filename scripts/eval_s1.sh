#!/bin/bash
if [ $# -lt 2 ]; then
    echo "usage: $0 <gpus> <cache_name>"
    echo "  e.g.  $0 0 hora_v1"
    echo
    echo "All 2 arguments are required. In particular an empty <gpus> expands to"
    echo "CUDA_VISIBLE_DEVICES= , which hides every GPU -- Isaac Sim then fails with"
    echo "'no CUDA-capable device is detected', which looks like a broken driver but is not."
    exit 1
fi


GPUS=$1
CACHE=$2
C=outputs/AllegroHandHora/"${CACHE}"/stage1_nn/best.pth
CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=AllegroHandHora headless=True \
task.env.numEnvs=20000 test=True task.on_evaluation=True \
task.env.object.type=cylinder_default \
train.algo=PPO \
task.env.randomization.randomizeMass=True \
task.env.randomization.randomizeCOM=True \
task.env.randomization.randomizeFriction=True \
task.env.randomization.randomizePDGains=True \
task.env.randomization.randomizeScale=True \
task.env.randomization.jointNoiseScale=0.005 \
task.env.reset_height_threshold=0.6 \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.ppo.priv_info=True \
train.ppo.output_name=AllegroHandHora/"${CACHE}" \
checkpoint="${C}"