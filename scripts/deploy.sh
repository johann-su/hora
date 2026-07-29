#!/bin/bash
CACHE=$1
python deploy.py checkpoint=outputs/AllegroHandHora/"${CACHE}"/stage2_nn/"${CHECKPOINT:-model_best.ckpt}"
