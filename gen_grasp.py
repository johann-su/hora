# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------

# NOTE: this entry point is not yet ported -- grasp generation is M3 of the IsaacGym ->
# IsaacLab migration (see docs/isaaclab_migration.md). It needs ContactSensor-based
# fingertip/object contact detection to replace IsaacGym's CPU-only
# get_env_rigid_contacts; see hora/tasks/allegro_hand_grasp.py.
#
# Use the published grasp caches (README.md) until then -- cache/*.npy covers every
# scale in randomizeScaleList.

import hydra
from omegaconf import DictConfig, OmegaConf

from hora.tasks import make_task  # noqa: F401  (M3 will build the grasp task through this)
from hora.utils.misc import set_np_formatting, set_seed
from hora.utils.reformat import omegaconf_to_dict


## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config.
# used primarily for num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    config.seed = set_seed(config.seed)

    raise NotImplementedError(
        'Grasp generation has not been ported to IsaacLab yet (M3 of '
        'docs/isaaclab_migration.md). Use the published caches in cache/ for now.')


if __name__ == '__main__':
    main()
