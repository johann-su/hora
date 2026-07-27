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
# Ported from IsaacGym to IsaacLab. See docs/isaaclab_migration.md.
# --------------------------------------------------------

# ---- Isaac Sim must be launched before anything imports isaaclab (it pulls in `carb`).
# This block therefore runs before the hora/isaaclab imports further down, and hydra is
# only invoked afterwards. Reordering these will produce confusing extension-load errors.
import argparse
import sys

from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser(add_help=False)
AppLauncher.add_app_launcher_args(_parser)
_app_args, _hydra_argv = _parser.parse_known_args()
# Hydra parses sys.argv itself, so hand it only the overrides meant for it.
sys.argv = [sys.argv[0]] + _hydra_argv

# scripts/*.sh drive this with hydra-style `headless=True`, but the app is launched
# before hydra ever runs, so mirror it onto AppLauncher's flag. The token stays in argv
# for hydra as well -- `headless` is still a real key in configs/config.yaml.
for _arg in _hydra_argv:
    if _arg.startswith('headless='):
        _app_args.headless = _arg.split('=', 1)[1].lower() in ('true', '1', 'yes')
    elif _arg.startswith('pipeline='):
        _app_args.device = 'cpu' if _arg.split('=', 1)[1].lower() == 'cpu' else _app_args.device

app_launcher = AppLauncher(_app_args)
simulation_app = app_launcher.app

# ---- Everything below runs with the app live. -------------------------------------
import datetime  # noqa: E402
import os  # noqa: E402

import hydra  # noqa: E402
from hydra.utils import to_absolute_path  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from termcolor import cprint  # noqa: E402

from hora.algo.padapt.padapt import ProprioAdapt  # noqa: E402,F401
from hora.algo.ppo.ppo import PPO  # noqa: E402,F401
from hora.tasks import make_task  # noqa: E402
from hora.utils.misc import git_diff_config, git_hash, set_np_formatting, set_seed  # noqa: E402
from hora.utils.reformat import omegaconf_to_dict  # noqa: E402

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config.
# used primarily for num_envs
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


@hydra.main(config_name='config', config_path='configs', version_base='1.1')
def main(config: DictConfig):
    if config.checkpoint:
        config.checkpoint = to_absolute_path(config.checkpoint)

    set_np_formatting()
    config.seed = set_seed(config.seed)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    env = make_task(
        config.task_name,
        omegaconf_to_dict(config.task),
        num_envs=config.num_envs if config.num_envs != '' else None,
    )

    output_dif = os.path.join('outputs', config.train.ppo.output_name)
    os.makedirs(output_dif, exist_ok=True)
    agent = eval(config.train.algo)(env, output_dif, full_config=config)
    if config.test:
        agent.restore_test(config.train.load_path)
        agent.test()
    else:
        date = str(datetime.datetime.now().strftime('%m%d%H'))
        print(git_diff_config('./'))
        os.system(f'git diff HEAD > {output_dif}/gitdiff.patch')
        with open(os.path.join(output_dif, f'config_{date}_{git_hash()}.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(config))

        # check whether execute train by mistake:
        best_ckpt_path = os.path.join(
            'outputs', config.train.ppo.output_name,
            'stage1_nn' if config.train.algo == 'PPO' else 'stage2_nn', 'best.pth'
        )
        if os.path.exists(best_ckpt_path):
            user_input = input(
                f'are you intentionally going to overwrite files in {config.train.ppo.output_name}, type yes to continue \n')
            if user_input != 'yes':
                exit()

        agent.restore_train(config.train.load_path)
        agent.train()


if __name__ == '__main__':
    import traceback  # noqa: E402

    _exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        _exit_code = 1

    # Deliberately NOT calling simulation_app.close(): with a live SimulationContext it
    # blocks indefinitely, which looks exactly like training finishing and then hanging
    # with no output -- and because it blocks, any exit-code handling placed after it
    # never runs either. Checkpoints are written synchronously by torch.save and the
    # tensorboard writer is flushed at the end of train(), so there is nothing left to
    # lose by leaving directly. See "M0 findings" in docs/isaaclab_migration.md.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_code)
