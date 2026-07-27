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
import argparse
import sys

from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    '--report-contact-forces', action='store_true',
    help='Print the distribution of fingertip contact forces instead of collecting '
         'poses. Use this to calibrate task.env.contactForceThreshold.')
AppLauncher.add_app_launcher_args(_parser)
_app_args, _hydra_argv = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _hydra_argv

# scripts/gen_grasp.sh drives this with hydra-style `headless=True`; the app launches
# before hydra runs, so mirror it onto AppLauncher's flag.
for _arg in _hydra_argv:
    if _arg.startswith('headless='):
        _app_args.headless = _arg.split('=', 1)[1].lower() in ('true', '1', 'yes')

app_launcher = AppLauncher(_app_args)
simulation_app = app_launcher.app

# ---- Everything below runs with the app live. -------------------------------------
import os  # noqa: E402

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from termcolor import cprint  # noqa: E402

from hora.tasks import make_task  # noqa: E402
from hora.utils.misc import set_np_formatting, set_seed  # noqa: E402
from hora.utils.reformat import omegaconf_to_dict  # noqa: E402

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config.
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


def report_contact_forces(env, steps=200):
    """Sample fingertip contact forces so the threshold can be set from data.

    PhysX reports contact *forces* where IsaacGym reported contact *existence*, so grasp
    generation needs a cutoff that has no counterpart in the original. Picking it blind
    would silently change which grasps are accepted into the cache.
    """
    forces, net_forces, obj_z, stable = [], [], [], []
    cond_near, cond_touch, cond_held, tip_dist = [], [], [], []
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    for _ in range(steps):
        env.step(actions)
        forces.append(env.fingertip_contact_forces().flatten())
        # Unfiltered net force too: if this is nonzero while the filtered matrix is not,
        # the fingertips are touching *something* and the object filter is the problem.
        net_forces.append(torch.stack([
            torch.norm(s.data.net_forces_w.view(env.num_envs, -1, 3), dim=-1).sum(-1)
            for s in env._contact_sensors], dim=-1).flatten())
        obj_z.append(env.object_pos[:, 2])
        stable.append(env._grasp_is_stable().float().mean())
        _op = env.object_pos
        _ft = env.hand.data.body_pos_w[:, env._fingertip_idx] - env.scene.env_origins[:, None, :]
        _d = torch.norm(_op[:, None, :] - _ft, dim=-1)
        cond_near.append(torch.less(_d, 0.1).all(-1).float().mean())
        cond_touch.append((env.fingertip_contact_forces() >
                           env.contact_force_threshold).sum(-1).ge(2).float().mean())
        cond_held.append(torch.greater(_op[:, -1], env.reset_z_threshold).float().mean())
        tip_dist.append(_d.mean())

    print('\n---- diagnostics ----')
    print('force_matrix_w is None:',
          [s.data.force_matrix_w is None for s in env._contact_sensors])
    zs = torch.cat(obj_z)
    print(f'object z (env frame): min {zs.min():.4f} mean {zs.mean():.4f} max {zs.max():.4f}'
          f'   reset_z_threshold {env.reset_z_threshold}')
    nf = torch.cat(net_forces)
    print(f'unfiltered net fingertip force: nonzero {int((nf > 1e-6).sum())} / {nf.numel()}'
          f'  max {nf.max():.4f}')
    print(f'mean fraction of envs passing all 3 grasp conditions: '
          f'{torch.stack(stable).mean():.4f}')
    print(f'  cond1 near   (all tips < 0.1 m): {torch.stack(cond_near).mean():.4f}')
    print(f'  cond2 touch  (>=2 tips contact): {torch.stack(cond_touch).mean():.4f}')
    print(f'  cond3 held   (z > threshold)   : {torch.stack(cond_held).mean():.4f}')
    print(f'  mean fingertip-object distance : {torch.stack(tip_dist).mean():.4f} m')
    f = torch.cat(forces)
    nonzero = f[f > 1e-6]
    print('\n---- fingertip contact force distribution (N) ----')
    print(f'samples: {f.numel()}   nonzero: {nonzero.numel()} '
          f'({100.0 * nonzero.numel() / max(f.numel(), 1):.1f}%)')
    if nonzero.numel():
        qs = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], device=f.device)
        for q, v in zip(qs.tolist(), torch.quantile(nonzero, qs).tolist()):
            print(f'  p{int(q * 100):02d}: {v:.4f}')
        print(f'  max: {nonzero.max().item():.4f}')
    print('Set task.env.contactForceThreshold below the bulk of this distribution but '
          'above the noise floor.')


@hydra.main(config_name='config', config_path='configs', version_base='1.1')
def main(config: DictConfig):
    set_np_formatting()
    config.seed = set_seed(config.seed)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    env = make_task(
        config.task_name,
        omegaconf_to_dict(config.task),
        num_envs=config.num_envs if config.num_envs != '' else None,
    )
    env.reset()

    if _app_args.report_contact_forces:
        report_contact_forces(env)
        return

    # Zero actions throughout: on reset the position controller drives the hand to the
    # canonical pose and it simply holds there. Poses are filtered, not learned.
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    while not env.cache_complete:
        env.step(actions)


if __name__ == '__main__':
    import traceback  # noqa: E402

    _exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        _exit_code = 1

    # Not calling simulation_app.close(): it blocks with a live SimulationContext and
    # swallows the exit status. The cache is written synchronously by np.save before we
    # reach here. See "M0 findings" in docs/isaaclab_migration.md.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_code)
