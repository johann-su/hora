# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# M3 of the IsaacGym -> IsaacLab migration. See docs/isaaclab_migration.md.
# --------------------------------------------------------

"""Check that a generated grasp cache behaves like the published one.

Statistics alone do not settle this -- two caches can have similar joint histograms and
still differ in whether the poses actually *hold the object*. So the load-bearing test is
functional: reset the rotation env from the cache, step with zero actions, and measure how
many environments still hold the object after a while. A good cache starts the policy in a
stable grasp; a bad one drops the object before the policy does anything.

Because a SimulationContext cannot be recreated inside one process (see "M0 findings"),
this runs one cache per invocation. Run it twice and compare the printed summary::

    python scripts/validate_grasp_cache.py --cache-dir cache            # published
    python scripts/validate_grasp_cache.py --cache-dir /tmp/gencache    # generated
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cache-dir', type=str, default='cache',
                    help='Directory holding <name>_grasp_50k_s<scale>.npy.')
parser.add_argument('--scale', type=float, default=0.8, help='Object scale to test.')
parser.add_argument('--num-envs', type=int, default=1024)
parser.add_argument('--steps', type=int, default=100,
                    help='Zero-action steps to hold the grasp for.')
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs with the app live."""

import os  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

# Running as scripts/validate_grasp_cache.py puts scripts/ on sys.path, not the repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hora.tasks import make_task  # noqa: E402
from hora.tasks.allegro_hand_hora_cfg import HORA_FINGERTIP_BODIES  # noqa: E402


def cache_path(cache_dir: str, scale: float, name: str = 'internal_allegro') -> str:
    return os.path.join(
        cache_dir, f'{name}_grasp_50k_s{str(scale).replace(".", "")}.npy')


def report_statistics(path: str):
    data = np.load(path)
    print(f'file        : {path}')
    print(f'shape       : {data.shape}  dtype {data.dtype}')
    joints, pos, quat = data[:, :16], data[:, 16:19], data[:, 19:23]
    print(f'joint mean  : {np.round(joints.mean(0), 3).tolist()}')
    print(f'joint std   : {np.round(joints.std(0), 3).tolist()}')
    print(f'obj pos mean: {np.round(pos.mean(0), 4).tolist()}  '
          f'std {np.round(pos.std(0), 4).tolist()}')
    print(f'quat norm   : mean {np.linalg.norm(quat, axis=1).mean():.4f} '
          f'(should be 1.0)')
    # Real part last in the stored xyzw convention; a mean near +-1 means mostly-upright.
    print(f'quat mean   : {np.round(quat.mean(0), 4).tolist()}  (xyzw on disk)')


def build_task_cfg(cache_dir: str, scale: float, num_envs: int) -> dict:
    task = yaml.safe_load(open(os.path.join(REPO_ROOT, 'configs/task/AllegroHandHora.yaml')))
    task['physics_engine'] = 'physx'
    task['on_evaluation'] = False
    task['env']['numEnvs'] = num_envs
    task['env']['object']['type'] = 'simple_tennis_ball'
    task['env']['baseObjScale'] = scale
    task['env']['genGrasps'] = False
    task['env']['graspCacheDir'] = cache_dir
    task['env']['randomization']['randomizeScale'] = False
    # Isolate the cache: any other randomization would confound the comparison.
    for key in ('randomizeMass', 'randomizeCOM', 'randomizeFriction', 'randomizePDGains'):
        task['env']['randomization'][key] = False
    task['env']['randomization']['jointNoiseScale'] = 0.0
    task['sim']['use_gpu_pipeline'] = True
    task['sim']['physx'].update(num_threads=4, solver_type=1, use_gpu=True, num_subscenes=4)
    return task


def main() -> bool:
    path = cache_path(args_cli.cache_dir, args_cli.scale)
    if not os.path.isfile(path):
        print(f'not found: {path}')
        return False

    print('=' * 72)
    report_statistics(path)
    print('-' * 72)

    env = make_task('AllegroHandHora',
                    build_task_cfg(args_cli.cache_dir, args_cli.scale, args_cli.num_envs),
                    num_envs=args_cli.num_envs)
    env.reset()

    tip_idx, _ = env.hand.find_bodies(HORA_FINGERTIP_BODIES, preserve_order=True)
    tips = env.hand.data.body_pos_w[:, tip_idx] - env.scene.env_origins[:, None, :]
    dist0 = torch.norm(env.object_pos[:, None, :] - tips, dim=-1)
    print(f'at reset    : mean fingertip-object distance {dist0.mean():.4f} m')
    print(f'at reset    : object z mean {env.object_pos[:, 2].mean():.4f} '
          f'(threshold {env.reset_z_threshold})')

    # Zero actions: the policy contributes nothing, so anything that survives is the
    # grasp itself holding.
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    held_curve = []
    for step in range(args_cli.steps):
        env.step(actions)
        held = torch.greater(env.object_pos[:, 2], env.reset_z_threshold)
        held_curve.append(held.float().mean().item())

    for s in (0, 9, 24, 49, args_cli.steps - 1):
        if s < len(held_curve):
            print(f'held @ step {s + 1:>3}: {held_curve[s]:.3f}')
    print(f'held (mean over run): {float(np.mean(held_curve)):.3f}')
    print('=' * 72)
    return True


if __name__ == '__main__':
    ok = False
    try:
        ok = main()
    except BaseException:
        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    # See "M0 findings": simulation_app.close() blocks and swallows the exit status.
    os._exit(0 if ok else 1)
