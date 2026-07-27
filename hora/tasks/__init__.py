# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Ported from IsaacGym to IsaacLab. See docs/isaaclab_migration.md.
# --------------------------------------------------------

"""Task registry.

The tasks are registered with gymnasium for discoverability, but hora's entry points
construct the env class directly: ``gymnasium.make`` wraps the env in order-enforcing
wrappers that add nothing here, and hora's PPO drives the env itself.

Importing this module requires a running Isaac Sim app -- ``isaaclab`` pulls in ``carb``.
Call ``AppLauncher`` before importing.
"""

from hora.tasks.allegro_hand_hora import AllegroHandHora
from hora.tasks.allegro_hand_hora_cfg import AllegroHandHoraEnvCfg, make_env_cfg

# Name -> (env class, cfg factory). Replaces IsaacGym's `isaacgym_task_map`.
isaaclab_task_map = {
    'AllegroHandHora': (AllegroHandHora, make_env_cfg),
    'PublicAllegroHandHora': (AllegroHandHora, make_env_cfg),
}


def make_task(task_name: str, task_cfg: dict, num_envs=None, render_mode=None):
    """Build a task env from hora's hydra ``task`` config dict."""
    if task_name not in isaaclab_task_map:
        if 'Grasp' in task_name:
            raise NotImplementedError(
                f'{task_name} needs contact sensing, which is M3 of the migration '
                '(see docs/isaaclab_migration.md).')
        raise KeyError(f'unknown task {task_name!r}; known: {sorted(isaaclab_task_map)}')
    env_cls, cfg_factory = isaaclab_task_map[task_name]
    return env_cls(cfg_factory(task_cfg, num_envs=num_envs), task_cfg,
                   render_mode=render_mode)


def register_gym_envs():
    """Register the tasks with gymnasium. Safe to call more than once."""
    import gymnasium as gym
    for name, (env_cls, _) in isaaclab_task_map.items():
        env_id = f'Hora-{name}-v0'
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point='hora.tasks.allegro_hand_hora:AllegroHandHora',
            disable_env_checker=True,
            kwargs={'env_cfg_entry_point': AllegroHandHoraEnvCfg},
        )


__all__ = ['AllegroHandHora', 'AllegroHandHoraEnvCfg', 'make_env_cfg', 'make_task',
           'isaaclab_task_map', 'register_gym_envs']
