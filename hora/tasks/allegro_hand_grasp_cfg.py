# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Config for grasp-pose generation.

Adds one :class:`ContactSensor` per fingertip to the in-hand rotation config. Contact is
the only thing grasp generation needs that the rotation task does not.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from hora.tasks.allegro_hand_hora_cfg import (
    HORA_FINGERTIP_BODIES,
    REPO_ROOT,
    AllegroHandHoraEnvCfg,
    _object_assets,
    make_env_cfg,
)

# Finger labels in hora order, used to name the sensors.
FINGER_NAMES = ('index', 'thumb', 'middle', 'ring')


@configclass
class AllegroHandGraspEnvCfg(AllegroHandHoraEnvCfg):
    # One sensor per fingertip, in HORA_FINGERTIP_BODIES order.
    #
    # A separate sensor per body rather than one sensor matching all four: this mirrors
    # IsaacLab's own dexsuite Allegro config, and keeps `force_matrix_w` at a predictable
    # (N, 1, 1, 3) per sensor instead of depending on how a multi-body regex resolves.
    contact_sensors: tuple = ()


def make_grasp_env_cfg(task_cfg: dict, num_envs: int | None = None) -> AllegroHandGraspEnvCfg:
    """Build the grasp-generation env cfg from hora's hydra ``task`` dict."""
    base = make_env_cfg(task_cfg, num_envs=num_envs)

    cfg = AllegroHandGraspEnvCfg()
    for field in ('decimation', 'episode_length_s', 'action_space', 'observation_space',
                  'state_space', 'sim', 'scene', 'robot_cfg', 'object_cfg', 'object_scales'):
        setattr(cfg, field, getattr(base, field))

    cfg.scene.clone_in_fabric = False
    cfg.scene.replicate_physics = False

    body_name = _object_link_name(env_cfg_object_urdf(task_cfg))
    cfg.contact_sensors = tuple(
        ContactSensorCfg(
            prim_path=f'/World/envs/env_.*/hand/{body}',
            filter_prim_paths_expr=[f'/World/envs/env_.*/object/{body_name}'],
            update_period=0.0,
            history_length=0,
        )
        for body in HORA_FINGERTIP_BODIES
    )
    return cfg


def env_cfg_object_urdf(task_cfg: dict) -> str:
    """Path of the first URDF the object spawner draws from."""
    return _object_assets(task_cfg['env']['object']['type'])[0]


def _object_link_name(urdf_rel: str) -> str:
    """Name of the single link in an object URDF -- the prim the rigid body ends up on."""
    root = ET.parse(os.path.join(REPO_ROOT, urdf_rel)).getroot()
    links = [ln.get('name') for ln in root.findall('link')]
    if len(links) != 1:
        raise ValueError(f'expected a single-link object URDF, {urdf_rel} has {links}')
    return links[0]
