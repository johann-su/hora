# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# M1 of the IsaacGym -> IsaacLab migration. See docs/isaaclab_migration.md.
# --------------------------------------------------------

"""IsaacLab config for the in-hand rotation task.

The hydra YAML tree under ``configs/`` stays the source of truth -- all seven shell
scripts drive it with ``task.env.*=`` overrides, and hora's own PPO/ProprioAdapt read it
directly. This module only translates the parts IsaacLab needs into ``@configclass``
form; :func:`make_env_cfg` does the translation.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

# Joint order the policy works in: index, THUMB, middle, ring, four joints each.
#
# This is *not* the order PhysX reports (which is breadth-first, interleaving the
# fingers), and it is *not* URDF declaration order. It is the order produced by
# `_obs_allegro2hora` in algo/deploy/deploy.py, the order its allegro_dof_lower/upper
# arrays are written in, and -- verified empirically -- the order the joint columns of
# cache/*.npy are stored in. Everything policy-facing in this codebase uses it; the
# conversion to PhysX order happens only at the sim boundary.
#
# Run scripts/verify_hand_asset.py to re-derive the mapping against a converted asset.
HORA_JOINT_ORDER = (
    [f'joint_{i}_0' for i in range(0, 4)]      # index
    + [f'joint_{i}_0' for i in range(12, 16)]  # thumb
    + [f'joint_{i}_0' for i in range(4, 8)]    # middle
    + [f'joint_{i}_0' for i in range(8, 12)]   # ring
)

# Fingertip bodies, in HORA_JOINT_ORDER finger order. Used by M3's contact sensing.
HORA_FINGERTIP_BODIES = ['link_3_0', 'link_15_0', 'link_7_0', 'link_11_0']

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _default_joint_pos(urdf_rel: str) -> dict[str, float]:
    """Legal initial joint positions, derived from the URDF limits.

    IsaacLab rejects an articulation whose defaults fall outside the joint limits, and
    the Allegro thumb's joint_12_0 range is [0.263, 1.396] -- so a blanket 0.0 is
    invalid. Clamping 0.0 into each joint's own range is the honest fix; hardcoding the
    thumb exception would silently rot if the asset changed.

    Every joint is listed explicitly because IsaacLab refuses overlapping name patterns
    (a '.*' catch-all alongside a specific name is "Multiple matches").
    """
    path = os.path.join(REPO_ROOT, urdf_rel)
    joint_pos = {}
    for joint in ET.parse(path).getroot().findall('joint'):
        limit = joint.find('limit')
        if joint.get('type') == 'fixed' or limit is None:
            continue
        lo, hi = float(limit.get('lower', 0.0)), float(limit.get('upper', 0.0))
        joint_pos[joint.get('name')] = min(max(0.0, lo), hi)
    return joint_pos


def _usd_path(urdf_rel: str) -> str:
    """'assets/allegro/allegro_internal.urdf' -> absolute path of the converted USD."""
    rel = urdf_rel.replace('assets/', 'assets/usd/', 1)
    if rel.endswith('.urdf'):
        rel = rel[:-len('.urdf')] + '.usd'
    path = os.path.join(REPO_ROOT, rel)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'converted asset not found: {path}\n'
            'Run ./scripts/convert_assets.sh first (M0 of docs/isaaclab_migration.md).')
    return path


@configclass
class AllegroHandHoraEnvCfg(DirectRLEnvCfg):
    """Structural defaults. :func:`make_env_cfg` overwrites these from the hydra config."""

    decimation = 6              # 120 Hz sim / 6 -> 20 Hz policy
    episode_length_s = 20.0     # 400 policy steps at 20 Hz
    action_space = 16
    observation_space = 96      # 3 frames x (16 joint pos + 16 targets)
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=6,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8 * 1024 * 1024,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16384, env_spacing=0.25, replicate_physics=True, clone_in_fabric=True)

    robot_cfg: ArticulationCfg = None
    object_cfg: RigidObjectCfg = None

    # NOTE: hora's own hydra dict is deliberately NOT a field here. @configclass
    # deep-validates every field, and recursing through a large plain dict blows the
    # stack. The env takes it as a separate constructor argument instead.


def make_env_cfg(task_cfg: dict, num_envs: int | None = None) -> AllegroHandHoraEnvCfg:
    """Build an IsaacLab env cfg from hora's hydra ``task`` dict."""
    env = task_cfg['env']
    sim_cfg = task_cfg['sim']
    controller = env['controller']
    rand = env['randomization']

    cfg = AllegroHandHoraEnvCfg()

    # ---- sim / stepping -------------------------------------------------------------
    cfg.decimation = controller['controlFrequencyInv']
    cfg.sim.dt = sim_cfg['dt']
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.gravity = tuple(sim_cfg['gravity'])
    cfg.episode_length_s = env['episodeLength'] * cfg.decimation * cfg.sim.dt

    physx = sim_cfg.get('physx', {})
    cfg.sim.physx.bounce_threshold_velocity = physx.get('bounce_threshold_velocity', 0.2)
    cfg.sim.physx.gpu_max_rigid_contact_count = physx.get('max_gpu_contact_pairs', 8 * 1024 * 1024)

    cfg.action_space = env['numActions']
    cfg.observation_space = env['numObservations']

    # ---- scene ----------------------------------------------------------------------
    cfg.scene.num_envs = num_envs if num_envs is not None else env['numEnvs']
    cfg.scene.env_spacing = env['envSpacing']

    # M1 supports a single object scale, which is what lets replicate_physics stay True
    # (the fast cloning path). Per-env heterogeneous scale is M2 -- see "Decision 0" in
    # docs/isaaclab_migration.md.
    scale_list = rand['randomizeScaleList'] if rand['randomizeScale'] else [env['baseObjScale']]
    if len(scale_list) > 1:
        raise NotImplementedError(
            f'multi-scale randomization ({len(scale_list)} scales) is M2 of the migration; '
            'for M1 pass a single scale, e.g. '
            'task.env.randomization.randomizeScale=False task.env.baseObjScale=0.8')
    obj_scale = float(scale_list[0])

    # ---- hand -----------------------------------------------------------------------
    # Torque control: hora runs its own PD at sim rate in _apply_action, so the actuator
    # model must not also drive the joints. Zero stiffness/damping makes the implicit
    # actuator a pure effort passthrough. effort_limit matches the +-0.5 Nm clip hora
    # applies, and the armature/friction values come from the old DOF properties.
    torque_control = controller['torque_control']
    cfg.robot_cfg = ArticulationCfg(
        prim_path='/World/envs/env_.*/hand',
        spawn=sim_utils.UsdFileCfg(
            usd_path=_usd_path(env['asset']['handAsset']),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,          # IsaacGym: hand_asset_options.disable_gravity
                angular_damping=0.01,
                max_depenetration_velocity=physx.get('max_depenetration_velocity', 1000.0),
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=physx.get('num_position_iterations', 8),
                solver_velocity_iteration_count=physx.get('num_velocity_iterations', 0),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            # IsaacGym built this as Quat.from_axis_angle(y, -pi/2) * Quat.from_axis_angle(x, pi/2).
            # Worked through to IsaacLab's wxyz convention that is (0.5, 0.5, -0.5, 0.5).
            rot=(0.5, 0.5, -0.5, 0.5),
            # Overwritten per-env from the grasp cache on reset; this only has to be
            # legal. See _default_joint_pos for why it is not simply zeros.
            joint_pos=_default_joint_pos(env['asset']['handAsset']),
        ),
        actuators={
            'hand': ImplicitActuatorCfg(
                joint_names_expr=['.*'],
                effort_limit_sim=0.5,
                stiffness=0.0 if torque_control else controller['pgain'],
                damping=0.0 if torque_control else controller['dgain'],
                armature=0.001,
                friction=0.01,
            ),
        },
    )

    # ---- object ---------------------------------------------------------------------
    object_type = env['object']['type']
    cfg.object_cfg = RigidObjectCfg(
        prim_path='/World/envs/env_.*/object',
        spawn=sim_utils.UsdFileCfg(
            usd_path=_usd_path(_object_asset(object_type)),
            scale=(obj_scale, obj_scale, obj_scale),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=physx.get('max_depenetration_velocity', 1000.0),
                solver_position_iteration_count=physx.get('num_position_iterations', 8),
                solver_velocity_iteration_count=physx.get('num_velocity_iterations', 0),
            ),
        ),
        # z here only matters for the very first step -- reset overwrites it from the
        # grasp cache. 0.65 kept for parity with the IsaacGym _init_object_pose.
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.01, -0.01, 0.65), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    return cfg


def _object_asset(object_type: str) -> str:
    """Map hora's object type string onto a URDF path (converted to USD downstream).

    M1 handles the single-object types. The ``cuboid``/``cylinder`` families sample from
    a directory of ~80 meshes per env, which needs the multi-asset spawner and therefore
    lands with M2.
    """
    simple = {
        'simple_tennis_ball': 'assets/ball.urdf',
        'block': 'assets/cube.urdf',
        'cube': 'assets/cube.urdf',
        'cylinder': 'assets/cylinder.urdf',
        'ball': 'assets/ball.urdf',
    }
    if object_type in simple:
        return simple[object_type]
    raise NotImplementedError(
        f"object type {object_type!r} samples from an asset family, which needs the "
        f"multi-asset spawner (M2). For M1 use one of: {sorted(simple)}")
