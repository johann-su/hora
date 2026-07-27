# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""IsaacLab config for the in-hand rotation task.

The hydra YAML tree under ``configs/`` stays the source of truth -- all seven shell
scripts drive it with ``task.env.*=`` overrides, and hora's own PPO/ProprioAdapt read it
directly. This module only translates the parts IsaacLab needs into ``@configclass``
form; :func:`make_env_cfg` does the translation.
"""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg
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

    # Nominal object scales, in the order envs cycle through them: env i gets
    # object_scales[i % len(object_scales)]. The grasp cache is keyed by these values,
    # so this tuple is the single source of truth for the scale <-> cache pairing.
    object_scales: tuple = (0.8,)

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

    scale_list = [float(s) for s in
                  (rand['randomizeScaleList'] if rand['randomizeScale'] else [env['baseObjScale']])]
    cfg.object_scales = tuple(scale_list)

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
    # Object identity and scale are both per-env and both fixed at construction, so they
    # are expressed as spawner variants rather than runtime randomization. Variants are
    # ordered scale-minor:
    #
    #     variant c  ->  scale = scale_list[c % S],  asset = asset_list[c // S]
    #
    # MultiAssetSpawnerCfg with random_choice=False assigns variant (i % V) to env i, and
    # because S divides V that collapses to scale_list[i % S] -- exactly the `i %
    # num_scales` rule the IsaacGym code used, and the one _reset_idx relies on to pick
    # the matching grasp cache.
    asset_list = _object_assets(env['object']['type'])
    n_scales = len(scale_list)
    # Every variant spawns a prototype prim whether or not an env uses it, so cap the
    # asset count at what this many envs can actually reach. Rounding up keeps S | V.
    max_assets = max(1, -(-cfg.scene.num_envs // n_scales))
    asset_list = asset_list[:max_assets]

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=physx.get('max_depenetration_velocity', 1000.0),
        solver_position_iteration_count=physx.get('num_position_iterations', 8),
        solver_velocity_iteration_count=physx.get('num_velocity_iterations', 0),
    )
    variants = [
        sim_utils.UsdFileCfg(
            usd_path=_usd_path(asset), scale=(sc, sc, sc),
            activate_contact_sensors=True, rigid_props=rigid_props)
        for asset in asset_list for sc in scale_list
    ]

    if len(variants) == 1:
        spawn_cfg = variants[0]
    else:
        # Per-env geometry means physics cannot be replicated from a single prototype.
        # This is the cost of multi-scale: slower startup and a lower env ceiling.
        cfg.scene.replicate_physics = False
        # Fabric cloning must go too. It is faster, but the clones are not reachable
        # through USD APIs -- and the multi-asset spawner finds its targets by matching
        # USD prim paths, so with fabric on it saw only env_0 and spawned one object for
        # the whole scene (num_instances=1 for N envs, property writes hitting env 0).
        cfg.scene.clone_in_fabric = False
        spawn_cfg = MultiAssetSpawnerCfg(
            assets_cfg=variants, random_choice=False, activate_contact_sensors=True)

    cfg.object_cfg = RigidObjectCfg(
        prim_path='/World/envs/env_.*/object',
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.01, -0.01, _object_start_z(env)), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    return cfg


def _object_start_z(env: dict) -> float:
    """Object spawn height, following the IsaacGym _init_object_pose.

    For in-hand rotation this only matters on the very first step -- reset overwrites it
    from the grasp cache. For grasp *generation* it is load-bearing: the object has to
    start just above the fingertips so the closing hand can catch it. Starting it at the
    rotation task's height leaves it below reset_height_threshold almost immediately, so
    every env fails on step one and no poses are ever collected.
    """
    z = 0.66 if env['genGrasps'] else 0.65
    if 'internal' not in env['grasp_cache_name']:
        z -= 0.02
    return z


def _object_assets(object_type: str) -> list[str]:
    """Map hora's object type string onto the URDF paths it draws from.

    Mirrors ``_setup_object_info`` in the IsaacGym implementation: a bare name is a
    single asset, while ``cuboid_<subset>`` / ``cylinder_<subset>`` enumerate a directory.

    One deliberate behavioural difference: IsaacGym drew each env's object with
    ``np.random.choice(..., p=sampleProb)``, whereas the multi-asset spawner assigns
    round-robin. For the uniform weights hora actually uses these agree in distribution,
    and round-robin additionally guarantees exact balance across envs. Non-uniform
    ``sampleProb`` would need the list repeating in proportion.
    """
    simple = {
        'simple_tennis_ball': 'assets/ball.urdf',
        'block': 'assets/cube.urdf',
        'cube': 'assets/cube.urdf',
        'cylinder': 'assets/cylinder.urdf',
        'ball': 'assets/ball.urdf',
    }
    if object_type in simple:
        return [simple[object_type]]

    for family in ('cuboid', 'cylinder'):
        if object_type.startswith(family):
            subset = object_type.split('_', 1)[1] if '_' in object_type else 'default'
            pattern = os.path.join(REPO_ROOT, 'assets', family, subset, '*.urdf')
            found = sorted(glob.glob(pattern))
            if not found:
                raise FileNotFoundError(f'no assets matched {pattern}')
            return [os.path.relpath(f, REPO_ROOT) for f in found]

    raise ValueError(
        f"unknown object type {object_type!r}; expected one of {sorted(simple)} "
        f"or a cuboid_<subset> / cylinder_<subset> family")
