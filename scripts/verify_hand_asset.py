# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# M0 of the IsaacGym -> IsaacLab migration. See docs/isaaclab_migration.md.
# --------------------------------------------------------

"""Pin down the converted Allegro hand's DOF order and joint limits.

Two things must hold before any of the migrated env code can be trusted, and neither is
visible by reading the URDF:

1. **DOF order.** PhysX orders articulation DOFs breadth-first, so the four finger chains
   interleave by depth level. hora's policy works finger-major (index, thumb, middle,
   ring). The two differ, and every joint read and write has to be reordered. This prints
   the mapping so M1 can assert against it.

2. **Joint limits.** The converted USD's limits must agree with ``allegro_dof_lower`` /
   ``allegro_dof_upper`` in ``hora/algo/deploy/deploy.py``. If they drift, a policy
   trained in sim is clipped against different bounds than the one running on hardware --
   silent, and expensive to track down later.

Run after conversion, and again whenever the hand asset or the joint mapping changes::

    python scripts/verify_hand_asset.py assets/usd/allegro/allegro_internal.usd \\
                                        assets/allegro/allegro_internal.urdf

Exits non-zero if the limits disagree or the joint set is not the expected 16.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('usd', type=str, help='Converted hand USD.')
parser.add_argument('urdf', type=str, help='Source hand URDF (read for joint limits).')
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# isaaclab pulls in `carb`, which only exists inside a launched Isaac Sim app.
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs with the app live."""

import os  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

# Held module-level so shutdown can tear it down even when main() raises partway through.
_sim = None

# The joint order hora's policy works in: index, THUMB, middle, ring -- each finger's
# four joints contiguous. This is what `_obs_allegro2hora` in algo/deploy/deploy.py
# produces, and the order its allegro_dof_lower/upper arrays are written in.
#
# Note the underscores: USD prim paths cannot contain '.', so the source URDFs were
# renamed from `joint_0.0` to `joint_0_0`. Look joints up by the sanitized name.
HORA_JOINT_ORDER = (
    [f'joint_{i}_0' for i in range(0, 4)]      # index
    + [f'joint_{i}_0' for i in range(12, 16)]  # thumb
    + [f'joint_{i}_0' for i in range(4, 8)]    # middle
    + [f'joint_{i}_0' for i in range(8, 12)]   # ring
)

# From hora/algo/deploy/deploy.py, in HORA_JOINT_ORDER.
DEPLOY_DOF_LOWER = [
    -0.4700, -0.1960, -0.1740, -0.2270, 0.2630, -0.1050, -0.1890, -0.1620,
    -0.4700, -0.1960, -0.1740, -0.2270, -0.4700, -0.1960, -0.1740, -0.2270,
]
DEPLOY_DOF_UPPER = [
    0.4700, 1.6100, 1.7090, 1.6180, 1.3960, 1.1630, 1.6440, 1.7190,
    0.4700, 1.6100, 1.7090, 1.6180, 0.4700, 1.6100, 1.7090, 1.6180,
]


def urdf_joint_limits(urdf: Path) -> dict[str, tuple[float, float]]:
    """Joint name -> (lower, upper) from the URDF."""
    limits = {}
    for joint in ET.parse(urdf).getroot().findall('joint'):
        node = joint.find('limit')
        if joint.get('type') != 'fixed' and node is not None:
            limits[joint.get('name')] = (
                float(node.get('lower', 0.0)), float(node.get('upper', 0.0)))
    return limits


def main() -> bool:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg

    usd_path, urdf_path = Path(args_cli.usd), Path(args_cli.urdf)
    for p in (usd_path, urdf_path):
        if not p.is_file():
            # Return rather than raise SystemExit: the shutdown path below must run, or
            # the process hangs instead of reporting the missing file.
            print(f'not found: {p}')
            return False

    # IsaacLab refuses to initialise an articulation whose default joint positions fall
    # outside the joint limits; IsaacGym silently allowed it. The thumb's joint_12_0 has
    # a strictly positive range ([0.263, 1.396]), so the implicit default of 0.0 is
    # invalid. Seed by clamping 0.0 into each joint's range.
    limits = urdf_joint_limits(urdf_path)
    joint_pos = {name: min(max(0.0, lo), hi) for name, (lo, hi) in limits.items()}

    global _sim
    _sim = sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(device='cuda:0', dt=1 / 120))
    cfg = ArticulationCfg(
        prim_path='/World/hand',
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path.resolve())),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=joint_pos),
        # Zero-gain implicit actuators: hora runs its own PD at 120 Hz.
        actuators={'hand': ImplicitActuatorCfg(
            joint_names_expr=['.*'], stiffness=0.0, damping=0.0)},
    )
    hand = Articulation(cfg)
    sim.reset()

    names = list(hand.joint_names)
    print(f'DOFs: {len(names)}')
    print(f'PhysX order: {names}')

    if sorted(names) != sorted(HORA_JOINT_ORDER):
        print('!! joint SET mismatch -- this USD does not describe the hand we expect')
        print(f'missing: {sorted(set(HORA_JOINT_ORDER) - set(names))}')
        print(f'extra:   {sorted(set(names) - set(HORA_JOINT_ORDER))}')
        return False

    idx, _ = hand.find_joints(HORA_JOINT_ORDER, preserve_order=True)
    print(f'hora->PhysX index map: {idx}')
    if idx == list(range(len(idx))):
        print('PhysX order already matches hora order (no remap needed)')
    else:
        print('PhysX order differs from hora order -- env code MUST reorder using this map')

    lower = hand.data.joint_pos_limits[0, idx, 0].tolist()
    upper = hand.data.joint_pos_limits[0, idx, 1].tolist()
    tol = 1e-3
    lo_ok = all(abs(a - b) < tol for a, b in zip(lower, DEPLOY_DOF_LOWER))
    hi_ok = all(abs(a - b) < tol for a, b in zip(upper, DEPLOY_DOF_UPPER))
    print(f'limits in hora order, lower: {[round(v, 4) for v in lower]}')
    print(f'limits in hora order, upper: {[round(v, 4) for v in upper]}')
    if lo_ok and hi_ok:
        print('limits agree with deploy.py allegro_dof_lower/upper')
    else:
        print('!! LIMITS DISAGREE with deploy.py -- sim and hardware would clip differently')
        if not lo_ok:
            print(f'deploy lower: {DEPLOY_DOF_LOWER}')
        if not hi_ok:
            print(f'deploy upper: {DEPLOY_DOF_UPPER}')
    return lo_ok and hi_ok


if __name__ == '__main__':
    ok = False
    try:
        ok = main()
    except BaseException:
        traceback.print_exc()
        ok = False
    finally:
        # A live SimulationContext keeps callbacks and physics threads registered, which
        # is what makes this script print all its output and then never exit.
        if _sim is not None:
            _sim.clear_all_callbacks()
            _sim.clear_instance()

    # Deliberately NOT calling simulation_app.close(): it tears the process down itself
    # and always yields status 0, so a failed check would report success. Nothing here
    # writes to disk -- the app is only ever read from -- so exiting straight out is safe
    # and lets the exit code mean what it says.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 1)
