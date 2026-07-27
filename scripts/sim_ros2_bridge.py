# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Expose the Allegro hand in Isaac Sim as a ROS 2 joint-state/command endpoint.

This is the simulation half of "Isaac Sim as a hardware surrogate": it runs one hand (and
optionally the object) and speaks the two topics a ros2_control hardware interface needs,
so the *real* deployment stack -- controller_manager, the position controller, and
``deploy.py`` -- can drive simulation instead of a physical hand with nothing in the policy
path branching on which it is.

    /isaac_joint_states     (published)  sensor_msgs/JointState, hora joint names
    /isaac_joint_commands   (subscribed) sensor_msgs/JointState, hora joint names

**Joint names are hora's, not ros2_control's.** This publishes ``joint_0_0 .. joint_15_0``
(the converted hora asset), while the ros2_allegro stack uses ``ah_joint00 .. ah_joint33``.
The two models are different URDFs, so the rename happens in one documented place --
``ros2_allegro``-side relay, see docs/isaaclab_migration.md -- rather than being smeared
across the graph.

Requires the ROS 2 bridge environment *before* launch (see README_ROS2_ISAACLAB.md):

    export ROS_DISTRO=humble ROS_DOMAIN_ID=42
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.11/site-packages/\\
isaacsim/exts/isaacsim.ros2.bridge/humble/lib
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--hand-usd', type=str,
                    default='assets/usd/allegro/allegro_internal.usd',
                    help='Converted hand USD (relative to the repo root).')
parser.add_argument('--domain-id', type=int, default=42,
                    help='ROS_DOMAIN_ID. Must match the devcontainer (42).')
parser.add_argument('--state-topic', type=str, default='/isaac_joint_states')
parser.add_argument('--command-topic', type=str, default='/isaac_joint_commands')
parser.add_argument('--publish-only', action='store_true',
                    help='Skip the command subscriber; useful for checking connectivity.')
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs with the app live."""

import os  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

# The OmniGraph and ROS 2 bridge extensions have to be enabled before `omni.graph` is
# importable at all -- importing it at module scope fails with ModuleNotFoundError.
for _ext in ('omni.graph.action', 'omni.graph.nodes', 'isaacsim.core.nodes',
             'isaacsim.ros2.bridge'):
    enable_extension(_ext)

import isaaclab.sim as sim_utils  # noqa: E402
import omni.graph.core as og  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

from hora.tasks.allegro_hand_hora_cfg import HORA_JOINT_ORDER  # noqa: E402

HAND_PRIM = '/World/hand'
# Must live under /World: creating the graph prim at stage root fails with
# "Unable to create prim for graph".
GRAPH_PATH = '/World/ActionGraph'


def default_joint_pos(urdf_rel: str) -> dict[str, float]:
    """Legal initial joint positions -- see _default_joint_pos in the task cfg."""
    joint_pos = {}
    for joint in ET.parse(os.path.join(REPO_ROOT, urdf_rel)).getroot().findall('joint'):
        limit = joint.find('limit')
        if joint.get('type') == 'fixed' or limit is None:
            continue
        lo, hi = float(limit.get('lower', 0.0)), float(limit.get('upper', 0.0))
        joint_pos[joint.get('name')] = min(max(0.0, lo), hi)
    return joint_pos


def build_graph(domain_id: int, state_topic: str, command_topic: str, publish_only: bool):
    """Wire the ROS 2 publish/subscribe graph onto the hand articulation."""
    keys = og.Controller.Keys
    nodes = [
        ('OnTick', 'omni.graph.action.OnPlaybackTick'),
        ('Context', 'isaacsim.ros2.bridge.ROS2Context'),
        ('ReadSimTime', 'isaacsim.core.nodes.IsaacReadSimulationTime'),
        ('PublishJointState', 'isaacsim.ros2.bridge.ROS2PublishJointState'),
    ]
    connect = [
        ('OnTick.outputs:tick', 'PublishJointState.inputs:execIn'),
        ('Context.outputs:context', 'PublishJointState.inputs:context'),
        ('ReadSimTime.outputs:simulationTime', 'PublishJointState.inputs:timeStamp'),
    ]
    values = [
        ('Context.inputs:domain_id', domain_id),
        ('PublishJointState.inputs:topicName', state_topic),
        ('PublishJointState.inputs:targetPrim', [HAND_PRIM]),
    ]

    if not publish_only:
        nodes += [
            ('SubscribeJointState', 'isaacsim.ros2.bridge.ROS2SubscribeJointState'),
            ('ArticulationController', 'isaacsim.core.nodes.IsaacArticulationController'),
        ]
        connect += [
            ('OnTick.outputs:tick', 'SubscribeJointState.inputs:execIn'),
            ('Context.outputs:context', 'SubscribeJointState.inputs:context'),
            ('SubscribeJointState.outputs:execOut', 'ArticulationController.inputs:execIn'),
            ('SubscribeJointState.outputs:jointNames',
             'ArticulationController.inputs:jointNames'),
            ('SubscribeJointState.outputs:positionCommand',
             'ArticulationController.inputs:positionCommand'),
        ]
        values += [
            ('SubscribeJointState.inputs:topicName', command_topic),
            ('ArticulationController.inputs:targetPrim', [HAND_PRIM]),
        ]

    og.Controller.edit(
        {'graph_path': GRAPH_PATH, 'evaluator_name': 'execution'},
        {keys.CREATE_NODES: nodes, keys.CONNECT: connect, keys.SET_VALUES: values},
    )


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(device='cuda:0', dt=1 / 120))

    ground = sim_utils.GroundPlaneCfg()
    ground.func('/World/ground', ground)
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light.func('/World/Light', light)

    # Position control here, not hora's torque PD: the surrogate has to present the same
    # position interface the physical hand does, since that is what the ros2_control
    # position controller drives.
    urdf_rel = 'assets/allegro/allegro_internal.urdf'
    hand_cfg = ArticulationCfg(
        prim_path=HAND_PRIM,
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(REPO_ROOT, args_cli.hand_usd),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.5, 0.5, -0.5, 0.5),
            joint_pos=default_joint_pos(urdf_rel),
        ),
        actuators={'hand': ImplicitActuatorCfg(
            joint_names_expr=['.*'], effort_limit_sim=0.5, stiffness=3.0, damping=0.1)},
    )
    hand = Articulation(hand_cfg)

    # After reset: the articulation has to be initialised before the graph targets it.
    sim.reset()
    build_graph(args_cli.domain_id, args_cli.state_topic, args_cli.command_topic,
                args_cli.publish_only)
    idx, names = hand.find_joints(HORA_JOINT_ORDER, preserve_order=True)
    print(f'[bridge] hand DOFs      : {len(names)}', flush=True)
    print(f'[bridge] publishing     : {args_cli.state_topic}', flush=True)
    if not args_cli.publish_only:
        print(f'[bridge] subscribing    : {args_cli.command_topic}', flush=True)
    print(f'[bridge] ROS_DOMAIN_ID  : {args_cli.domain_id}', flush=True)
    print('[bridge] running -- Ctrl-C to stop', flush=True)

    while simulation_app.is_running():
        sim.step()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except KeyboardInterrupt:
        pass
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # See "M0 findings": simulation_app.close() blocks and swallows the exit status.
    os._exit(code)
