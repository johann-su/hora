# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on:
# https://github.com/felixduvallet/allegro-hand-ros/blob/master/allegro_hand/src/allegro_hand/liballegro.py
# Ported to ROS 2 for the ros2_allegro (ros2_control) stack.
# --------------------------------------------------------

import threading

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from lifecycle_msgs.msg import TransitionEvent
from controller_manager_msgs.srv import ListControllers, SwitchController
from allegro_hand_control_msgs.action import GraspCommand


class Allegro(Node):

    def __init__(self, joint_prefix='ah_', namespace='',
                 position_controller='allegro_hand_position_controller',
                 grasp_controller='allegro_hand_grasp_controller',
                 num_joints=16, spin_thread=True):
        """ Simple python interface to the Allegro Hand (ros2_allegro stack).

        The AllegroClient is a simple python interface to an allegro
        robot hand.  It enables you to command the hand directly through
        python library calls (joint positions or 'named' grasps).

        The hand is driven by ros2_control: joint positions go to a
        forward_command_controller and named grasps to the
        AllegroHandGraspController action server. Both controllers claim
        the same joints, so this client switches the active controller
        through the controller_manager as needed.

        :param joint_prefix: Prefix of the hand joint names in the URDF
        (default 'ah_', giving ah_joint00 .. ah_joint33). Positions and
        commands crossing this interface are in allegro SDK order --
        index, middle, ring, thumb, i.e. joint10..13, joint20..23,
        joint30..33, joint00..03 -- matching the old ROS 1 joint_cmd
        topic. Reordering to the controller's thumb-first joint order is
        handled internally.

        :param namespace: Optional namespace of the controller_manager and
        controllers (e.g. 'right' for a duo-hand launch).

        :param position_controller: Name of the forward position controller.

        :param grasp_controller: Name of the grasp (torque) controller.

        :param num_joints: Number of expected joints, used when
        commanding joint positions.

        :param spin_thread: If True, spin this node in a background
        thread so subscriptions and service calls work without the
        caller spinning.
        """
        super().__init__('allegro_client', namespace=namespace or None)

        self._num_joints = num_joints

        # ros2_allegro names the thumb joint00..03 and the fingers
        # joint10..33, so the URDF/controller order is thumb-first. The
        # allegro SDK (and the old ROS 1 joint_cmd topic, which hora's
        # observation/action remapping is written against) orders them
        # index, middle, ring, thumb -- see sdk_ordered_joint_base_names in
        # allegro_hand_hardwares/v4/hardware/src/v4_hardware_interface.cpp.
        # This class speaks SDK order; _controller_joint_names is the order
        # the forward_command_controller expects on the wire.
        self._joint_names = [
            '{}joint{}{}'.format(joint_prefix, finger, joint)
            for finger in (1, 2, 3, 0) for joint in range(4)
        ]
        self._controller_joint_names = [
            '{}joint{}{}'.format(joint_prefix, finger, joint)
            for finger in range(4) for joint in range(4)
        ]
        # _command_order[i] is the index into an SDK-ordered command of the
        # value that belongs in slot i of the controller's command array.
        self._command_order = [
            self._joint_names.index(name)
            for name in self._controller_joint_names
        ]

        ns = '/{}'.format(namespace.strip('/')) if namespace else ''
        topic_joint_command = '{}/{}/commands'.format(ns, position_controller)
        topic_joint_state = '{}/joint_states'.format(ns)

        self._position_controller = position_controller
        self._grasp_controller = grasp_controller
        self._active_controller = None

        self.pub_joint = self.create_publisher(
            Float64MultiArray, topic_joint_command, 10)
        self.create_subscription(
            JointState, topic_joint_state, self._joint_state_callback, 10)
        self._joint_state = None
        self._joint_state_event = threading.Event()

        self._grasp_client = ActionClient(
            self, GraspCommand,
            '{}/{}/grasp_cmd'.format(ns, grasp_controller))
        self._envelop_torque = -1.0  # -1 means controller default

        self._list_controllers_client = self.create_client(
            ListControllers, '{}/controller_manager/list_controllers'.format(ns))
        self._switch_controller_client = self.create_client(
            SwitchController, '{}/controller_manager/switch_controller'.format(ns))

        # _active_controller caches which controller we last activated so the
        # 20 Hz command path does not need a list_controllers round trip. The
        # cache goes stale if anything else switches controllers (or a
        # controller deactivates on error), which would silently publish
        # commands nobody is listening to. Watch each controller's lifecycle
        # transitions so the cache is invalidated without polling.
        for controller in (position_controller, grasp_controller):
            self.create_subscription(
                TransitionEvent,
                '{}/{}/transition_event'.format(ns, controller),
                lambda msg, name=controller: self._transition_callback(
                    name, msg),
                10)

        self._executor = None
        self._spin_thread = None
        if spin_thread:
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self)
            self._spin_thread = threading.Thread(
                target=self._executor.spin, daemon=True)
            self._spin_thread.start()

        self.get_logger().info(
            'Allegro Client started (position controller: {})'.format(
                position_controller))

        # "Named" grasps provided by the grasp (torque) controller. These
        # can be commanded directly and the hand will execute them. The
        # keys are more human-friendly names, the values are the command
        # strings of the GraspCommand action. Multiple strings mapping to
        # the same value are allowed.
        self._named_grasps_mappings = {
            'home': 'home',
            'ready': 'ready',
            'three_finger_grasp': 'grasp_3',
            'three finger grasp': 'grasp_3',
            'four_finger_grasp': 'grasp_4',
            'four finger grasp': 'grasp_4',
            'index_pinch': 'pinch_it',
            'index pinch': 'pinch_it',
            'middle_pinch': 'pinch_mt',
            'middle pinch': 'pinch_mt',
            'envelop': 'envelop',
            'off': 'off',
            'gravity_compensation': 'gravcomp',
            'gravity compensation': 'gravcomp',
            'gravity': 'gravcomp'
        }

    def disconnect(self):
        """
        Disconnect the allegro client from the hand by sending the 'off'
        command. This is principally a convenience binding.

        Note that we don't actually 'disconnect', so you could technically
        continue sending other commands after this.
        """
        self.command_hand_configuration('off')
        if self._executor is not None:
            self._executor.shutdown()
            # The spin thread must be joined before the node (and the rclpy
            # context behind it) is torn down, otherwise it is still inside
            # spin() during interpreter shutdown and the process aborts.
            if self._spin_thread is not None:
                self._spin_thread.join(timeout=5.0)
            self._executor = None
            self._spin_thread = None

    def _transition_callback(self, controller, msg):
        """Track controller lifecycle changes so _active_controller stays honest."""
        if msg.goal_state.label == 'active':
            self._active_controller = controller
        elif self._active_controller == controller:
            self._active_controller = None

    def _joint_state_callback(self, data):
        # The joint_state_broadcaster does not guarantee the same joint
        # ordering as the position controller command, so reorder by name.
        try:
            index = [data.name.index(name) for name in self._joint_names]
        except ValueError:
            self.get_logger().warn(
                'joint_states is missing expected hand joints; got: {}'.format(
                    data.name),
                throttle_duration_sec=5.0)
            return
        reordered = JointState()
        reordered.header = data.header
        reordered.name = list(self._joint_names)
        if data.position:
            reordered.position = [data.position[i] for i in index]
        if data.velocity:
            reordered.velocity = [data.velocity[i] for i in index]
        if data.effort:
            reordered.effort = [data.effort[i] for i in index]
        self._joint_state = reordered
        self._joint_state_event.set()

    def _call_service(self, client, request, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                'Service {} not available.'.format(client.srv_name))
            return None
        future = client.call_async(request)
        if self._executor is not None:
            if not future.done():
                event = threading.Event()
                future.add_done_callback(lambda _: event.set())
                if not event.wait(timeout):
                    self.get_logger().error(
                        'Call to {} timed out.'.format(client.srv_name))
                    return None
        else:
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    def _activate_controller(self, controller):
        """
        Make sure `controller` is the active one, deactivating the other
        hand controller if necessary. Returns True on success.
        """
        if self._active_controller == controller:
            return True

        result = self._call_service(
            self._list_controllers_client, ListControllers.Request())
        active = set()
        if result is not None:
            active = {c.name for c in result.controller if c.state == 'active'}
            if controller in active:
                self._active_controller = controller
                return True

        request = SwitchController.Request()
        request.activate_controllers = [controller]
        request.deactivate_controllers = [
            c for c in (self._position_controller, self._grasp_controller)
            if c != controller and c in active
        ]
        request.strictness = SwitchController.Request.STRICT
        result = self._call_service(self._switch_controller_client, request)
        if result is None or not result.ok:
            self.get_logger().error(
                'Failed to activate controller {}.'.format(controller))
            return False
        self._active_controller = controller
        return True

    def command_joint_position(self, desired_pose):
        """
        Command a specific desired hand pose.

        The desired pose must be the correct dimensionality (self._num_joints)
        and ordered index, middle, ring, thumb (joint10..13, joint20..23,
        joint30..33, joint00..03), the same ordering as the old allegroHand
        joint_cmd topic. Only the pose is commanded, and **no bound-checking
        happens here**: any commanded pose must be valid or Bad Things May
        Happen. (Generally, values between 0.0 and 1.5 are fine, but use
        this at your own risk.)

        :param desired_pose: The desired joint configurations.
        :return: True if pose is published, False otherwise.
        """

        # Check that the desired pose can have len() applied to it, and that
        # the number of dimensions is the same as the number of hand joints.
        if (not hasattr(desired_pose, '__len__') or
                len(desired_pose) != self._num_joints):
            self.get_logger().warn(
                'Desired pose must be a {}-d array: got {}.'
                .format(self._num_joints, desired_pose))
            return False

        if not self._activate_controller(self._position_controller):
            return False

        msg = Float64MultiArray()
        try:
            values = [float(x) for x in desired_pose]
            # Reorder from SDK order (index, middle, ring, thumb) into the
            # controller's joint order (thumb first) before publishing.
            msg.data = [values[i] for i in self._command_order]
            self.pub_joint.publish(msg)
            self.get_logger().debug('Published desired pose.')
            return True
        except (TypeError, ValueError):
            self.get_logger().warn(
                'Incorrect type for desired pose: {}.'.format(desired_pose))
            return False

    def command_joint_torques(self, desired_torques):
        """
        Command a desired torque for each joint.

        Not supported by the ros2_allegro forward position controller: the
        default controller configuration has no per-joint effort interface.
        Configure allegro_hand_position_effort_controller if you need this.

        :param desired_torques: The desired joint torques.
        :return: False (unsupported).
        """
        self.get_logger().warn(
            'Per-joint torque commands are not supported by the '
            'ros2_allegro position controller; ignoring.')
        return False

    def poll_joint_position(self, wait=False, timeout=5.0):
        """ Get the current joint positions of the hand.

        :param wait: If true, waits for a 'fresh' state reading.
        :param timeout: Max seconds to wait for a fresh reading.
        :return: (positions, efforts) in SDK order (index, middle, ring,
        thumb), or None if no state has been received.
        """
        if wait:  # Clear joint state and wait for the next reading.
            self._joint_state = None
            self._joint_state_event.clear()
            if not self._joint_state_event.wait(timeout):
                self.get_logger().warn('Timed out waiting for joint state.')
                return None

        if self._joint_state:
            return (self._joint_state.position, self._joint_state.effort)
        else:
            return None

    def command_hand_configuration(self, hand_config, wait_for_result=False):
        """
        Command a named hand configuration (e.g., index_pinch, envelop,
        gravity_compensation).

        The configurations are executed by the AllegroHandGraspController
        (GraspCommand action). More human-friendly names are used by
        defining them as 'shortcuts' in the _named_grasps_mapping variable.
        Multiple strings can map to the same commanded configuration.

        :param hand_config: A human-friendly string of the desired
        configuration.
        :param wait_for_result: If True, block until the grasp goal
        completes.
        :return: True if the grasp was known and commanded, false otherwise.
        """

        # Only use known named grasps.
        if hand_config not in self._named_grasps_mappings:
            self.get_logger().warn(
                'Unable to command unknown grasp {}'.format(hand_config))
            return False

        if not self._activate_controller(self._grasp_controller):
            return False

        goal = GraspCommand.Goal()
        goal.command = self._named_grasps_mappings[hand_config]
        goal.effort = self._envelop_torque
        self.get_logger().debug('Commanding grasp: {}'.format(goal.command))

        if not self._grasp_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Grasp action server not available.')
            return False

        if wait_for_result:
            self._grasp_client.send_goal(goal)
        else:
            self._grasp_client.send_goal_async(goal)
        return True

    def list_hand_configurations(self):
        """
        :return: List of valid strings for named hand configurations (including
        duplicates).
        """
        return self._named_grasps_mappings.keys()

    def set_envelop_torque(self, torque):
        """
        Set the envelop grasping torque, applied by the next 'envelop'
        hand configuration command.

        :param torque: Desired torque, between 0 and 1. Values outside this
        range are clamped.
        :return: True.
        """

        self._envelop_torque = max(0.0, min(1.0, torque))
        return True
