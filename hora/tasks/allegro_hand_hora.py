# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Ported from IsaacGym to IsaacLab. See docs/isaaclab_migration.md.
# --------------------------------------------------------

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationCfg  # noqa: F401  (re-exported for cfg typing)
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

from hora.tasks.allegro_hand_hora_cfg import HORA_JOINT_ORDER, AllegroHandHoraEnvCfg
from hora.utils.math_utils import quat_xyzw_to_wxyz, tensor_clamp, torch_rand_float, unscale
from hora.utils.misc import tprint


class AllegroHandHora(DirectRLEnv):
    """In-hand object rotation.

    Ordering note, because it is the single easiest thing to get wrong here: everything
    policy-facing (observations, actions, targets, joint limits, the grasp cache) is in
    *hora order* -- index, thumb, middle, ring. PhysX reports joints breadth-first, in a
    different order entirely. ``self._joint_idx`` maps hora order onto PhysX order and is
    passed as ``joint_ids=`` on every read and write, so no other code in this file has
    to think about it.
    """

    cfg: AllegroHandHoraEnvCfg

    def __init__(self, cfg: AllegroHandHoraEnvCfg, task_cfg: dict,
                 render_mode: str | None = None, **kwargs):
        self.task_cfg = task_cfg
        self._setup_domain_rand_config(task_cfg['env']['randomization'])
        self._setup_priv_option_config(task_cfg['env']['privInfo'])
        self._setup_reward_config(task_cfg['env']['reward'])

        controller = task_cfg['env']['controller']
        self.torque_control = controller['torque_control']
        self.p_gain_val = controller['pgain']
        self.d_gain_val = controller['dgain']
        self.control_freq_inv = controller['controlFrequencyInv']

        self.clip_obs = task_cfg['env'].get('clipObservations', np.inf)
        self.clip_actions = task_cfg['env'].get('clipActions', np.inf)
        self.reset_z_threshold = task_cfg['env']['reset_height_threshold']
        self.grasp_cache_name = task_cfg['env']['grasp_cache_name']
        self.base_obj_scale = task_cfg['env']['baseObjScale']
        self.save_init_pose = task_cfg['env']['genGrasps']
        self.evaluate = task_cfg['on_evaluation']

        self.prop_hist_len = task_cfg['env']['hora']['propHistoryLen']
        self.num_env_factors = task_cfg['env']['hora']['privInfoDim']
        self.priv_info_dict = {
            'obj_position': (0, 3),
            'obj_scale': (3, 4),
            'obj_mass': (4, 5),
            'obj_friction': (5, 6),
            'obj_com': (6, 9),
        }

        super().__init__(cfg, render_mode, **kwargs)

        # hora's PPO/ProprioAdapt read `.shape`, `.low` and `.high` off these and expect
        # the *single*-env spaces, not the batched ones DirectRLEnv installs.
        self.observation_space = self.single_observation_space['policy']
        self.action_space = self.single_action_space

        self.num_actions = self.cfg.action_space
        self.num_obs = self.cfg.observation_space

        self._resolve_joint_order()
        self._allocate_buffers()
        self._load_grasp_cache()

    # ------------------------------------------------------------------ setup

    def _resolve_joint_order(self):
        """Pin the hora <-> PhysX joint mapping, and fail loudly if the asset changed."""
        idx, names = self.hand.find_joints(HORA_JOINT_ORDER, preserve_order=True)
        if names != HORA_JOINT_ORDER:
            raise RuntimeError(
                f'hand asset joints do not match the expected set.\n'
                f'  wanted: {HORA_JOINT_ORDER}\n  got:    {names}')
        self._joint_idx = idx

        limits = self.hand.data.joint_pos_limits[0, self._joint_idx]
        self.hand_dof_lower = limits[:, 0].clone()
        self.hand_dof_upper = limits[:, 1].clone()
        self.num_hand_dofs = len(idx)

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        spawn_ground_plane(self)

        # copy_from_source=False keeps the clones as USD references into one source prim,
        # which is what makes replicate_physics viable at this env count.
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations['hand'] = self.hand
        self.scene.rigid_objects['object'] = self.object

        light_cfg = _dome_light_cfg()
        light_cfg.func('/World/Light', light_cfg)

    def _allocate_buffers(self):
        n, device = self.num_envs, self.device

        # NOT `obs_buf`: DirectRLEnv owns that attribute and stores the observation
        # dict in it on every step, which would silently clobber this tensor.
        self._obs = torch.zeros((n, self.num_obs), device=device)
        # 80 frames of (16 joint pos + 16 targets); observations read the last 3 and the
        # proprioceptive history reads the last `prop_hist_len`.
        self.obs_buf_lag_history = torch.zeros((n, 80, self.num_obs // 3), device=device)
        self.priv_info_buf = torch.zeros((n, self.num_env_factors), device=device)
        self.proprio_hist_buf = torch.zeros((n, self.prop_hist_len, 32), device=device)

        self.at_reset_buf = torch.ones(n, device=device, dtype=torch.long)
        self.prev_targets = torch.zeros((n, self.num_hand_dofs), device=device)
        self.cur_targets = torch.zeros((n, self.num_hand_dofs), device=device)
        self.init_pose_buf = torch.zeros((n, self.num_hand_dofs), device=device)
        self.actions = torch.zeros((n, self.num_actions), device=device)
        self.torques = torch.zeros((n, self.num_actions), device=device)
        self.dof_vel_finite_diff = torch.zeros((n, self.num_hand_dofs), device=device)
        self.rot_axis_buf = torch.zeros((n, 3), device=device)

        self.p_gain = torch.ones((n, self.num_actions), device=device) * self.p_gain_val
        self.d_gain = torch.ones((n, self.num_actions), device=device) * self.d_gain_val

        self.object_pos_prev = self.object_pos.clone()
        self.object_rot_prev = self.object_rot.clone()

        # random-force perturbation
        self.force_scale = self.task_cfg['env'].get('forceScale', 0.0)
        self.random_force_prob_scalar = self.task_cfg['env'].get('randomForceProbScalar', 0.0)
        self.force_decay = self.task_cfg['env'].get('forceDecay', 0.99)
        self.force_decay_interval = self.task_cfg['env'].get('forceDecayInterval', 0.08)
        self.rb_forces = torch.zeros((n, 1, 3), device=device)

        # evaluation statistics
        self.stat_sum_rewards = 0
        self.stat_sum_rotate_rewards = 0
        self.stat_sum_episode_length = 0
        self.stat_sum_obj_linvel = 0
        self.stat_sum_torques = 0
        self.env_evaluated = 0
        self.max_evaluate_envs = 500000

    def _load_grasp_cache(self):
        """Load the per-scale grasp caches, converting them out of IsaacGym conventions.

        Layout is [16 joint pos (hora order), 3 object xyz, 4 object quat]. The quat is
        **xyzw** because the caches were generated by the old IsaacGym code; the joint
        columns are already in hora order (verified against the thumb's strictly-positive
        joint_12_0 range).
        """
        scales = self.randomize_scale_list if self.randomize_scale else [self.base_obj_scale]
        self.saved_grasping_states = {}
        for s in scales:
            name = f'cache/{self.grasp_cache_name}_grasp_50k_s{str(s).replace(".", "")}.npy'
            if not os.path.isfile(name):
                raise FileNotFoundError(
                    f'grasp cache not found: {name}\n'
                    'Download the published caches (see README.md) or generate them with '
                    'gen_grasp.py (M3 of docs/isaaclab_migration.md).')
            data = torch.from_numpy(np.load(name)).float().to(self.device)
            data[:, 19:23] = quat_xyzw_to_wxyz(data[:, 19:23])
            self.saved_grasping_states[str(s)] = data

    # ------------------------------------------------------------------ properties

    @property
    def object_pos(self) -> torch.Tensor:
        """Object position in *env-local* frame.

        ``root_pos_w`` includes the env origin; every threshold and offset in this task
        is expressed relative to the hand, so the origin has to come back out.
        """
        return self.object.data.root_pos_w - self.scene.env_origins

    @property
    def object_rot(self) -> torch.Tensor:
        return self.object.data.root_quat_w

    @property
    def hand_dof_pos(self) -> torch.Tensor:
        return self.hand.data.joint_pos[:, self._joint_idx]

    @property
    def hand_dof_vel(self) -> torch.Tensor:
        return self.hand.data.joint_vel[:, self._joint_idx]

    # ------------------------------------------------------------------ stepping

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = torch.clamp(actions.clone(), -self.clip_actions, self.clip_actions)
        targets = self.prev_targets + 1 / 24 * self.actions
        self.cur_targets[:] = tensor_clamp(targets, self.hand_dof_lower, self.hand_dof_upper)
        self.prev_targets[:] = self.cur_targets.clone()

        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos

        if self.force_scale > 0.0:
            self._apply_random_forces()

    def _apply_random_forces(self):
        self.rb_forces *= torch.pow(
            torch.tensor(self.force_decay, device=self.device),
            self.cfg.sim.dt / self.force_decay_interval)
        obj_mass = self.object.data.default_mass.to(self.device).sum(-1)
        force_indices = torch.less(
            torch.rand(self.num_envs, device=self.device), self.random_force_prob_scalar)
        rand = torch.randn((self.num_envs, 1, 3), device=self.device)
        self.rb_forces[force_indices] = (
            rand[force_indices] * obj_mass[force_indices, None, None] * self.force_scale)
        self.object.set_external_force_and_torque(
            self.rb_forces, torch.zeros_like(self.rb_forces))

    def _apply_action(self):
        """Called once per physics step inside the decimation loop -- hora's 120 Hz PD."""
        previous_dof_pos = self.hand_dof_pos.clone()
        self.hand.update(self.cfg.sim.dt)

        if self.torque_control:
            dof_pos = self.hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.cfg.sim.dt
            self.dof_vel_finite_diff = dof_vel.clone()
            torques = self.p_gain * (self.cur_targets - dof_pos) - self.d_gain * dof_vel
            self.torques = torch.clip(torques, -0.5, 0.5).clone()
            self.hand.set_joint_effort_target(self.torques, joint_ids=self._joint_idx)
        else:
            self.hand.set_joint_position_target(self.cur_targets, joint_ids=self._joint_idx)

    # ------------------------------------------------------------------ observations

    def _get_observations(self) -> dict:
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()

        joint_noise = (torch.rand_like(self.hand_dof_pos) * 2.0 - 1.0) * self.joint_noise_scale
        cur_obs_buf = unscale(
            joint_noise + self.hand_dof_pos, self.hand_dof_lower, self.hand_dof_upper
        ).clone().unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # Freshly reset envs have no history, so backfill every frame with the current
        # pose rather than letting zeros leak into the observation.
        at_reset = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(at_reset) > 0:
            self.obs_buf_lag_history[at_reset, :, 0:16] = unscale(
                self.hand_dof_pos[at_reset], self.hand_dof_lower, self.hand_dof_upper
            ).clone().unsqueeze(1)
            self.obs_buf_lag_history[at_reset, :, 16:32] = \
                self.hand_dof_pos[at_reset].unsqueeze(1)
            self.at_reset_buf[at_reset] = 0

        t_buf = self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1).clone()
        self._obs[:, :t_buf.shape[1]] = t_buf
        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()
        self._update_priv_buf(slice(None), 'obj_position', self.object_pos.clone())

        obs = torch.clamp(self._obs, -self.clip_obs, self.clip_obs)
        # 'policy' is the key IsaacLab/gymnasium expect; 'obs' is the alias hora's
        # trainers use. Same tensor, so the duplication costs nothing.
        return {
            'policy': obs,
            'obs': obs,
            'priv_info': self.priv_info_buf,
            'proprio_hist': self.proprio_hist_buf,
        }

    def _update_priv_buf(self, env_id, name, value, lower=None, upper=None):
        s, e = self.priv_info_dict[name]
        if getattr(self, f'enable_priv_{name}'):
            if isinstance(value, list):
                value = torch.tensor(value, dtype=torch.float, device=self.device)
            if lower is not None and upper is not None:
                value = (2.0 * value - upper - lower) / (upper - lower)
            self.priv_info_buf[env_id, s:e] = value
        else:
            self.priv_info_buf[env_id, s:e] = 0

    # ------------------------------------------------------------------ reward / done

    def _get_rewards(self) -> torch.Tensor:
        self.rot_axis_buf[:, -1] = -1

        pose_diff_penalty = ((self.hand_dof_pos - self.init_pose_buf) ** 2).sum(-1)
        torque_penalty = (self.torques ** 2).sum(-1)
        work_penalty = ((self.torques * self.dof_vel_finite_diff).sum(-1)) ** 2

        # Angular velocity from the frame-to-frame quaternion delta. axis_angle_from_quat
        # is IsaacLab's wxyz-native replacement for hora's hand-rolled xyzw helper.
        angdiff = axis_angle_from_quat(
            quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
        object_angvel = angdiff / (self.control_freq_inv * self.cfg.sim.dt)
        vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
        rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=self.angvel_clip_min)

        object_linvel = (self.object_pos - self.object_pos_prev) / \
            (self.control_freq_inv * self.cfg.sim.dt)
        object_linvel_penalty = torch.norm(object_linvel, p=1, dim=-1)

        reward = (
            self.rotate_reward_scale * rotate_reward
            + object_linvel_penalty * self.object_linvel_penalty_scale
            + pose_diff_penalty * self.pose_diff_penalty_scale
            + torque_penalty * self.torque_penalty_scale
            + work_penalty * self.work_penalty_scale
        )

        self.extras['rotation_reward'] = rotate_reward.mean()
        self.extras['object_linvel_penalty'] = object_linvel_penalty.mean()
        self.extras['pose_diff_penalty'] = pose_diff_penalty.mean()
        self.extras['work_done'] = work_penalty.mean()
        self.extras['torques'] = torque_penalty.mean()
        self.extras['roll'] = object_angvel[:, 0].mean()
        self.extras['pitch'] = object_angvel[:, 1].mean()
        self.extras['yaw'] = object_angvel[:, 2].mean()

        if self.evaluate:
            self._update_eval_stats(reward, rotate_reward)
        return reward

    def _update_eval_stats(self, reward, rotate_reward):
        dropped = torch.less(self.object_pos[:, -1], self.reset_z_threshold)
        self.stat_sum_rewards += reward.sum()
        self.stat_sum_rotate_rewards += rotate_reward.sum()
        self.stat_sum_torques += self.torques.abs().sum()
        self.stat_sum_obj_linvel += (self.object.data.root_lin_vel_w ** 2).sum(-1).sum()
        self.stat_sum_episode_length += (~dropped).sum()
        self.env_evaluated += dropped.sum()
        if self.env_evaluated > 0:
            tprint(
                f'progress {self.env_evaluated} / {self.max_evaluate_envs} | '
                f'reward: {self.stat_sum_rewards / self.env_evaluated:.2f} | '
                f'eps length: {self.stat_sum_episode_length / self.env_evaluated:.2f} | '
                f'rotate reward: {self.stat_sum_rotate_rewards / self.env_evaluated:.2f} | '
                f'lin vel (x100): '
                f'{self.stat_sum_obj_linvel * 100 / self.stat_sum_episode_length:.4f} | '
                f'command torque: {self.stat_sum_torques / self.stat_sum_episode_length:.2f}')

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """IsaacGym merged these into one reset_buf; gymnasium wants them apart.

        Keeping them separate is what lets PPO bootstrap the value function through a
        timeout without treating it as a real terminal state.
        """
        dropped = torch.less(self.object_pos[:, -1], self.reset_z_threshold)
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return dropped, timed_out

    # ------------------------------------------------------------------ reset

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper,
                (len(env_ids), self.num_actions), device=self.device)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper,
                (len(env_ids), self.num_actions), device=self.device)

        scales = self.randomize_scale_list if self.randomize_scale else [self.base_obj_scale]
        num_scales = len(scales)
        for n_s, obj_scale in enumerate(scales):
            s_ids = env_ids[(env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)]
            if len(s_ids) == 0:
                continue
            cache = self.saved_grasping_states[str(obj_scale)]
            sampled = cache[torch.randint(cache.shape[0], (len(s_ids),), device=self.device)]

            root_state = self.object.data.default_root_state[s_ids].clone()
            root_state[:, 0:3] = sampled[:, 16:19] + self.scene.env_origins[s_ids]
            root_state[:, 3:7] = sampled[:, 19:23]
            root_state[:, 7:13] = 0
            self.object.write_root_state_to_sim(root_state, env_ids=s_ids)

            pos = sampled[:, :16]
            self.hand.write_joint_state_to_sim(
                pos, torch.zeros_like(pos), joint_ids=self._joint_idx, env_ids=s_ids)
            self.prev_targets[s_ids] = pos
            self.cur_targets[s_ids] = pos
            self.init_pose_buf[s_ids] = pos.clone()
            self._update_priv_buf(s_ids, 'obj_scale', obj_scale)

        if not self.torque_control:
            self.hand.set_joint_position_target(
                self.prev_targets[env_ids], joint_ids=self._joint_idx, env_ids=env_ids)

        self._obs[env_ids] = 0
        self.obs_buf_lag_history[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    # ------------------------------------------------------------------ config plumbing

    def _setup_domain_rand_config(self, rand_config):
        self.randomize_mass = rand_config['randomizeMass']
        self.randomize_mass_lower = rand_config['randomizeMassLower']
        self.randomize_mass_upper = rand_config['randomizeMassUpper']
        self.randomize_com = rand_config['randomizeCOM']
        self.randomize_com_lower = rand_config['randomizeCOMLower']
        self.randomize_com_upper = rand_config['randomizeCOMUpper']
        self.randomize_friction = rand_config['randomizeFriction']
        self.randomize_friction_lower = rand_config['randomizeFrictionLower']
        self.randomize_friction_upper = rand_config['randomizeFrictionUpper']
        self.randomize_scale = rand_config['randomizeScale']
        self.scale_list_init = rand_config['scaleListInit']
        self.randomize_scale_list = rand_config['randomizeScaleList']
        self.randomize_pd_gains = rand_config['randomizePDGains']
        self.randomize_p_gain_lower = rand_config['randomizePGainLower']
        self.randomize_p_gain_upper = rand_config['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_config['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_config['randomizeDGainUpper']
        self.joint_noise_scale = rand_config['jointNoiseScale']

    def _setup_priv_option_config(self, p_config):
        self.enable_priv_obj_position = p_config['enableObjPos']
        self.enable_priv_obj_mass = p_config['enableObjMass']
        self.enable_priv_obj_scale = p_config['enableObjScale']
        self.enable_priv_obj_com = p_config['enableObjCOM']
        self.enable_priv_obj_friction = p_config['enableObjFriction']

    def _setup_reward_config(self, r_config):
        self.angvel_clip_min = r_config['angvelClipMin']
        self.angvel_clip_max = r_config['angvelClipMax']
        self.rotate_reward_scale = r_config['rotateRewardScale']
        self.object_linvel_penalty_scale = r_config['objLinvelPenaltyScale']
        self.pose_diff_penalty_scale = r_config['poseDiffPenaltyScale']
        self.torque_penalty_scale = r_config['torquePenaltyScale']
        self.work_penalty_scale = r_config['workPenaltyScale']


def _dome_light_cfg():
    import isaaclab.sim as sim_utils
    return sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))


def spawn_ground_plane(env):
    import isaaclab.sim as sim_utils
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func('/World/ground', cfg)


__all__ = ['AllegroHandHora', 'gym']
