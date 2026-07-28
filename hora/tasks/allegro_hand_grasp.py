# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Rewritten for IsaacLab. See docs/isaaclab_migration.md.
# --------------------------------------------------------

"""Grasp-pose generation.

Produces the ``cache/*.npy`` files that the in-hand rotation task resets from. This is a
filter, not a policy: close the hand from a canonical pose plus noise, step with zero
actions under position control, and keep the environments where the object survived.

The IsaacGym version enumerated every contact in an env with
``gym.get_env_rigid_contacts()`` (CPU-only, hence its ``assert device == 'cpu'``) and
identified fingertip/object pairs by hashing global rigid-body indices. IsaacLab has no
equivalent query by design -- materialising full contact reports for thousands of envs on
GPU is the cost its tensor API exists to avoid. Instead contacts are *declared* up front:
a :class:`ContactSensor` per fingertip, filtered to the object, giving per-pair forces.

That makes this a net upgrade -- contact sensing runs on GPU, so grasp generation is no
longer pinned to the CPU pipeline.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from isaaclab.sensors import ContactSensor

from hora.tasks.allegro_hand_grasp_cfg import FINGER_NAMES, AllegroHandGraspEnvCfg
from hora.tasks.allegro_hand_hora import AllegroHandHora
from hora.tasks.allegro_hand_hora_cfg import HORA_FINGERTIP_BODIES
from hora.utils.math_utils import quat_wxyz_to_xyzw, tensor_clamp, torch_rand_float

# Number of poses collected before a cache file is written.
CACHE_TARGET = 50_000

# Fingertip-to-object distance below which a fingertip counts as "near" (metres).
NEAR_OBJECT_DIST = 0.1


class AllegroHandGrasp(AllegroHandHora):
    """Collects stable grasp poses into cache/<name>_grasp_50k_s<scale>.npy."""

    cfg: AllegroHandGraspEnvCfg

    def __init__(self, cfg: AllegroHandGraspEnvCfg, task_cfg: dict,
                 render_mode: str | None = None, **kwargs):
        super().__init__(cfg, task_cfg, render_mode=render_mode, **kwargs)

        # The pose the hand closes from. Values are in hora joint order.
        self.canonical_pose = torch.tensor([
            0.082, 1.244, 0.265, 0.298, 1.104, 1.163, 0.953, -0.138,
            0.005, 1.096, 0.080, 0.150, 0.029, 1.337, 0.285, 0.317,
        ], device=self.device)

        # [16 joint pos, 3 object xyz, 4 object quat] -- see _record_successes for the
        # quaternion convention.
        self.saved_grasping_states = torch.zeros((0, 23), device=self.device)
        self.cache_complete = False

        idx, names = self.hand.find_bodies(HORA_FINGERTIP_BODIES, preserve_order=True)
        if names != HORA_FINGERTIP_BODIES:
            raise RuntimeError(
                f'fingertip bodies not found as expected.\n'
                f'  wanted: {HORA_FINGERTIP_BODIES}\n  got:    {names}')
        self._fingertip_idx = idx

        # PhysX reports contact *forces*, where IsaacGym reported contact *existence*, so
        # "is this fingertip touching the object" needs a cutoff that did not exist
        # before. It changes which grasps are accepted, so it is configurable rather
        # than buried.
        #
        # 0.1 N comes from the measured distribution (gen_grasp.py
        # --report-contact-forces): fingertip/object forces run p05 0.068, p50 0.40,
        # p99 2.44 N. 0.1 clears the noise floor while keeping ~93% of real contacts.
        # Re-measure if the hand, object set or controller gains change.
        self.contact_force_threshold = float(
            task_cfg['env'].get('contactForceThreshold', 0.1))

        # Steps at the start of an episode during which failure is not evaluated.
        #
        # The IsaacGym version had none: it reset an env the instant the three
        # conditions stopped holding, which only works if the object is already touching
        # the fingers the moment it spawns. That was true of its exact collision
        # geometry; the URDF->USD conversion produces slightly different colliders, and
        # here the object spawns ~1 cm clear of the fingertips. With no grace period it
        # fails on step 1, is teleported back to the spawn height, and never falls in --
        # the object literally never moves and no pose is ever collected.
        self.settle_steps = int(task_cfg['env'].get('graspSettleSteps', 10))

        # Where cache files are written. Defaults to cache/, which is also where the
        # *published* caches live -- point this elsewhere when validating generated
        # output so a half-finished run cannot clobber known-good data.
        self.cache_dir = str(task_cfg['env'].get('graspCacheDir', 'cache'))

    # ------------------------------------------------------------------ scene

    def _setup_extra_sensors(self):
        self._contact_sensors = []
        for name, sensor_cfg in zip(FINGER_NAMES, self.cfg.contact_sensors):
            sensor = ContactSensor(sensor_cfg)
            self.scene.sensors[f'contact_{name}'] = sensor
            self._contact_sensors.append(sensor)

    # ------------------------------------------------------------------ contact

    def fingertip_contact_forces(self) -> torch.Tensor:
        """Per-fingertip contact force magnitude against the object. Shape (num_envs, 4).

        Each sensor is filtered to the object alone, so ``force_matrix_w`` is
        (num_envs, 1, 1, 3) and collapses to one magnitude per env per finger.
        """
        forces = []
        for sensor in self._contact_sensors:
            matrix = sensor.data.force_matrix_w
            if matrix is None:
                # No filtered pair resolved -- treat as no contact rather than crashing
                # mid-collection, but make it visible.
                forces.append(torch.zeros(self.num_envs, device=self.device))
                continue
            forces.append(torch.norm(matrix.view(self.num_envs, -1, 3), dim=-1).sum(-1))
        return torch.stack(forces, dim=-1)

    # ------------------------------------------------------------------ success test

    def _grasp_is_stable(self) -> torch.Tensor:
        """The three conditions the IsaacGym version used, unchanged in meaning."""
        object_pos = self.object_pos
        fingertip_pos = (self.hand.data.body_pos_w[:, self._fingertip_idx]
                         - self.scene.env_origins[:, None, :])

        # 1) every fingertip close to the object. A relative distance, so the env-origin
        #    offset would cancel anyway -- subtracted above for consistency.
        dist = torch.norm(object_pos[:, None, :] - fingertip_pos, dim=-1)
        near = torch.less(dist, NEAR_OBJECT_DIST).all(-1)

        # 2) at least two fingertips actually touching it
        contacts = torch.greater(self.fingertip_contact_forces(),
                                 self.contact_force_threshold).sum(-1)
        touching = torch.greater_equal(contacts, 2)

        # 3) object has not been dropped
        held = torch.greater(object_pos[:, -1], self.reset_z_threshold)

        return near & touching & held

    def _get_rewards(self) -> torch.Tensor:
        # Grasp generation has no reward; poses are filtered, not optimised. The
        # IsaacGym version also computed none -- it only overrode compute_reward to set
        # reset flags, which is _get_dones here.
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        settling = self.episode_length_buf < self.settle_steps
        failed = (~self._grasp_is_stable()) & (~settling)
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        # A pose is only harvested if it survived a full episode, so "reached the time
        # limit" is success here -- the opposite of its meaning in the rotation task.
        return failed, timed_out

    # ------------------------------------------------------------------ collection

    def _record_successes(self, env_ids):
        """Append envs that survived a full episode, and write the file when full."""
        survived = self.episode_length_buf[env_ids] >= self.max_episode_length - 1
        if not survived.any():
            return
        winners = env_ids[survived]

        root_pos = self.object.data.root_pos_w[winners] - self.scene.env_origins[winners]
        root_quat = self.object.data.root_quat_w[winners]
        states = torch.cat([
            self.hand_dof_pos[winners],
            root_pos,
            # Stored xyzw, matching the published caches exactly, so generated and
            # downloaded files stay interchangeable. _load_grasp_cache converts to
            # IsaacLab's wxyz on the way in.
            quat_wxyz_to_xyzw(root_quat),
        ], dim=1)
        self.saved_grasping_states = torch.cat([self.saved_grasping_states, states])

        n = self.saved_grasping_states.shape[0]
        print(f'\rgrasp cache: {n} / {CACHE_TARGET}', end='', flush=True)
        if n >= CACHE_TARGET:
            self.save_cache()
            self.cache_complete = True

    def save_cache(self) -> str:
        scale = self.cfg.object_scales[0]
        name = os.path.join(
            self.cache_dir,
            f'{self.grasp_cache_name}_grasp_50k_s{str(scale).replace(".", "")}.npy')
        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(name, self.saved_grasping_states[:CACHE_TARGET].cpu().numpy())
        print(f'\nwrote {name} '
              f'({min(self.saved_grasping_states.shape[0], CACHE_TARGET)} poses)')
        return name

    # ------------------------------------------------------------------ reset

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # Harvest before anything is overwritten by the reset itself.
        if len(env_ids) > 0 and hasattr(self, 'saved_grasping_states'):
            self._record_successes(env_ids)

        # Skip AllegroHandHora._reset_idx: it resets *from* the grasp cache, which is
        # exactly what does not exist yet here.
        super(AllegroHandHora, self)._reset_idx(env_ids)

        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper,
                (len(env_ids), self.num_actions), device=self.device)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper,
                (len(env_ids), self.num_actions), device=self.device)

        # Object back to its spawn pose, at rest and unrotated.
        root_state = self.object.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] += self.scene.env_origins[env_ids]
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        root_state[:, 7:13] = 0.0
        self.object.write_root_state_to_sim(root_state, env_ids=env_ids)

        # Hand to the canonical pose plus noise, which is what makes the resulting cache
        # a distribution of grasps rather than one repeated pose.
        pos = self.canonical_pose[None].repeat(len(env_ids), 1)
        pos = pos + 0.25 * torch_rand_float(
            -1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        pos = tensor_clamp(pos, self.hand_dof_lower, self.hand_dof_upper)

        self.hand.write_joint_state_to_sim(
            pos, torch.zeros_like(pos), joint_ids=self._joint_idx, env_ids=env_ids)
        self.prev_targets[env_ids] = pos
        self.cur_targets[env_ids] = pos
        self.init_pose_buf[env_ids] = pos.clone()
        # Same reason as in AllegroHandHora._reset_idx: the joints were just teleported,
        # so the finite-difference buffer has to follow or the first substep sees a
        # spurious velocity spike.
        self._prev_dof_pos[env_ids] = pos
        if not self.torque_control:
            self.hand.set_joint_position_target(
                pos, joint_ids=self._joint_idx, env_ids=env_ids)

        self._obs[env_ids] = 0
        self.obs_buf_lag_history[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1
