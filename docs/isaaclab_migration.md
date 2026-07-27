# IsaacGym → IsaacLab Migration Plan

Companion to `README_ROS2_ISAACLAB.md`. The ROS 1 → ROS 2 half is done and verified;
this document covers the simulation half.

**Target end state:** no `isaacgym`, no `gym`, no `rospy` imports anywhere. Training and
sim-eval run in the `env_isaaclab` conda env on the host with no ROS involvement.
Deployment runs in the ROS 2 devcontainer, against either hardware or a ROS-bridged
Isaac Sim.

---

## Decision 0: object scale randomization (resolved)

This constrains the scene layout, so it is settled before anything else.

Hora assigns object scale **once at env creation** and never changes it:

```python
obj_scale = np.random.uniform(scale_list[i % 9] - 0.025, scale_list[i % 9] + 0.025)
self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)
```

Reset then selects a grasp cache by *nominal* scale via `env_ids % num_scales == n_s`.
So there is no runtime rescaling to reproduce — only per-env heterogeneous geometry at
construction time.

**Approach:** a single `RigidObjectCfg` whose spawner is a `MultiAssetSpawnerCfg`
holding one `UsdFileCfg` per scale variant (same USD, different uniform `scale=(s,s,s)`),
with `random_choice=False` so variants are assigned round-robin — reproducing `i % 9`
exactly. Verify the round-robin semantics of that flag against the installed IsaacLab
version before relying on it; if it differs, spawn per-env explicitly in a loop, which
`replicate_physics=False` permits anyway.

**Consequences, all accepted:**

- Requires `InteractiveSceneCfg.replicate_physics = False`. Scene construction is much
  slower and heavier, capping env count. Not a new limit in practice — a 12 GB 3060 will
  not reach `numEnvs: 16384` under IsaacLab regardless.
- The ±0.025 jitter is lost if only 9 variants are used. Recover it by emitting
  9 × k variants (e.g. 45) — they are xform scales on one USD, not distinct meshes, so
  extra variants are nearly free. Keep the count a multiple of 9 so `i % 9` still selects
  the right grasp cache.
- Because assignment is deterministic, the exact per-env scale is known at construction
  and can be written straight into `priv_info` (`obj_scale`), as today.
- Mass is overridden by a randomization event term, so USD scale not recomputing mass is
  harmless.

**Staging:** milestone M1 runs single-scale with `replicate_physics=True` (fast, validates
everything else); M2 turns on multi-scale and eats the construction cost.

---

## Milestone map

| M | Goal | Exit criterion |
|---|---|---|
| M0 | Env + assets ready ✅ **done** | `assets/usd/` populated (96/96); IsaacLab 2.3.1 installed in `env_isaaclab`; `verify_hand_asset.py` passes |
| M1 | Core env ports, single scale, no ROS | `train.py` runs stage 1 PPO; reward curve rises |
| M2 | Domain randomization + multi-scale | Stage 1 parity with published HORA results |
| M3 | Grasp generation | `gen_grasp.py` regenerates the 9 caches |
| M4 | ROS 2 sim-in-the-loop | `deploy.py` drives sim through ros2_control, unmodified |

M3 depends on M0 only, so it can run in parallel with M1/M2 if the caches must be
regenerated early. Note `cache/` is currently **empty** — either re-download the published
caches or finish M3 before M2 can be validated.

---

## M0 — Environment and assets

### `environment_isaaclab.yaml` — MODIFY

Currently stops at `isaacsim`. Add the IsaacLab source install and the host-side training
deps that the devcontainer does *not* provide (training runs on the host, so the
container's `requirements.txt` is irrelevant to it).

- Keep `isaacsim[all,extscache]==5.1.0` and torch 2.7.0+cu128.
- Install IsaacLab from source into the same env (`git clone` + `./isaaclab.sh -i`; it
  detects and reuses the existing pip `isaacsim`). Pin to the release line that pairs with
  Isaac Sim 5.1 — check the compatibility table rather than assuming a version number.
- Add `termcolor`. `hydra-core` and `omegaconf` arrive as IsaacLab dependencies;
  `tensorboard` arrives with the Isaac Sim stack.
- Do **not** add `gym`.

Source install rather than `pip install isaaclab`, for three reasons specific to this port:
the `Isaac-Repose-Cube-Allegro-Direct-v0` reference env (same robot, same task family,
same `DirectRLEnv` base) is only in the repo; `scripts/tools/convert_urdf.py` is only in
the repo; and the scale work above will likely need IsaacLab internals inspected.

### `assets/allegro/*.urdf` and `assets/allegro/meshes/` — MODIFY (done)

The two hand URDFs could not be imported at all as shipped. Fixed at the source rather
than worked around at conversion time, so no bespoke converter has to be maintained:

- Link and joint names sanitized: `joint_0.0` → `joint_0_0`, `link_0.0` → `link_0_0`.
  `.` is illegal in a USD prim path.
- The 12 dotted mesh files renamed to match (`link_0.0.obj` → `link_0_0.obj`), via
  `git mv` so history follows.
- Mesh references rebased from `allegro/meshes/allegro/X.obj` (relative to `assets/`, an
  IsaacGym `load_asset` convention) to `meshes/allegro/X.obj` (relative to the URDF,
  which is what every other importer expects).

Nothing in the codebase referenced these names — hora indexes joints positionally — so
the rename is contained. `assets/ycb/*.urdf` and the cuboid/cylinder primitives needed no
change. **This does break the old IsaacGym asset path**, which is fine: `isaacgym` is not
installed here and that path is being deleted anyway.

### `scripts/convert_assets.sh` — NEW

~40-line loop over IsaacLab's own `scripts/tools/convert_urdf.py`, covering all 96 assets:
`assets/allegro/*.urdf`, `assets/{ball,cube,cylinder}.urdf`, `assets/cuboid/**/*.urdf`,
`assets/cylinder/**/*.urdf`, `assets/ycb/*.urdf`. Output mirrors the source tree under
`assets/usd/` (gitignored), so `_setup_object_info`'s glob logic ports over with a
directory/extension swap.

Deliberately not our own converter — deferring to the upstream script means nothing here
needs updating as IsaacLab moves. The cost is one Isaac Sim launch per asset (~1 hour for
96), which is fine for a one-time step.

Hands get `--fix-base --merge-joints` (matching IsaacGym's `fix_base_link` /
`collapse_fixed_joints`) plus zeroed drive gains, since hora runs its own PD and PhysX
must not also control the joints. Objects are single free rigid bodies.

### `scripts/verify_hand_asset.py` — NEW

Loads the converted hand, resolves hora's joint order against PhysX's, and cross-checks
the resulting limits against `allegro_dof_lower`/`upper` in `algo/deploy/deploy.py`. Kept
separate from conversion because it is a migration invariant, not build tooling — M1 will
re-run it whenever the asset or the joint mapping changes. It earned its keep immediately;
see "M0 findings" below.

### M0 findings

Five things turned up during M0 that change assumptions made elsewhere in this plan.

**1. PhysX orders DOFs breadth-first, not depth-first.** This plan originally claimed a
DFS over the URDF gives hora's order. That is wrong. PhysX interleaves the four finger
chains by depth level:

```
PhysX: j0, j12, j4, j8 | j1, j13, j5, j9 | j2, j14, j6, j10 | j3, j15, j7, j11
       \___ level 0 __/  \___ level 1 __/  \___ level 2 ___/  \___ level 3 ___/
hora:  j0, j1, j2, j3   | j12, j13, j14, j15 | j4..j7        | j8..j11
       \___ index ____/  \_____ thumb ______/  \_ middle __/   \_ ring __/
```

hora is finger-major, PhysX is level-major, so the two differ by a 4×4 transpose
(`physx_index = 4 * (hora_index % 4) + hora_index // 4`). Note the finger order *within*
each PhysX level is index, thumb, middle, ring — which is exactly hora's finger order, and
the same reordering `_obs_allegro2hora` performs. **M1 must reorder on every joint read
and write.** Use `find_joints(HORA_JOINT_ORDER, preserve_order=True)`; never assume URDF
declaration order.

**2. Dotted names are illegal in USD, and conversion fails on them.** hora named links and
joints `link_0.0` / `joint_0.0`. The importer sanitizes *link* names to `link_0_0` itself,
but derives visual prim names from the mesh file stem (`link_0.0.obj`) without sanitizing,
producing the ill-formed path `</visuals/link_0_0/link_0.0>` and aborting with
`RuntimeError: Used null prim`. Fixed at the source (see above). **Consequence: joints are
`joint_0_0`… everywhere now.** Anything looking a joint up by name must use that form.

**3. Mesh paths resolved from a different base.** hora's URDFs referenced
`allegro/meshes/allegro/base_link.obj`, relative to `assets/` (an IsaacGym `load_asset`
convention). Every other importer resolves relative to the URDF's own directory, where
that path does not exist. Also fixed at the source.

**4. IsaacLab rejects default joint positions outside joint limits.** IsaacGym allowed it
silently. The thumb's `joint_12_0` has a strictly positive range `[0.263, 1.396]`, so the
implicit default of 0.0 raises at `sim.reset()`. M1's `ArticulationCfg.init_state` must
seed a valid pose — the canonical pose or a grasp-cache entry, not zeros. Worth checking
`allegro_hand_grasp.py`'s `canonical_pose` against the limits before relying on it.

**5. `SimulationContext` cannot be recreated within one app session.** Tearing one down
and building another hangs indefinitely at the second `reset()`. This constrains any
tooling that wants to inspect several assets: one sim per process. Relevant to M3, where
grasp generation may be tempted to loop over scales in a single run.

**6. Shutting an IsaacLab script down is its own problem.** Every entry point (M1's
`train.py`, M3's `gen_grasp.py`) needs the same three-part teardown, or it misbehaves in
ways that look like the script itself is broken:

- A live `SimulationContext` keeps callbacks and physics threads registered, and the
  process **prints all its output and then never exits**. Call `clear_all_callbacks()`
  and `clear_instance()` before shutting down.
- `simulation_app.close()` tears the process down itself and **always yields exit status
  0**, so a failed check or crashed training run reports success. Anything whose exit code
  is load-bearing should skip `close()` and `os._exit(code)` instead — safe for
  read-only tools, but a script that writes checkpoints must flush them first.
- Even after a clean close, non-daemon threads can keep the interpreter alive.

`scripts/verify_hand_asset.py` shows the working pattern.

### `README_ROS2_ISAACLAB.md` — MODIFY

Four fixes, all independently confirmed:

- `conda env create -f environment.yaml` → `environment_isaaclab.yaml`.
- `ROS_DISTRO=jazzy` → `humble`, and the `LD_LIBRARY_PATH` to the bridge's `humble/`
  directory. The container is `osrf/ros:humble-desktop`; the bridge ships both.
- Add `export ROS_DOMAIN_ID=42` and a matching `ROS_AUTOMATIC_DISCOVERY_RANGE` on the
  host. The container sets both; without them on the host the two never discover
  each other.
- Split the instructions by workload: training (host, no ROS), sim eval (host, no ROS),
  deploy (container).

Staying on Humble is deliberate: it is ROS 2, supported to 2027, and the ROS half is
already verified. Jazzy is a separate follow-up, not part of this migration.

---

## M1 — Core environment port

### `hora/tasks/base/vec_task.py` — **DELETE**

389 lines, ~90% replaced by `DirectRLEnv` + `DirectRLEnvCfg`. What must survive, and where
it goes:

| Current | Destination |
|---|---|
| `_allocate_buffers` (obs/rew/reset/progress/lag-history) | `_setup_scene` / `__init__` of the new env |
| `control_freq_inv` inner physics loop | `DirectRLEnvCfg.decimation = 6` |
| `update_low_level_control` hook | `DirectRLEnv._apply_action()` (called per physics step within decimation — an exact match for hora's 120 Hz PD inside a 20 Hz control step) |
| `clip_obs` / `clip_actions` | keep as explicit clamps |
| `_parse_sim_params` | `SimulationCfg` / `PhysxCfg` (see config table below) |
| `_set_viewer`, `render`, `create_sim`, `_create_ground_plane` | delete — handled by `AppLauncher` and `_setup_scene` |
| `spaces.Box` (the `gym` import) | delete — declare `observation_space`/`action_space` in the cfg |

`hora/tasks/base/__init__.py` — delete with it.

### `hora/tasks/allegro_hand_hora_cfg.py` — NEW

`@configclass` definitions: `AllegroHandHoraEnvCfg(DirectRLEnvCfg)` with nested
`SimulationCfg`, `InteractiveSceneCfg`, `ArticulationCfg` (hand), `RigidObjectCfg`
(object, with the multi-asset spawner from Decision 0), and the event terms for M2.

Config value mapping — the IsaacGym `sim:` block does **not** transfer key-for-key:

| `configs/task/*.yaml` | IsaacLab |
|---|---|
| `dt: 0.0083333` | `SimulationCfg.dt` (unchanged) |
| `controller.controlFrequencyInv: 6` | `DirectRLEnvCfg.decimation` |
| `substeps: 1` | no equivalent — drop |
| `up_axis: 'z'` | default |
| `use_gpu_pipeline` / `pipeline` | `SimulationCfg.device` |
| `solver_type` | `PhysxCfg.solver_type` |
| `num_position_iterations` / `num_velocity_iterations` | `ArticulationRootPropertiesCfg.solver_{position,velocity}_iteration_count` (per-actor, not global) |
| `contact_offset` / `rest_offset` | `CollisionPropertiesCfg` (per-collider) |
| `max_depenetration_velocity` | `RigidBodyPropertiesCfg` |
| `bounce_threshold_velocity` | `PhysxCfg` |
| `max_gpu_contact_pairs` | `PhysxCfg.gpu_max_rigid_contact_count` |
| `num_threads`, `num_subscenes`, `default_buffer_size_multiplier`, `contact_collection` | drop |
| `envSpacing: 0.25` | `InteractiveSceneCfg.env_spacing` |

Actuators: `ImplicitActuatorCfg` with `stiffness=0.0`, `damping=0.0`, effort limit 0.5,
because hora computes its own PD and clips to ±0.5. Do not let IsaacLab's actuator model
also apply a PD — that would double-control the joints.

**Initial hand pose.** The IsaacGym pose is built from a product of two axis-angle quats;
IsaacLab needs it as a literal **wxyz** quaternion. Working it through gives approximately
`(0.5, 0.5, -0.5, 0.5)` for `init_state.rot`, with `pos=(0, 0, 0.5)`. Confirm visually in
the viewer before trusting it — a wrong hand orientation will look plausible in logs and
silently ruin every grasp cache.

### `hora/tasks/allegro_hand_hora.py` — **REWRITE** (the bulk of the work)

663 lines. Becomes `AllegroHandHora(DirectRLEnv)`. Method-by-method:

| Current | New | Notes |
|---|---|---|
| `_create_envs` | `_setup_scene` | Drop aggregates entirely (no equivalent, no loss). Hand → `Articulation`, object → `RigidObject`. Per-env property randomization moves out to event terms (M2). |
| `_create_ground_plane` | `_setup_scene` | `sim_utils.GroundPlaneCfg` |
| `_create_object_asset` | cfg spawners | `AssetOptions` → `UsdFileCfg` + `RigidBodyPropertiesCfg` |
| `_refresh_gym` | **delete** | `articulation.data.*` / `rigid_object.data.*` are refreshed by the scene; no manual refresh calls |
| `pre_physics_step` | `_pre_physics_step` | logic unchanged (target integration, `object_{rot,pos}_prev` snapshot) |
| `update_low_level_control` | `_apply_action` | `set_dof_actuation_force_tensor` → `set_joint_effort_target`; `set_dof_position_target_tensor` → `set_joint_position_target` |
| `compute_observations` | `_get_observations` | returns the dict instead of filling `self.obs_dict` |
| `compute_reward` | `_get_rewards` | see quaternion warning |
| `check_termination` | `_get_dones` | must return `(terminated, truncated)` separately — currently conflated into one `reset_buf` |
| `reset_idx` | `_reset_idx` | `set_actor_root_state_tensor_indexed` → `write_root_state_to_sim(..., env_ids)`; `set_dof_state_tensor_indexed` → `write_joint_state_to_sim(..., env_ids)` |
| force perturbation in `pre_physics_step` | `_pre_physics_step` | `apply_rigid_body_force_tensors(ENV_SPACE)` → `RigidObject.set_external_force_and_torque()`. Forces are isotropic Gaussian, so the env→world frame change is immaterial. |
| debug viz (`gym.add_lines`) | `debug_draw` or drop | lowest priority |

**Three correctness traps, in priority order:**

1. **Quaternion convention.** IsaacGym is xyzw; IsaacLab/USD is **wxyz**. This hits
   `quat_to_axis_angle` (line ~650, hardcoded real-part-last), the rotation reward that
   depends on it, and the grasp-cache `.npy` layout `[16 joint pos, 3 xyz, 4 quat_xyzw]`.
   Replace the local helper with `isaaclab.utils.math.axis_angle_from_quat` and convert
   caches on load. A silent wrong-reward bug if missed — the policy will still train, just
   toward the wrong objective.
2. **World vs env frame.** `root_pos_w` includes the env origin. Subtract
   `scene.env_origins` before the `reset_z_threshold` check in `check_termination` and
   before writing `obj_position` into `priv_info`. Relative quantities
   (`object_pos - object_pos_prev`) are unaffected.
3. **`_get_dones` split.** IsaacGym's single `reset_buf` merges the height-threshold
   failure and the episode-length timeout. `extras['time_outs']` already distinguishes
   them for value bootstrapping, so map height → `terminated`, length → `truncated` and
   keep bootstrapping behaviour identical.

**Ordering assertions.** Add a hard assert at construction that
`articulation.find_joints(...)` returns `joint_0.0 … joint_15.0` in hora's order, and that
DOF limits match `allegro_dof_lower`/`upper` in `deploy.py`. Cheap, and it converts the
whole class of silent reordering bugs — the same class as commit `bacfc08` on the ROS
side — into a startup failure.

### `hora/tasks/__init__.py` — REWRITE

Drop `isaacgym_task_map`. Register the four task variants with `gymnasium.register`,
pointing at the env classes and their cfg entry points.

### `hora/utils/math_utils.py` — NEW

Small replacements for `isaacgym.torch_utils`. Most have IsaacLab equivalents in
`isaaclab.utils.math` (`quat_apply`, `quat_mul`, `quat_conjugate`, `quat_from_angle_axis`,
`sample_uniform`, `axis_angle_from_quat`). The rest are one-liners to write locally:
`to_torch`, `unscale`, `tensor_clamp`, `torch_rand_float`.

### `train.py` — MODIFY

`import isaacgym` → `AppLauncher`. The ordering constraint is strict: parse args, launch
the app, and *only then* import anything from `isaaclab` or `hora.tasks`. Getting this
wrong produces confusing extension-loading errors. Keep the hydra `@hydra.main` decorator
and all four `OmegaConf` resolvers — `isaaclab.utils.hydra.hydra_task_config` is the
sanctioned bridge between `@configclass` and the existing YAML tree, so all seven shell
scripts keep working with their `task.env.*=` overrides.

Replace `isaacgym_task_map[config.task_name](...)` with `gymnasium.make(...)`.

### `hora/algo/ppo/experience.py` — MODIFY (1 line)

Delete `import gym` on line 13. Confirmed dead — never referenced in the file.

### `hora/algo/ppo/ppo.py`, `hora/algo/padapt/padapt.py` — MODIFY (small)

Sim-agnostic; the env contract is preserved deliberately so these barely change.

- `from tensorboardX import SummaryWriter` → `torch.utils.tensorboard`.
- `torch.load(fn)` → pass `weights_only` explicitly (4 sites across the two files).
  Host torch is 2.7, container torch is 2.1.2; 2.6+ flipped the default.
- Verify `action_space.low/high` still expose numpy arrays through the gymnasium space
  that `DirectRLEnv` builds (`ppo.py:35-36` calls `.copy()` on them).

Unchanged, contract-wise: `observation_space`, `action_space`, `prop_hist_len`,
`reset()`/`step()` returning `(obs_dict{obs, priv_info, proprio_hist}, rew, done, extras)`
with `extras['time_outs']`.

### `hora/algo/models/*` — UNCHANGED

### `configs/` — MODIFY

- `configs/config.yaml`: drop `physics_engine`, `pipeline`, `num_threads`,
  `num_subscenes`, `graphics_device_id`; add IsaacLab's `device` / `--headless` wiring.
- `configs/task/*.yaml` (all four): rewrite the `sim:` block per the mapping table.
  The `env:` block largely survives.
- `configs/train/*.yaml`: `minibatch_size: 32768` with `horizon_length: 8` assumes
  `numEnvs=16384` (batch 131072 → 4 minibatches). At a 3060-realistic 2048–4096 envs the
  batch drops to 16k–32k and the minibatch must scale down with it, or PPO silently
  degenerates to one minibatch per epoch.

### `scripts/*.sh` — MODIFY

`pipeline=cpu|gpu` → IsaacLab device flags; `headless=True` → `--headless`. `gen_grasp.sh`
loses `pipeline=cpu` entirely (see M3).

### `requirements.txt` — MODIFY

Remove `gym`, `tensorboardx`. This file now describes the **container** (deploy) env only;
the host env is `environment_isaaclab.yaml`. Worth a comment saying so.

---

## M2 — Domain randomization and multi-scale

Per-env properties currently set inline in `_create_envs` move to `EventTermCfg`:

| Current | Event term |
|---|---|
| `set_actor_rigid_body_properties` (mass) | `randomize_rigid_body_mass` |
| `prop[0].com` (COM) | `randomize_rigid_body_com` |
| `set_actor_rigid_shape_properties` (friction, hand + object) | `randomize_rigid_body_material` |
| `set_actor_scale` | Decision 0 — construction-time, not an event term |
| PD gain randomization in `reset_idx` | keep in `_reset_idx`; it is plain tensor work |

Then flip `replicate_physics=False` and enable the multi-asset spawner.

**Grasp cache loader.** `_reset_idx` reads `cache/{name}_grasp_50k_s{scale}.npy`,
`[16 joint pos, 3 xyz, 4 quat_xyzw]`. Needs: quat → wxyz, joint order asserted against
`find_joints`, and object position offset by `scene.env_origins` before
`write_root_state_to_sim`. Do this in one loader function, not inline, so M3's regenerated
caches and the published ones go through the same path.

---

## M3 — Grasp generation

### `hora/tasks/allegro_hand_grasp.py` — **REWRITE**

Blocked on contact sensing, not a port. The current implementation calls
`gym.get_env_rigid_contacts()` and hard-asserts `device == 'cpu'`; there is no IsaacLab
equivalent.

Replace with a `ContactSensor` on the four fingertip bodies, using
`filter_prim_paths_expr` pointed at the object to get pairwise `force_matrix_w`. Then
`compute_reward`'s three conditions map over directly:

- cond1 (all fingertips within 0.1 m of object) — unchanged tensor math
- cond2 (≥2 fingers contacting) — `force_matrix_w` norm > threshold, summed over fingers.
  This replaces the rigid-body-index hashing (`obj_id * 10000 + link_id`), which was
  always a hack around the CPU contact API.
- cond3 (object above `reset_z_threshold`) — remember the env-origin subtraction

This is a net **upgrade**: contact sensing now runs on GPU, so grasp generation is no
longer pinned to the CPU pipeline. The `gen_grasp.sh` comments about CPU-only and
"no custom PD because bug in CPU mode" become obsolete — delete them.

### `hora/tasks/allegro_hand_grasp_cfg.py` — NEW

Cfg with the contact sensors added and randomization mostly disabled, mirroring
`configs/task/AllegroHandGrasp.yaml`.

### `gen_grasp.py` — MODIFY

Same `AppLauncher` treatment as `train.py`. Saves with the M2 loader's conventions
(wxyz quats) — and bump the filename or add a header field so old xyzw caches cannot be
loaded by mistake.

---

## M4 — ROS 2 sim-in-the-loop

Goal: `deploy.py` and the whole ros2_control stack run **unmodified** against Isaac Sim.
Nothing in the policy path should branch on sim vs hardware.

### `ros2_allegro/.../allegro_hand.ros2_control.xacro` — MODIFY

The launch files already advertise `ros2_control_hardware_type:=isaac`, but it is
vapourware — copy-paste from a MoveIt template with no matching `<xacro:if>` in the
hardware block. Add the branch.

Two implementations, in order of preference:

1. **`topic_based_ros2_control`** (PickNikRobotics) — a generic hardware interface that
   reads/writes `JointState` topics. Isaac Sim then needs only the stock
   `ROS2PublishJointState` / `ROS2SubscribeJointState` OmniGraph nodes, and **no custom
   C++ is written at all**. Confirm Humble availability before committing to this.
2. **Custom `isaac` hardware plugin** — more work, but `allegro_hand_hardwares/gazebo/plugin`
   is a working structural template already in the tree.

Do not bypass ros2_control by having Isaac Sim publish the controller topics directly:
`Allegro._activate_controller()` makes real `controller_manager` service calls, which
would have nothing to answer them.

### Sim-side bridge scene — NEW

An Isaac Sim scene/script for the single-env deployment case that publishes
`/joint_states` and subscribes to joint commands, with the joint-name mapping
`joint_N.0` ↔ `ah_joint{finger}{joint}`. Note this is a *third* joint ordering in the
system, alongside hora order and controller order — put the mapping in one place with the
same kind of assertion used in M1.

### Verification

Run `./scripts/deploy.sh` with `ros2_control_hardware_type:=isaac` and confirm it behaves
as it does against `mock_components`, then against hardware. Because the stack is
identical, a discrepancy localises to the sim, not the plumbing.

---

## Files that do not change

- `hora/algo/models/models.py`, `running_mean_std.py`
- `hora/algo/deploy/robots/allegro.py` — already clean rclpy + ros2_control
- `deploy.py` (root) — hydra entry point, sim-agnostic
- `hora/utils/reformat.py`
- `.devcontainer/*` — the container is the deploy environment and is already correct
- `hora/algo/deploy/deploy.py` — one-line `weights_only` fix aside
