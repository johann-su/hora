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
| M1 | Core env ports, single scale, no ROS ✅ **done** | `train.py` runs stage 1 PPO; reward curve rises |
| M2 | Domain randomization + multi-scale ✅ **done** | `train_s1.sh` runs unmodified; per-env scale/mass/CoM/friction verified |
| M3 | Grasp generation ✅ **done** | `gen_grasp.py` regenerates a cache validated equivalent to the published one |
| M4 | ROS 2 sim-in-the-loop ⚠️ **partial** | `deploy.py` drives sim through ros2_control, unmodified |

M3 depends on M0 only, so it can run in parallel with M1/M2 if the caches must be
regenerated early. The published caches are downloaded (26 files, every scale in
`randomizeScaleList`), so M1/M2 train without needing M3 finished.

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

### M1 findings

**1. `DirectRLEnv` owns `self.obs_buf`.** `step()` assigns the dict returned by
`_get_observations()` to it, so hora's flat observation tensor of the same name was
silently replaced with a dict on the first step. Renamed to `self._obs`. Worth checking
any other buffer name the port carries over against the base class.

**2. A raw dict cannot be a `@configclass` field.** Storing hora's hydra config dict on
the env cfg sends `configclass`'s recursive validator into a ~1000-frame `RecursionError`.
The dict is passed to the env constructor as a separate argument instead.

**3. Overlapping joint-name patterns are rejected.** `{'joint_12_0': 0.263, '.*': 0.0}`
fails with "Multiple matches". Initial joint positions are now derived by clamping 0.0
into each joint's URDF limits (`_default_joint_pos`), which is also the honest fix for
finding M0-4 rather than hardcoding the thumb exception.

**4. Shutdown bites here too, and worse.** `simulation_app.close()` hung after training
completed — three 10-minute test runs that were really 13 seconds of work plus a hung
shutdown. Because `close()` blocks, any exit-code handling placed *after* it never runs,
so the earlier "flush then `os._exit`" guard was dead code. `train.py` now skips `close()`
entirely; `PPO.train()` flushes the tensorboard writer so nothing is lost.

**5. Trainers moved to the gymnasium API.** `reset()` returns `(obs, info)` and `step()`
returns the 5-tuple with `terminated`/`truncated` separate. Eight call sites across
`ppo.py` and `padapt.py`. `_get_observations` returns `'policy'` (the IsaacLab contract)
and `'obs'` (what hora's trainers read) pointing at the same tensor.

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

### M2 findings

**Performance cost is much smaller than feared.** At 1024 envs, 81 cuboids × 9 scales
with full randomization runs at **15.7k FPS** against M1's single-scale, replicated
**19.4k FPS** — roughly 19%, not the order of magnitude "Decision 0" braced for. The env
ceiling on a 12 GB card is still the binding constraint, not `replicate_physics`.

**`clone_in_fabric` must be off for heterogeneous scenes, and fails silently if not.**
Fabric clones are not reachable through USD APIs, and the multi-asset spawner locates its
targets by matching USD prim paths — so with fabric on it found only `env_0` and spawned
one object for the entire scene. The symptom was `num_instances == 1` for 64 envs, i.e.
every property write landing on env 0 while training carried on looking healthy. This is
the most dangerous thing found in M2: nothing errors.

**`clone_environments()` must be guarded on `replicate_physics`.** When it is False,
`InteractiveScene.__init__` has *already* deep-cloned the env xforms — it must, so the
`env_.*` paths exist before the spawner's regex runs. Calling it again from `_setup_scene`
(as every replicated DirectRLEnv example does) undoes that.

**A single-body `RigidObject` reports CoM as `(num_envs, 7)`**, not the
`(num_envs, num_bodies, 7)` that IsaacLab's own `randomize_rigid_body_com` indexes —
that term is written for articulations.

**Randomization is applied by hand, not through `EventManager`.** hora's privileged
observation must contain the *actual* sampled mass/CoM/friction, and event terms sample
internally without reporting values back. `_randomize_object_properties` samples once at
construction (matching IsaacGym, which randomized at env creation rather than per reset)
and writes straight into `priv_info_buf`.

**Object choice is round-robin, not weighted sampling.** IsaacGym drew each env's object
with `np.random.choice(..., p=sampleProb)`; `MultiAssetSpawnerCfg(random_choice=False)`
assigns cyclically. For the uniform weights hora uses these agree in distribution, and
round-robin also guarantees exact balance. A non-uniform `sampleProb` would need the
variant list repeated in proportion.

**The ±0.025 scale jitter is not reproduced.** Envs get exactly the nine nominal scales.
Restoring it means emitting more variants (9 × k, keeping the count a multiple of 9 so
`i % 9` still selects the right grasp cache) — cheap, since variants are xform scales over
one USD, but not currently done.

**`train.py`'s overwrite guard blocks unattended runs.** Re-using an `output_name` that
already holds `best.pth` prompts on stdin and waits forever. Worth knowing before queuing
a long sweep.

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

Same `AppLauncher` treatment as `train.py`. `--report-contact-forces` samples the
fingertip/object force distribution instead of collecting, which is how the contact
threshold gets calibrated rather than guessed.

**On-disk format deliberately unchanged.** An earlier draft of this plan proposed writing
wxyz quaternions and renaming the files. That was the wrong call: keeping the published
**xyzw** layout means generated and downloaded caches stay interchangeable, and there is
one conversion point (`_load_grasp_cache`) instead of two formats in circulation.

### `scripts/validate_grasp_cache.py` — NEW

The exit criterion for M3. Statistics alone cannot settle whether a generated cache is
usable — two caches can have similar joint histograms and still differ in whether the
poses actually hold the object. So the load-bearing check is functional: reset the
rotation env from the cache, step with zero actions, and measure how many envs still hold
the object. Zero actions means anything that survives is the grasp itself, not a policy.

One cache per invocation (`SimulationContext` cannot be rebuilt in-process, M0 finding 5);
run it twice and compare.

**Published-cache baseline** (scale 0.8, 512 envs, 100 steps): fingertip-object distance
0.0845 m at reset, object z 0.6519, and **held = 1.000 at every step**. A generated cache
should reproduce that.

---

### M3 findings

**Validated against the published cache.** A full 50,000-pose cache was generated for
scale 0.8 (2048 envs, ~28 min) and compared with the published one both statistically and
functionally:

| | published | generated |
|---|---|---|
| rows | 50,000 | 50,000 |
| joint mean abs difference | — | max 0.026 rad (1.5°), avg 0.006 rad |
| object pos mean (x, y, z) | -0.0093, -0.0055, 0.6518 | -0.0088, -0.0042, 0.6525 |
| quaternion norm | 1.0000 | 1.0000 |
| fingertip-object distance at reset | 0.0845 m | 0.0850 m |
| **held, zero actions, 100 steps** | **1.000** | **1.000** |

Joint ranges stay inside the published envelope; object-position spread is slightly wider
(std 0.0107 vs 0.0072 in x), i.e. marginally more diverse rather than degenerate.

Only scale 0.8 was regenerated and validated — the remaining eight scales are the same
command with a different `baseObjScale` (see the loop in `scripts/gen_grasp.sh`), roughly
30-60 min each. The published caches are still in `cache/` and untouched.

**The contact filter must name the rigid body, not the spawn path.** The URDF importer
wraps each link in an Xform, so the USD default prim is `/ball` and the body is at
`/ball/ball`; spawning at `.../object` puts the body at `.../object/<link_name>`.
Filtering on `.../object` yields a valid-but-empty pair set, so `force_matrix_w` is
**all zeros forever rather than None** — indistinguishable from "nothing is touching".
PhysX does warn (`GPU contact filter for collider ... is not supported`), but it is one
line in thousands of startup messages. Grep for it when filtered contacts read as zero.

**Contact needs a settling period that the original did not.** IsaacGym reset an env the
instant its three conditions stopped holding, which only works if the object is already
touching the fingers at spawn. The converted colliders differ enough that the object
spawns ~1 cm clear, so it failed on step 1, was teleported back, and *never moved* —
object z was identical to the spawn height on every step of a 200-step run.
`graspSettleSteps` (default 10) suppresses failure while the object drops in.

**The force threshold is calibrated, not guessed.** Measured fingertip/object contact
forces run p05 0.068 N, p50 0.40 N, p99 2.44 N. The default is 0.1 N — above the noise
floor, keeping ~93% of real contacts. An initial guess of 0.5 N sat above the *median*
and discarded most genuine contacts. Re-measure with
`gen_grasp.py --report-contact-forces` if the hand, objects or gains change.

**Filtered contacts need `replicate_physics=False`**, so grasp generation does not get
M2's replicated fast path.

**Hand orientation is confirmed correct.** `rot=(0.5, 0.5, -0.5, 0.5)` was carried
through M1/M2 unverified. Two independent checks now agree: a Kabsch solve recovering the
rotation from the grasp cache returns the same quaternion (residual 0.0659 vs 0.0672 m,
i.e. no improvement available), and the GUI shows the hand palm-up in a natural cupped
pose. The ~8 cm fingertip-to-object distance seen when replaying cached grasps is
body-origin to object-centre, not a surface gap — the merged `_tip` links put each
fingertip body's origin well behind its contact surface, which is why IsaacGym's own
condition used a generous 0.1 m.

## M4 — ROS 2 sim-in-the-loop

Goal: `deploy.py` and the whole ros2_control stack run **unmodified** against Isaac Sim.
Nothing in the policy path should branch on sim vs hardware.

**No physical hand is needed for any of this.** Isaac Sim *replaces* the hardware, so M4
is implementable and largely verifiable without an Allegro. What the hand is needed for is
the separate question of sim-to-real parity — see "Needs hardware" below.

### Verified without hardware ✅

**The ROS 2 deployment path works end to end against `mock_components`.** With the
devcontainer running:

```bash
ros2 launch allegro_hand_bringup allegro_hand.launch.py \
     ros2_control_hardware_type:=mock_components
./scripts/deploy.sh hora_v0.0.2
```

Confirmed live: `allegro_hand_position_controller` **active** with all 16 joints claimed;
`/allegro_hand_position_controller/commands` showing publisher 1 / subscriber 1; and
`/joint_states` reporting varied non-zero positions tracking the policy's commands. So the
policy → controller_manager → hardware-interface path is sound, and the only thing M4 adds
is swapping the hardware interface underneath it.

Note `deploy.py` prints progress with `tprint` (carriage return, no newline), so its stdout
looks empty while it is in fact running. Check `ros2 topic info` rather than the console.

**No custom C++ is required.** `ros-humble-topic-based-ros2-control` is packaged for
Humble (0.2.0) and is now installed in the container. Its `TopicBasedSystem` exchanges
`sensor_msgs/JointState` with any simulator, so the Isaac-specific hardware plugin the
plan originally contemplated is unnecessary.

**The `isaac` hardware type is now real.** It was advertised in every launch file's help
string but had no matching branch in the xacro. Added, and verified to expand:

```bash
xacro .../allegro_hand.urdf.xacro ros2_control_hardware_type:=isaac
# -> <plugin>topic_based_ros2_control/TopicBasedSystem</plugin>
#    joint_states_topic=/isaac_joint_states  joint_commands_topic=/isaac_joint_commands
```

Deliberately **not** `/joint_states`: `joint_state_broadcaster` publishes there, and
reusing it would feed the stack its own output.

**The Isaac ROS 2 bridge extension loads from a standalone IsaacLab script**, exposing the
nodes the design needs (`ROS2Context`, `ROS2PublishJointState`, `ROS2SubscribeJointState`)
— given the `ROS_DISTRO` / `RMW_IMPLEMENTATION` / `LD_LIBRARY_PATH` exports from the
README.

### Blocked ⚠️

`scripts/sim_ros2_bridge.py` is written but **does not run**. Building the OmniGraph fails:

```
[omni.graph.core.plugin] Unable to create prim for graph at /ActionGraph
OmniGraphError: Failed to wrap graph in node given {'graph_path': ..., 'evaluator_name': 'execution'}
```

Moving the graph under `/World` and constructing it after `sim.reset()` did not help. Two
things are already established and worth keeping: `omni.graph` is not importable until
`omni.graph.action` / `isaacsim.core.nodes` / `isaacsim.ros2.bridge` are enabled, so the
enables must precede the import; and the bridge extension itself loads fine.

Two candidate ways forward, neither yet attempted:

1. **Fix the OmniGraph construction.** Most likely the graph needs creating against an
   explicitly-set stage, or via `omni.graph.core.Controller` with the ROS 2 bridge's own
   sample as the template rather than a hand-written node list. Closest reference is
   Isaac Sim's own ROS 2 joint-control sample.
2. **Bypass OmniGraph entirely.** Have the Isaac script speak a plain socket, and run a
   small `rclpy` relay inside the devcontainer that bridges socket ↔ ROS topics. This
   sidesteps OmniGraph *and* the rclpy-version problem (the host conda env is Python 3.11
   with no rclpy; Humble's rclpy is 3.10), and it is the natural home for the joint-name
   remap below. Less "native", equally valid for the purpose, and much easier to debug.

### Joint-name remapping — required either way

This is a third naming boundary in the system and must live in exactly one place:

| | names | order |
|---|---|---|
| hora policy / cache / `deploy.py` | — | index, thumb, middle, ring |
| hora sim asset (converted USD) | `joint_0_0 … joint_15_0` | PhysX breadth-first |
| ros2_allegro stack | `ah_joint00 … ah_joint33` | thumb-first (`joint00..03` = thumb) |

The hora asset and the ros2_allegro description are **different URDFs of the same hand**,
so the sim cannot simply adopt the ROS names. `deploy.py` already handles hora ↔ SDK order
(`_obs_allegro2hora` / `_action_hora2allegro`) and `algo/deploy/robots/allegro.py` already
handles SDK ↔ controller order; M4 adds sim-asset ↔ `ah_*`, which belongs in the relay or
the bridge, asserted at startup the way `scripts/verify_hand_asset.py` asserts the others.

### Needs hardware ❌

These cannot be closed without a physical Allegro, and none of them block the M4
implementation:

- **Sim-to-real parity.** Whether a policy that works against the Isaac surrogate behaves
  the same on the real hand — the actual reason M4 exists.
- **The `physical_device` path after the xacro change.** The `isaac` branch is additive
  and `mock_components` still expands correctly, but only real hardware proves the
  physical branch is unaffected.
- **Timing and latency.** The real hand runs a CAN bus with its own rate and jitter;
  `topic_based_ros2_control` over DDS has a different profile. Only relevant once a policy
  is being transferred.

### Verification recipe, once the bridge runs

```bash
# host                                    # container
export ROS_DOMAIN_ID=42 ...               ros2 launch allegro_hand_bringup \
python scripts/sim_ros2_bridge.py              allegro_hand.launch.py \
                                               ros2_control_hardware_type:=isaac
                                          ./scripts/deploy.sh hora_v0.0.2
```

Expected: `/isaac_joint_states` visible from inside the container (this also proves
host↔container DDS, which `--net=host` plus matching `ROS_DOMAIN_ID=42` and
`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` should give); the position controller active;
and the simulated hand tracking the policy exactly as the mock hand does above. Because
the stack is identical to the `mock_components` run, any discrepancy localises to the
simulator rather than the plumbing.

## Files that do not change

- `hora/algo/models/models.py`, `running_mean_std.py`
- `hora/algo/deploy/robots/allegro.py` — already clean rclpy + ros2_control
- `deploy.py` (root) — hydra entry point, sim-agnostic
- `hora/utils/reformat.py`
- `.devcontainer/*` — the container is the deploy environment and is already correct
- `hora/algo/deploy/deploy.py` — one-line `weights_only` fix aside
