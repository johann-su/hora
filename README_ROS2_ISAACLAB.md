# Hora converted to ros2 & isaaclab

## How to use

Isaaclab runs on the host (Isaac Sim from pip, IsaacLab from source, packaged as a conda env). Ros2 runs in a devcontainer.

There are three workloads, and it matters which is which:

| | Runs where | ROS 2? |
|---|---|---|
| **Training** (PPO stage 1, ProprioAdapt stage 2) | `env_isaaclab`, host | none, and none wanted — 16k envs through DDS is a non-starter |
| **Sim eval / visualization** | `env_isaaclab`, host | none by default; ROS-bridged variant is opt-in |
| **Hardware deployment** | devcontainer | yes |

ROS 2 is never on the training path. The bridge exists so the *real* deployment stack (controller_manager, position controller, `deploy.py`) can be driven against Isaac Sim instead of hardware, with nothing in the policy path branching on which it is talking to.

## Setup

### Ros

(inside dev container): setup ros workspace:
```bash
source /opt/ros/humble/setup.bash
source /home/lasr_ws/install/setup.bash   # only if that install/ dir exists

colcon build --symlink-install
```

(inside dev container - terminal 1): launch allegro hand controller
```bash
ros2 launch allegro_hand_bringup allegro_hand.launch.py
ros2 launch allegro_hand_bringup allegro_hand.launch.py ros2_control_hardware_type:=mock_components # run with no hardware connected
```

(inside dev container - terminal 2): run the hora policy
```bash
./scripts/deploy.sh hora_v0.0.2
```

### Isaaclab:

#### Isaacsim

https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-sim

Create conda env:
```bash
conda env create -f environment_isaaclab.yaml
conda activate env_isaaclab
```

launch isaacsim
```bash
isaacsim
```

To talk to the devcontainer over the ROS 2 bridge, the host must match the container's DDS settings or the two will never discover each other — the container sets `ROS_DOMAIN_ID=42` and `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`. The bridge ships both `humble/` and `jazzy/`; use the one matching the container.

```bash
export ROS_DISTRO=humble
export ROS_DOMAIN_ID=42
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib
``` 

#### Isaaclab

https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-lab

Clone Isaaclab:
```bash
cd ../
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.1   # pairs with Isaac Sim 5.1
```

Install
```bash
sudo apt install cmake build-essential # if not already installed
./isaaclab.sh -i none   # 'none' -> skip rl_games/rsl_rl/skrl/sb3; hora has its own PPO
```

#### Convert .urdf -> .usd

Isaaclab expects 3D models to be in .usd format. Previously, all assets where in .urdf format. A conversion script exists to make the transition easier.
```bash
./scripts/convert_assets.sh
```

This wraps IsaacLab's own `scripts/tools/convert_urdf.py`, which launches Isaac Sim once per asset — takes a while to complete. Output goes to `assets/usd/` (gitignored), and re-running skips what is already converted, so an interrupted run can just be restarted. Paths are derived from the script's own location, so it works from any working directory.
Set `ISAACLAB=` if the IsaacLab repo is not at `~/code/IsaacLab`.

## Training

### Grasp Position Cache

One scale per invocation -- a SimulationContext cannot be rebuilt in-process, so each scale needs its own process.

```bash
# e.g. ./scripts/gen_grasp.sh 0 0.8
./scripts/gen_grasp.sh ${GPU_ID} ${SCALE}
```

`task.env.randomization.randomizeScaleList` is the list that matters -- generate every entry, since `_load_grasp_cache` looks up one cache file per scale in `cfg.object_scales`:

```bash
for s in 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86; do ./scripts/gen_grasp.sh 0 $s; done
```

Output lands in `cache/${grasp_cache_name}_grasp_50k_s${scale_without_dot}.npy` (0.8 -> `s08`), 50000 rows of `[16 joint pos (hora order), 3 object xyz, 4 object quat (xyzw)]`.

### Stage 1

```bash
# e.g. ./scripts/train_s1.sh 0 0 debug
./scripts/train_s1.sh ${GPU_ID} ${SEED_ID} ${OUTPUT_NAME}
```

tensorboard
```bash
tensorboard --logdir=outputs/AllegroHandHora/debug
```

### Train Stage 2

```bash
# e.g. ./scripts/train_s2.sh 0 0 debug
./scripts/train_s2.sh ${GPU_ID} ${SEED_ID} ${OUTPUT_NAME}
```

### TUD HPC cluster (Capella)

Everything cluster specific is in `scripts/hpc/`.

- Cluster container docs: https://compendium.hpc.tu-dresden.de/software/containers/
- IsaacLab cluster docs: https://isaac-sim.github.io/IsaacLab/main/source/deployment/cluster.html

#### One-time setup

Build isaaclab container (from published isaaclab image, not build from source):
```bash
apptainer build isaac-lab-2.3.1.sif docker://nvcr.io/nvidia/isaac-lab:2.3.1

docker save my-isaac-lab:latest -o my-isaac-lab.tar
rsync -P my-isaac-lab.tar <login>:/data/horse/ws/$USER-hora/sif/
apptainer build my-isaac-lab.sif docker-archive://my-isaac-lab.tar
```

**SSH:** follow https://doc.zih.tu-dresden.de/access/ssh_login/.

**Workspace.**

```bash
# eg ws_allocate -F horse hora 90
ws_allocate -F <fs> hora <alloc-days>     # -> /data/horse/ws/<user>-hora
```

Change config in `scripts/hpc/cluster.env`

**Push the code, build the image, install the missing deps:**

```bash
./scripts/hpc/sync.sh                                    # local -> workspace

ssh login1.capella.hpc.tu-dresden.de
cd /data/horse/ws/$USER-hora/hora
./scripts/hpc/build_image.sh --submit                    # ~8 min batch job
./scripts/hpc/setup_deps.sh                              # after the build finishes
```

#### Running jobs

Run from the login node, in the synced copy:
```bash
cd /data/horse/ws/$USER-hora/hora

./scripts/hpc/submit.sh gen_grasp 0.8                    # one scale per job
./scripts/hpc/submit.sh train_s1 0 hora_v1 overwrite=True
./scripts/hpc/submit.sh train_s2 0 hora_v1 overwrite=True
./scripts/hpc/submit.sh raw scripts/eval_s1.sh 0 hora_v1 # anything else

# every scale the training config expects
for s in 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86; do
    ./scripts/hpc/submit.sh gen_grasp $s
done
```

Resources default to one GPU's share and 8 h; override per submission:
```bash
HORA_TIME=23:00:00 HORA_MEM=240G ./scripts/hpc/submit.sh train_s1 0 hora_v1
```

Logs land in `/data/horse/ws/$USER-hora/slurm-logs/<job-name>-<jobid>.out`.

#### Getting results back

```bash
./scripts/hpc/fetch.sh              # all of outputs/
./scripts/hpc/fetch.sh hora_v1      # one run
./scripts/hpc/fetch.sh --cache      # grasp caches generated on the cluster
./scripts/hpc/fetch.sh --logs       # SLURM logs
```

`sync.sh` excludes `outputs/` by default so a push cannot clobber a running job's
checkpoints; use `--with-outputs` when stage 2 needs a stage-1 checkpoint that
only exists locally.

#### Interactive debugging

```bash
srun -A p_lasr_students -p capella --gres=gpu:1 -c 14 --mem=180G -t 2:00:00 --pty bash
cd /data/horse/ws/$USER-hora/hora
./scripts/hpc/run_in_container.sh bash                   # shell inside the image
./scripts/hpc/run_in_container.sh python train.py ...    # -> /isaac-sim/python.sh
```
