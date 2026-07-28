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

```bash
# e.g. ./scripts/gen_grasp.sh 0 2
./scripts/gen_grasp.sh ${GPU_ID} ${SCALE}
```

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

### TUD HPC cluster

If training on the hpc cluster:

configure ssh according to: https://doc.zih.tu-dresden.de/access/ssh_login/#connecting-via-visual-studio-code

allocate and connect to interactive compute node:
```bash
salloc --gres=gpu:1 --nodes 1 # optional -A p_lasr_students
# connect to node: ssh c123 (capella compute node 123)
```

```bash
ws_allocate conda_virtual_environment 1 # https://doc.zih.tu-dresden.de/software/python_virtual_environments/#conda-virtual-environment
```

```bash
mkdir /data/horse/ws/<username>-conda_virtual_environment/conda-env    #Create directory for environment
module load Anaconda3    #Load Anaconda
conda config --set channel_priority strict
conda env create --prefix /data/horse/ws/<username>-conda_virtual_environment/conda-env --file environment_isaaclab.yaml    #Create conda env in directory with packages from environment.yml file
```

TODO: HPC cluster cannot install isaaclab via pip (incompatible) - migrate to docker based deployment via singularity (https://compendium.hpc.tu-dresden.de/software/containers/)

## Deployment

```bash
scripts/deploy.sh ${OUTPUT_NAME}
```