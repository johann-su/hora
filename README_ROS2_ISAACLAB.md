# Hora converted to ros2 & isaaclab

## How to use

Isaaclab runs on the host, installed via pip (packaged as a conda env). Ros2 runs in a devcontainer.

## Setup

### Isaaclab:

Create conda env:
```bash
conda env create -f environment.yaml # (based on https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
conda activate env_isaaclab
```

launch isaacsim
```bash
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib

isaacsim
```

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
```

(inside dev container - terminal 2): launch hora
```bash
python -m hora.algo.deploy.deploy ...
```
