# Docker Guide

This repository can be built and run inside the CUDA-enabled ROS 2 Humble image
defined at `docker/Dockerfile.humble-cuda`.

The current Dockerfile has been verified to build successfully. It uses:

- `nvidia/cuda:12.1.1-devel-ubuntu22.04`
- ROS 2 Humble
- Gazebo, RViz, `xacro`, `robot_state_publisher`, and `joint_state_publisher`
- a Python virtual environment at `/opt/exact_mppi/venv`
- CUDA 12 JAX via `jax[cuda12]`
- `python3-tk` for `gctl`
- `lxml` for Gazebo `spawn_entity.py` when the venv is active
- editable installs of `ir-sim_mppi` and `EXACT_MPPI_core`

Ubuntu and ROS 2 apt sources are configured to use the Tsinghua mirrors in the
image, which is useful for builds from China.

## Prerequisites

Install these on the host before building or running the image:

- Docker
- NVIDIA driver
- NVIDIA Container Toolkit
- X11 support if you want to open Gazebo or RViz windows

Quick host checks:

```bash
docker --version
nvidia-smi
echo "$DISPLAY"
ls /tmp/.X11-unix
```

If `nvidia-smi` fails on the host, fix the host NVIDIA driver/toolkit setup
before debugging the container.

## Build the Image

Run this from the repository root:

```bash
docker build -f docker/Dockerfile.humble-cuda -t exact-mppi:humble-cuda .
```

If you need a fully fresh rebuild:

```bash
docker build --no-cache -f docker/Dockerfile.humble-cuda -t exact-mppi:humble-cuda .
```

## Run the Container

For command-line development:

```bash
docker run -it --rm \
  --gpus all \
  --network host \
  --ipc host \
  -v "$(pwd):/workspace/EXACT-mppi" \
  -w /workspace/EXACT-mppi \
  --name exact-mppi-dev \
  exact-mppi:humble-cuda
```

For Gazebo or RViz, allow local X11 access on the host first. Run these on the
host, not inside Docker:

```bash
xhost +SI:localuser:root
xhost +local:root
```

Then start the GUI-enabled container:

```bash
docker run -it --rm \
  --gpus all \
  --network host \
  --ipc host \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd):/workspace/EXACT-mppi" \
  -w /workspace/EXACT-mppi \
  --name exact-mppi-dev \
  exact-mppi:humble-cuda
```

After GUI work, revoke the temporary X11 permission on the host:

```bash
xhost -SI:localuser:root
xhost -local:root
```

## Python and ROS Environment

Interactive shells source ROS 2 Humble and activate the Python virtual
environment through `/root/.bashrc`.

Inside the container, verify the active environment:

```bash
which python
python -m pip --version
echo "$ROS_DISTRO"
```

For non-interactive Docker commands, source the environment explicitly:

```bash
source /opt/ros/humble/setup.bash
source /opt/exact_mppi/venv/bin/activate
```

Because the repository is mounted at `/workspace/EXACT-mppi`, local source edits
are visible inside the container immediately. Rebuild the image when package
metadata, dependencies, or Dockerfile layers change.

## Verify the Image

Check that JAX can see the GPU:

```bash
python -c "import jax; print(jax.devices())"
```

Check that the local Python packages import:

```bash
python -c "import irsim, lxml; from exact_mppi.mppi_jax.controller import MPPIController; print('imports ok')"
python -c "import tkinter, gctl; print('gctl ok')"
```

Check that ROS 2 and RViz are available:

```bash
ros2 pkg list | head
which rviz2
```

## Build the ROS 2 Workspace

The ROS 2 bridge workspace is in `mosaic_mppi_ros2`.

Inside Docker, you run as `root`, and the current image does not install
`sudo`. Use `apt-get` directly for any remaining ROS workspace dependencies:

```bash
apt-get update && apt-get install -y \
  ros-humble-tf-transformations \
  ros-humble-tf2-tools \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs \
  ros-humble-nav-msgs \
  ros-humble-sensor-msgs \
  ros-humble-geometry-msgs \
  ros-humble-visualization-msgs \
  ros-humble-joint-state-publisher-gui \
  libeigen3-dev \
  libyaml-cpp-dev \
  cmake
```

Then build:

```bash
cd /workspace/EXACT-mppi/mosaic_mppi_ros2
./build.sh
source install/setup.bash
```

If the scripts are not executable:

```bash
chmod +x setup.sh build.sh
```

`./setup.sh` is still useful for native host setup, but inside this Docker image
the direct `apt-get` command above is the safer path because the container runs
as `root`.

## Run Examples

Core Python example:

```bash
cd /workspace/EXACT-mppi
python EXACT_MPPI_core/example/corridor_dynamic_random_jax/mppi_jax_anyshape.py --robot-shape f
```

ROS 2 corridor simulator:

```bash
cd /workspace/EXACT-mppi/mosaic_mppi_ros2
source install/setup.bash
ros2 launch exact_mppi_jax sim_corridor_external_ref_launch.py
```

ROS 2 T-shape simulator:

```bash
cd /workspace/EXACT-mppi/mosaic_mppi_ros2
source install/setup.bash
ros2 launch exact_mppi_jax sim_tshape_external_ref_launch.py
```

Gazebo LIMO corridor simulation:

```bash
cd /workspace/EXACT-mppi/mosaic_mppi_ros2
source install/setup.bash
ros2 launch exact_mppi_jax sim_limo_corridor_launch.py auto_goal:=true
```

## Common Issues

`rviz2: command not found`

```bash
apt-get update && apt-get install -y ros-humble-rviz2
```

`package 'gazebo_ros' not found`

```bash
apt-get update && apt-get install -y ros-humble-gazebo-ros-pkgs
```

`file not found: xacro`

```bash
apt-get update && apt-get install -y ros-humble-xacro
```

`sudo: command not found`

The Docker image runs as `root`, so use `apt-get` directly instead of `sudo`.

Gazebo or RViz cannot open a window:

- make sure `echo "$DISPLAY"` is not empty on the host
- make sure `/tmp/.X11-unix` is mounted into the container
- run `xhost +SI:localuser:root` and `xhost +local:root` on the host before
  starting the GUI container

`Authorization required, but no authorization protocol specified`

This is an X11 authorization failure. In a separate host terminal, run:

```bash
xhost +SI:localuser:root
xhost +local:root
```

Then retry `rviz2` in the already running container. If you restart the
container later, keep the same Docker run command.

JAX does not use the GPU:

- make sure `nvidia-smi` works on the host
- start the container with `--gpus all`
- keep the CUDA-enabled JAX install from the Dockerfile: `jax[cuda12]`
