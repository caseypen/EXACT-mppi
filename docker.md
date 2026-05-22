# Docker Guide

This repository can be developed and tested inside Docker. The recommended setup is:

- Ubuntu 22.04
- ROS 2 Humble
- NVIDIA CUDA 12
- Python virtual environment included in the image

The repository already includes a base image definition at `docker/Dockerfile.humble-cuda`.

## What This Container Is For

Use the Docker environment when you want:

- a reproducible Ubuntu 22.04 + ROS 2 Humble workspace
- GPU support for JAX
- Gazebo and RViz-based ROS 2 demos without polluting the host system

The provided Dockerfile includes Gazebo, `gazebo_ros`, `xacro`, and `rviz2` for ROS 2 simulation workflows in `mosaic_mppi_ros2`.

## Prerequisites

Before building the container, make sure the host machine has:

- Docker
- NVIDIA Container Toolkit
- a working NVIDIA driver
- X11 available if you want to run Gazebo or RViz with GUI

Quick checks on the host:

```bash
docker --version
nvidia-smi
echo "$DISPLAY"
ls /tmp/.X11-unix
```

If `nvidia-smi` fails on the host, fix the GPU driver stack before debugging the container.

## Build the Image

From the repository root:

```bash
cd EXACT-mppi
docker build -f docker/Dockerfile.humble-cuda -t exact-mppi:humble-cuda .
```

## Run the Container

From the repository root, start a headless container for building and command-line work:

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

If you want to open Gazebo or RViz windows from inside the container, allow X11 access first:

```bash
xhost +local:root
```

Then start the GUI-enabled container:

```bash
docker run -it --rm \
  --gpus all \
  --network host \
  --ipc host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd):/workspace/EXACT-mppi" \
  -w /workspace/EXACT-mppi \
  --name exact-mppi-dev \
  exact-mppi:humble-cuda
```

This command mounts the repository into the container and drops you into the project workspace.

After GUI work, you can revoke the X11 permission:

```bash
xhost -local:root
```

## Use the Python Environment

The Docker image creates a Python virtual environment at `/opt/exact_mppi/venv`, puts it on `PATH`, and installs:

- `jax[cuda12]`
- `ir-sim_mppi` in editable mode
- `EXACT_MPPI_core` in editable mode

Inside the container, confirm that the venv Python and local packages are active:

```bash
which python
python -m pip --version
python -c "import irsim; from exact_mppi.mppi_jax.controller import MPPIController; import jax; print(jax.devices())"
```

Because these are editable installs and the repository is mounted at `/workspace/EXACT-mppi`, normal Python source edits are picked up without reinstalling. If package metadata or dependencies change, rebuild the Docker image.

## Build the ROS 2 Workspace

The ROS 2 bridge workspace is located in `mosaic_mppi_ros2`.

Inside the container:

```bash
cd mosaic_mppi_ros2
./setup.sh
./build.sh
source install/setup.bash
```

If `setup.sh` is not executable:

```bash
chmod +x setup.sh build.sh
```

## Verification

Check that ROS 2 is available:

```bash
source ~/.bashrc
ros2 --version
```

Check that JAX can see the GPU:

```bash
python -c "import jax; print(jax.devices())"
```

Check that RViz is installed:

```bash
source ~/.bashrc
rviz2
```

## Run a First Example

Core Python example:

```bash
source .exact_mppi/bin/activate
python EXACT_MPPI_core/example/corridor_dynamic_random_jax/mppi_jax_anyshape.py --robot-shape f
```

ROS 2 simulation example:

```bash
cd mosaic_mppi_ros2
source install/setup.bash
ros2 launch exact_mppi_jax sim_limo_corridor_launch.py auto_goal:=true
```

## Common Issues

`rviz2: command not found`

Install:

```bash
apt install -y ros-humble-rviz2
```

`package 'gazebo_ros' not found`

Install:

```bash
apt install -y ros-humble-gazebo-ros-pkgs
```

`file not found: xacro`

Install:

```bash
apt install -y ros-humble-xacro
```

Gazebo or RViz cannot open a window

Check:

- `echo $DISPLAY` is not empty
- the X11 socket directory is mounted into the container
- `xhost +local:root` was run on the host

JAX does not use GPU

Check:

- `nvidia-smi` works on the host
- the container was started with `--gpus all`
- the installed JAX package matches the CUDA major version
