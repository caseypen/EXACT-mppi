# Docker Guide

This repository contains two local Python packages:

- `ir-sim_mppi`: simulator package used by the examples
- `EXACT_MPPI_core`: MPPI controller package and example scripts

The recommended way to build and run everything is the provided Docker image, which ships a fully configured CUDA + ROS 2 Humble environment. With it you do **not** need to install ROS, JAX, or any Python dependency on the host — build the image, start the container, and run an example.

## What the Docker image provides

The image is defined in `docker/Dockerfile.humble-cuda` and has been verified to build successfully. It includes:

- `nvidia/cuda:12.1.1-devel-ubuntu22.04`
- ROS 2 Humble (`ros-base`) with all workspace dependencies pre-installed
- Gazebo, RViz, `xacro`, `robot_state_publisher`, and `joint_state_publisher`
- a Python virtual environment at `/opt/exact_mppi/venv`
- CUDA 12 JAX via `jax[cuda12]`
- `python3-tk` and `python3-pil.imagetk` for `gctl` GUI components
- `lxml` for Gazebo `spawn_entity.py` when the venv is active
- editable installs of `ir-sim_mppi` and `EXACT_MPPI_core`

ROS 2 and the virtual environment are sourced automatically in every shell (via `~/.bashrc`), so the environment is ready to use the moment the container starts.

> **Apt mirrors:** Ubuntu and ROS 2 apt sources are configured to use the Tsinghua mirrors, which accelerates builds inside China. If you are outside China, comment out the mirror configuration lines in the Dockerfile.

## Prerequisites

Install these on the host before building or running the image:

- Docker
- NVIDIA driver
- NVIDIA Container Toolkit (required for `--gpus all`)
- X11 support — only needed if you want to open Gazebo or RViz windows

Quick host checks:

```bash
docker --version
nvidia-smi
echo "$DISPLAY"
ls /tmp/.X11-unix
```

## Build the image

The helper scripts live in the `docker/` directory. Make them executable once:

```bash
chmod +x docker/build_docker.sh docker/run_docker.sh
```

Then build:

```bash
./docker/build_docker.sh
```

The script uses the repository root as the build context and tags the result `exact-mppi:humble-cuda`. The first build downloads ROS 2, CUDA JAX, and other large packages and can take a while; subsequent builds reuse the cache.

## Run the container

```bash
./docker/run_docker.sh
```

This starts an interactive container with:

- `--gpus all` and full NVIDIA capabilities
- host networking (`--net=host`)
- X11 forwarding so Gazebo / RViz can open windows on your desktop
- the repository mounted read-write at `/workspace/EXACT-mppi`

Because the repository is mounted live and the two packages are installed in editable mode, **Python source edits you make on the host take effect immediately inside the container** — no rebuild required.

## Quick check

Inside the container:

```bash
python -c "import irsim; from exact_mppi.mppi_jax.controller import MPPIController; print('setup ok')"
```

## Run examples

All commands below are run inside the container.

### Core Python example (no ROS required)

```bash
cd /workspace/EXACT-mppi
python EXACT_MPPI_core/example/corridor_dynamic_random_jax/mppi_jax_anyshape.py --robot-shape f
```

Each core example prints the detected JAX backend and device list before the simulation starts, so you can confirm whether it is running on CPU or GPU. For other scenarios, run a different script under `EXACT_MPPI_core/example/`.

### ROS 2 corridor simulator

```bash
cd /workspace/EXACT-mppi/mosaic_mppi_ros2
source install/setup.bash
ros2 launch exact_mppi_jax sim_corridor_external_ref_launch.py
```

### ROS 2 T-shape simulator

```bash
cd /workspace/EXACT-mppi/mosaic_mppi_ros2
source install/setup.bash
ros2 launch exact_mppi_jax sim_tshape_external_ref_launch.py
```

### Gazebo LIMO corridor simulation

```bash
cd /workspace/EXACT-mppi/mosaic_mppi_ros2
source install/setup.bash
ros2 launch exact_mppi_jax sim_limo_corridor_launch.py auto_goal:=true
```

The ROS 2 and Gazebo examples open RViz / Gazebo windows, so make sure X11 forwarding is working (see the run-container note above).

## Local install without Docker (optional)

If you prefer to set things up directly on the host instead of using the image, use Python 3.10+ (3.10 recommended) and a virtual environment:

```bash
python3 -m venv .exact_mppi
source .exact_mppi/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ./ir-sim_mppi
python -m pip install -U "jax[cuda12]"   # or "jax[cpu]" for CPU-only
python -m pip install -e ./EXACT_MPPI_core
```

Match the JAX wheel to your CUDA **major** version only (e.g. CUDA 12.x → `jax[cuda12]`). For the exact Python/CUDA combination on your machine, follow the official guide: https://docs.jax.dev/en/latest/installation.html
