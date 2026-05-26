<h1 align="center">EXACT-MPPI</h1>

<p align="center">
  <strong>Exact Non-Convex Shape-Aware MPPI for Agile Robot Navigation</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10" /></a>
  <a href="#"><img src="https://img.shields.io/badge/JAX-GPU%20%7C%20CPU-FF6F00?logo=google&logoColor=white" alt="JAX" /></a>
  <a href="#"><img src="https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros&logoColor=white" alt="ROS 2 Humble" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker" /></a>
</p>

<p align="center">
  <img src="videos/figure1_system_architecture.png" width="90%" alt="EXACT-MPPI system architecture" />
</p>

---

## Overview

Ground robots equipped with payloads, tools, or structural attachments often have **complex, non-convex footprints** that conventional planners cannot handle faithfully. Standard approaches approximate the robot body with a convex hull or inflated bounding shape, and rasterize sensor data into occupancy grids or distance fields. Both choices discard feasible motions when the available clearance is comparable to the true footprint geometry.

EXACT-MPPI is a **training-free** local navigation framework that bypasses these limitations. It maps raw **point-cloud observations** and a sparse global guidance path directly to motion commands — with **no intermediate map representation**. The core of the framework is an analytic, exact signed-distance evaluator embedded inside an MPPI controller: during every sampled rollout, obstacle points are transformed into the predicted robot body frame and evaluated against the full footprint geometry, making collision costs exact rather than approximate at every candidate trajectory.

The footprint is represented as a **simple polygon** for general convex or concave planar shapes, with a **rectangle-cover specialization** (union of axis-aligned rectangles) for faster evaluation of rectilinear bodies. No convex decomposition, shape inflation, or learned encoder is required. All rollout computations are batched in **JAX** for GPU-parallel, real-time receding-horizon control.

The same codebase deploys across **differential-drive, Ackermann, omnidirectional, and hybrid-mode platforms** by changing only the footprint description and motion model — no per-platform retraining.

---

## Key Features

- **Exact signed-distance evaluation** — analytic footprint SDF computed per rollout step directly against raw point clouds; no occupancy grid, distance field, or inflation layer
- **Arbitrary footprint support** — simple polygon for convex/concave shapes; rectangle-cover specialization for rectilinear bodies (e.g. T-shape, F-shape)
- **Training-free** — no learned encoder or per-platform training; only the footprint description and motion model are platform-specific
- **Direct point-cloud input** — obstacle points consumed as-is from the sensor, transformed into the predicted body frame at each rollout step
- **Dynamic obstacle handling** — moving obstacles handled online through receding-horizon rollout re-evaluation
- **Multi-platform** — validated on differential-drive, Ackermann, omnidirectional, and hybrid-mode robots without retraining
- **JAX GPU backend** — JIT-compiled batched rollouts for real-time control; runs on GPU or CPU with no code changes
- **ROS 2 Humble + Gazebo + Docker** — full simulation stack with a ready-to-run container

---

## Simulation Demonstrations

<table>
  <tr>
    <th align="center">Cluttered Corridor — T-Shape Robot</th>
    <th align="center">Dynamic Obstacles — F-Shape Robot</th>
    <th align="center">Narrow-Gap Navigation</th>
  </tr>
  <tr>
    <td align="center">
      <img src="videos/corridor_cluttered.gif" width="100%" alt="Cluttered corridor navigation" />
    </td>
    <td align="center">
      <img src="videos/corridor_dynamic_anyshape_f.gif" width="100%" alt="F-shape robot with dynamic obstacles" />
    </td>
    <td align="center">
      <img src="videos/narrow_gaps.gif" width="100%" alt="Narrow-gap navigation" />
    </td>
  </tr>
  <tr>
    <td align="center"><sub>Rectangle-cover footprint navigating through static polygon obstacles in a tight corridor</sub></td>
    <td align="center"><sub>F-shape footprint avoiding 8 randomly moving polygon obstacles</sub></td>
    <td align="center"><sub>Exact footprint SDF preserving feasible motion through gaps near robot width</sub></td>
  </tr>
</table>

---

## Real-World Deployment

EXACT-MPPI has been deployed on three hardware platforms without hardware-specific tuning — only the footprint description and motion model change between robots.

<table>
  <tr>
    <th align="center">Differential Drive Robot</th>
    <th align="center">Swerve Drive Robot</th>
    <th align="center">Quadrupedal Robot</th>
  </tr>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/60513fe5-9388-4718-be83-48ee510ae98c" width="100%" controls></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/8300cdb7-86d7-48e7-a77c-482bb2e17e55" width="100%" controls></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/80982db2-d9c7-454a-9023-6f732b195ab2" width="100%" controls></video>
    </td>
  </tr>
</table>

---

## Getting Started

The fastest path is Docker — no local ROS or JAX install needed.

```bash
# Build the image (first build takes several minutes)
chmod +x docker/build_docker.sh docker/run_docker.sh
./docker/build_docker.sh

# Start the container (GPU + X11 forwarding)
./docker/run_docker.sh

# Inside the container — run the F-shape dynamic obstacle demo
python EXACT_MPPI_core/example/corridor_dynamic_random_jax/mppi_jax_anyshape.py --robot-shape f

# Save a GIF (add -a; --gif-speed 2 plays back at 2× speed)
python EXACT_MPPI_core/example/corridor_dynamic_random_jax/mppi_jax_anyshape.py --robot-shape f -a --gif-speed 2
```

For a local (non-Docker) install, see the detailed instructions below.

---

## Repository Structure

```
EXACT-mppi/
├── ir-sim_mppi/               # Simulator package (irsim fork)
├── EXACT_MPPI_core/           # MPPI controller and Python examples
│   └── example/
│       ├── corridor_dynamic_random_jax/   # Dynamic obstacles, any robot shape
│       ├── corridor_jax_cluttered/        # Static cluttered corridor
│       ├── narrow_gaps_jax/              # Narrow-passage stress test
│       └── T-shape_trap_jax/             # T-shape trap scenario
├── mosaic_mppi_ros2/          # ROS 2 workspace (Humble + Gazebo)
├── docker/                    # Dockerfile and helper scripts
└── videos/                    # System architecture figure, demo GIFs, and deployment videos
```

---

## Installation

There are two ways to build and run everything:

1. **[Docker (recommended)](#1-docker-recommended)** — a fully configured CUDA + ROS 2 Humble image.
2. **[Local install (no Docker)](#2-local-install-without-docker)** — set everything up directly on the host.

### 1. Docker (recommended)

#### What the image provides

The image is defined in `docker/Dockerfile.humble-cuda` and includes:

- `nvidia/cuda:12.1.1-devel-ubuntu22.04`
- ROS 2 Humble (`ros-base`) with all workspace dependencies pre-installed
- Gazebo, RViz, `xacro`, `robot_state_publisher`, and `joint_state_publisher`
- a Python virtual environment at `/opt/exact_mppi/venv`
- CUDA 12 JAX via `jax[cuda12]`
- `python3-tk` and `python3-pil.imagetk` for GUI components
- editable installs of `ir-sim_mppi` and `EXACT_MPPI_core`

ROS 2 and the virtual environment are sourced automatically in every shell via `~/.bashrc`.

#### Prerequisites

- Docker
- NVIDIA driver
- NVIDIA Container Toolkit (required for `--gpus all`)
- X11 support (only needed for Gazebo / RViz windows)

```bash
docker --version && nvidia-smi && echo "DISPLAY=$DISPLAY"
```

#### Build

```bash
chmod +x docker/build_docker.sh docker/run_docker.sh
./docker/build_docker.sh
```

Tags the image `exact-mppi:humble-cuda`. First build downloads large packages and takes a while; subsequent builds reuse the cache.

#### Run

```bash
./docker/run_docker.sh
```

Starts an interactive container with `--gpus all`, host networking, X11 forwarding, and the repository mounted read-write at `/workspace/EXACT-mppi`. Editable-mode installs mean **host edits take effect immediately** — no rebuild needed.

---

### 2. Local install (without Docker)

Tested on **Ubuntu 22.04** with **Python 3.10**.

#### 2a. Minimal (core Python only)

```bash
# System packages
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip \
    python3-tk python3-pil.imagetk

# Virtual environment
python3 -m venv --system-site-packages .exact_mppi
source .exact_mppi/bin/activate

pip install --upgrade pip setuptools wheel
pip install lxml
pip install -e ./ir-sim_mppi
pip install -U "jax[cuda12]"   # or "jax[cpu]" for CPU-only
pip install -e ./EXACT_MPPI_core
```

> `--system-site-packages` is required so the venv sees `python3-tk` / `python3-pil.imagetk` and (in the full setup) ROS 2's `rclpy`.

Match the JAX wheel to your CUDA **major** version only (CUDA 12.x → `jax[cuda12]`).
Reference: https://docs.jax.dev/en/latest/installation.html

#### 2b. Full setup (ROS 2 + Gazebo)

Do this **in addition to 2a** for the ROS 2 / Gazebo examples.

```bash
# 1. Install ROS 2 Humble — https://docs.ros.org/en/humble/Installation.html

# 2. Simulation tools and build deps
sudo apt install -y \
    ros-humble-ros-base gazebo \
    ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins ros-humble-gazebo-msgs \
    ros-humble-xacro ros-humble-rviz2 \
    ros-humble-joint-state-publisher ros-humble-joint-state-publisher-gui \
    ros-humble-robot-state-publisher \
    ros-humble-tf-transformations ros-humble-tf2-tools ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs ros-humble-nav-msgs ros-humble-sensor-msgs \
    ros-humble-geometry-msgs ros-humble-visualization-msgs \
    libeigen3-dev libyaml-cpp-dev cmake \
    python3-rosdep python3-argcomplete python3-colcon-common-extensions

sudo rosdep init || true && rosdep update

# 3. Build the ROS 2 workspace
source /opt/ros/humble/setup.bash
source .exact_mppi/bin/activate
cd mosaic_mppi_ros2
colcon build
source install/setup.bash
```

---

## Running the Examples

> **Docker users:** the environment is pre-sourced — run commands directly.
> **Local users:** activate the venv (and source ROS 2 for ROS examples) first.

```bash
# Sanity check
python -c "import irsim; from exact_mppi.mppi_jax.controller import MPPIController; print('OK')"
```

### Core Python examples (no ROS)

| Scenario | Command |
|---|---|
| Dynamic obstacles — any shape | `python EXACT_MPPI_core/example/corridor_dynamic_random_jax/mppi_jax_anyshape.py --robot-shape f` |
| Dynamic obstacles — T/rect shape | `python EXACT_MPPI_core/example/corridor_dynamic_random_jax/mppi_jax_test.py --robot_shape t` |
| Cluttered corridor | `python EXACT_MPPI_core/example/corridor_jax_cluttered/mppi_jax_test.py` |
| Narrow gaps | `python EXACT_MPPI_core/example/narrow_gaps_jax/mppi_jax_test.py` |
| T-shape trap | `python EXACT_MPPI_core/example/T-shape_trap_jax/mppi_jax_test.py` |

Add `-a` to save a GIF; `--gif-speed 2` for 2× playback speed.

### ROS 2 examples

```bash
cd mosaic_mppi_ros2 && source install/setup.bash

# Corridor simulator
ros2 launch exact_mppi_jax sim_corridor_external_ref_launch.py

# T-shape simulator
ros2 launch exact_mppi_jax sim_tshape_external_ref_launch.py

# Gazebo LIMO
ros2 launch exact_mppi_jax sim_limo_corridor_launch.py auto_goal:=true
```

---

## Notes

- **Apt mirrors.** The Dockerfile uses Tsinghua mirrors to accelerate builds inside China. Comment out those lines if you are building from outside China.
- **JAX / CUDA.** Match the wheel to your CUDA **major** version (`jax[cuda12]` for CUDA 12.x; `jax[cpu]` for CPU-only). See https://docs.jax.dev/en/latest/installation.html
- **X11.** GUI examples need a working display. Inside Docker this requires X11 forwarding; locally it uses your current desktop session.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{exactmppi2025,
  title   = {EXACT-MPPI: Exact Signed-Distance Navigation for Arbitrary-Footprint Robots
             from Point Clouds via Path Integral Control},
  author  = {},
  journal = {},
  year    = {2025},
  keywords = {Robot Navigation, Model Predictive Path Integral Control,
              Signed-Distance Collision Avoidance, Arbitrary Footprint}
}
```
