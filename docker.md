
ROS 官方提供 `ros:humble` / `osrf/ros:humble-desktop` 等镜像；Humble 对应 Ubuntu 22.04。你的项目需要 JAX GPU，因此更推荐以 NVIDIA CUDA Ubuntu22.04 镜像为基础，再安装 ROS 2 Humble。ROS 官方 Docker 教程也使用 `osrf/ros:humble-desktop` 作为 Humble 示例镜像。([ROS][1])
JAX 官方推荐通过 pip 安装 CUDA 版本，例如 `jax[cuda12]`，这适合 Docker 内部环境。([jax.dev][2])

---

## 一、进入项目目录

你的项目路径是：

```bash
cd "/home/hxy-ubuntu2404/文档/teacher's_project/EXACT-mppi"
```

建议在项目中创建 Docker 文件夹，便于后续开源：

```bash
mkdir -p docker
```

---

## 二、创建 Dockerfile

创建文件：

```bash
gedit docker/Dockerfile.humble-cuda
```

写入以下内容：

```dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV ROS_DISTRO=humble
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# 基础系统依赖
RUN apt-get update && apt-get install -y \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    build-essential \
    git \
    wget \
    vim \
    nano \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-colcon-common-extensions \
    python3-argcomplete \
    && locale-gen en_US en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# 添加 ROS 2 Humble 软件源
RUN add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    > /etc/apt/sources.list.d/ros2.list

# 安装 ROS 2 Humble ros-base
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# 初始化 rosdep，网络失败不终止构建
RUN rosdep init || true
RUN rosdep update || true

# Python 基础工具
RUN python3 -m pip install --upgrade pip setuptools wheel

# 容器启动时自动加载 ROS 2 Humble
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

WORKDIR /workspace/EXACT-mppi

CMD ["/bin/bash"]
```

---

## 三、创建 `.dockerignore`

建议一起开源，但不要把虚拟环境、缓存、训练输出打进镜像上下文。

```bash
gedit .dockerignore
```

写入：

```gitignore
.git
__pycache__/
*.pyc
*.pyo
*.pyd

.exact_mppi/
.venv/
venv/
env/

build/
install/
log/

runs/
outputs/
wandb/
.cache/
.pytest_cache/

*.bag
*.db3
*.pt
*.pth
*.onnx
```

---

## 四、构建镜像

在项目根目录执行：

```bash
cd "/home/hxy-ubuntu2404/文档/teacher's_project/EXACT-mppi"

docker build \
  -f docker/Dockerfile.humble-cuda \
  -t exact-mppi:humble-cuda12.1 \
  .
```

构建完成后检查：

```bash
docker images | grep exact-mppi
```

应该能看到：

```bash
exact-mppi    humble-cuda12.1
```

---

## 五、创建可长期使用的容器实例

注意：**不要加 `--rm`**，否则退出后容器会被删除。

```bash
docker run -it \
  --gpus all \
  --name EXACT-mppi_humble_cuda \
  --network host \
  --ipc host \
  --shm-size=8g \
  -v "/home/hxy-ubuntu2404/文档/teacher's_project/EXACT-mppi:/workspace/EXACT-mppi" \
  exact-mppi:humble-cuda12.1 \
  /bin/bash
```

以后退出容器后，可以随时重新进入：

```bash
docker start -ai EXACT-mppi_humble_cuda
```

如果容器已经在后台运行，用：

```bash
docker exec -it EXACT-mppi_humble_cuda bash
```

---

## 六、在 Docker 中确认 GPU 可用

进入容器后先执行：

```bash
nvidia-smi
```

如果能看到 RTX 4090 / A100 / 你的显卡信息，说明 Docker GPU 透传正常。

再检查 CUDA 编译器：

```bash
nvcc --version
```

然后检查 ROS：

```bash
source /opt/ros/humble/setup.bash
ros2 --version
```

---

## 七、在容器中安装 EXACT-mppi 项目环境

容器内执行：

```bash
cd /workspace/EXACT-mppi
```

创建 Python 虚拟环境：

```bash
python3 -m venv .exact_mppi
source .exact_mppi/bin/activate
```

升级基础包：

```bash
python -m pip install --upgrade pip setuptools wheel
```

按照你项目说明，推荐顺序为：

```bash
python -m pip install -e ./ir-sim_mppi
python -m pip install -U "jax[cuda12]"
python -m pip install -e ./EXACT_MPPI_core
```

如果旧版 torch 组件也要运行，再装：

```bash
python -m pip install torch arm-pytorch-utilities
```

---

## 八、验证 JAX 是否使用 GPU

容器内、虚拟环境激活后执行：

```bash
python -c "import jax; print(jax.devices())"
```

正常 GPU 输出类似：

```bash
[CudaDevice(id=0)]
```

如果输出是：

```bash
[CpuDevice(id=0)]
```

说明 JAX 没有识别 GPU，需要检查：

```bash
nvidia-smi
python -c "import jax; print(jax.default_backend())"
python -m pip show jax jaxlib
```

---

## 九、验证项目安装是否成功

```bash
python -c "import irsim; from exact_mppi.mppi_jax.controller import MPPIController; print('setup ok')"
```

如果输出：

```bash
setup ok
```

说明项目基础依赖安装成功。

---

## 十、运行 EXACT_MPPI_core 示例

进入示例目录：

```bash
cd /workspace/EXACT-mppi/EXACT_MPPI_core/example
```

先看有哪些文件：

```bash
ls
```

然后运行对应示例，例如如果里面有：

```bash
python xxx.py
```

你就直接执行：

```bash
python xxx.py
```

运行前建议先确认 JAX 后端：

```bash
python -c "import jax; print('backend:', jax.default_backend()); print(jax.devices())"
```

---

## 十一、推荐加入 README 的启动说明

你后续开源时，可以在 README 中加入：

````markdown
## Docker Environment

This project provides a Docker environment based on Ubuntu 22.04, ROS 2 Humble, CUDA 12.1 and Python 3.10.

### Build image

```bash
docker build -f docker/Dockerfile.humble-cuda -t exact-mppi:humble-cuda12.1 .
````

### Start container with GPU

```bash
docker run -it \
  --gpus all \
  --name EXACT-mppi_humble_cuda \
  --network host \
  --ipc host \
  --shm-size=8g \
  -v "$(pwd):/workspace/EXACT-mppi" \
  exact-mppi:humble-cuda12.1 \
  /bin/bash
```

### Install Python dependencies

```bash
cd /workspace/EXACT-mppi
python3 -m venv .exact_mppi
source .exact_mppi/bin/activate
python -m pip install --upgrade pip setuptools wheel

python -m pip install -e ./ir-sim_mppi
python -m pip install -U "jax[cuda12]"
python -m pip install -e ./EXACT_MPPI_core
```

### Check GPU

```bash
nvidia-smi
python -c "import jax; print(jax.devices())"
```

````

---

## 十二、如果 Docker 拉取超时

你的上一条报错：

```bash
context deadline exceeded
````

是 Docker Hub 网络超时。可以先单独拉基础镜像：

```bash
docker pull nvidia/cuda:12.1.1-devel-ubuntu22.04
```

失败就重复执行，或者用循环重试：

```bash
until docker pull nvidia/cuda:12.1.1-devel-ubuntu22.04; do
  echo "pull failed, retrying..."
  sleep 5
done
```

---

## 最终你应执行的核心命令汇总

```bash
cd "/home/hxy-ubuntu2404/文档/teacher's_project/EXACT-mppi"

mkdir -p docker
gedit docker/Dockerfile.humble-cuda
gedit .dockerignore

docker build \
  -f docker/Dockerfile.humble-cuda \
  -t exact-mppi:humble-cuda12.1 \
  .

docker run -it \
  --gpus all \
  --name EXACT-mppi_humble_cuda \
  --network host \
  --ipc host \
  --shm-size=8g \
  -v "/home/hxy-ubuntu2404/文档/teacher's_project/EXACT-mppi:/workspace/EXACT-mppi" \
  exact-mppi:humble-cuda12.1 \
  /bin/bash
```

容器内：

```bash
nvidia-smi
source /opt/ros/humble/setup.bash

cd /workspace/EXACT-mppi
python3 -m venv .exact_mppi
source .exact_mppi/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ./ir-sim_mppi
python -m pip install -U "jax[cuda12]"
python -m pip install -e ./EXACT_MPPI_core

python -c "import jax; print(jax.devices())"
python -c "import irsim; from exact_mppi.mppi_jax.controller import MPPIController; print('setup ok')"
```

[1]: https://docs.ros.org/en/humble/How-To-Guides/Run-2-nodes-in-single-or-separate-docker-containers.html?utm_source=chatgpt.com "Running ROS 2 nodes in Docker [community-contributed]"
[2]: https://jax.dev/?utm_source=chatgpt.com "JAX: High performance array computing — JAX documentation"
