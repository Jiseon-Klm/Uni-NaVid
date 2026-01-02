# syntax=docker/dockerfile:1.7

# DGX Spark(GB10, sm_121)에서는 CUDA 13 계열/Blackwell 대응 PyTorch가 들어간 컨테이너가 필요해요.
# NGC PyTorch 25.11은 CUDA 13.0.2 + PyTorch 포함(Blackwell 최적화 언급) :contentReference[oaicite:3]{index=3}
ARG NGC_TAG=25.11-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ---- system deps (빌드/런타임 최소 안정 세트) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    ca-certificates curl git \
    build-essential pkg-config \
    python3-venv python3-dev \
    cmake ninja-build \
    ffmpeg \
    libsentencepiece-dev \
    && rm -rf /var/lib/apt/lists/*

# locale
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# ---- Rust 최신 설치 (tokenizers가 소스 빌드로 떨어져도 edition2024로 터지지 않게) ----
# Ubuntu apt의 rust/cargo가 구버전이면 edition2024 에러가 나요 -> rustup으로 최신 stable 사용 :contentReference[oaicite:4]{index=4}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal && \
    source /root/.cargo/env && \
    rustup toolchain install stable && rustup default stable && \
    rustc --version && cargo --version
ENV PATH=/root/.cargo/bin:$PATH

# ---- venv (system-site-packages로 "컨테이너에 원래 있던 torch"를 그대로 씀) ----
# 핵심: torch/cuda는 NGC에 들어있는 걸 그대로 쓰고, pip가 torch를 갈아엎지 못하게 격리해요.
RUN python3 -m venv --system-site-packages /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /when2reason
COPY . /when2reason

# ---- (중요) 로컬 패키지는 의존성 설치를 "절대" 같이 하지 않아요 ----
# pyproject.toml의 오래된 핀(예: transformers==4.31.0)이 Py3.12에서 소스빌드를 유발해서 지옥문 열어요.
RUN pip install -e . --no-deps

# ---- DGX Spark/py3.12에서 wheel 잘 잡히는 쪽으로 runtime deps만 설치 ----
# (너 pyproject.toml은 건드리지 않고, 컨테이너에서만 안전 세트로 설치)
# tokenizers는 Py3.12 wheel이 있는 버전대(>=0.19.x)를 쓰는 게 안정적이라는 케이스가 많아요. :contentReference[oaicite:5]{index=5}
RUN cat > /tmp/requirements.dgx_spark.txt <<'REQ'
numpy<2
packaging
pyyaml
tqdm
pillow
opencv-python-headless

einops
timm

# HF stack (Py3.12 호환 + tokenizers 최신 계열)
transformers>=4.44.0
tokenizers>=0.19.1
accelerate>=0.30.0
peft>=0.10.0
safetensors
huggingface_hub
sentencepiece

# Uni-NaVid 쪽에서 import 흔한 것들
httpx==0.24.0
wandb
shortuuid
fairscale

# 너 로그에 deepspeed가 올라오니까 포함.
# CUDA 커널 ops 빌드는 피하고(DS_BUILD_OPS=0) pure python로 쓰는 게 실패 확률이 훨씬 낮아요.
deepspeed==0.9.5
REQ

ENV DS_BUILD_OPS=0
RUN pip install --no-cache-dir -r /tmp/requirements.dgx_spark.txt

# ---- sanity check: torch/cuda/device capability ----
RUN python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime (torch.version.cuda):", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
else:
    print("CUDA not available inside container (check --gpus all / nvidia-container-runtime)")
PY

# ---- Intel RealSense SDK (librealsense2) Full Setup ----
# 1. 필수 의존성 설치 (OpenSSL, USB, 그래픽 라이브러리 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev libusb-1.0-0-dev pkg-config \
    libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    libx11-dev libxcursor-dev libxinerama-dev libxrandr-dev libxi-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. 소스 빌드 및 설치 (GUI 도구 및 Python 바인딩 포함)
# FORCE_RSUSB_BACKEND: 커널 패치 없이 Docker에서 장치 인식을 위해 필수
# BUILD_GRAPHICAL_EXAMPLES: realsense-viewer 빌드를 위해 필수
RUN git clone https://github.com/IntelRealSense/librealsense.git /tmp/librealsense && \
    cd /tmp/librealsense && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_PYTHON_BINDINGS:bool=true \
        -DPYTHON_EXECUTABLE=$(which python) \
        -DBUILD_EXAMPLES:bool=true \
        -DBUILD_GRAPHICAL_EXAMPLES:bool=true \
        -DFORCE_RSUSB_BACKEND=true && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])") && \
    cp $PY_SITE/pyrealsense2/pyrealsense2*.so $PY_SITE/ && \
    rm -rf /tmp/librealsense
    
    # ---- ROS2 Setup (OS 버전에 맞춰 Humble/Jazzy 자동 선택) ----
# 1. 리포지토리 설정 및 키 등록
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 2. OS 버전에 따른 ROS 배포판 감지 및 설치
# Ubuntu 22.04(jammy) -> humble / Ubuntu 24.04(noble) -> jazzy
RUN export UBUNTU_CODENAME=$(. /etc/os-release && echo $UBUNTU_CODENAME) && \
    if [ "$UBUNTU_CODENAME" = "jammy" ]; then \
        export ROS_DISTRO=humble; \
    elif [ "$UBUNTU_CODENAME" = "noble" ]; then \
        export ROS_DISTRO=jazzy; \
    fi && \
    echo "Selected ROS_DISTRO: $ROS_DISTRO for $UBUNTU_CODENAME" && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-ros-base \
    ros-dev-tools \
    python3-colcon-common-extensions \
    ros-${ROS_DISTRO}-realsense2-camera \
    && rm -rf /var/lib/apt/lists/* && \
    echo $ROS_DISTRO > /tmp/ros_distro_name

# 3. venv와 ROS2 환경 병합 (PYTHONPATH 유지)
# NGC 25.11-py3는 Ubuntu 24.04(Jazzy)일 가능성이 높으므로 기본값을 jazzy로 설정하되,
# 필요시 수동으로 humble로 변경 가능합니다.
ENV ROS_DISTRO=jazzy
ENV AMENT_PREFIX_PATH=/opt/ros/${ROS_DISTRO}
ENV COLCON_PREFIX_PATH=/opt/ros/${ROS_DISTRO}
ENV LD_LIBRARY_PATH=/opt/ros/${ROS_DISTRO}/lib:${LD_LIBRARY_PATH}
ENV PATH=/opt/ros/${ROS_DISTRO}/bin:${PATH}
# venv 내부에서 시스템 ROS 패키지를 import할 수 있도록 설정
ENV PYTHONPATH=/opt/ros/${ROS_DISTRO}/lib/python3/dist-packages:${PYTHONPATH}

# 4. Entrypoint 및 환경 설정
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]