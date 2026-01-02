#!/bin/bash
# Uni-NaVid Docker 컨테이너 실행 스크립트

set -e

# 이미지 이름
IMAGE_NAME="uninavid:latest"
CONTAINER_NAME="uninavid-container"

# 프로젝트 루트 디렉토리 (호스트)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
HOST_WORKDIR="${PROJECT_ROOT}"

# 컨테이너 내부 작업 디렉토리
CONTAINER_WORKDIR="/when2reason/Uni-NaVid"

# X11 디스플레이 설정 (GUI 앱이 필요한 경우)
X11_SOCK="/tmp/.X11-unix"
DISPLAY_VAR="${DISPLAY:-:0}"

echo "=========================================="
echo "Uni-NaVid Docker 컨테이너 실행"
echo "=========================================="
echo "이미지: ${IMAGE_NAME}"
echo "컨테이너 이름: ${CONTAINER_NAME}"
echo "호스트 작업 디렉토리: ${HOST_WORKDIR}"
echo "컨테이너 작업 디렉토리: ${CONTAINER_WORKDIR}"
echo ""

# 이미지 존재 확인
if ! docker images | grep -q "uninavid.*latest"; then
    echo "❌ 오류: Docker 이미지를 찾을 수 없습니다: ${IMAGE_NAME}"
    echo ""
    echo "먼저 이미지를 빌드해주세요:"
    echo "  ./docker_build.sh"
    exit 1
fi

# 실행 중인 컨테이너가 있으면 제거 옵션
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "⚠️  기존 컨테이너 '${CONTAINER_NAME}'가 발견되었습니다."
    read -p "기존 컨테이너를 제거하고 새로 시작하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "기존 컨테이너 제거 중..."
        docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    else
        echo "기존 컨테이너를 사용합니다."
        echo "컨테이너에 접속: docker exec -it ${CONTAINER_NAME} bash"
        exit 0
    fi
fi

# GPU 확인
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU 감지됨"
    GPU_FLAG="--gpus all"
    # GPU 정보 출력
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "⚠️  NVIDIA GPU를 찾을 수 없습니다. CPU 모드로 실행합니다."
    GPU_FLAG=""
fi

echo ""
echo "컨테이너 실행 중..."
echo ""

# X11 소켓 권한 설정
if [ -S "${X11_SOCK}" ]; then
    xhost +local:docker 2>/dev/null || true
fi

# Docker 컨테이너 실행
docker run -it \
    --name "${CONTAINER_NAME}" \
    ${GPU_FLAG} \
    --privileged \
    --net=host \
    -e DISPLAY="${DISPLAY_VAR}" \
    -e QT_X11_NO_MITSHM=1 \
    -v "${X11_SOCK}:${X11_SOCK}" \
    -v "${HOST_WORKDIR}:${CONTAINER_WORKDIR}" \
    -v "${PROJECT_ROOT}/model_zoo:/when2reason/model_zoo" \
    -w "${CONTAINER_WORKDIR}" \
    --shm-size=8g \
    "${IMAGE_NAME}" \
    /bin/bash

# X11 권한 복원
xhost -local:docker 2>/dev/null || true

