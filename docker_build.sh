#!/bin/bash
# Uni-NaVid Docker 이미지 빌드 스크립트

set -e

# 이미지 이름과 태그 설정
IMAGE_NAME="uninavid"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# 프로젝트 루트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

echo "=========================================="
echo "Uni-NaVid Docker 이미지 빌드"
echo "=========================================="
echo "이미지 이름: ${FULL_IMAGE_NAME}"
echo "프로젝트 디렉토리: ${PROJECT_ROOT}"
echo ""

# Dockerfile 존재 확인
if [ ! -f "${PROJECT_ROOT}/Dockerfile" ]; then
    echo "❌ 오류: Dockerfile을 찾을 수 없습니다: ${PROJECT_ROOT}/Dockerfile"
    exit 1
fi

# NVIDIA Container Toolkit 확인
if ! command -v nvidia-container-toolkit &> /dev/null; then
    echo "⚠️  경고: nvidia-container-toolkit이 설치되지 않았습니다."
    echo "   GPU 지원을 위해서는 설치가 필요합니다."
fi

# Docker 빌드 시작
echo "Docker 이미지 빌드 시작..."
echo ""

cd "${PROJECT_ROOT}"

# BuildKit 사용하여 빌드 (더 빠르고 효율적)
DOCKER_BUILDKIT=1 docker build \
    -t "${FULL_IMAGE_NAME}" \
    -f Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 빌드 완료!"
    echo ""
    echo "빌드된 이미지 확인:"
    docker images | grep "${IMAGE_NAME}"
    echo ""
    echo "다음 명령어로 컨테이너를 실행할 수 있습니다:"
    echo "  ./docker_run.sh"
    echo ""
    echo "또는 직접 실행:"
    echo "  docker run -it --gpus all ${FULL_IMAGE_NAME}"
else
    echo ""
    echo "❌ 빌드 실패"
    exit 1
fi

