#!/bin/bash
# Uni-NaVid Docker 컨테이너에 명령 실행 스크립트

CONTAINER_NAME="uninavid-container"

# 컨테이너가 실행 중인지 확인
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ 오류: 컨테이너 '${CONTAINER_NAME}'가 실행 중이 아닙니다."
    echo ""
    echo "컨테이너를 시작하려면:"
    echo "  ./docker_run.sh"
    exit 1
fi

# 명령이 제공되었으면 실행, 아니면 bash 세션 시작
if [ $# -eq 0 ]; then
    echo "컨테이너 '${CONTAINER_NAME}'에 접속합니다..."
    docker exec -it "${CONTAINER_NAME}" /bin/bash
else
    echo "컨테이너 '${CONTAINER_NAME}'에서 명령 실행: $@"
    docker exec -it "${CONTAINER_NAME}" "$@"
fi

