# Docker 사용 가이드

이 문서는 Uni-NaVid 프로젝트를 Docker로 빌드하고 실행하는 방법을 설명합니다.

## 필요한 사전 요구사항

1. **Docker** (20.10 이상)
2. **NVIDIA Container Toolkit** (GPU 사용 시)
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

## 빠른 시작

### 1. Docker 이미지 빌드

```bash
./docker_build.sh
```

이 스크립트는:
- Dockerfile을 사용하여 이미지를 빌드합니다
- 이미지 이름: `uninavid:latest`
- BuildKit을 사용하여 최적화된 빌드를 수행합니다

### 2. Docker 컨테이너 실행

```bash
./docker_run.sh
```

이 스크립트는:
- GPU 지원 (`--gpus all`)
- X11 디스플레이 지원 (GUI 앱용)
- 프로젝트 디렉토리를 컨테이너에 마운트
- `model_zoo` 디렉토리를 마운트
- 네트워크 호스트 모드
- 8GB 공유 메모리

### 3. 실행 중인 컨테이너에 접속

```bash
./docker_exec.sh
```

또는 특정 명령 실행:
```bash
./docker_exec.sh python offline_eval_uninavid.py test_cases/vln_1 output_dir
```

## 수동 실행 (고급)

### 이미지 빌드

```bash
docker build -t uninavid:latest -f Dockerfile .
```

### 컨테이너 실행

```bash
docker run -it \
    --name uninavid-container \
    --gpus all \
    --privileged \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/when2reason/Uni-NaVid \
    -v $(pwd)/model_zoo:/when2reason/model_zoo \
    -w /when2reason/Uni-NaVid \
    --shm-size=8g \
    uninavid:latest \
    /bin/bash
```

## 유용한 Docker 명령어

### 컨테이너 상태 확인
```bash
docker ps -a | grep uninavid
```

### 컨테이너 중지
```bash
docker stop uninavid-container
```

### 컨테이너 제거
```bash
docker rm uninavid-container
```

### 컨테이너 로그 확인
```bash
docker logs uninavid-container
```

### 이미지 제거
```bash
docker rmi uninavid:latest
```

## 문제 해결

### GPU가 인식되지 않는 경우

1. NVIDIA Container Toolkit이 설치되어 있는지 확인:
   ```bash
   nvidia-container-toolkit --version
   ```

2. Docker가 GPU를 사용할 수 있는지 테스트:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

### X11 디스플레이 오류

GUI 애플리케이션이 필요한 경우:
```bash
xhost +local:docker
```

### 권한 오류

Docker 그룹에 사용자 추가:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 컨테이너 내부에서 작업

컨테이너 내부 작업 디렉토리는 `/when2reason/Uni-NaVid`입니다.

### 평가 실행 예시

```bash
# 컨테이너 내부에서
cd /when2reason/Uni-NaVid
python offline_eval_uninavid.py test_cases/vln_1 output_dir
```

### 학습 실행 예시

```bash
# 컨테이너 내부에서
cd /when2reason/Uni-NaVid
bash scripts/uninavid_stage_1.sh
```

## 볼륨 마운트 정보

- **프로젝트 루트**: 호스트의 프로젝트 디렉토리가 `/when2reason/Uni-NaVid`로 마운트
- **model_zoo**: `/when2reason/model_zoo`로 마운트 (모델 체크포인트용)

호스트의 변경사항은 컨테이너에서 즉시 반영되며, 컨테이너 내부의 변경사항도 호스트에 저장됩니다.

