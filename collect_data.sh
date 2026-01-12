# Docker 내부 터미널
ros2 launch realsense2_camera rs_launch.py \
  rgb_camera.profile:=640x480x10 \
  depth_module.profile:=640x480x10 \
  enable_depth:=true \
  align_depth.enable:=true \
  pointcloud.enable:=false

# Docker 내부 터미널
# 저장할 디렉토리로 이동 후
cd /data_storage 

ros2 bag record -s mcap \
  -o session_01 \
  /camera/color/image_raw
