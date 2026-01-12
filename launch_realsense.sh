ros2 launch realsense2_camera rs_launch.py \
  rgb_camera.profile:=640x480x10 \
  enable_color:=true \
  enable_depth:=false \
  enable_infra:=false \
  enable_infra1:=false \
  enable_infra2:=false \
  align_depth.enable:=false \
  pointcloud.enable:=false
