source /when2reason/install/scout_mini_base/share/scout_mini_base/local_setup.bash
source /when2reason/install/scout_mini_description/share/scout_mini_description/local_setup.bash
sudo ip link set can0 up type can bitrate 500000
ros2 launch scout_mini_base base_launch.py
