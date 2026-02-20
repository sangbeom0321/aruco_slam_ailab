
# ============================================================
# ROS2 Humble Setup
# ============================================================
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash 2>/dev/null

# Colcon autocomplete
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash 2>/dev/null

# ROS2 Domain
export ROS_DOMAIN_ID=0

# ============================================================
# 편의 명령어
# ============================================================

# colcon build
alias cb='cd /ros2_ws && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'
alias cbs='cd /ros2_ws && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select'
alias cbr='cd /ros2_ws && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release'

# source
alias sb='source /root/.bashrc'
alias si='source /ros2_ws/install/setup.bash'

# ros2 shortcuts
alias rt='ros2 topic list'
alias rte='ros2 topic echo'
alias rth='ros2 topic hz'
alias rti='ros2 topic info'
alias rn='ros2 node list'
alias rni='ros2 node info'
alias rp='ros2 param list'
alias rpg='ros2 param get'
alias rs='ros2 service list'
alias rsc='ros2 service call'
alias rb='ros2 bag record'
alias rbl='ros2 bag info'
alias rl='ros2 launch'

# tf2
alias tf='ros2 run tf2_tools view_frames'
alias tfe='ros2 run tf2_ros tf2_echo'

# workspace
alias ws='cd /ros2_ws'
alias src='cd /ros2_ws/src'
alias slam='cd /ros2_ws/src/hunter/aruco_sam_ailab'

# 자주 쓰는 launch
alias slam_hw='ros2 launch aruco_sam_ailab hardware.launch.py'
alias rviz_slam='rviz2 -d /ros2_ws/src/hunter/aruco_sam_ailab/config/slam_rviz.rviz'

# kill all ros nodes
killros() {
  pkill -9 -f 'ros2|rviz2|realsense|aruco_det|imu_pre|graph_opt' 2>/dev/null
  echo "All ROS processes killed"
}

# colcon clean
alias cc='rm -rf /ros2_ws/build /ros2_ws/install /ros2_ws/log && echo Clean done'
