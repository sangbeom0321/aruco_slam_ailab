# Graph optimization 결과: 키프레임 시점의 최적화된 pose/velocity/bias
# IMU 사전적분 노드가 이 값으로 재초기화하여 누적 오차를 리셋
std_msgs/Header header
geometry_msgs/Pose pose          # base_link frame
geometry_msgs/Vector3 velocity  # base_link frame
float64[] bias                  # 6: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z (rad/s for gyr)
