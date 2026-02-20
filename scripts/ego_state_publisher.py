#!/usr/bin/env python3
"""
EgoState Publisher (v4)
========================
SLAM 파이프라인의 최종 출력 노드.
/w_odom (SLAM wheel odom) → /ego_state (차량 상태: 위치, 속도, 각속도)
                            → /ego_state_pose_viz (RViz 시각화용 PoseStamped)

Gazebo GT /odom 의존성 완전 제거. TF lookup 불필요 (/w_odom이 pose+velocity 모두 포함).
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Quaternion
from hunter_msgs2.msg import EgoState
import tf_transformations


def quaternion_to_yaw(q) -> float:
    """쿼터니언 → yaw 각도 변환 (radians)."""
    explicit_quat = [q.x, q.y, q.z, q.w]
    _, _, yaw = tf_transformations.euler_from_quaternion(explicit_quat)
    return yaw


def yaw_to_quaternion_msg(yaw) -> Quaternion:
    """yaw → geometry_msgs/Quaternion 변환."""
    q_array = tf_transformations.quaternion_from_euler(0, 0, yaw)
    msg = Quaternion()
    msg.x = q_array[0]
    msg.y = q_array[1]
    msg.z = q_array[2]
    msg.w = q_array[3]
    return msg


class EgoStatePublisher(Node):
    def __init__(self):
        super().__init__('ego_state_publisher')

        # /w_odom 구독 (wheel_odom_node 발행, odom frame, pose + velocity)
        self.sub_odom = self.create_subscription(
            Odometry, '/w_odom', self._wodom_callback, 10)

        # /ego_state 발행 (planner + controller 구독)
        self.pub_ego = self.create_publisher(EgoState, '/ego_state', 10)

        # /ego_state_pose_viz 발행 (RViz PoseStamped 시각화용)
        self.pub_ego_viz = self.create_publisher(PoseStamped, '/ego_state_pose_viz', 10)

        self._prev_v = 0.0
        self._prev_stamp = None
        # slam_params.yaml에 정의된 파라미터 로드
        self.declare_parameter('ego_state_accel_clamp', 3.0)
        self._accel_clamp = self.get_parameter('ego_state_accel_clamp').value

        self.get_logger().info('EgoStatePublisher initialized (v5: Acceleration enabled)')

    def _wodom_callback(self, msg: Odometry):
        """/w_odom에서 pose+velocity 추출 → EgoState + PoseStamped 발행."""
        stamp = Time.from_msg(msg.header.stamp)
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)

        # EgoState 발행 — SLAM 최종 출력 (차량 전체 상태)
        ego = EgoState()
        ego.header.stamp = msg.header.stamp
        ego.header.frame_id = 'odom'
        ego.x = msg.pose.pose.position.x
        ego.y = msg.pose.pose.position.y
        ego.yaw = yaw
        ego.v = msg.twist.twist.linear.x
        ego.yaw_rate = msg.twist.twist.angular.z

        # 가속도 계산: Finite Difference
        if self._prev_stamp is not None:
            dt = (stamp - self._prev_stamp).nanoseconds * 1e-9
            if dt > 0.02:  # 50Hz 이상이면 계산 (너무 작은 dt는 노이즈 유발)
                a_raw = (ego.v - self._prev_v) / dt
                ego.a = float(max(-self._accel_clamp, min(self._accel_clamp, a_raw)))
            else:
                # dt가 너무 작으면 이전 가속도 유지 또는 계산 건너뜀
                # 여기서는 간단히 0.0 (또는 이전 값 유지 로직 추가 가능)
                ego.a = 0.0
        else:
            ego.a = 0.0

        self._prev_v = ego.v
        self._prev_stamp = stamp

        self.pub_ego.publish(ego)

        # PoseStamped viz 발행 (RViz용)
        pose_viz = PoseStamped()
        pose_viz.header = ego.header
        pose_viz.pose.position.x = ego.x
        pose_viz.pose.position.y = ego.y
        pose_viz.pose.orientation = yaw_to_quaternion_msg(yaw)
        self.pub_ego_viz.publish(pose_viz)


def main(args=None):
    rclpy.init(args=args)
    node = EgoStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
