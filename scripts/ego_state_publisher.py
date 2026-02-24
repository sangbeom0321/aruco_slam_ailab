#!/usr/bin/env python3
"""
EgoState Publisher (v7)
========================
/ekf/odom (EKF Smoother, IMU+SLAM 융합, map frame) → /ego_state
EKF smoother 출력이 map frame pose + velocity를 모두 포함하므로 TF lookup 불필요.
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from hunter_msgs2.msg import EgoState
import tf_transformations


def quaternion_to_yaw(q) -> float:
    explicit_quat = [q.x, q.y, q.z, q.w]
    _, _, yaw = tf_transformations.euler_from_quaternion(explicit_quat)
    return yaw


class EgoStatePublisher(Node):
    def __init__(self):
        super().__init__('ego_state_publisher')

        self.sub_ekf = self.create_subscription(
            Odometry, '/ekf/odom', self._ekf_callback, 10)

        self.pub_ego = self.create_publisher(EgoState, '/ego_state', 10)

        self._prev_v = 0.0
        self._prev_stamp = None

        self.declare_parameter('ego_state_accel_clamp', 3.0)
        self._accel_clamp = self.get_parameter('ego_state_accel_clamp').value

        self.get_logger().info('EgoStatePublisher initialized (v7: EKF only)')

    def _ekf_callback(self, msg: Odometry):
        stamp = Time.from_msg(msg.header.stamp)

        if self._prev_stamp is not None and stamp.nanoseconds < self._prev_stamp.nanoseconds:
            self.get_logger().warn('Time jump detected in EgoState! Resetting.')
            self._prev_v = 0.0
            self._prev_stamp = stamp
            return

        ego = EgoState()
        ego.header.stamp = msg.header.stamp
        ego.header.frame_id = 'map'

        ego.x = msg.pose.pose.position.x
        ego.y = msg.pose.pose.position.y
        ego.yaw = quaternion_to_yaw(msg.pose.pose.orientation)

        ego.v = msg.twist.twist.linear.x
        ego.yaw_rate = msg.twist.twist.angular.z

        if self._prev_stamp is not None:
            dt = float((stamp - self._prev_stamp).nanoseconds) * 1e-9
            if dt > 0.005:
                a_raw = float((ego.v - self._prev_v) / dt)
                ego.a = float(max(float(-self._accel_clamp), min(float(self._accel_clamp), a_raw)))
            else:
                ego.a = 0.0
        else:
            ego.a = 0.0

        self._prev_v = ego.v
        self._prev_stamp = stamp

        self.pub_ego.publish(ego)


def main(args=None):
    rclpy.init(args=args)
    node = EgoStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
