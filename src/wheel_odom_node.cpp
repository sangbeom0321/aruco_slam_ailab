/**
 * Wheel Odometry Node (Re-propagation)
 * ====================================
 * Hunter2 Ackermann: rear_left/right_joint -> v_linear, omega
 *
 * Re-propagation 방식:
 * - 버퍼링: joint_states 속도/시간을 history_buffer_에 저장
 * - 보정 수신: Anchor 업데이트 + corr_time 이전 버퍼 삭제 + 남은 버퍼로 재적분
 * - 결과: 위치 점프 없이 부드러운 보정
 */

#include <cmath>
#include <memory>
#include <string>
#include <deque>
#include <mutex>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>

// 속도와 시간을 저장할 구조체
struct VelocityMeasurement {
  rclcpp::Time stamp;
  double v_linear;
  double omega;
  double dt;  // 이전 프레임과의 시간 차
};

class WheelOdomNode : public rclcpp::Node {
public:
  WheelOdomNode() : Node("wheel_odom_node") {
    // --- Parameters ---
    declare_parameter("wheel_radius", 0.165);
    declare_parameter("track_width", 0.605);
    declare_parameter("base_frame", "base_link");
    declare_parameter("odom_frame", "odom");
    declare_parameter("publish_tf", false);
    declare_parameter("wheel_odom_topic", "wheel_odom");
    declare_parameter("sync_with_slam_topic", "/aruco_slam/odom");
    declare_parameter("wheel_odom_correction_topic", "/odometry/wheel_odom_correction");
    declare_parameter("buffer_duration", 5.0);  // 버퍼 보관 시간 (초)

    wheel_radius_ = get_parameter("wheel_radius").as_double();
    track_width_ = get_parameter("track_width").as_double();
    base_frame_ = get_parameter("base_frame").as_string();
    odom_frame_ = get_parameter("odom_frame").as_string();
    publish_tf_ = get_parameter("publish_tf").as_bool();
    buffer_duration_ = get_parameter("buffer_duration").as_double();

    std::string odom_topic = get_parameter("wheel_odom_topic").as_string();
    std::string correction_topic = get_parameter("wheel_odom_correction_topic").as_string();
    std::string sync_topic = get_parameter("sync_with_slam_topic").as_string();

    // --- ROS Interface ---
    sub_joint_ = create_subscription<sensor_msgs::msg::JointState>(
        "joint_states", 100, std::bind(&WheelOdomNode::jointStatesCallback, this, std::placeholders::_1));

    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic, 10);

    // SLAM 보정 (Re-propagation Trigger)
    if (!correction_topic.empty()) {
      sub_correction_ = create_subscription<nav_msgs::msg::Odometry>(
          correction_topic, 10, std::bind(&WheelOdomNode::correctionCallback, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(), "Will correct from %s (Re-propagation)", correction_topic.c_str());
    }

    // 초기화: SLAM sync는 선택적 (correction이 먼저 올 수 있음)
    // [중요] graph_optimizer가 wheel odom을 기다리므로, 반드시 바로 발행해야 함 (데드락 방지)
    initialized_ = true;  // 0,0,0에서 시작하여 즉시 발행
    if (!sync_topic.empty()) {
      sub_slam_init_ = create_subscription<nav_msgs::msg::Odometry>(
          sync_topic, 10, std::bind(&WheelOdomNode::slamInitCallback, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(), "Optional sync with %s (correction이 초기화에 사용됨)", sync_topic.c_str());
    }

    if (publish_tf_) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    RCLCPP_INFO(get_logger(), "Wheel Odom Node Started with Re-propagation Buffer (%.1fs).", buffer_duration_);
  }

private:
  // =================================================================================
  // 1. Joint State Callback: 실시간 적분 및 버퍼링
  // =================================================================================
  void jointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    rclcpp::Time now = msg->header.stamp;
    if (now.seconds() == 0.0) now = get_clock()->now();

    // 조인트 속도 추출
    double v_l = 0.0, v_r = 0.0;
    bool l_found = false, r_found = false;
    for (size_t i = 0; i < msg->name.size(); ++i) {
      if (i < msg->velocity.size()) {
        if (msg->name[i] == "rear_left_joint") {
          v_l = msg->velocity[i];
          l_found = true;
        } else if (msg->name[i] == "rear_right_joint") {
          v_r = msg->velocity[i];
          r_found = true;
        }
      }
    }
    if (!l_found || !r_found) return;

    std::lock_guard<std::mutex> lock(mtx_);

    // dt 계산
    if (!last_time_initialized_) {
      last_time_ = now;
      last_time_initialized_ = true;
      return;
    }
    double dt = (now - last_time_).seconds();
    last_time_ = now;
    if (dt <= 0.0 || dt > 1.0) return;

    // 운동학 계산 (Differential / Ackermann approximate)
    double v_linear = (v_l + v_r) * wheel_radius_ / 2.0;
    double omega = (v_r - v_l) * wheel_radius_ / track_width_;

    // 1) 버퍼에 저장 (나중에 재적분을 위해)
    VelocityMeasurement meas;
    meas.stamp = now;
    meas.v_linear = v_linear;
    meas.omega = omega;
    meas.dt = dt;
    history_buffer_.push_back(meas);

    // 버퍼 사이즈 관리 (너무 오래된 데이터 삭제)
    while (!history_buffer_.empty()) {
      if ((now - history_buffer_.front().stamp).seconds() > buffer_duration_) {
        history_buffer_.pop_front();
      } else {
        break;
      }
    }

    // 2) 현재 상태 업데이트 (Dead Reckoning)
    updatePoseStep(current_pose_x_, current_pose_y_, current_pose_yaw_, v_linear, omega, dt);

    // 3) Publish
    if (initialized_) {
      publishOdometry(now, v_linear, omega);
    }
  }

  // =================================================================================
  // 2. Correction Callback: 핵심 Re-propagation 로직
  // =================================================================================
  void correctionCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mtx_);

    // 1) Anchor(기준점)를 최적화된 위치로 업데이트
    rclcpp::Time corr_time = msg->header.stamp;
    double anchor_x = msg->pose.pose.position.x;
    double anchor_y = msg->pose.pose.position.y;
    double anchor_yaw = getYaw(msg->pose.pose.orientation);

    // 2) 버퍼 정리: 보정된 시간(corr_time)보다 이전 데이터는 모두 삭제
    while (!history_buffer_.empty()) {
      if (history_buffer_.front().stamp <= corr_time) {
        history_buffer_.pop_front();
      } else {
        break;
      }
    }

    // 3) Re-propagation (재적분)
    current_pose_x_ = anchor_x;
    current_pose_y_ = anchor_y;
    current_pose_yaw_ = anchor_yaw;

    for (const auto& meas : history_buffer_) {
      updatePoseStep(current_pose_x_, current_pose_y_, current_pose_yaw_,
                     meas.v_linear, meas.omega, meas.dt);
    }

    if (!initialized_) initialized_ = true;
  }

  void slamInitCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (initialized_) return;

    current_pose_x_ = msg->pose.pose.position.x;
    current_pose_y_ = msg->pose.pose.position.y;
    current_pose_yaw_ = getYaw(msg->pose.pose.orientation);
    initialized_ = true;
    RCLCPP_INFO(get_logger(), "Initialized Pose by SLAM: x=%.3f, y=%.3f", current_pose_x_, current_pose_y_);
  }

  // =================================================================================
  // Helper Functions
  // =================================================================================

  void updatePoseStep(double& x, double& y, double& yaw, double v, double w, double dt) {
    x += v * std::cos(yaw) * dt;
    y += v * std::sin(yaw) * dt;
    yaw += w * dt;
    yaw = std::atan2(std::sin(yaw), std::cos(yaw));
  }

  double getYaw(const geometry_msgs::msg::Quaternion& q_msg) {
    tf2::Quaternion q(q_msg.x, q_msg.y, q_msg.z, q_msg.w);
    tf2::Matrix3x3 m(q);
    double r, p, y;
    m.getRPY(r, p, y);
    return y;
  }

  void publishOdometry(const rclcpp::Time& now, double v_linear, double omega) {
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = now;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id = base_frame_;

    odom.pose.pose.position.x = current_pose_x_;
    odom.pose.pose.position.y = current_pose_y_;
    odom.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0, 0, current_pose_yaw_);
    odom.pose.pose.orientation = tf2::toMsg(q);

    odom.twist.twist.linear.x = v_linear;
    odom.twist.twist.angular.z = omega;

    odom.pose.covariance[0] = 0.01;
    odom.pose.covariance[7] = 0.01;
    odom.pose.covariance[35] = 0.01;

    pub_odom_->publish(odom);

    if (publish_tf_) {
      geometry_msgs::msg::TransformStamped t;
      t.header.stamp = now;
      t.header.frame_id = odom_frame_;
      t.child_frame_id = base_frame_;
      t.transform.translation.x = current_pose_x_;
      t.transform.translation.y = current_pose_y_;
      t.transform.translation.z = 0.0;
      t.transform.rotation = tf2::toMsg(q);
      tf_broadcaster_->sendTransform(t);
    }
  }

  // --- Variables ---
  double wheel_radius_;
  double track_width_;
  std::string base_frame_;
  std::string odom_frame_;
  bool publish_tf_;
  double buffer_duration_;

  std::mutex mtx_;
  bool initialized_ = false;
  bool last_time_initialized_ = false;
  rclcpp::Time last_time_;

  double current_pose_x_ = 0.0;
  double current_pose_y_ = 0.0;
  double current_pose_yaw_ = 0.0;

  std::deque<VelocityMeasurement> history_buffer_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_correction_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_slam_init_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WheelOdomNode>());
  rclcpp::shutdown();
  return 0;
}
