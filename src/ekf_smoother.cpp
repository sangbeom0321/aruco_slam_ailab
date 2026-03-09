// EKF Smoother: SE(3) Extended Kalman Filter for smooth SLAM odometry
// - Prediction: IMU odom delta (~200Hz, odom frame)
// - Update: ISAM2 optimized pose (~10Hz when markers visible, map frame)
// - Output: smoothed odometry in map frame

#include "aruco_sam_ailab/utility.hpp"
#include "aruco_sam_ailab/msg/optimized_keyframe_state.hpp"

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <gtsam/geometry/Pose3.h>
#include <Eigen/Dense>
#include <mutex>
#include <cmath>

namespace aruco_sam_ailab {

class EkfSmoother : public ParamServer {
public:
    // 6-DOF state in SE(3) tangent space: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    static constexpr int N = 6;

    // EKF state (map frame)
    gtsam::Pose3 statePose_;
    Eigen::Matrix<double, N, N> P_;  // covariance in tangent space
    Eigen::Matrix<double, N, N> Q_;  // process noise (per second)
    Eigen::Matrix<double, N, N> R_;  // measurement noise

    bool initialized_ = false;

    // Previous IMU odom for delta computation
    gtsam::Pose3 lastImuOdomPose_;
    rclcpp::Time lastImuOdomTime_{0, 0, RCL_ROS_TIME};
    bool hasLastImuOdom_ = false;

    // Last published timestamp (prevent time regression)
    rclcpp::Time lastPublishedTime_{0, 0, RCL_ROS_TIME};

    // Jump detection: velocity-based + absolute distance-based
    static constexpr double MAX_SPEED = 5.0;      // m/s
    static constexpr double MAX_ANG_VEL = 3.0;    // rad/s
    static constexpr double MAX_DELTA_TRANS = 0.5; // m — absolute max per single step
    static constexpr double MAX_DELTA_ROT = 1.0;   // rad — absolute max per single step

    // Innovation gating: reject SLAM corrections that are too large (steady-state)
    static constexpr double MAX_INNOVATION_TRANS = 5.0;  // m
    static constexpr double MAX_INNOVATION_ROT = 2.0;    // rad
    // Convergence: accept first N corrections unconditionally
    // (ISAM2 warmup shifts pose significantly before EKF gets corrections)
    static constexpr int CONVERGENCE_UPDATES = 10;
    int updateCount_ = 0;

    // --- 속도(Twist): IMU preintegration에서 직접 수신 ---
    double current_v_ = 0.0;
    double current_yaw_rate_ = 0.0;

    // ROS interface
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subImuOdom_;
    rclcpp::Subscription<aruco_sam_ailab::msg::OptimizedKeyframeState>::SharedPtr subSlamCorrection_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdom_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;

    nav_msgs::msg::Path path_;
    std::mutex mtx_;

    // EKF tuning parameters
    double qPos_, qRot_, rPos_, rRot_;
    bool lock_z_output_ = true;

    // LPF state for smooth output (v, yaw_rate)
    double smoothed_v_ = 0.0;
    double smoothed_yaw_rate_ = 0.0;
    double lpf_alpha_v_ = 0.1;     // 속도는 반응성 중요
    double lpf_alpha_yaw_ = 0.1;   // 각속도는 부드러움 중요

    // Sliding Window for additional smoothing
    std::deque<double> v_window_;
    std::deque<double> yaw_window_;
    size_t v_window_size_ = 3;      // ~25ms delay at 200Hz
    size_t yaw_window_size_ = 20;   // ~75ms delay at 200Hz

    EkfSmoother(const rclcpp::NodeOptions& options) : ParamServer("ekf_smoother", options) {
        // EKF noise parameters
        declare_parameter("ekf_process_noise_pos", 0.1);
        declare_parameter("ekf_process_noise_rot", 0.05);
        declare_parameter("ekf_measurement_noise_pos", 0.05);
        declare_parameter("ekf_measurement_noise_rot", 0.05);
        declare_parameter("ekf_lock_z_output", true);

        get_parameter("ekf_process_noise_pos", qPos_);
        get_parameter("ekf_process_noise_rot", qRot_);
        get_parameter("ekf_measurement_noise_pos", rPos_);
        get_parameter("ekf_measurement_noise_rot", rRot_);
        get_parameter("ekf_lock_z_output", lock_z_output_);

        // Initialize covariance matrices
        P_ = Eigen::Matrix<double, N, N>::Identity() * 0.01;

        // Process noise (per second): GTSAM tangent space = [rot(3), trans(3)]
        Q_ = Eigen::Matrix<double, N, N>::Zero();
        Q_(0, 0) = Q_(1, 1) = Q_(2, 2) = qRot_ * qRot_;
        Q_(3, 3) = Q_(4, 4) = Q_(5, 5) = qPos_ * qPos_;

        // Measurement noise
        R_ = Eigen::Matrix<double, N, N>::Zero();
        R_(0, 0) = R_(1, 1) = R_(2, 2) = rRot_ * rRot_;
        R_(3, 3) = R_(4, 4) = R_(5, 5) = rPos_ * rPos_;

        // Subscribers
        subImuOdom_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odometry/imu_incremental", rclcpp::SensorDataQoS(),
            std::bind(&EkfSmoother::imuOdomCallback, this, std::placeholders::_1));

        subSlamCorrection_ = create_subscription<aruco_sam_ailab::msg::OptimizedKeyframeState>(
            "/optimized_keyframe_state", 10,
            std::bind(&EkfSmoother::slamCorrectionCallback, this, std::placeholders::_1));

        // Publishers
        pubOdom_ = create_publisher<nav_msgs::msg::Odometry>("/ekf/odom", 200);
        pubPath_ = create_publisher<nav_msgs::msg::Path>("/ekf/path", 10);
        tfBroadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        RCLCPP_INFO(get_logger(),
                    "EKF Smoother initialized (Q_pos=%.3f, Q_rot=%.3f, R_pos=%.3f, R_rot=%.3f, lock_z_output=%s)",
                    qPos_, qRot_, rPos_, rRot_, lock_z_output_ ? "true" : "false");
    }

    // ─── Prediction step: apply relative motion from IMU odom ───
    void predict(const gtsam::Pose3& delta, double dt) {
        statePose_ = statePose_.compose(delta);
        P_ += Q_ * dt;
    }

    // ─── Update step: correct state toward SLAM measurement ───
    // Returns false if correction was rejected (innovation too large)
    bool update(const gtsam::Pose3& measurement) {
        // Innovation in SE(3) tangent space: log(state⁻¹ · measurement)
        gtsam::Pose3 errorPose = statePose_.between(measurement);
        Eigen::Matrix<double, N, 1> innovation = gtsam::Pose3::Logmap(errorPose);

        // Innovation gating: reject corrections that are too large (outlier protection)
        // During convergence period, accept all corrections unconditionally
        // (ISAM2 warmup causes large initial discrepancy with EKF state)
        double transInnovation = innovation.tail<3>().norm();
        double rotInnovation = innovation.head<3>().norm();
        if (updateCount_ >= CONVERGENCE_UPDATES &&
            (transInnovation > MAX_INNOVATION_TRANS || rotInnovation > MAX_INNOVATION_ROT)) {
            RCLCPP_WARN(get_logger(),
                "[EKF] Large SLAM correction rejected: trans=%.3fm rot=%.3frad (max: %.1f/%.1f)",
                transInnovation, rotInnovation, MAX_INNOVATION_TRANS, MAX_INNOVATION_ROT);
            return false;
        }
        if (updateCount_ < CONVERGENCE_UPDATES) {
            RCLCPP_DEBUG(get_logger(),
                "[EKF] Convergence %d/%d: trans=%.3fm rot=%.3frad (accepted)",
                updateCount_ + 1, CONVERGENCE_UPDATES, transInnovation, rotInnovation);
        }

        // Kalman gain: K = P · (P + R)⁻¹
        Eigen::Matrix<double, N, N> S = P_ + R_;
        Eigen::Matrix<double, N, N> K = P_ * S.inverse();

        // Apply correction via exponential map
        Eigen::Matrix<double, N, 1> correction = K * innovation;
        statePose_ = statePose_.compose(gtsam::Pose3::Expmap(correction));

        // Joseph form for numerical stability: P = (I - K) P (I - K)^T + K R K^T
        Eigen::Matrix<double, N, N> IKH = Eigen::Matrix<double, N, N>::Identity() - K;
        P_ = IKH * P_ * IKH.transpose() + K * R_ * K.transpose();

        // Ensure symmetry
        P_ = (P_ + P_.transpose()) / 2.0;
        updateCount_++;
        return true;
    }

    // ─── IMU odom callback (~200Hz): EKF prediction ───
    void imuOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);

        gtsam::Pose3 currOdom = poseMsgToGtsam(msg->pose.pose);
        rclcpp::Time currTime = msg->header.stamp;

        if (!hasLastImuOdom_) {
            lastImuOdomPose_ = currOdom;
            lastImuOdomTime_ = currTime;
            hasLastImuOdom_ = true;
            return;
        }

        double dt = (currTime - lastImuOdomTime_).seconds();
        if (dt <= 0 || dt > 1.0) {
            lastImuOdomPose_ = currOdom;
            lastImuOdomTime_ = currTime;
            return;
        }

        if (!initialized_) {
            lastImuOdomPose_ = currOdom;
            lastImuOdomTime_ = currTime;
            return;
        }

        // Compute relative motion (body-frame, works across any reference frame)
        gtsam::Pose3 delta = lastImuOdomPose_.between(currOdom);

        // --- 속도: IMU preintegration이 이미 계산한 값을 직접 사용 + Sliding Window + LPF 적용 ---
        double raw_v = msg->twist.twist.linear.x;
        double raw_yaw_rate = msg->twist.twist.angular.z;

        // 1. Sliding Window (Moving Average)
        v_window_.push_back(raw_v);
        if (v_window_.size() > v_window_size_) v_window_.pop_front();
        double ma_v = std::accumulate(v_window_.begin(), v_window_.end(), 0.0) / v_window_.size();

        yaw_window_.push_back(raw_yaw_rate);
        if (yaw_window_.size() > yaw_window_size_) yaw_window_.pop_front();
        double ma_yaw = std::accumulate(yaw_window_.begin(), yaw_window_.end(), 0.0) / yaw_window_.size();

        // 2. Low-Pass Filter (MA 결과를 입력으로)
        smoothed_v_ = lpf_alpha_v_ * ma_v + (1.0 - lpf_alpha_v_) * smoothed_v_;
        smoothed_yaw_rate_ = lpf_alpha_yaw_ * ma_yaw + (1.0 - lpf_alpha_yaw_) * smoothed_yaw_rate_;

        current_v_ = smoothed_v_;
        current_yaw_rate_ = smoothed_yaw_rate_;

        // Jump detection: if delta implies physically impossible motion, skip it.
        // This happens when imu_preintegration resets its state after receiving
        // an optimized keyframe correction.
        double transNorm = delta.translation().norm();
        double rotAngle = std::abs(delta.rotation().axisAngle().second);

        // Jump detection: velocity-based OR absolute distance-based
        // Absolute check catches jumps even when dt is large (e.g., 50ms gap)
        bool velocityJump = dt > 0 && (transNorm / dt > MAX_SPEED || rotAngle / dt > MAX_ANG_VEL);
        bool absoluteJump = transNorm > MAX_DELTA_TRANS || rotAngle > MAX_DELTA_ROT;

        if (velocityJump || absoluteJump) {
            if (enableTopicDebugLog) {
                RCLCPP_WARN(get_logger(),
                    "[EKF] Jump detected: trans=%.3fm rot=%.3frad dt=%.4fs (vel=%s abs=%s)",
                    transNorm, rotAngle, dt,
                    velocityJump ? "YES" : "no", absoluteJump ? "YES" : "no");
            }
            lastImuOdomPose_ = currOdom;
            lastImuOdomTime_ = currTime;
            publishState(currTime);
            return;
        }

        // Normal prediction
        predict(delta, dt);
        publishState(currTime);

        lastImuOdomPose_ = currOdom;
        lastImuOdomTime_ = currTime;
    }

    // ─── SLAM correction callback (~10Hz): EKF update ───
    void slamCorrectionCallback(const aruco_sam_ailab::msg::OptimizedKeyframeState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);

        gtsam::Pose3 measurement = poseMsgToGtsam(msg->pose);

        if (!initialized_) {
            statePose_ = measurement;
            P_ = Eigen::Matrix<double, N, N>::Identity() * 0.01;
            initialized_ = true;
            RCLCPP_INFO(get_logger(), "EKF initialized at (%.2f, %.2f, %.2f)",
                        measurement.translation().x(),
                        measurement.translation().y(),
                        measurement.translation().z());
            publishState(msg->header.stamp);
            return;
        }

        // EKF update: partially correct toward optimized pose (Kalman gain based)
        update(measurement);

        if (enableTopicDebugLog) {
            // Log Kalman gain diagonal for tuning visibility
            Eigen::Matrix<double, N, N> S = P_ + R_;
            Eigen::Matrix<double, N, N> K = P_ * S.inverse();
            RCLCPP_INFO(get_logger(),
                        "[EKF] Update: K_trans=%.2f K_rot=%.2f P_diag=(%.4f,%.4f,%.4f,%.4f,%.4f,%.4f)",
                        K(3, 3), K(0, 0),
                        P_(0, 0), P_(1, 1), P_(2, 2), P_(3, 3), P_(4, 4), P_(5, 5));
        }

        publishState(msg->header.stamp);
    }

    // ─── Publish smoothed odometry, TF, and path ───
    void publishState(const rclcpp::Time& stamp) {
        // Prevent time regression: SLAM corrections may arrive with older timestamps
        // than the last IMU odom publish. Use max(stamp, lastPublished) to keep monotonic.
        rclcpp::Time pubStamp = stamp;
        if (lastPublishedTime_.nanoseconds() > 0 && stamp < lastPublishedTime_) {
            pubStamp = lastPublishedTime_;
        }
        lastPublishedTime_ = pubStamp;

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = pubStamp;
        odom.header.frame_id = odomFrame;
        odom.child_frame_id = baseLinkFrame;
        odom.pose.pose = gtsamToPoseMsg(statePose_);
        if (lock_z_output_) {
            odom.pose.pose.position.z = 0.0;
        }

        // 속도: IMU preintegration에서 수신한 값을 그대로 전달
        odom.twist.twist.linear.x = current_v_;
        odom.twist.twist.linear.y = 0.0;
        odom.twist.twist.linear.z = 0.0;
        odom.twist.twist.angular.x = 0.0;
        odom.twist.twist.angular.y = 0.0;
        odom.twist.twist.angular.z = current_yaw_rate_;
        for (int i = 0; i < 36; i++) odom.twist.covariance[i] = 0.0;
        odom.twist.covariance[0]  = 0.01;  // v_x 분산
        odom.twist.covariance[35] = 0.01;  // yaw_rate 분산

        // Fill covariance from P_ (GTSAM order: rot,trans → ROS order: trans,rot)
        for (int i = 0; i < 36; i++) odom.pose.covariance[i] = 0.0;
        odom.pose.covariance[0]  = P_(3, 3);  // x
        odom.pose.covariance[7]  = P_(4, 4);  // y
        odom.pose.covariance[14] = P_(5, 5);  // z
        odom.pose.covariance[21] = P_(0, 0);  // roll
        odom.pose.covariance[28] = P_(1, 1);  // pitch
        odom.pose.covariance[35] = P_(2, 2);  // yaw

        pubOdom_->publish(odom);

        // TF: odom → base_footprint
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = pubStamp;
        t.header.frame_id = odomFrame;
        t.child_frame_id = baseLinkFrame;
        t.transform.translation.x = odom.pose.pose.position.x;
        t.transform.translation.y = odom.pose.pose.position.y;
        t.transform.translation.z = lock_z_output_ ? 0.0 : odom.pose.pose.position.z;
        t.transform.rotation = odom.pose.pose.orientation;
        tfBroadcaster_->sendTransform(t);

        // Path
        geometry_msgs::msg::PoseStamped ps;
        ps.header = odom.header;
        ps.pose = odom.pose.pose;
        path_.header = odom.header;
        path_.poses.push_back(ps);
        if (path_.poses.size() > 10000) {
            path_.poses.erase(path_.poses.begin());
        }
        pubPath_->publish(path_);
    }
};

} // namespace aruco_sam_ailab

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);

    auto node = std::make_shared<aruco_sam_ailab::EkfSmoother>(options);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> EKF Smoother Started.\033[0m");

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
