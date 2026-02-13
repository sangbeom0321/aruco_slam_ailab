#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <deque>
#include <mutex>
#include <string>

namespace aruco_slam_ailab {

// Utility functions
template<typename T>
double stamp2Sec(const T& stamp) {
    return rclcpp::Time(stamp).seconds();
}

// Convert geometry_msgs::Pose to GTSAM Pose3
gtsam::Pose3 poseMsgToGtsam(const geometry_msgs::msg::Pose& pose_msg) {
    gtsam::Point3 translation(pose_msg.position.x, pose_msg.position.y, pose_msg.position.z);
    gtsam::Rot3 rotation = gtsam::Rot3::Quaternion(
        pose_msg.orientation.w,
        pose_msg.orientation.x,
        pose_msg.orientation.y,
        pose_msg.orientation.z
    );
    return gtsam::Pose3(rotation, translation);
}

// Convert GTSAM Pose3 to geometry_msgs::Pose
geometry_msgs::msg::Pose gtsamToPoseMsg(const gtsam::Pose3& pose) {
    geometry_msgs::msg::Pose pose_msg;
    pose_msg.position.x = pose.translation().x();
    pose_msg.position.y = pose.translation().y();
    pose_msg.position.z = pose.translation().z();
    auto q = pose.rotation().toQuaternion();
    pose_msg.orientation.w = q.w();
    pose_msg.orientation.x = q.x();
    pose_msg.orientation.y = q.y();
    pose_msg.orientation.z = q.z();
    return pose_msg;
}

// Parameter server base class
class ParamServer : public rclcpp::Node {
public:
    // Topics
    std::string imuTopic;
    std::string wheelOdomTopic;
    std::string odomTopic;  // /odom for vision-only OdomBetweenFactor
    std::string arucoPosesTopic;
    
    // Frames
    std::string baseLinkFrame;
    std::string imuFrame;
    std::string cameraFrame;
    std::string odomFrame;
    std::string mapFrame;
    
    // IMU parameters
    double imuAccNoise;
    double imuGyrNoise;
    double imuAccBiasN;
    double imuGyrBiasN;
    double imuGravity;
    
    // Extrinsic calibrations
    Eigen::Vector3d extTransBaseImu;
    Eigen::Matrix3d extRotBaseImu;
    Eigen::Vector3d extTransBaseCam;
    Eigen::Matrix3d extRotBaseCam;
    Eigen::Vector3d extTransBaseDepthCam;
    Eigen::Matrix3d extRotBaseDepthCam;

    // Wheel odometry parameters
    double wheelOdomTransNoise;
    double wheelOdomRotNoise;
    bool useWheelOdom;

    // Odom BetweenFactor (vision-only): /odom topic noise
    double odomBetweenTransNoise;
    double odomBetweenRotNoise;
    
    // IMU usage flag
    bool useImu;
    
    // ArUco observation parameters
    double arucoTransNoise;
    double arucoRotNoise;
    
    // Keyframe policy
    double keyframeTimeInterval;  // seconds
    double keyframeDistanceThreshold;  // meters
    double keyframeAngleThreshold;  // radians

    // ISAM2 parameters
    double isamRelinearizeThreshold;
    int isamRelinearizeSkip;

    // Debug: topic receive logging (throttled/first-received)
    bool enableTopicDebugLog;
    
    ParamServer(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node(node_name, options) {
        
        // Topics
        declare_parameter("imu_topic", "/imu");
        declare_parameter("wheel_odom_topic", "/wheel_odom");
        declare_parameter("odom_topic", "/w_odom");
        declare_parameter("aruco_poses_topic", "/aruco_poses");
        
        get_parameter("imu_topic", imuTopic);
        get_parameter("wheel_odom_topic", wheelOdomTopic);
        get_parameter("odom_topic", odomTopic);
        get_parameter("aruco_poses_topic", arucoPosesTopic);
        
        // Frames
        declare_parameter("base_link_frame", "base_link");
        declare_parameter("imu_frame", "imu_link");
        declare_parameter("camera_frame", "camera_link");
        declare_parameter("odom_frame", "odom");
        declare_parameter("map_frame", "map");
        
        get_parameter("base_link_frame", baseLinkFrame);
        get_parameter("imu_frame", imuFrame);
        get_parameter("camera_frame", cameraFrame);
        get_parameter("odom_frame", odomFrame);
        get_parameter("map_frame", mapFrame);
        
        // IMU noise parameters
        declare_parameter("imu_acc_noise", 1e-2);
        declare_parameter("imu_gyr_noise", 1e-3);
        declare_parameter("imu_acc_bias_n", 1e-4);
        declare_parameter("imu_gyr_bias_n", 1e-5);
        declare_parameter("imu_gravity", 9.81);
        
        get_parameter("imu_acc_noise", imuAccNoise);
        get_parameter("imu_gyr_noise", imuGyrNoise);
        get_parameter("imu_acc_bias_n", imuAccBiasN);
        get_parameter("imu_gyr_bias_n", imuGyrBiasN);
        get_parameter("imu_gravity", imuGravity);
        
        // Extrinsic calibrations
        std::vector<double> ext_trans_base_imu_v = {0.0, 0.0, 0.0};
        std::vector<double> ext_rot_base_imu_v = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        std::vector<double> ext_trans_base_cam_v = {0.0, 0.0, 0.0};
        std::vector<double> ext_rot_base_cam_v = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        
        declare_parameter("ext_trans_base_imu", ext_trans_base_imu_v);
        declare_parameter("ext_rot_base_imu", ext_rot_base_imu_v);
        declare_parameter("ext_trans_base_cam", ext_trans_base_cam_v);
        declare_parameter("ext_rot_base_cam", ext_rot_base_cam_v);
        std::vector<double> ext_trans_base_depth_cam_v = {0.411, 0.011, 0.037};
        std::vector<double> ext_rot_base_depth_cam_v = {0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0};
        declare_parameter("ext_trans_base_depth_cam", ext_trans_base_depth_cam_v);
        declare_parameter("ext_rot_base_depth_cam", ext_rot_base_depth_cam_v);

        get_parameter("ext_trans_base_imu", ext_trans_base_imu_v);
        get_parameter("ext_rot_base_imu", ext_rot_base_imu_v);
        get_parameter("ext_trans_base_cam", ext_trans_base_cam_v);
        get_parameter("ext_rot_base_cam", ext_rot_base_cam_v);
        get_parameter("ext_trans_base_depth_cam", ext_trans_base_depth_cam_v);
        get_parameter("ext_rot_base_depth_cam", ext_rot_base_depth_cam_v);

        extTransBaseImu = Eigen::Map<const Eigen::Vector3d>(ext_trans_base_imu_v.data());
        extRotBaseImu = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(ext_rot_base_imu_v.data(), 3, 3);
        extTransBaseCam = Eigen::Map<const Eigen::Vector3d>(ext_trans_base_cam_v.data());
        extRotBaseCam = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(ext_rot_base_cam_v.data(), 3, 3);
        extTransBaseDepthCam = Eigen::Map<const Eigen::Vector3d>(ext_trans_base_depth_cam_v.data());
        extRotBaseDepthCam = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(ext_rot_base_depth_cam_v.data(), 3, 3);
        
        // IMU usage flag
        declare_parameter("use_imu", true);
        get_parameter("use_imu", useImu);
        
        // Wheel odometry parameters
        declare_parameter("use_wheel_odom", false);
        declare_parameter("wheel_odom_trans_noise", 0.1);
        declare_parameter("wheel_odom_rot_noise", 0.1);
        
        get_parameter("use_wheel_odom", useWheelOdom);
        get_parameter("wheel_odom_trans_noise", wheelOdomTransNoise);
        get_parameter("wheel_odom_rot_noise", wheelOdomRotNoise);

        // Odom BetweenFactor (vision-only)
        declare_parameter("odom_between_trans_noise", 0.15);
        declare_parameter("odom_between_rot_noise", 0.15);
        get_parameter("odom_between_trans_noise", odomBetweenTransNoise);
        get_parameter("odom_between_rot_noise", odomBetweenRotNoise);
        
        // ArUco observation parameters
        declare_parameter("aruco_trans_noise", 0.05);
        declare_parameter("aruco_rot_noise", 0.1);
        
        get_parameter("aruco_trans_noise", arucoTransNoise);
        get_parameter("aruco_rot_noise", arucoRotNoise);
        
        // Keyframe policy
        declare_parameter("keyframe_time_interval", 2.0);
        declare_parameter("keyframe_distance_threshold", 0.5);
        declare_parameter("keyframe_angle_threshold", 0.0524);  // 3 deg

        get_parameter("keyframe_time_interval", keyframeTimeInterval);
        get_parameter("keyframe_distance_threshold", keyframeDistanceThreshold);
        get_parameter("keyframe_angle_threshold", keyframeAngleThreshold);

        // ISAM2 parameters
        declare_parameter("isam_relinearize_threshold", 0.1);
        declare_parameter("isam_relinearize_skip", 1);
        
        get_parameter("isam_relinearize_threshold", isamRelinearizeThreshold);
        get_parameter("isam_relinearize_skip", isamRelinearizeSkip);

        declare_parameter("enable_topic_debug_log", rclcpp::ParameterValue(false));
        rclcpp::Parameter p;
        if (get_parameter("enable_topic_debug_log", p)) {
            if (p.get_type() == rclcpp::ParameterType::PARAMETER_BOOL)
                enableTopicDebugLog = p.as_bool();
            else if (p.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
                enableTopicDebugLog = (p.as_string() == "true" || p.as_string() == "1");
        }
    }
};

} // namespace aruco_slam_ailab
