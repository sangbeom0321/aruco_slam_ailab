#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include "aruco_slam_ailab/msg/marker_array.hpp"
#include "aruco_slam_ailab/msg/marker_observation.hpp"
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <memory>
#include <vector>
#include <map>
#include <cmath>

class ArucoDetectorNode : public rclcpp::Node {
public:
    ArucoDetectorNode() : Node("aruco_detector_node") {
        // Parameters
        declare_parameter("camera_topic", "/kinect_camera/kinect_camera/image_raw");
        declare_parameter("camera_info_topic", "/kinect_camera/kinect_camera/camera_info");
        declare_parameter("marker_size", 0.30);
        declare_parameter("aruco_dict_type", "DICT_4X4_50");

        std::string camera_topic = get_parameter("camera_topic").as_string();
        std::string camera_info_topic = get_parameter("camera_info_topic").as_string();
        marker_size_ = get_parameter("marker_size").as_double();
        std::string aruco_dict_type = get_parameter("aruco_dict_type").as_string();

        // Initialize ArUco dictionary
        initArucoDictionary(aruco_dict_type);

        // Target marker IDs: 0 to 9
        for (int i = 0; i < 10; ++i) {
            target_marker_ids_.insert(i);
        }

        // Subscribers
        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            camera_topic, 10,
            std::bind(&ArucoDetectorNode::imageCallback, this, std::placeholders::_1));

        camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic, 10,
            std::bind(&ArucoDetectorNode::cameraInfoCallback, this, std::placeholders::_1));

        // Publishers
        // Publish custom MarkerArray directly for SLAM backend
        aruco_poses_pub_ = create_publisher<aruco_slam_ailab::msg::MarkerArray>(
            "/aruco_poses", 10);
        // Publish visualization markers for RViz
        visualization_markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            "/aruco_markers", 10);
        debug_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
            "/aruco_debug_image", 10);

        RCLCPP_INFO(get_logger(), "ArUco Detector Node started");
        RCLCPP_INFO(get_logger(), "  Camera topic: %s", camera_topic.c_str());
        RCLCPP_INFO(get_logger(), "  Marker size: %.2f m", marker_size_);
        RCLCPP_INFO(get_logger(), "  ArUco dictionary: %s", aruco_dict_type.c_str());
    }

private:
    void initArucoDictionary(const std::string& dict_type) {
        std::map<std::string, cv::aruco::PREDEFINED_DICTIONARY_NAME> dict_map = {
            {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
            {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
            {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
            {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
            {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
            {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
            {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
            {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
            {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
            {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
            {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
            {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
        };

        auto it = dict_map.find(dict_type);
        if (it != dict_map.end()) {
            aruco_dict_ = cv::aruco::getPredefinedDictionary(it->second);
        } else {
            RCLCPP_WARN(get_logger(), "Unknown dictionary type: %s, using DICT_4X4_50", dict_type.c_str());
            aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        }

        aruco_params_ = cv::makePtr<cv::aruco::DetectorParameters>();
    }

    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        if (!camera_matrix_initialized_) {
            camera_matrix_ = cv::Mat(3, 3, CV_64F);
            dist_coeffs_ = cv::Mat(msg->d.size(), 1, CV_64F);

            // Copy camera matrix
            for (int i = 0; i < 9; ++i) {
                camera_matrix_.at<double>(i / 3, i % 3) = msg->k[i];
            }

            // Copy distortion coefficients
            for (size_t i = 0; i < msg->d.size(); ++i) {
                dist_coeffs_.at<double>(i) = msg->d[i];
            }

            camera_frame_ = msg->header.frame_id;
            camera_matrix_initialized_ = true;

            RCLCPP_INFO(get_logger(), "Camera intrinsics received. Frame: %s", camera_frame_.c_str());
        }
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (!camera_matrix_initialized_) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for camera info...");
            return;
        }

        try {
            // Convert ROS Image to OpenCV
            cv_bridge::CvImagePtr cv_ptr;
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat cv_image = cv_ptr->image;
            cv::Mat gray;
            cv::cvtColor(cv_image, gray, cv::COLOR_BGR2GRAY);

            // Detect ArUco markers
            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f>> corners;
            std::vector<std::vector<cv::Point2f>> rejected;

            cv::aruco::detectMarkers(gray, aruco_dict_, corners, ids, aruco_params_, rejected);

            // Debug image
            cv::Mat debug_image = cv_image.clone();

            if (!ids.empty()) {
                // Draw detected markers
                cv::aruco::drawDetectedMarkers(debug_image, corners, ids);

                // Estimate poses for all detected markers (no filtering)
                std::vector<cv::Vec3d> rvecs, tvecs;
                cv::aruco::estimatePoseSingleMarkers(
                    corners, marker_size_,
                    camera_matrix_, dist_coeffs_,
                    rvecs, tvecs);

                // Create custom MarkerArray for SLAM backend
                aruco_slam_ailab::msg::MarkerArray marker_array;
                marker_array.header.stamp = msg->header.stamp;
                marker_array.header.frame_id = camera_frame_;

                for (size_t idx = 0; idx < ids.size(); ++idx) {
                    cv::Vec3d rvec = rvecs[idx];
                    cv::Vec3d tvec = tvecs[idx];
                    int marker_id = ids[idx];

                    // Draw axis on debug image
                    cv::drawFrameAxes(debug_image, camera_matrix_, dist_coeffs_,
                                     rvec, tvec, marker_size_ * 0.5);

                    // Create MarkerObservation
                    aruco_slam_ailab::msg::MarkerObservation observation;
                    observation.id = marker_id;
                    
                    // Set position
                    observation.pose.position.x = tvec[0];
                    observation.pose.position.y = tvec[1];
                    observation.pose.position.z = tvec[2];

                    // Convert rotation vector to quaternion
                    cv::Mat rotation_matrix;
                    cv::Rodrigues(rvec, rotation_matrix);
                    Quaternion quat = rotationMatrixToQuaternion(rotation_matrix);

                    observation.pose.orientation.w = quat.w;
                    observation.pose.orientation.x = quat.x;
                    observation.pose.orientation.y = quat.y;
                    observation.pose.orientation.z = quat.z;

                    marker_array.markers.push_back(observation);
                }

                // Publish custom MarkerArray directly for SLAM backend
                aruco_poses_pub_->publish(marker_array);
                
                // Publish visualization markers for RViz
                visualization_msgs::msg::MarkerArray vis_marker_array;
                vis_marker_array.markers.resize(ids.size());
                
                for (size_t idx = 0; idx < ids.size(); ++idx) {
                    visualization_msgs::msg::Marker& vis_marker = vis_marker_array.markers[idx];
                    vis_marker.header = marker_array.header;
                    vis_marker.ns = "aruco_markers";
                    vis_marker.id = ids[idx];
                    vis_marker.type = visualization_msgs::msg::Marker::CUBE;
                    vis_marker.action = visualization_msgs::msg::Marker::ADD;
                    
                    // Use the same pose as the observation
                    vis_marker.pose = marker_array.markers[idx].pose;
                    
                    // Set marker size (cube)
                    vis_marker.scale.x = marker_size_;
                    vis_marker.scale.y = marker_size_;
                    vis_marker.scale.z = 0.01;  // Thin cube
                    
                    // Set color based on marker ID
                    vis_marker.color.a = 0.7;
                    vis_marker.color.r = ((ids[idx] * 37) % 255) / 255.0;
                    vis_marker.color.g = ((ids[idx] * 73) % 255) / 255.0;
                    vis_marker.color.b = ((ids[idx] * 113) % 255) / 255.0;
                    
                    // Set lifetime (0 = infinite)
                    vis_marker.lifetime.sec = 0;
                    vis_marker.lifetime.nanosec = 0;
                    
                    // Add text with marker ID
                    vis_marker.text = std::to_string(ids[idx]);
                }
                
                visualization_markers_pub_->publish(vis_marker_array);
            } else {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "No ArUco markers detected in image");
            }

            // Publish debug image
            sensor_msgs::msg::Image::SharedPtr debug_msg =
                cv_bridge::CvImage(msg->header, "bgr8", debug_image).toImageMsg();
            debug_image_pub_->publish(*debug_msg);

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Error processing image: %s", e.what());
        }
    }


    struct Quaternion {
        double w, x, y, z;
        Quaternion(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) {}
    };

    Quaternion rotationMatrixToQuaternion(const cv::Mat& R) {
        double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);
        double w, x, y, z;

        if (trace > 0) {
            double s = 0.5 / std::sqrt(trace + 1.0);
            w = 0.25 / s;
            x = (R.at<double>(2, 1) - R.at<double>(1, 2)) * s;
            y = (R.at<double>(0, 2) - R.at<double>(2, 0)) * s;
            z = (R.at<double>(1, 0) - R.at<double>(0, 1)) * s;
        } else if (R.at<double>(0, 0) > R.at<double>(1, 1) && R.at<double>(0, 0) > R.at<double>(2, 2)) {
            double s = 2.0 * std::sqrt(1.0 + R.at<double>(0, 0) - R.at<double>(1, 1) - R.at<double>(2, 2));
            w = (R.at<double>(2, 1) - R.at<double>(1, 2)) / s;
            x = 0.25 * s;
            y = (R.at<double>(0, 1) + R.at<double>(1, 0)) / s;
            z = (R.at<double>(0, 2) + R.at<double>(2, 0)) / s;
        } else if (R.at<double>(1, 1) > R.at<double>(2, 2)) {
            double s = 2.0 * std::sqrt(1.0 + R.at<double>(1, 1) - R.at<double>(0, 0) - R.at<double>(2, 2));
            w = (R.at<double>(0, 2) - R.at<double>(2, 0)) / s;
            x = (R.at<double>(0, 1) + R.at<double>(1, 0)) / s;
            y = 0.25 * s;
            z = (R.at<double>(1, 2) + R.at<double>(2, 1)) / s;
        } else {
            double s = 2.0 * std::sqrt(1.0 + R.at<double>(2, 2) - R.at<double>(0, 0) - R.at<double>(1, 1));
            w = (R.at<double>(1, 0) - R.at<double>(0, 1)) / s;
            x = (R.at<double>(0, 2) + R.at<double>(2, 0)) / s;
            y = (R.at<double>(1, 2) + R.at<double>(2, 1)) / s;
            z = 0.25 * s;
        }

        return Quaternion(w, x, y, z);
    }

    // Member variables
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Publisher<aruco_slam_ailab::msg::MarkerArray>::SharedPtr aruco_poses_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr visualization_markers_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;

    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    std::string camera_frame_;
    bool camera_matrix_initialized_ = false;

    double marker_size_;
    std::set<int> target_marker_ids_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArucoDetectorNode>();
    
    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Exception: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}
