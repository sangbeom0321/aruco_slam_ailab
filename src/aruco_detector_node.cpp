#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include "aruco_slam_ailab/msg/marker_array.hpp"
#include "aruco_slam_ailab/msg/marker_observation.hpp"
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <mutex>

class ArucoDetectorNode : public rclcpp::Node {
public:
    ArucoDetectorNode() : Node("aruco_detector_node") {
        // Parameters
        declare_parameter("camera_topic", "/camera/rgb/image_raw");
        declare_parameter("camera_info_topic", "/camera/rgb/camera_info");
        declare_parameter("depth_topic", "/camera/depth/depth/image_raw");
        declare_parameter("depth_camera_info_topic", "/camera/depth/depth/camera_info");
        declare_parameter("marker_size", 0.30);
        declare_parameter("aruco_dict_type", "DICT_4X4_50");
        declare_parameter("debug", false);

        std::string camera_topic = get_parameter("camera_topic").as_string();
        std::string camera_info_topic = get_parameter("camera_info_topic").as_string();
        std::string depth_topic = get_parameter("depth_topic").as_string();
        std::string depth_camera_info_topic = get_parameter("depth_camera_info_topic").as_string();
        marker_size_ = get_parameter("marker_size").as_double();
        std::string aruco_dict_type = get_parameter("aruco_dict_type").as_string();
        debug_enabled_ = get_parameter("debug").as_bool();

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

        depth_image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            depth_topic, 10,
            std::bind(&ArucoDetectorNode::depthImageCallback, this, std::placeholders::_1));

        depth_camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            depth_camera_info_topic, 10,
            std::bind(&ArucoDetectorNode::depthCameraInfoCallback, this, std::placeholders::_1));

        // Publishers
        // Publish custom MarkerArray directly for SLAM backend
        aruco_poses_pub_ = create_publisher<aruco_slam_ailab::msg::MarkerArray>(
            "/aruco_poses", 10);
        // Publish visualization markers for RViz (MarkerArray display accepts any topic name, no "_array" suffix required)
        visualization_markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            "/aruco_markers", 10);
        debug_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
            "/aruco_debug_image", 10);

        if (debug_enabled_) {
            RCLCPP_INFO(get_logger(), "ArUco Detector Node started");
            RCLCPP_INFO(get_logger(), "  Camera topic: %s", camera_topic.c_str());
            RCLCPP_INFO(get_logger(), "  Depth topic: %s", depth_topic.c_str());
            RCLCPP_INFO(get_logger(), "  Marker size: %.2f m", marker_size_);
            RCLCPP_INFO(get_logger(), "  ArUco dictionary: %s", aruco_dict_type.c_str());
        }
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
            if (debug_enabled_) RCLCPP_WARN(get_logger(), "Unknown dictionary type: %s, using DICT_4X4_50", dict_type.c_str());
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

            if (debug_enabled_) RCLCPP_INFO(get_logger(), "Camera intrinsics received. Frame: %s", camera_frame_.c_str());
        }
    }

    void depthImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            std::lock_guard<std::mutex> lock(depth_mutex_);
            if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
                latest_depth_image_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
            } else if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
                latest_depth_image_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
            } else {
                if (debug_enabled_) RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                    "Depth image encoding not supported: %s (use 16UC1 or 32FC1)", msg->encoding.c_str());
                return;
            }
            depth_image_valid_ = true;
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "depth cv_bridge: %s", e.what());
        }
    }

    void depthCameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        (void)msg;
        if (!depth_camera_info_received_) {
            depth_camera_info_received_ = true;
            if (debug_enabled_) RCLCPP_INFO(get_logger(), "Depth camera info received");
        }
    }

    // RGB 이미지 좌표 (cx, cy)를 depth 이미지에서 샘플링. 깊이 해상도가 다를 수 있음.
    // 반환: 미터 단위 깊이. 유효하지 않으면 -1.0
    double sampleDepthAtMarkerCenter(const cv::Mat& depth, int rgb_cols, int rgb_rows,
                                      const std::vector<cv::Point2f>& corners) {
        if (depth.empty() || rgb_cols <= 0 || rgb_rows <= 0) return -1.0;
        float cx = 0, cy = 0;
        for (const auto& p : corners) { cx += p.x; cy += p.y; }
        cx /= 4.0f; cy /= 4.0f;
        int du = static_cast<int>(cx * depth.cols / rgb_cols);
        int dv = static_cast<int>(cy * depth.rows / rgb_rows);
        if (du < 0 || du >= depth.cols || dv < 0 || dv >= depth.rows) return -1.0;

        double z_m = -1.0;
        if (depth.type() == CV_16UC1) {
            uint16_t v = depth.at<uint16_t>(dv, du);
            if (v == 0) return -1.0;
            z_m = v * 0.001;  // mm -> m
        } else if (depth.type() == CV_32FC1) {
            float v = depth.at<float>(dv, du);
            if (std::isnan(v) || v <= 0.0f) return -1.0;
            z_m = static_cast<double>(v);
        }
        return z_m;
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (!camera_matrix_initialized_) {
            if (debug_enabled_) RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for camera info...");
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
                // Estimate poses for all detected markers
                std::vector<cv::Vec3d> rvecs, tvecs;
                cv::aruco::estimatePoseSingleMarkers(
                    corners, marker_size_,
                    camera_matrix_, dist_coeffs_,
                    rvecs, tvecs);

                // ID 0~9만 사용 (이외는 탐색 결과에서 제외)
                std::vector<int> ids_filt;
                std::vector<std::vector<cv::Point2f>> corners_filt;
                std::vector<cv::Vec3d> rvecs_filt, tvecs_filt;
                for (size_t idx = 0; idx < ids.size(); ++idx) {
                    if (target_marker_ids_.count(ids[idx]) == 0) continue;
                    ids_filt.push_back(ids[idx]);
                    corners_filt.push_back(corners[idx]);
                    rvecs_filt.push_back(rvecs[idx]);
                    tvecs_filt.push_back(tvecs[idx]);
                }

                if (ids_filt.empty()) {
                    // 마커 없음: 시각화만 스킵 (DELETE 안 함 → 기존 마커 누적 유지)
                } else {
                // Draw detected markers (0~9만)
                cv::aruco::drawDetectedMarkers(debug_image, corners_filt, ids_filt);

                // Create custom MarkerArray for SLAM backend
                aruco_slam_ailab::msg::MarkerArray marker_array;
                marker_array.header.stamp = msg->header.stamp;
                marker_array.header.frame_id = camera_frame_;

                for (size_t idx = 0; idx < ids_filt.size(); ++idx) {
                    cv::Vec3d rvec = rvecs_filt[idx];
                    cv::Vec3d tvec = tvecs_filt[idx];
                    int marker_id = ids_filt[idx];

                    // Depth 추정 정확도 로그: 재투영 오차 + RGB(ArUco) 추정 vs 센서 깊이
                    {
                        // 마커 3D 코너 (마커 좌표계)
                        std::vector<cv::Point3f> obj_pts = {
                            cv::Point3f(-marker_size_/2,  marker_size_/2, 0),
                            cv::Point3f( marker_size_/2,  marker_size_/2, 0),
                            cv::Point3f( marker_size_/2, -marker_size_/2, 0),
                            cv::Point3f(-marker_size_/2, -marker_size_/2, 0)
                        };
                        std::vector<cv::Point2f> proj_pts;
                        cv::projectPoints(obj_pts, rvec, tvec, camera_matrix_, dist_coeffs_, proj_pts);
                        double err_sum = 0;
                        for (size_t k = 0; k < 4; ++k) {
                            double dx = proj_pts[k].x - corners_filt[idx][k].x;
                            double dy = proj_pts[k].y - corners_filt[idx][k].y;
                            err_sum += dx*dx + dy*dy;
                        }
                        double reproj_error_px = std::sqrt(err_sum / 4.0);
                        double depth_aruco = tvec[2];

                        cv::Mat depth_for_sample;
                        {
                            std::lock_guard<std::mutex> lock(depth_mutex_);
                            if (depth_image_valid_ && !latest_depth_image_.empty())
                                depth_for_sample = latest_depth_image_.clone();
                        }
                        double depth_sensor = sampleDepthAtMarkerCenter(
                            depth_for_sample, cv_image.cols, cv_image.rows, corners_filt[idx]);

                        if (depth_sensor > 0.0) {
                            double err_m = depth_aruco - depth_sensor;
                            double err_pct = (depth_sensor > 1e-6) ? (100.0 * err_m / depth_sensor) : 0.0;
                            // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                            //     "[Depth accuracy] id=%d | ArUco(RGB)=%.3f m | sensor=%.3f m | err=%.3f m (%.1f%%) | reproj=%.2f px",
                            //     marker_id, depth_aruco, depth_sensor, err_m, err_pct, reproj_error_px);
                        } else {
                            if (debug_enabled_) RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                                "[Depth] id=%d | ArUco(RGB)=%.3f m | sensor=invalid | reproj_err=%.2f px",
                                marker_id, depth_aruco, reproj_error_px);
                        }
                    }

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
                
                // Publish 3D visualization for RViz: 누적 (DELETE 안 함, 매 프레임 고유 ID로 ADD만)
                // stamp를 now()로 하여 map 기준 TF lookup 시 extrapolation into future 방지
                std_msgs::msg::Header vis_header;
                vis_header.stamp = now();
                vis_header.frame_id = camera_frame_;

                visualization_msgs::msg::MarkerArray vis_marker_array;
                const double axis_scale = marker_size_ * 0.6;
                const size_t n = ids_filt.size();
                vis_marker_array.markers.reserve(n * 5);

                for (size_t idx = 0; idx < n; ++idx) {
                    int mid = ids_filt[idx];
                    const auto& obs = marker_array.markers[idx];
                    double r = ((mid * 37) % 255) / 255.0;
                    double g = ((mid * 73) % 255) / 255.0;
                    double b = ((mid * 113) % 255) / 255.0;

                    // 1) 마커 평면 큐브 (고유 ID로 누적)
                    visualization_msgs::msg::Marker cube;
                    cube.header = vis_header;
                    cube.ns = "aruco_cubes";
                    cube.id = vis_marker_id_counter_++;
                    cube.type = visualization_msgs::msg::Marker::CUBE;
                    cube.action = visualization_msgs::msg::Marker::ADD;
                    cube.pose = obs.pose;
                    cube.scale.x = marker_size_;
                    cube.scale.y = marker_size_;
                    cube.scale.z = 0.01;
                    cube.color.a = 0.6;
                    cube.color.r = r;
                    cube.color.g = g;
                    cube.color.b = b;
                    cube.lifetime.sec = 0;
                    cube.lifetime.nanosec = 0;
                    vis_marker_array.markers.push_back(cube);

                    // 2) 마커 중심 3D 위치 구
                    visualization_msgs::msg::Marker sphere;
                    sphere.header = vis_header;
                    sphere.ns = "aruco_centers";
                    sphere.id = vis_marker_id_counter_++;
                    sphere.type = visualization_msgs::msg::Marker::SPHERE;
                    sphere.action = visualization_msgs::msg::Marker::ADD;
                    sphere.pose.position = obs.pose.position;
                    sphere.pose.orientation.w = 1.0;
                    sphere.pose.orientation.x = sphere.pose.orientation.y = sphere.pose.orientation.z = 0.0;
                    sphere.scale.x = sphere.scale.y = sphere.scale.z = marker_size_ * 0.15;
                    sphere.color.a = 0.9;
                    sphere.color.r = r;
                    sphere.color.g = g;
                    sphere.color.b = b;
                    sphere.lifetime.sec = 0;
                    sphere.lifetime.nanosec = 0;
                    vis_marker_array.markers.push_back(sphere);

                    // 3) ID 텍스트
                    visualization_msgs::msg::Marker text;
                    text.header = vis_header;
                    text.ns = "aruco_labels";
                    text.id = vis_marker_id_counter_++;
                    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                    text.action = visualization_msgs::msg::Marker::ADD;
                    text.pose.position = obs.pose.position;
                    text.pose.position.z += marker_size_ * 0.6;
                    text.pose.orientation.w = 1.0;
                    text.pose.orientation.x = text.pose.orientation.y = text.pose.orientation.z = 0.0;
                    text.scale.z = marker_size_ * 0.4;
                    text.color.a = 1.0;
                    text.color.r = 1.0;
                    text.color.g = 1.0;
                    text.color.b = 1.0;
                    text.text = "ID:" + std::to_string(mid);
                    text.lifetime.sec = 0;
                    text.lifetime.nanosec = 0;
                    vis_marker_array.markers.push_back(text);

                    // 4) 좌표축: X(빨강), Y(초록), Z(파랑) 화살표
                    geometry_msgs::msg::Point origin;
                    origin.x = obs.pose.position.x;
                    origin.y = obs.pose.position.y;
                    origin.z = obs.pose.position.z;
                    auto q = obs.pose.orientation;
                    auto rot = [&q](double x, double y, double z) {
                        geometry_msgs::msg::Point p;
                        double qw = q.w, qx = q.x, qy = q.y, qz = q.z;
                        p.x = (1 - 2*(qy*qy + qz*qz)) * x + 2*(qx*qy - qw*qz) * y + 2*(qx*qz + qw*qy) * z;
                        p.y = 2*(qx*qy + qw*qz) * x + (1 - 2*(qx*qx + qz*qz)) * y + 2*(qy*qz - qw*qx) * z;
                        p.z = 2*(qx*qz - qw*qy) * x + 2*(qy*qz + qw*qx) * y + (1 - 2*(qx*qx + qy*qy)) * z;
                        return p;
                    };
                    auto arrow = [&](int axis_id, double ax, double ay, double az, float cr, float cg, float cb) {
                        (void)axis_id;
                        visualization_msgs::msg::Marker ar;
                        ar.header = vis_header;
                        ar.ns = "aruco_axes";
                        ar.id = vis_marker_id_counter_++;
                        ar.type = visualization_msgs::msg::Marker::ARROW;
                        ar.action = visualization_msgs::msg::Marker::ADD;
                        geometry_msgs::msg::Point end;
                        geometry_msgs::msg::Point pd = rot(ax, ay, az);
                        end.x = origin.x + pd.x * axis_scale;
                        end.y = origin.y + pd.y * axis_scale;
                        end.z = origin.z + pd.z * axis_scale;
                        ar.points.resize(2);
                        ar.points[0] = origin;
                        ar.points[1] = end;
                        ar.scale.x = axis_scale * 0.08;
                        ar.scale.y = axis_scale * 0.16;
                        ar.color.a = 0.9;
                        ar.color.r = cr;
                        ar.color.g = cg;
                        ar.color.b = cb;
                        ar.lifetime.sec = 0;
                        ar.lifetime.nanosec = 0;
                        vis_marker_array.markers.push_back(ar);
                    };
                    arrow(0, 1, 0, 0, 1.0f, 0.0f, 0.0f);  // X
                    arrow(1, 0, 1, 0, 0.0f, 1.0f, 0.0f);  // Y
                    arrow(2, 0, 0, 1, 0.0f, 0.0f, 1.0f);  // Z
                }

                visualization_markers_pub_->publish(vis_marker_array);
                }
            } else {
                // 마커 전혀 없음: 시각화만 스킵 (누적 유지)
                if (debug_enabled_) RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
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
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_camera_info_sub_;
    rclcpp::Publisher<aruco_slam_ailab::msg::MarkerArray>::SharedPtr aruco_poses_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr visualization_markers_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;

    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    std::string camera_frame_;
    bool debug_enabled_ = false;
    bool camera_matrix_initialized_ = false;

    double marker_size_;
    std::set<int> target_marker_ids_;
    int vis_marker_id_counter_ = 0;  // 3D 시각화 마커 누적용 고유 ID

    std::mutex depth_mutex_;
    cv::Mat latest_depth_image_;
    bool depth_image_valid_ = false;
    bool depth_camera_info_received_ = false;
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
