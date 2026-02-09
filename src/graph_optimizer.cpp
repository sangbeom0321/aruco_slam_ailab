#include "aruco_slam_ailab/utility.hpp"
#include "aruco_slam_ailab/msg/marker_observation.hpp"
#include "aruco_slam_ailab/msg/marker_array.hpp"

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Pose3.h>

#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <set>
#include <limits>
#include <cmath>

using gtsam::symbol_shorthand::X; // Pose3 for robot states
using gtsam::symbol_shorthand::L; // Pose3 for landmarks
using gtsam::symbol_shorthand::V; // Velocity
using gtsam::symbol_shorthand::B; // Bias

namespace aruco_slam_ailab {

class GraphOptimizer : public ParamServer {
public:
    std::mutex mtx_;

    // Subscribers
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subImuOdometry_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subWheelOdom_;
    rclcpp::Subscription<aruco_slam_ailab::msg::MarkerArray>::SharedPtr subArucoPoses_;

    // Publishers
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubGlobalOdometry_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;

    // TF
    std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;

    // GTSAM
    gtsam::ISAM2 isam_;
    gtsam::NonlinearFactorGraph graphFactors_;
    gtsam::Values graphValues_;

    // Noise models
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr wheelOdomNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr arucoObsNoise_;
    gtsam::Vector noiseModelBetweenBias_;

    // State tracking
    int keyframeIdx_ = 0;
    bool systemInitialized_ = false;
    int frameIdx_ = 0;  // Frame counter for continuous optimization (like aruco-slam)

    // Previous states
    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::imuBias::ConstantBias prevBias_;

    // Data buffers
    std::deque<nav_msgs::msg::Odometry> imuOdomQueue_;
    std::deque<nav_msgs::msg::Odometry> wheelOdomQueue_;
    std::deque<aruco_slam_ailab::msg::MarkerArray> arucoQueue_;

    // Landmark tracking: marker_id -> landmark symbol key
    std::map<int, gtsam::Key> landmarkMap_;

    // Keyframe tracking
    double lastKeyframeTime_ = -1.0;
    gtsam::Pose3 lastKeyframePose_;

    // Path for visualization
    nav_msgs::msg::Path globalPath_;

    // Logging: topic receive check
    uint64_t imuOdomCallbackCount_ = 0;
    uint64_t wheelOdomCallbackCount_ = 0;
    uint64_t arucoCallbackCount_ = 0;
    bool imuOdomFirstLogged_ = false;
    bool wheelOdomFirstLogged_ = false;
    bool arucoFirstLogged_ = false;

    // Extrinsic transformations
    gtsam::Pose3 baseToImu_;
    gtsam::Pose3 baseToCam_;
    gtsam::Pose3 opticalToCameraLink_;  // OpenCV optical frame to ROS camera_link frame

    GraphOptimizer(const rclcpp::NodeOptions& options) : ParamServer("graph_optimizer", options) {
        // Subscribers
        if (useImu) {
            subImuOdometry_ = create_subscription<nav_msgs::msg::Odometry>(
                "/odometry/imu_incremental", 2000,
                std::bind(&GraphOptimizer::imuOdometryHandler, this, std::placeholders::_1));
        }

        if (useWheelOdom) {
            subWheelOdom_ = create_subscription<nav_msgs::msg::Odometry>(
                wheelOdomTopic, 100,
                std::bind(&GraphOptimizer::wheelOdomHandler, this, std::placeholders::_1));
        }

        subArucoPoses_ = create_subscription<aruco_slam_ailab::msg::MarkerArray>(
            arucoPosesTopic, 100,
            std::bind(&GraphOptimizer::arucoHandler, this, std::placeholders::_1));

        // Publishers
        pubGlobalOdometry_ = create_publisher<nav_msgs::msg::Odometry>(
            "/odometry/global", 10);
        pubPath_ = create_publisher<nav_msgs::msg::Path>("/path", 10);

        // TF broadcaster
        tfBroadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        // Setup ISAM2
        gtsam::ISAM2Params isamParams;
        isamParams.relinearizeThreshold = isamRelinearizeThreshold;
        isamParams.relinearizeSkip = isamRelinearizeSkip;
        isam_ = gtsam::ISAM2(isamParams);

        // Setup noise models
        // Z축에 대한 더 강한 제약 (로봇이 지면에 있다고 가정)
        priorPoseNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.01, 0.01, 0.001, 0.01, 0.01, 0.01).finished());  // Z축 노이즈를 더 작게 설정
        priorVelNoise_ = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);  // 초기 정지 상태 가정
        priorBiasNoise_ = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);

        wheelOdomNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << wheelOdomRotNoise, wheelOdomRotNoise, wheelOdomRotNoise,
                                 wheelOdomTransNoise, wheelOdomTransNoise, wheelOdomTransNoise).finished());

        arucoObsNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << arucoRotNoise, arucoRotNoise, arucoRotNoise,
                                 arucoTransNoise, arucoTransNoise, arucoTransNoise).finished());

        noiseModelBetweenBias_ = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN,
                                                       imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

        // Setup extrinsic transformations
        gtsam::Rot3 rot_base_imu(extRotBaseImu);
        gtsam::Point3 trans_base_imu(extTransBaseImu.x(), extTransBaseImu.y(), extTransBaseImu.z());
        baseToImu_ = gtsam::Pose3(rot_base_imu, trans_base_imu);

        gtsam::Rot3 rot_base_cam(extRotBaseCam);
        gtsam::Point3 trans_base_cam(extTransBaseCam.x(), extTransBaseCam.y(), extTransBaseCam.z());
        baseToCam_ = gtsam::Pose3(rot_base_cam, trans_base_cam);
        
        // optical frame -> camera_link: inverse of xacro camera_color_optical_joint
        // Ref: hunter2_description/xacro/include/camera/d455.xacro
        //      camera_color_optical_joint has rpy="${-M_PI/2} 0 ${-M_PI/2}" (camera_link -> optical)
        // So optical -> camera_link = RotZ(90)*RotX(90)
        gtsam::Rot3 rotZ90 = gtsam::Rot3::Rz(M_PI/2);
        gtsam::Rot3 rotX90 = gtsam::Rot3::Rx(M_PI/2);
        gtsam::Rot3 opticalToCamera = rotZ90 * rotX90;
        opticalToCameraLink_ = gtsam::Pose3(opticalToCamera, gtsam::Point3(0, 0, 0));

        RCLCPP_INFO(get_logger(), "Extrinsics: base_link->camera_link trans=(%.3f, %.3f, %.3f) [ref: hunter2_core.xacro]",
                    extTransBaseCam.x(), extTransBaseCam.y(), extTransBaseCam.z());

        RCLCPP_INFO(get_logger(), "Graph Optimizer node initialized (topic_debug_log=%s, use_imu=%s)", 
            enableTopicDebugLog ? "on" : "off", useImu ? "true" : "false");
        if (useImu) {
            RCLCPP_INFO(get_logger(), "  subscribe: /odometry/imu_incremental, %s, %s",
                (useWheelOdom ? wheelOdomTopic.c_str() : "(wheel_odom off)"), arucoPosesTopic.c_str());
        } else {
            RCLCPP_INFO(get_logger(), "  subscribe: %s (vision-only mode, no IMU)",
                arucoPosesTopic.c_str());
        }
    }

    void imuOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg) {
        std::lock_guard<std::mutex> lock(mtx_);
        imuOdomCallbackCount_++;
        if (enableTopicDebugLog) {
            if (!imuOdomFirstLogged_) {
                imuOdomFirstLogged_ = true;
                RCLCPP_INFO(get_logger(), "[GraphOpt] First /odometry/imu_incremental received (stamp=%.3f)", stamp2Sec(odomMsg->header.stamp));
            }
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                "[GraphOpt] IMU odom: count=%lu queue=%zu", imuOdomCallbackCount_, imuOdomQueue_.size());
        }
        imuOdomQueue_.push_back(*odomMsg);
        
        // Keep queue size reasonable
        while (imuOdomQueue_.size() > 100) {
            imuOdomQueue_.pop_front();
        }

        // If not initialized and we have ArUco markers, try marker-based initialization first
        if (!systemInitialized_ && !arucoQueue_.empty()) {
            // Find latest ArUco observation
            for (const auto& markerArray : arucoQueue_) {
                if (!markerArray.markers.empty()) {
                    initializeFromFirstMarker(markerArray);
                    return;  // Don't process keyframes yet, wait for next IMU odom
                }
            }
        }

        // Process every frame (like aruco-slam: optimize on every observation)
        if (systemInitialized_) {
            processFrame();
        } else {
            // Process keyframes for initialization
            processKeyframes();
        }
    }
    
    void processFrame() {
        if (imuOdomQueue_.empty()) {
            return;
        }
        
        // Get latest IMU odometry
        nav_msgs::msg::Odometry latestImuOdom = imuOdomQueue_.back();
        double currentTime = stamp2Sec(latestImuOdom.header.stamp);
        gtsam::Pose3 currentPose = poseMsgToGtsam(latestImuOdom.pose.pose);
        
        // Convert to IMU frame
        gtsam::Pose3 imuPose = currentPose.compose(baseToImu_);
        gtsam::Vector3 vel(latestImuOdom.twist.twist.linear.x,
                           latestImuOdom.twist.twist.linear.y,
                           latestImuOdom.twist.twist.linear.z);
        
        // Get current camera pose estimate (like aruco-slam: camera_pose = current_estimate.atPose3(X(i)))
        gtsam::Pose3 currentCameraPose = prevPose_.compose(baseToImu_);
        
        // Add ArUco landmark observations (like aruco-slam: for idx, pose in zip(ids, poses))
        addArucoObservations(frameIdx_, currentTime);
        
        // Add odometry factor and estimate (like aruco-slam: add_odom_factor_and_estimate)
        // Zero motion model: assume no change between frames
        gtsam::Pose3 zeroMotion = gtsam::Pose3();  // Identity pose (no rotation, no translation)
        
        // Add between factor with zero motion model (like aruco-slam line 177-185)
        graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
            X(frameIdx_), X(frameIdx_ + 1), zeroMotion, wheelOdomNoise_));
        
        // Add initial estimate for next frame (like aruco-slam line 172-175)
        // Use current pose from IMU odometry as initial estimate
        graphValues_.insert(X(frameIdx_ + 1), imuPose);
        graphValues_.insert(V(frameIdx_ + 1), vel);
        graphValues_.insert(B(frameIdx_ + 1), prevBias_);
        
        // Optimize every frame (like aruco-slam: don't optimize on first iteration)
        if (frameIdx_ == 0) {
            // First iteration: just set current estimate (like aruco-slam line 149-150)
            // Values are already inserted above
        } else {
            // Update ISAM2 and get optimized result (like aruco-slam line 152-153)
            isam_.update(graphFactors_, graphValues_);
            graphFactors_.resize(0);
            graphValues_.clear();
            
            // Get optimized result
            gtsam::Values result = isam_.calculateEstimate();
            gtsam::Pose3 optimizedImuPose = result.at<gtsam::Pose3>(X(frameIdx_ + 1));
            gtsam::Pose3 optimizedBasePose = optimizedImuPose.compose(baseToImu_.inverse());
            
            gtsam::Vector3 optimizedVel = result.at<gtsam::Vector3>(V(frameIdx_ + 1));
            prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(frameIdx_ + 1));
            
            prevPose_ = optimizedBasePose;
            prevVel_ = optimizedVel;
            
            // Publish results every frame
            publishOdometry(optimizedBasePose, latestImuOdom.header.stamp);
            publishTF(optimizedBasePose, latestImuOdom.header.stamp);
            updatePath(optimizedBasePose, latestImuOdom.header.stamp);
            
            // Update keyframe tracking for visualization
            lastKeyframeTime_ = currentTime;
            lastKeyframePose_ = optimizedBasePose;
        }
        
        frameIdx_++;
    }
    
    void processFrameVisionOnly(const aruco_slam_ailab::msg::MarkerArray& markerArray) {
        if (markerArray.markers.empty()) {
            return;
        }
        
        double currentTime = stamp2Sec(markerArray.header.stamp);
        
        // Get current camera pose estimate (like aruco-slam: camera_pose = current_estimate.atPose3(X(i)))
        gtsam::Pose3 currentCameraPose = prevPose_.compose(baseToCam_);
        
        // Add ArUco landmark observations (like aruco-slam: for idx, pose in zip(ids, poses))
        addArucoObservations(frameIdx_, currentTime);
        
        // Add odometry factor and estimate (like aruco-slam: add_odom_factor_and_estimate)
        // Zero motion model: assume no change between frames
        gtsam::Pose3 zeroMotion = gtsam::Pose3();  // Identity pose (no rotation, no translation)
        
        // Add between factor with zero motion model (like aruco-slam line 177-185)
        graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
            X(frameIdx_), X(frameIdx_ + 1), zeroMotion, wheelOdomNoise_));
        
        // Add initial estimate for next frame (like aruco-slam line 172-175)
        // Use current pose estimate as initial estimate (vision-only: no IMU odometry)
        gtsam::Pose3 nextCameraPose = currentCameraPose;  // Use current pose as initial estimate
        gtsam::Pose3 nextBasePose = nextCameraPose.compose(baseToCam_.inverse());
        gtsam::Pose3 nextImuPose = nextBasePose.compose(baseToImu_);
        
        graphValues_.insert(X(frameIdx_ + 1), nextImuPose);
        graphValues_.insert(V(frameIdx_ + 1), gtsam::Vector3(0, 0, 0));  // Zero velocity in vision-only mode
        graphValues_.insert(B(frameIdx_ + 1), prevBias_);
        
        // Optimize every frame (like aruco-slam: don't optimize on first iteration)
        if (frameIdx_ == 0) {
            // First iteration: just set current estimate (like aruco-slam line 149-150)
            // Values are already inserted above
        } else {
            // Update ISAM2 and get optimized result (like aruco-slam line 152-153)
            isam_.update(graphFactors_, graphValues_);
            graphFactors_.resize(0);
            graphValues_.clear();
            
            // Get optimized result
            gtsam::Values result = isam_.calculateEstimate();
            gtsam::Pose3 optimizedImuPose = result.at<gtsam::Pose3>(X(frameIdx_ + 1));
            gtsam::Pose3 optimizedBasePose = optimizedImuPose.compose(baseToImu_.inverse());
            
            gtsam::Vector3 optimizedVel = result.at<gtsam::Vector3>(V(frameIdx_ + 1));
            prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(frameIdx_ + 1));
            
            prevPose_ = optimizedBasePose;
            prevVel_ = optimizedVel;
            
            // Publish results every frame
            rclcpp::Time stamp = markerArray.header.stamp;
            publishOdometry(optimizedBasePose, stamp);
            publishTF(optimizedBasePose, stamp);
            updatePath(optimizedBasePose, stamp);
            
            // Update keyframe tracking for visualization
            lastKeyframeTime_ = currentTime;
            lastKeyframePose_ = optimizedBasePose;
        }
        
        frameIdx_++;
    }

    void wheelOdomHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg) {
        std::lock_guard<std::mutex> lock(mtx_);
        wheelOdomCallbackCount_++;
        if (enableTopicDebugLog) {
            if (!wheelOdomFirstLogged_) {
                wheelOdomFirstLogged_ = true;
                RCLCPP_INFO(get_logger(), "[GraphOpt] First wheel_odom received on %s (stamp=%.3f)", wheelOdomTopic.c_str(), stamp2Sec(odomMsg->header.stamp));
            }
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                "[GraphOpt] wheel odom: count=%lu queue=%zu", wheelOdomCallbackCount_, wheelOdomQueue_.size());
        }
        wheelOdomQueue_.push_back(*odomMsg);
        
        while (wheelOdomQueue_.size() > 100) {
            wheelOdomQueue_.pop_front();
        }
    }

    void arucoHandler(const aruco_slam_ailab::msg::MarkerArray::SharedPtr markerArrayMsg) {
        std::lock_guard<std::mutex> lock(mtx_);
        arucoCallbackCount_++;
        size_t numMarkers = markerArrayMsg->markers.size();
        if (enableTopicDebugLog) {
            if (!arucoFirstLogged_) {
                arucoFirstLogged_ = true;
                RCLCPP_INFO(get_logger(), "[GraphOpt] First %s received (stamp=%.3f, markers=%zu)", arucoPosesTopic.c_str(), stamp2Sec(markerArrayMsg->header.stamp), numMarkers);
            }
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                "[GraphOpt] aruco: count=%lu queue=%zu markers=%zu", arucoCallbackCount_, arucoQueue_.size(), numMarkers);
        }
        arucoQueue_.push_back(*markerArrayMsg);
        
        while (arucoQueue_.size() > 50) {
            arucoQueue_.pop_front();
        }
        
        // If not using IMU, process ArUco markers directly (vision-only mode)
        if (!useImu) {
            if (!systemInitialized_ && numMarkers > 0) {
                // Initialize from first ArUco marker
                initializeFromFirstMarker(*markerArrayMsg);
            } else if (systemInitialized_) {
                // Process frame with ArUco observations
                processFrameVisionOnly(*markerArrayMsg);
            }
            return;
        }
        
        // Trigger initialization from first ArUco marker if system not initialized (IMU mode)
        // Priority: ArUco marker-based initialization over IMU odometry-based
        if (!systemInitialized_ && numMarkers > 0) {
            if (!imuOdomQueue_.empty()) {
                // We have both ArUco and IMU odometry, initialize from marker
                initializeFromFirstMarker(*markerArrayMsg);
            } else {
                // ArUco arrived first, wait for IMU odometry (will be handled in imuOdometryHandler)
                RCLCPP_INFO(get_logger(), "[GraphOpt] First ArUco marker received, waiting for IMU odometry to initialize...");
            }
        }
    }

    bool shouldCreateKeyframe(const nav_msgs::msg::Odometry& odom) {
        double currentTime = stamp2Sec(odom.header.stamp);
        
        if (lastKeyframeTime_ < 0) {
            return true;  // First keyframe
        }

        // Time-based check
        if (currentTime - lastKeyframeTime_ < keyframeTimeInterval) {
            return false;
        }

        // Distance/angle-based check
        gtsam::Pose3 currentPose = poseMsgToGtsam(odom.pose.pose);
        
        if (keyframeIdx_ > 0) {
            gtsam::Pose3 delta = lastKeyframePose_.between(currentPose);
            double dist = delta.translation().norm();
            double angle = delta.rotation().axisAngle().second;  // angle in radians

            if (dist < keyframeDistanceThreshold && angle < keyframeAngleThreshold) {
                return false;
            }
        }

        return true;
    }

    void processKeyframes() {
        if (imuOdomQueue_.empty()) {
            return;
        }

        // Get latest IMU odometry
        nav_msgs::msg::Odometry latestImuOdom = imuOdomQueue_.back();
        
        if (!shouldCreateKeyframe(latestImuOdom)) {
            if (enableTopicDebugLog && systemInitialized_) {
                double currentTime = stamp2Sec(latestImuOdom.header.stamp);
                gtsam::Pose3 currentPose = poseMsgToGtsam(latestImuOdom.pose.pose);
                if (lastKeyframeTime_ >= 0) {
                    gtsam::Pose3 delta = lastKeyframePose_.between(currentPose);
                    double dist = delta.translation().norm();
                    double angle = delta.rotation().axisAngle().second;
                    RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000,
                        "[GraphOpt] Skipping keyframe: time_diff=%.3f (need>=%.3f), dist=%.3f (need>=%.3f), angle=%.3f (need>=%.3f)",
                        currentTime - lastKeyframeTime_, keyframeTimeInterval,
                        dist, keyframeDistanceThreshold, angle, keyframeAngleThreshold);
                }
            }
            if (enableTopicDebugLog && systemInitialized_) {
                RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000,
                    "[GraphOpt] Skipping keyframe: time_interval=%.3f, dist=%.3f, angle=%.3f",
                    stamp2Sec(latestImuOdom.header.stamp) - lastKeyframeTime_,
                    (lastKeyframePose_.between(poseMsgToGtsam(latestImuOdom.pose.pose))).translation().norm(),
                    (lastKeyframePose_.between(poseMsgToGtsam(latestImuOdom.pose.pose))).rotation().axisAngle().second);
            }
            return;
        }

        double keyframeTime = stamp2Sec(latestImuOdom.header.stamp);
        gtsam::Pose3 keyframePose = poseMsgToGtsam(latestImuOdom.pose.pose);

        // Initialize system if needed
        if (!systemInitialized_) {
            initializeSystem(keyframePose, latestImuOdom);
            return;
        }

        // Add new keyframe
        addKeyframe(keyframeTime, keyframePose, latestImuOdom);

        lastKeyframeTime_ = keyframeTime;
        lastKeyframePose_ = keyframePose;
    }

    // Initialize system from first ArUco marker observation
    void initializeFromFirstMarker(const aruco_slam_ailab::msg::MarkerArray& markerArray) {
        if (markerArray.markers.empty()) {
            return;
        }
        
        // For vision-only mode, we don't need IMU odometry
        nav_msgs::msg::Odometry dummyOdom;
        dummyOdom.header.stamp = markerArray.header.stamp;
        dummyOdom.twist.twist.linear.x = 0.0;
        dummyOdom.twist.twist.linear.y = 0.0;
        dummyOdom.twist.twist.linear.z = 0.0;
        
        nav_msgs::msg::Odometry latestImuOdom = dummyOdom;
        if (!useImu && imuOdomQueue_.empty()) {
            // Vision-only mode: use dummy odometry
            latestImuOdom = dummyOdom;
        } else if (useImu && imuOdomQueue_.empty()) {
            // IMU mode but no IMU odometry yet
            return;
        } else {
            // IMU mode: use actual IMU odometry
            latestImuOdom = imuOdomQueue_.back();
        }
        
        // Use first marker (assume marker ID 0 is at origin, or use first marker as reference)
        const auto& firstMarker = markerArray.markers[0];
        geometry_msgs::msg::Pose markerPoseMsg = firstMarker.pose;
        
        // Convert observation to base_link frame
        gtsam::Pose3 markerObsBase;
        if (arucoObservationFrame == "camera") {
            // Observation is in OpenCV optical frame, convert to base_link
            // T_base_marker = T_base_cam * T_cam_optical * T_optical_marker
            // First convert from optical to camera_link, then to base_link
            gtsam::Pose3 markerObsOptical = poseMsgToGtsam(markerPoseMsg);
            gtsam::Pose3 markerObsCameraLink = opticalToCameraLink_.compose(markerObsOptical);
            markerObsBase = baseToCam_.compose(markerObsCameraLink);
        } else {
            // Observation is already in base_link frame
            markerObsBase = poseMsgToGtsam(markerPoseMsg);
        }
        
        // Assume first marker (ID 0) is at origin in map frame
        // If marker is at origin, then robot pose = -markerObsBase (inverse)
        // T_map_base = T_map_marker * T_marker_base = identity * markerObsBase.inverse()
        gtsam::Pose3 initialBasePose = markerObsBase.inverse();
        
        // If marker ID is not 0, we can't assume it's at origin
        // For now, use identity pose and let optimization correct it
        if (firstMarker.id != 0) {
            RCLCPP_WARN(get_logger(), "[GraphOpt] First marker ID=%d (not 0), using identity pose. Consider using marker ID 0 at origin.", firstMarker.id);
            initialBasePose = gtsam::Pose3();  // Identity
        }
        
        RCLCPP_INFO(get_logger(), "[GraphOpt] Initializing from first ArUco marker (ID=%d)", firstMarker.id);
        initializeSystem(initialBasePose, latestImuOdom);
    }

    void initializeSystem(const gtsam::Pose3& initialPose, const nav_msgs::msg::Odometry& odom) {
        // Initial pose in base_link frame
        prevPose_ = initialPose;
        prevVel_ = gtsam::Vector3(odom.twist.twist.linear.x,
                                  odom.twist.twist.linear.y,
                                  odom.twist.twist.linear.z);
        prevBias_ = gtsam::imuBias::ConstantBias();

        // Convert to IMU frame for IMU factors
        gtsam::Pose3 imuPose = initialPose.compose(baseToImu_);

        // Add prior factors
        graphFactors_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), imuPose, priorPoseNoise_));
        graphFactors_.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), prevVel_, priorVelNoise_));
        graphFactors_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), prevBias_, priorBiasNoise_));

        // Add initial values
        graphValues_.insert(X(0), imuPose);
        graphValues_.insert(V(0), prevVel_);
        graphValues_.insert(B(0), prevBias_);

        // Optimize
        isam_.update(graphFactors_, graphValues_);
        graphFactors_.resize(0);
        graphValues_.clear();

        keyframeIdx_ = 1;
        frameIdx_ = 0;  // Start frame index at 0 (like aruco-slam)
        systemInitialized_ = true;
        
        // Set initial keyframe tracking for next keyframe creation
        lastKeyframeTime_ = stamp2Sec(odom.header.stamp);
        lastKeyframePose_ = initialPose;

        RCLCPP_INFO(get_logger(), "System initialized at frame 0");
    }

    void addKeyframe(double time, const gtsam::Pose3& keyframePose, const nav_msgs::msg::Odometry& odom) {
        // Convert to IMU frame
        gtsam::Pose3 imuPose = keyframePose.compose(baseToImu_);
        gtsam::Vector3 vel(odom.twist.twist.linear.x,
                           odom.twist.twist.linear.y,
                           odom.twist.twist.linear.z);

        // Add IMU factor (simplified - in real implementation, integrate IMU measurements)
        // For now, use odometry as measurement
        gtsam::Pose3 deltaPose = prevPose_.between(keyframePose);
        gtsam::Pose3 deltaImuPose = gtsam::Pose3(prevPose_.compose(baseToImu_)).between(imuPose);

        // Add wheel odometry factor if available
        if (useWheelOdom && !wheelOdomQueue_.empty()) {
            // Find closest wheel odom measurement
            nav_msgs::msg::Odometry closestWheelOdom;
            double minTimeDiff = std::numeric_limits<double>::max();
            for (const auto& wheelOdom : wheelOdomQueue_) {
                double timeDiff = std::abs(stamp2Sec(wheelOdom.header.stamp) - time);
                if (timeDiff < minTimeDiff) {
                    minTimeDiff = timeDiff;
                    closestWheelOdom = wheelOdom;
                }
            }

            if (minTimeDiff < 0.1) {  // Within 100ms
                gtsam::Pose3 wheelDelta = poseMsgToGtsam(closestWheelOdom.pose.pose);
                graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                    X(keyframeIdx_ - 1), X(keyframeIdx_), wheelDelta, wheelOdomNoise_));
            }
        } else {
            // Use odometry-based between factor
            graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                X(keyframeIdx_ - 1), X(keyframeIdx_), deltaImuPose, wheelOdomNoise_));
        }

        // Add ArUco landmark observations
        addArucoObservations(keyframeIdx_, time);

        // Force Z-axis to be near 0 (ground plane constraint)
        // Extract translation and set Z to 0
        gtsam::Point3 translation = imuPose.translation();
        gtsam::Point3 constrainedTranslation(translation.x(), translation.y(), 0.0);
        gtsam::Pose3 constrainedImuPose(imuPose.rotation(), constrainedTranslation);
        
        // Also constrain Z-axis velocity to be near 0
        gtsam::Vector3 constrainedVel(vel.x(), vel.y(), 0.0);

        // Add initial estimate with Z-axis constraints
        graphValues_.insert(X(keyframeIdx_), constrainedImuPose);
        graphValues_.insert(V(keyframeIdx_), constrainedVel);
        graphValues_.insert(B(keyframeIdx_), prevBias_);
        
        // Add Z-axis prior factor for ground plane constraint
        gtsam::noiseModel::Diagonal::shared_ptr zAxisNoise = 
            gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(1) << 0.001).finished());
        // Note: GTSAM doesn't have direct partial pose prior, so we rely on initial value constraint

        // Optimize
        isam_.update(graphFactors_, graphValues_);
        graphFactors_.resize(0);
        graphValues_.clear();

        // Get optimized result
        gtsam::Values result = isam_.calculateEstimate();
        gtsam::Pose3 optimizedImuPose = result.at<gtsam::Pose3>(X(keyframeIdx_));
        gtsam::Pose3 optimizedBasePose = optimizedImuPose.compose(baseToImu_.inverse());
        
        // Apply ground plane constraint: force Z-axis to be near 0
        gtsam::Point3 baseTranslation = optimizedBasePose.translation();
        if (std::abs(baseTranslation.z()) > 0.05) {  // If Z is more than 5cm from ground
            // Reset Z to 0 and update pose
            gtsam::Point3 constrainedTranslation(baseTranslation.x(), baseTranslation.y(), 0.0);
            gtsam::Pose3 constrainedBasePose(optimizedBasePose.rotation(), constrainedTranslation);
            optimizedBasePose = constrainedBasePose;
            
            // Also update IMU pose
            optimizedImuPose = constrainedBasePose.compose(baseToImu_);
        }
        
        gtsam::Vector3 optimizedVel = result.at<gtsam::Vector3>(V(keyframeIdx_));
        // Constrain Z-axis velocity to be near 0
        if (std::abs(optimizedVel.z()) > 0.1) {
            optimizedVel = gtsam::Vector3(optimizedVel.x(), optimizedVel.y(), 0.0);
        }

        prevPose_ = optimizedBasePose;
        prevVel_ = optimizedVel;
        prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(keyframeIdx_));

        // Publish results
        publishOdometry(optimizedBasePose, odom.header.stamp);
        publishTF(optimizedBasePose, odom.header.stamp);
        updatePath(optimizedBasePose, odom.header.stamp);

        keyframeIdx_++;

        // Z축 발산 감지 및 경고
        double zPos = optimizedBasePose.translation().z();
        if (std::abs(zPos) > 0.1) {  // Z축이 0.1m 이상 벗어나면 경고
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "[GraphOpt] WARNING: Z-axis divergence detected! z=%.3f (expected near 0.0)", zPos);
        }
        
        RCLCPP_INFO(get_logger(), "[GraphOpt] Added keyframe %d, total landmarks: %zu, pose: [%.3f, %.3f, %.3f]", 
                     keyframeIdx_ - 1, landmarkMap_.size(),
                     optimizedBasePose.translation().x(),
                     optimizedBasePose.translation().y(),
                     optimizedBasePose.translation().z());
    }

    void addArucoObservations(int keyframeIdx, double time) {
        size_t markersProcessed = 0;
        size_t newLandmarksAdded = 0;
        size_t loopClosuresAdded = 0;
        
        // Find the closest ArUco observation to this keyframe time
        const aruco_slam_ailab::msg::MarkerArray* closestMarkerArray = nullptr;
        double minTimeDiff = std::numeric_limits<double>::max();
        
        for (const auto& markerArray : arucoQueue_) {
            double obsTime = stamp2Sec(markerArray.header.stamp);
            double timeDiff = std::abs(obsTime - time);
            if (timeDiff < minTimeDiff && timeDiff <= 0.2) {  // Within 200ms
                minTimeDiff = timeDiff;
                closestMarkerArray = &markerArray;
            }
        }
        
        if (closestMarkerArray == nullptr) {
            // No ArUco observations near this keyframe time
            return;
        }
        
        markersProcessed = closestMarkerArray->markers.size();

        // Get current pose estimate
        gtsam::Pose3 currentBasePose = prevPose_;
        gtsam::Pose3 currentImuPose = currentBasePose.compose(baseToImu_);
        
        // Track processed marker IDs to avoid duplicates within the same array
        std::set<int> processedMarkerIds;

        for (const auto& marker : closestMarkerArray->markers) {
            int markerId = marker.id;
            
            // Skip if this marker ID was already processed in this keyframe
            if (processedMarkerIds.find(markerId) != processedMarkerIds.end()) {
                continue;
            }
            processedMarkerIds.insert(markerId);
            
            geometry_msgs::msg::Pose markerPoseMsg = marker.pose;

                // Convert observation to base_link frame if needed
                gtsam::Pose3 markerObsBase;
                if (arucoObservationFrame == "camera") {
                    // Observation is in OpenCV optical frame, convert to base_link
                    // T_base_marker = T_base_cam * T_cam_optical * T_optical_marker
                    // First convert from optical to camera_link, then to base_link
                    gtsam::Pose3 markerObsOptical = poseMsgToGtsam(markerPoseMsg);
                    gtsam::Pose3 markerObsCameraLink = opticalToCameraLink_.compose(markerObsOptical);
                    markerObsBase = baseToCam_.compose(markerObsCameraLink);
                } else {
                    // Observation is already in base_link frame
                    markerObsBase = poseMsgToGtsam(markerPoseMsg);
                }

                // Check if landmark exists
                gtsam::Key landmarkKey;
                if (landmarkMap_.find(markerId) != landmarkMap_.end()) {
                    // Existing landmark - add observation factor (loop closure)
                    landmarkKey = landmarkMap_[markerId];
                    
                    // Compute relative pose: T_base_marker = T_base_base * T_base_marker_obs
                    // In map frame: T_map_marker = T_map_base * T_base_marker
                    gtsam::Pose3 relativePose = markerObsBase;
                    
                    graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                        X(keyframeIdx), landmarkKey, relativePose, arucoObsNoise_));

                    loopClosuresAdded++;
                    if (enableTopicDebugLog) {
                        RCLCPP_INFO(get_logger(), "[GraphOpt] Added loop closure: keyframe %d -> landmark %d (total landmarks: %zu)", 
                                     keyframeIdx, markerId, landmarkMap_.size());
                    }
                } else {
                    // New landmark - initialize and add observation
                    landmarkKey = L(landmarkMap_.size());
                    landmarkMap_[markerId] = landmarkKey;

                    // Initialize landmark pose in map frame
                    // T_map_marker = T_map_base * T_base_marker
                    gtsam::Pose3 landmarkPoseMap = currentBasePose.compose(markerObsBase);
                    
                    graphValues_.insert(landmarkKey, landmarkPoseMap);

                    // Add observation factor
                    graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                        X(keyframeIdx), landmarkKey, markerObsBase, arucoObsNoise_));

                    newLandmarksAdded++;
                    RCLCPP_INFO(get_logger(), "[GraphOpt] Added new landmark ID=%d at keyframe %d (total landmarks: %zu, pose: [%.3f, %.3f, %.3f])", 
                                 markerId, keyframeIdx, landmarkMap_.size(),
                                 landmarkPoseMap.translation().x(),
                                 landmarkPoseMap.translation().y(),
                                 landmarkPoseMap.translation().z());
                }
        }
        
        if (enableTopicDebugLog && markersProcessed > 0) {
            RCLCPP_INFO(get_logger(), "[GraphOpt] Processed %zu markers at keyframe %d: %zu new landmarks, %zu loop closures", 
                         markersProcessed, keyframeIdx, newLandmarksAdded, loopClosuresAdded);
        }
    }

    void publishOdometry(const gtsam::Pose3& pose, const rclcpp::Time& stamp) {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = odomFrame;  // Changed: odom->base_link (SLAM estimate)
        odom.child_frame_id = baseLinkFrame;

        odom.pose.pose = gtsamToPoseMsg(pose);
        
        // Set covariance (simplified)
        for (int i = 0; i < 36; i++) {
            odom.pose.covariance[i] = 0.0;
        }
        odom.pose.covariance[0] = 0.01;   // x
        odom.pose.covariance[7] = 0.01;   // y
        odom.pose.covariance[14] = 0.01;  // z
        odom.pose.covariance[21] = 0.01;  // roll
        odom.pose.covariance[28] = 0.01;  // pitch
        odom.pose.covariance[35] = 0.01;  // yaw

        pubGlobalOdometry_->publish(odom);
    }

    void publishTF(const gtsam::Pose3& pose, const rclcpp::Time& stamp) {
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.stamp = stamp;
        transformStamped.header.frame_id = odomFrame;  // odom->base_link (SLAM estimate)
        transformStamped.child_frame_id = baseLinkFrame;

        transformStamped.transform.translation.x = pose.translation().x();
        transformStamped.transform.translation.y = pose.translation().y();
        transformStamped.transform.translation.z = pose.translation().z();

        auto q = pose.rotation().toQuaternion();
        transformStamped.transform.rotation.w = q.w();
        transformStamped.transform.rotation.x = q.x();
        transformStamped.transform.rotation.y = q.y();
        transformStamped.transform.rotation.z = q.z();

        tfBroadcaster_->sendTransform(transformStamped);
        
        if (enableTopicDebugLog) {
            RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 1000,
                "[GraphOpt] Published TF: %s -> %s, pose: [%.3f, %.3f, %.3f]",
                odomFrame.c_str(), baseLinkFrame.c_str(),
                pose.translation().x(), pose.translation().y(), pose.translation().z());
        }
    }

    void updatePath(const gtsam::Pose3& pose, const rclcpp::Time& stamp) {
        geometry_msgs::msg::PoseStamped poseStamped;
        poseStamped.header.stamp = stamp;
        poseStamped.header.frame_id = mapFrame;
        poseStamped.pose = gtsamToPoseMsg(pose);

        globalPath_.header.stamp = stamp;
        globalPath_.header.frame_id = mapFrame;
        globalPath_.poses.push_back(poseStamped);

        // Keep path size reasonable
        if (globalPath_.poses.size() > 1000) {
            globalPath_.poses.erase(globalPath_.poses.begin());
        }

        pubPath_->publish(globalPath_);
    }
};

} // namespace aruco_slam_ailab

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);

    auto node = std::make_shared<aruco_slam_ailab::GraphOptimizer>(options);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Graph Optimizer Started.\033[0m");

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
