#include "aruco_slam_ailab/utility.hpp"
#include "aruco_slam_ailab/msg/marker_observation.hpp"
#include "aruco_slam_ailab/msg/marker_array.hpp"
#include "aruco_slam_ailab/msg/optimized_keyframe_state.hpp"

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Pose3.h>

#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <set>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <mutex>
#include <chrono>

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
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLandmarks_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLandmarksOdom_;  // odom 프레임, 맵 저장용
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubKeyframes_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubOdometryEdges_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLoopClosureEdges_;
    rclcpp::Publisher<aruco_slam_ailab::msg::OptimizedKeyframeState>::SharedPtr pubOptimizedKeyframeState_;

    // TF (독립 타이머로 발행 → SLAM 연산 지연 시에도 RViz 끊김 방지)
    std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;
    rclcpp::CallbackGroup::SharedPtr tfCallbackGroup_;
    rclcpp::TimerBase::SharedPtr tfTimer_;
    std::mutex poseMtx_;
    gtsam::Pose3 currentEstimatePose_;

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

    // 관측 엣지 (키프레임–랜드마크): 루프 클로저 시각화용
    std::vector<std::pair<int, gtsam::Key>> observationEdges_;

    // Keyframe tracking
    double lastKeyframeTime_ = -1.0;
    gtsam::Pose3 lastKeyframePose_;           // 마지막 키프레임의 최적화된 포즈 (Vision 보정 반영)
    gtsam::Pose3 imuPoseAtLastKeyframe_;      // 마지막 키프레임 시점의 IMU 오도메트리 포즈 (키프레임 간 보간용)
    std::set<int> markerIdsAtLastKeyframe_;   // 마커 등장/퇴장 감지용

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
    gtsam::Pose3 baseToDepthCam_;  // base_link -> camera_depth_optical_frame (RViz 랜드마크 시각화용)
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
        pubLandmarks_ = create_publisher<visualization_msgs::msg::MarkerArray>("/slam/landmarks", 10);
        pubLandmarksOdom_ = create_publisher<visualization_msgs::msg::MarkerArray>("/slam/landmarks_odom", 10);  // 맵 저장용: odom 프레임
        pubKeyframes_ = create_publisher<visualization_msgs::msg::MarkerArray>("/slam/keyframes", 10);
        pubOdometryEdges_ = create_publisher<visualization_msgs::msg::MarkerArray>("/slam/odometry_edges", 10);
        pubLoopClosureEdges_ = create_publisher<visualization_msgs::msg::MarkerArray>("/slam/loop_closure_edges", 10);
        if (useImu) {
            pubOptimizedKeyframeState_ = create_publisher<aruco_slam_ailab::msg::OptimizedKeyframeState>(
                "/optimized_keyframe_state", 10);
        }

        // TF broadcaster
        tfBroadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
        currentEstimatePose_ = gtsam::Pose3();

        // TF 발행 전용 타이머 (50Hz) — SLAM 연산과 분리해 RViz 대기열 끊김 방지
        tfCallbackGroup_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        tfTimer_ = create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&GraphOptimizer::tfTimerCallback, this),
            tfCallbackGroup_);

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

        gtsam::Rot3 rot_base_depth_cam(extRotBaseDepthCam);
        gtsam::Point3 trans_base_depth_cam(extTransBaseDepthCam.x(), extTransBaseDepthCam.y(), extTransBaseDepthCam.z());
        baseToDepthCam_ = gtsam::Pose3(rot_base_depth_cam, trans_base_depth_cam);

        // baseToCam_ = T_base_camera_color_optical (TF 직접 사용) → ArUco 관측은 이미 optical frame
        opticalToCameraLink_ = gtsam::Pose3();  // identity (추가 변환 없음)

        RCLCPP_INFO(get_logger(), "Extrinsics: base->camera_gyro_optical trans=(%.3f, %.3f, %.3f)",
                    extTransBaseImu.x(), extTransBaseImu.y(), extTransBaseImu.z());
        RCLCPP_INFO(get_logger(), "         base->camera_color_optical trans=(%.3f, %.3f, %.3f)",
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

        // LIO-SAM style: 키프레임만 노드, 키프레임 사이는 IMU factor + ArUco(키프레임 시점) factor
        if (systemInitialized_) {
            bool keyframeAdded = processKeyframes();
            if (!keyframeAdded) {
                // 키프레임 간: 마지막 최적화 포즈 + IMU 상대 운동으로 보간 (원시 IMU만 쓰면 드리프트 누적)
                nav_msgs::msg::Odometry latest = imuOdomQueue_.back();
                gtsam::Pose3 currentImuPose = poseMsgToGtsam(latest.pose.pose);
                gtsam::Pose3 poseToPublish;
                if (lastKeyframeTime_ >= 0) {
                    gtsam::Pose3 delta = imuPoseAtLastKeyframe_.inverse().compose(currentImuPose);
                    poseToPublish = lastKeyframePose_.compose(delta);
                } else {
                    poseToPublish = currentImuPose;
                }
                publishOdometry(poseToPublish, latest.header.stamp);
                { std::lock_guard<std::mutex> lock(poseMtx_); currentEstimatePose_ = poseToPublish; }
            }
        } else {
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
        
        // Between factor: use IMU delta so trajectory follows dead-reckoning (IMU-only 시 궤적이 끊기지 않음)
        gtsam::Pose3 prevImuPose = prevPose_.compose(baseToImu_);
        gtsam::Pose3 deltaImuPose = prevImuPose.between(imuPose);
        graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
            X(frameIdx_), X(frameIdx_ + 1), deltaImuPose, wheelOdomNoise_));
        
        // Add initial estimate for next frame (like aruco-slam line 172-175)
        // Use current pose from IMU odometry as initial estimate
        graphValues_.insert(X(frameIdx_ + 1), imuPose);
        graphValues_.insert(V(frameIdx_ + 1), vel);
        graphValues_.insert(B(frameIdx_ + 1), prevBias_);
        
        // Optimize every frame (like aruco-slam: don't optimize on first iteration)
        if (frameIdx_ == 0) {
            // First iteration: just set current estimate (like aruco-slam line 149-150)
            // Values are already inserted above
            frameIdx_++;
        } else {
            try {
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
                { std::lock_guard<std::mutex> lock(poseMtx_); currentEstimatePose_ = optimizedBasePose; }
                updatePath(optimizedBasePose, latestImuOdom.header.stamp);
                publishLandmarks(latestImuOdom.header.stamp);
                publishGraphVisualization(latestImuOdom.header.stamp);
                
                // Update keyframe tracking for visualization
                lastKeyframeTime_ = currentTime;
                lastKeyframePose_ = optimizedBasePose;
                frameIdx_++;
            } catch (const std::exception& e) {
                // 예외 시 퍼블리시 끊기지 않도록 IMU 포즈로 폴백; frameIdx_ 유지해 다음 콜백에서 재시도
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                    "[GraphOpt] ISAM2 update/estimate exception (frame=%d): %s; publishing IMU pose",
                    frameIdx_ + 1, e.what());
                graphFactors_.resize(0);
                graphValues_.clear();
                prevPose_ = currentPose;
                prevVel_ = vel;
                publishOdometry(currentPose, latestImuOdom.header.stamp);
                { std::lock_guard<std::mutex> lock(poseMtx_); currentEstimatePose_ = currentPose; }
                updatePath(currentPose, latestImuOdom.header.stamp);
                lastKeyframeTime_ = currentTime;
                lastKeyframePose_ = currentPose;
                frameIdx_++;  // 예외 후에도 진행해 그래프가 끊기지 않도록
            }
        }
    }
    
    void processFrameVisionOnly(const aruco_slam_ailab::msg::MarkerArray& markerArray) {
        if (markerArray.markers.empty()) {
            return;
        }

        // Vision-Only: 키프레임 정책 적용 (ArUco 30Hz 전부 그래프에 넣지 않도록)
        nav_msgs::msg::Odometry dummyOdom;
        dummyOdom.header = markerArray.header;
        dummyOdom.pose.pose = gtsamToPoseMsg(prevPose_);
        std::set<int> currentMarkerIds;
        for (const auto& m : markerArray.markers) {
            currentMarkerIds.insert(m.id);
        }
        if (!shouldCreateKeyframe(dummyOdom, currentMarkerIds)) {
            return;
        }

        double currentTime = stamp2Sec(markerArray.header.stamp);

        gtsam::Pose3 currentCameraPose = prevPose_.compose(baseToCam_);

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
            lastKeyframeTime_ = currentTime;
            lastKeyframePose_ = prevPose_;
        } else {
            isam_.update(graphFactors_, graphValues_);
            graphFactors_.resize(0);
            graphValues_.clear();
            
            gtsam::Values result = isam_.calculateEstimate();
            gtsam::Pose3 optimizedImuPose = result.at<gtsam::Pose3>(X(frameIdx_ + 1));
            gtsam::Pose3 optimizedBasePose = optimizedImuPose.compose(baseToImu_.inverse());
            
            gtsam::Vector3 optimizedVel = result.at<gtsam::Vector3>(V(frameIdx_ + 1));
            prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(frameIdx_ + 1));
            
            prevPose_ = optimizedBasePose;
            prevVel_ = optimizedVel;
            
            rclcpp::Time stamp = markerArray.header.stamp;
            publishOdometry(optimizedBasePose, stamp);
            { std::lock_guard<std::mutex> lock(poseMtx_); currentEstimatePose_ = optimizedBasePose; }
            updatePath(optimizedBasePose, stamp);
            publishLandmarks(stamp);
            publishGraphVisualization(stamp);
            
            lastKeyframeTime_ = currentTime;
            lastKeyframePose_ = optimizedBasePose;
        }
        markerIdsAtLastKeyframe_ = currentMarkerIds;

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

    // 키프레임 정책: 최소 간격(쓰로틀) + 최대 간격(keep-alive) + 움직임 + 마커 변화
    // 데드락 제거: 시간 체크에서 먼저 리턴하지 않고, 최소 간격 통과 후에만 다른 조건 검사
    bool shouldCreateKeyframe(const nav_msgs::msg::Odometry& odom,
                              const std::set<int>& currentMarkerIds) {
        double currentTime = stamp2Sec(odom.header.stamp);

        if (lastKeyframeTime_ < 0) {
            return true;  // 1. 첫 프레임은 무조건 키프레임
        }

        double timeSince = currentTime - lastKeyframeTime_;

        // 2. [Hard Limit] 최소 시간 간격 (예: 0.2초 = 5Hz) → 키프레임 폭주 방지
        const double keyframeMinInterval = 0.2;
        if (timeSince < keyframeMinInterval) {
            return false;
        }

        // 3. [Keep Alive] 최대 시간 간격 경과 시 그래프 연결 유지를 위해 생성
        if (timeSince >= keyframeTimeInterval) {
            return true;
        }

        // 4. [Motion] 움직임 체크
        if (keyframeIdx_ > 0) {
            gtsam::Pose3 currentPose = poseMsgToGtsam(odom.pose.pose);
            gtsam::Pose3 delta = lastKeyframePose_.between(currentPose);
            double dist = delta.translation().norm();
            double angle = std::abs(delta.rotation().axisAngle().second);
            if (dist >= keyframeDistanceThreshold || angle >= keyframeAngleThreshold) {
                return true;
            }
        }

        // 5. [Context] 마커 등장/퇴장 시 위치 보정을 위해 키프레임 추가
        for (int id : currentMarkerIds) {
            if (markerIdsAtLastKeyframe_.count(id) == 0) {
                return true;  // 새 마커 등장
            }
        }
        for (int id : markerIdsAtLastKeyframe_) {
            if (currentMarkerIds.count(id) == 0) {
                return true;  // 기존 마커 퇴장
            }
        }

        return false;
    }

    // LIO-SAM style: 키프레임만 노드. 키프레임 간 엣지 = IMU between factor + 키프레임 시점 ArUco 랜드마크 factor.
    // 반환: 키프레임이 추가되었으면 true.
    bool processKeyframes() {
        if (imuOdomQueue_.empty()) {
            return false;
        }

        nav_msgs::msg::Odometry latestImuOdom = imuOdomQueue_.back();
        double keyframeTime = stamp2Sec(latestImuOdom.header.stamp);

        // 현재 시각에 가장 가까운 ArUco 관측에서 마커 ID 집합 구하기 (마커 등장/퇴장 판단용)
        std::set<int> currentMarkerIds;
        const aruco_slam_ailab::msg::MarkerArray* closestMarkerArray = nullptr;
        double minTimeDiff = std::numeric_limits<double>::max();
        for (const auto& markerArray : arucoQueue_) {
            double obsTime = stamp2Sec(markerArray.header.stamp);
            double timeDiff = std::abs(obsTime - keyframeTime);
            if (timeDiff < minTimeDiff && timeDiff <= 0.2) {
                minTimeDiff = timeDiff;
                closestMarkerArray = &markerArray;
            }
        }
        if (closestMarkerArray) {
            for (const auto& m : closestMarkerArray->markers) {
                currentMarkerIds.insert(m.id);
            }
        }

        if (!shouldCreateKeyframe(latestImuOdom, currentMarkerIds)) {
            if (enableTopicDebugLog && systemInitialized_) {
                double currentTime = keyframeTime;
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
            return false;
        }

        gtsam::Pose3 keyframePose = poseMsgToGtsam(latestImuOdom.pose.pose);

        if (!systemInitialized_) {
            initializeSystem(gtsam::Pose3(), latestImuOdom);
            lastKeyframeTime_ = keyframeTime;
            lastKeyframePose_ = prevPose_;
            imuPoseAtLastKeyframe_ = keyframePose;
            markerIdsAtLastKeyframe_ = currentMarkerIds;
            return true;
        }

        addKeyframe(keyframeTime, keyframePose, latestImuOdom);
        lastKeyframeTime_ = keyframeTime;
        lastKeyframePose_ = prevPose_;              // 최적화된 포즈 사용 (Vision 보정 반영)
        imuPoseAtLastKeyframe_ = keyframePose;     // 이 시점의 IMU 포즈 저장 (다음 키프레임까지 보간용)
        markerIdsAtLastKeyframe_ = currentMarkerIds;
        return true;
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
        
        // 초기 pose 고정: 항상 (0, 0, 0) 위치 + identity 자세 (0 0 0 0 0 0 0)
        gtsam::Pose3 initialBasePose = gtsam::Pose3();
        
        RCLCPP_INFO(get_logger(), "[GraphOpt] Initializing with fixed initial pose (0, 0, 0, 0, 0, 0)");
        initializeSystem(initialBasePose, latestImuOdom);
    }

    void initializeSystem(const gtsam::Pose3& initialPose, const nav_msgs::msg::Odometry& odom) {
        // Initial pose in base_link frame
        prevPose_ = initialPose;
        { std::lock_guard<std::mutex> lock(poseMtx_); currentEstimatePose_ = initialPose; }
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

    // LIO-SAM style: 키프레임-키프레임 간 엣지 = IMU로 추정한 상대 포즈 (between factor) + 이 키프레임 시점 ArUco 랜드마크 factor.
    void addKeyframe(double time, const gtsam::Pose3& keyframePose, const nav_msgs::msg::Odometry& odom) {
        gtsam::Pose3 imuPose = keyframePose.compose(baseToImu_);
        gtsam::Vector3 vel(odom.twist.twist.linear.x,
                           odom.twist.twist.linear.y,
                           odom.twist.twist.linear.z);

        // 키프레임 구간 IMU 누적 상대 포즈 (last keyframe -> this keyframe)
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

        // IMU 노드에 최적화 결과 전달: 다음 구간 적분을 이 포즈/바이어스로 초기화 (따로 노는 것 방지)
        if (pubOptimizedKeyframeState_) {
            aruco_slam_ailab::msg::OptimizedKeyframeState msg;
            msg.header.stamp = odom.header.stamp;
            msg.header.frame_id = odomFrame;
            msg.pose = gtsamToPoseMsg(optimizedBasePose);
            msg.velocity.x = optimizedVel.x();
            msg.velocity.y = optimizedVel.y();
            msg.velocity.z = optimizedVel.z();
            gtsam::Vector6 b = prevBias_.vector();
            msg.bias.resize(6);
            for (int i = 0; i < 6; i++) msg.bias[i] = b(i);
            pubOptimizedKeyframeState_->publish(msg);
        }

        // Publish results
        publishOdometry(optimizedBasePose, odom.header.stamp);
        { std::lock_guard<std::mutex> lock(poseMtx_); currentEstimatePose_ = optimizedBasePose; }
        updatePath(optimizedBasePose, odom.header.stamp);
        publishLandmarks(odom.header.stamp);
        publishGraphVisualization(odom.header.stamp);

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

                // X(keyframeIdx) is in IMU frame; L(landmark) is in odom frame.
                // BetweenFactor(X, L, rel) means L = X * rel => rel = X^{-1}*L = T_imu_marker.
                gtsam::Pose3 markerObsImu = baseToImu_.inverse().compose(markerObsBase);

                // Check if landmark exists
                gtsam::Key landmarkKey;
                if (landmarkMap_.find(markerId) != landmarkMap_.end()) {
                    // Existing landmark - add observation factor (loop closure)
                    landmarkKey = landmarkMap_[markerId];
                    
                    graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                        X(keyframeIdx), landmarkKey, markerObsImu, arucoObsNoise_));
                    observationEdges_.emplace_back(keyframeIdx, landmarkKey);

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

                    // Add observation factor (same frame as X: IMU)
                    graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                        X(keyframeIdx), landmarkKey, markerObsImu, arucoObsNoise_));
                    observationEdges_.emplace_back(keyframeIdx, landmarkKey);

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

    void tfTimerCallback() {
        gtsam::Pose3 poseToSend;
        {
            std::lock_guard<std::mutex> lock(poseMtx_);
            poseToSend = currentEstimatePose_;
        }
        rclcpp::Time stampToSend = get_clock()->now();

        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.stamp = stampToSend;
        transformStamped.header.frame_id = odomFrame;
        transformStamped.child_frame_id = baseLinkFrame;
        transformStamped.transform.translation.x = poseToSend.translation().x();
        transformStamped.transform.translation.y = poseToSend.translation().y();
        transformStamped.transform.translation.z = poseToSend.translation().z();
        auto q = poseToSend.rotation().toQuaternion();
        transformStamped.transform.rotation.w = q.w();
        transformStamped.transform.rotation.x = q.x();
        transformStamped.transform.rotation.y = q.y();
        transformStamped.transform.rotation.z = q.z();
        tfBroadcaster_->sendTransform(transformStamped);
    }

    void updatePath(const gtsam::Pose3& pose, const rclcpp::Time& stamp) {
        geometry_msgs::msg::PoseStamped poseStamped;
        poseStamped.header.stamp = stamp;
        poseStamped.header.frame_id = odomFrame;  // SLAM 세계 좌표 = odom (odom->base_link TF와 일치)
        poseStamped.pose = gtsamToPoseMsg(pose);

        globalPath_.header.stamp = stamp;
        globalPath_.header.frame_id = odomFrame;
        globalPath_.poses.push_back(poseStamped);

        // Keep path size reasonable
        if (globalPath_.poses.size() > 1000) {
            globalPath_.poses.erase(globalPath_.poses.begin());
        }

        pubPath_->publish(globalPath_);
    }

    void publishLandmarks(const rclcpp::Time& stamp) {
        visualization_msgs::msg::MarkerArray arr;
        visualization_msgs::msg::MarkerArray arrOdom;
        if (landmarkMap_.empty()) {
            pubLandmarks_->publish(arr);
            pubLandmarksOdom_->publish(arrOdom);
            return;
        }
        gtsam::Values est;
        try {
            est = isam_.calculateEstimate();
        } catch (...) {
            pubLandmarks_->publish(arr);
            pubLandmarksOdom_->publish(arrOdom);
            return;
        }
        // RViz: ArUco 관측과 동일한 camera_color_optical_frame 사용 → /aruco_markers와 겹쳐 보이도록
        const std::string vis_frame_id = cameraFrame;
        gtsam::Pose3 odomToBase = prevPose_;
        gtsam::Pose3 odomToVis = odomToBase.compose(baseToCam_);
        gtsam::Pose3 visToOdom = odomToVis.inverse();

        for (const auto& [markerId, landmarkKey] : landmarkMap_) {
            if (!est.exists(landmarkKey)) continue;
            gtsam::Pose3 poseOdom = est.at<gtsam::Pose3>(landmarkKey);

            // ---- /slam/landmarks: vis_frame (기본: color = ArUco와 동일 프레임)
            gtsam::Pose3 poseInVis = visToOdom.compose(poseOdom);
            geometry_msgs::msg::Pose poseMsgVis = gtsamToPoseMsg(poseInVis);
            visualization_msgs::msg::Marker cube;
            cube.header.stamp = stamp;
            cube.header.frame_id = vis_frame_id;
            cube.ns = "landmarks";
            cube.id = markerId;
            cube.type = visualization_msgs::msg::Marker::CUBE;
            cube.action = visualization_msgs::msg::Marker::ADD;
            cube.pose = poseMsgVis;
            cube.scale.x = 0.28;
            cube.scale.y = 0.28;
            cube.scale.z = 0.04;
            cube.color.r = 0.2f;
            cube.color.g = 0.6f;
            cube.color.b = 1.0f;
            cube.color.a = 0.9f;
            cube.lifetime = rclcpp::Duration(0, 0);
            arr.markers.push_back(cube);
            visualization_msgs::msg::Marker text;
            text.header.stamp = stamp;
            text.header.frame_id = vis_frame_id;
            text.ns = "landmark_ids";
            text.id = markerId;
            text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text.action = visualization_msgs::msg::Marker::ADD;
            text.pose = poseMsgVis;
            text.pose.position.z += 0.18;
            text.scale.z = 0.42;
            text.color.r = 0.0f;
            text.color.g = 1.0f;
            text.color.b = 0.4f;
            text.color.a = 1.0f;
            text.text = std::to_string(markerId);
            text.lifetime = rclcpp::Duration(0, 0);
            arr.markers.push_back(text);

            // ---- /slam/landmarks_odom: odom 프레임 (맵 저장용, camera_color_optical_frame에서 관측했지만 월드는 odom)
            geometry_msgs::msg::Pose poseMsgOdom = gtsamToPoseMsg(poseOdom);
            visualization_msgs::msg::Marker cubeOdom;
            cubeOdom.header.stamp = stamp;
            cubeOdom.header.frame_id = odomFrame;
            cubeOdom.ns = "landmarks";
            cubeOdom.id = markerId;
            cubeOdom.type = visualization_msgs::msg::Marker::CUBE;
            cubeOdom.action = visualization_msgs::msg::Marker::ADD;
            cubeOdom.pose = poseMsgOdom;
            cubeOdom.scale.x = 0.28;
            cubeOdom.scale.y = 0.28;
            cubeOdom.scale.z = 0.04;
            cubeOdom.color.r = 0.2f;
            cubeOdom.color.g = 0.6f;
            cubeOdom.color.b = 1.0f;
            cubeOdom.color.a = 0.9f;
            cubeOdom.lifetime = rclcpp::Duration(0, 0);
            arrOdom.markers.push_back(cubeOdom);
            visualization_msgs::msg::Marker textOdom;
            textOdom.header.stamp = stamp;
            textOdom.header.frame_id = odomFrame;
            textOdom.ns = "landmark_ids";
            textOdom.id = markerId;
            textOdom.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            textOdom.action = visualization_msgs::msg::Marker::ADD;
            textOdom.pose = poseMsgOdom;
            textOdom.pose.position.z += 0.18;
            textOdom.scale.z = 0.42;
            textOdom.color.r = 0.0f;
            textOdom.color.g = 1.0f;
            textOdom.color.b = 0.4f;
            textOdom.color.a = 1.0f;
            textOdom.text = std::to_string(markerId);
            textOdom.lifetime = rclcpp::Duration(0, 0);
            arrOdom.markers.push_back(textOdom);
        }
        pubLandmarks_->publish(arr);
        pubLandmarksOdom_->publish(arrOdom);
    }

    void publishGraphVisualization(const rclcpp::Time& stamp) {
        gtsam::Values est;
        try {
            est = isam_.calculateEstimate();
        } catch (...) {
            return;
        }

        // Pose 노드 개수: X(0), X(1), ... 존재하는 최대 인덱스+1
        int numPoseNodes = 0;
        while (est.exists(X(numPoseNodes))) {
            numPoseNodes++;
        }
        if (numPoseNodes == 0) return;

        const gtsam::Pose3 imuToBase = baseToImu_.inverse();

        // ----- 1) 키프레임 노드 (로봇이 있었던 시점의 위치: 구체)
        visualization_msgs::msg::MarkerArray keyframeArr;
        for (int i = 0; i < numPoseNodes; i++) {
            gtsam::Pose3 imuPose = est.at<gtsam::Pose3>(X(i));
            gtsam::Pose3 basePose = imuPose.compose(imuToBase);
            gtsam::Point3 t = basePose.translation();

            visualization_msgs::msg::Marker node;
            node.header.stamp = stamp;
            node.header.frame_id = odomFrame;
            node.ns = "keyframes";
            node.id = i;
            node.type = visualization_msgs::msg::Marker::SPHERE;
            node.action = visualization_msgs::msg::Marker::ADD;
            node.pose.position.x = t.x();
            node.pose.position.y = t.y();
            node.pose.position.z = t.z();
            node.pose.orientation.w = 1.0;
            node.scale.x = node.scale.y = node.scale.z = 0.08;
            node.color.r = 0.0f;
            node.color.g = 0.8f;
            node.color.b = 0.0f;
            node.color.a = 0.9f;
            node.lifetime = rclcpp::Duration(0, 0);
            keyframeArr.markers.push_back(node);
        }
        pubKeyframes_->publish(keyframeArr);

        // ----- 2) 오도메트리 엣지 (인접 노드 사이 선)
        visualization_msgs::msg::MarkerArray odomArr;
        visualization_msgs::msg::Marker lineOdom;
        lineOdom.header.stamp = stamp;
        lineOdom.header.frame_id = odomFrame;
        lineOdom.ns = "odometry_edges";
        lineOdom.id = 0;
        lineOdom.type = visualization_msgs::msg::Marker::LINE_LIST;
        lineOdom.action = visualization_msgs::msg::Marker::ADD;
        lineOdom.scale.x = 0.03;
        lineOdom.color.r = 0.2f;
        lineOdom.color.g = 0.6f;
        lineOdom.color.b = 1.0f;
        lineOdom.color.a = 0.9f;
        lineOdom.lifetime = rclcpp::Duration(0, 0);
        for (int i = 0; i + 1 < numPoseNodes; i++) {
            gtsam::Point3 p0 = est.at<gtsam::Pose3>(X(i)).compose(imuToBase).translation();
            gtsam::Point3 p1 = est.at<gtsam::Pose3>(X(i + 1)).compose(imuToBase).translation();
            geometry_msgs::msg::Point pt0, pt1;
            pt0.x = p0.x(); pt0.y = p0.y(); pt0.z = p0.z();
            pt1.x = p1.x(); pt1.y = p1.y(); pt1.z = p1.z();
            lineOdom.points.push_back(pt0);
            lineOdom.points.push_back(pt1);
        }
        if (!lineOdom.points.empty()) {
            odomArr.markers.push_back(lineOdom);
        }
        pubOdometryEdges_->publish(odomArr);

        // ----- 3) 루프 클로저(관측) 엣지 (키프레임–랜드마크 선)
        visualization_msgs::msg::MarkerArray loopArr;
        visualization_msgs::msg::Marker lineLoop;
        lineLoop.header.stamp = stamp;
        lineLoop.header.frame_id = odomFrame;
        lineLoop.ns = "loop_closure_edges";
        lineLoop.id = 0;
        lineLoop.type = visualization_msgs::msg::Marker::LINE_LIST;
        lineLoop.action = visualization_msgs::msg::Marker::ADD;
        lineLoop.scale.x = 0.02;
        lineLoop.color.r = 1.0f;
        lineLoop.color.g = 0.4f;
        lineLoop.color.b = 0.0f;
        lineLoop.color.a = 0.7f;
        lineLoop.lifetime = rclcpp::Duration(0, 0);
        for (const auto& [kfIdx, lkKey] : observationEdges_) {
            if (!est.exists(X(kfIdx)) || !est.exists(lkKey)) continue;
            gtsam::Point3 pk = est.at<gtsam::Pose3>(X(kfIdx)).compose(imuToBase).translation();
            gtsam::Point3 pl = est.at<gtsam::Pose3>(lkKey).translation();
            geometry_msgs::msg::Point pt0, pt1;
            pt0.x = pk.x(); pt0.y = pk.y(); pt0.z = pk.z();
            pt1.x = pl.x(); pt1.y = pl.y(); pt1.z = pl.z();
            lineLoop.points.push_back(pt0);
            lineLoop.points.push_back(pt1);
        }
        if (!lineLoop.points.empty()) {
            loopArr.markers.push_back(lineLoop);
        }
        pubLoopClosureEdges_->publish(loopArr);
    }
};

} // namespace aruco_slam_ailab

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);

    auto node = std::make_shared<aruco_slam_ailab::GraphOptimizer>(options);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Graph Optimizer Started (MultiThreadedExecutor).\033[0m");

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
