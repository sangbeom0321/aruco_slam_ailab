#include "aruco_sam_ailab/utility.hpp"
#include "aruco_sam_ailab/msg/marker_array.hpp"
#include "aruco_sam_ailab/msg/optimized_keyframe_state.hpp"
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuBias.h>

#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <mutex>
#include <deque>
#include <set>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <chrono>

using gtsam::symbol_shorthand::X; // Robot Pose (base_link in map frame)
using gtsam::symbol_shorthand::L; // Landmark Pose (map frame)
using gtsam::symbol_shorthand::V; // Velocity (map frame)
using gtsam::symbol_shorthand::B; // IMU Bias

namespace aruco_sam_ailab {

class GraphOptimizer : public ParamServer {
public:
    // ═══ Mutexes ═══
    // queueMtx_: protects IMU queue (very fast, microseconds)
    // slamMtx_: protects ISAM2 + graph state (held ~100ms during optimization)
    // pendingMtx_: protects latest ArUco observation for timer
    // snapMtx_: protects SLAM snapshot for interpolation
    std::mutex queueMtx_;
    std::mutex slamMtx_;
    std::mutex pendingMtx_;
    std::mutex snapMtx_;

    // ═══ Core ISAM2 (guarded by slamMtx_) ═══
    gtsam::ISAM2 isam_;
    gtsam::NonlinearFactorGraph graphFactors_;
    gtsam::Values graphValues_;

    // ═══ State (guarded by slamMtx_) ═══
    gtsam::Pose3 baseToCam_;  // const after constructor
    gtsam::Pose3 currentEstimate_;
    gtsam::Vector3 currentVelocity_ = gtsam::Vector3::Zero();
    gtsam::imuBias::ConstantBias currentBias_;
    bool systemInitialized_ = false;
    int frameIdx_ = 0;

    // ═══ Keyframe Selection (guarded by slamMtx_) ═══
    rclcpp::Time lastKeyframeTime_{0, 0, RCL_ROS_TIME};
    std::set<int> lastVisibleMarkers_;
    double keyframeTimeThresh_;

    // ═══ Raw IMU queue (guarded by queueMtx_) ═══
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subRawImu_;
    std::deque<sensor_msgs::msg::Imu> rawImuQueue_;
    Eigen::Vector3d lpfAcc_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d lpfGyr_ = Eigen::Vector3d::Zero();
    bool lpfInitialized_ = false;
    boost::shared_ptr<gtsam::PreintegrationParams> imuParams_;
    gtsam::PreintegratedImuMeasurements* imuPreintegrator_ = nullptr;  // guarded by slamMtx_

    // ═══ Wheel Odom queue (guarded by queueMtx_) ═══
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subWheelOdom_;
    std::deque<nav_msgs::msg::Odometry> wheelOdomQueue_;
    gtsam::noiseModel::Diagonal::shared_ptr wheelOdomNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr wheelOdomZeroNoise_;  // 정지 시 tight constraint

    // Wheel odom delta 결과 (delta + 정지 여부)
    struct WheelOdomResult {
        gtsam::Pose3 delta;
        bool isStationary;
    };

    // ═══ Pending ArUco (guarded by pendingMtx_) ═══
    aruco_sam_ailab::msg::MarkerArray pendingMarkers_;  // already in base_link
    rclcpp::Time pendingStamp_{0, 0, RCL_ROS_TIME};
    bool pendingArucoValid_ = false;

    // ═══ SLAM Snapshot for interpolation (guarded by snapMtx_) ═══
    gtsam::Pose3 snapEstimate_;
    bool snapValid_ = false;

    // ═══ Wheel Odom Repropagation (guarded by snapMtx_) ═══
    gtsam::Pose3 slamPoseAtLastKF_;   // SLAM 보정 pose at last keyframe
    gtsam::Pose3 odomPoseAtLastKF_;   // Wheel odom pose at last keyframe
    bool wheelOdomRepropValid_ = false;

    // ═══ Landmarks (guarded by slamMtx_) ═══
    std::map<int, gtsam::Key> landmarkIdToKey_;

    // ═══ ROS Interface ═══
    rclcpp::Subscription<aruco_sam_ailab::msg::MarkerArray>::SharedPtr subAruco_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubGlobalOdom_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLandmarks_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubDebugMap_;
    rclcpp::Publisher<aruco_sam_ailab::msg::OptimizedKeyframeState>::SharedPtr pubOptimizedState_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr saveLandmarksSrv_;
    std::string landmarksSavePath_;
    rclcpp::TimerBase::SharedPtr slamTimer_;

    // ═══ Callback Groups (enable true concurrency with MultiThreadedExecutor) ═══
    rclcpp::CallbackGroup::SharedPtr queueCbGroup_;   // IMU subscriber
    rclcpp::CallbackGroup::SharedPtr arucoCbGroup_;    // ArUco subscriber (lightweight)
    rclcpp::CallbackGroup::SharedPtr timerCbGroup_;    // SLAM timer (heavy ISAM2)

    // ═══ Mode ═══
    bool isLocalizationMode_ = false;
    std::string mapPath_;
    gtsam::noiseModel::Diagonal::shared_ptr fixedLandmarkNoise_;

    nav_msgs::msg::Path globalPath_;
    nav_msgs::msg::Path correctedOdomPath_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;

    // ═══ Wheel Odom Repropagation Publishers ═══
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubCorrectedOdom_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubCorrectedPath_;

    // ═══ Noise Models ═══
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr biasRandomWalkNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr obsNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr landmarkPriorNoise_;

    // ═══ ArUco observation filter params ═══
    double maxArucoRange_;       // 최대 관측 거리 (m), 초과 시 스킵
    double minViewingAngle_;     // 최소 viewing angle (rad), 정면 관측 스킵

    // ═══════════════════════════════════════════════════════════
    //  Constructor
    // ═══════════════════════════════════════════════════════════
    GraphOptimizer(const rclcpp::NodeOptions& options) : ParamServer("graph_optimizer", options) {

        // ─── 0. Callback Groups (allow concurrent execution) ───
        queueCbGroup_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        arucoCbGroup_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        timerCbGroup_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions queueSubOpts;
        queueSubOpts.callback_group = queueCbGroup_;
        rclcpp::SubscriptionOptions arucoSubOpts;
        arucoSubOpts.callback_group = arucoCbGroup_;

        // ─── 1. Subscribers ───

        // 1a. Raw IMU (for ImuFactor preintegration in factor graph)
        subRawImu_ = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(queueMtx_);
                sensor_msgs::msg::Imu imu_base = *msg;
                if (imuFrame != baseLinkFrame) {
                    Eigen::Vector3d acc(msg->linear_acceleration.x,
                                       msg->linear_acceleration.y,
                                       msg->linear_acceleration.z);
                    Eigen::Vector3d gyr(msg->angular_velocity.x,
                                       msg->angular_velocity.y,
                                       msg->angular_velocity.z);
                    acc = extRotBaseImu * acc;
                    gyr = extRotBaseImu * gyr;
                    imu_base.linear_acceleration.x = acc.x();
                    imu_base.linear_acceleration.y = acc.y();
                    imu_base.linear_acceleration.z = acc.z();
                    imu_base.angular_velocity.x = gyr.x();
                    imu_base.angular_velocity.y = gyr.y();
                    imu_base.angular_velocity.z = gyr.z();
                }
                // Apply low-pass filter to reduce vibration spikes
                if (lpfAlpha < 1.0) {
                    Eigen::Vector3d a(imu_base.linear_acceleration.x,
                                      imu_base.linear_acceleration.y,
                                      imu_base.linear_acceleration.z);
                    Eigen::Vector3d g(imu_base.angular_velocity.x,
                                      imu_base.angular_velocity.y,
                                      imu_base.angular_velocity.z);
                    if (!lpfInitialized_) {
                        lpfAcc_ = a;
                        lpfGyr_ = g;
                        lpfInitialized_ = true;
                    } else {
                        lpfAcc_ = lpfAlpha * a + (1.0 - lpfAlpha) * lpfAcc_;
                        lpfGyr_ = lpfAlpha * g + (1.0 - lpfAlpha) * lpfGyr_;
                    }
                    imu_base.linear_acceleration.x = lpfAcc_.x();
                    imu_base.linear_acceleration.y = lpfAcc_.y();
                    imu_base.linear_acceleration.z = lpfAcc_.z();
                    imu_base.angular_velocity.x = lpfGyr_.x();
                    imu_base.angular_velocity.y = lpfGyr_.y();
                    imu_base.angular_velocity.z = lpfGyr_.z();
                }
                rawImuQueue_.push_back(imu_base);
                if (rawImuQueue_.size() > 4000) rawImuQueue_.pop_front();

                // [DEBUG] base frame gyro 값 확인 (5초마다)
                if (enableTopicDebugLog) {
                    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                        "[IMU raw→base] gyr=(%.5f, %.5f, %.5f) acc=(%.3f, %.3f, %.3f)",
                        imu_base.angular_velocity.x, imu_base.angular_velocity.y, imu_base.angular_velocity.z,
                        imu_base.linear_acceleration.x, imu_base.linear_acceleration.y, imu_base.linear_acceleration.z);
                }
            }, queueSubOpts);

        // 1b. Wheel Odometry (for BetweenFactor + repropagation)
        if (useWheelOdom) {
            subWheelOdom_ = create_subscription<nav_msgs::msg::Odometry>(
                wheelOdomTopic, rclcpp::SensorDataQoS(),
                [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                    // Queue에 push (BetweenFactor 용)
                    {
                        std::lock_guard<std::mutex> lock(queueMtx_);
                        wheelOdomQueue_.push_back(*msg);
                        if (wheelOdomQueue_.size() > 2000) wheelOdomQueue_.pop_front();
                    }
                    // Repropagation: SLAM 보정 pose 기준으로 연속 위치 발행
                    {
                        std::lock_guard<std::mutex> lock(snapMtx_);
                        if (!wheelOdomRepropValid_) return;
                        gtsam::Pose3 currentOdom = poseMsgToGtsam(msg->pose.pose);
                        gtsam::Pose3 delta = odomPoseAtLastKF_.between(currentOdom);
                        gtsam::Pose3 corrected = slamPoseAtLastKF_.compose(delta);

                        nav_msgs::msg::Odometry out;
                        out.header.stamp = msg->header.stamp;
                        out.header.frame_id = odomFrame;
                        out.child_frame_id = baseLinkFrame;
                        out.pose.pose = gtsamToPoseMsg(corrected);
                        pubCorrectedOdom_->publish(out);

                        geometry_msgs::msg::PoseStamped ps;
                        ps.header = out.header;
                        ps.pose = out.pose.pose;
                        correctedOdomPath_.header = out.header;
                        correctedOdomPath_.poses.push_back(ps);
                        pubCorrectedPath_->publish(correctedOdomPath_);
                    }
                }, queueSubOpts);
        }

        // 1c. ArUco (lightweight — never blocks on ISAM2)
        subAruco_ = create_subscription<aruco_sam_ailab::msg::MarkerArray>(
            arucoPosesTopic, 10,
            std::bind(&GraphOptimizer::arucoHandler, this, std::placeholders::_1),
            arucoSubOpts);

        // ─── 2. Publishers ───
        pubGlobalOdom_ = create_publisher<nav_msgs::msg::Odometry>("/aruco_slam/odom", 10);
        pubPath_ = create_publisher<nav_msgs::msg::Path>("/aruco_slam/path", 10);
        pubLandmarks_ = create_publisher<visualization_msgs::msg::MarkerArray>("/aruco_slam/landmarks", 10);
        pubDebugMap_ = create_publisher<visualization_msgs::msg::MarkerArray>("/aruco_slam/debug_markers", 10);
        pubOptimizedState_ = create_publisher<aruco_sam_ailab::msg::OptimizedKeyframeState>("/optimized_keyframe_state", 10);
        if (useWheelOdom) {
            pubCorrectedOdom_ = create_publisher<nav_msgs::msg::Odometry>("/aruco_slam/wheel_odom", 10);
            pubCorrectedPath_ = create_publisher<nav_msgs::msg::Path>("/aruco_slam/wheel_odom_path", 10);
            correctedOdomPath_.header.frame_id = odomFrame;
        }
        tfBroadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // ─── 2.5. Service ───
        declare_parameter("landmarks_save_path", "landmarks_map.json");
        get_parameter("landmarks_save_path", landmarksSavePath_);
        saveLandmarksSrv_ = create_service<std_srvs::srv::Trigger>(
            "save_landmarks",
            std::bind(&GraphOptimizer::saveLandmarksCallback, this,
                      std::placeholders::_1, std::placeholders::_2));

        // ─── 2.6. Run Mode ───
        declare_parameter("run_mode", "mapping");
        declare_parameter("map_path", "landmarks_map.json");
        std::string runModeStr;
        get_parameter("run_mode", runModeStr);
        get_parameter("map_path", mapPath_);
        isLocalizationMode_ = (runModeStr == "localization");
        fixedLandmarkNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());

        // ─── 2.8. Keyframe params ───
        keyframeTimeThresh_ = keyframeTimeInterval;
        RCLCPP_INFO(get_logger(), "Keyframe policy: time=%.2fs (marker-triggered + time fallback)",
                    keyframeTimeThresh_);

        // ─── 3. ISAM2 ───
        gtsam::ISAM2Params params;
        params.relinearizeThreshold = isamRelinearizeThreshold;
        params.relinearizeSkip = isamRelinearizeSkip;
        isam_ = gtsam::ISAM2(params);

        // ─── 4. IMU Preintegration Params ───
        imuParams_ = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        imuParams_->accelerometerCovariance = gtsam::Matrix33::Identity() * pow(imuAccNoise, 2);
        imuParams_->gyroscopeCovariance = gtsam::Matrix33::Identity() * pow(imuGyrNoise, 2);
        imuParams_->integrationCovariance = gtsam::Matrix33::Identity() * pow(1e-4, 2);
        // Set IMU lever arm (IMU position in body frame)
        imuParams_->body_P_sensor = gtsam::Pose3(
            gtsam::Rot3::Identity(),
            gtsam::Point3(extTransBaseImu.x(), extTransBaseImu.y(), extTransBaseImu.z()));
        imuPreintegrator_ = new gtsam::PreintegratedImuMeasurements(imuParams_, currentBias_);

        // ─── 5. Noise Models ───
        // GTSAM Pose3 tangent order: [rot_x(roll), rot_y(pitch), rot_z(yaw), trans_x, trans_y, trans_z]
        // 2D UGV: roll/pitch/z ≈ const → tight, yaw/x/y → 자유도
        priorPoseNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001).finished());
        // Localization: 초기 포즈가 ArUco 단일 관측에서 계산되므로 x/y/yaw 완화, roll/pitch/z tight (2D UGV)
        if (isLocalizationMode_) {
            priorPoseNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << 0.01, 0.01, 0.15, 0.10, 0.10, 0.01).finished());
        }
        priorVelNoise_ = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
        priorBiasNoise_ = gtsam::noiseModel::Isotropic::Sigma(6, 0.1);  // 초기 bias 불확실성 허용

        // Bias random walk: acc/gyro 분리 스케일링
        // Acc bias 100x: accelerometer 드리프트가 크므로 bias 보정 허용
        // Gyro bias 10x: gyro bias를 안정적으로 유지 → 실제 회전이 bias로 흡수되는 것 방지
        //   (100x일 때 optimizer가 회전을 bias 변화로 설명 → heading 실시간성 저하)
        biasRandomWalkNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << imuAccBiasN * 100, imuAccBiasN * 100, imuAccBiasN * 100,
                                  imuGyrBiasN * 10, imuGyrBiasN * 10, imuGyrBiasN * 10).finished());

        // ArUco obsNoise: 기본 노이즈 (addLandmarkFactors에서 거리/각도별 dynamic noise로 대체됨)
        // 이 값은 fallback용으로 유지
        obsNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.05, 0.05, arucoRotNoise,
                                  arucoTransNoise, arucoTransNoise, 0.03).finished());

        // ArUco observation filter params
        maxArucoRange_ = arucoMaxRange;
        minViewingAngle_ = arucoMinViewingAngle;
        RCLCPP_INFO(get_logger(), "ArUco filter: max_range=%.1fm, min_viewing_angle=%.1fdeg",
                    maxArucoRange_, minViewingAngle_ * 180.0 / M_PI);
        // Landmark prior: soft regularization → 관측이 지배하도록 큰 σ 사용
        // 초기 추정값이 로봇 pose 오차를 포함하므로 tight prior는 오히려 해로움
        landmarkPriorNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.1, 0.1, 0.5, 1.0, 1.0, 0.1).finished());

        // Wheel odom noise: GTSAM Pose3 tangent order [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
        if (useWheelOdom) {
            wheelOdomNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << wheelOdomNoiseRoll, wheelOdomNoisePitch, wheelOdomNoiseYaw,
                                      wheelOdomNoiseX, wheelOdomNoiseY, wheelOdomNoiseZ).finished());
            // 정지 시 zero-motion constraint: 모든 축 매우 tight
            wheelOdomZeroNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001).finished());
            RCLCPP_INFO(get_logger(), "Wheel odom enabled: topic=%s noise=[%.2f,%.2f,%.2f | %.2f,%.2f,%.2f]",
                        wheelOdomTopic.c_str(),
                        wheelOdomNoiseRoll, wheelOdomNoisePitch, wheelOdomNoiseYaw,
                        wheelOdomNoiseX, wheelOdomNoiseY, wheelOdomNoiseZ);
        }

        // ─── 6. Extrinsic (Color camera for ArUco) ───
        gtsam::Rot3 rot(extRotBaseCam);
        gtsam::Point3 trans(extTransBaseCam.x(), extTransBaseCam.y(), extTransBaseCam.z());
        baseToCam_ = gtsam::Pose3(rot, trans);

        // ─── 7. Mode init ───
        if (isLocalizationMode_) {
            loadMap(mapPath_);
            RCLCPP_INFO(get_logger(), "Graph Optimizer (ImuFactor) - Localization Mode (%zu landmarks)", landmarkIdToKey_.size());
        } else {
            RCLCPP_INFO(get_logger(), "Graph Optimizer (ImuFactor) - Mapping Mode");
        }

        // ─── 8. SLAM Timer (decoupled from ArUco callback) ───
        slamTimer_ = create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&GraphOptimizer::slamTimerCallback, this),
            timerCbGroup_);

        RCLCPP_INFO(get_logger(), "Mutex split: queueMtx_ / slamMtx_ / pendingMtx_ / snapMtx_");
        RCLCPP_INFO(get_logger(), "Callback groups: queue / aruco / timer (concurrent with MultiThreadedExecutor)");
    }

    ~GraphOptimizer() {
        if (imuPreintegrator_) delete imuPreintegrator_;
    }

    // ═══════════════════════════════════════════════════════════
    //  ArUco Callback (lightweight — never blocks on ISAM2)
    //  Runs at camera rate (~30Hz). Stores pending data for timer.
    //  Publishes last SLAM estimate for EKF correction.
    // ═══════════════════════════════════════════════════════════
    void arucoHandler(const aruco_sam_ailab::msg::MarkerArray::SharedPtr msg) {
        // 1. Transform markers to base_link (lock-free, baseToCam_ is const)
        aruco_sam_ailab::msg::MarkerArray markersInBase;
        markersInBase.header = msg->header;
        for (const auto& rawMarker : msg->markers) {
            gtsam::Pose3 poseInCam = poseMsgToGtsam(rawMarker.pose);
            gtsam::Pose3 poseInBase = baseToCam_.compose(poseInCam);
            auto newMarker = rawMarker;
            newMarker.pose = gtsamToPoseMsg(poseInBase);
            markersInBase.markers.push_back(newMarker);
        }

        // 2. Store latest observation for SLAM timer (always update to keep timer alive)
        {
            std::lock_guard<std::mutex> lock(pendingMtx_);
            pendingMarkers_ = markersInBase;
            pendingStamp_ = msg->header.stamp;
            pendingArucoValid_ = true;
        }

        // 3. Publish last SLAM estimate (~30Hz)
        {
            gtsam::Pose3 snapEst;
            bool valid;
            {
                std::lock_guard<std::mutex> lock(snapMtx_);
                snapEst = snapEstimate_;
                valid = snapValid_;
            }
            if (valid) {
                publishSlamPose(snapEst, msg->header.stamp);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  SLAM Timer Callback (heavy ISAM2 work, decoupled from ArUco)
    //  Runs at ~50ms intervals. Processes stored ArUco data.
    //  ISAM2 optimization happens here, not in arucoHandler.
    // ═══════════════════════════════════════════════════════════
    void slamTimerCallback() {
        // 1. Check for pending ArUco data
        aruco_sam_ailab::msg::MarkerArray markers;
        rclcpp::Time stamp(0, 0, RCL_ROS_TIME);
        bool hasPending = false;
        {
            std::lock_guard<std::mutex> lock(pendingMtx_);
            if (pendingArucoValid_) {
                markers = pendingMarkers_;
                stamp = pendingStamp_;
                hasPending = true;
                pendingArucoValid_ = false;
            }
        }

        if (!hasPending) return;

        // 2. Lock SLAM state for processing
        std::lock_guard<std::mutex> lock(slamMtx_);

        // 3. Marker IDs
        std::vector<int> currMarkerIds;
        for (const auto& m : markers.markers) currMarkerIds.push_back(m.id);

        // 4. System initialization
        //    시퀀스: [1] IMU 대기 → [2] ArUco 첫 최적화 → 초기화
        //    초기화 후에만 IMU 적분 + keyframe 최적화 시작
        if (!systemInitialized_) {
            // ── 센서 데이터 준비 확인 (IMU 필요) ──
            {
                std::lock_guard<std::mutex> qlock(queueMtx_);
                if (rawImuQueue_.size() < 50) {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                        "[Init 1/2] Waiting for IMU data (%zu/50 samples)...", rawImuQueue_.size());
                    return;
                }
            }
            // [Init 2/2] ArUco 마커 기반 초기화
            if (isLocalizationMode_) {
                for (const auto& m : markers.markers) {
                    if (landmarkIdToKey_.count(static_cast<int>(m.id))) {
                        int mid = static_cast<int>(m.id);
                        gtsam::Pose3 L_map = isam_.calculateEstimate().at<gtsam::Pose3>(L(mid));
                        gtsam::Pose3 L_base = poseMsgToGtsam(m.pose);
                        gtsam::Pose3 initialPose = L_map.compose(L_base.inverse());
                        initializeSystemAt(initialPose, stamp);
                        lastVisibleMarkers_.clear();
                        for (const auto& mk : markers.markers) lastVisibleMarkers_.insert(mk.id);
                        addLandmarkFactors(0, markers);
                        isam_.update(graphFactors_, graphValues_);
                        graphFactors_.resize(0);
                        graphValues_.clear();
                        currentEstimate_ = isam_.calculateEstimate().at<gtsam::Pose3>(X(0));
                        updateSnapshot();
                        publishLandmarks(stamp);
                        return;
                    }
                }
                std::string detectedIds, mapIds;
                for (const auto& m : markers.markers) detectedIds += std::to_string(m.id) + " ";
                for (const auto& p : landmarkIdToKey_) mapIds += std::to_string(p.first) + " ";
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "Localization: Waiting for known marker. Detected: [%s] | Map has: [%s]",
                    detectedIds.empty() ? "(none)" : detectedIds.c_str(),
                    mapIds.empty() ? "(load failed?)" : mapIds.c_str());
                return;
            } else {
                if (markers.markers.empty()) {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                        "Mapping: Waiting for ArUco markers to initialize...");
                    return;
                }
                initializeSystem(markers, stamp);
                updateSnapshot();
                return;
            }
        }

        // 5. Keyframe decision + ISAM2 optimization
        if (needNewKeyframe(stamp, currMarkerIds)) {
            processKeyframe(markers, stamp);
            lastKeyframeTime_ = stamp;
            lastVisibleMarkers_.clear();
            for (int id : currMarkerIds) lastVisibleMarkers_.insert(id);
            updateSnapshot();
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  Update SLAM snapshot (called after ISAM2, under slamMtx_)
    // ═══════════════════════════════════════════════════════════
    void updateSnapshot() {
        std::lock_guard<std::mutex> lock(snapMtx_);
        snapEstimate_ = currentEstimate_;
        snapValid_ = true;
    }

    // ═══════════════════════════════════════════════════════════
    //  IMU Preintegration for Factor Graph
    //  (locks queueMtx_ internally)
    // ═══════════════════════════════════════════════════════════
    int preintegrateImu(const rclcpp::Time& fromTime, const rclcpp::Time& toTime) {
        std::lock_guard<std::mutex> lock(queueMtx_);

        imuPreintegrator_->resetIntegrationAndSetBias(currentBias_);

        double lastT = fromTime.seconds();
        int count = 0;

        for (const auto& imu : rawImuQueue_) {
            double t = stamp2Sec(imu.header.stamp);
            if (t <= fromTime.seconds()) continue;
            if (t > toTime.seconds()) break;

            double dt = t - lastT;
            if (dt <= 0 || dt > 1.0) { lastT = t; continue; }

            imuPreintegrator_->integrateMeasurement(
                gtsam::Vector3(imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z),
                gtsam::Vector3(imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z),
                dt);
            lastT = t;
            count++;
        }

        // Trim old data (keep 0.5s buffer before fromTime)
        double cutoff = fromTime.seconds() - 0.5;
        while (!rawImuQueue_.empty() && stamp2Sec(rawImuQueue_.front().header.stamp) < cutoff) {
            rawImuQueue_.pop_front();
        }

        return count;
    }

    // ═══════════════════════════════════════════════════════════
    //  Wheel Odometry Delta (pose-differencing)
    //  (locks queueMtx_ internally)
    // ═══════════════════════════════════════════════════════════
    std::optional<WheelOdomResult> computeWheelOdomDelta(const rclcpp::Time& fromTime, const rclcpp::Time& toTime) {
        std::lock_guard<std::mutex> lock(queueMtx_);

        if (wheelOdomQueue_.empty()) return std::nullopt;

        double fromSec = fromTime.seconds();
        double toSec = toTime.seconds();
        constexpr double TOLERANCE = 0.1;  // 100ms

        // fromTime에 가장 가까운 odom 메시지 검색
        const nav_msgs::msg::Odometry* bestFrom = nullptr;
        double bestFromDt = TOLERANCE;
        const nav_msgs::msg::Odometry* bestTo = nullptr;
        double bestToDt = TOLERANCE;

        for (const auto& odom : wheelOdomQueue_) {
            double t = stamp2Sec(odom.header.stamp);
            double dtFrom = std::abs(t - fromSec);
            double dtTo = std::abs(t - toSec);
            if (dtFrom < bestFromDt) { bestFrom = &odom; bestFromDt = dtFrom; }
            if (dtTo < bestToDt) { bestTo = &odom; bestToDt = dtTo; }
        }

        if (!bestFrom || !bestTo || bestFrom == bestTo) return std::nullopt;

        // 정지 감지: fromTime~toTime 구간 내 모든 odom의 twist.linear 속도 체크
        constexpr double ZERO_VEL_THRESH = 0.01;  // m/s — 이하면 정지로 판단
        bool allStationary = true;
        for (const auto& odom : wheelOdomQueue_) {
            double t = stamp2Sec(odom.header.stamp);
            if (t < fromSec - 0.05 || t > toSec + 0.05) continue;
            double vx = odom.twist.twist.linear.x;
            double vy = odom.twist.twist.linear.y;
            double speed = std::sqrt(vx * vx + vy * vy);
            if (speed > ZERO_VEL_THRESH) {
                allStationary = false;
                break;
            }
        }

        // Pose differencing: delta = poseFrom.between(poseTo)
        gtsam::Pose3 poseFrom = poseMsgToGtsam(bestFrom->pose.pose);
        gtsam::Pose3 poseTo = poseMsgToGtsam(bestTo->pose.pose);
        gtsam::Pose3 delta = poseFrom.between(poseTo);

        // Sanity check: >3m 이동이면 odom reset으로 판단하여 reject
        double dist = delta.translation().norm();
        double dt = toSec - fromSec;
        if (dt > 0 && dist / dt > 3.0) {
            RCLCPP_WARN(get_logger(), "Wheel odom delta rejected: %.2fm in %.2fs (%.1fm/s)", dist, dt, dist / dt);
            return std::nullopt;
        }

        // 오래된 데이터 정리 (fromTime 0.5초 이전 제거)
        double cutoff = fromSec - 0.5;
        while (!wheelOdomQueue_.empty() && stamp2Sec(wheelOdomQueue_.front().header.stamp) < cutoff) {
            wheelOdomQueue_.pop_front();
        }

        return WheelOdomResult{allStationary ? gtsam::Pose3() : delta, allStationary};
    }

    // ═══════════════════════════════════════════════════════════
    //  Keyframe Policy (marker-triggered + time fallback)
    // ═══════════════════════════════════════════════════════════
    bool needNewKeyframe(const rclcpp::Time& currTime, const std::vector<int>& currMarkerIds) {
        if (frameIdx_ == 0) return true;

        double timeDiff = (currTime - lastKeyframeTime_).seconds();
        bool hasMarkers = !currMarkerIds.empty();

        double minInterval = hasMarkers ? 0.1 : 0.2;
        if (timeDiff < minInterval) return false;

        if (hasMarkers) return true;

        if (timeDiff > keyframeTimeThresh_) return true;

        return false;
    }

    // ═══════════════════════════════════════════════════════════
    //  Initialization
    // ═══════════════════════════════════════════════════════════
    void initializeSystem(const aruco_sam_ailab::msg::MarkerArray& markers, const rclcpp::Time& stamp) {
        initializeSystemAt(gtsam::Pose3(), stamp);
        lastVisibleMarkers_.clear();
        for (const auto& m : markers.markers) lastVisibleMarkers_.insert(m.id);
        addLandmarkFactors(0, markers);
        isam_.update(graphFactors_, graphValues_);
        graphFactors_.resize(0);
        graphValues_.clear();

        // Extract initial estimates after optimization with landmarks
        gtsam::Values result = isam_.calculateEstimate();
        currentEstimate_ = result.at<gtsam::Pose3>(X(0));
        currentVelocity_ = result.at<gtsam::Vector3>(V(0));
        currentBias_ = result.at<gtsam::imuBias::ConstantBias>(B(0));

        RCLCPP_INFO(get_logger(), "System Initialized at Frame 0 (Markers: %zu)", markers.markers.size());
    }

    // ─── Compute initial IMU bias from buffered stationary data ───
    // (locks queueMtx_ internally)
    gtsam::imuBias::ConstantBias computeInitialBias() {
        std::lock_guard<std::mutex> lock(queueMtx_);

        if (rawImuQueue_.size() < 50) {
            RCLCPP_WARN(get_logger(), "[GraphOpt] Not enough IMU data for bias estimation (%zu), using zero bias",
                        rawImuQueue_.size());
            return gtsam::imuBias::ConstantBias();
        }

        // Use the first 1 second of raw IMU data (already rotated to base_link)
        Eigen::Vector3d accSum(0, 0, 0), gyrSum(0, 0, 0);
        int count = 0;
        double startT = stamp2Sec(rawImuQueue_.front().header.stamp);

        for (const auto& imu : rawImuQueue_) {
            double t = stamp2Sec(imu.header.stamp);
            if (t - startT > 1.0) break;  // 최대 1초

            accSum += Eigen::Vector3d(imu.linear_acceleration.x,
                                      imu.linear_acceleration.y,
                                      imu.linear_acceleration.z);
            gyrSum += Eigen::Vector3d(imu.angular_velocity.x,
                                      imu.angular_velocity.y,
                                      imu.angular_velocity.z);
            count++;
        }
        if (count < 10) return gtsam::imuBias::ConstantBias();

        Eigen::Vector3d accMean = accSum / count;
        Eigen::Vector3d gyrMean = gyrSum / count;

        // Accel bias: measured - expected gravity
        // In base_link (Z-up), gravity reads as (0, 0, +g) at rest
        Eigen::Vector3d gravityInBody = accMean.normalized() * imuGravity;
        Eigen::Vector3d accBias = accMean - gravityInBody;

        // Gyro bias: mean gyro during stationary = bias
        Eigen::Vector3d gyrBias = gyrMean;

        RCLCPP_INFO(get_logger(),
            "[GraphOpt] Initial bias from %d samples: acc_bias=(%.4f,%.4f,%.4f) gyr_bias=(%.5f,%.5f,%.5f)",
            count, accBias.x(), accBias.y(), accBias.z(),
            gyrBias.x(), gyrBias.y(), gyrBias.z());

        return gtsam::imuBias::ConstantBias(
            (gtsam::Vector(6) << accBias.x(), accBias.y(), accBias.z(),
                                  gyrBias.x(), gyrBias.y(), gyrBias.z()).finished());
    }

    void initializeSystemAt(const gtsam::Pose3& startPose, const rclcpp::Time& stamp) {
        currentEstimate_ = startPose;
        currentVelocity_ = gtsam::Vector3::Zero();
        lastKeyframeTime_ = stamp;
        lastVisibleMarkers_.clear();

        // Compute initial bias from buffered stationary IMU data
        currentBias_ = computeInitialBias();

        // Pose prior
        graphValues_.insert(X(0), startPose);
        graphFactors_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), startPose, priorPoseNoise_));

        // Velocity prior (stationary)
        gtsam::Vector3 zeroVel(0, 0, 0);
        graphValues_.insert(V(0), zeroVel);
        graphFactors_.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), zeroVel, priorVelNoise_));

        // Bias prior (estimated from stationary data)
        graphValues_.insert(B(0), currentBias_);
        graphFactors_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), currentBias_, priorBiasNoise_));

        systemInitialized_ = true;
        frameIdx_ = 0;

        // IMU + Wheel Odom 큐 정리: 초기화 시점 이전 데이터 제거
        {
            std::lock_guard<std::mutex> lock(queueMtx_);
            double initTime = stamp.seconds();
            while (!rawImuQueue_.empty() && stamp2Sec(rawImuQueue_.front().header.stamp) < initTime - 0.1) {
                rawImuQueue_.pop_front();
            }
            while (!wheelOdomQueue_.empty() && stamp2Sec(wheelOdomQueue_.front().header.stamp) < initTime - 0.1) {
                wheelOdomQueue_.pop_front();
            }
            RCLCPP_INFO(get_logger(), "[Init] Queue trimmed: IMU=%zu, WheelOdom=%zu",
                        rawImuQueue_.size(), wheelOdomQueue_.size());
        }

        imuPreintegrator_->resetIntegrationAndSetBias(currentBias_);

        RCLCPP_INFO(get_logger(), "System Initialized at pose (%.2f, %.2f, %.2f)",
                    startPose.translation().x(), startPose.translation().y(), startPose.translation().z());
    }

    // ═══════════════════════════════════════════════════════════
    //  Process Keyframe (ImuFactor + ArUco)
    //  Called from slamTimerCallback under slamMtx_
    // ═══════════════════════════════════════════════════════════
    void processKeyframe(const aruco_sam_ailab::msg::MarkerArray& markers,
                         const rclcpp::Time& stamp) {
        frameIdx_++;

        // 1. IMU Preintegration: lastKeyframeTime_ → stamp
        //    (locks queueMtx_ internally)
        int imuCount = preintegrateImu(lastKeyframeTime_, stamp);

        // 2. Predict state using IMU
        gtsam::Pose3 predictedPose;
        gtsam::Vector3 predictedVel;

        if (imuCount > 0) {
            gtsam::NavState prevNavState(currentEstimate_, currentVelocity_);
            gtsam::NavState predictedState = imuPreintegrator_->predict(prevNavState, currentBias_);
            predictedPose = predictedState.pose();
            predictedVel = predictedState.velocity();

            // [DEBUG] IMU preintegration 결과 확인
            double prevYaw = currentEstimate_.rotation().yaw();
            double predYaw = predictedPose.rotation().yaw();
            auto deltaR = imuPreintegrator_->deltaRij();
            RCLCPP_INFO(get_logger(),
                "[IMU DEBUG] %d samples | deltaRot(r=%.4f p=%.4f y=%.4f) | prevYaw=%.3f predYaw=%.3f dYaw=%.4f rad (%.1f deg)"
                " | gyrBias=(%.5f,%.5f,%.5f)",
                imuCount,
                deltaR.roll(), deltaR.pitch(), deltaR.yaw(),
                prevYaw, predYaw, predYaw - prevYaw, (predYaw - prevYaw) * 180.0 / M_PI,
                currentBias_.gyroscope().x(), currentBias_.gyroscope().y(), currentBias_.gyroscope().z());
        } else {
            // No IMU — identity motion (정지 가정)
            predictedPose = currentEstimate_;
            predictedVel = currentVelocity_;
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "No IMU data for preintegration, using identity motion");
        }

        // 3. Insert new variables
        graphValues_.insert(X(frameIdx_), predictedPose);
        graphValues_.insert(V(frameIdx_), predictedVel);
        graphValues_.insert(B(frameIdx_), currentBias_);

        // 4. ImuFactor or identity/wheel-odom fallback
        //    Wheel odom delta 계산 (queueMtx_ lock 내부)
        std::optional<WheelOdomResult> wheelResult;
        if (useWheelOdom) {
            wheelResult = computeWheelOdomDelta(lastKeyframeTime_, stamp);
        }

        if (imuCount > 0) {
            graphFactors_.add(gtsam::ImuFactor(
                X(frameIdx_ - 1), V(frameIdx_ - 1),
                X(frameIdx_), V(frameIdx_),
                B(frameIdx_ - 1),
                *imuPreintegrator_));

            // Wheel odom: IMU와 함께 추가 제약으로 삽입
            if (wheelResult.has_value()) {
                auto& noise = wheelResult->isStationary ? wheelOdomZeroNoise_ : wheelOdomNoise_;
                graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                    X(frameIdx_ - 1), X(frameIdx_), wheelResult->delta, noise));
                if (enableTopicDebugLog) {
                    RCLCPP_INFO(get_logger(), "[WheelOdom] BetweenFactor added (w/ IMU)%s: dx=%.3f dy=%.3f dyaw=%.3f",
                                wheelResult->isStationary ? " [STATIONARY]" : "",
                                wheelResult->delta.translation().x(), wheelResult->delta.translation().y(),
                                wheelResult->delta.rotation().yaw());
                }
            }
        } else if (wheelResult.has_value()) {
            // IMU 없음 — wheel odom을 identity fallback 대신 사용
            auto& noise = wheelResult->isStationary ? wheelOdomZeroNoise_ : wheelOdomNoise_;
            graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                X(frameIdx_ - 1), X(frameIdx_), wheelResult->delta, noise));
            RCLCPP_INFO(get_logger(), "[WheelOdom] BetweenFactor replacing identity fallback%s: dx=%.3f dy=%.3f",
                        wheelResult->isStationary ? " [STATIONARY]" : "",
                        wheelResult->delta.translation().x(), wheelResult->delta.translation().y());
        } else {
            // 둘 다 없음 — identity fallback (그래프 under-constrained 방지)
            // 2D UGV: roll/pitch/z tight, yaw/x/y loose
            graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                X(frameIdx_ - 1), X(frameIdx_), gtsam::Pose3(),
                gtsam::noiseModel::Diagonal::Sigmas(
                    (gtsam::Vector(6) << 0.01, 0.01, 0.1, 0.1, 0.1, 0.01).finished())));
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "No IMU and no wheel odom, using identity motion fallback");
        }

        // 5. Bias random walk constraint
        graphFactors_.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
            B(frameIdx_ - 1), B(frameIdx_),
            gtsam::imuBias::ConstantBias(),
            biasRandomWalkNoise_));

        // 6. Landmark factors
        addLandmarkFactors(frameIdx_, markers);

        // 7. Optimize
        try {
            isam_.update(graphFactors_, graphValues_);
            graphFactors_.resize(0);
            graphValues_.clear();

            gtsam::Values result = isam_.calculateEstimate();
            currentEstimate_ = result.at<gtsam::Pose3>(X(frameIdx_));
            currentVelocity_ = result.at<gtsam::Vector3>(V(frameIdx_));
            currentBias_ = result.at<gtsam::imuBias::ConstantBias>(B(frameIdx_));

            // Reset preintegrator with updated bias
            imuPreintegrator_->resetIntegrationAndSetBias(currentBias_);

            // Wheel odom repropagation reference 업데이트
            if (useWheelOdom) {
                // 현재 keyframe 시점에 가장 가까운 odom pose 찾기
                gtsam::Pose3 odomAtKF;
                bool foundOdom = false;
                {
                    std::lock_guard<std::mutex> qlock(queueMtx_);
                    double stampSec = stamp.seconds();
                    double bestDt = 0.2;
                    for (const auto& odom : wheelOdomQueue_) {
                        double dt = std::abs(stamp2Sec(odom.header.stamp) - stampSec);
                        if (dt < bestDt) {
                            bestDt = dt;
                            odomAtKF = poseMsgToGtsam(odom.pose.pose);
                            foundOdom = true;
                        }
                    }
                }
                if (foundOdom) {
                    std::lock_guard<std::mutex> slock(snapMtx_);
                    slamPoseAtLastKF_ = currentEstimate_;
                    odomPoseAtLastKF_ = odomAtKF;
                    wheelOdomRepropValid_ = true;
                }
            }

            // Publish landmarks (visualization)
            publishLandmarks(stamp);

            // Warmup: 초반 N 키프레임 동안은 그래프가 불안정하므로
            // imu_preintegration/EKF에 보정값을 보내지 않음
            static constexpr int WARMUP_FRAMES = 5;
            if (frameIdx_ >= WARMUP_FRAMES) {
                publishOptimizedKeyframeState(currentEstimate_, stamp);
            } else {
                RCLCPP_INFO(get_logger(), "[ISAM2] Warmup frame %d/%d — not publishing correction yet",
                            frameIdx_, WARMUP_FRAMES);
            }

            RCLCPP_INFO(get_logger(),
                "[ISAM2] frame=%d pos=(%.3f,%.3f) yaw=%.2f imu=%d markers=%zu",
                frameIdx_,
                currentEstimate_.translation().x(), currentEstimate_.translation().y(),
                currentEstimate_.rotation().yaw(), imuCount, markers.markers.size());
            if (enableTopicDebugLog) {
                RCLCPP_INFO(get_logger(),
                    "[ISAM2]   vel=(%.2f,%.2f,%.2f) bias_acc=(%.4f,%.4f,%.4f) bias_gyr=(%.5f,%.5f,%.5f)",
                    currentVelocity_.x(), currentVelocity_.y(), currentVelocity_.z(),
                    currentBias_.accelerometer().x(), currentBias_.accelerometer().y(), currentBias_.accelerometer().z(),
                    currentBias_.gyroscope().x(), currentBias_.gyroscope().y(), currentBias_.gyroscope().z());
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "ISAM2 Update Failed: %s", e.what());
            graphFactors_.resize(0);
            graphValues_.clear();
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  Publishers
    // ═══════════════════════════════════════════════════════════
    void publishOptimizedKeyframeState(const gtsam::Pose3& optimizedPose, const rclcpp::Time& stamp) {
        aruco_sam_ailab::msg::OptimizedKeyframeState msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = mapFrame;
        msg.pose = gtsamToPoseMsg(optimizedPose);
        // Optimized velocity from graph
        msg.velocity.x = currentVelocity_.x();
        msg.velocity.y = currentVelocity_.y();
        msg.velocity.z = currentVelocity_.z();
        // Optimized bias from graph
        gtsam::Vector6 biasVec = currentBias_.vector();
        msg.bias.resize(6);
        for (int i = 0; i < 6; i++) msg.bias[i] = biasVec(i);
        pubOptimizedState_->publish(msg);
    }

    void publishSlamPose(const gtsam::Pose3& estimatedPose, const rclcpp::Time& stamp) {
        geometry_msgs::msg::PoseStamped p;
        p.header.stamp = stamp;
        p.header.frame_id = mapFrame;
        p.pose = gtsamToPoseMsg(estimatedPose);
        globalPath_.header = p.header;
        globalPath_.poses.push_back(p);
        pubPath_->publish(globalPath_);
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = mapFrame;
        odom.child_frame_id = baseLinkFrame;
        odom.pose.pose = p.pose;
        pubGlobalOdom_->publish(odom);
    }

    // ═══════════════════════════════════════════════════════════
    //  Landmark Factors
    // ═══════════════════════════════════════════════════════════
    void addLandmarkFactors(int currentFrameIdx, const aruco_sam_ailab::msg::MarkerArray& markers) {
        for (const auto& marker : markers.markers) {
            int mid = marker.id;
            gtsam::Pose3 measurement = poseMsgToGtsam(marker.pose);

            // ── Range filter: 원거리 관측은 depth 오차가 커서 제외 ──
            double range = measurement.translation().norm();
            if (range > maxArucoRange_) {
                continue;
            }

            // ── Viewing angle filter ──
            // marker Z axis (normal, 마커 면에서 외부 방향) in base_link frame
            Eigen::Vector3d marker_z = measurement.rotation().matrix().col(2);
            // viewing ray: base_link → marker 방향
            Eigen::Vector3d view_ray = measurement.translation().normalized();
            // 정면 관측: marker normal ≈ -view_ray → |cos| ≈ 1
            double cos_angle = std::abs(marker_z.dot(-view_ray));
            double viewing_angle = std::acos(std::clamp(cos_angle, 0.0, 1.0));

            if (viewing_angle < minViewingAngle_) {
                continue;  // 정면 ±15° 이내, PnP depth 불안정 → 스킵
            }

            // ── Dynamic noise model ──
            // 거리 비례: 1m→1.3x, 3m→1.9x, 4m→2.2x
            double range_factor = 1.0 + 0.3 * range;
            // 정면일수록 depth 불확실: head-on(cos≈1)→2.5x, 45°→1.75x, edge-on(cos≈0)→1x
            double angle_factor = 1.0 + 1.5 * cos_angle * cos_angle;
            double sigma_t = arucoTransNoise * range_factor * angle_factor;

            auto dynamicNoise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << 0.05, 0.05, arucoRotNoise,
                                      sigma_t, sigma_t, 0.03).finished());

            bool knownLandmark = (landmarkIdToKey_.find(mid) != landmarkIdToKey_.end());

            if (isLocalizationMode_) {
                if (!knownLandmark) continue;
                gtsam::Key landmarkKey = landmarkIdToKey_[mid];
                graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                    X(currentFrameIdx), landmarkKey, measurement, dynamicNoise));
            } else {
                gtsam::Key landmarkKey;
                if (!knownLandmark) {
                    landmarkKey = L(mid);
                    landmarkIdToKey_[mid] = landmarkKey;
                    gtsam::Pose3 initialLandmarkPose;
                    if (graphValues_.exists(X(currentFrameIdx))) {
                        initialLandmarkPose = graphValues_.at<gtsam::Pose3>(X(currentFrameIdx)).compose(measurement);
                    } else {
                        initialLandmarkPose = currentEstimate_.compose(measurement);
                    }
                    graphValues_.insert(landmarkKey, initialLandmarkPose);
                    graphFactors_.add(gtsam::PriorFactor<gtsam::Pose3>(landmarkKey, initialLandmarkPose, landmarkPriorNoise_));
                } else {
                    landmarkKey = landmarkIdToKey_[mid];
                }
                graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                    X(currentFrameIdx), landmarkKey, measurement, dynamicNoise));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  Helpers
    // ═══════════════════════════════════════════════════════════
    void publishLandmarks(const rclcpp::Time& stamp) {
        visualization_msgs::msg::MarkerArray arr;
        if (landmarkIdToKey_.empty()) { pubLandmarks_->publish(arr); return; }
        gtsam::Values est;
        try { est = isam_.calculateEstimate(); }
        catch (...) { pubLandmarks_->publish(arr); return; }
        for (const auto& [markerId, landmarkKey] : landmarkIdToKey_) {
            if (!est.exists(landmarkKey)) continue;
            gtsam::Pose3 poseMap = est.at<gtsam::Pose3>(landmarkKey);
            geometry_msgs::msg::Pose poseMsgMap = gtsamToPoseMsg(poseMap);
            visualization_msgs::msg::Marker sphere;
            sphere.header.stamp = stamp; sphere.header.frame_id = mapFrame;
            sphere.ns = "landmarks"; sphere.id = markerId;
            sphere.type = visualization_msgs::msg::Marker::SPHERE;
            sphere.action = visualization_msgs::msg::Marker::ADD;
            sphere.pose.position.x = poseMap.translation().x();
            sphere.pose.position.y = poseMap.translation().y();
            sphere.pose.position.z = poseMap.translation().z();
            sphere.pose.orientation.w = 1.0;
            sphere.pose.orientation.x = sphere.pose.orientation.y = sphere.pose.orientation.z = 0.0;
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.15;
            sphere.color.r = 0.2f; sphere.color.g = 0.6f; sphere.color.b = 1.0f; sphere.color.a = 0.9f;
            sphere.lifetime = rclcpp::Duration(0, 0);
            arr.markers.push_back(sphere);
            visualization_msgs::msg::Marker arrow;
            arrow.header.stamp = stamp; arrow.header.frame_id = mapFrame;
            arrow.ns = "landmark_normal"; arrow.id = markerId;
            arrow.type = visualization_msgs::msg::Marker::ARROW;
            arrow.action = visualization_msgs::msg::Marker::ADD;
            gtsam::Matrix33 R = poseMap.rotation().matrix();
            geometry_msgs::msg::Point startPt, endPt;
            startPt.x = poseMap.translation().x(); startPt.y = poseMap.translation().y(); startPt.z = poseMap.translation().z();
            endPt.x = startPt.x + 0.3 * R(0, 2); endPt.y = startPt.y + 0.3 * R(1, 2); endPt.z = startPt.z + 0.3 * R(2, 2);
            arrow.points.push_back(startPt); arrow.points.push_back(endPt);
            arrow.scale.x = 0.02; arrow.scale.y = 0.04; arrow.scale.z = 0.0;
            arrow.color.r = 0.3f; arrow.color.g = 0.3f; arrow.color.b = 1.0f; arrow.color.a = 0.9f;
            arrow.lifetime = rclcpp::Duration(0, 0);
            arr.markers.push_back(arrow);
            visualization_msgs::msg::Marker text;
            text.header.stamp = stamp; text.header.frame_id = mapFrame;
            text.ns = "landmark_ids"; text.id = markerId;
            text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text.action = visualization_msgs::msg::Marker::ADD;
            text.pose.position.x = poseMap.translation().x();
            text.pose.position.y = poseMap.translation().y();
            text.pose.position.z = poseMap.translation().z() + 0.15;
            text.pose.orientation = poseMsgMap.orientation;
            text.scale.z = 0.3;
            text.color.r = 1.0f; text.color.g = 1.0f; text.color.b = 1.0f; text.color.a = 1.0f;
            text.text = std::to_string(markerId);
            text.lifetime = rclcpp::Duration(0, 0);
            arr.markers.push_back(text);
        }
        pubLandmarks_->publish(arr);
    }

    // ═══════════════════════════════════════════════════════════
    //  Map Load/Save
    // ═══════════════════════════════════════════════════════════
    static double extractJsonDouble(const std::string& s, const std::string& key, size_t start = 0) {
        std::string pattern = "\"" + key + "\":";
        size_t p = s.find(pattern, start);
        if (p == std::string::npos) return 0.0;
        p += pattern.size();
        return std::stod(s.substr(p));
    }
    static int extractJsonInt(const std::string& s, const std::string& key, size_t start = 0) {
        std::string pattern = "\"" + key + "\":";
        size_t p = s.find(pattern, start);
        if (p == std::string::npos) return 0;
        p += pattern.size();
        return std::stoi(s.substr(p));
    }

    void loadMap(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) { RCLCPP_ERROR(get_logger(), "loadMap: Failed to open %s", filename.c_str()); return; }
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        size_t landmarksPos = content.find("\"landmarks\"");
        if (landmarksPos == std::string::npos) { RCLCPP_ERROR(get_logger(), "loadMap: No 'landmarks' key"); return; }
        size_t arrStart = content.find('[', landmarksPos);
        if (arrStart == std::string::npos) return;
        graphFactors_.resize(0); graphValues_.clear(); landmarkIdToKey_.clear();
        size_t pos = arrStart + 1;
        while (pos < content.size()) {
            size_t objStart = content.find('{', pos);
            if (objStart == std::string::npos) break;
            int depth = 1; size_t objEnd = objStart + 1;
            for (; objEnd < content.size() && depth > 0; objEnd++) {
                if (content[objEnd] == '{') depth++;
                else if (content[objEnd] == '}') depth--;
            }
            if (depth != 0) break;
            std::string obj = content.substr(objStart, objEnd - objStart + 1);
            int id = extractJsonInt(obj, "id");
            size_t posIdx = obj.find("\"position\""); size_t oriIdx = obj.find("\"orientation\"");
            if (posIdx == std::string::npos || oriIdx == std::string::npos) { pos = objEnd + 1; continue; }
            std::string posBlock = obj.substr(posIdx, oriIdx - posIdx);
            std::string oriBlock = obj.substr(oriIdx);
            double x = extractJsonDouble(posBlock, "x"), y = extractJsonDouble(posBlock, "y"), z = extractJsonDouble(posBlock, "z");
            double qw = extractJsonDouble(oriBlock, "w"), qx = extractJsonDouble(oriBlock, "x");
            double qy = extractJsonDouble(oriBlock, "y"), qz = extractJsonDouble(oriBlock, "z");
            gtsam::Key key = L(id); landmarkIdToKey_[id] = key;
            gtsam::Pose3 pose(gtsam::Rot3::Quaternion(qw, qx, qy, qz), gtsam::Point3(x, y, z));
            graphValues_.insert(key, pose);
            graphFactors_.add(gtsam::PriorFactor<gtsam::Pose3>(key, pose, fixedLandmarkNoise_));
            pos = objEnd + 1;
        }
        isam_.update(graphFactors_, graphValues_);
        graphFactors_.resize(0); graphValues_.clear();
        std::string idsStr;
        for (const auto& p : landmarkIdToKey_) idsStr += std::to_string(p.first) + " ";
        RCLCPP_INFO(get_logger(), "loadMap: Loaded %zu landmarks [ids: %s]", landmarkIdToKey_.size(), idsStr.c_str());
    }

    void saveLandmarksCallback(const std_srvs::srv::Trigger::Request::SharedPtr,
                               std_srvs::srv::Trigger::Response::SharedPtr response) {
        std::lock_guard<std::mutex> lock(slamMtx_);
        if (landmarkIdToKey_.empty()) { response->success = false; response->message = "No landmarks."; return; }
        gtsam::Values est;
        try { est = isam_.calculateEstimate(); }
        catch (const std::exception& e) { response->success = false; response->message = e.what(); return; }
        std::ostringstream json;
        json << "{\n  \"frame_id\": \"" << mapFrame << "\",\n";
        json << "  \"timestamp\": \"" << now().seconds() << "\",\n  \"landmarks\": [\n";
        bool first = true; size_t savedCount = 0;
        for (const auto& [markerId, landmarkKey] : landmarkIdToKey_) {
            if (!est.exists(landmarkKey)) continue;
            gtsam::Pose3 poseMap = est.at<gtsam::Pose3>(landmarkKey);
            auto q = poseMap.rotation().toQuaternion();
            if (!first) json << ",\n";
            json << "    {\"id\": " << markerId << ", \"position\": {\"x\": " << poseMap.translation().x()
                 << ", \"y\": " << poseMap.translation().y() << ", \"z\": " << poseMap.translation().z()
                 << "}, \"orientation\": {\"w\": " << q.w() << ", \"x\": " << q.x()
                 << ", \"y\": " << q.y() << ", \"z\": " << q.z() << "}}";
            first = false; savedCount++;
        }
        json << "\n  ]\n}\n";
        std::ofstream ofs(landmarksSavePath_);
        if (!ofs) { response->success = false; response->message = "Cannot open " + landmarksSavePath_; return; }
        ofs << json.str(); ofs.close();
        response->success = true;
        response->message = "Saved " + std::to_string(savedCount) + " landmarks to " + landmarksSavePath_;
        RCLCPP_INFO(get_logger(), "save_landmarks: %s", response->message.c_str());
    }
};

} // namespace aruco_sam_ailab

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    auto node = std::make_shared<aruco_sam_ailab::GraphOptimizer>(options);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Graph Optimizer Started.\033[0m");
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
