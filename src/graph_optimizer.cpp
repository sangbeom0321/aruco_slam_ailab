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
    // ═══ Mutexes (split for concurrency) ═══
    // queueMtx_: protects IMU/odom queues (very fast, microseconds)
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
    gtsam::Pose3 lastOdomPose_;
    bool systemInitialized_ = false;
    int frameIdx_ = 0;

    // ═══ Keyframe Selection (guarded by slamMtx_) ═══
    gtsam::Pose3 lastKeyframeOdomPose_;
    rclcpp::Time lastKeyframeTime_{0, 0, RCL_ROS_TIME};
    std::set<int> lastVisibleMarkers_;
    double keyframeDistThresh_;
    double keyframeAngleThresh_;
    double keyframeTimeThresh_;

    // ═══ Raw IMU queue (guarded by queueMtx_) ═══
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subRawImu_;
    std::deque<sensor_msgs::msg::Imu> rawImuQueue_;
    boost::shared_ptr<gtsam::PreintegrationParams> imuParams_;
    gtsam::PreintegratedImuMeasurements* imuPreintegrator_ = nullptr;  // guarded by slamMtx_

    // ═══ IMU Odom queue (guarded by queueMtx_) ═══
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subImuOdom_;
    std::deque<nav_msgs::msg::Odometry> imuOdomQueue_;

    // ═══ Wheel Odometry queue (guarded by queueMtx_) ═══
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subWheelOdom_;
    std::deque<nav_msgs::msg::Odometry> wheelOdomQueue_;
    bool wheelOdomActive_ = false;
    gtsam::Pose3 firstWheelOdomPose_;
    bool firstWheelOdomReceived_ = false;
    gtsam::Pose3 lastKfWheelRelPose_;  // guarded by slamMtx_

    // ═══ Pending ArUco (guarded by pendingMtx_) ═══
    aruco_sam_ailab::msg::MarkerArray pendingMarkers_;  // already in base_link
    gtsam::Pose3 pendingOdomPose_;
    nav_msgs::msg::Odometry pendingOdom_;
    rclcpp::Time pendingStamp_{0, 0, RCL_ROS_TIME};
    bool pendingArucoValid_ = false;

    // ═══ SLAM Snapshot for interpolation (guarded by snapMtx_) ═══
    gtsam::Pose3 snapEstimate_;
    gtsam::Pose3 snapKfOdomPose_;
    bool snapValid_ = false;

    // ═══ Landmarks (guarded by slamMtx_) ═══
    std::map<int, gtsam::Key> landmarkIdToKey_;

    // ═══ ROS Interface ═══
    rclcpp::Subscription<aruco_sam_ailab::msg::MarkerArray>::SharedPtr subAruco_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubGlobalOdom_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubWheelOdomCorrection_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLandmarks_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubDebugMap_;
    rclcpp::Publisher<aruco_sam_ailab::msg::OptimizedKeyframeState>::SharedPtr pubOptimizedState_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr saveLandmarksSrv_;
    std::string landmarksSavePath_;
    rclcpp::TimerBase::SharedPtr slamTimer_;

    // ═══ Callback Groups (enable true concurrency with MultiThreadedExecutor) ═══
    rclcpp::CallbackGroup::SharedPtr queueCbGroup_;   // IMU/odom subscribers
    rclcpp::CallbackGroup::SharedPtr arucoCbGroup_;    // ArUco subscriber (lightweight)
    rclcpp::CallbackGroup::SharedPtr timerCbGroup_;    // SLAM timer (heavy ISAM2)

    // ═══ Mode ═══
    bool isLocalizationMode_ = false;
    std::string mapPath_;
    gtsam::noiseModel::Diagonal::shared_ptr fixedLandmarkNoise_;

    nav_msgs::msg::Path globalPath_;  // only written by arucoHandler (arucoCbGroup_)
    std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;

    // ═══ Noise Models ═══
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr biasRandomWalkNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr odomFallbackNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr wheelOdomNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr obsNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr landmarkPriorNoise_;

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

        // 1a. IMU odom (for keyframe triggering & dead reckoning between keyframes)
        subImuOdom_ = create_subscription<nav_msgs::msg::Odometry>(
            odomTopic, 100,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(queueMtx_);
                imuOdomQueue_.push_back(*msg);
                if (imuOdomQueue_.size() > 200) imuOdomQueue_.pop_front();
            }, queueSubOpts);

        // 1b. Raw IMU (for ImuFactor preintegration in factor graph)
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
                rawImuQueue_.push_back(imu_base);
                if (rawImuQueue_.size() > 4000) rawImuQueue_.pop_front();
            }, queueSubOpts);

        // 1c. Wheel Odometry (optional — uses relative pose from first received)
        subWheelOdom_ = create_subscription<nav_msgs::msg::Odometry>(
            wheelOdomTopic, 100,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(queueMtx_);
                if (!firstWheelOdomReceived_) {
                    firstWheelOdomPose_ = poseMsgToGtsam(msg->pose.pose);
                    firstWheelOdomReceived_ = true;
                    wheelOdomActive_ = true;
                    RCLCPP_INFO(get_logger(), "Wheel odom active on %s (using relative from first pose)",
                                wheelOdomTopic.c_str());
                }
                wheelOdomQueue_.push_back(*msg);
                if (wheelOdomQueue_.size() > 200) wheelOdomQueue_.pop_front();
            }, queueSubOpts);

        // 1d. ArUco (lightweight — never blocks on ISAM2)
        subAruco_ = create_subscription<aruco_sam_ailab::msg::MarkerArray>(
            arucoPosesTopic, 10,
            std::bind(&GraphOptimizer::arucoHandler, this, std::placeholders::_1),
            arucoSubOpts);

        // ─── 2. Publishers ───
        pubGlobalOdom_ = create_publisher<nav_msgs::msg::Odometry>("/aruco_slam/odom", 10);
        pubWheelOdomCorrection_ = create_publisher<nav_msgs::msg::Odometry>("/odometry/wheel_odom_correction", 10);
        pubPath_ = create_publisher<nav_msgs::msg::Path>("/aruco_slam/path", 10);
        pubLandmarks_ = create_publisher<visualization_msgs::msg::MarkerArray>("/aruco_slam/landmarks", 10);
        pubDebugMap_ = create_publisher<visualization_msgs::msg::MarkerArray>("/aruco_slam/debug_markers", 10);
        pubOptimizedState_ = create_publisher<aruco_sam_ailab::msg::OptimizedKeyframeState>("/optimized_keyframe_state", 10);
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
        keyframeDistThresh_ = keyframeDistanceThreshold;
        keyframeAngleThresh_ = keyframeAngleThreshold;
        keyframeTimeThresh_ = keyframeTimeInterval;
        RCLCPP_INFO(get_logger(), "Keyframe policy: dist=%.2fm, angle=%.2frad, time=%.2fs",
                    keyframeDistThresh_, keyframeAngleThresh_, keyframeTimeThresh_);

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
        priorPoseNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001).finished());
        priorVelNoise_ = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
        priorBiasNoise_ = gtsam::noiseModel::Isotropic::Sigma(6, 0.1);  // 초기 bias 불확실성 허용

        // Bias random walk: σ_continuous * √(dt_keyframe) ≈ 0.0001*√0.5 is too tight
        // Scale up 100x to allow graph to actually correct bias
        biasRandomWalkNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << imuAccBiasN * 100, imuAccBiasN * 100, imuAccBiasN * 100,
                                  imuGyrBiasN * 100, imuGyrBiasN * 100, imuGyrBiasN * 100).finished());

        odomFallbackNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());

        wheelOdomNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << wheelOdomRotNoise, wheelOdomRotNoise, wheelOdomRotNoise,
                                  wheelOdomTransNoise, wheelOdomTransNoise, wheelOdomTransNoise).finished());

        obsNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << arucoRotNoise, arucoRotNoise, arucoRotNoise,
                                  arucoTransNoise, arucoTransNoise, arucoTransNoise).finished());
        // Landmark prior: 관측 noise와 비슷한 수준으로 조여서 초기 수렴 안정화
        landmarkPriorNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.15, 0.15, 0.15, 0.15, 0.15, 0.15).finished());

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
    //  Always publishes interpolated SLAM pose for EKF correction.
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

        // 2. Sync IMU odom (locks queueMtx_ internally, fast)
        nav_msgs::msg::Odometry currOdom;
        if (!getSyncedOdom(msg->header.stamp, currOdom)) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for IMU Odometry...");
            return;
        }
        gtsam::Pose3 currOdomPose = poseMsgToGtsam(currOdom.pose.pose);

        // 3. Store latest observation for SLAM timer (only when markers detected)
        if (!markersInBase.markers.empty()) {
            std::lock_guard<std::mutex> lock(pendingMtx_);
            pendingMarkers_ = markersInBase;
            pendingOdomPose_ = currOdomPose;
            pendingOdom_ = currOdom;
            pendingStamp_ = msg->header.stamp;
            pendingArucoValid_ = true;
        }

        // 4. Publish interpolated SLAM pose (~30Hz, even during ISAM2)
        gtsam::Pose3 snapEst;
        gtsam::Pose3 snapKfOdom;
        bool valid;
        {
            std::lock_guard<std::mutex> lock(snapMtx_);
            snapEst = snapEstimate_;
            snapKfOdom = snapKfOdomPose_;
            valid = snapValid_;
        }

        if (valid) {
            gtsam::Pose3 odomDelta = snapKfOdom.between(currOdomPose);
            gtsam::Pose3 interpolated = snapEst.compose(odomDelta);
            // TF + path + odom
            publishTFOnly(interpolated, msg->header.stamp, currOdomPose);
            // EKF correction (~30Hz instead of sporadic keyframe rate)
            publishCorrection(interpolated, msg->header.stamp);
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
        gtsam::Pose3 odomPose;
        nav_msgs::msg::Odometry odom;
        rclcpp::Time stamp(0, 0, RCL_ROS_TIME);
        bool hasPending = false;
        {
            std::lock_guard<std::mutex> lock(pendingMtx_);
            if (pendingArucoValid_) {
                markers = pendingMarkers_;
                odomPose = pendingOdomPose_;
                odom = pendingOdom_;
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
        if (!systemInitialized_) {
            if (isLocalizationMode_) {
                for (const auto& m : markers.markers) {
                    if (landmarkIdToKey_.count(static_cast<int>(m.id))) {
                        int mid = static_cast<int>(m.id);
                        gtsam::Pose3 L_map = isam_.calculateEstimate().at<gtsam::Pose3>(L(mid));
                        gtsam::Pose3 L_base = poseMsgToGtsam(m.pose);
                        gtsam::Pose3 initialPose = L_map.compose(L_base.inverse());
                        initializeSystemAt(initialPose, odom);
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
                initializeSystem(markers, odom);
                updateSnapshot();
                return;
            }
        }

        // 5. Keyframe decision + ISAM2 optimization
        if (needNewKeyframe(odomPose, stamp, currMarkerIds)) {
            processKeyframe(markers, odomPose, stamp);
            lastKeyframeOdomPose_ = odomPose;
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
        snapKfOdomPose_ = lastKeyframeOdomPose_;
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
    //  Wheel Odometry Sync (locks queueMtx_ internally)
    // ═══════════════════════════════════════════════════════════
    bool getSyncedWheelOdom(const rclcpp::Time& stamp, gtsam::Pose3& relPose) {
        std::lock_guard<std::mutex> lock(queueMtx_);

        if (!wheelOdomActive_ || wheelOdomQueue_.empty()) return false;

        double minDiff = std::numeric_limits<double>::max();
        size_t bestIdx = 0;
        for (size_t i = 0; i < wheelOdomQueue_.size(); ++i) {
            double diff = std::abs((rclcpp::Time(wheelOdomQueue_[i].header.stamp) - stamp).seconds());
            if (diff < minDiff) { minDiff = diff; bestIdx = i; }
        }
        if (minDiff > 0.5) return false;

        gtsam::Pose3 absPose = poseMsgToGtsam(wheelOdomQueue_[bestIdx].pose.pose);
        relPose = firstWheelOdomPose_.between(absPose);
        return true;
    }

    // ═══════════════════════════════════════════════════════════
    //  Keyframe Policy
    // ═══════════════════════════════════════════════════════════
    bool needNewKeyframe(const gtsam::Pose3& currOdomPose, const rclcpp::Time& currTime,
                         const std::vector<int>& currMarkerIds) {
        if (frameIdx_ == 0) return true;

        double timeDiff = (currTime - lastKeyframeTime_).seconds();
        bool hasMarkers = !currMarkerIds.empty();

        double minInterval = hasMarkers ? 0.1 : 0.2;
        if (timeDiff < minInterval) return false;

        if (hasMarkers) return true;

        gtsam::Pose3 delta = lastKeyframeOdomPose_.between(currOdomPose);
        if (delta.translation().norm() > keyframeDistThresh_) return true;
        double rotAngle = std::abs(delta.rotation().axisAngle().second);
        if (rotAngle > keyframeAngleThresh_) return true;

        if (timeDiff > keyframeTimeThresh_) return true;

        return false;
    }

    // ═══════════════════════════════════════════════════════════
    //  Initialization
    // ═══════════════════════════════════════════════════════════
    void initializeSystem(const aruco_sam_ailab::msg::MarkerArray& markers, const nav_msgs::msg::Odometry& odom) {
        initializeSystemAt(gtsam::Pose3(), odom);
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

    void initializeSystemAt(const gtsam::Pose3& startPose, const nav_msgs::msg::Odometry& odom) {
        currentEstimate_ = startPose;
        currentVelocity_ = gtsam::Vector3::Zero();
        lastOdomPose_ = poseMsgToGtsam(odom.pose.pose);
        lastKeyframeOdomPose_ = lastOdomPose_;
        lastKeyframeTime_ = odom.header.stamp;
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

        // Initialize wheel odom reference if available
        if (wheelOdomActive_) {
            gtsam::Pose3 wheelRel;
            if (getSyncedWheelOdom(odom.header.stamp, wheelRel)) {
                lastKfWheelRelPose_ = wheelRel;
            }
        }

        systemInitialized_ = true;
        frameIdx_ = 0;

        imuPreintegrator_->resetIntegrationAndSetBias(currentBias_);

        publishMapToOdomTF(currentEstimate_, lastOdomPose_, odom.header.stamp);
        publishCorrection(currentEstimate_, odom.header.stamp);

        RCLCPP_INFO(get_logger(), "System Initialized at pose (%.2f, %.2f, %.2f)",
                    startPose.translation().x(), startPose.translation().y(), startPose.translation().z());
    }

    // ═══════════════════════════════════════════════════════════
    //  Process Keyframe (ImuFactor + optional WheelOdom + ArUco)
    //  Called from slamTimerCallback under slamMtx_
    // ═══════════════════════════════════════════════════════════
    void processKeyframe(const aruco_sam_ailab::msg::MarkerArray& markers,
                         const gtsam::Pose3& currOdomPose, const rclcpp::Time& stamp) {
        frameIdx_++;

        // 1. IMU Preintegration: lastKeyframeTime_ → stamp
        //    (locks queueMtx_ internally)
        int imuCount = preintegrateImu(lastKeyframeTime_, stamp);

        // 2. Predict state using IMU or odom fallback
        gtsam::Pose3 predictedPose;
        gtsam::Vector3 predictedVel;

        if (imuCount > 0) {
            gtsam::NavState prevNavState(currentEstimate_, currentVelocity_);
            gtsam::NavState predictedState = imuPreintegrator_->predict(prevNavState, currentBias_);
            predictedPose = predictedState.pose();
            predictedVel = predictedState.velocity();
        } else {
            // Fallback: use odom delta
            gtsam::Pose3 odomDelta = lastKeyframeOdomPose_.between(currOdomPose);
            predictedPose = currentEstimate_.compose(odomDelta);
            predictedVel = currentVelocity_;
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "No IMU data for preintegration, using odom fallback");
        }

        // 3. Insert new variables
        graphValues_.insert(X(frameIdx_), predictedPose);
        graphValues_.insert(V(frameIdx_), predictedVel);
        graphValues_.insert(B(frameIdx_), currentBias_);

        // 4. ImuFactor or fallback BetweenFactor
        if (imuCount > 0) {
            graphFactors_.add(gtsam::ImuFactor(
                X(frameIdx_ - 1), V(frameIdx_ - 1),
                X(frameIdx_), V(frameIdx_),
                B(frameIdx_ - 1),
                *imuPreintegrator_));
        } else {
            gtsam::Pose3 odomDelta = lastKeyframeOdomPose_.between(currOdomPose);
            graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                X(frameIdx_ - 1), X(frameIdx_), odomDelta, odomFallbackNoise_));
        }

        // 5. Bias random walk constraint
        graphFactors_.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
            B(frameIdx_ - 1), B(frameIdx_),
            gtsam::imuBias::ConstantBias(),
            biasRandomWalkNoise_));

        // 6. Wheel Odometry BetweenFactor (optional, locks queueMtx_ internally)
        if (wheelOdomActive_) {
            gtsam::Pose3 currWheelRel;
            if (getSyncedWheelOdom(stamp, currWheelRel)) {
                gtsam::Pose3 wheelDelta = lastKfWheelRelPose_.between(currWheelRel);
                graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                    X(frameIdx_ - 1), X(frameIdx_), wheelDelta, wheelOdomNoise_));
                lastKfWheelRelPose_ = currWheelRel;
            }
        }

        // 7. Landmark factors
        addLandmarkFactors(frameIdx_, markers);

        // 8. Optimize
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

            if (enableTopicDebugLog) {
                RCLCPP_INFO(get_logger(),
                    "[ISAM2] frame=%d imu=%d vel=(%.2f,%.2f,%.2f) bias_acc=(%.4f,%.4f,%.4f) bias_gyr=(%.5f,%.5f,%.5f)",
                    frameIdx_, imuCount,
                    currentVelocity_.x(), currentVelocity_.y(), currentVelocity_.z(),
                    currentBias_.accelerometer().x(), currentBias_.accelerometer().y(), currentBias_.accelerometer().z(),
                    currentBias_.gyroscope().x(), currentBias_.gyroscope().y(), currentBias_.gyroscope().z());
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "ISAM2 Update Failed: %s", e.what());
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  Publishers
    // ═══════════════════════════════════════════════════════════
    void publishCorrection(const gtsam::Pose3& optimizedPose, const rclcpp::Time& stamp) {
        nav_msgs::msg::Odometry correctionMsg;
        correctionMsg.header.stamp = stamp;
        correctionMsg.header.frame_id = mapFrame;
        correctionMsg.child_frame_id = baseLinkFrame;
        correctionMsg.pose.pose = gtsamToPoseMsg(optimizedPose);
        for (int i = 0; i < 36; i++) correctionMsg.pose.covariance[i] = 0.0;
        correctionMsg.pose.covariance[0] = 0.01;
        correctionMsg.pose.covariance[7] = 0.01;
        correctionMsg.pose.covariance[14] = 0.01;
        correctionMsg.pose.covariance[21] = 0.01;
        correctionMsg.pose.covariance[28] = 0.01;
        correctionMsg.pose.covariance[35] = 0.01;
        pubWheelOdomCorrection_->publish(correctionMsg);
    }

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

    void publishMapToOdomTF(const gtsam::Pose3& mapToBase, const gtsam::Pose3& odomToBase, const rclcpp::Time& stamp) {
        gtsam::Pose3 mapToOdom = mapToBase.compose(odomToBase.inverse());
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp;
        t.header.frame_id = mapFrame;
        t.child_frame_id = odomFrame;
        t.transform.translation.x = mapToOdom.translation().x();
        t.transform.translation.y = mapToOdom.translation().y();
        t.transform.translation.z = mapToOdom.translation().z();
        auto q = mapToOdom.rotation().toQuaternion();
        t.transform.rotation.w = q.w();
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        tfBroadcaster_->sendTransform(t);
    }

    void publishTFOnly(const gtsam::Pose3& estimatedPose, const rclcpp::Time& stamp,
                      const gtsam::Pose3& odomToBase) {
        publishMapToOdomTF(estimatedPose, odomToBase, stamp);
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
            bool knownLandmark = (landmarkIdToKey_.find(mid) != landmarkIdToKey_.end());

            if (isLocalizationMode_) {
                if (!knownLandmark) continue;
                gtsam::Key landmarkKey = landmarkIdToKey_[mid];
                graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                    X(currentFrameIdx), landmarkKey, measurement, obsNoise_));
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
                    X(currentFrameIdx), landmarkKey, measurement, obsNoise_));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  Helpers
    // ═══════════════════════════════════════════════════════════

    // getSyncedOdom: locks queueMtx_ internally
    bool getSyncedOdom(const rclcpp::Time& stamp, nav_msgs::msg::Odometry& result) {
        std::lock_guard<std::mutex> lock(queueMtx_);

        if (imuOdomQueue_.empty()) return false;
        double minDiff = std::numeric_limits<double>::max();
        size_t bestIdx = 0;
        for (size_t i = 0; i < imuOdomQueue_.size(); ++i) {
            double diff = std::abs((rclcpp::Time(imuOdomQueue_[i].header.stamp) - stamp).seconds());
            if (diff < minDiff) { minDiff = diff; bestIdx = i; }
        }
        if (minDiff > 0.5) return false;
        result = imuOdomQueue_[bestIdx];
        return true;
    }

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
