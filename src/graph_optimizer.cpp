#include "aruco_sam_ailab/utility.hpp"
#include "aruco_sam_ailab/msg/marker_array.hpp"
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Pose3.h>

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

using gtsam::symbol_shorthand::X; // Robot Pose (base_link)
using gtsam::symbol_shorthand::L; // Landmark Pose (World/Map frame)

namespace aruco_sam_ailab {

class GraphOptimizer : public ParamServer {
public:
    // Core Components
    std::mutex mtx_;
    gtsam::ISAM2 isam_;
    gtsam::NonlinearFactorGraph graphFactors_;
    gtsam::Values graphValues_;

    // Extrinsics & State
    gtsam::Pose3 baseToCam_;          // T_base_cam
    gtsam::Pose3 currentEstimate_;    // 현재 로봇 위치 (Map frame)
    gtsam::Pose3 lastOdomPose_;       // 직전 프레임의 오도메트리 값
    bool systemInitialized_ = false;
    int frameIdx_ = 0;                // Robot Pose Node Index (X0, X1...)

    // Keyframe Selection
    gtsam::Pose3 lastKeyframeOdomPose_;  // 마지막 키프레임 시점의 Odom 포즈
    rclcpp::Time lastKeyframeTime_{0};   // 마지막 키프레임 시간 (0 = 미초기화)
    std::set<int> lastVisibleMarkers_;    // 마지막에 보였던 마커 ID들

    // Keyframe Selection Thresholds (Fix 19)
    double keyframeDistThresh_;
    double keyframeAngleThresh_;
    double keyframeTimeThresh_;

    // TF Smoothing (Fix 18)
    double tfSmoothAlpha_;
    gtsam::Pose3 lastPublishedTF_;
    bool tfInitialized_ = false;

    // ZUPT 파라미터 (정지 판단 임계값)
    double zuptTransThresh_ = 0.01;   // 1cm
    double zuptRotThresh_ = 0.008;    // 약 0.5도 (rad)

    // Data Buffers & Maps
    std::deque<nav_msgs::msg::Odometry> odomQueue_;
    std::map<int, gtsam::Key> landmarkIdToKey_; // MarkerID -> GTSAM Key(L)

    // ============================================
    // [추가] 3D 포즈를 2D 평면(Z=0, Roll=0, Pitch=0)으로 강제 투영
    // ============================================
    gtsam::Pose3 flattenPose(const gtsam::Pose3& pose) const {
        double yaw = pose.rotation().yaw();
        double x = pose.translation().x();
        double y = pose.translation().y();
        return gtsam::Pose3(gtsam::Rot3::Rz(yaw), gtsam::Point3(x, y, 0.0));
    }


    // ROS Interface
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subWheelOdom_;
    rclcpp::Subscription<aruco_sam_ailab::msg::MarkerArray>::SharedPtr subAruco_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubGlobalOdom_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubWheelOdomCorrection_;  // Wheel Odom 보정 신호
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLandmarks_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubDebugMap_;

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr saveLandmarksSrv_;
    std::string landmarksSavePath_;

    // Mapping vs Localization
    bool isLocalizationMode_ = false;
    std::string mapPath_;
    gtsam::noiseModel::Diagonal::shared_ptr fixedLandmarkNoise_;

    nav_msgs::msg::Path globalPath_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;

    // Noise Models
    gtsam::noiseModel::Diagonal::shared_ptr priorNoise_; // 첫 시작 고정용
    gtsam::noiseModel::Diagonal::shared_ptr odomNoise_;  // 휠 오도메트리 신뢰도 (이동 시)
    gtsam::noiseModel::Diagonal::shared_ptr odomNoiseStationary_; // 정지 시: 휠 오돔 강하게 신뢰
    gtsam::noiseModel::Diagonal::shared_ptr obsNoise_;   // ArUco 관측 신뢰도
    gtsam::noiseModel::Diagonal::shared_ptr landmarkPriorNoise_; // 새 랜드마크 초기 고정용

    GraphOptimizer(const rclcpp::NodeOptions& options) : ParamServer("graph_optimizer", options) {
        // 1. Subscribers
        subWheelOdom_ = create_subscription<nav_msgs::msg::Odometry>(
            wheelOdomTopic, 100,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(mtx_);
                odomQueue_.push_back(*msg);
                if(odomQueue_.size() > 200) odomQueue_.pop_front();
            });

        subAruco_ = create_subscription<aruco_sam_ailab::msg::MarkerArray>(
            arucoPosesTopic, 10,
            std::bind(&GraphOptimizer::arucoHandler, this, std::placeholders::_1));

        // 2. Publishers
        pubGlobalOdom_ = create_publisher<nav_msgs::msg::Odometry>("/aruco_slam/odom", 10);
        pubWheelOdomCorrection_ = create_publisher<nav_msgs::msg::Odometry>("/odometry/wheel_odom_correction", 10);
        pubPath_ = create_publisher<nav_msgs::msg::Path>("/aruco_slam/path", 10);
        pubLandmarks_ = create_publisher<visualization_msgs::msg::MarkerArray>("/aruco_slam/landmarks", 10);
        pubDebugMap_ = create_publisher<visualization_msgs::msg::MarkerArray>("/aruco_slam/debug_markers", 10);
        tfBroadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // 2.5. Save Landmarks Service
        declare_parameter("landmarks_save_path", "landmarks_map.json");
        get_parameter("landmarks_save_path", landmarksSavePath_);
        saveLandmarksSrv_ = create_service<std_srvs::srv::Trigger>(
            "save_landmarks",
            std::bind(&GraphOptimizer::saveLandmarksCallback, this,
                      std::placeholders::_1, std::placeholders::_2));

    // Fix 19: Keyframe Threshold Parameterization
    // Thresholds are already declared in ParamServer constructor, just get values
    get_parameter("keyframe_distance_threshold", keyframeDistThresh_);
    get_parameter("keyframe_angle_threshold", keyframeAngleThresh_);
    get_parameter("keyframe_time_interval", keyframeTimeThresh_);

    // Fix 18: TF Smoothing
    tfSmoothAlpha_ = declare_parameter("tf_smooth_alpha", 0.3);

    RCLCPP_INFO(get_logger(), "GraphOptimizer Initialized (v5b: Param fixes applied).");
        // 2.6. Run Mode (mapping vs localization)
        declare_parameter("run_mode", "mapping");
        declare_parameter("map_path", "landmarks_map.json");

        // 2.7. ZUPT (Zero Motion Update) 임계값
        declare_parameter("zupt_trans_thresh", 0.01);
        declare_parameter("zupt_rot_thresh", 0.008);
        get_parameter("zupt_trans_thresh", zuptTransThresh_);
        get_parameter("zupt_rot_thresh", zuptRotThresh_);
        std::string runModeStr;
        get_parameter("run_mode", runModeStr);
        get_parameter("map_path", mapPath_);
        isLocalizationMode_ = (runModeStr == "localization");
        fixedLandmarkNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());

        // 3. GTSAM Settings
        gtsam::ISAM2Params params;
        params.relinearizeThreshold = 0.1;
        params.relinearizeSkip = 1;
        isam_ = gtsam::ISAM2(params);

        // 4. Noise Models Setup
        // (X, Y, Z, Roll, Pitch, Yaw)
        priorNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001).finished()); // 시작점 고정
        odomNoise_  = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // 이동 불확실성
        odomNoiseStationary_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001).finished()); // 정지 시: 휠 오돔 강하게 신뢰
        obsNoise_   = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());    // 관측 불확실성 (카메라 노이즈)
        landmarkPriorNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
             (gtsam::Vector(6) << 10.0, 10.0, 10.0, 10.0, 10.0, 10.0).finished()); // Weak Prior (첫 관측 시 특이점 방지)

        // 5. Extrinsic Setup (Camera -> Base)
        // 기존 코드의 extRot, extTrans 값을 그대로 사용
        gtsam::Rot3 rot(extRotBaseDepthCam);
        gtsam::Point3 trans(extTransBaseDepthCam.x(), extTransBaseDepthCam.y(), extTransBaseDepthCam.z());
        baseToCam_ = gtsam::Pose3(rot, trans);

        // 6. Localization Mode: Load map (fixed landmarks)
        if (isLocalizationMode_) {
            loadMap(mapPath_);
            RCLCPP_INFO(get_logger(), "Graph Optimizer Initialized (Localization Mode, %zu landmarks loaded).",
                        landmarkIdToKey_.size());
        } else {
            RCLCPP_INFO(get_logger(), "Graph Optimizer Initialized (Mapping Mode).");
        }
    }

    // 메인 콜백: ArUco 데이터가 들어오면 SLAM 수행
    void arucoHandler(const aruco_sam_ailab::msg::MarkerArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);

        // === [추가] 시간 역행(Time Jump) 감지 및 ISAM2 완전 리셋 로직 ===
        static double last_msg_sec = 0.0;
        double current_sec = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        
        if (last_msg_sec > 0.0 && current_sec < last_msg_sec) {
            RCLCPP_WARN(get_logger(), "\033[1;31mTime jump detected in SLAM! Initiating Total Reset.\033[0m");

            // 1. GTSAM ISAM2 최적화기 완전 파괴 및 재생성
            gtsam::ISAM2Params params;
            params.relinearizeThreshold = 0.1;
            params.relinearizeSkip = 1;
            isam_ = gtsam::ISAM2(params);

            // 2. 그래프와 변수 모두 삭제
            graphFactors_.resize(0);
            graphValues_.clear();

            // 3. 큐와 마커 기록 초기화
            odomQueue_.clear();
            lastVisibleMarkers_.clear();

            // 4. 상태 플래그 초기화
            systemInitialized_ = false;
            tfInitialized_ = false;
            frameIdx_ = 0;

            // 5. Localization 모드라면 고정된 랜드마크 맵을 다시 불러옴
            if (isLocalizationMode_) {
                loadMap(mapPath_);
            }
            // [추가] TF 트리가 끊어지는 것을 방지하기 위해 원점 TF를 즉시 발행
            publishMapToOdomTF(gtsam::Pose3(), gtsam::Pose3(), msg->header.stamp);
            lastPublishedTF_ = gtsam::Pose3(); // TF 스무딩 변수도 리셋

            last_msg_sec = current_sec;
            return; // 꼬여버린 이번 프레임은 버리고 탈출
        }
        last_msg_sec = current_sec;

        // [중요] 1. 들어온 마커 데이터를 즉시 "로봇 좌표계(base_link)"로 변환
        // 이후 모든 로직은 Camera 좌표계를 신경 쓰지 않음.
        aruco_sam_ailab::msg::MarkerArray markersInBase;
        markersInBase.header = msg->header;

        for (const auto& rawMarker : msg->markers) {
            // Raw: Camera Optical Frame
            gtsam::Pose3 poseInCam = poseMsgToGtsam(rawMarker.pose);
            // Transformed: Base Link Frame
            gtsam::Pose3 poseInBase = baseToCam_.compose(poseInCam);

            auto newMarker = rawMarker;
            newMarker.pose = gtsamToPoseMsg(poseInBase);
            markersInBase.markers.push_back(newMarker);
        }

        // 2. 오도메트리 동기화 (가장 가까운 시간의 휠 오도메트리 찾기)
        nav_msgs::msg::Odometry currOdom;
        if (!getSyncedOdom(msg->header.stamp, currOdom)) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "wheel_odom 동기화 대기 중 (queue=%zu개, ArUco_stamp=%.3fs)",
                         odomQueue_.size(), rclcpp::Time(msg->header.stamp).seconds());
            return;
        }
        gtsam::Pose3 currOdomPose = poseMsgToGtsam(currOdom.pose.pose);

        // 3. 현재 보이는 마커 ID 추출
        std::vector<int> currMarkerIds;
        for (const auto& m : markersInBase.markers) currMarkerIds.push_back(m.id);

        // [ZUPT] 3.5. 정지 상태 판단: 오도메트리 변화량이 극히 작으면 ArUco 관측 무시
        bool isStationary = false;
        if (systemInitialized_) {
            gtsam::Pose3 odomDelta = lastKeyframeOdomPose_.between(currOdomPose);
            double transNorm = odomDelta.translation().norm();
            double rotAngle = std::abs(odomDelta.rotation().axisAngle().second);
            // ZUPT: 임계값 미만 움직임이면 정지 상태로 간주
            if (transNorm < zuptTransThresh_ && rotAngle < zuptRotThresh_) {
                isStationary = true;
            }
        }

        // 4. SLAM Process
        if (!systemInitialized_) {
            if (isLocalizationMode_) {
                // 아는 마커가 보일 때까지 대기, 보이면 역산으로 초기화
                for (const auto& m : markersInBase.markers) {
                    if (landmarkIdToKey_.count(static_cast<int>(m.id))) {
                        int mid = static_cast<int>(m.id);
                        gtsam::Pose3 L_map = isam_.calculateEstimate().at<gtsam::Pose3>(L(mid));
                        gtsam::Pose3 L_base = poseMsgToGtsam(m.pose);
                        gtsam::Pose3 initialPose = L_map.compose(L_base.inverse());
                        initializeSystemAt(initialPose, currOdom);
                        lastVisibleMarkers_.clear();
                        for (const auto& mk : markersInBase.markers) lastVisibleMarkers_.insert(mk.id);
                        addLandmarkFactors(0, markersInBase);
                        isam_.update(graphFactors_, graphValues_);
                        graphFactors_.resize(0);
                        graphValues_.clear();
                        currentEstimate_ = isam_.calculateEstimate().at<gtsam::Pose3>(X(0));
                        publishResults(msg->header.stamp, lastOdomPose_);
                        return;
                    }
                }
                // 디버그: 감지된 ID vs 맵에 있는 ID
                std::string detectedIds, mapIds;
                for (const auto& m : markersInBase.markers) detectedIds += std::to_string(m.id) + " ";
                for (const auto& p : landmarkIdToKey_) mapIds += std::to_string(p.first) + " ";
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "Localization: Waiting for known marker. Detected: [%s] | Map has: [%s]",
                    detectedIds.empty() ? "(none)" : detectedIds.c_str(),
                    mapIds.empty() ? "(load failed?)" : mapIds.c_str());
                return;
            } else {
                initializeSystem(markersInBase, currOdom);
            }
        } else {
            // [ZUPT] 정지 상태면 ArUco Factor 추가/최적화 생략 → 카메라 노이즈로 인한 진동 방지
            if (isStationary) {
                gtsam::Pose3 odomDelta = lastKeyframeOdomPose_.between(currOdomPose);
                gtsam::Pose3 estimatedPose = currentEstimate_.compose(odomDelta);
                publishTFOnly(estimatedPose, msg->header.stamp, currOdomPose);
                return;
            }

            // [핵심] 키프레임인지 판단
            if (needNewKeyframe(currOdomPose, msg->header.stamp, currMarkerIds)) {
                // Case A: 키프레임 → 그래프 추가 + 최적화
                processKeyframe(markersInBase, currOdomPose, msg->header.stamp);
                lastKeyframeOdomPose_ = currOdomPose;
                lastKeyframeTime_ = msg->header.stamp;
                lastVisibleMarkers_.clear();
                for (int id : currMarkerIds) lastVisibleMarkers_.insert(id);
            } else {
                // Case B: 키프레임 아님 → Dead Reckoning (TF map->odom, Path 부드럽게 발행)
                gtsam::Pose3 odomDelta = lastKeyframeOdomPose_.between(currOdomPose);
                gtsam::Pose3 estimatedPose = currentEstimate_.compose(odomDelta);
                publishTFOnly(estimatedPose, msg->header.stamp, currOdomPose);
            }
        }
    }

    bool needNewKeyframe(const gtsam::Pose3& currOdomPose, const rclcpp::Time& currTime,
                         const std::vector<int>& currMarkerIds) {
        // 1. 첫 프레임이면 무조건 True
        if (frameIdx_ == 0) return true;

        // 2. 시간 체크 (너무 빠르면 무조건 Skip - 0.2초 미만은 무시)
        double timeDiff = (currTime - lastKeyframeTime_).seconds();
        if (timeDiff < 0.2) return false;

        // 3. 이동량 체크 (Odom 기준)
        gtsam::Pose3 delta = lastKeyframeOdomPose_.between(currOdomPose);
        if (delta.translation().norm() > keyframeDistThresh_) return true;
        // 회전 각도 (axisAngle: axis, angle)
        double rotAngle = std::abs(delta.rotation().axisAngle().second);
        if (rotAngle > keyframeAngleThresh_) return true;

        // 4. 마커 구성 변경 체크 (새로운 마커 등장 시)
        for (int id : currMarkerIds) {
            if (lastVisibleMarkers_.find(id) == lastVisibleMarkers_.end()) {
                return true;  // 못 보던 마커가 나타나면 키프레임 추가!
            }
        }

        // 5. 너무 오래 지났으면 추가 (Keep Alive)
        if (timeDiff > keyframeTimeThresh_) return true;

        return false;
    }


    void initializeSystem(const aruco_sam_ailab::msg::MarkerArray& markers, const nav_msgs::msg::Odometry& odom) {
        initializeSystemAt(gtsam::Pose3(), odom);
        lastVisibleMarkers_.clear();
        for (const auto& m : markers.markers) lastVisibleMarkers_.insert(m.id);
        addLandmarkFactors(0, markers);
        isam_.update(graphFactors_, graphValues_);
        graphFactors_.resize(0);
        graphValues_.clear();
        RCLCPP_INFO(get_logger(), "System Initialized at Frame 0 (Markers: %zu)", markers.markers.size());
    }

    void initializeSystemAt(const gtsam::Pose3& startPose, const nav_msgs::msg::Odometry& odom) {
        // [수정] 들어온 시작 위치를 무조건 2D로 먼저 완벽하게 투영
        gtsam::Pose3 flatStartPose = flattenPose(startPose);
        
        currentEstimate_ = flatStartPose;
        lastOdomPose_ = poseMsgToGtsam(odom.pose.pose);
        lastKeyframeOdomPose_ = lastOdomPose_;
        lastKeyframeTime_ = odom.header.stamp;
        lastVisibleMarkers_.clear();

        // [수정] 원본 startPose 대신 평탄화된 flatStartPose를 그래프에 삽입
        graphValues_.insert(X(0), flatStartPose);
        graphFactors_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), flatStartPose, priorNoise_));

        systemInitialized_ = true;
        frameIdx_ = 0;

        publishMapToOdomTF(currentEstimate_, lastOdomPose_, odom.header.stamp);
        publishCorrection(currentEstimate_, odom.header.stamp);

        RCLCPP_INFO(get_logger(), "System Initialized at flat pose (%.2f, %.2f, 0.0)",
                    flatStartPose.translation().x(), flatStartPose.translation().y());
    }

    void processKeyframe(const aruco_sam_ailab::msg::MarkerArray& markers,
                         const gtsam::Pose3& currOdomPose, const rclcpp::Time& stamp) {
        frameIdx_++;

        // 1. Odometry Factor (직전 키프레임 Odom ~ 현재 Odom)
        gtsam::Pose3 odomDelta = lastKeyframeOdomPose_.between(currOdomPose);

        // 2. 현재 로봇 위치 예측 (Prediction)
        gtsam::Pose3 predictedPose = currentEstimate_.compose(odomDelta);
        graphValues_.insert(X(frameIdx_), predictedPose);

        // 3. Odometry Factor 추가 (정지 시 휠 오돔 강하게 신뢰 → landmark 노이즈로 인한 흔들림 방지)
        double transNorm = odomDelta.translation().norm();
        double rotAngle = std::abs(odomDelta.rotation().axisAngle().second);
        const double stationaryThresh = 0.02;  // 2cm, ~1deg
        auto odomNoise = (transNorm < stationaryThresh && rotAngle < stationaryThresh)
            ? odomNoiseStationary_ : odomNoise_;
        graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
            X(frameIdx_ - 1), X(frameIdx_), odomDelta, odomNoise));

        // 4. Landmark Factor 추가 (이미 base_link로 변환된 마커들 사용)
        addLandmarkFactors(frameIdx_, markers);

        // 5. 최적화 수행
        try {
            isam_.update(graphFactors_, graphValues_);
            graphFactors_.resize(0);
            graphValues_.clear();

            // 결과 추출
            gtsam::Values result = isam_.calculateEstimate();
            currentEstimate_ = result.at<gtsam::Pose3>(X(frameIdx_));

            // [추가] 최적화된 3D 추정치를 강제로 지면(2D)으로 끌어내림
            currentEstimate_ = flattenPose(currentEstimate_);


            // 시각화, TF(map->odom) 발행 및 Wheel Odom 보정 신호 송출
            publishResults(stamp, currOdomPose);
            publishCorrection(currentEstimate_, stamp);

        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "ISAM2 Update Failed: %s", e.what());
        }
    }

    void publishCorrection(const gtsam::Pose3& optimizedPose, const rclcpp::Time& stamp) {
        nav_msgs::msg::Odometry correctionMsg;
        correctionMsg.header.stamp = stamp;
        correctionMsg.header.frame_id = mapFrame;
        correctionMsg.child_frame_id = baseLinkFrame;
        correctionMsg.pose.pose = gtsamToPoseMsg(optimizedPose);

        // Covariance (필요 시 확장)
        for (int i = 0; i < 36; i++) correctionMsg.pose.covariance[i] = 0.0;
        correctionMsg.pose.covariance[0] = 0.01;   // x
        correctionMsg.pose.covariance[7] = 0.01;   // y
        correctionMsg.pose.covariance[14] = 0.01;  // z
        correctionMsg.pose.covariance[21] = 0.01;  // roll
        correctionMsg.pose.covariance[28] = 0.01;  // pitch
        correctionMsg.pose.covariance[35] = 0.01;  // yaw

        pubWheelOdomCorrection_->publish(correctionMsg);
    }

    void publishMapToOdomTF(const gtsam::Pose3& map_to_base, const gtsam::Pose3& odom_to_base,
                          const rclcpp::Time& stamp) {
    // REP-105: map->odom = map->base * (odom->base)^-1
    gtsam::Pose3 mapToOdomRaw = map_to_base.compose(odom_to_base.inverse());

    // [추가] TF 트리 연산 결과도 무조건 2D로 고정
        mapToOdomRaw = flattenPose(mapToOdomRaw);


    // Fix 18: TF Smoothing
    gtsam::Pose3 mapToOdom = tfInitialized_ ? smoothPose(lastPublishedTF_, mapToOdomRaw) : mapToOdomRaw;

    lastPublishedTF_ = mapToOdom;
    tfInitialized_ = true;

    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = mapFrame;
    t.child_frame_id = odomFrame;
    t.transform.translation.x = mapToOdom.translation().x();
    t.transform.translation.y = mapToOdom.translation().y();
    t.transform.translation.z = mapToOdom.translation().z();
    gtsam::Quaternion q = mapToOdom.rotation().toQuaternion();
    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();

    tfBroadcaster_->sendTransform(t);
  }

    gtsam::Pose3 smoothPose(const gtsam::Pose3& prev, const gtsam::Pose3& curr) const {
    const double dist = (curr.translation() - prev.translation()).norm();
    const double alpha = (dist < 0.05) ? 1.0 : tfSmoothAlpha_;

    // 1. auto 대신 명시적으로 gtsam::Point3 타입으로 연산 결과를 받습니다.
    gtsam::Point3 t_raw = prev.translation() + alpha * (curr.translation() - prev.translation());

    // 2. Z축이 들썩이지 않도록 강제 평탄화한 새로운 변수 t_flat을 만듭니다.
    gtsam::Point3 t_flat(t_raw.x(), t_raw.y(), 0.0);

    // Rotation smoothing (2D yaw focus)
    const double yawPrev = prev.rotation().yaw();
    const double yawCurr = curr.rotation().yaw();
    double yawDiff = yawCurr - yawPrev;
    while (yawDiff > M_PI) yawDiff -= 2.0 * M_PI;
    while (yawDiff < -M_PI) yawDiff += 2.0 * M_PI;

    const double yaw = yawPrev + alpha * yawDiff;

    // 3. 최종 반환 시 평탄화된 t_flat을 사용합니다.
    return gtsam::Pose3(gtsam::Rot3::Rz(yaw), t_flat);
  }


    void publishTFOnly(const gtsam::Pose3& estimatedPose, const rclcpp::Time& stamp,
                      const gtsam::Pose3& odomToBase) {
        publishMapToOdomTF(estimatedPose, odomToBase, stamp);

        // [추가] 추정된 위치를 발행하기 전에 2D로 깎아냄
        gtsam::Pose3 flatEstimatedPose = flattenPose(estimatedPose);

        publishMapToOdomTF(flatEstimatedPose, odomToBase, stamp);



        // Path (부드러운 시각화용)
        geometry_msgs::msg::PoseStamped p;
        p.header.stamp = stamp;
        p.header.frame_id = mapFrame;
        p.pose = gtsamToPoseMsg(estimatedPose);
        globalPath_.header = p.header;
        globalPath_.poses.push_back(p);
        pubPath_->publish(globalPath_);

        // Global Odometry
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = mapFrame;
        odom.child_frame_id = baseLinkFrame;
        odom.pose.pose = p.pose;
        pubGlobalOdom_->publish(odom);
    }

    // [핵심 수정] 큐 탐색 없이 인자로 받은 마커(이미 base_link 기준)를 바로 사용
    void addLandmarkFactors(int currentFrameIdx, const aruco_sam_ailab::msg::MarkerArray& markers) {
        for (const auto& marker : markers.markers) {
            int mid = marker.id;
            gtsam::Pose3 measurement = poseMsgToGtsam(marker.pose);  // T_base_marker
            bool knownLandmark = (landmarkIdToKey_.find(mid) != landmarkIdToKey_.end());

            if (isLocalizationMode_) {
                // Localization: 지도에 있는 마커만 사용 (로봇 위치 보정용)
                if (!knownLandmark) continue;  // 모르는 마커 무시
                gtsam::Key landmarkKey = landmarkIdToKey_[mid];
                graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                    X(currentFrameIdx), landmarkKey, measurement, obsNoise_));
            } else {
                // Mapping: 기존 로직 (새 랜드마크 추가 가능)
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

    // Helper: 시간 동기화
    bool getSyncedOdom(const rclcpp::Time& stamp, nav_msgs::msg::Odometry& result) {
        if (odomQueue_.empty()) return false;

        // 가장 가까운 시간의 odom 탐색 (back()만 쓰면 최신 데이터가 오래된 마커 타임스탬프와 불일치)
        double minDiff = std::numeric_limits<double>::max();
        size_t bestIdx = 0;
        for (size_t i = 0; i < odomQueue_.size(); ++i) {
            double diff = std::abs((rclcpp::Time(odomQueue_[i].header.stamp) - stamp).seconds());
            if (diff < minDiff) {
                minDiff = diff;
                bestIdx = i;
            }
        }

        // Gazebo 초기화 지연 대비: 2.0초까지 허용 (기존 0.5초에서 완화)
        if (minDiff > 2.0) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "Wheel odom time gap too large: %.3fs (ArUco: %.3f, Odom: %.3f)",
                minDiff, stamp.seconds(),
                rclcpp::Time(odomQueue_[bestIdx].header.stamp).seconds());
            return false;
        }

        result = odomQueue_[bestIdx];
        return true;
    }

    // Helper: 시각화 모음
    void publishResults(const rclcpp::Time& stamp, const gtsam::Pose3& odomToBase) {
        // 1. Path (map 기준)
        geometry_msgs::msg::PoseStamped p;
        p.header.stamp = stamp;
        p.header.frame_id = mapFrame;
        p.pose = gtsamToPoseMsg(currentEstimate_);
        globalPath_.header = p.header;
        globalPath_.poses.push_back(p);
        pubPath_->publish(globalPath_);

        // 2. TF (map -> odom) REP-105: 제어는 odom->base만 사용
        publishMapToOdomTF(currentEstimate_, odomToBase, stamp);

        // 3. Global Odometry (map 기준, 경로 계획용)
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = mapFrame;
        odom.child_frame_id = baseLinkFrame;
        odom.pose.pose = p.pose;
        pubGlobalOdom_->publish(odom);

        // 4. Landmarks
        publishLandmarks(stamp);
    }

    void publishLandmarks(const rclcpp::Time& stamp) {
        visualization_msgs::msg::MarkerArray arr;
        if (landmarkIdToKey_.empty()) {
            pubLandmarks_->publish(arr);
            return;
        }
        gtsam::Values est;
        try {
            est = isam_.calculateEstimate();
        } catch (...) {
            pubLandmarks_->publish(arr);
            return;
        }
        for (const auto& [markerId, landmarkKey] : landmarkIdToKey_) {
            if (!est.exists(landmarkKey)) continue;
            gtsam::Pose3 poseMap = est.at<gtsam::Pose3>(landmarkKey);
            geometry_msgs::msg::Pose poseMsgMap = gtsamToPoseMsg(poseMap);

            visualization_msgs::msg::Marker sphere;
            sphere.header.stamp = stamp;
            sphere.header.frame_id = mapFrame;
            sphere.ns = "landmarks";
            sphere.id = markerId;
            sphere.type = visualization_msgs::msg::Marker::SPHERE;
            sphere.action = visualization_msgs::msg::Marker::ADD;
            sphere.pose.position.x = poseMap.translation().x();
            sphere.pose.position.y = poseMap.translation().y();
            sphere.pose.position.z = poseMap.translation().z();
            sphere.pose.orientation.w = 1.0;
            sphere.pose.orientation.x = sphere.pose.orientation.y = sphere.pose.orientation.z = 0.0;
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.2;
            sphere.color.r = 0.2f;
            sphere.color.g = 0.6f;
            sphere.color.b = 1.0f;
            sphere.color.a = 0.9f;
            sphere.lifetime = rclcpp::Duration(0, 0);
            arr.markers.push_back(sphere);

            visualization_msgs::msg::Marker text;
            text.header.stamp = stamp;
            text.header.frame_id = mapFrame;
            text.ns = "landmark_ids";
            text.id = markerId;
            text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text.action = visualization_msgs::msg::Marker::ADD;
            text.pose.position.x = poseMap.translation().x();
            text.pose.position.y = poseMap.translation().y();
            text.pose.position.z = poseMap.translation().z() + 0.15;
            text.pose.orientation = poseMsgMap.orientation;
            text.scale.z = 0.3;
            text.color.r = 1.0f;
            text.color.g = 1.0f;
            text.color.b = 1.0f;
            text.color.a = 1.0f;
            text.text = std::to_string(markerId);
            text.lifetime = rclcpp::Duration(0, 0);
            arr.markers.push_back(text);
        }
        pubLandmarks_->publish(arr);
    }

    // JSON에서 숫자 추출: "key": value 형태
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
        if (!file.is_open()) {
            RCLCPP_ERROR(get_logger(), "loadMap: Failed to open %s", filename.c_str());
            return;
        }
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        size_t landmarksPos = content.find("\"landmarks\"");
        if (landmarksPos == std::string::npos) {
            RCLCPP_ERROR(get_logger(), "loadMap: No 'landmarks' key in %s", filename.c_str());
            return;
        }
        size_t arrStart = content.find('[', landmarksPos);
        if (arrStart == std::string::npos) return;

        graphFactors_.resize(0);
        graphValues_.clear();
        landmarkIdToKey_.clear();

        size_t pos = arrStart + 1;
        while (pos < content.size()) {
            size_t objStart = content.find('{', pos);
            if (objStart == std::string::npos) break;
            // 중첩된 {} 매칭: position{"x":..} 등으로 첫 }가 잘못된 객체 끝이 됨
            int depth = 1;
            size_t objEnd = objStart + 1;
            for (; objEnd < content.size() && depth > 0; objEnd++) {
                if (content[objEnd] == '{') depth++;
                else if (content[objEnd] == '}') depth--;
            }
            if (depth != 0) break;

            std::string obj = content.substr(objStart, objEnd - objStart + 1);
            int id = extractJsonInt(obj, "id");
            size_t posIdx = obj.find("\"position\"");
            size_t oriIdx = obj.find("\"orientation\"");
            if (posIdx == std::string::npos || oriIdx == std::string::npos) {
                RCLCPP_WARN(get_logger(), "loadMap: Skip landmark id=%d (missing position/orientation)", id);
                pos = objEnd + 1;
                continue;
            }
            std::string posBlock = obj.substr(posIdx, oriIdx - posIdx);
            std::string oriBlock = obj.substr(oriIdx);
            double x = extractJsonDouble(posBlock, "x");
            double y = extractJsonDouble(posBlock, "y");
            double z = extractJsonDouble(posBlock, "z");
            double qw = extractJsonDouble(oriBlock, "w");
            double qx = extractJsonDouble(oriBlock, "x");
            double qy = extractJsonDouble(oriBlock, "y");
            double qz = extractJsonDouble(oriBlock, "z");

            gtsam::Key key = L(id);
            landmarkIdToKey_[id] = key;
            gtsam::Pose3 pose(gtsam::Rot3::Quaternion(qw, qx, qy, qz), gtsam::Point3(x, y, z));
            graphValues_.insert(key, pose);
            graphFactors_.add(gtsam::PriorFactor<gtsam::Pose3>(key, pose, fixedLandmarkNoise_));

            pos = objEnd + 1;
        }

        isam_.update(graphFactors_, graphValues_);
        graphFactors_.resize(0);
        graphValues_.clear();
        std::string idsStr;
        for (const auto& p : landmarkIdToKey_) idsStr += std::to_string(p.first) + " ";
        RCLCPP_INFO(get_logger(), "loadMap: Loaded %zu landmarks from %s [ids: %s]", landmarkIdToKey_.size(),
                    filename.c_str(), idsStr.c_str());
    }

    void saveLandmarksCallback(const std_srvs::srv::Trigger::Request::SharedPtr /*request*/,
                               std_srvs::srv::Trigger::Response::SharedPtr response) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (landmarkIdToKey_.empty()) {
            response->success = false;
            response->message = "No landmarks to save.";
            RCLCPP_WARN(get_logger(), "save_landmarks: No landmarks in graph.");
            return;
        }
        gtsam::Values est;
        try {
            est = isam_.calculateEstimate();
        } catch (const std::exception& e) {
            response->success = false;
            response->message = std::string("Failed to get estimate: ") + e.what();
            RCLCPP_ERROR(get_logger(), "save_landmarks: %s", e.what());
            return;
        }
        std::ostringstream json;
        json << "{\n";
        json << "  \"frame_id\": \"" << mapFrame << "\",\n";
        json << "  \"timestamp\": \"" << now().seconds() << "\",\n";
        json << "  \"landmarks\": [\n";
        bool first = true;
        size_t savedCount = 0;
        for (const auto& [markerId, landmarkKey] : landmarkIdToKey_) {
            if (!est.exists(landmarkKey)) continue;
            gtsam::Pose3 poseMap = est.at<gtsam::Pose3>(landmarkKey);
            auto q = poseMap.rotation().toQuaternion();
            if (!first) json << ",\n";
            json << "    {\n";
            json << "      \"id\": " << markerId << ",\n";
            json << "      \"position\": {\"x\": " << poseMap.translation().x() << ", \"y\": "
                 << poseMap.translation().y() << ", \"z\": " << poseMap.translation().z() << "},\n";
            json << "      \"orientation\": {\"w\": " << q.w() << ", \"x\": " << q.x()
                 << ", \"y\": " << q.y() << ", \"z\": " << q.z() << "}\n";
            json << "    }";
            first = false;
            savedCount++;
        }
        json << "\n  ]\n}\n";
        std::ofstream ofs(landmarksSavePath_);
        if (!ofs) {
            response->success = false;
            response->message = "Failed to open file: " + landmarksSavePath_;
            RCLCPP_ERROR(get_logger(), "save_landmarks: Cannot open %s", landmarksSavePath_.c_str());
            return;
        }
        ofs << json.str();
        ofs.close();
        response->success = true;
        response->message = "Saved " + std::to_string(savedCount) +
                           " landmarks to " + landmarksSavePath_;
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
