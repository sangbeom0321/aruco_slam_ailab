#include "aruco_slam_ailab/utility.hpp"
#include "aruco_slam_ailab/msg/marker_array.hpp"
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Pose3.h>

#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <mutex>
#include <deque>
#include <set>
#include <vector>
#include <cmath>

using gtsam::symbol_shorthand::X; // Robot Pose (base_link)
using gtsam::symbol_shorthand::L; // Landmark Pose (World/Map frame)

namespace aruco_slam_ailab {

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

    // Keyframe 파라미터 (튜닝 가능)
    static constexpr double keyframeDistThresh_ = 0.3;   // 30cm
    static constexpr double keyframeAngleThresh_ = 0.2; // 약 11.5도 (rad)
    static constexpr double keyframeTimeThresh_ = 2.0;   // 2초 (Keep Alive)

    // Data Buffers & Maps
    std::deque<nav_msgs::msg::Odometry> odomQueue_;
    std::map<int, gtsam::Key> landmarkIdToKey_; // MarkerID -> GTSAM Key(L)

    // ROS Interface
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subWheelOdom_;
    rclcpp::Subscription<aruco_slam_ailab::msg::MarkerArray>::SharedPtr subAruco_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubGlobalOdom_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubWheelOdomCorrection_;  // Wheel Odom 보정 신호
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLandmarks_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubDebugMap_;

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

        subAruco_ = create_subscription<aruco_slam_ailab::msg::MarkerArray>(
            arucoPosesTopic, 10,
            std::bind(&GraphOptimizer::arucoHandler, this, std::placeholders::_1));

        // 2. Publishers
        pubGlobalOdom_ = create_publisher<nav_msgs::msg::Odometry>("/aruco_slam/odom", 10);
        pubWheelOdomCorrection_ = create_publisher<nav_msgs::msg::Odometry>("/odometry/wheel_odom_correction", 10);
        pubPath_ = create_publisher<nav_msgs::msg::Path>("/aruco_slam/path", 10);
        pubLandmarks_ = create_publisher<visualization_msgs::msg::MarkerArray>("/aruco_slam/landmarks", 10);
        pubDebugMap_ = create_publisher<visualization_msgs::msg::MarkerArray>("/aruco_slam/debug_markers", 10);
        tfBroadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

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

        RCLCPP_INFO(get_logger(), "Graph Optimizer Initialized.");
    }

    // 메인 콜백: ArUco 데이터가 들어오면 SLAM 수행
    void arucoHandler(const aruco_slam_ailab::msg::MarkerArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);

        // [중요] 1. 들어온 마커 데이터를 즉시 "로봇 좌표계(base_link)"로 변환
        // 이후 모든 로직은 Camera 좌표계를 신경 쓰지 않음.
        aruco_slam_ailab::msg::MarkerArray markersInBase;
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
            RCLCPP_WARN(get_logger(), "Waiting for Wheel Odometry...");
            return;
        }
        gtsam::Pose3 currOdomPose = poseMsgToGtsam(currOdom.pose.pose);

        // 3. 현재 보이는 마커 ID 추출
        std::vector<int> currMarkerIds;
        for (const auto& m : markersInBase.markers) currMarkerIds.push_back(m.id);

        // 4. SLAM Process
        if (!systemInitialized_) {
            initializeSystem(markersInBase, currOdom);
        } else {
            // [핵심] 키프레임인지 판단
            if (needNewKeyframe(currOdomPose, msg->header.stamp, currMarkerIds)) {
                // Case A: 키프레임 → 그래프 추가 + 최적화
                processKeyframe(markersInBase, currOdomPose, msg->header.stamp);
                lastKeyframeOdomPose_ = currOdomPose;
                lastKeyframeTime_ = msg->header.stamp;
                lastVisibleMarkers_.clear();
                for (int id : currMarkerIds) lastVisibleMarkers_.insert(id);
            } else {
                // Case B: 키프레임 아님 → Dead Reckoning (TF/Path만 부드럽게 발행)
                gtsam::Pose3 odomDelta = lastKeyframeOdomPose_.between(currOdomPose);
                gtsam::Pose3 estimatedPose = currentEstimate_.compose(odomDelta);
                publishTFOnly(estimatedPose, msg->header.stamp);
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
        if (delta.translation().norm() > keyframeDistThresh_) return true;  // 거리 30cm
        // 회전 각도 (axisAngle: axis, angle)
        double rotAngle = std::abs(delta.rotation().axisAngle().second);
        if (rotAngle > keyframeAngleThresh_) return true;  // 약 11.5도

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

    void initializeSystem(const aruco_slam_ailab::msg::MarkerArray& markers, const nav_msgs::msg::Odometry& odom) {
        // [핵심] 마커가 없어도 초기화 진행! (점프 현상 방지)
        // 로봇이 움직이는 동안 SLAM 좌표계도 Dead Reckoning으로 이동하고,
        // 나중에 마커 발견 시 연속적으로 이어짐 (0,0,0 강제 리셋 없음)

        // 시스템 원점을 (0,0,0)으로 고정
        currentEstimate_ = gtsam::Pose3();
        lastOdomPose_ = poseMsgToGtsam(odom.pose.pose);
        lastKeyframeOdomPose_ = lastOdomPose_;
        lastKeyframeTime_ = odom.header.stamp;
        lastVisibleMarkers_.clear();
        for (const auto& m : markers.markers) lastVisibleMarkers_.insert(m.id);

        // Prior Factor 추가 (X0 고정)
        graphValues_.insert(X(0), currentEstimate_);
        graphFactors_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), currentEstimate_, priorNoise_));

        // 첫 프레임의 랜드마크 추가
        addLandmarkFactors(0, markers);

        // 초기 최적화
        isam_.update(graphFactors_, graphValues_);
        graphFactors_.resize(0);
        graphValues_.clear();

        systemInitialized_ = true;
        frameIdx_ = 0;

        // 초기화 직후에도 보정 신호 전송 (0점에서 시작하라고 Wheel Odom에 알림)
        publishCorrection(currentEstimate_, odom.header.stamp);

        RCLCPP_INFO(get_logger(), "System Initialized at Frame 0 (Markers: %zu)", markers.markers.size());
    }

    void processKeyframe(const aruco_slam_ailab::msg::MarkerArray& markers,
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

            // 시각화, TF 발행 및 Wheel Odom 보정 신호 송출
            publishResults(stamp);
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

    void publishTFOnly(const gtsam::Pose3& estimatedPose, const rclcpp::Time& stamp) {
        // TF (map -> base_link)
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp;
        t.header.frame_id = mapFrame;
        t.child_frame_id = baseLinkFrame;
        t.transform.translation.x = estimatedPose.translation().x();
        t.transform.translation.y = estimatedPose.translation().y();
        t.transform.translation.z = estimatedPose.translation().z();
        auto q = estimatedPose.rotation().toQuaternion();
        t.transform.rotation.w = q.w();
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        tfBroadcaster_->sendTransform(t);

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
    void addLandmarkFactors(int currentFrameIdx, const aruco_slam_ailab::msg::MarkerArray& markers) {
        for (const auto& marker : markers.markers) {
            int mid = marker.id;
            // Measurement: T_base_marker (이미 변환됨)
            gtsam::Pose3 measurement = poseMsgToGtsam(marker.pose);

            // 랜드마크 키 관리
            gtsam::Key landmarkKey;
            if (landmarkIdToKey_.find(mid) == landmarkIdToKey_.end()) {
                // 새로운 랜드마크 발견!
                landmarkKey = L(mid); // ID를 그대로 사용하거나, 순차적으로 부여
                landmarkIdToKey_[mid] = landmarkKey;

                // 초기 위치 추정: T_map_marker = T_map_base * T_base_marker
                // 주의: 이때 currentEstimate_는 아직 최적화 전(Prediction) 상태일 수 있음.
                gtsam::Pose3 initialLandmarkPose;
                if(graphValues_.exists(X(currentFrameIdx))) {
                     initialLandmarkPose = graphValues_.at<gtsam::Pose3>(X(currentFrameIdx)).compose(measurement);
                } else {
                     initialLandmarkPose = currentEstimate_.compose(measurement);
                }

                graphValues_.insert(landmarkKey, initialLandmarkPose);

                // [특이점 방지] 처음 본 랜드마크에 약한 Prior를 걸어 계산 폭주 방지
                graphFactors_.add(gtsam::PriorFactor<gtsam::Pose3>(landmarkKey, initialLandmarkPose, landmarkPriorNoise_));
            } else {
                landmarkKey = landmarkIdToKey_[mid];
            }

            // Measurement Factor 추가 (BetweenFactor: Robot -> Landmark)
            graphFactors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                X(currentFrameIdx), landmarkKey, measurement, obsNoise_));
        }
    }

    // Helper: 시간 동기화
    bool getSyncedOdom(const rclcpp::Time& stamp, nav_msgs::msg::Odometry& result) {
        if (odomQueue_.empty()) return false;

        // 간단하게 가장 최신 것 사용 (실제로는 보간이나 근사 검색 필요)
        // 여기서는 예시로 가장 뒤에 있는(최신) 데이터를 씀
        result = odomQueue_.back();

        // 시간 차이가 너무 크면 무시 (0.5초 이상)
        double diff = std::abs((rclcpp::Time(result.header.stamp) - stamp).seconds());
        if (diff > 0.5) return false;

        return true;
    }

    // Helper: 시각화 모음
    void publishResults(const rclcpp::Time& stamp) {
        // 1. Path
        geometry_msgs::msg::PoseStamped p;
        p.header.stamp = stamp;
        p.header.frame_id = mapFrame; // Global Frame
        p.pose = gtsamToPoseMsg(currentEstimate_);
        globalPath_.header = p.header;
        globalPath_.poses.push_back(p);
        pubPath_->publish(globalPath_);

        // 2. TF (map -> base_link)
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp;
        t.header.frame_id = mapFrame;
        t.child_frame_id = baseLinkFrame; // SLAM 결과가 base_link의 위치임
        t.transform.translation.x = p.pose.position.x;
        t.transform.translation.y = p.pose.position.y;
        t.transform.translation.z = p.pose.position.z;
        t.transform.rotation = p.pose.orientation;
        tfBroadcaster_->sendTransform(t);

        // 3. Global Odometry
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
            sphere.pose.position = poseMsgMap.position;
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
            text.pose = poseMsgMap;
            text.pose.position.z += 0.15;
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
};

} // namespace aruco_slam_ailab

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);

    auto node = std::make_shared<aruco_slam_ailab::GraphOptimizer>(options);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Graph Optimizer Started.\033[0m");

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
