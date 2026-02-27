#include "aruco_sam_ailab/utility.hpp"
#include "aruco_sam_ailab/msg/optimized_keyframe_state.hpp"

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

using gtsam::symbol_shorthand::X; // Pose3
using gtsam::symbol_shorthand::V; // Velocity
using gtsam::symbol_shorthand::B; // Bias

namespace aruco_sam_ailab {

class IMUPreintegration : public ParamServer {
public:
    std::mutex mtx_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu_;
    rclcpp::Subscription<aruco_sam_ailab::msg::OptimizedKeyframeState>::SharedPtr subOptimizedKeyframeState_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;

    bool systemInitialized_ = false;
    std::string runMode_ = "mapping";

    // GTSAM noise models
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise_;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise_;

    // IMU preintegration
    gtsam::PreintegratedImuMeasurements* imuIntegratorImu_;

    // IMU data buffer
    std::deque<sensor_msgs::msg::Imu> imuQueue_;

    // Previous state
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    double lastImuTime_ = -1.0;
    int keyframeIdx_ = 0;

    // Stationary detection for velocity reset (accelerometer-based)
    std::deque<double> recentAccNorm_;
    static constexpr size_t ACC_WINDOW = 100;            // 0.5s at 200Hz (faster response)
    static constexpr double ACC_STATIONARY_THRESH = 0.15; // m/s² std-dev (lower = more sensitive)
    static constexpr double GYRO_STATIONARY_THRESH = 0.05; // rad/s (magnitude)
    static constexpr double VELOCITY_MAX = 1.0;           // max reasonable velocity m/s
    static constexpr double VELOCITY_DECAY = 0.995;       // per-step decay: 0.995^200 ≈ 0.37 after 1s

    // Logging: topic receive check
    uint64_t imuCallbackCount_ = 0;
    bool imuFirstReceivedLogged_ = false;

    // IMU Low-Pass Filter state
    Eigen::Vector3d lpfAcc_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d lpfGyr_ = Eigen::Vector3d::Zero();
    bool lpfInitialized_ = false;

    // Initial bias estimation (from stationary period)
    double initStartTime_ = -1.0;
    std::vector<Eigen::Vector3d> initGyroSamples_;
    std::vector<Eigen::Vector3d> initAccelSamples_;
    static constexpr double INIT_STATIONARY_DURATION = 1.0;  // 1 second stationary period

    // ISAM2 for incremental optimization (optional, mainly for prediction)
    gtsam::ISAM2 optimizer_;
    gtsam::NonlinearFactorGraph graphFactors_;
    gtsam::Values graphValues_;

    // Extrinsic: base_link to imu
    gtsam::Pose3 baseToImu_;

    IMUPreintegration(const rclcpp::NodeOptions& options) : ParamServer("imu_preintegration", options) {
        declare_parameter("run_mode", "mapping");
        get_parameter("run_mode", runMode_);

        subImu_ = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, rclcpp::SensorDataQoS(),
            std::bind(&IMUPreintegration::imuHandler, this, std::placeholders::_1));

        subOptimizedKeyframeState_ = create_subscription<aruco_sam_ailab::msg::OptimizedKeyframeState>(
            "/optimized_keyframe_state", 10,
            std::bind(&IMUPreintegration::optimizedKeyframeStateCallback, this, std::placeholders::_1));

        pubImuOdometry_ = create_publisher<nav_msgs::msg::Odometry>(
            "/odometry/imu_incremental", 2000);
        tfBroadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // Setup IMU preintegration parameters
        boost::shared_ptr<gtsam::PreintegrationParams> p =
            gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2);
        p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);
        p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);

        gtsam::imuBias::ConstantBias prior_imu_bias(
            (gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());

        // Setup noise models (improved for realistic initialization)
        priorPoseNoise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01).finished());
        priorVelNoise_ = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);  // 초기 정지 상태 가정
        priorBiasNoise_ = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);

        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);

        // Setup extrinsic transformation
        gtsam::Rot3 rot_base_imu = gtsam::Rot3(extRotBaseImu);
        gtsam::Point3 trans_base_imu(extTransBaseImu.x(), extTransBaseImu.y(), extTransBaseImu.z());
        baseToImu_ = gtsam::Pose3(rot_base_imu, trans_base_imu);

        RCLCPP_INFO(get_logger(), "IMU Preintegration node initialized, subscribing to: %s (topic_debug_log=%s)", imuTopic.c_str(), enableTopicDebugLog ? "on" : "off");
    }

    ~IMUPreintegration() {
        if (imuIntegratorImu_) {
            delete imuIntegratorImu_;
        }
    }

    void resetOptimization() {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = isamRelinearizeThreshold;
        optParameters.relinearizeSkip = isamRelinearizeSkip;
        optimizer_ = gtsam::ISAM2(optParameters);

        graphFactors_.resize(0);
        graphValues_.clear();
    }

    void initializeSystem(const gtsam::Pose3& initialPose) {
        resetOptimization();

        // Measurements are already in base_link frame, so track state in base_link
        gtsam::Pose3 imuPose = initialPose;
        gtsam::Vector3 initialVel(0, 0, 0);

        // Compute initial bias from stationary period data
        gtsam::imuBias::ConstantBias initialBias = computeInitialBias();

        // Add prior factors
        graphFactors_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), imuPose, priorPoseNoise_));
        graphFactors_.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), initialVel, priorVelNoise_));
        graphFactors_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), initialBias, priorBiasNoise_));

        // Add initial values
        graphValues_.insert(X(0), imuPose);
        graphValues_.insert(V(0), initialVel);
        graphValues_.insert(B(0), initialBias);

        // Optimize once
        optimizer_.update(graphFactors_, graphValues_);
        graphFactors_.resize(0);
        graphValues_.clear();

        // Get optimized initial state
        gtsam::Values result = optimizer_.calculateEstimate();
        prevState_ = gtsam::NavState(
            result.at<gtsam::Pose3>(X(0)),
            result.at<gtsam::Vector3>(V(0))
        );
        prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(0));

        imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);

        keyframeIdx_ = 1;
        systemInitialized_ = true;

        RCLCPP_INFO(get_logger(), "System initialized with initial pose");
    }

    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imuMsg) {
        std::lock_guard<std::mutex> lock(mtx_);

        imuCallbackCount_++;
        if (enableTopicDebugLog) {
            if (!imuFirstReceivedLogged_) {
                imuFirstReceivedLogged_ = true;
                RCLCPP_INFO(get_logger(), "[IMU] First message received on %s (stamp=%.3f)", imuTopic.c_str(), stamp2Sec(imuMsg->header.stamp));
            }
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                "[IMU] receiving: count=%lu queue=%zu initialized=%d", imuCallbackCount_, imuQueue_.size(), systemInitialized_ ? 1 : 0);
        }

        // Convert IMU to base_link frame if needed
        sensor_msgs::msg::Imu imu_base = *imuMsg;

        // Apply extrinsic rotation if IMU frame is different
        if (imuFrame != baseLinkFrame) {
            Eigen::Vector3d acc(imuMsg->linear_acceleration.x,
                                imuMsg->linear_acceleration.y,
                                imuMsg->linear_acceleration.z);
            Eigen::Vector3d gyr(imuMsg->angular_velocity.x,
                                imuMsg->angular_velocity.y,
                                imuMsg->angular_velocity.z);

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
            Eigen::Vector3d acc(imu_base.linear_acceleration.x,
                                imu_base.linear_acceleration.y,
                                imu_base.linear_acceleration.z);
            Eigen::Vector3d gyr(imu_base.angular_velocity.x,
                                imu_base.angular_velocity.y,
                                imu_base.angular_velocity.z);
            if (!lpfInitialized_) {
                lpfAcc_ = acc;
                lpfGyr_ = gyr;
                lpfInitialized_ = true;
            } else {
                lpfAcc_ = lpfAlpha * acc + (1.0 - lpfAlpha) * lpfAcc_;
                lpfGyr_ = lpfAlpha * gyr + (1.0 - lpfAlpha) * lpfGyr_;
            }
            imu_base.linear_acceleration.x = lpfAcc_.x();
            imu_base.linear_acceleration.y = lpfAcc_.y();
            imu_base.linear_acceleration.z = lpfAcc_.z();
            imu_base.angular_velocity.x = lpfGyr_.x();
            imu_base.angular_velocity.y = lpfGyr_.y();
            imu_base.angular_velocity.z = lpfGyr_.z();
        }

        imuQueue_.push_back(imu_base);

        // Keep queue size reasonable
        while (imuQueue_.size() > 2000) {
            imuQueue_.pop_front();
        }

        // Collect IMU data during stationary period for bias estimation
        if (!systemInitialized_) {
            double currentTime = stamp2Sec(imu_base.header.stamp);

            // Start collecting samples
            if (initStartTime_ < 0) {
                initStartTime_ = currentTime;
                initGyroSamples_.clear();
                initAccelSamples_.clear();
                RCLCPP_INFO(get_logger(), "[IMU] Starting stationary period for bias estimation...");
            }

            // Collect samples during stationary period (1 second)
            double elapsed = currentTime - initStartTime_;
            if (elapsed < INIT_STATIONARY_DURATION) {
                // Collect gyro and accel samples
                Eigen::Vector3d gyr(imu_base.angular_velocity.x,
                                    imu_base.angular_velocity.y,
                                    imu_base.angular_velocity.z);
                Eigen::Vector3d acc(imu_base.linear_acceleration.x,
                                    imu_base.linear_acceleration.y,
                                    imu_base.linear_acceleration.z);
                initGyroSamples_.push_back(gyr);
                initAccelSamples_.push_back(acc);

                if (enableTopicDebugLog) {
                    RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 500,
                        "[IMU] Collecting bias samples: %.2f/%.2f sec, samples=%zu",
                        elapsed, INIT_STATIONARY_DURATION, initGyroSamples_.size());
                }
                return;
            }

            // After 1 second stationary period
            if (runMode_ == "localization") {
                // Localization: 바이어스 수집 완료, SLAM 초기 위치 추정 대기
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000,
                    "[IMU] Localization mode: bias samples ready (%zu), waiting for SLAM initial pose...",
                    initGyroSamples_.size());
                return;
            }
            // Mapping: (0,0)에서 즉시 시작
            if (imuQueue_.size() >= 10) {
                RCLCPP_INFO(get_logger(), "[IMU] Mapping mode: auto-initializing at origin (%zu bias samples)",
                            initGyroSamples_.size());
                gtsam::Pose3 identityPose = gtsam::Pose3();
                initializeSystem(identityPose);
            } else {
                if (enableTopicDebugLog) {
                    RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 1000,
                        "[IMU] system not initialized yet, queue size=%zu (need >=10)", imuQueue_.size());
                }
                return;
            }
        }

        double imuTime = stamp2Sec(imu_base.header.stamp);
        if (lastImuTime_ >= 0 && imuTime <= lastImuTime_) {
            return;  // keyframe 리셋 이후 오래된 stamp 무시
        }
        double dt = (lastImuTime_ < 0) ? (1.0 / 200.0) : (imuTime - lastImuTime_);
        lastImuTime_ = imuTime;

        // Track acceleration magnitude for stationary detection
        double accNorm = Eigen::Vector3d(imu_base.linear_acceleration.x,
                                            imu_base.linear_acceleration.y,
                                            imu_base.linear_acceleration.z).norm();
        recentAccNorm_.push_back(accNorm);
        while (recentAccNorm_.size() > ACC_WINDOW) recentAccNorm_.pop_front();

        // Integrate IMU measurement
        imuIntegratorImu_->integrateMeasurement(
            gtsam::Vector3(imu_base.linear_acceleration.x,
                            imu_base.linear_acceleration.y,
                            imu_base.linear_acceleration.z),
            gtsam::Vector3(imu_base.angular_velocity.x,
                            imu_base.angular_velocity.y,
                            imu_base.angular_velocity.z),
            dt);

        // Predict state
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevState_, prevBias_);

        // Per-step state update (prevents numerical issues with long integrations)
        prevState_ = currentState;
        imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);

        // Velocity damping: exponential decay prevents runaway from accelerometer noise
        // At 200Hz: 0.995^200 ≈ 0.37 → velocity decays to 37% after 1 second without real acceleration
        {
            gtsam::Vector3 vel = prevState_.velocity();
            vel *= VELOCITY_DECAY;
            prevState_ = gtsam::NavState(prevState_.pose(), vel);
        }

        // Stationary detection: if |acc| magnitude is stable (low std-dev) AND angular velocity is small
        if (recentAccNorm_.size() >= ACC_WINDOW) {
            double mean = 0.0;
            for (double a : recentAccNorm_) mean += a;
            mean /= recentAccNorm_.size();

            double var = 0.0;
            for (double a : recentAccNorm_) var += (a - mean) * (a - mean);
            var /= recentAccNorm_.size();
            double stddev = std::sqrt(var);

            double gyrNorm = Eigen::Vector3d(imu_base.angular_velocity.x,
                                            imu_base.angular_velocity.y,
                                            imu_base.angular_velocity.z).norm();

            if (enableTopicDebugLog) {
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                    "[IMU] acc_stddev=%.4f gyro_norm=%.4f vel=(%.3f,%.3f,%.3f) stationary=%s",
                    stddev, gyrNorm,
                    prevState_.velocity().x(), prevState_.velocity().y(), prevState_.velocity().z(),
                    (stddev < ACC_STATIONARY_THRESH && gyrNorm < GYRO_STATIONARY_THRESH) ? "YES" : "NO");
            }
            if (stddev < ACC_STATIONARY_THRESH && gyrNorm < GYRO_STATIONARY_THRESH) {
                // Device is stationary: zero out velocity
                prevState_ = gtsam::NavState(prevState_.pose(), gtsam::Vector3::Zero());
            }
        }

        // Safety: clamp velocity to prevent runaway
        gtsam::Vector3 vel = prevState_.velocity();
        if (vel.norm() > VELOCITY_MAX) {
            vel = vel.normalized() * VELOCITY_MAX;
            prevState_ = gtsam::NavState(prevState_.pose(), vel);
        }

        // Publish incremental odometry (use prevState_ which has corrections applied)
        gtsam::Vector3 currentAngularVel(imu_base.angular_velocity.x,
                                        imu_base.angular_velocity.y,
                                        imu_base.angular_velocity.z);
        publishOdometry(prevState_, imu_base.header.stamp, currentAngularVel);
    }

    void publishOdometry(const gtsam::NavState& state, const rclcpp::Time& stamp, const gtsam::Vector3& angular_vel) {
        // State is already tracked in base_link frame (measurements are in base_link)
        gtsam::Pose3 basePose = state.pose();

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = odomFrame;
        odom.child_frame_id = baseLinkFrame;

        odom.pose.pose.position.x = basePose.translation().x();
        odom.pose.pose.position.y = basePose.translation().y();
        odom.pose.pose.position.z = basePose.translation().z();

        auto q = basePose.rotation().toQuaternion();
        odom.pose.pose.orientation.w = q.w();
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();

        odom.twist.twist.linear.x = state.velocity().x();
        odom.twist.twist.linear.y = state.velocity().y();
        odom.twist.twist.linear.z = state.velocity().z();

        odom.twist.twist.angular.x = angular_vel.x();
        odom.twist.twist.angular.y = angular_vel.y();
        odom.twist.twist.angular.z = angular_vel.z();

        pubImuOdometry_->publish(odom);
        // TF는 ekf_smoother가 odom → base_footprint로 발행
    }

    // Compute initial bias from stationary period samples
    gtsam::imuBias::ConstantBias computeInitialBias() {
        gtsam::Vector3 accBias(0, 0, 0);
        gtsam::Vector3 gyrBias(0, 0, 0);

        if (!initGyroSamples_.empty() && !initAccelSamples_.empty()) {
            // Compute gyroscope bias (average during stationary period)
            Eigen::Vector3d gyrSum(0, 0, 0);
            for (const auto& gyr : initGyroSamples_) {
                gyrSum += gyr;
            }
            gyrBias = gtsam::Vector3(gyrSum.x() / initGyroSamples_.size(),
                                    gyrSum.y() / initGyroSamples_.size(),
                                    gyrSum.z() / initGyroSamples_.size());

            // Compute accelerometer bias
            // During stationary period, measured acc = gravity_in_body + bias
            // Gravity direction is inferred from the mean acceleration vector
            Eigen::Vector3d accSum(0, 0, 0);
            for (const auto& acc : initAccelSamples_) {
                accSum += acc;
            }
            Eigen::Vector3d accMean = accSum / initAccelSamples_.size();

            // Expected gravity vector: same direction as accMean, magnitude = imuGravity
            // This handles tilted cameras correctly (doesn't assume Z-up)
            Eigen::Vector3d gravityInBody = accMean.normalized() * imuGravity;
            accBias = gtsam::Vector3(accMean.x() - gravityInBody.x(),
                                    accMean.y() - gravityInBody.y(),
                                    accMean.z() - gravityInBody.z());

            RCLCPP_INFO(get_logger(), "[IMU] Initial bias computed from %zu samples:", initGyroSamples_.size());
            RCLCPP_INFO(get_logger(), "  Gyro bias: [%.6f, %.6f, %.6f] rad/s",
                        gyrBias.x(), gyrBias.y(), gyrBias.z());
            RCLCPP_INFO(get_logger(), "  Accel mean: [%.3f, %.3f, %.3f] m/s^2 (norm: %.3f, expected: %.3f)",
                        accMean.x(), accMean.y(), accMean.z(), accMean.norm(), imuGravity);
            RCLCPP_INFO(get_logger(), "  Accel bias: [%.6f, %.6f, %.6f] m/s^2",
                        accBias.x(), accBias.y(), accBias.z());
        } else {
            RCLCPP_WARN(get_logger(), "[IMU] No samples collected, using zero bias");
        }

        // Combine into ConstantBias: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
        return gtsam::imuBias::ConstantBias(
            (gtsam::Vector(6) << accBias.x(), accBias.y(), accBias.z(),
                                gyrBias.x(), gyrBias.y(), gyrBias.z()).finished());
    }

    // Called by graph_optimizer to initialize system
    void setInitialPose(const gtsam::Pose3& pose) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!systemInitialized_) {
            initializeSystem(pose);
        }
    }

    // Get current IMU preintegration result for keyframe creation
    bool getPreintegratedMeasurement(double time, gtsam::PreintegratedImuMeasurements& preint) {
        std::lock_guard<std::mutex> lock(mtx_);

        if (!systemInitialized_ || imuQueue_.empty()) {
            return false;
        }

        // Integrate IMU measurements up to the specified time
        gtsam::PreintegratedImuMeasurements tempPreint = *imuIntegratorImu_;

        double lastTime = lastImuTime_;
        for (const auto& imu : imuQueue_) {
            double imuTime = stamp2Sec(imu.header.stamp);
            if (imuTime > lastTime && imuTime <= time) {
                double dt = (lastTime < 0) ? (1.0 / 200.0) : (imuTime - lastTime);
                tempPreint.integrateMeasurement(
                    gtsam::Vector3(imu.linear_acceleration.x,
                                    imu.linear_acceleration.y,
                                    imu.linear_acceleration.z),
                    gtsam::Vector3(imu.angular_velocity.x,
                                    imu.angular_velocity.y,
                                    imu.angular_velocity.z),
                    dt);
                lastTime = imuTime;
            }
        }

        preint = tempPreint;
        return true;
    }

    // Reset preintegration after keyframe
    void resetPreintegration(const gtsam::imuBias::ConstantBias& bias) {
        std::lock_guard<std::mutex> lock(mtx_);
        imuIntegratorImu_->resetIntegrationAndSetBias(bias);
        prevBias_ = bias;
    }

    // Update state after optimization
    void updateState(const gtsam::NavState& state, const gtsam::imuBias::ConstantBias& bias) {
        std::lock_guard<std::mutex> lock(mtx_);
        prevState_ = state;
        prevBias_ = bias;
    }

    // Graph optimization 결과 수신: 이 포즈/속도/바이어스로 재초기화하고,
    // 그 시점부터 현재까지 쌓인 IMU 데이터를 "재적분(Re-propagation)" 해야 튐/발산 방지
    void optimizedKeyframeStateCallback(const aruco_sam_ailab::msg::OptimizedKeyframeState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);

        // Localization: 첫 SLAM 보정으로 시스템 초기화
        if (!systemInitialized_) {
            gtsam::Pose3 initialPose = poseMsgToGtsam(msg->pose);
            RCLCPP_INFO(get_logger(), "[IMU] Localization: initializing from SLAM pose (x=%.3f, y=%.3f)",
                        initialPose.translation().x(), initialPose.translation().y());
            initializeSystem(initialPose);
        }

        double optTime = stamp2Sec(msg->header.stamp);

        // 1. 최적화된 상태(Pose, Velocity, Bias) 파싱
        gtsam::Pose3 basePose = poseMsgToGtsam(msg->pose);
        gtsam::Vector3 vel(msg->velocity.x, msg->velocity.y, msg->velocity.z);

        // Bias: if provided (size==6), use it; otherwise keep current prevBias_
        if (msg->bias.size() == 6) {
            prevBias_ = gtsam::imuBias::ConstantBias(
                (gtsam::Vector(6) << msg->bias[0], msg->bias[1], msg->bias[2],
                                    msg->bias[3], msg->bias[4], msg->bias[5]).finished());
        }

        // 2. 큐 정리: 최적화 시점(optTime)보다 오래된 데이터는 삭제
        while (!imuQueue_.empty()) {
            if (stamp2Sec(imuQueue_.front().header.stamp) < optTime) {
                imuQueue_.pop_front();
            } else {
                break;
            }
        }

        // 3. 상태 리셋: 기준 상태(prevState_)를 최적화된 값으로 덮어씀
        //    velocity=0 from graph optimizer → resets accumulated velocity drift
        prevState_ = gtsam::NavState(basePose, vel);

        // 4. 적분기 리셋: 현재 바이어스 적용
        imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);

        // 5. [핵심] 재전파 (Re-propagation): t_opt ~ t_now 큐 데이터를 새 바이어스/시작위치로 재적분
        double dt_accum = 0.0;
        double last_t = optTime;

        if (!imuQueue_.empty()) {
            for (const auto& imu : imuQueue_) {
                double t = stamp2Sec(imu.header.stamp);
                double dt = t - last_t;

                if (dt > 0) {
                    imuIntegratorImu_->integrateMeasurement(
                        gtsam::Vector3(imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z),
                        gtsam::Vector3(imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z),
                        dt);
                    last_t = t;
                    dt_accum += dt;
                }
            }
        }

        // 6. 재적분 결과로 prevState_를 "현재" 시점으로 갱신 (다음 imuHandler가 작은 dt로 적분하도록)
        if (dt_accum > 0) {
            prevState_ = imuIntegratorImu_->predict(prevState_, prevBias_);
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
        }

        // 7. 시간 동기화: lastImuTime_을 큐의 마지막 시간(현재)으로 설정 → 다음 imuHandler에서 dt 꼬임 방지
        if (dt_accum > 0) {
            lastImuTime_ = last_t;
        } else {
            lastImuTime_ = optTime;
        }

        if (enableTopicDebugLog) {
            RCLCPP_INFO(get_logger(), "[IMU] Corrected & Repropagated: time gap %.4fs updated, queue=%zu",
                dt_accum, imuQueue_.size());
        }
    }
};

} // namespace aruco_sam_ailab

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);

    auto node = std::make_shared<aruco_sam_ailab::IMUPreintegration>(options);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> IMU Preintegration Started.\033[0m");

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
