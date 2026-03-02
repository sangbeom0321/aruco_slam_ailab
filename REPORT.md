# ArUco-SAM: ArUco 마커 기반 Smoothing and Mapping 시스템 상세 보고서

---

## 목차

1. [ArUco-SAM 개념 및 이론적 배경](#1-aruco-sam-개념-및-이론적-배경)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [시뮬레이션 환경 평가 지표](#3-시뮬레이션-환경-평가-지표)
4. [실제 환경 평가 방법론](#4-실제-환경-평가-방법론)
5. [코드 리뷰 및 개선 제안](#5-코드-리뷰-및-개선-제안)

---

## 1. ArUco-SAM 개념 및 이론적 배경

### 1.1 ArUco-SAM의 명칭과 의미

**ArUco-SAM**은 다음 두 요소의 결합이다:

- **ArUco** (Augmented Reality University of Córdoba): OpenCV에서 제공하는 정사각형 바이너리 피듀셜 마커 시스템. 각 마커는 고유한 정수 ID를 가지며, 단일 카메라 이미지에서 4개 코너점의 검출과 PnP(Perspective-n-Point) 알고리즘을 통해 6-DOF 상대 포즈를 추정할 수 있다.

- **SAM** (Smoothing and Mapping): 전체 로봇 궤적과 랜드마크 위치를 **동시에 최적화**하는 SLAM 백엔드 방법론. "Smoothing"은 전체 궤적의 사후확률(posterior)을 일관되게 추정하는 것을 의미하며, 이는 필터 기반 방법(EKF-SLAM 등)이 현재 상태만 추정하는 것과 대비된다.

### 1.2 LIO-SAM과의 구조적 비교

**LIO-SAM** (LiDAR Inertial Odometry via Smoothing and Mapping, T. Shan et al., IROS 2020)은 본 시스템의 직접적 설계 참조이다.

| 구성 요소 | LIO-SAM | ArUco-SAM (본 시스템) |
|---|---|---|
| **관측 센서** | 3D LiDAR (포인트 클라우드 매칭) | RGB-D 카메라 (ArUco 마커 검출) |
| **관성 센서** | 9축 IMU | 6축 IMU (RealSense D455 내장 BMI055) |
| **관측 모델** | Scan-to-map ICP/NDT → 상대 포즈 | PnP + Depth Correction → 마커 상대 포즈 |
| **Factor Graph 요소** | IMU Factor + LiDAR Odom Factor + GPS Factor + Loop Closure Factor | IMU Factor + Wheel Odom Factor + ArUco Landmark Factor |
| **최적화 백엔드** | GTSAM ISAM2 | GTSAM ISAM2 |
| **보조 오도메트리** | IMU preintegration (고주파 데드레코닝) | IMU preintegration + Wheel Odometry |
| **출력 스무딩** | IMU 적분 기반 보간 | EKF Smoother (SE(3) 칼만 필터) |
| **랜드마크 표현** | 암묵적 (스캔 매칭) | 명시적 (마커 ID별 Pose3 변수) |

**구조적 유사점:**
- **Factor Graph + ISAM2 백엔드** 구조가 동일하다. LIO-SAM이 제안한 "tightly-coupled lidar inertial odometry via factor graph" 아키텍처를 그대로 따르되, LiDAR scan matching을 ArUco landmark observation으로 대체하였다.
- **IMU preintegration** 노드가 고주파(~200Hz) 데드레코닝을 수행하고, 그래프 최적화 결과로 상태를 보정(re-propagation)하는 패턴이 LIO-SAM과 동일하다.
- **키프레임 기반 최적화**: 모든 센서 데이터가 아닌, 키프레임 시점에서만 Factor를 추가하고 ISAM2를 업데이트한다.

**구조적 차이점:**
- LIO-SAM은 LiDAR 포인트 클라우드에서 **암묵적으로** 환경을 인식하므로 사전 준비 없이 임의 환경에서 동작하지만, ArUco-SAM은 **명시적 랜드마크(마커)**가 환경에 배치되어야 한다.
- 반대로, ArUco 마커는 고유 ID를 가지므로 **데이터 연관(data association)** 문제가 발생하지 않으며, loop closure를 별도로 검출할 필요 없이 동일 마커 재관측이 자동으로 loop closing 역할을 수행한다.
- **Wheel Odometry**를 추가 오도메트리 소스로 사용한다. 이는 LiDAR scan matching이 제공하던 inter-keyframe 상대 포즈 역할을 일부 대체한다.

### 1.3 Factor Graph SLAM의 수학적 정의

SLAM 문제를 **Maximum a Posteriori (MAP) 추정**으로 정식화한다.

**상태 변수 집합:**

$$\mathcal{X} = \{x_0, x_1, \ldots, x_K, v_0, v_1, \ldots, v_K, b_0, b_1, \ldots, b_K, l_1, l_2, \ldots, l_M\}$$

여기서:
- $x_i \in SE(3)$: 키프레임 $i$에서의 로봇 포즈 (위치 + 자세)
- $v_i \in \mathbb{R}^3$: 키프레임 $i$에서의 속도
- $b_i \in \mathbb{R}^6$: 키프레임 $i$에서의 IMU 바이어스 (가속도계 3 + 자이로스코프 3)
- $l_j \in SE(3)$: 랜드마크 $j$의 맵 프레임 포즈

**MAP 추정 문제:**

$$\mathcal{X}^* = \arg\max_{\mathcal{X}} p(\mathcal{X} \mid \mathcal{Z})$$

여기서 $\mathcal{Z}$는 모든 관측의 집합이다. Bayes 정리와 조건부 독립 가정에 의해:

$$\mathcal{X}^* = \arg\min_{\mathcal{X}} \left[ \underbrace{\|r_0\|^2_{\Sigma_0}}_{\text{Prior}} + \sum_{i=1}^{K} \underbrace{\|r_i^{\text{IMU}}\|^2_{\Sigma_i^{\text{IMU}}}}_{\text{IMU Factor}} + \sum_{i=1}^{K} \underbrace{\|r_i^{\text{odom}}\|^2_{\Sigma_i^{\text{odom}}}}_{\text{Wheel Odom Factor}} + \sum_{(i,j)} \underbrace{\|r_{ij}^{\text{ArUco}}\|^2_{\Sigma_{ij}^{\text{ArUco}}}}_{\text{Landmark Factor}} + \sum_{i=1}^{K} \underbrace{\|r_i^{\text{bias}}\|^2_{\Sigma_i^{\text{bias}}}}_{\text{Bias RW}} \right]$$

각 잔차(residual) $r$은 Mahalanobis norm $\|r\|^2_\Sigma = r^T \Sigma^{-1} r$로 가중된다.

**본 시스템의 Factor 종류별 잔차 정의:**

**(a) Prior Factor** (초기 상태 고정):
$$r_0^{\text{pose}} = \text{Log}(x_0^{-1} \cdot \bar{x}_0) \in \mathbb{R}^6$$
$$r_0^{\text{vel}} = v_0 - \bar{v}_0 \in \mathbb{R}^3$$
$$r_0^{\text{bias}} = b_0 - \bar{b}_0 \in \mathbb{R}^6$$

**(b) IMU Factor** (GTSAM `ImuFactor`):

키프레임 $i-1$에서 $i$까지의 IMU 사전적분(preintegration) 결과 $\Delta\hat{R}_{i-1,i}, \Delta\hat{v}_{i-1,i}, \Delta\hat{p}_{i-1,i}$를 사용하여:

$$r_i^{\text{IMU}} = \begin{bmatrix} \text{Log}(\Delta\hat{R}_{i-1,i}^T \cdot R_{i-1}^T \cdot R_i) \\ R_{i-1}^T(v_i - v_{i-1} - g \cdot \Delta t) - \Delta\hat{v}_{i-1,i} \\ R_{i-1}^T(p_i - p_{i-1} - v_{i-1}\Delta t - \frac{1}{2}g\Delta t^2) - \Delta\hat{p}_{i-1,i} \end{bmatrix}$$

여기서 사전적분 값은 바이어스에 대한 1차 보정을 포함한다:
$$\Delta\hat{R}_{i-1,i} \approx \Delta\tilde{R}_{i-1,i} \cdot \text{Exp}(J_R^g \cdot \delta b_g)$$

**(c) ArUco Landmark Factor** (`BetweenFactor<Pose3>`):

로봇 포즈 $x_i$에서 랜드마크 $l_j$까지의 상대 관측 $\tilde{z}_{ij}$에 대해:
$$r_{ij}^{\text{ArUco}} = \text{Log}(\tilde{z}_{ij}^{-1} \cdot x_i^{-1} \cdot l_j) \in \mathbb{R}^6$$

본 시스템에서 $\tilde{z}_{ij}$는 카메라 프레임의 ArUco PnP 결과를 base_link 프레임으로 변환한 값이다:
$$\tilde{z}_{ij} = T_{\text{base}}^{\text{cam}} \cdot T_{\text{cam}}^{\text{marker}}$$

**(d) Wheel Odometry Factor** (`BetweenFactor<Pose3>`):

연속 키프레임 간 휠 오도메트리 상대 포즈 $\Delta \tilde{o}_{i-1,i}$에 대해:
$$r_i^{\text{odom}} = \text{Log}(\Delta\tilde{o}_{i-1,i}^{-1} \cdot x_{i-1}^{-1} \cdot x_i)$$

정지 감지 시(구간 내 모든 twist 속도 < 0.01 m/s), zero-motion constraint로 대체:
$$r_i^{\text{zero}} = \text{Log}(x_{i-1}^{-1} \cdot x_i), \quad \Sigma_{\text{zero}} = \text{diag}(0.001^2 \cdot \mathbf{1}_6)$$

**(e) Bias Random Walk Factor** (`BetweenFactor<imuBias::ConstantBias>`):
$$r_i^{\text{bias}} = b_i - b_{i-1}$$

### 1.4 증분형(ISAM2) vs 배치형(Batch) 최적화

Factor Graph의 비선형 최소제곱 문제를 풀 때, 증분형과 배치형 최적화는 근본적으로 다른 전략을 사용한다.

#### 1.4.1 배치형 최적화 (Gauss-Newton / Levenberg-Marquardt)

모든 Factor와 변수에 대해 **전체 Jacobian**을 구성하고 한 번에 풀다:

$$\delta^* = \arg\min_\delta \|J\delta - r\|^2$$

정규방정식:
$$\underbrace{J^T J}_{H \text{ (정보 행렬)}} \delta^* = \underbrace{-J^T r}_{b}$$

여기서 $J \in \mathbb{R}^{m \times n}$은 전체 Jacobian ($m$: 잔차 차원, $n$: 상태 차원), $H \in \mathbb{R}^{n \times n}$은 정보 행렬이다.

**계산 복잡도:** 변수 수 $n$에 대해 $H$는 **sparse** 구조를 가지므로 sparse Cholesky 분해를 사용하면 $O(n^{1.5})$ ~ $O(n^2)$ 정도이다. 그러나 **매 키프레임마다 전체 재분해**가 필요하므로, 궤적이 길어지면 실시간 처리가 어렵다.

| 키프레임 수 | $H$ 차원 (6-DOF 포즈 + 3-DOF 속도 + 6-DOF 바이어스 + 랜드마크) | 대략적 연산량 |
|---|---|---|
| 100 | ~1,600 | ~수 ms |
| 1,000 | ~16,000 | ~수백 ms |
| 10,000 | ~160,000 | ~수 초 이상 |

#### 1.4.2 증분형 최적화 (ISAM2)

ISAM2 (Incremental Smoothing and Mapping with Fluid Relinearization, Kaess et al., IJRR 2012)는 Factor Graph를 **Bayes Tree** 자료구조로 관리하여, 새로운 Factor가 추가될 때 **영향 받는 변수만 재계산**한다.

**Bayes Tree 구조:**

1. Factor Graph를 **variable elimination** 순서에 따라 정규방정식으로 변환
2. 결과를 **Bayes Net** (방향 그래프)으로 표현
3. Bayes Net을 **Bayes Tree** (루트가 있는 트리)로 변환

새 Factor 추가 시:
1. 영향 받는 clique만 식별 (트리의 일부 경로)
2. 해당 clique만 재선형화(relinearization) 및 재분해
3. 나머지 트리는 그대로 유지

**Fluid Relinearization:** 선형화 오차가 임계값(`relinearizeThreshold = 0.1`)을 초과하는 변수만 재선형화한다. `relinearizeSkip = 1`이면 매 업데이트마다 체크한다.

**계산 복잡도:**
- 최악의 경우 (모든 변수 영향): 배치와 동일
- 일반적인 경우 (새 Factor가 소수 변수에만 영향): $O(k \log n)$, 여기서 $k$는 새로 추가된 변수/Factor 수

| 상황 | ISAM2 복잡도 | 배치 복잡도 |
|---|---|---|
| 새 키프레임 추가 (이전 키프레임과만 연결) | $O(\log n)$ | $O(n^{1.5})$ |
| 루프 클로저 추가 (먼 키프레임 연결) | $O(n)$ (최악) | $O(n^{1.5})$ |
| 랜드마크 재관측 | $O(d \log n)$, $d$=depth | $O(n^{1.5})$ |

**본 시스템의 선택: ISAM2**

본 시스템은 20Hz SLAM 타이머에서 ISAM2를 사용한다. ArUco 마커의 재관측이 빈번하게 발생하는데, 이는 loop closure와 유사한 효과를 가지므로 Bayes Tree의 상당 부분이 재계산될 수 있다. 그럼에도 명시적 랜드마크 수가 10개로 제한되어 있어 전체 그래프 규모가 작으므로 실시간 처리에 문제가 없다.

---

## 2. 시스템 아키텍처

### 2.1 전체 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ArUco-SAM 시스템 아키텍처                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [RGB-D Camera]──image──▶ aruco_detector_node ──/aruco_poses──┐    │
│       │                   (PnP + Depth Corr.)                  │    │
│       │                                                        ▼    │
│  [IMU 200Hz]──────────────────────────────────▶ graph_optimizer     │
│       │                                         (ISAM2 Backend)     │
│       │  [Wheel Odom]────/w_odom──────────────▶     │    │         │
│       │                                              │    │         │
│       │                    /optimized_keyframe_state◀─┘    │         │
│       │                         │         │                │         │
│       ▼                         ▼         │                │         │
│  imu_preintegration ◀───────────┘         │                │         │
│  (Dead Reckoning ~200Hz)                  │                │         │
│       │                                   │                │         │
│       │ /odometry/imu_incremental         │                │         │
│       ▼                                   ▼                │         │
│  ekf_smoother ◀───────────────────────────┘                │         │
│  (SE(3) EKF)                                               │         │
│       │                                                    │         │
│       │ /ekf/odom + TF(odom→base_footprint)                │         │
│       ▼                                                    │         │
│  ego_state_publisher ──/ego_state──▶ [Planning System]     │         │
│                                                            │         │
│  landmark_boundary_occupancy_grid ◀────────────────────────┘         │
│  (/global_map, /local_map) ────────▶ [Planning System]              │
│  (localization 모드에서만 활성)                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

TF Tree: map ──(static identity)──▶ odom ──(ekf_smoother)──▶ base_footprint
```

### 2.2 노드별 상세 설명

#### 2.2.1 aruco_detector_node — 비전 프론트엔드

**역할:** RGB 이미지에서 ArUco 마커를 검출하고, 깊이 이미지를 활용하여 3D 포즈를 정밀화한다.

**알고리즘:**

1. **ArUco 검출**: `cv::aruco::detectMarkers()`로 DICT_4X4_50 사전에서 마커 검출
2. **PnP 포즈 추정**: `cv::aruco::estimatePoseSingleMarkers()`로 4개 코너점과 알려진 마커 크기(0.30m)를 사용하여 P3P 알고리즘 적용

$$\text{minimize} \sum_{k=1}^{4} \|u_k - \pi(T_c^m \cdot p_k^m)\|^2$$

여기서 $\pi$는 핀홀 투영, $T_c^m \in SE(3)$는 마커→카메라 변환, $p_k^m$은 마커 좌표계의 코너 3D 좌표, $u_k$는 검출된 2D 코너점이다.

3. **깊이 보정** (시뮬레이션에서 활성): 마커 중심 주변 깊이 픽셀을 샘플링하여 중앙값(median) 깊이를 구하고, 역투영(back-projection)으로 3D 위치를 보정한다:

$$X = \frac{(u - c_x) \cdot d}{f_x}, \quad Y = \frac{(v - c_y) \cdot d}{f_y}, \quad Z = d$$

PnP의 회전은 유지하고 병진만 깊이 기반 값으로 대체한다.

**발행 토픽:** `/aruco_poses` (`MarkerArray` — 마커 ID + camera frame 상대 포즈 배열)

#### 2.2.2 graph_optimizer — ISAM2 SLAM 백엔드

**역할:** Factor Graph를 구성 및 유지하고, ISAM2 증분 최적화를 수행한다. 시스템의 핵심 노드이다.

**키프레임 정책:**
- **마커 트리거**: ArUco 마커가 하나라도 관측되면 키프레임 생성 (최소 간격 0.1초)
- **시간 폴백**: 마커 없이 `keyframeTimeInterval`(0.5초) 경과 시 데드레코닝 키프레임 생성

**키프레임당 처리 순서:**

```
1. IMU Preintegration: lastKF → currentKF 구간의 IMU 데이터 적분
2. State Prediction: NavState_prev + PreintegratedIMU → NavState_predicted
3. New Variable Insertion: X(i), V(i), B(i) with predicted values
4. Factor Addition:
   a. ImuFactor(X(i-1), V(i-1), X(i), V(i), B(i-1), preint)
   b. BetweenFactor<Pose3> (Wheel Odom delta 또는 zero-motion)
   c. BetweenFactor<ConstantBias> (Bias Random Walk)
   d. BetweenFactor<Pose3> (ArUco landmark observations)
5. ISAM2 Update: isam_.update(newFactors, newValues)
6. Extract Results: currentEstimate_, currentVelocity_, currentBias_
7. Publish: OptimizedKeyframeState → imu_preintegration, ekf_smoother
```

**노이즈 모델 설계 (GTSAM Pose3 tangent order: `[rx, ry, rz, tx, ty, tz]`):**

| Factor | Roll/Pitch | Yaw | X/Y | Z | 설계 근거 |
|---|---|---|---|---|---|
| ArUco 관측 | 0.05 rad | `arucoRotNoise` (0.5 rad) | `arucoTransNoise` (0.08 m) | 0.03 m | 2D UGV: roll/pitch/z 안정, yaw는 PnP 부정확 |
| Wheel Odom | 0.01 rad | 1.0 rad | 0.05 m | 0.01 m | Ackermann yaw 부정확, 위치는 신뢰 |
| Zero-Motion | 0.001 rad | 0.001 rad | 0.001 m | 0.001 m | 정지 시 매우 타이트한 제약 |
| Bias Random Walk | — | — | — | — | acc: `imuAccBiasN × 100`, gyro: `imuGyrBiasN × 10` |

**Bias Random Walk 노이즈 비대칭 설계의 의미:**
- 가속도계 바이어스 `× 100`: 바이어스 변화를 크게 허용 → 옵티마이저가 가속도계 드리프트를 적극 보정
- 자이로스코프 바이어스 `× 10`: 바이어스 변화를 상대적으로 제한 → 실제 회전이 바이어스 변화로 흡수되는 것을 방지 (heading 실시간성 확보)

**동작 모드:**
- **Mapping**: 원점에서 시작, 마커 최초 관측 시 `PriorFactor`로 랜드마크 등록
- **Localization**: 저장된 맵의 랜드마크를 `σ = 10^{-6}`의 극도로 타이트한 `PriorFactor`로 고정, 기존 마커 관측으로 초기 포즈 계산 후 시작

#### 2.2.3 imu_preintegration — 고주파 데드레코닝

**역할:** IMU 원시 데이터를 ~200Hz로 적분하여 연속적 오도메트리를 생성하고, 그래프 최적화 결과로 상태를 보정한다.

**IMU 스트랩다운 적분:**

단일 타임스텝 $\Delta t$에 대해:
$$R_{k+1} = R_k \cdot \text{Exp}((\omega_k - b_g) \Delta t)$$
$$v_{k+1} = v_k + R_k \cdot (a_k - b_a) \Delta t + g \Delta t$$
$$p_{k+1} = p_k + v_k \Delta t + \frac{1}{2} R_k \cdot (a_k - b_a) \Delta t^2 + \frac{1}{2} g \Delta t^2$$

**주요 특징:**
- **Per-step reset 패턴**: 매 IMU 샘플마다 `predict() → prevState_ = currentState → resetIntegration()`을 수행. 일반적인 preintegration(구간 적분 후 일괄 예측)과 달리 단일 스텝 적분으로 사용. 이는 수치 안정성을 위한 선택이나, preintegration의 공분산 누적 이점을 포기한다.

- **속도 감쇠 (Velocity Decay)**: `v *= 0.995` (매 스텝). 200Hz에서 1초 후 $0.995^{200} \approx 0.37$로 감쇠. 가속도계 노이즈에 의한 속도 발산을 방지하는 경험적(heuristic) 기법이다.

- **정지 감지**: 가속도 크기의 표준편차 < 0.15 m/s² AND 자이로 크기 < 0.05 rad/s이면 정지로 판단, 속도를 0으로 리셋.

- **Re-propagation**: `OptimizedKeyframeState` 수신 시:
  1. `prevState_`를 최적화된 포즈/속도로 덮어쓰기
  2. IMU 큐에서 최적화 시점 이전 데이터 제거
  3. 최적화 시점 ~ 현재까지의 IMU 데이터를 새 바이어스로 재적분
  4. 재적분 결과로 현재 상태 갱신

**저역통과 필터 (LPF):**
$$y[n] = \alpha \cdot x[n] + (1 - \alpha) \cdot y[n-1]$$
`lpfAlpha` 값에 따라 진동/바이브레이션 스파이크를 억제한다.

#### 2.2.4 ekf_smoother — SE(3) 확장 칼만 필터

**역할:** 고주파 IMU 오도메트리(~200Hz)와 저주파 SLAM 보정(~10Hz, 마커 가시 시)을 융합하여 부드러운 최종 오도메트리를 출력한다.

**상태 공간:**
- 상태: $x \in SE(3)$ (6-DOF 포즈)
- 공분산: $P \in \mathbb{R}^{6 \times 6}$ (SE(3) 접선 공간, GTSAM 순서: `[rot, trans]`)

**예측 단계** (IMU 오도메트리 delta $\Delta T$ 수신 시):
$$\hat{x}_{k|k-1} = x_{k-1|k-1} \circ \Delta T$$
$$P_{k|k-1} = P_{k-1|k-1} + Q \cdot \Delta t$$

여기서 $\circ$는 SE(3) compose 연산이고, $F = I$ 가정 (미소 변위에서의 1차 근사). 엄밀히는 $F = \text{Ad}_{\Delta T^{-1}}$ (adjoint 표현)이어야 하나, ~200Hz의 미소 $\Delta T$에서 $I$와 차이가 매우 작다.

**갱신 단계** (SLAM 보정 포즈 $z$ 수신 시):

$$\text{innovation} = \text{Log}(\hat{x}^{-1} \circ z) \in \mathbb{R}^6$$
$$K = P \cdot (P + R)^{-1}$$
$$x_{k|k} = \hat{x}_{k|k-1} \circ \text{Exp}(K \cdot \text{innovation})$$
$$P_{k|k} = (I - K) P_{k|k-1} (I - K)^T + K R K^T \quad \text{(Joseph form)}$$

여기서 $H = I$ 가정 (관측이 직접 전체 상태를 제공하므로 관측 Jacobian이 항등행렬).

**Innovation Gating:**
- 수렴 기간(첫 10회 업데이트): 모든 보정 무조건 수용
- 정상 상태: 병진 innovation > 5.0m 또는 회전 innovation > 2.0rad이면 이상치로 거부

**EKF 노이즈 파라미터:**

| 파라미터 | Sim/Real 값 | 의미 |
|---|---|---|
| `ekf_process_noise_pos` | 0.15 | IMU 데드레코닝 위치 불확실성 (√s당 m) |
| `ekf_process_noise_rot` | 0.02 | IMU 데드레코닝 자세 불확실성 (√s당 rad) |
| `ekf_measurement_noise_pos` | 0.05 | SLAM 위치 보정 신뢰도 (m) |
| `ekf_measurement_noise_rot` | 0.15 | SLAM 자세 보정 신뢰도 (rad) |

설계 의도: 위치는 SLAM 보정을 빠르게 반영(낮은 R_pos)하고, heading은 IMU를 신뢰하여 SLAM 보정을 완만하게 반영(높은 R_rot). 이는 IMU의 자이로스코프가 단기 heading에서 높은 정확도를 보이지만, ArUco PnP의 회전 추정이 상대적으로 부정확하기 때문이다.

#### 2.2.5 ego_state_publisher — 상태 변환

**역할:** `/ekf/odom` (nav_msgs/Odometry)를 `/ego_state` (hunter_msgs2/EgoState)로 변환하여 플래닝 시스템에 제공.

**가속도 추정 파이프라인:**
```
v(t) → 수치 미분 (dv/dt) → Clamping (±3.0 m/s²) → LPF (α=0.1) → Moving Average (window=20)
```

#### 2.2.6 landmark_boundary_occupancy_grid_node — 점유 격자 생성

**역할:** Localization 모드에서만 활성화. 랜드마크 위치를 연결하여 경계 폴리곤을 구성하고, 이를 점유 격자(OccupancyGrid)로 변환하여 플래너에 제공.

**알고리즘:**
- **Bresenham 직선 래스터화**: 연속 랜드마크 쌍을 잇는 벽을 격자에 표현
- **Ray-casting 내부 판정**: 폴리곤 내부 = FREE(0), 외부 = UNKNOWN(-1), 벽 = OCCUPIED(100)
- **로봇 중심 로컬맵**: TF로 로봇 위치를 조회하여 10m × 10m 윈도우 추출

### 2.3 초기화 시퀀스

```
t=0s   : imu_preintegration 시작 → IMU 바이어스 샘플 수집 (정지 1초)
t=1s   : imu_preintegration 바이어스 추정 완료
         - Mapping: 원점에서 즉시 초기화, 데드레코닝 시작
         - Localization: SLAM 초기 포즈 대기
t=5s   : graph_optimizer 시작 (TimerAction 지연)
         - IMU 큐 50샘플 이상 확인
         - Mapping: ArUco 마커 최초 관측 시 Factor Graph 초기화
         - Localization: 기존 맵의 마커 관측 시 역포즈 계산으로 초기화
t=5s+  : ISAM2 키프레임 최적화 시작
         - Warmup 5프레임 동안 OptimizedKeyframeState 미발행
t=~7s  : Warmup 완료 → OptimizedKeyframeState 발행 시작
         → imu_preintegration Re-propagation 시작
         → ekf_smoother 초기화 및 출력 시작
```

---

## 3. 시뮬레이션 환경 평가 지표

LIO-SAM 논문 및 관련 SLAM 벤치마크(KITTI, EuRoC, TUM)에서 사용하는 표준 지표를 기반으로, ArUco-SAM의 시뮬레이션 환경 평가에 적합한 지표를 제안한다.

### 3.1 Absolute Pose Error (APE)

**정의:** 추정 궤적과 Ground Truth 궤적 간의 절대 오차. 전역 정확도를 평가한다.

시간 $t_i$에서의 절대 포즈 오차:
$$E_i = T_{\text{GT},i}^{-1} \cdot T_{\text{est},i}$$

**통계량:**
- **APE Translation RMSE**: $\text{RMSE}_{\text{trans}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \|t_{E_i}\|^2}$ — 가장 대표적인 단일 지표
- **APE Translation Mean/Median/Std**: 분포 특성 파악
- **APE Rotation RMSE**: $\text{RMSE}_{\text{rot}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \|\text{Log}(R_{E_i})\|^2}$ (단위: rad)

**도구:** `evo` 패키지 (`evo_ape`)

```bash
# Gazebo에서 /gazebo/model_states 또는 /ground_truth/odom으로 GT 확보
evo_ape tum gt.txt est.txt -p --plot_mode=xz
```

**ArUco-SAM에서의 적용:**
- Gazebo 시뮬레이터의 model state를 Ground Truth로 사용
- `/ekf/odom` (최종 출력)과 비교
- **추가 비교**: `/aruco_slam/odom` (ISAM2 직접 출력), `/aruco_slam/wheel_odom` (보정된 wheel odom)과 개별 비교하여 각 모듈의 기여도 분석

### 3.2 Relative Pose Error (RPE)

**정의:** 일정 거리/시간 간격에서의 상대 포즈 오차. 로컬 일관성과 drift를 평가한다.

구간 $\Delta$에서의 상대 포즈 오차:
$$E_{\Delta,i} = (T_{\text{GT},i}^{-1} \cdot T_{\text{GT},i+\Delta})^{-1} \cdot (T_{\text{est},i}^{-1} \cdot T_{\text{est},i+\Delta})$$

**권장 $\Delta$ 값:**
- **단거리**: $\Delta = 1$m — 로컬 정확도 (주행 제어에 직접 영향)
- **중거리**: $\Delta = 5$m — 중기 drift
- **장거리**: $\Delta = 10$m (전체 환경 크기 8m×6m 고려) — 전역 일관성

**도구:** `evo_rpe`

### 3.3 Drift Rate

**정의:** 주행 거리 대비 누적 오차 비율. SLAM 시스템의 장기 안정성을 나타낸다.

$$\text{Drift Rate} = \frac{\|T_{\text{GT},\text{end}}^{-1} \cdot T_{\text{est},\text{end}}\|_{\text{trans}}}{\sum_{i=0}^{N-1} \|p_{\text{GT},i+1} - p_{\text{GT},i}\|} \times 100\%$$

LIO-SAM 논문 기준 실외 LiDAR SLAM은 약 0.5~2%의 drift rate를 보고한다. ArUco-SAM은 랜드마크 재관측 시 loop closing이 자동 수행되므로 **폐루프 주행 시** 이보다 현저히 낮은 drift rate가 기대된다.

### 3.4 Landmark Estimation Accuracy

ArUco-SAM에 고유한 지표. Mapping 모드에서 최적화된 랜드마크 위치와 Gazebo 내 실제 배치 좌표를 비교한다.

$$\text{Landmark Error}_j = \|p_{\text{est},j} - p_{\text{GT},j}\|$$

- **Mean Landmark Error**: 전체 랜드마크의 평균 추정 오차
- **Max Landmark Error**: 최악 랜드마크의 오차

Gazebo world 파일에서 각 마커 모델의 배치 좌표를 Ground Truth로 추출하고, `save_landmarks` 서비스로 저장된 맵과 비교한다.

### 3.5 Loop Closing 효과 분석

**실험 설계:**
1. 마커를 관측하며 한 바퀴 주행
2. 동일 마커 재관측 전후의 APE 변화 기록

**정량 지표:**
- **Pre-revisit APE**: 마커 재관측 직전의 APE
- **Post-revisit APE**: 마커 재관측 및 ISAM2 최적화 직후의 APE
- **Correction Magnitude**: 재관측으로 인한 포즈 보정량 (OptimizedKeyframeState의 innovation)

### 3.6 계산 시간 지표

**키프레임당 ISAM2 최적화 시간:**
- `isam_.update()` + `calculateEstimate()` 소요 시간
- 키프레임 수 증가에 따른 시간 추이 (ISAM2의 증분 특성 검증)

**EKF 스텝 실행 시간:**
- predict + publish 한 사이클의 소요 시간 (200Hz 주기 = 5ms 이내여야 함)

**센서 처리 지연 (End-to-End Latency):**
- 카메라 이미지 수신 → ArUco 검출 → Factor Graph 추가 → ISAM2 최적화 → EKF 출력까지의 총 지연

### 3.7 센서별 Ablation Study

IMU, Wheel Odometry, ArUco 관측의 기여도를 정량화하기 위한 ablation 실험을 권장한다:

| 구성 | IMU | Wheel Odom | ArUco | 예상 결과 |
|---|:---:|:---:|:---:|---|
| Full | O | O | O | 최고 성능 (baseline) |
| No Wheel | O | X | O | Yaw drift 증가, 위치 drift 소폭 증가 |
| No ArUco (dead reckoning) | O | O | X | 시간에 따른 누적 drift |
| ArUco only | X | X | O | 마커 미관측 구간에서 위치 손실 |

---

## 4. 실제 환경 평가 방법론

실제 환경에서는 Gazebo와 같은 정확한 Ground Truth를 얻기 어렵다. 따라서 여러 방법을 조합하여 신뢰도 있는 평가를 수행한다.

### 4.1 Ground Truth 확보 방법

#### 방법 A: 체크포인트 기반 수동 측정 (권장, 저비용)

1. 환경 내 여러 지점에 체크포인트를 설정하고 줄자/레이저 거리계로 정밀 좌표를 측정
2. 로봇을 각 체크포인트에 정밀 정렬 후 `/ekf/odom` 포즈를 기록
3. 측정 좌표 vs 추정 좌표의 오차 계산

**장점:** 특수 장비 불필요, 즉시 수행 가능
**단점:** 이산적 포인트만 평가, 동적 정확도 미평가
**정확도:** 줄자 ±5mm, 레이저 거리계 ±2mm

#### 방법 B: 폐루프 일관성 실험 (권장, 자기 평가)

**Start-to-End 오차:**
1. 시작점에 마커를 배치하고 초기 위치를 기록
2. 임의 경로로 주행 후 시작점으로 복귀
3. 최종 위치 추정과 시작점의 오차 측정

$$e_{\text{loop}} = \|p_{\text{end}} - p_{\text{start}}\|$$

이는 **필요 조건**이다 (오차가 작다고 전체 궤적이 정확한 것은 아니지만, 오차가 크면 확실히 문제가 있다).

**다중 루프 실험:** 동일 경로를 N회 반복하여 궤적의 재현성(repeatability)을 평가.

#### 방법 C: 외부 참조 시스템 (고비용, 고정밀)

- Motion Capture (OptiTrack/Vicon): mm급 정확도, 고비용
- Total Station: 실외 cm급 정확도
- RTK-GPS: 실외 cm급, 실내 불가

### 4.2 내부 일관성 지표 (Ground Truth 불필요)

Ground Truth 없이도 시스템의 건강 상태를 평가할 수 있는 내재적 지표:

#### 4.2.1 ISAM2 Factor Graph 잔차 (Residual)

최적화 후 각 Factor의 잔차를 모니터링한다:

$$\text{Total Residual} = \sum_k \|r_k\|^2_{\Sigma_k}$$

- **잔차 추이**: 키프레임 증가에 따라 안정적으로 유지되어야 한다. 급격한 증가는 일관성 상실을 의미한다.
- **Factor별 잔차**: IMU Factor, ArUco Factor, Wheel Odom Factor의 개별 잔차를 분리 관찰하여 어떤 센서가 비일관적인지 진단.

**구현:** `graph_optimizer.cpp`에 잔차 로깅을 추가하거나, GTSAM의 `NonlinearFactorGraph::error(Values)` 호출.

#### 4.2.2 IMU 바이어스 수렴

$$b_a(t) = [b_{ax}, b_{ay}, b_{az}](t), \quad b_g(t) = [b_{gx}, b_{gy}, b_{gz}](t)$$

- 바이어스가 시간에 따라 수렴하여 안정적 값을 유지해야 한다
- 급격한 바이어스 변화는 옵티마이저가 실제 모션과 센서 노이즈를 구분하지 못하는 상황을 의미
- `OptimizedKeyframeState`의 bias 필드를 시계열로 기록하여 확인

**정상 범위 (RealSense D455 BMI055 기준):**
- 가속도계 바이어스: ±0.5 m/s² 이내
- 자이로스코프 바이어스: ±0.01 rad/s 이내

#### 4.2.3 EKF 공분산 추이

$$P(t) = \text{diag}(P_{rx}, P_{ry}, P_{rz}, P_{tx}, P_{ty}, P_{tz})$$

- 마커 관측 시 $P$ 감소 (update 단계), 미관측 시 $P$ 증가 (prediction 단계)
- 공분산의 시간적 패턴이 마커 가시성과 일치하는지 확인
- 공분산이 발산하면 시스템 불안정 신호

EKF smoother에서 이미 publish하는 `odom.pose.covariance`를 `rqt_plot`으로 모니터링 가능.

#### 4.2.4 랜드마크 재투영 일관성 (Landmark Reprojection Consistency)

다른 키프레임에서 동일 마커를 관측했을 때, 최적화된 포즈로 역투영한 마커 위치가 일관적인지 확인:

$$e_{\text{reproj}, j} = \max_{i, i'} \|T_{\text{est},i} \cdot \tilde{z}_{ij} - T_{\text{est},i'} \cdot \tilde{z}_{i'j}\|$$

이는 서로 다른 시점에서 같은 마커를 봤을 때 추정된 마커 위치의 **산포도(spread)**를 측정한다.

### 4.3 비교 실험 설계

#### 4.3.1 주행 시나리오별 실험

| 시나리오 | 설명 | 평가 초점 |
|---|---|---|
| **직선 왕복** | 시작점에서 직선 왕복 후 복귀 | 기본 정확도, 루프 오차 |
| **사각 루프** | 환경 내 사각형 경로 주행 | 4방향 drift, 코너링 정확도 |
| **마커 밀집 구간** | 여러 마커가 동시 관측되는 구간 | 다중 관측의 보정 효과 |
| **마커 공백 구간** | 의도적으로 마커 없는 구간 장기 주행 | Dead reckoning drift |
| **반복 주행** | 동일 경로 5~10회 반복 | 재현성 (repeatability) |
| **속도 변화** | 저속(0.3m/s) / 고속(1.5m/s) | 속도별 정확도 변화 |

#### 4.3.2 정량 지표 요약

| 지표 | 유형 | GT 필요 | 설명 |
|---|---|:---:|---|
| 폐루프 복귀 오차 | 외적 | X | 시작점 복귀 오차 (m, deg) |
| 체크포인트 위치 오차 | 외적 | O (수동) | N개 체크포인트의 RMSE (m) |
| ISAM2 잔차 추이 | 내적 | X | 최적화 잔차의 시계열 안정성 |
| IMU 바이어스 수렴 | 내적 | X | 바이어스의 시간적 수렴 여부 |
| EKF 공분산 추이 | 내적 | X | 공분산의 마커 가시성 연동 패턴 |
| 마커 재투영 산포 | 내적 | X | 동일 마커의 다중 관측 일관성 |
| End-to-End 지연 | 성능 | X | 센서 입력 → 포즈 출력 시간 |
| ISAM2 최적화 시간 | 성능 | X | 키프레임당 소요 시간 |

---

## 5. 코드 리뷰 및 개선 제안

### 5.1 발견된 이슈

#### [Issue 1] IMU preintegration의 속도가 World Frame으로 발행됨

**위치:** `imu_preintegration.cpp:401`

```cpp
odom.twist.twist.linear.x = state.velocity().x();
odom.twist.twist.linear.y = state.velocity().y();
odom.twist.twist.linear.z = state.velocity().z();
```

`gtsam::NavState::velocity()`는 **world frame** (map/odom frame) 속도를 반환한다. 그러나 ROS의 `nav_msgs/Odometry` 관례에서 `twist.twist.linear`는 **child_frame (body frame)** 속도여야 한다.

**영향:** `ekf_smoother`가 이 값을 `current_v_`로 받아 `/ekf/odom`의 twist에 넣고, `ego_state_publisher`가 이를 `/ego_state.v`로 변환한다. 직선 주행 시에는 body x축 속도와 world frame 속도가 유사하지만, **회전 중에는 오차가 발생**한다.

**권장 수정:**
```cpp
// World velocity를 body frame으로 변환
gtsam::Vector3 bodyVel = state.pose().rotation().unrotate(state.velocity());
odom.twist.twist.linear.x = bodyVel.x();
```

#### [Issue 2] Raw pointer 사용

**위치:** `graph_optimizer.cpp:73`, `imu_preintegration.cpp:36`

```cpp
gtsam::PreintegratedImuMeasurements* imuPreintegrator_ = nullptr;
// ...
imuPreintegrator_ = new gtsam::PreintegratedImuMeasurements(imuParams_, currentBias_);
// ...
if (imuPreintegrator_) delete imuPreintegrator_;
```

예외 발생 시 메모리 누수 가능. `std::unique_ptr`로 대체 권장.

#### [Issue 3] JSON 파서의 취약성

**위치:** `graph_optimizer.cpp:1068-1081`

수동 문자열 파싱(`extractJsonDouble`, `extractJsonInt`)을 사용한다. 공백 변화, 숫자 포맷 변화(과학적 표기법 등)에 취약하다.

**권장:** `nlohmann/json` 라이브러리 사용. 이미 C++17을 요구하므로 호환성 문제 없음.

#### [Issue 4] utility.hpp의 ODR 위반 가능성

**위치:** `include/aruco_sam_ailab/utility.hpp`

`poseMsgToGtsam()`과 `gtsamToPoseMsg()` 함수가 헤더에 비-inline으로 정의되어 있다. 현재는 각 `.cpp`가 별도 실행 파일로 컴파일되므로 문제 없지만, 라이브러리로 통합 시 링커 오류 발생.

**권장:** `inline` 키워드 추가.

#### [Issue 5] 미사용 파라미터

**위치:** `utility.hpp`에서 선언된 `keyframeDistanceThreshold`, `keyframeAngleThreshold`

이 파라미터들이 선언되고 YAML에서 로드되지만, 실제 키프레임 정책(`needNewKeyframe()`)에서 사용되지 않는다. 시간 기반 + 마커 트리거 정책만 사용 중.

**권장:** 거리/각도 기반 키프레임 정책이 필요 없다면 파라미터를 제거하여 혼란 방지. 필요하다면 `needNewKeyframe()`에 통합.

#### [Issue 6] ArUco 검출기의 RGB-Depth 비동기

**위치:** `aruco_detector_node.cpp`

RGB와 Depth 이미지가 `message_filters::TimeSynchronizer` 없이 개별 콜백으로 수신된다. 로봇 이동 중 RGB 프레임에 대응하지 않는 Depth 프레임이 사용될 수 있다.

**영향:** 시뮬레이션에서는 프레임 동기가 보장되므로 문제 없으나, 실제 RealSense에서는 프레임 드롭이나 지연으로 불일치 가능.

**권장:** `message_filters::ApproximateTimeSynchronizer` 사용.

### 5.2 아키텍처 개선 제안

#### [제안 1] 마커 가시성 기반 적응형 노이즈

현재 ArUco 관측 노이즈가 모든 마커에 동일하게 적용된다. 거리에 따른 PnP 정확도 차이를 반영하면 성능 향상이 기대된다:

$$\sigma_{\text{trans}}(d) = \sigma_0 \cdot (1 + \alpha \cdot d^2)$$

여기서 $d$는 마커까지의 거리, $\alpha$는 스케일 계수. 먼 마커일수록 PnP 정확도가 떨어지므로 노이즈를 키워 가중치를 낮춘다.

#### [제안 2] 다중 마커 동시 관측 활용

현재 각 마커를 개별 `BetweenFactor`로 처리한다. 동시에 여러 마커가 관측될 때 마커 간 상대 포즈를 추가 제약으로 활용하면 관측 일관성이 향상된다:

$$r_{jk}^{\text{inter}} = \text{Log}(\tilde{z}_{ij}^{-1} \cdot \tilde{z}_{ik}) - \text{Log}(l_j^{-1} \cdot l_k)$$

#### [제안 3] EKF에서 속도 상태 추가

현재 EKF의 상태가 포즈(6-DOF)뿐이다. 속도를 상태에 포함하면(15-DOF: 포즈 + 속도 + 바이어스) 더 정확한 예측이 가능하지만, 현재 구현의 단순함과 200Hz 업데이트 주기를 고려하면 필수적이지는 않다.
