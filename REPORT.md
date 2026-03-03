# ArUco-SAM: ArUco 마커 기반 Smoothing and Mapping 시스템 상세 보고서

---

## 목차

1. [ArUco-SAM 개념 및 이론적 배경](#1-aruco-sam-개념-및-이론적-배경)
   - 1.1 명칭과 의미 / 1.2 LIO-SAM 비교 / 1.3 Factor Graph 수학 / 1.4 ISAM2 vs Batch
2. [시스템 아키텍처](#2-시스템-아키텍처)
   - 2.1 데이터 흐름 / 2.2 노드별 상세 / **2.3 이중 추정기: Graph Opt + EKF를 동시에 쓰는 이유** / 2.4 초기화 시퀀스
3. [시뮬레이션 환경 평가 지표](#3-시뮬레이션-환경-평가-지표)
   - APE / RPE / Drift Rate / Landmark Accuracy / Ablation Study
4. [실제 환경 평가 방법론](#4-실제-환경-평가-방법론)
   - **4.1 바닥 마킹 기반 Ground Truth** / 4.2 내부 일관성 지표 / 4.3 비교 실험 설계
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
| ArUco 관측 (정적) | 0.05 rad | `arucoRotNoise` (0.5 rad) | `arucoTransNoise` (0.08 m) | 0.03 m | 2D UGV: roll/pitch/z 안정, yaw는 PnP 부정확 |
| ArUco 관측 (동적) | — | — | $\sigma_0 \cdot (1 + 0.3d) \cdot (1 + 1.5\cos^2\theta)$ | — | 거리·시야각 적응형 (아래 참조) |
| Wheel Odom | 0.01 rad | 1.0 rad | 0.05 m | 0.01 m | Ackermann yaw 부정확, 위치는 신뢰 |
| Zero-Motion | 0.001 rad | 0.001 rad | 0.001 m | 0.001 m | 정지 시 매우 타이트한 제약 |
| Landmark Prior | 0.5 rad | 1.0 rad | 10.0 m | 0.5 m | 발산 방지 regularizer (아래 참조) |
| Bias Random Walk | — | — | — | — | acc: `imuAccBiasN × 100`, gyro: `imuGyrBiasN × 10` |

**동적 ArUco 노이즈 모델:**

ArUco PnP 추정 정확도는 마커까지의 거리 $d$와 시야각(viewing angle) $\theta$에 강하게 의존한다. 이를 반영한 적응형 노이즈 모델:

$$\sigma_{\text{trans}}(d, \theta) = \sigma_0 \cdot (1 + 0.3 \cdot d) \cdot (1 + 1.5 \cdot \cos^2\theta)$$

- **거리 계수** $(1 + 0.3d)$: PnP의 병진 오차는 거리에 비례하여 증가. 1m에서 ×1.3, 5m에서 ×2.5.
- **시야각 계수** $(1 + 1.5\cos^2\theta)$: 정면 관측($\theta \approx 0$)은 깊이 방향 불확실성이 크므로 ×2.5, 45° 관측은 ×1.75.
- **시야각 필터**: $\theta < 0.17$ rad (약 10°) 미만의 극단적 정면 관측은 PnP 깊이 추정이 불안정하여 완전히 필터링.
- **거리 필터**: $d > 5.0$m 초과 관측도 필터링.

**Landmark PriorFactor 설계:**

Mapping 모드에서 마커 최초 관측 시 초기 추정 위치에 PriorFactor를 부여한다. 이 Prior의 σ는 의도적으로 매우 느슨하게(10m) 설정되어 있다:

$$\Sigma_{\text{prior}} = \text{diag}(0.5, 0.5, 1.0, 10.0, 10.0, 0.5)^2$$

이는 **발산 방지 regularizer** 역할만 수행한다. ArUco BetweenFactor의 σ가 ~0.1m이므로, 3~4회 관측 이후에는 BetweenFactor가 PriorFactor를 완전히 지배하여 랜드마크 위치가 관측 데이터에 의해 결정된다. PriorFactor 없이는 ISAM2가 수치적으로 발산할 수 있다.

**Bias Random Walk 노이즈 비대칭 설계의 의미:**
- 가속도계 바이어스 `× 100`: 바이어스 변화를 크게 허용 → 옵티마이저가 가속도계 드리프트를 적극 보정
- 자이로스코프 바이어스 `× 10`: 바이어스 변화를 상대적으로 제한 → 실제 회전이 바이어스 변화로 흡수되는 것을 방지 (heading 실시간성 확보)

**동작 모드:**
- **Mapping**: 원점에서 시작, 마커 최초 관측 시 느슨한 PriorFactor(σ=10m)로 랜드마크 등록, 이후 BetweenFactor 관측으로 위치 수렴
- **Localization**: 저장된 맵의 랜드마크를 $σ = 10^{-6}$의 극도로 타이트한 PriorFactor로 고정, 기존 마커 관측으로 초기 포즈 계산 후 시작

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

### 2.3 이중 추정기 아키텍처: Graph Optimization과 EKF를 동시에 사용하는 이유

본 시스템은 **ISAM2 Factor Graph Optimizer**와 **SE(3) EKF Smoother**라는 두 개의 추정기를 동시에 운용한다. 이는 설계 중복이 아니라, 각각이 해결하는 문제가 근본적으로 다르기 때문이다.

#### 2.3.1 문제 정의: Rate-Accuracy Tradeoff

로봇 제어 시스템은 두 가지 상충하는 요구사항을 동시에 만족해야 한다:

| 요구사항 | 필요 주파수 | 필요 정확도 | 담당 추정기 |
|---|---|---|---|
| **경로 추종 제어** | 50~200 Hz | 연속적, 부드러운 포즈 | EKF Smoother |
| **전역 위치 정확도** | 2~10 Hz | 절대 좌표계 기준 일관성 | Graph Optimizer |

단일 추정기로는 **두 요구사항을 동시에 만족할 수 없다**. 아래에서 각 추정기만 사용할 경우의 문제를 분석한다.

#### 2.3.2 Graph Optimizer만 사용할 경우의 한계

**(a) 이산적 출력 (10 Hz)**

ISAM2 최적화는 키프레임 단위로만 실행된다. 본 시스템에서 키프레임 주기는:
- 마커 관측 시: 최소 100ms 간격 → 최대 **10 Hz**
- 마커 미관측 시: 500ms 간격 → **2 Hz**

제어 루프가 50 Hz로 동작한다면, 10 Hz SLAM 출력 사이의 40ms 동안 포즈 정보가 없다. 이 기간 동안 로봇은 "눈을 감고" 주행하는 것과 같다.

**(b) 마커 공백 구간에서 포즈 정지**

마커가 2초간 보이지 않으면, ISAM2는 IMU/wheel odom 데드레코닝 키프레임만 생성한다. 이 키프레임들의 포즈는 누적 drift를 포함하며, 제어기에 직접 전달하면 **불연속적 점프**가 발생한다 (특히 마커 재관측 후 ISAM2가 과거 궤적을 일괄 보정할 때).

**(c) 이상치 내성 부재**

Graph Optimizer의 출력을 직접 사용하면, ISAM2 warmup 기간(초반 5 키프레임)이나 일시적 최적화 실패 시 비정상적 포즈가 그대로 제어기에 전달된다.

#### 2.3.3 EKF만 사용할 경우의 한계

**(a) 누적 Drift**

EKF prediction은 IMU 적분에 의존하며, 프로세스 노이즈 $Q$에 의해 공분산이 선형으로 증가한다:

$$P(t) = P(0) + Q \cdot t$$

본 시스템의 위치 프로세스 노이즈 $q_{\text{pos}} = 0.15$ m/√s에서, SLAM 보정 없이 10초 경과 시:
$$\sigma_{\text{pos}}(10\text{s}) = \sqrt{0.15^2 \times 10} = 0.47\text{ m} \quad (1\sigma)$$

이는 8m×6m 환경에서 **환경 크기의 6%**에 해당하며, 장기 주행에서는 발산한다.

**(b) 절대 위치 부재**

EKF + IMU + Wheel Odom만으로는 **상대적** 위치 추적만 가능하다. 지도 프레임(map frame)에서의 절대 좌표를 알 수 없으므로:
- 사전 정의된 목표 지점으로의 자율 주행 불가
- 다중 세션 간 위치 연속성 불가
- 맵 기반 경로 계획 불가

**(c) Loop Closure 불가**

동일 지점을 재방문해도 누적 drift를 보정할 메커니즘이 없다. EKF는 **현재 상태만** 유지하므로 과거 궤적을 소급 보정할 수 없다.

#### 2.3.4 이중 추정기의 역할 분담

```
시간 →  0ms     5ms    10ms   ...  100ms   105ms  ...  200ms
        │       │       │           │       │           │
ISAM2   ■━━━━━━━━━━━━━━━━━━━━━━━━━━■━━━━━━━━━━━━━━━━━━━■
        KF#n                       KF#n+1                KF#n+2
        │                          │                     │
        ▼ OptimizedKeyframeState   ▼                     ▼
        │                          │                     │
EKF     ■─■─■─■─■─■─■─■─■─■─■─■─■─■─■─■─■─■─■─■─■─■─■─■
        P  P  P  P  P  P  P  P  P U  P  P  P  P  P  P  U
                                   ↑                     ↑
                                   SLAM Update           SLAM Update

■ = EKF prediction (IMU delta), P = predict, U = update (SLAM correction)
```

| 컴포넌트 | 역할 | 주파수 | 출력 특성 |
|---|---|---|---|
| **Graph Optimizer** | 전역 최적화, loop closure, 랜드마크 매핑, 바이어스 추정 | 2~10 Hz | 정확하지만 이산적, 점프 가능 |
| **IMU Preintegration** | 고주파 데드레코닝, Re-propagation | ~200 Hz | 연속적이지만 drift 있음 |
| **EKF Smoother** | 두 소스를 Kalman Gain으로 최적 융합, TF 발행 | ~200 Hz | 연속적 + 전역 정확 |

**핵심 통찰:** Graph Optimizer는 **"어디에 있는가"**를 정확히 답하고, EKF는 **"지금 이 순간 어디에 있는가"**를 부드럽게 답한다.

#### 2.3.5 Kalman Gain의 수학적 분석: 센서별 신뢰도

EKF의 Kalman Gain $K$는 프로세스 노이즈 $Q$와 관측 노이즈 $R$의 비율로 결정되며, 이는 **IMU 예측과 SLAM 보정 중 어느 쪽을 더 신뢰할지**를 수학적으로 결정한다.

**본 시스템의 Q, R 행렬** (GTSAM tangent space 순서: `[rot, trans]`):

$$Q = \begin{bmatrix} 0.02^2 \cdot I_3 & 0 \\ 0 & 0.15^2 \cdot I_3 \end{bmatrix} = \begin{bmatrix} 4 \times 10^{-4} \cdot I_3 & 0 \\ 0 & 2.25 \times 10^{-2} \cdot I_3 \end{bmatrix} \text{ (per second)}$$

$$R = \begin{bmatrix} 0.15^2 \cdot I_3 & 0 \\ 0 & 0.05^2 \cdot I_3 \end{bmatrix} = \begin{bmatrix} 2.25 \times 10^{-2} \cdot I_3 & 0 \\ 0 & 2.5 \times 10^{-3} \cdot I_3 \end{bmatrix}$$

**정상 상태 Kalman Gain** (SLAM 보정이 $\Delta t_c = 0.1$초 간격으로 도착할 때):

예측 단계에서 $\Delta t_c$ 동안 축적되는 공분산:
$$P_{\text{pred}} = P_{\text{post}} + Q \cdot \Delta t_c$$

갱신 단계의 Kalman Gain:
$$K = P_{\text{pred}} (P_{\text{pred}} + R)^{-1}$$

정상 상태에서 $P_{\text{post}} \approx P_{\text{pred}} - K P_{\text{pred}}$이므로, 이산 대수 Riccati 방정식의 근사해:

$$P_\infty \approx \sqrt{Q \cdot \Delta t_c \cdot R}$$

| 상태 변수 | $Q \cdot \Delta t_c$ | $R$ | $K_\infty$ | 해석 |
|---|---|---|---|---|
| **회전** (roll, pitch, yaw) | $4 \times 10^{-5}$ | $2.25 \times 10^{-2}$ | **~0.02** | SLAM 보정을 **2%만** 반영 → **IMU heading을 98% 신뢰** |
| **병진** (x, y, z) | $2.25 \times 10^{-3}$ | $2.5 \times 10^{-3}$ | **~0.90** | SLAM 위치를 **90%** 반영 → **SLAM 위치를 강하게 신뢰** |

**물리적 의미:**

- **회전 $K \approx 0.02$**: IMU 자이로스코프의 단기 heading 정확도가 높으므로 (`imuGyrNoise = 0.0005`), EKF는 IMU의 heading을 거의 그대로 사용한다. ArUco PnP의 yaw 추정은 상대적으로 노이즈가 크므로 (`arucoRotNoise = 0.5`), SLAM 보정의 yaw 성분은 2%만 반영된다. 이것이 **회전 방향에서 튀지 않는 부드러운 heading**을 만든다.

- **병진 $K \approx 0.90$**: IMU 가속도계만으로는 위치가 빠르게 발산하므로 (`imuAccNoise = 0.005`, 이중 적분 특성상 $t^2$ 발산), SLAM의 위치 보정을 즉시 90% 반영한다. 이것이 **마커 관측 시 위치가 빠르게 수렴**하는 이유이다.

#### 2.3.6 마커 미관측 구간의 동작 분석

마커가 $T$초 동안 보이지 않을 때, EKF는 prediction만 수행하고 update는 없다:

$$P(T) = P(0) + Q \cdot T$$

| 경과 시간 $T$ | $\sigma_{\text{rot}}(T)$ (rad) | $\sigma_{\text{pos}}(T)$ (m) | 상태 |
|---|---|---|---|
| 0.1초 | 0.006 | 0.047 | 정상 (마커 재관측 대기) |
| 1.0초 | 0.020 | 0.150 | 제어 가능 (drift 미미) |
| 5.0초 | 0.045 | 0.335 | 위치 열화 시작 |
| 10.0초 | 0.063 | 0.474 | 위치 불확실, heading은 아직 유효 |

**핵심:** 회전 불확실성은 10초 후에도 0.063 rad(3.6°)로 제어 가능 범위이나, 위치는 0.47m까지 열화된다. 이때 마커가 재관측되면:
1. Graph Optimizer가 ISAM2 최적화 수행 → 절대 위치 보정
2. `OptimizedKeyframeState` → EKF update 단계 실행
3. $K_{\text{pos}} \approx 0.90$이므로 위치가 즉시 보정됨
4. $K_{\text{rot}} \approx 0.02$이므로 heading은 미세 조정만

#### 2.3.7 Re-propagation: ISAM2 보정과 IMU 연속성의 연결

EKF 외에도, `imu_preintegration` 노드가 ISAM2 보정을 수신하면 **Re-propagation**을 수행한다. 이는 이중 추정기 아키텍처의 핵심 메커니즘이다.

```
ISAM2 보정 시점: t_opt (과거)
현재 시점:      t_now

IMU 큐: [imu(t_opt), imu(t_opt+5ms), ..., imu(t_now)]

Re-propagation 수순:
1. prevState_ ← ISAM2가 추정한 (pose, velocity) at t_opt
2. prevBias_  ← ISAM2가 추정한 bias at t_opt
3. 적분기 리셋: resetIntegrationAndSetBias(prevBias_)
4. for each imu in queue[t_opt ... t_now]:
       integrateMeasurement(acc, gyr, dt)
5. prevState_ ← predict(prevState_, prevBias_)  // t_now 시점으로 전진
```

**왜 필요한가:** ISAM2 보정 없이 단순히 `prevState_`만 덮어쓰면, 현재 시점의 IMU 적분 상태와 ISAM2 보정 시점 사이에 시간 간극이 발생한다. Re-propagation은 **과거의 정확한 상태**에서 **현재까지** 새로운 바이어스로 재적분하여, 연속적이면서도 보정된 상태를 생성한다.

#### 2.3.8 요약: 왜 둘 다 필요한가

```
┌──────────────────────────────────────────────────────────┐
│                  하나만 쓰면 안 되는 이유                    │
├────────────────────┬─────────────────────────────────────┤
│ Graph Optimizer만  │ ✗ 10Hz 이산 출력 → 제어 불가          │
│                    │ ✗ 마커 없으면 포즈 정지                │
│                    │ ✗ 출력 점프 → 플래너 불안정             │
├────────────────────┼─────────────────────────────────────┤
│ EKF만              │ ✗ 누적 drift 발산                    │
│                    │ ✗ 절대 위치 없음 → 맵 기반 주행 불가    │
│                    │ ✗ 과거 보정 불가 (loop closure 없음)   │
├────────────────────┼─────────────────────────────────────┤
│ Graph + EKF        │ ✓ 200Hz 연속 출력 (EKF)              │
│ (본 시스템)         │ ✓ 전역 정확도 (Graph → EKF update)    │
│                    │ ✓ 마커 공백에도 heading 유지 (IMU)     │
│                    │ ✓ 마커 재관측 시 즉시 보정 (K≈0.9)     │
│                    │ ✓ Innovation gating으로 이상치 내성    │
└────────────────────┴─────────────────────────────────────┘
```

이 아키텍처는 LIO-SAM이 "IMU preintegration + Factor Graph"로 달성한 것과 동일한 패턴이며, ArUco-SAM은 이에 명시적 EKF 레이어를 추가하여 **센서 퓨전의 투명성과 튜닝 가능성**을 높였다.

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

### 2.4 초기화 시퀀스

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

LIO-SAM 논문 및 관련 SLAM 벤치마크(KITTI, EuRoC, TUM)에서 사용하는 표준 지표를 기반으로, ArUco-SAM의 시뮬레이션 환경 평가에 적합한 지표를 정의한다.

**평가 도구:** `scripts/evaluate_slam.py` — rosbag 후처리 기반 자동 평가 스크립트. ROS 의존성 없이 `rosbags` + `evo` 라이브러리만으로 동작하며, TUM 파일 내보내기, Umeyama SE(3) 정렬, APE/RPE/Drift/Landmark 지표 산출, 시각화를 자동 수행한다.

```bash
# 기본 사용 (GT/추정 토픽 자동 감지)
python3 evaluate_slam.py ./rosbag_run/
# 랜드마크 평가 포함
python3 evaluate_slam.py ./rosbag_run/ --landmarks-map ./map/landmarks_map.json
```

### 3.1 Absolute Pose Error (APE)

**정의:** 추정 궤적과 Ground Truth 궤적 간의 절대 오차. 전역 정확도를 평가한다.

시간 $t_i$에서의 절대 포즈 오차:
$$E_i = T_{\text{GT},i}^{-1} \cdot T_{\text{est},i}$$

**통계량:**
- **APE Translation RMSE**: $\text{RMSE}_{\text{trans}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \|t_{E_i}\|^2}$ — 가장 대표적인 단일 지표
- **APE Translation Mean/Median/Std**: 분포 특성 파악
- **APE Rotation RMSE**: $\text{RMSE}_{\text{rot}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \|\text{Log}(R_{E_i})\|^2}$ (단위: rad)

**도구:** `evaluate_slam.py` (내부적으로 `evo` 패키지 사용), 또는 TUM 파일 내보내기 후 `evo_ape` CLI 직접 호출

```bash
# evaluate_slam.py가 자동으로 TUM 파일 생성 후 APE/RPE 산출
python3 evaluate_slam.py ./rosbag/

# 또는 생성된 TUM 파일로 evo CLI 직접 사용 (교차 검증)
evo_ape tum results/gt.tum results/ekf_odom.tum -p --plot_mode=xz
```

**ArUco-SAM에서의 적용:**
- Gazebo 시뮬레이터의 `/odom_gt` (wheel_odom_node에서 model state 변환)을 Ground Truth로 사용
- `/ekf/odom` (EKF 최종 출력)과 `/aruco_slam/odom` (ISAM2 직접 출력)을 자동 비교하여 각 모듈의 기여도 분석
- **rosbag 녹화**: `scripts/record_slam_bag.sh`로 이미지 제외 최소 토픽만 녹화 (용량 절약)

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

ArUco-SAM에 고유한 지표. Mapping 모드에서 최적화된 랜드마크 위치·방향과 Gazebo 내 실제 배치를 비교한다.

**위치 오차:**
$$\text{Landmark Error}_j = \|p_{\text{est},j} - p_{\text{GT},j}\|$$

**방향(Yaw) 오차:** SLAM이 추정한 마커 법선 방향과 GT 법선 방향의 차이.

- **Mean Landmark Error**: 전체 랜드마크의 평균 위치 추정 오차
- **Max Landmark Error**: 최악 랜드마크의 오차
- **Mean Yaw Error**: 전체 랜드마크의 평균 방향 추정 오차

**GT 위치 보정:** Gazebo의 마커 모델은 0.5m×0.5m×0.6m 박스이며, ArUco 마커는 박스의 -X 면에 부착되어 있다. 따라서 GT 위치는 **박스 중심이 아닌 마커 면 위치**(중심에서 0.251m offset)를 사용한다. 마커 면의 법선 방향은 환경 중심점 $(1.495, -2.425)$를 향한다.

SLAM 추정값(`landmarks_map.json`)은 Umeyama 정렬의 변환 행렬 $T_{\text{align}}$으로 GT 좌표계로 변환한 후 비교한다.

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

### 3.8 시뮬레이션 평가 결과

Gazebo 시뮬레이션 환경(8m×6m, ArUco 마커 10개, ID 11~20)에서 Mapping 모드로 약 47초, 20.6m 주행 후 평가한 결과이다. Ground Truth는 Gazebo의 `/odom_gt`(모델 상태)를 사용하였으며, Umeyama SE(3) 정렬로 좌표계를 일치시켰다.

**평가 도구:** `evaluate_slam.py` (evo 기반, rosbag 후처리)

#### 3.8.1 궤적 정확도 (EKF 최종 출력 vs Graph Optimizer 직접 출력)

| 지표 | EKF (`/ekf/odom`) | ISAM2 (`/aruco_slam/odom`) |
|---|---|---|
| **APE Trans RMSE** | **0.122 m** | 0.128 m |
| APE Trans Mean | 0.106 m | 0.110 m |
| APE Trans Max | 0.342 m | 0.451 m |
| **APE Rot RMSE** | **2.22°** | 2.86° |
| APE Rot Max | 3.89° | 15.51° |
| RPE Δ=1m Trans | 0.097 m | 0.095 m |
| RPE Δ=5m Trans | 0.274 m | 0.318 m |
| RPE Δ=10m Trans | 0.343 m | 0.353 m |
| **Drift Rate** | **0.099%** | 0.098% |
| Endpoint Error | 0.020 m | 0.020 m |
| Endpoint Yaw Error | 0.67° | 1.14° |

**분석:**

- **EKF가 ISAM2 직접 출력보다 우수하다.** APE RMSE 0.122m vs 0.128m, 특히 회전 최대 오차에서 3.89° vs 15.51°로 큰 차이를 보인다. 이는 EKF의 innovation gating이 ISAM2의 일시적 최적화 불안정을 효과적으로 필터링함을 입증한다.
- **Drift Rate 0.099%** — 시작점 복귀 오차가 0.020m에 불과하다. ArUco 마커 재관측에 의한 자동 loop closing이 drift를 거의 완벽히 제거하였다. 참고로 LIO-SAM 논문의 실외 LiDAR SLAM은 0.5~2% drift rate를 보고한다.
- **RPE Δ=1m에서 ~10cm**: 1m 이동 구간의 로컬 정확도가 10cm 수준으로, 경로 추종 제어에 충분한 정밀도이다.

#### 3.8.2 랜드마크 추정 정확도

Mapping 모드에서 ISAM2가 추정한 10개 랜드마크 위치를 Gazebo 배치 좌표(마커 면 기준)와 비교한 결과이다.

| 마커 ID | 위치 오차 (m) | Yaw 오차 (°) | 비고 |
|:---:|:---:|:---:|---|
| 11 | 0.295 | 2.2 | 환경 가장자리 |
| 12 | 0.240 | 1.8 | |
| 13 | 0.138 | 0.9 | |
| 14 | **0.027** | 1.0 | 최소 오차 |
| 15 | 0.056 | 1.1 | |
| 16 | 0.115 | 1.1 | |
| 17 | 0.192 | 1.3 | |
| 18 | 0.289 | 1.6 | |
| 19 | **0.395** | 2.4 | 최대 오차 |
| 20 | 0.274 | 1.0 | |
| **평균** | **0.202** | **1.43°** | |

**분석:**

- **평균 위치 오차 0.202m, 평균 yaw 오차 1.43°** — yaw 추정은 매우 정확하다.
- **지리적 패턴이 존재한다**: 마커 14~16(로봇 주행 경로 근처)의 오차가 3~12cm으로 가장 낮고, 마커 11, 12, 19(환경 가장자리)의 오차가 24~40cm으로 높다. 이는 가장자리 마커가 먼 거리(>5m)에서 관측되어 PnP 정확도가 떨어지기 때문이다.
- **개선 방향**: 2회 이상 환경 순회(multi-lap mapping), ArUco 관측 노이즈 모델 정밀화, 깊이 보정 활성화 등으로 5cm 이하 정확도를 목표로 개선 중이다.

#### 3.8.3 EKF 이중 추정기 효과 검증

| 비교 항목 | EKF | ISAM2 직접 | EKF 개선 |
|---|---|---|---|
| APE Trans RMSE | 0.122 m | 0.128 m | -5% |
| APE Rot Max | 3.89° | 15.51° | **-75%** |
| 출력 주파수 | ~200 Hz | ~10 Hz | ×20 |

EKF smoother의 핵심 기여는 **회전 이상치 억제**이다. ISAM2의 일시적 yaw 점프(최대 15.51°)가 EKF의 낮은 회전 Kalman Gain($K_\text{rot} \approx 0.02$)에 의해 3.89°로 억제되었다. 이는 §2.3에서 분석한 이중 추정기 아키텍처의 설계 의도를 정량적으로 검증한다.

---

## 4. 실제 환경 평가 방법론

실제 환경에서는 Gazebo와 같은 연속적 Ground Truth를 얻기 어렵다. 본 장에서는 **바닥 마킹 기반 체크포인트 평가**를 중심으로 실용적이면서도 정량적인 평가 프로토콜을 제시한다.

### 4.1 바닥 마킹 기반 Ground Truth 시스템

#### 4.1.1 좌표계 설정

바닥 마킹의 좌표계는 **SLAM의 map frame과 일치**시켜야 한다. Mapping 모드에서 SLAM이 원점(0,0)에서 시작하므로:

1. **원점 (0,0)**: 로봇의 SLAM 시작 위치 (Mapping 모드에서의 `base_footprint` 초기 위치)
2. **X축 양의 방향**: 로봇의 초기 전진 방향
3. **Y축 양의 방향**: X축에서 반시계 90° (ROS REP-103 규약)

**원점 마킹 절차:**
1. 로봇을 시작 위치에 정렬
2. `base_footprint`의 접지점(바퀴 중심축의 바닥 투영)을 바닥에 표시 → 이것이 원점
3. 로봇 전진 방향을 줄자로 연장하여 X축 표시
4. 직각자(L-square)로 Y축 표시

#### 4.1.2 체크포인트 그리드 설계

환경 크기(약 8m × 6m)를 고려한 체크포인트 배치:

```
Y(m)
 5 ┤  ·  ·  ·  ·  ·  ·  ·  ·  ·    · = 체크포인트 (총 54개)
 4 ┤  ·  ·  ·  ·  ·  ·  ·  ·  ·    간격: 1m × 1m
 3 ┤  ·  ·  ·  ·  ·  ·  ·  ·  ·    M = ArUco 마커 (벽면)
 2 ┤  ·  ·  ·  ·  ·  ·  ·  ·  ·
 1 ┤  ·  ·  ·  ·  ·  ·  ·  ·  ·
 0 ┤  ◎  ·  ·  ·  ·  ·  ·  ·  ·    ◎ = 원점 (로봇 시작점)
   └──┬──┬──┬──┬──┬──┬──┬──┬──→ X(m)
     -2 -1  0  1  2  3  4  5  6
```

**마킹 방법:**
- **십자 테이프** (폭 5mm 이하의 컬러 비닐 테이프): 각 체크포인트에 10cm 길이의 십자(+) 마킹
- **체크포인트 ID**: 각 교차점에 `(x,y)` 좌표를 직접 기입 (예: "P(2,3)")
- **정밀도 확보**: 1m 간격은 줄자로 측정 (오차 ±3mm), 직각은 3-4-5 삼각형 또는 레이저 직각기로 검증

**최소 세트 (빠른 평가용):** 전체 54개 중 핵심 8~12개만 선택

```
선택 기준:
- 원점 근처: P(0,0), P(1,0), P(0,1) — 초기 정확도
- 환경 중앙: P(2,2), P(3,3) — 중거리 정확도
- 환경 가장자리: P(-2,0), P(6,0), P(0,5) — 최대 거리 정확도
- 코너: P(-2,5), P(6,5) — 대각 방향 정확도
- 시작점 재방문: P(0,0) — 폐루프 일관성
```

#### 4.1.3 실험 프로토콜 (5단계)

**[1단계] 좌표계 수립 및 바닥 마킹 (1회, 환경 설치 시)**

```bash
준비물: 줄자 (5m+), 직각자 또는 레이저 직각기, 컬러 비닐 테이프, 마커 펜
소요 시간: 약 30분 (1m 격자 54포인트 기준)
```

1. 원점 마킹 (로봇 시작점)
2. X축 방향 줄자 전개 (0m ~ 6m, 1m 간격 마킹)
3. 각 X축 포인트에서 Y축 방향으로 직각 전개 (0m ~ 5m, 1m 간격 마킹)
4. 직각 검증: 대각선 길이 확인 (피타고라스 정리, 예: 3m-4m 변의 대각선 = 5m)
5. 모든 교차점에 십자 테이프 + 좌표 라벨

**[2단계] Mapping 모드 실행 (맵 구축)**

```bash
# 원점에 로봇 정렬 후 Mapping 시작
ros2 launch aruco_sam_ailab aruco_slam.launch.py use_sim_time:=false run_mode:=mapping

# 환경 내 모든 마커가 관측되도록 주행
# 주행 완료 후 맵 저장
ros2 service call /save_landmarks std_srvs/srv/Trigger "{}"
```

**[3단계] Localization 모드 실행 및 체크포인트 통과 주행**

```bash
# 원점에 로봇 정렬 후 Localization 시작 + rosbag 기록
ros2 launch aruco_sam_ailab aruco_slam.launch.py use_sim_time:=false run_mode:=localization &

# rosbag 기록 시작
ros2 bag record /ekf/odom /aruco_slam/odom /odometry/imu_incremental \
    /optimized_keyframe_state /aruco_poses -o checkpoint_run_001
```

**주행 프로토콜:**
1. 원점에서 출발 → X축 방향으로 1m 간격 체크포인트 순회
2. 각 체크포인트에서 **정지** (3초간) → 로봇 `base_footprint` 접지점을 십자 중앙에 정렬
3. 정지 시점의 타임스탬프를 기록 (스톱워치 또는 `ros2 topic echo /clock`의 시간)
4. 모든 체크포인트 순회 후 원점으로 복귀
5. 원점에서 정지 (폐루프 오차 측정)

**[4단계] 데이터 후처리**

rosbag에서 체크포인트 통과 시점의 포즈를 추출하고 Ground Truth와 비교:

```python
# 후처리 스크립트 개요 (Python + rosbag2_py)
import numpy as np
from rosbags.rosbag2 import Reader

# 1. rosbag에서 /ekf/odom 타임시리즈 추출
# 2. 각 체크포인트 정지 시점(±1초)의 포즈 평균 계산
# 3. GT 좌표와 비교

checkpoints_gt = {
    'P(0,0)': (0.0, 0.0),
    'P(1,0)': (1.0, 0.0),
    'P(2,0)': (2.0, 0.0),
    # ...
}

for cp_id, (gt_x, gt_y) in checkpoints_gt.items():
    est_x, est_y = extract_pose_at_checkpoint(bag, cp_id, stop_timestamps[cp_id])
    error = np.sqrt((gt_x - est_x)**2 + (gt_y - est_y)**2)
    print(f"{cp_id}: GT=({gt_x:.2f},{gt_y:.2f}) Est=({est_x:.2f},{est_y:.2f}) Error={error:.3f}m")
```

**[5단계] 통계 분석 및 시각화**

$$\text{RMSE}_{\text{checkpoint}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \|p_{\text{est},i} - p_{\text{GT},i}\|^2}$$

시각화:
- **오차 히트맵**: 각 체크포인트에서의 위치 오차를 색상 원으로 바닥 지도 위에 표시
- **오차 벡터 플롯**: 각 포인트에서 GT→추정 방향의 화살표 → drift 방향 패턴 파악
- **거리별 오차 분포**: 원점으로부터의 거리 vs 위치 오차 산포도

#### 4.1.4 폐루프 복귀 오차 실험

체크포인트 평가와 **동시에** 수행할 수 있는 가장 간단한 정량 지표:

**프로토콜:**
1. 원점 P(0,0)에서 출발, 초기 포즈 기록: $p_0 = (x_0, y_0, \theta_0)$
2. 환경 내 임의 경로 주행 (총 주행 거리 $D$ 기록)
3. 원점 P(0,0)으로 복귀, 최종 포즈 기록: $p_f = (x_f, y_f, \theta_f)$

**정량 지표:**
$$e_{\text{trans}} = \sqrt{(x_f - x_0)^2 + (y_f - y_0)^2}$$
$$e_{\text{rot}} = |\theta_f - \theta_0| \mod 2\pi$$
$$\text{Drift Rate} = \frac{e_{\text{trans}}}{D} \times 100\%$$

**반복 실험:** 동일 경로를 5회 반복하여 평균 ± 표준편차 보고:
$$\bar{e}_{\text{trans}} \pm \sigma_e, \quad \text{95\% CI} = \bar{e} \pm 1.96 \cdot \frac{\sigma_e}{\sqrt{N}}$$

#### 4.1.5 주행 시나리오 설계

| 시나리오 | 경로 | 주행거리 | 평가 초점 |
|---|---|---|---|
| **A. 직선 왕복** | P(0,0)→P(6,0)→P(0,0) | 12m | 기본 직선 정확도, 복귀 오차 |
| **B. 사각 루프** | P(0,0)→P(6,0)→P(6,5)→P(0,5)→P(0,0) | 22m | 4방향 drift, 코너링 |
| **C. 격자 순회** | 모든 체크포인트를 S자로 순회 | ~60m | 전체 환경 커버리지 |
| **D. 마커 공백 주행** | 마커가 안 보이는 영역을 의도적으로 장기 주행 | 가변 | Dead reckoning 성능 |
| **E. 반복 주행** | 시나리오 B를 5회 반복 | 110m | 재현성 (repeatability) |
| **F. 속도 변화** | 동일 경로를 0.3m/s / 1.0m/s / 1.5m/s로 | 가변 | 속도 의존성 |

#### 4.1.6 결과 보고 형식

```
┌─────────────────────────────────────────────────────┐
│        ArUco-SAM 실환경 정확도 평가 결과 (예시)       │
├─────────────────────────────────────────────────────┤
│ 환경: 실내 8m × 6m, 마커 10개 (ID 11-20)            │
│ 로봇: Hunter 2.0, RealSense D455                    │
│ 체크포인트: 12개 (1m 격자에서 선택)                    │
│ 반복 횟수: 5회                                       │
├────────────┬────────────────────────────────────────┤
│ 지표       │ 결과                                    │
├────────────┼────────────────────────────────────────┤
│ CP RMSE    │ 0.XX ± 0.XX m (5회 평균)                │
│ CP Max     │ 0.XX m (at P(x,y))                     │
│ 폐루프 오차 │ 0.XX ± 0.XX m / 0.X ± 0.X° (5회)       │
│ Drift Rate │ X.X% (총 주행거리 대비)                  │
│ Heading 오차│ X.X ± X.X° (체크포인트 정지 시)          │
└────────────┴────────────────────────────────────────┘
```

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

#### [제안 1] ~~마커 가시성 기반 적응형 노이즈~~ → **구현 완료**

거리와 시야각에 따른 동적 노이즈 모델이 구현되었다 (§2.2.2 참조):

$$\sigma_{\text{trans}}(d, \theta) = \sigma_0 \cdot (1 + 0.3d) \cdot (1 + 1.5\cos^2\theta)$$

추가로 시야각 < 10° 및 거리 > 5m 필터링도 적용되어 극단적 관측을 제거한다.

#### [제안 2] 다중 마커 동시 관측 활용

현재 각 마커를 개별 `BetweenFactor`로 처리한다. 동시에 여러 마커가 관측될 때 마커 간 상대 포즈를 추가 제약으로 활용하면 관측 일관성이 향상된다:

$$r_{jk}^{\text{inter}} = \text{Log}(\tilde{z}_{ij}^{-1} \cdot \tilde{z}_{ik}) - \text{Log}(l_j^{-1} \cdot l_k)$$

#### [제안 3] EKF에서 속도 상태 추가

현재 EKF의 상태가 포즈(6-DOF)뿐이다. 속도를 상태에 포함하면(15-DOF: 포즈 + 속도 + 바이어스) 더 정확한 예측이 가능하지만, 현재 구현의 단순함과 200Hz 업데이트 주기를 고려하면 필수적이지는 않다.
