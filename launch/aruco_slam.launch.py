#!/usr/bin/env python3
# IMU+Vision SLAM: ArUco detector + graph_optimizer + imu_preintegration + ekf_smoother.
# ArUco 관측: camera_color_optical_frame 사용.

import yaml
from os.path import join
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share_aruco = get_package_share_directory('aruco_sam_ailab')
    config_file = join(pkg_share_aruco, 'config', 'slam_params.yaml')

    # slam_params.yaml에서 기본값 읽기 (launch 인자 미지정 시 fallback)
    run_mode_default = 'mapping'
    map_path_default = join(pkg_share_aruco, 'map', 'landmarks_map.json')
    try:
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        params = cfg.get('/**', {}).get('ros__parameters', {})
        run_mode_default = params.get('run_mode', 'mapping')
        map_path_default = params.get('map_path', map_path_default)
    except Exception:
        pass

    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    run_mode_arg = DeclareLaunchArgument(
        'run_mode',
        default_value=run_mode_default,
        description='SLAM run mode: "mapping" or "localization"'
    )

    map_path_arg = DeclareLaunchArgument(
        'map_path',
        default_value=map_path_default,
        description='Landmark map file path for localization mode'
    )

    marker_size_arg = DeclareLaunchArgument(
        'marker_size',
        default_value='0.30',
        description='ArUco marker size in meters'
    )

    enable_topic_debug_log_arg = DeclareLaunchArgument(
        'enable_topic_debug_log',
        default_value='false',
        description='Enable topic receive debug logs (aruco) for diagnostics'
    )

    imu_topic_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value='/camera/imu',
        description='Raw IMU topic (sim: /camera/imu, real: /camera/camera/imu)'
    )

    # Launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    run_mode = LaunchConfiguration('run_mode')
    map_path = LaunchConfiguration('map_path')
    marker_size = LaunchConfiguration('marker_size')
    enable_topic_debug_log = LaunchConfiguration('enable_topic_debug_log', default='true')
    imu_topic = LaunchConfiguration('imu_topic', default='/camera/imu')

    # ArUco detector node for front camera (camera_color_optical_frame)
    # use_depth_correction: Depth로 Z(거리) 보정 → PnP Jitter 감소
    aruco_detector_front = Node(
        package='aruco_sam_ailab',
        executable='aruco_detector_node',
        name='aruco_detector_front',
        parameters=[{
            'use_sim_time': use_sim_time,
            'use_depth_correction': True,
            'depth_max_range': 5.0,
            'depth_sample_radius': 3,
        }],
        output='screen'
    )

    # SLAM Backend
    graph_optimizer_node = Node(
        package='aruco_sam_ailab',
        executable='graph_optimizer',
        name='graph_optimizer',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': use_sim_time,
             'run_mode': run_mode,
             'map_path': map_path,
             'enable_topic_debug_log': enable_topic_debug_log},
        ],
    )

    # Landmark boundary occupancy grid (run_mode=="localization"일 때만)
    # id 0~9를 직선으로 연결한 벽 테두리 → /map (nav_msgs/OccupancyGrid)
    is_localization = PythonExpression(["'", run_mode, "' == 'localization'"])
    landmark_boundary_map_node = Node(
        package='aruco_sam_ailab',
        executable='landmark_boundary_occupancy_grid_node.py',
        name='landmark_boundary_occupancy_grid_node',
        output='screen',
        condition=IfCondition(is_localization),
        respawn=True,
        respawn_delay=2.0,
        parameters=[
            config_file,
            {'use_sim_time': use_sim_time},
            {'map_path': map_path},
            {'frame_id': 'map'},
            {'resolution': 0.05},
            {'wall_thickness': 1},
            {'publish_rate': 1.0},
        ],
    )

    # RViz node with aruco_viz.rviz config
    rviz_config = join(pkg_share_aruco, 'rviz', 'aruco_viz.rviz')
    # rviz_config = join(pkg_share_aruco, 'rviz', 'aruco_detection_debug.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
    )

    # IMU Preintegration: raw IMU → dead reckoning odom (~200Hz)
    # mapping: origin에서 즉시 시작, localization: SLAM 초기 위치 추정 대기
    imu_preintegration_node = Node(
        package='aruco_sam_ailab',
        executable='imu_preintegration',
        name='imu_preintegration',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': use_sim_time,
             'run_mode': run_mode,
             'imu_topic': imu_topic},
        ],
    )

    # EKF Smoother: IMU odom + SLAM correction → smoothed odom (map frame, ~200Hz)
    ekf_smoother_node = Node(
        package='aruco_sam_ailab',
        executable='ekf_smoother',
        name='ekf_smoother',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': use_sim_time,
             'enable_topic_debug_log': enable_topic_debug_log},
        ],
    )

    # EgoState publisher: /ekf/odom (EKF 융합) → /ego_state (~200Hz)
    ego_state_node = Node(
        package='aruco_sam_ailab',
        executable='ego_state_publisher.py',
        name='ego_state_publisher',
        output='screen',
        respawn=True,           # 추가: 죽으면 자동 부활
        respawn_delay=2.0,      # 추가: 부활 간격 2초
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # map → odom: static identity TF (EKF smoother가 odom → base_footprint 발행)
    static_tf_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )

    # graph_optimizer: 5-second delay for startup race condition mitigation
    graph_optimizer_action = TimerAction(
        period=5.0,
        actions=[graph_optimizer_node]
    )

    return LaunchDescription([
        use_sim_time_arg,
        run_mode_arg,
        map_path_arg,
        marker_size_arg,
        enable_topic_debug_log_arg,
        imu_topic_arg,
        static_tf_map_odom,
        imu_preintegration_node,
        aruco_detector_front,
        graph_optimizer_action,
        ekf_smoother_node,
        landmark_boundary_map_node,
        ego_state_node,
        # RViz는 hunter2_bringup/navigation.launch.py에서 통합 실행
        # (단독 실행 시 필요하면 아래 주석 해제)
        # rviz_node,
    ])
