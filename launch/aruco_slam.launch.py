#!/usr/bin/env python3
# IMU+Vision SLAM: ArUco detector + graph_optimizer + imu_preintegration + ekf_smoother.
# ArUco 관측: camera_color_optical_frame 사용.
# use_sim_time에 따라 slam_params_sim.yaml 또는 slam_params_real.yaml 자동 선택.

import yaml
from os.path import join
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    pkg_share_aruco = get_package_share_directory('aruco_sam_ailab')

    # use_sim_time 값에 따라 config 파일 선택
    use_sim_time_str = context.launch_configurations['use_sim_time']
    is_sim = use_sim_time_str.lower() in ('true', '1')
    config_name = 'slam_params_sim.yaml' if is_sim else 'slam_params_real.yaml'
    config_file = join(pkg_share_aruco, 'config', config_name)

    # Launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    run_mode = LaunchConfiguration('run_mode')
    map_path = LaunchConfiguration('map_path')
    enable_topic_debug_log = LaunchConfiguration('enable_topic_debug_log', default='true')

    # ArUco detector node for front camera (camera_color_optical_frame)
    # use_depth_correction: Depth로 Z(거리) 보정 → PnP Jitter 감소
    # 하드웨어(RealSense): /camera/color/*, /camera/depth/image_rect_raw
    # 시뮬레이션(Gazebo):  /camera/rgb/*, /camera/depth/depth/* (노드 기본값)
    # boundary_id_order에 해당하는 실제 마커 ID만 허용 (거짓 양성 방지)
    aruco_params = {
        'use_sim_time': use_sim_time,
        'depth_max_range': 5.0,
        'depth_sample_radius': 3,
        'allowed_marker_ids': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    }
    if is_sim:
        aruco_params['use_depth_correction'] = True
    else:
        aruco_params['use_depth_correction'] = False
        aruco_params['camera_topic'] = '/camera/color/image_raw'
        aruco_params['camera_info_topic'] = '/camera/color/camera_info'
        aruco_params['depth_topic'] = '/camera/depth/image_rect_raw'
        aruco_params['depth_camera_info_topic'] = '/camera/depth/camera_info'

    aruco_detector_front = Node(
        package='aruco_sam_ailab',
        executable='aruco_detector_node',
        name='aruco_detector_front',
        parameters=[aruco_params],
        output='screen',
        respawn=True,
        respawn_delay=2.0,
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
    # id 11~20을 직선으로 연결한 벽 테두리 → /map (nav_msgs/OccupancyGrid)
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
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # IMU Preintegration: raw IMU → dead reckoning odom (~200Hz)
    # mapping: origin에서 즉시 시작, localization: SLAM 초기 위치 추정 대기
    # imu_topic은 config 파일에서 자동 로드 (sim/real 분리)
    imu_preintegration_node = Node(
        package='aruco_sam_ailab',
        executable='imu_preintegration',
        name='imu_preintegration',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': use_sim_time,
             'run_mode': run_mode},
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
        respawn=True,
        respawn_delay=2.0,
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # map → odom: static identity TF (EKF smoother가 odom → base_footprint 발행)
    static_tf_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # graph_optimizer: 5-second delay for startup race condition mitigation
    graph_optimizer_action = TimerAction(
        period=5.0,
        actions=[graph_optimizer_node]
    )

    return [
        static_tf_map_odom,
        imu_preintegration_node,
        aruco_detector_front,
        graph_optimizer_action,
        ekf_smoother_node,
        landmark_boundary_map_node,
        ego_state_node,
        rviz_node,
    ]


def generate_launch_description():
    # Launch arguments (OpaqueFunction 내부에서 resolve)
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    run_mode_arg = DeclareLaunchArgument(
        'run_mode',
        default_value='mapping',
        description='SLAM run mode: "mapping" or "localization"'
    )

    map_path_arg = DeclareLaunchArgument(
        'map_path',
        default_value='/ros_ws/src/aruco_sam_ailab/map/landmarks_map.json',
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

    return LaunchDescription([
        use_sim_time_arg,
        run_mode_arg,
        map_path_arg,
        marker_size_arg,
        enable_topic_debug_log_arg,
        OpaqueFunction(function=launch_setup),
    ])
