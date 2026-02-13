#!/usr/bin/env python3
# Vision-only SLAM: ArUco detector + graph_optimizer (use_imu=False).
# ArUco 관측: camera_color_optical_frame 사용.

import yaml
from os.path import join
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share_aruco = get_package_share_directory('aruco_sam_ailab')
    config_file = join(pkg_share_aruco, 'config', 'slam_params.yaml')

    # slam_params.yaml에서 run_mode, map_path 읽기 (occupancy grid 노드 조건·경로용)
    run_mode_from_config = 'mapping'
    map_path_from_config = join(pkg_share_aruco, 'map', 'landmarks_map.json')
    try:
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        params = cfg.get('/**', {}).get('ros__parameters', {})
        run_mode_from_config = params.get('run_mode', 'mapping')
        map_path_from_config = params.get('map_path', map_path_from_config)
    except Exception:
        pass

    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
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

    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/w_odom',
        description='Wheel odom topic for OdomBetweenFactor (nav_msgs/Odometry, base_link).'
    )

    # run_mode, map_path: slam_params.yaml에서만 설정 (launch 인자 제거)
    # Launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    marker_size = LaunchConfiguration('marker_size')
    enable_topic_debug_log = LaunchConfiguration('enable_topic_debug_log', default='true')
    odom_topic = LaunchConfiguration('odom_topic', default='/w_odom')

    # Wheel Odometry (joint_states -> wheel_odom)
    # use_sim_time: Gazebo와 함께 실행 시 true (TF·odom stamp가 /clock 사용)
    wheel_odom_node = Node(
        package='aruco_sam_ailab',
        executable='wheel_odom_node',
        name='wheel_odom_node',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'wheel_radius': 0.165},
            {'track_width': 0.605},
            {'base_frame': 'base_link'},
            {'odom_frame': 'odom'},
            {'publish_tf': True},  # TF odom->base_link (graph_optimizer는 publish_tf=false)
            {'wheel_odom_topic': '/w_odom'},
            {'sync_with_slam_topic': '/aruco_slam/odom'},
            {'wheel_odom_correction_topic': '/odometry/wheel_odom_correction'},
        ],
    )

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
             'enable_topic_debug_log': enable_topic_debug_log,
             'use_imu': False,
             'odom_topic': odom_topic},
        ],
        remappings=[
            ('/aruco_poses', '/aruco_poses'),  # Needs MarkerArray from detector
            ('/odometry/global', '/odometry/global'),
            ('/path', '/path'),
        ]
    )

    # Landmark boundary occupancy grid (slam_params.yaml run_mode=="localization"일 때만)
    # id 0~9를 직선으로 연결한 벽 테두리 → /map (nav_msgs/OccupancyGrid)
    landmark_boundary_map_node = Node(
        package='aruco_sam_ailab',
        executable='landmark_boundary_occupancy_grid_node.py',
        name='landmark_boundary_occupancy_grid_node',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'map_path': map_path_from_config},
            {'frame_id': 'map'},
            {'resolution': 0.05},
            {'wall_thickness': 2},
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
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    ld_actions = [
        use_sim_time_arg,
        marker_size_arg,
        enable_topic_debug_log_arg,
        odom_topic_arg,
        wheel_odom_node,
        aruco_detector_front,
        graph_optimizer_node,
    ]
    if run_mode_from_config == 'localization':
        ld_actions.append(landmark_boundary_map_node)
    ld_actions.append(rviz_node)

    return LaunchDescription(ld_actions)
