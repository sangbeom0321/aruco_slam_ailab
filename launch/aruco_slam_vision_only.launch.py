#!/usr/bin/env python3

from os.path import join
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share_aruco = get_package_share_directory('aruco_slam_ailab')
    config_file = join(pkg_share_aruco, 'config', 'slam_params.yaml')

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
        default_value='true',
        description='Enable topic receive debug logs (aruco) for diagnostics'
    )
    
    # Launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    marker_size = LaunchConfiguration('marker_size')
    enable_topic_debug_log = LaunchConfiguration('enable_topic_debug_log', default='true')

    # ArUco detector node for front camera (C++ version)
    # Directly publishes aruco_slam_ailab/msg/MarkerArray for SLAM backend
    aruco_detector_front = Node(
        package='aruco_slam_ailab',
        executable='aruco_detector_node',
        name='aruco_detector_front',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'camera_topic': '/camera/rgb/image_raw'},
            {'camera_info_topic': '/camera/rgb/camera_info'},
            {'marker_size': marker_size},
            {'aruco_dict_type': 'DICT_4X4_50'},
        ],
        remappings=[
            ('/aruco_poses', '/aruco_poses'),  # Direct output for SLAM backend
            ('/aruco_debug_image', '/aruco_debug_image_front')
        ]
    )

    # SLAM Backend (Vision-only: Graph Optimizer without IMU)
    graph_optimizer_node = Node(
        package='aruco_slam_ailab',
        executable='graph_optimizer',
        name='graph_optimizer',
        output='screen',
        parameters=[
            config_file, 
            {'use_sim_time': use_sim_time, 
             'enable_topic_debug_log': enable_topic_debug_log,
             'use_imu': False}  # Disable IMU for vision-only mode
        ],
        remappings=[
            ('/aruco_poses', '/aruco_poses'),  # Needs MarkerArray from detector
            ('/odometry/global', '/odometry/global'),
            ('/path', '/path'),
        ]
    )

    # Static TF: map -> odom (identity; new-style args to avoid deprecation warning)
    static_tf_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_map_odom',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '0', '--frame-id', 'map', '--child-frame-id', 'odom'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # RViz node with aruco_viz.rviz config
    rviz_config = join(pkg_share_aruco, 'rviz', 'aruco_viz.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        marker_size_arg,
        enable_topic_debug_log_arg,
        aruco_detector_front,
        graph_optimizer_node,
        static_tf_map_odom,
        rviz_node
    ])
