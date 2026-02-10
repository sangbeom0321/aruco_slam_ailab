#!/usr/bin/env python3
# IMU only 모드: ArUco/휠 오도메트리 없이 IMU 사전적분만으로 경향 확인 (발산 예상)

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

    enable_topic_debug_log_arg = DeclareLaunchArgument(
        'enable_topic_debug_log',
        default_value='true',
        description='Enable topic receive debug logs (IMU) for diagnostics'
    )

    # Launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_topic_debug_log = LaunchConfiguration('enable_topic_debug_log', default='true')

    # IMU 사전적분: /imu -> /odometry/imu_incremental
    imu_preintegration_node = Node(
        package='aruco_slam_ailab',
        executable='imu_preintegration',
        name='imu_preintegration',
        output='screen',
        parameters=[config_file, {'use_sim_time': use_sim_time, 'enable_topic_debug_log': enable_topic_debug_log}],
        remappings=[
            ('/imu', '/camera/imu'),
            ('/odometry/imu_incremental', '/odometry/imu_incremental'),
        ]
    )

    # SLAM Backend (IMU only: no vision, no wheel odom)
    # 키프레임( factor ) 추가는 0.1초에 한 번만 (IMU only 시 그래프/연산 부담 완화)
    graph_optimizer_node = Node(
        package='aruco_slam_ailab',
        executable='graph_optimizer',
        name='graph_optimizer',
        output='screen',
        parameters=[
            config_file,
            {
                'use_sim_time': use_sim_time,
                'enable_topic_debug_log': enable_topic_debug_log,
                'use_imu': True,
                'use_wheel_odom': False,
                'keyframe_time_interval': 0.1,  # IMU only: 0.1초에 한 번만 factor 추가
            }
        ],
        remappings=[
            ('/odometry/imu_incremental', '/odometry/imu_incremental'),
            ('/aruco_poses', '/aruco_poses'),
            ('/odometry/global', '/odometry/global'),
            ('/path', '/path'),
        ]
    )

    # Static TF: map -> odom
    static_tf_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_map_odom',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '0', '--frame-id', 'map', '--child-frame-id', 'odom'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # RViz
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
        enable_topic_debug_log_arg,
        imu_preintegration_node,
        graph_optimizer_node,
        static_tf_map_odom,
        rviz_node
    ])
