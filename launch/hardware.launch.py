#!/usr/bin/env python3
# 실제 하드웨어용: RealSense 카메라 + ArUco 디텍터 + IMU Preintegration + Graph Optimizer
# Camera-only SLAM (no wheel odometry)

import yaml
from os.path import join
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share_aruco = get_package_share_directory('aruco_sam_ailab')
    config_file = join(pkg_share_aruco, 'config', 'slam_params.yaml')

    # Launch arguments
    marker_size_arg = DeclareLaunchArgument(
        'marker_size',
        default_value='0.105',
        description='ArUco marker size in meters'
    )
    marker_size = LaunchConfiguration('marker_size')

    # RealSense 카메라 (Color + Depth + IMU)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ]),
        launch_arguments={
            'pointcloud.enable': 'false',
            'enable_gyro': 'true',
            'enable_accel': 'true',
            'unite_imu_method': '2',
            'align_depth.enable': 'true',
        }.items(),
    )

    # ArUco 디텍터 (RealSense 토픽에 맞게 설정)
    aruco_detector = Node(
        package='aruco_sam_ailab',
        executable='aruco_detector_node',
        name='aruco_detector_front',
        parameters=[{
            'use_sim_time': True,
            'camera_topic': '/camera/camera/color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'depth_topic': '/camera/camera/depth/image_rect_raw',
            'depth_camera_info_topic': '/camera/camera/depth/camera_info',
            'marker_size': marker_size,
            'use_depth_correction': False,
            'depth_max_range': 5.0,
            'depth_sample_radius': 3,
            'allowed_marker_ids': [0, 1],
        }],
        output='screen',
    )

    # IMU Preintegration (RealSense IMU → /odometry/imu_incremental)
    imu_preintegration = Node(
        package='aruco_sam_ailab',
        executable='imu_preintegration',
        name='imu_preintegration',
        parameters=[config_file, {'use_sim_time': True}],
        output='screen',
    )

    # Graph Optimizer (SLAM backend: ArUco + IMU odom)
    graph_optimizer = Node(
        package='aruco_sam_ailab',
        executable='graph_optimizer',
        name='graph_optimizer',
        parameters=[config_file, {'use_sim_time': True}],
        output='screen',
    )

    # EKF Smoother (IMU odom + SLAM correction → smooth odometry)
    ekf_smoother = Node(
        package='aruco_sam_ailab',
        executable='ekf_smoother',
        name='ekf_smoother',
        parameters=[config_file, {'use_sim_time': True}],
        output='screen',
    )

    # RViz2
    rviz_config = join(pkg_share_aruco, 'config', 'slam_rviz.rviz')
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    return LaunchDescription([
        marker_size_arg,
        # realsense_launch,
        aruco_detector,
        imu_preintegration,
        graph_optimizer,
        ekf_smoother,
        rviz2,
    ])
