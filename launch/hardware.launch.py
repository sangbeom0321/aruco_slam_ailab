#!/usr/bin/env python3
# 실제 하드웨어용: RealSense 카메라 + ArUco 디텍터 + rqt_image_view
# docker compose up 시 자동 실행됨

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

    # Launch arguments
    marker_size_arg = DeclareLaunchArgument(
        'marker_size',
        default_value='0.30',
        description='ArUco marker size in meters'
    )
    marker_size = LaunchConfiguration('marker_size')

    # RealSense 카메라
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ]),
        launch_arguments={
            'pointcloud.enable': 'true',
        }.items(),
    )

    # ArUco 디텍터 (RealSense 토픽에 맞게 설정)
    aruco_detector = Node(
        package='aruco_sam_ailab',
        executable='aruco_detector_node',
        name='aruco_detector_front',
        parameters=[{
            'use_sim_time': False,
            'camera_topic': '/camera/camera/color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'depth_topic': '/camera/camera/depth/image_rect_raw',
            'depth_camera_info_topic': '/camera/camera/depth/camera_info',
            'marker_size': marker_size,
            'use_depth_correction': True,
            'depth_max_range': 5.0,
            'depth_sample_radius': 3,
        }],
        output='screen',
    )

    return LaunchDescription([
        marker_size_arg,
        realsense_launch,
        aruco_detector,
    ])
