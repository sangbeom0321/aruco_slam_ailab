cmake_minimum_required(VERSION 3.8)
project(aruco_slam_ailab)

# Set C++ standard
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# C++ dependencies
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# GTSAM dependency
# Try both common naming conventions
find_package(GTSAM QUIET)
if(NOT GTSAM_FOUND)
  find_package(gtsam QUIET)
  if(NOT gtsam_FOUND)
    message(FATAL_ERROR "GTSAM not found. Please install libgtsam-dev or build GTSAM from source.")
  endif()
endif()

# Message generation
find_package(rosidl_default_generators REQUIRED)

# Custom message definitions
# rosidl_generate_interfaces automatically generates and installs both C++ and Python bindings
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MarkerObservation.msg"
  "msg/MarkerArray.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

# Install Python package
# Note: rosidl_generate_interfaces already installs Python message bindings
# We install our custom Python modules here
# The Python package installation must come AFTER rosidl_generate_interfaces
# to avoid target conflicts, but rosidl_generate_interfaces creates Python targets
# that conflict with ament_python_install_package
# 
# Solution: Only install Python package if it doesn't conflict
# We'll install our Python scripts manually instead

# Install Python package manually to avoid conflicts with rosidl_generate_interfaces
# We install the Python package directory and scripts separately
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}
    DESTINATION lib/python3.10/site-packages
    FILES_MATCHING PATTERN "*.py"
)

# Note: Python scripts are no longer used - we use C++ executables instead
# Old Python scripts are kept for reference but not installed


# Install launch files
install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}/
)

# Install models directory
install(DIRECTORY models
    DESTINATION share/${PROJECT_NAME}/
)

# Install rviz directory
install(DIRECTORY rviz
    DESTINATION share/${PROJECT_NAME}/
)

# Install config directory
install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME}/
)

# C++ executables
# Note: Source files will be created in Step 3-4
# For now, we prepare the structure

# Include directories
include_directories(include)

# imu_preintegration executable (to be implemented in Step 3)
add_executable(imu_preintegration src/imu_preintegration.cpp)
ament_target_dependencies(imu_preintegration
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)
# Link GTSAM library (handle different naming conventions)
if(TARGET GTSAM::gtsam)
  target_link_libraries(imu_preintegration GTSAM::gtsam)
elseif(TARGET gtsam)
  target_link_libraries(imu_preintegration gtsam)
else()
  # Fallback: link directly if targets are not available
  target_link_libraries(imu_preintegration ${GTSAM_LIBRARIES})
endif()
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(imu_preintegration "${cpp_typesupport_target}")

# graph_optimizer executable (to be implemented in Step 4)
add_executable(graph_optimizer src/graph_optimizer.cpp)
ament_target_dependencies(graph_optimizer
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)
# Link GTSAM library (handle different naming conventions)
if(TARGET GTSAM::gtsam)
  target_link_libraries(graph_optimizer GTSAM::gtsam)
elseif(TARGET gtsam)
  target_link_libraries(graph_optimizer gtsam)
else()
  # Fallback: link directly if targets are not available
  target_link_libraries(graph_optimizer ${GTSAM_LIBRARIES})
endif()
rosidl_get_typesupport_target(cpp_typesupport_target2
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(graph_optimizer "${cpp_typesupport_target2}")

# aruco_detector_node executable
add_executable(aruco_detector_node src/aruco_detector_node.cpp)
ament_target_dependencies(aruco_detector_node
  rclcpp
  sensor_msgs
  visualization_msgs
  cv_bridge
)
rosidl_get_typesupport_target(cpp_typesupport_target3
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(aruco_detector_node
  ${OpenCV_LIBS}
  "${cpp_typesupport_target3}"
)

# Install executables
install(TARGETS
  imu_preintegration
  graph_optimizer
  aruco_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
