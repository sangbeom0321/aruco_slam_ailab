# MarkerArray.msg
# Array of marker observations with header timestamp and frame_id
#
# header: Standard ROS header with timestamp and frame_id
# markers: Array of MarkerObservation messages

std_msgs/Header header
MarkerObservation[] markers
