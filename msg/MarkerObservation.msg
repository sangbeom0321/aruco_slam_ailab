# MarkerObservation.msg
# Represents a single ArUco marker observation with ID and relative pose
#
# id: ArUco marker ID (int32)
# pose: Relative pose T_(observer_frame -> marker_frame)
#       This is the pose of the marker frame relative to the observer frame
#       (typically base_link or camera frame, configurable via parameter)

int32 id
geometry_msgs/Pose pose
