#!/usr/bin/env python3
"""
랜드마크 벽 테두리 Occupancy Grid 노드
=====================================
Localization 모드에서 landmarks_map.json의 id 0~9를 직선으로 연결하여
벽 테두리로 하는 2D occupancy grid를 발행합니다.
"""

import json
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


def bresenham_line(x0: int, y0: int, x1: int, y1: int):
    """Bresenham's line algorithm - yields (x, y) cell indices."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        yield (x, y)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


class LandmarkBoundaryOccupancyGridNode(Node):
    def __init__(self):
        super().__init__('landmark_boundary_occupancy_grid_node')

        self.declare_parameter('map_path', '')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('resolution', 0.05)  # m per cell
        self.declare_parameter('wall_thickness', 2)  # cells (벽 두께)
        self.declare_parameter('publish_rate', 1.0)  # Hz

        map_path = self.get_parameter('map_path').get_parameter_value().string_value
        if not map_path:
            pkg_share = self._get_package_share_directory()
            if pkg_share:
                map_path = str(Path(pkg_share) / 'map' / 'landmarks_map.json')
            else:
                self.get_logger().error('map_path가 비어있고 패키지 경로를 찾을 수 없습니다.')
                raise ValueError('map_path required')

        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.resolution = self.get_parameter('resolution').get_parameter_value().double_value
        self.wall_thickness = self.get_parameter('wall_thickness').get_parameter_value().integer_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        self.landmarks_xy = self._load_landmarks_0_to_9(map_path)
        if not self.landmarks_xy:
            self.get_logger().error('랜드마크 0~9를 로드할 수 없습니다. map_path=%s' % map_path)
            raise ValueError('No landmarks 0-9 in map file')

        self.landmarks_all = self._load_all_landmarks(map_path)

        self.pub_map = self.create_publisher(OccupancyGrid, '/map', 10)
        self.pub_landmarks = self.create_publisher(MarkerArray, '/aruco_slam/landmarks', 10)
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_map)
        self.get_logger().info(
            'Landmark boundary occupancy grid 노드 시작 (랜드마크 %d개, resolution=%.3f, viz %d개)'
            % (len(self.landmarks_xy), self.resolution, len(self.landmarks_all))
        )

    def _get_package_share_directory(self):
        try:
            from ament_index_python.packages import get_package_share_directory
            return get_package_share_directory('aruco_sam_ailab')
        except Exception:
            return None

    def _load_landmarks_0_to_9(self, map_path: str) -> list:
        """맵 파일에서 id 0~9 랜드마크를 id 순으로 (x, y) 리스트 반환."""
        path = Path(map_path)
        if not path.exists():
            self.get_logger().warn('맵 파일이 없습니다: %s' % map_path)
            return []

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        landmarks = data.get('landmarks', [])
        id_to_xy = {}
        for lm in landmarks:
            lid = lm.get('id')
            if lid is not None and 0 <= lid <= 9:
                pos = lm.get('position', {})
                x = pos.get('x', 0.0)
                y = pos.get('y', 0.0)
                id_to_xy[lid] = (x, y)

        # id 0,1,2,...,7,9,8 순서로 정렬 (8번·9번 순서 교환, 없는 id는 건너뜀)
        result = []
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]:
            if i in id_to_xy:
                result.append(id_to_xy[i])
        return result

    def _load_all_landmarks(self, map_path: str) -> list:
        """맵 파일에서 모든 랜드마크를 (id, x, y, z) 리스트로 반환."""
        path = Path(map_path)
        if not path.exists():
            return []

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        result = []
        for lm in data.get('landmarks', []):
            lid = lm.get('id')
            if lid is None:
                continue
            pos = lm.get('position', {})
            x = pos.get('x', 0.0)
            y = pos.get('y', 0.0)
            z = pos.get('z', 0.0)
            result.append((lid, x, y, z))
        return result

    def _world_to_grid(self, x: float, y: float, origin_x: float, origin_y: float) -> tuple:
        """월드 좌표를 그리드 인덱스로 변환."""
        gx = int(round((x - origin_x) / self.resolution))
        gy = int(round((y - origin_y) / self.resolution))
        return (gx, gy)

    def _draw_wall_line(self, grid: list, width: int, height: int,
                        x0: int, y0: int, x1: int, y1: int):
        """두 그리드 셀을 잇는 선을 OCCUPIED(100)로 그리기. 두께 적용."""
        for (gx, gy) in bresenham_line(x0, y0, x1, y1):
            for dx in range(-self.wall_thickness, self.wall_thickness + 1):
                for dy in range(-self.wall_thickness, self.wall_thickness + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        idx = ny * width + nx
                        grid[idx] = 100  # OCCUPIED

    def _point_in_polygon(self, px: float, py: float, polygon: list) -> bool:
        """Ray casting: (px, py)가 polygon 내부인지 판단. polygon: [(x,y), ...]"""
        n = len(polygon)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if (yj - yi) != 0 and ((yi > py) != (yj > py)):
                if px < (xj - xi) * (py - yi) / (yj - yi) + xi:
                    inside = not inside
            j = i
        return inside

    def _build_occupancy_grid(self) -> OccupancyGrid:
        """랜드마크 0~9를 직선으로 연결한 벽 테두리 occupancy grid 생성."""
        if len(self.landmarks_xy) < 2:
            return None

        xs = [p[0] for p in self.landmarks_xy]
        ys = [p[1] for p in self.landmarks_xy]
        margin = 1.0  # m
        min_x = min(xs) - margin
        max_x = max(xs) + margin
        min_y = min(ys) - margin
        max_y = max(ys) + margin

        width = int(math.ceil((max_x - min_x) / self.resolution))
        height = int(math.ceil((max_y - min_y) / self.resolution))
        width = max(width, 10)
        height = max(height, 10)

        # -1: UNKNOWN, 0: FREE, 100: OCCUPIED
        grid = [-1] * (width * height)

        # 랜드마크를 그리드 좌표로 변환 (ROS map: row 0 = bottom, origin at min_x, min_y)
        points = []
        for (wx, wy) in self.landmarks_xy:
            gx, gy = self._world_to_grid(wx, wy, min_x, min_y)
            gy = max(0, min(height - 1, gy))
            gx = max(0, min(width - 1, gx))
            points.append((gx, gy))

        # 0->1->2->...->9->0 직선 연결 (벽 = OCCUPIED)
        for i in range(len(points)):
            j = (i + 1) % len(points)
            self._draw_wall_line(
                grid, width, height,
                points[i][0], points[i][1],
                points[j][0], points[j][1]
            )

        # 벽 안쪽: FREE(0), 바깥: UNKNOWN(-1)
        for gy in range(height):
            for gx in range(width):
                idx = gy * width + gx
                if grid[idx] == 100:
                    continue  # 벽은 그대로
                cx, cy = gx + 0.5, gy + 0.5  # 셀 중심
                if self._point_in_polygon(cx, cy, points):
                    grid[idx] = 0  # FREE

        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.resolution
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = min_x
        msg.info.origin.position.y = min_y
        msg.info.origin.position.z = 0.0  # z=0 평면
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid
        return msg

    def _build_landmarks_marker_array(self) -> MarkerArray:
        """JSON 랜드마크를 RViz용 MarkerArray로 생성 (graph_optimizer와 동일 포맷)."""
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for (lid, x, y, z) in self.landmarks_all:
            sphere = Marker()
            sphere.header.stamp = stamp
            sphere.header.frame_id = self.frame_id
            sphere.ns = 'landmarks'
            sphere.id = int(lid)
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.2
            sphere.color.r = 0.2
            sphere.color.g = 0.6
            sphere.color.b = 1.0
            sphere.color.a = 0.9
            arr.markers.append(sphere)

            text = Marker()
            text.header.stamp = stamp
            text.header.frame_id = self.frame_id
            text.ns = 'landmark_ids'
            text.id = int(lid)
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = z + 0.15
            text.pose.orientation.w = 1.0
            text.scale.z = 0.3
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = str(lid)
            arr.markers.append(text)
        return arr

    def publish_map(self):
        msg = self._build_occupancy_grid()
        if msg:
            self.pub_map.publish(msg)
        if self.landmarks_all:
            self.pub_landmarks.publish(self._build_landmarks_marker_array())


def main(args=None):
    rclpy.init(args=args)
    node = LandmarkBoundaryOccupancyGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
