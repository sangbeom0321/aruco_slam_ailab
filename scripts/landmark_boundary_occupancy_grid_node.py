#!/usr/bin/env python3
"""
랜드마크 벽 테두리 Occupancy Grid 노드
=====================================
Localization 모드에서 landmarks_map.json의 id 11~20을 직선으로 연결하여
벽 테두리로 하는 2D occupancy grid를 발행합니다.

발행 토픽:
  /global_map (OccupancyGrid): 전체 ArUco 경계 맵
  /local_map  (OccupancyGrid): 로봇 주변 local_map_size × local_map_size 윈도우
"""

import json
import math
from pathlib import Path

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener


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
        self.declare_parameter('resolution', 0.1)  # m per cell (성능 최적화: 0.05 → 0.1)
        self.declare_parameter('wall_thickness', 1)  # cells (벽 두께)
        self.declare_parameter('publish_rate', 0.5)  # Hz
        self.declare_parameter('local_map_size', 10.0)  # m (로봇 주변 윈도우 크기)
        self.declare_parameter('boundary_id_order', [11, 12, 13, 14, 15, 16, 17, 18, 20, 19])  # 벽 연결 순서

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
        self.local_map_size = self.get_parameter('local_map_size').get_parameter_value().double_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        self.boundary_id_order = self.get_parameter('boundary_id_order').get_parameter_value().integer_array_value
        self.landmarks_xy = self._load_landmarks_by_order(map_path)
        if not self.landmarks_xy:
            self.get_logger().error('랜드마크 11~20을 로드할 수 없습니다. map_path=%s' % map_path)
            raise ValueError('No landmarks 11-20 in map file')

        self.landmarks_all = self._load_all_landmarks(map_path)

        # TF2 Buffer & Listener (map → base_footprint)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers — TransientLocal QoS: planner_controller_node 구독자와 QoS 일치
        # (Volatile publisher + TransientLocal subscriber = 연결 불가)
        latched_qos = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
        )
        self.pub_map = self.create_publisher(OccupancyGrid, '/global_map', latched_qos)
        self.pub_local_map = self.create_publisher(OccupancyGrid, '/local_map', latched_qos)
        self.pub_landmarks = self.create_publisher(MarkerArray, '/aruco_slam/landmarks', 10)

        # 수동 장애물 (RViz Publish Point 도구 → /clicked_point)
        self.manual_obstacles = []  # [(x, y), ...]
        self.sub_clicked = self.create_subscription(
            PointStamped, '/clicked_point', self._clicked_point_callback, 10)

        self.timer = self.create_timer(1.0 / publish_rate, self.publish_map)
        self.get_logger().info(
            'Landmark boundary occupancy grid 노드 시작 (랜드마크 %d개, resolution=%.3f, '
            'local_map_size=%.1fm, viz %d개)'
            % (len(self.landmarks_xy), self.resolution,
                self.local_map_size, len(self.landmarks_all))
        )

    def _get_package_share_directory(self):
        try:
            from ament_index_python.packages import get_package_share_directory
            return get_package_share_directory('aruco_sam_ailab')
        except Exception:
            return None

    def _load_landmarks_by_order(self, map_path: str) -> list:
        """맵 파일에서 boundary_id_order 순서로 (x, y) 리스트 반환."""
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
            if lid is not None:
                pos = lm.get('position', {})
                x = pos.get('x', 0.0)
                y = pos.get('y', 0.0)
                id_to_xy[lid] = (x, y)

        result = []
        for i in self.boundary_id_order:
            if i in id_to_xy:
                result.append(id_to_xy[i])
            else:
                self.get_logger().warn('boundary_id_order에 지정된 ID %d가 맵에 없습니다' % i)
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

    def _clicked_point_callback(self, msg: PointStamped):
        """RViz Publish Point 도구 클릭 → 해당 좌표에 수동 장애물 추가."""
        self.manual_obstacles.append((msg.point.x, msg.point.y))
        self.get_logger().info(
            f'장애물 추가: ({msg.point.x:.2f}, {msg.point.y:.2f}), 총 {len(self.manual_obstacles)}개')

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
        """랜드마크 11~20을 직선으로 연결한 벽 테두리 occupancy grid 생성."""
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

        # 수동 장애물 마킹 (반경 0.3m 원형)
        obstacle_radius_cells = max(1, int(0.3 / self.resolution))
        for (ox, oy) in self.manual_obstacles:
            ogx, ogy = self._world_to_grid(ox, oy, min_x, min_y)
            for dx in range(-obstacle_radius_cells, obstacle_radius_cells + 1):
                for dy in range(-obstacle_radius_cells, obstacle_radius_cells + 1):
                    if dx * dx + dy * dy <= obstacle_radius_cells * obstacle_radius_cells:
                        nx, ny = ogx + dx, ogy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            grid[ny * width + nx] = 100  # OCCUPIED

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

    def _build_local_occupancy_grid(self, robot_x: float, robot_y: float,
                                    full_grid: OccupancyGrid) -> OccupancyGrid:
        """로봇 주변 local_map_size × local_map_size 윈도우를 full_grid에서 잘라 반환."""
        half = self.local_map_size / 2.0
        local_origin_x = robot_x - half
        local_origin_y = robot_y - half
        n_cells = int(self.local_map_size / self.resolution)

        full_ox = full_grid.info.origin.position.x
        full_oy = full_grid.info.origin.position.y
        full_w = full_grid.info.width
        full_h = full_grid.info.height

        local_data = []
        for gy in range(n_cells):
            for gx in range(n_cells):
                wx = local_origin_x + (gx + 0.5) * self.resolution
                wy = local_origin_y + (gy + 0.5) * self.resolution
                fgx = int((wx - full_ox) / self.resolution)
                fgy = int((wy - full_oy) / self.resolution)
                if 0 <= fgx < full_w and 0 <= fgy < full_h:
                    local_data.append(full_grid.data[fgy * full_w + fgx])
                else:
                    local_data.append(-1)  # unknown (맵 범위 밖)

        msg = OccupancyGrid()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.resolution
        msg.info.width = n_cells
        msg.info.height = n_cells
        msg.info.origin.position.x = local_origin_x
        msg.info.origin.position.y = local_origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = local_data
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
        # [1] 전체 경계 맵 발행 → /global_map
        full_msg = self._build_occupancy_grid()
        if full_msg is None:
            return
        self.pub_map.publish(full_msg)

        # [2] 로컬 윈도우 맵 발행 → /local_map (TF로 로봇 위치 조회 후 crop)
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
            rx = t.transform.translation.x
            ry = t.transform.translation.y
            local_msg = self._build_local_occupancy_grid(rx, ry, full_msg)
            self.pub_local_map.publish(local_msg)
        except Exception:
            pass  # TF 미준비 시 local_map 발행 생략 (초기화 중)

        # [3] 랜드마크 시각화 마커 발행
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
        if rclpy.ok():          # 추가: rclpy가 아직 켜져 있을 때만 끕니다.
            rclpy.shutdown()


if __name__ == '__main__':
    main()
