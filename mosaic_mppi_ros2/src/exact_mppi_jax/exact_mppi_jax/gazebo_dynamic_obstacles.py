#!/usr/bin/env python

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rclpy
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Empty
import yaml

from exact_mppi_jax.utils import yaw_to_quat


Point = Tuple[float, float, float]


def _as_point(value: Sequence[float], default_z: float) -> Point:
    if len(value) < 2:
        raise ValueError(f"Point requires at least x and y: {value}")
    z = float(value[2]) if len(value) >= 3 else float(default_z)
    return (float(value[0]), float(value[1]), z)


def _wrap_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass
class ObstacleTrajectory:
    name: str
    points: List[Point]
    speed: float
    loop: str
    yaw_mode: str
    start_delay: float
    default_yaw: float

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError(f"Obstacle '{self.name}' needs at least two trajectory points")
        if self.speed <= 0:
            raise ValueError(f"Obstacle '{self.name}' has invalid speed: {self.speed}")
        if isinstance(self.loop, bool):
            self.loop = "ping_pong" if self.loop else "once"
        self.loop = str(self.loop).lower()
        self.yaw_mode = self.yaw_mode.lower()
        if self.loop not in {"cycle", "ping_pong", "once"}:
            raise ValueError(f"Obstacle '{self.name}' has unsupported loop mode: {self.loop}")
        if self.yaw_mode not in {"tangent", "fixed"}:
            raise ValueError(f"Obstacle '{self.name}' has unsupported yaw_mode: {self.yaw_mode}")

        self._cycle_points = list(self.points)
        if self.loop == "ping_pong":
            self._cycle_points = self.points + self.points[-2::-1]

        self._segments: List[Tuple[Point, Point, float]] = []
        for i in range(len(self._cycle_points) - 1):
            self._add_segment(self._cycle_points[i], self._cycle_points[i + 1])
        if self.loop == "cycle":
            self._add_segment(self._cycle_points[-1], self._cycle_points[0])

        self._total_distance = sum(seg[2] for seg in self._segments)
        if self._total_distance <= 1e-9:
            raise ValueError(f"Obstacle '{self.name}' trajectory has zero length")
        self._duration = self._total_distance / self.speed

    def _add_segment(self, start: Point, end: Point) -> None:
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        if distance > 1e-9:
            self._segments.append((start, end, distance))

    def sample(self, elapsed_s: float) -> Tuple[Point, float, Tuple[float, float]]:
        t = max(0.0, elapsed_s - self.start_delay)
        if self.loop in {"cycle", "ping_pong"}:
            distance_along = (t * self.speed) % self._total_distance
        else:
            distance_along = min(t * self.speed, self._total_distance)

        for start, end, distance in self._segments:
            if distance_along <= distance:
                ratio = distance_along / distance
                x = start[0] + (end[0] - start[0]) * ratio
                y = start[1] + (end[1] - start[1]) * ratio
                z = start[2] + (end[2] - start[2]) * ratio
                vx = (end[0] - start[0]) / distance * self.speed
                vy = (end[1] - start[1]) / distance * self.speed
                yaw = self.default_yaw
                if self.yaw_mode == "tangent":
                    yaw = math.atan2(end[1] - start[1], end[0] - start[0])
                return (x, y, z), _wrap_pi(yaw), (vx, vy)
            distance_along -= distance

        end = self._segments[-1][1]
        return end, self.default_yaw, (0.0, 0.0)


class GazeboDynamicObstaclesNode(Node):
    """Drive Gazebo model poses from editable YAML trajectories."""

    def __init__(self) -> None:
        super().__init__("gazebo_dynamic_obstacles", automatically_declare_parameters_from_overrides=True)

        if not self.has_parameter("config_file"):
            self.declare_parameter("config_file", "neupan_pedestrians.yaml")

        config_file = self.get_parameter("config_file").get_parameter_value().string_value
        config_path = self._resolve_config_path(config_file)
        cfg = self._load_yaml(config_path)
        root = cfg.get("dynamic_obstacles", cfg)
        if not isinstance(root, dict):
            raise ValueError(f"Invalid dynamic obstacle config: {config_path}")

        self.reference_frame = str(root.get("reference_frame", "world"))
        self.service_names = [str(name) for name in root.get("service_names", ["/gazebo/set_entity_state", "/set_entity_state"])]
        self.update_rate = float(root.get("update_rate", 30.0))
        if self.update_rate <= 0.0:
            raise ValueError(f"Invalid dynamic obstacle update_rate: {self.update_rate}")
        self.start_paused = bool(root.get("start_paused", False))
        self.start_topic = str(root.get("start_topic", "/start_dynamic_obstacles"))
        self.reset_on_start = bool(root.get("reset_on_start", True))

        default_z = float(root.get("default_z", 0.0))
        default_yaw = float(root.get("default_yaw", 0.0))
        obstacles_cfg = root.get("obstacles", [])
        if not isinstance(obstacles_cfg, list) or not obstacles_cfg:
            raise ValueError(f"No dynamic obstacles configured in {config_path}")

        self.trajectories = self._parse_obstacles(obstacles_cfg, default_z, default_yaw)
        if not self.service_names:
            self.service_names = ["/gazebo/set_entity_state", "/set_entity_state"]

        self._set_state_clients = {
            service_name: self.create_client(SetEntityState, service_name)
            for service_name in self.service_names
        }
        self.active_service_name: Optional[str] = None
        self.pending: Dict[str, Any] = {}
        self.start_time_ns: Optional[int] = None
        self.started = not self.start_paused
        self.start_requested = not self.start_paused
        self._reset_after_start = False
        self._last_service_warn_s = 0.0
        self._last_failure_log_s = 0.0
        if self.start_paused:
            self.create_subscription(Empty, self.start_topic, self._start_cb, 10)

        self.timer = self.create_timer(1.0 / self.update_rate, self._on_timer)
        self.get_logger().info(
            f"Loaded {len(self.trajectories)} dynamic obstacle trajectories from {config_path}; "
            f"trying services: {', '.join(self.service_names)}"
        )
        if self.start_paused:
            self.get_logger().info(
                f"Dynamic obstacles are paused; waiting for {self.start_topic}"
            )

    def _resolve_config_path(self, config_file: str) -> str:
        if os.path.isabs(config_file):
            return config_file
        pkg_dir = get_package_share_directory("exact_mppi_jax")
        return os.path.join(pkg_dir, "config", "dynamic_obstacles", config_file)

    def _load_yaml(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dynamic obstacle config not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _parse_obstacles(self, obstacles_cfg: List[dict], default_z: float, default_yaw: float) -> List[ObstacleTrajectory]:
        trajectories: List[ObstacleTrajectory] = []
        for item in obstacles_cfg:
            if not isinstance(item, dict):
                raise ValueError(f"Obstacle entry must be a map: {item}")
            name = str(item["name"])
            kind = str(item.get("type", "waypoints")).lower()
            if kind == "linear":
                points = [
                    _as_point(item["start"], default_z),
                    _as_point(item["end"], default_z),
                ]
            elif kind in {"waypoint", "waypoints", "scripted"}:
                points = [_as_point(point, default_z) for point in item.get("waypoints", [])]
            else:
                raise ValueError(f"Obstacle '{name}' has unsupported type: {kind}")

            trajectories.append(
                ObstacleTrajectory(
                    name=name,
                    points=points,
                    speed=float(item.get("speed", 0.2)),
                    loop=item.get("loop", "ping_pong"),
                    yaw_mode=str(item.get("yaw_mode", "tangent")),
                    start_delay=float(item.get("start_delay", 0.0)),
                    default_yaw=float(item.get("yaw", default_yaw)),
                )
            )
        return trajectories

    def _start_cb(self, _msg: Empty) -> None:
        self.start_requested = True
        self.started = False
        self.start_time_ns = None
        self._reset_after_start = self.reset_on_start
        self.pending.clear()
        self.get_logger().info("Received dynamic obstacle start trigger")

    def _select_client(self) -> Optional[Any]:
        if self.active_service_name:
            client = self._set_state_clients[self.active_service_name]
            if client.service_is_ready():
                return client
            self.active_service_name = None

        for service_name, client in self._set_state_clients.items():
            if client.service_is_ready():
                self.active_service_name = service_name
                self.get_logger().info(f"Using Gazebo SetEntityState service: {service_name}")
                return client

        now_s = time.monotonic()
        if now_s - self._last_service_warn_s > 2.0:
            self.get_logger().warn(
                f"Waiting for Gazebo SetEntityState service ({', '.join(self.service_names)})"
            )
            self._last_service_warn_s = now_s
        return None

    def _on_timer(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        client = self._select_client()
        if client is None:
            return

        if not self.started:
            if not self.start_requested:
                return
            self.start_time_ns = now_ns
            self.started = True
            self.get_logger().info("Starting dynamic obstacles from t=0")

        if self.start_time_ns is None:
            self.start_time_ns = now_ns
        elapsed_s = max(0.0, (now_ns - self.start_time_ns) * 1e-9)
        if self._reset_after_start:
            elapsed_s = 0.0
            self._reset_after_start = False

        for trajectory in self.trajectories:
            pending = self.pending.get(trajectory.name)
            if pending is not None and hasattr(pending, "done") and not pending.done():
                continue

            pose, yaw, velocity = trajectory.sample(elapsed_s)
            request = SetEntityState.Request()
            request.state = self._build_state(trajectory.name, pose, yaw, velocity)
            future = client.call_async(request)
            future.add_done_callback(lambda fut, name=trajectory.name: self._log_result(name, fut))
            self.pending[trajectory.name] = future

    def _build_state(self, name: str, pose_xyz: Point, yaw: float, velocity_xy: Tuple[float, float]) -> EntityState:
        state = EntityState()
        state.name = name
        state.reference_frame = self.reference_frame
        state.pose.position.x = pose_xyz[0]
        state.pose.position.y = pose_xyz[1]
        state.pose.position.z = pose_xyz[2]
        state.pose.orientation = yaw_to_quat(yaw)
        state.twist = Twist()
        state.twist.linear.x = velocity_xy[0]
        state.twist.linear.y = velocity_xy[1]
        state.twist.angular.z = 0.0
        return state

    def _log_result(self, name: str, future) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self._throttled_failure_log(f"SetEntityState failed for '{name}': {exc}")
            return
        if result is not None and hasattr(result, "success") and not result.success:
            message = getattr(result, "status_message", "")
            self._throttled_failure_log(f"SetEntityState rejected '{name}': {message}")

    def _throttled_failure_log(self, message: str) -> None:
        now_s = time.monotonic()
        if now_s - self._last_failure_log_s > 2.0:
            self.get_logger().warn(message)
            self._last_failure_log_s = now_s


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GazeboDynamicObstaclesNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
