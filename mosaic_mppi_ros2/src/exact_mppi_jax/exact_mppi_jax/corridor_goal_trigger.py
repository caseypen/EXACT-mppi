#!/usr/bin/env python3
"""Publish a fixed corridor goal once after launch startup."""

import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class CorridorGoalTrigger(Node):
    def __init__(self) -> None:
        super().__init__("corridor_goal_trigger")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("goal_x", 25.0)
        self.declare_parameter("goal_y", 9.5)
        self.declare_parameter("goal_yaw", 0.0)
        self.declare_parameter("initial_delay", 2.0)
        self.declare_parameter("publish_count", 10)
        self.declare_parameter("publish_period", 0.2)

        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.goal_yaw = float(self.get_parameter("goal_yaw").value)
        self.initial_delay = max(0.0, float(self.get_parameter("initial_delay").value))
        self.publish_count = max(1, int(self.get_parameter("publish_count").value))
        self.publish_period = max(0.0, float(self.get_parameter("publish_period").value))

        self.publisher = self.create_publisher(PoseStamped, self.goal_topic, 10)

    def publish_goal_burst(self) -> None:
        self.get_logger().info(
            f"Publishing corridor goal ({self.goal_x:.2f}, {self.goal_y:.2f}, "
            f"{self.goal_yaw:.2f}) on {self.goal_topic} after {self.initial_delay:.1f}s"
        )
        self._sleep_wall(self.initial_delay)

        for index in range(self.publish_count):
            if not rclpy.ok():
                break
            self.publisher.publish(self._goal_msg())
            self.get_logger().info(
                f"Published corridor goal trigger {index + 1}/{self.publish_count}"
            )
            if index + 1 < self.publish_count:
                self._sleep_wall(self.publish_period)

    def _goal_msg(self) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = self.goal_x
        msg.pose.position.y = self.goal_y
        msg.pose.orientation.z = math.sin(self.goal_yaw * 0.5)
        msg.pose.orientation.w = math.cos(self.goal_yaw * 0.5)
        return msg

    def _sleep_wall(self, duration: float) -> None:
        end_time = time.monotonic() + duration
        while rclpy.ok():
            remaining = end_time - time.monotonic()
            if remaining <= 0.0:
                return
            rclpy.spin_once(self, timeout_sec=min(0.05, remaining))


def main() -> None:
    rclpy.init()
    node = CorridorGoalTrigger()
    try:
        node.publish_goal_burst()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
