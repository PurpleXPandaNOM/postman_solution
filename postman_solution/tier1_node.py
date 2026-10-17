#!/usr/bin/env python3
"""Tier 1 dead-reckoning waypoint navigation for the Sim2Real hackathon."""

import itertools
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseArray, TransformStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from tf2_ros import TransformBroadcaster


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_to_quaternion(yaw: float):
    half_yaw = yaw * 0.5
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


class SolutionNode(Node):
    def __init__(self):
        super().__init__("solution_node")

        # Robot dimensions and actuator limits from the competition setup.
        self.wheel_radius = 0.0225
        self.wheel_separation = 0.159
        self.max_linear_speed = 0.30
        self.max_angular_speed = 1.50

        # The robot starts at the origin, facing along +x in the odom frame.
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0
        self.imu_yaw_rate = 0.0
        self.last_joint_time = None

        self.waypoints = []
        self.visited = set()
        self.route = []
        self.target_dwell_start = None
        self.home_dwell_start = None

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            JointState, "/joint_states", self._on_joint_states, 20
        )
        self.create_subscription(Imu, "/imu", self._on_imu, 20)

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            PoseArray, "/waypoints", self._on_waypoints, latched_qos
        )
        self.create_subscription(
            PoseArray,
            "/dynamic_waypoints",
            self._on_dynamic_waypoints,
            latched_qos,
        )

        self.create_timer(0.05, self._control_loop)
        self.get_logger().info("Tier 1 solution node initialized.")

    def _on_imu(self, msg: Imu) -> None:
        self.imu_yaw_rate = float(msg.angular_velocity.z)

    def _on_joint_states(self, msg: JointState) -> None:
        try:
            left_index = msg.name.index("left_wheel_joint")
            right_index = msg.name.index("right_wheel_joint")
            self.left_wheel_speed = float(msg.velocity[left_index])
            self.right_wheel_speed = float(msg.velocity[right_index])
        except (ValueError, IndexError):
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_joint_time is None:
            self.last_joint_time = now
            self._publish_odometry()
            return

        dt = now - self.last_joint_time
        self.last_joint_time = now

        # Ignore invalid intervals, for example immediately after a sim pause.
        if dt <= 0.0 or dt > 0.10:
            self._publish_odometry()
            return
        dt = min(dt, 0.05)

        left_linear = self.left_wheel_speed * self.wheel_radius
        right_linear = self.right_wheel_speed * self.wheel_radius
        self.linear_speed = 0.5 * (left_linear + right_linear)

        wheel_yaw_rate = (right_linear - left_linear) / self.wheel_separation
        # Fuse the low-noise wheel estimate with the IMU gyroscope.
        self.angular_speed = 0.80 * wheel_yaw_rate + 0.20 * self.imu_yaw_rate

        delta_yaw = self.angular_speed * dt
        midpoint_yaw = self.yaw + 0.5 * delta_yaw
        distance = self.linear_speed * dt
        self.x += distance * math.cos(midpoint_yaw)
        self.y += distance * math.sin(midpoint_yaw)
        self.yaw = normalize_angle(self.yaw + delta_yaw)
        self._publish_odometry()

    def _publish_odometry(self) -> None:
        stamp = self.get_clock().now().to_msg()
        qx, qy, qz, qw = yaw_to_quaternion(self.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = self.linear_speed
        odom.twist.twist.angular.z = self.angular_speed
        self.odom_pub.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = self.x
        transform.transform.translation.y = self.y
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)

    def _on_waypoints(self, msg: PoseArray) -> None:
        self._add_waypoints(msg)

    def _on_dynamic_waypoints(self, msg: PoseArray) -> None:
        self._add_waypoints(msg)

    def _add_waypoints(self, msg: PoseArray) -> None:
        added = False
        for pose in msg.poses:
            point = (float(pose.position.x), float(pose.position.y))
            # Static, dynamic, and combined lists can contain duplicates.
            if any(
                math.hypot(point[0] - old[0], point[1] - old[1]) < 0.03
                for old in self.waypoints
            ):
                continue
            self.waypoints.append(point)
            added = True

        if added:
            self.home_dwell_start = None
            self.target_dwell_start = None
            self._plan_route()
            self.get_logger().info(
                f"Registered {len(self.waypoints)} waypoint(s); "
                f"{len(self.route)} remain."
            )

    def _plan_route(self) -> None:
        remaining = [p for p in self.waypoints if p not in self.visited]
        if not remaining:
            self.route = []
            return

        # At most seven targets: exhaustive ordering gives the shortest route,
        # including the return trip to the home point (0, 0).
        start = (self.x, self.y)
        best_route = None
        best_distance = float("inf")
        for order in itertools.permutations(remaining):
            distance = 0.0
            previous = start
            for point in order:
                distance += math.hypot(
                    point[0] - previous[0], point[1] - previous[1]
                )
                previous = point
            distance += math.hypot(previous[0], previous[1])
            if distance < best_distance:
                best_distance = distance
                best_route = list(order)
        self.route = best_route or []

    def _publish_command(self, linear: float, angular: float) -> None:
        command = Twist()
        command.linear.x = clamp(linear, 0.0, self.max_linear_speed)
        command.angular.z = clamp(
            angular, -self.max_angular_speed, self.max_angular_speed
        )
        self.cmd_pub.publish(command)

    def _drive_to(self, target_x: float, target_y: float, speed_limit: float):
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.hypot(dx, dy)
        heading_error = normalize_angle(math.atan2(dy, dx) - self.yaw)

        # Reverse is prohibited, so turn in place when the target is behind.
        if abs(heading_error) > 0.35:
            self._publish_command(
                0.0,
                clamp(2.0 * heading_error, -self.max_angular_speed,
                      self.max_angular_speed),
            )
            return

        linear = clamp(0.8 * distance, 0.05, speed_limit)
        if abs(heading_error) > 0.15:
            linear *= 0.65
        angular = clamp(1.8 * heading_error, -1.0, 1.0)
        self._publish_command(linear, angular)

    def _control_loop(self) -> None:
        self._publish_odometry()

        if self.route:
            target = self.route[0]
            distance = math.hypot(target[0] - self.x, target[1] - self.y)

            # Stop well inside the evaluator's 0.20 m capture radius.
            if distance <= 0.12:
                self._publish_command(0.0, 0.0)
                if abs(self.linear_speed) <= 0.025:
                    now = self.get_clock().now().nanoseconds * 1e-9
                    if self.target_dwell_start is None:
                        self.target_dwell_start = now
                    elif now - self.target_dwell_start >= 2.30:
                        self.visited.add(target)
                        self.target_dwell_start = None
                        self.get_logger().info(
                            f"Waypoint collected at ({target[0]:.2f}, "
                            f"{target[1]:.2f}); "
                            f"{len(self.waypoints) - len(self.visited)} remain."
                        )
                        self._plan_route()
                else:
                    self.target_dwell_start = None
                return

            self.target_dwell_start = None
            self.home_dwell_start = None
            self._drive_to(target[0], target[1], 0.22)
            return

        # All known targets are collected; return to the start zone and stop.
        home_distance = math.hypot(self.x, self.y)
        if home_distance <= 0.12:
            self._publish_command(0.0, 0.0)
            if abs(self.linear_speed) <= 0.025:
                now = self.get_clock().now().nanoseconds * 1e-9
                if self.home_dwell_start is None:
                    self.home_dwell_start = now
                elif now - self.home_dwell_start >= 2.30:
                    self.get_logger().info("Home dwell complete.")
            else:
                self.home_dwell_start = None
            return

        self.home_dwell_start = None
        self._drive_to(0.0, 0.0, 0.18)


def main() -> None:
    rclpy.init()
    node = SolutionNode()
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
