#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState, Imu, Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseArray, TransformStamped
from tf2_ros import TransformBroadcaster

class SolutionNode(Node):
    def __init__(self):
        super().__init__('solution_node')

        self.wheel_radius = 0.0225
        self.wheel_separation = 0.159

        self.min_linear_speed = 0.0
        self.max_linear_speed = 0.22
        self.max_angular_speed = 1.2

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 20)
        self.create_subscription(Imu, '/imu', self._on_imu, 20)
        self.create_subscription(Image, '/overhead_camera/image_raw', self._on_image, 10)
        self.create_subscription(CameraInfo, '/overhead_camera/camera_info', self._on_camera_info, 10)

        qos_latched = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(PoseArray, '/waypoints', self._on_waypoints, qos_latched)
        self.create_subscription(PoseArray, '/dynamic_waypoints', self._on_dynamic_waypoints, qos_latched)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_left_pos = None
        self.last_right_pos = None

        self.unvisited_waypoints = []
        self.completed_waypoints = set()
        self.current_target = None
        self.static_received = False
        self.returning_to_base = False
        self.mission_complete = False

        self.wp_capture_radius = 0.10
        self.dwell_start_sim_time = None
        self.dwell_duration_sim = 2.80  # Generous margin over DWELL_TIME_REQ (2.0s)

        self.create_timer(0.05, self._control_loop)
        self.get_logger().info('Solution node active with sim-time dwell tracking.')

    def _get_sim_time(self) -> float:
        t = self.get_clock().now()
        return t.nanoseconds * 1e-9

    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _on_waypoints(self, msg: PoseArray) -> None:
        for p in msg.poses:
            wp = (round(p.position.x, 3), round(p.position.y, 3))
            if wp not in self.completed_waypoints and wp not in self.unvisited_waypoints:
                if self.current_target is None or wp != self.current_target:
                    self.unvisited_waypoints.append(wp)
        self.static_received = True
        self.get_logger().info(f'Static targets registered: {len(self.unvisited_waypoints)}')
        if self.current_target is None:
            self._select_next_nearest_waypoint()

    def _on_dynamic_waypoints(self, msg: PoseArray) -> None:
        added = 0
        for p in msg.poses:
            wp = (round(p.position.x, 3), round(p.position.y, 3))
            if wp not in self.completed_waypoints and wp not in self.unvisited_waypoints:
                if self.current_target is None or wp != self.current_target:
                    self.unvisited_waypoints.append(wp)
                    added += 1
        if added > 0:
            self.get_logger().info(f'Dynamic Waypoints Inserted: {added}. Total Queue: {len(self.unvisited_waypoints)}')
            if self.returning_to_base:
                self.returning_to_base = False
                self._select_next_nearest_waypoint()

    def _select_next_nearest_waypoint(self) -> None:
        if self.unvisited_waypoints:
            best_idx = 0
            min_dist = float('inf')
            for i, (wx, wy) in enumerate(self.unvisited_waypoints):
                dist = math.hypot(wx - self.x, wy - self.y)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            self.current_target = self.unvisited_waypoints.pop(best_idx)
            self.get_logger().info(f'Target Selected: {self.current_target} (Distance: {min_dist:.2f} m)')
        elif self.static_received and not self.returning_to_base and not self.mission_complete:
            self.returning_to_base = True
            self.current_target = (0.0, 0.0)
            self.get_logger().info('All targets cleared! Returning to Start Base (0.0, 0.0)...')
        else:
            self.current_target = None
            self.mission_complete = True
            self.get_logger().info('Mission Finished! Home base reached and secured.')

    def _on_imu(self, msg: Imu) -> None:
        pass

    def _on_image(self, msg: Image) -> None:
        pass

    def _on_camera_info(self, msg: CameraInfo) -> None:
        pass

    def _on_joint_states(self, msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        left_pos = name_to_pos.get('left_wheel_joint', None)
        right_pos = name_to_pos.get('right_wheel_joint', None)

        if left_pos is None or right_pos is None:
            return

        if self.last_left_pos is not None and self.last_right_pos is not None:
            d_left = (left_pos - self.last_left_pos) * self.wheel_radius
            d_right = (right_pos - self.last_right_pos) * self.wheel_radius

            ds = (d_right + d_left) / 2.0
            dtheta = (d_right - d_left) / self.wheel_separation

            self.x += ds * math.cos(self.yaw + dtheta / 2.0)
            self.y += ds * math.sin(self.yaw + dtheta / 2.0)
            self.yaw = self._normalize_angle(self.yaw + dtheta)

            now_stamp = self.get_clock().now().to_msg()

            odom = Odometry()
            odom.header.stamp = now_stamp
            odom.header.frame_id = 'odom'
            odom.child_frame_id = 'base_link'
            odom.pose.pose.position.x = self.x
            odom.pose.pose.position.y = self.y
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
            odom.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
            self.odom_pub.publish(odom)

            t = TransformStamped()
            t.header.stamp = now_stamp
            t.header.frame_id = 'odom'
            t.child_frame_id = 'base_link'
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0
            t.transform.rotation.z = math.sin(self.yaw / 2.0)
            t.transform.rotation.w = math.cos(self.yaw / 2.0)
            self.tf_broadcaster.sendTransform(t)

        self.last_left_pos = left_pos
        self.last_right_pos = right_pos

    def _control_loop(self) -> None:
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        if self.current_target is None or self.mission_complete:
            self.cmd_pub.publish(cmd)
            return

        target_x, target_y = self.current_target
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.hypot(dx, dy)
        current_sim_time = self._get_sim_time()

        # Dwell execution (Strict zero velocity in sim time)
        if self.dwell_start_sim_time is not None:
            if (current_sim_time - self.dwell_start_sim_time) < self.dwell_duration_sim:
                self.cmd_pub.publish(cmd)
                return
            else:
                self.get_logger().info(f'Dwell complete for {self.current_target}.')
                self.dwell_start_sim_time = None
                if not self.returning_to_base:
                    self.completed_waypoints.add(self.current_target)
                self._select_next_nearest_waypoint()
                return

        # Arrival trigger within waypoint tolerance
        if distance < self.wp_capture_radius:
            self.get_logger().info(f'Arrived at {self.current_target}. Starting sim-time dwell...')
            self.dwell_start_sim_time = current_sim_time
            self.cmd_pub.publish(cmd)
            return

        target_angle = math.atan2(dy, dx)
        angle_err = self._normalize_angle(target_angle - self.yaw)

        # Proportional Steering Controller
        if abs(angle_err) > 0.30:
            cmd.linear.x = 0.0
            cmd.angular.z = max(min(1.4 * angle_err, self.max_angular_speed), -self.max_angular_speed)
        else:
            speed = min(0.18, 0.40 * distance)
            cmd.linear.x = max(0.05, speed)
            cmd.angular.z = max(min(1.0 * angle_err, self.max_angular_speed), -self.max_angular_speed)

        self.cmd_pub.publish(cmd)

def main():
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

if __name__ == '__main__':
    main()
