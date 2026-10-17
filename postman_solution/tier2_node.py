#!/usr/bin/env python3
"""Standalone Tier 2 solution for the supplied Sim2Real arena.

Uses only public camera, CameraInfo, IMU and noisy wheel-state topics.
No waypoint-coordinate, evaluator, internal Gazebo, or ground-truth inputs.
Dependencies: rclpy, numpy, sensor_msgs, geometry_msgs, nav_msgs, tf2_ros.
Run through the existing tier2_node entry point, or directly:
  python3 tier2_node.py --ros-args -p use_sim_time:=true
Start at the initial robot pose; restart this node after resetting the world.

Vision uses NumPy green segmentation and connected components, so cv_bridge
and OpenCV are not required. Camera geometry matches the supplied arena.sdf.
"""
import itertools
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, Imu, Image, CameraInfo
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def green_components(rgb):
    """Return (area, mean_u, mean_v, width, height) for green components.

    Row runs plus union-find give 4-connected components without an OpenCV
    dependency. Pixel coordinates are column u and row v.
    """
    r, g, b = (rgb[:, :, i].astype(np.float32) for i in range(3))
    mask = (g > 45) & (g > 1.45 * r) & (g > 1.25 * b)
    parents, runs = [], []

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    previous = []
    for y, row in enumerate(mask):
        edges = np.diff(np.concatenate(([0], row, [0])).astype(np.int8))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        current = []
        j = 0
        for a, end in zip(starts, ends):
            a, z = int(a), int(end) - 1
            idx = len(parents)
            parents.append(idx)
            runs.append((a, z, y))
            while j < len(previous) and previous[j][1] < a:
                j += 1
            k = j
            while k < len(previous) and previous[k][0] <= z:
                other = root(previous[k][2])
                here = root(idx)
                if here != other:
                    parents[here] = other
                k += 1
            current.append((a, z, idx))
        previous = current

    groups = {}
    for idx, (a, z, y) in enumerate(runs):
        key = root(idx)
        n = z - a + 1
        if key not in groups:
            groups[key] = [0, 0.0, 0.0, a, z, y, y]
        s = groups[key]
        s[0] += n
        s[1] += n * (a + z) / 2.0
        s[2] += n * y
        s[3], s[4] = min(s[3], a), max(s[4], z)
        s[5], s[6] = min(s[5], y), max(s[6], y)
    return [(s[0], s[1] / s[0], s[2] / s[0], s[4] - s[3] + 1,
             s[6] - s[5] + 1) for s in groups.values()]


def detect_axis_zeroes(rgb):
    """Find the floor axes' zero-coordinate pixels from their colors.

    The supplied world draws a red world-X axis at world y=0 and a thin green
    world-Y axis at world x=0. These visible public image features determine
    image-axis direction and intercept; this avoids guessing camera image
    orientation from RViz/ROS optical-frame conventions.
    """
    r, g, b = (rgb[:, :, i].astype(np.int16) for i in range(3))
    red = (r > 45) & (r > 1.5 * g) & (r > 1.5 * b)
    green = (g > 45) & (g > 1.45 * r) & (g > 1.25 * b)
    # The red x-axis runs vertically in this overhead image: its strongest
    # column is the pixel intercept for world y=0. Likewise the green y-axis
    # runs horizontally: its strongest row is the intercept for world x=0.
    red_columns = red.sum(axis=0)
    green_rows = green.sum(axis=1)
    u0 = int(np.argmax(red_columns))
    v0 = int(np.argmax(green_rows))
    h, w = rgb.shape[:2]
    if red_columns[u0] < 0.20 * h or green_rows[v0] < 0.20 * w:
        return None
    return u0, v0, int(red_columns[u0]), int(green_rows[v0])


def detect_targets(rgb, intrinsics, camera, pixel_zeroes):
    """Segment disks, reject floor-axis lines, and project centers to meters.

    Gazebo camera looks along local +X. At pitch +pi/2 and yaw zero, image
    right points world -Y and image down points world -X. Therefore the
    pixel axes must be swapped as well as negated. Disk top is at z=0.02 m.
    Camera pose is an arena invariant, not a hardcoded target position.
    """
    fx, fy, cx, cy = intrinsics
    camera_x, camera_y, camera_z = camera
    u0, v0 = pixel_zeroes
    height = camera_z - 0.02
    rx, ry = fx * 0.15 / height, fy * 0.15 / height
    expected_area = math.pi * rx * ry
    points = []
    for area, u, v, width, tall in green_components(rgb):
        if not 0.65 * expected_area <= area <= 1.40 * expected_area:
            continue
        if not 0.72 <= (width / tall) / (fx / fy) <= 1.38:
            continue
        if not (1.5 * rx <= width <= 2.5 * rx and
                1.5 * ry <= tall <= 2.5 * ry):
            continue
        if area / (width * tall) < 0.55:
            continue
        # Calibrate signs and pixel origins against the visible arena axes.
        # The principal point selects the direction of increasing world coords;
        # the detected axis lines anchor world x=0 and world y=0.
        x_sign = 1.0 if v0 < cy else -1.0
        y_sign = 1.0 if u0 < cx else -1.0
        x = x_sign * (v - v0) * height / fy
        y = y_sign * (u - u0) * height / fx
        if 0.20 <= x <= 2.65 and 0.20 <= y <= 2.65:
            points.append((float(x), float(y)))
    return points


def shortest_route(start, points, pending):
    """Exact shortest center-to-center ordering, including the return home."""
    def length(order):
        total, previous = 0.0, start
        for idx in order:
            point = points[idx]
            total += math.dist(previous, point)
            previous = point
        return total + math.hypot(*previous)
    return list(min(itertools.permutations(pending), key=length)) if pending else []


class SolutionNode(Node):
    def __init__(self):
        super().__init__('solution_node')
        self.declare_parameter('camera_x', 1.5)
        self.declare_parameter('camera_y', 1.5)
        self.declare_parameter('camera_z', 4.0)
        self.camera = tuple(float(self.get_parameter('camera_' + axis).value)
                            for axis in ('x', 'y', 'z'))
        self.x = self.y = self.yaw = 0.0
        self.v = self.w = self.gyro = 0.0
        self.last_joint_stamp = self.last_joint_received = None
        self.last_imu_received = None
        self.intrinsics = None
        self.info_size = None
        self.pixel_zeroes = None
        self.last_image_processed = -math.inf
        self.last_image_received = None
        self.last_vision_error = -math.inf
        self.targets, self.candidates, self.route = [], [], []
        self.visited = set()
        self.dwell_start = None
        self.dwelling = False
        self.done = False
        self.last_wait_log = -math.inf
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf = TransformBroadcaster(self)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, '/joint_states', self.on_joints, qos)
        self.create_subscription(Imu, '/imu', self.on_imu, qos)
        self.create_subscription(CameraInfo, '/overhead_camera/camera_info',
                                 self.on_camera_info, image_qos)
        self.create_subscription(Image, '/overhead_camera/image_raw',
                                 self.on_image, image_qos)
        self.create_timer(0.05, self.control)
        self.get_logger().info('Tier 2: waiting for sensors and five visible targets.')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_imu(self, msg):
        if math.isfinite(msg.angular_velocity.z):
            self.gyro = float(msg.angular_velocity.z)
            self.last_imu_received = self.now()

    def on_joints(self, msg):
        try:
            left = float(msg.velocity[msg.name.index('left_wheel_joint')]) * 0.0225
            right = float(msg.velocity[msg.name.index('right_wheel_joint')]) * 0.0225
        except (ValueError, IndexError):
            return
        if not all(map(math.isfinite, (left, right))):
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        received = self.now()
        if t == 0.0:
            t = received
        self.last_joint_received = received
        self.v = (left + right) / 2.0
        wheel_w = (right - left) / 0.159
        imu_fresh = (self.last_imu_received is not None and
                     0.0 <= received - self.last_imu_received < 0.20)
        self.w = 0.8 * wheel_w + 0.2 * self.gyro if imu_fresh else wheel_w
        # Suppress gyro noise integration when both wheels are stationary.
        if max(abs(left), abs(right)) < 0.0001:
            self.w = 0.0
        if self.last_joint_stamp is not None:
            dt = t - self.last_joint_stamp
            if 0.0 < dt <= 0.25:
                dyaw = self.w * dt
                self.x += self.v * dt * math.cos(self.yaw + 0.5 * dyaw)
                self.y += self.v * dt * math.sin(self.yaw + 0.5 * dyaw)
                self.yaw = wrap(self.yaw + dyaw)
        self.last_joint_stamp = t

    def on_camera_info(self, msg):
        k = msg.k
        if (msg.width > 0 and msg.height > 0 and k[0] > 0 and k[4] > 0
                and all(math.isfinite(float(n)) for n in k)):
            self.intrinsics = (float(k[0]), float(k[4]), float(k[2]), float(k[5]))
            self.info_size = (msg.width, msg.height)

    def on_image(self, msg):
        now = self.now()
        if self.intrinsics is None or now - self.last_image_processed < 0.20:
            return
        self.last_image_processed = now
        try:
            # Respect encoding and row padding in ROS Image; no cv_bridge needed.
            channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}[msg.encoding]
            if msg.width <= 0 or msg.height <= 0 or msg.step < msg.width * channels:
                raise ValueError('Invalid image dimensions or step')
            rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            rgb = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
            rgb = rgb[:, :, :3]
            if msg.encoding.startswith('bgr'):
                rgb = rgb[:, :, ::-1]
            sx, sy = msg.width / self.info_size[0], msg.height / self.info_size[1]
            fx, fy, cx, cy = self.intrinsics
            # Recheck calibration continuously; keep the prior valid result if
            # the robot briefly occludes an axis.
            axes = detect_axis_zeroes(rgb)
            if axes is not None:
                self.pixel_zeroes = (axes[0] * sx, axes[1] * sy)
            if self.pixel_zeroes is None:
                raise ValueError('Could not calibrate camera from red/green floor axes')
            points = detect_targets(
                rgb, (fx * sx, fy * sy, cx * sx, cy * sy),
                self.camera, self.pixel_zeroes)
        except (KeyError, ValueError, TypeError) as exc:
            if now - self.last_vision_error > 5.0:
                self.get_logger().warning(f'Camera frame rejected: {exc}')
                self.last_vision_error = now
            return
        self.last_image_received = now
        self.register_detections(points, now)

    def register_detections(self, points, now):
        # Keep confirmed targets even when covered by the robot or already visited.
        self.candidates = [c for c in self.candidates if now - c['last'] < 1.0]
        used = set()
        added = False
        for point in points:
            if any(math.dist(point, old) < 0.20 for old in self.targets):
                continue
            match = next((i for i, c in enumerate(self.candidates)
                          if i not in used and math.dist(point, c['point']) < 0.12), None)
            if match is None:
                self.candidates.append({'point': point, 'hits': 1, 'last': now})
                used.add(len(self.candidates) - 1)
                continue
            used.add(match)
            c = self.candidates[match]
            n = c['hits']
            c['point'] = tuple((n * a + b) / (n + 1) for a, b in zip(c['point'], point))
            c['hits'] += 1
            c['last'] = now
            if c['hits'] >= 3 and len(self.targets) < 7:
                self.targets.append(c['point'])
                added = True
                self.get_logger().info(
                    f'Vision target {len(self.targets)}/7: '
                    f'({c["point"][0]:.3f}, {c["point"][1]:.3f})')
        if added:
            self.replan()

    def replan(self):
        pending = [i for i in range(len(self.targets)) if i not in self.visited]
        # Finish an ongoing dwell instead of restarting it when a new disk appears.
        fixed = self.route[0] if self.dwelling and self.route else None
        if fixed in pending:
            pending.remove(fixed)
            self.route = [fixed] + shortest_route(self.targets[fixed], self.targets, pending)
        else:
            self.route = shortest_route((self.x, self.y), self.targets, pending)
            self.dwelling = False
            self.dwell_start = None

    def command(self, v=0.0, w=0.0):
        msg = Twist()
        msg.linear.x = clamp(v, 0.0, 0.30)
        msg.angular.z = clamp(w, -1.50, 1.50)
        self.cmd_pub.publish(msg)

    def publish_odometry(self):
        stamp = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.x, self.y
        odom.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = self.v, self.w
        self.odom_pub.publish(odom)
        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x, tf.transform.translation.y = self.x, self.y
        tf.transform.rotation = odom.pose.pose.orientation
        self.tf.sendTransform(tf)

    def control(self):
        self.publish_odometry()
        now = self.now()
        healthy = (self.last_joint_received is not None and
                   0.0 <= now - self.last_joint_received < 0.30 and
                   self.last_imu_received is not None and
                   0.0 <= now - self.last_imu_received < 0.30 and
                   self.last_image_received is not None and
                   0.0 <= now - self.last_image_received < 2.0)
        if self.done or not healthy or len(self.targets) < 5:
            self.command()
            self.dwell_start = None
            return
        home = not self.route
        if home and len(self.visited) < 7:
            # Returning home prematurely could end evaluation before all 7 count.
            self.command()
            if now - self.last_wait_log > 5.0:
                self.get_logger().info('Waiting for remaining visual targets; not returning home yet.')
                self.last_wait_log = now
            return
        point = (0.0, 0.0) if home else self.targets[self.route[0]]
        distance = math.dist((self.x, self.y), point)
        if distance <= 0.10:
            self.command()
            self.dwelling = True
            if abs(self.v) <= 0.020 and abs(self.w) <= 0.08:
                if self.dwell_start is None:
                    self.dwell_start = now
                elif now - self.dwell_start >= 2.40:
                    if home:
                        self.done = True
                        self.get_logger().info('All seven targets collected; home dwell complete.')
                    else:
                        self.visited.add(self.route[0])
                        self.get_logger().info(f'Collected {len(self.visited)}/7 targets.')
                        self.dwelling = False
                        self.dwell_start = None
                        self.replan()
            else:
                self.dwell_start = None
            return
        self.dwelling = False
        self.dwell_start = None
        error = wrap(math.atan2(point[1] - self.y, point[0] - self.x) - self.yaw)
        if abs(error) > 0.35:
            self.command(0.0, clamp(2.0 * error, -1.5, 1.5))
        else:
            v = clamp(0.65 * distance, 0.04, 0.18 if home else 0.22)
            if abs(error) > 0.15:
                v *= 0.65
            self.command(v, clamp(1.8 * error, -1.0, 1.0))


def main(args=None):
    rclpy.init(args=args)
    node = SolutionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.command()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
