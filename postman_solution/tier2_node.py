"""Standalone Tier 2 solution for the supplied Sim2Real arena."""
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
    r, g, b = (rgb[:, :, i].astype(np.int16) for i in range(3))
    red = (r > 45) & (r > 1.5 * g) & (r > 1.5 * b)
    green = (g > 45) & (g > 1.45 * r) & (g > 1.25 * b)
    red_columns = red.sum(axis=0)
    green_rows = green.sum(axis=1)
    u0 = int(np.argmax(red_columns))
    v0 = int(np.argmax(green_rows))
    h, w = rgb.shape[:2]
    if red_columns[u0] < 0.20 * h or green_rows[v0] < 0.20 * w:
        return None
    return u0, v0, int(red_columns[u0]), int(green_rows[v0])


def detect_targets(rgb, intrinsics, camera, pixel_zeroes):
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
        x_sign = 1.0 if v0 < cy else -1.0
        y_sign = 1.0 if u0 < cx else -1.0
        x = x_sign * (v - v0) * height
