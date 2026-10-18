#!/usr/bin/env python3
"""Tier 3 """
import itertools



import heapq



import math



import numpy as np



import rclpy



from rclpy.node import Node



from rclpy.executors import ExternalShutdownException



from rclpy.qos import QoSProfile, ReliabilityPolicy



from sensor_msgs.msg import JointState, Imu, Image, CameraInfo



from geometry_msgs.msg import Twist, TransformStamped, PoseStamped



from nav_msgs.msg import Odometry, Path



from tf2_ros import TransformBroadcaster



from visualization_msgs.msg import Marker, MarkerArray



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



def decode_image(msg):
    """Read supported ROS Image encodings, including padded rows."""
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(msg.encoding)
    if channels is None:
        raise ValueError('Expected rgb8, bgr8, rgba8 or bgra8 camera image')
    if msg.width <= 0 or msg.height <= 0 or msg.step < msg.width * channels:
        raise ValueError('Invalid image dimensions or row stride')
    if len(msg.data) != msg.height * msg.step:
        raise ValueError('Image data length does not match height * step')
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    rgb = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    rgb = rgb[:, :, :3]
    return rgb[:, :, ::-1] if msg.encoding.startswith('bgr') else rgb



def pixel_to_world(u, v, intrinsics, camera, plane_z=0.02):
    """Intersect the supplied downward camera ray with a horizontal plane.

    Gazebo camera coordinates are forward +X, left +Y, up +Z. A pixel ray
    in that frame is (1, -(u-cx)/fx, -(v-cy)/fy). Rotating about Y by +pi/2
    and intersecting z=plane_z gives the two expressions below.
    """
    fx, fy, cx, cy = intrinsics
    camera_x, camera_y, camera_z = camera
    height = camera_z - plane_z
    if fx <= 0 or fy <= 0 or height <= 0:
        raise ValueError('Invalid camera calibration')
    return (camera_x - (v - cy) * height / fy,
            camera_y - (u - cx) * height / fx)



def detect_targets(rgb, intrinsics, camera):
    """Segment disks, reject floor-axis lines, and project centers to meters.

    Gazebo camera looks along local +X. At pitch +pi/2 and yaw zero, image
    right points world -Y and image down points world -X. Therefore the
    pixel axes must be swapped as well as negated. Disk top is at z=0.02 m.
    Camera pose is an arena invariant, not a hardcoded target position.
    """
    fx, fy, cx, cy = intrinsics
    camera_x, camera_y, camera_z = camera
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
        x, y = pixel_to_world(u, v, intrinsics, camera)
        if 0.20 <= x <= 2.65 and 0.20 <= y <= 2.65:
            points.append((float(x), float(y)))
    return points



def detect_obstacles(rgb, intrinsics, camera):
    r, g, b = (rgb[:, :, i].astype(np.float32) for i in range(3))
    mask = (r > 55) & (b > 55) & (r > 1.5*g) & (b > 1.5*g)
    vv, uu = np.nonzero(mask)
    remaining = set(zip(vv.tolist(), uu.tolist()))
    obstacles = []
    fx, fy, cx, cy = intrinsics
    height = camera[2] - 0.20
    expected_area = fx * fy * 0.50 * 0.06 / (height * height)
    while remaining:
        seed = remaining.pop()
        stack, pixels = [seed], []
        while stack:
            v, u = stack.pop()
            pixels.append((u, v))
            for q in ((v-1,u), (v+1,u), (v,u-1), (v,u+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        if not 0.50*expected_area <= len(pixels) <= 1.70*expected_area:
            continue
        uv = np.asarray(pixels, dtype=float)
        # Wall tops are 20 cm above the floor: using the floor depth would
        # displace them in the map and reduce the collision clearance.
        xy = np.column_stack((camera[0]-(uv[:,1]-cy)*height/fy,
                              camera[1]-(uv[:,0]-cx)*height/fx))
        mean = xy.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov(xy-mean, rowvar=False))
        along = vectors[:, -1]
        across = np.array([-along[1], along[0]])
        a, b = (xy-mean) @ along, (xy-mean) @ across
        amin, amax, bmin, bmax = a.min(), a.max(), b.min(), b.max()
        length, width = amax-amin, bmax-bmin
        if not (0.36 <= length <= 0.64 and 0.025 <= width <= 0.12):
            continue
        centre = mean + along*(amin+amax)/2 + across*(bmin+bmax)/2
        if not (0.0 < centre[0] < 2.75 and 0.0 < centre[1] < 2.75):
            continue
        # Include raster quantization and the visible cap's narrow side faces.
        pad = max(height/fx, height/fy) + 0.004
        obstacles.append((float(centre[0]), float(centre[1]),
                          math.atan2(along[1], along[0]) % math.pi,
                          float(length/2+pad), float(width/2+pad)))
    return sorted(obstacles, key=lambda q: (q[0], q[1]))



def local_point(p, box):
    dx, dy = p[0]-box[0], p[1]-box[1]
    c, s = math.cos(box[2]), math.sin(box[2])
    return c*dx+s*dy, -s*dx+c*dy



def advance_pose(pose, distance, turn):
    """Integrate a constant-curvature encoder increment without Euler bias."""
    x,y,yaw=pose
    scale=1.0 if abs(turn)<1e-9 else math.sin(turn/2)/(turn/2)
    return (x+distance*scale*math.cos(yaw+turn/2),
            y+distance*scale*math.sin(yaw+turn/2),wrap(yaw+turn))



def intersects_box(a, b, box, padding=0.0):
    """Closed segment/rotated-rectangle intersection by interval clipping."""
    p, q = local_point(a,box), local_point(b,box)
    lo, hi = 0.0, 1.0
    for k, extent in enumerate((box[3]+padding, box[4]+padding)):
        delta = q[k]-p[k]
        if abs(delta) < 1e-12:
            if abs(p[k]) > extent:
                return False
        else:
            t1, t2 = (-extent-p[k])/delta, (extent-p[k])/delta
            lo, hi = max(lo,min(t1,t2)), min(hi,max(t1,t2))
            if lo > hi:
                return False
    return True



class SafePlanner:
    """Visibility graph around camera-detected, expanded wall rectangles."""
    clearance = 0.27
    lower, upper = 0.14, 2.61
    gate = (0.30, 0.30)

    def __init__(self, obstacles):
        self.obstacles = obstacles
        self.cache = {}
        self.vertices = []
        for ox, oy, yaw, hl, hw in obstacles:
            c, s = math.cos(yaw), math.sin(yaw)
            for a in (-hl-self.clearance-0.025, hl+self.clearance+0.025):
                for b in (-hw-self.clearance-0.025, hw+self.clearance+0.025):
                    p = (ox+c*a-s*b, oy+s*a+c*b)
                    if self.free(p):
                        self.vertices.append(p)

    def free(self, p):
        return (all(self.lower <= x <= self.upper for x in p) and
                not any(intersects_box(p,p,b,self.clearance) for b in self.obstacles))

    def visible(self, a, b):
        return (self.free(a) and self.free(b) and
                not any(intersects_box(a,b,r,self.clearance) for r in self.obstacles))

    def path(self, start, goal):
        start, goal = tuple(start), tuple(goal)
        key = (start, goal)
        if key in self.cache:
            return list(self.cache[key])
        if not self.free(start) or not self.free(goal):
            raise ValueError('Start or goal is outside the map clearance; stopping for inspection.')
        if self.visible(start,goal):
            path = [start,goal]
        else:
            nodes = [start,goal] + self.vertices
            edges = [[] for _ in nodes]
            for i in range(len(nodes)):
                for j in range(i):
                    if self.visible(nodes[i],nodes[j]):
                        d = math.dist(nodes[i],nodes[j])
                        edges[i].append((j,d)); edges[j].append((i,d))
            dist, parent, queue = {0:0.0}, {}, [(0.0,0)]
            while queue:
                cost,i = heapq.heappop(queue)
                if cost > dist.get(i,math.inf):
                    continue
                if i == 1:
                    break
                for j,d in edges[i]:
                    if cost+d < dist.get(j,math.inf):
                        dist[j],parent[j] = cost+d,i
                        heapq.heappush(queue,(cost+d,j))
            if 1 not in dist:
                raise ValueError('No safe route through the detected obstacle map.')
            path, index = [goal], 1
            while index != 0:
                index = parent[index]
                path.append(nodes[index])
            path.reverse()
        self.cache[key] = tuple(path)
        self.cache[(goal,start)] = tuple(reversed(path))
        return path

    def distance(self,a,b):
        p = self.path(a,b)
        return sum(math.dist(x,y) for x,y in zip(p,p[1:]))

    def order(self,start,points,pending):
        """Exact ordering for <=7 targets using safe graph path lengths."""
        if not pending:
            return []
        first = {i:self.distance(start,points[i]) for i in pending}
        home = {i:self.distance(points[i],self.gate) for i in pending}
        pair = {(i,j):self.distance(points[i],points[j])
                for i in pending for j in pending if i != j}
        def length(order):
            return first[order[0]] + sum(pair[i,j] for i,j in zip(order,order[1:])) + home[order[-1]]
        return list(min(itertools.permutations(pending),key=length))



def boxes_overlap(a,b):
    """Exact OBB overlap using the separating-axis theorem."""
    au=(math.cos(a[2]),math.sin(a[2])); av=(-au[1],au[0])
    bu=(math.cos(b[2]),math.sin(b[2])); bv=(-bu[1],bu[0])
    def dot(p,q): return p[0]*q[0]+p[1]*q[1]
    d=(b[0]-a[0],b[1]-a[1])
    for axis in (au,av,bu,bv):
        ra=a[3]*abs(dot(au,axis))+a[4]*abs(dot(av,axis))
        rb=b[3]*abs(dot(bu,axis))+b[4]*abs(dot(bv,axis))
        if abs(dot(d,axis)) > ra+rb:
            return False
    return True



def footprint_safe(pose, obstacles, margin=0.010):
    x,y,yaw=pose
    c,s=math.cos(yaw),math.sin(yaw)
    robot=(x+0.095*c,y+0.095*s,yaw,0.105+margin,0.0885+margin)
    for a in (-robot[3],robot[3]):
        for b in (-robot[4],robot[4]):
            wx,wy=robot[0]+c*a-s*b,robot[1]+s*a+c*b
            if min(wx,wy)<-0.125 or max(wx,wy)>2.875:
                return False
    return not any(boxes_overlap(robot,box) for box in obstacles)



def braking_motion_safe(pose,v,w,target_v,target_w,obstacles):
    """Check the full body through 0.10 s reaction + a braking trajectory."""
    x,y,yaw=pose
    if not footprint_safe((x,y,yaw),obstacles):
        return False
    dt=0.025
    for k in range(24):
        cv,cw=(target_v,target_w) if k<4 else (0.0,0.0)
        v+=clamp(cv-v,-dt,dt)
        w+=clamp(cw-w,-4*dt,4*dt)
        x+=v*dt*math.cos(yaw+w*dt/2)
        y+=v*dt*math.sin(yaw+w*dt/2)
        yaw=wrap(yaw+w*dt)
        if not footprint_safe((x,y,yaw),obstacles):
            return False
    return True



def direct_approach_safe(pose, goal, obstacles, max_distance=1.20, margin=.04):
    """Check a short rotate-then-drive approach using the actual body.

    Only used from rest. By default keep 4 cm clearance; departure uses
    2.5 cm because the known spawn is close to the perimeter. Sample every
    centimetre and allow +/-0.10 rad of heading error along the straight leg.
    Longer routes retain the conservative visibility graph, except a checked
    initial departure (maximum 4 m).
    """
    x,y,yaw = pose
    distance = math.dist((x,y),goal)
    if distance > max_distance or distance < .01:
        return False
    heading = math.atan2(goal[1]-y,goal[0]-x)
    turn = wrap(heading-yaw)
    for t in np.linspace(0,1,max(2,int(abs(turn)/.025)+2)):
        if not footprint_safe((x,y,wrap(yaw+t*turn)),obstacles,margin):
            return False
    for t in np.linspace(0,1,max(2,int(distance/.01)+2)):
        px,py = x+t*(goal[0]-x),y+t*(goal[1]-y)
        for error in (-.10,0.,.10):
            if not footprint_safe((px,py,heading+error),obstacles,margin):
                return False
    return True


def select_home_goal(pose, obstacles):
    """Find a side approach with clearance for the asymmetric chassis.

    A diagonal dive toward zero puts the front bumper through the perimeter.
    Stage inside the arena, then enter the official 20 cm home circle slowly.
    All candidates are navigation aids; no target or wall locations are stored.
    """
    x,y,yaw = pose
    candidates = []
    for gx in np.linspace(.02,.17,16):
        for gy in np.linspace(.02,.17,16):
            if math.hypot(gx,gy)>.17:
                continue
            for stage in (None,(.80,.24),(.24,.80),(.50,.18),(.18,.50)):
                start = (x,y) if stage is None else stage
                heading = math.atan2(gy-start[1],gx-start[0])
                if not footprint_safe((gx,gy,heading),obstacles,.025):
                    continue
                path = [(float(gx),float(gy))] if stage is None else [stage,(float(gx),float(gy))]
                # Travel distance plus a small turn penalty chooses less awkward
                # approaches; this is not a proof of a global shortest route.
                cost = math.dist((x,y),start)+math.dist(start,(gx,gy))
                first_heading = math.atan2(path[0][1]-y,path[0][0]-x)
                cost += .08*abs(wrap(first_heading-yaw))
                if stage is not None:
                    cost += .08*abs(wrap(heading-first_heading))
                candidates.append((cost,path))
    for _,path in sorted(candidates):
        px,py,h = pose
        samples = []
        for tx,ty in path:
            heading = math.atan2(ty-py,tx-px)
            samples += [(px,py,wrap(h+t*wrap(heading-h))) for t in np.linspace(0,1,41)]
            samples += [(px+t*(tx-px),py+t*(ty-py),heading) for t in np.linspace(0,1,81)]
            px,py,h = tx,ty,heading
        if all(footprint_safe(p,obstacles,.025) for p in samples):
            return path
    raise ValueError('No checked home approach from the current pose. '
                     f'Position=({x:.3f},{y:.3f},{yaw:.3f}).')


class SolutionNode(Node):
    def __init__(self):
        super().__init__('solution_node')
        self.declare_parameter('camera_x', 1.5)
        self.declare_parameter('camera_y', 1.5)
        self.declare_parameter('camera_z', 4.0)
        self.declare_parameter('camera_horizontal_fov', 1.2217)
        self.camera = tuple(float(self.get_parameter('camera_' + axis).value)
                            for axis in ('x', 'y', 'z'))
        self.x = self.y = self.yaw = 0.0
        self.v = self.w = self.gyro = 0.0
        self.encoder_yaw = 0.0
        self.last_wheel_angles = None
        self.home_path = []
        self.home_stopped = False
        self.last_joint_stamp = self.last_joint_received = None
        self.last_imu_received = None
        self.intrinsics = None
        self.info_size = None
        self.started_at = self.now()
        self.camera_mode = None
        self.fault = None
        self.obstacles = []
        self.obstacle_candidate = None
        self.obstacle_hits = 0
        self.planner = None
        self.phase = 'exit'
        self.departure_goal = None
        self.direct_departure = False
        self.path = []
        self.path_goal = None
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
        self.path_pub = self.create_publisher(Path, '/tier3/path', 1)
        from rclpy.qos import DurabilityPolicy
        marker_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoint_markers', marker_qos)
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
        self.get_logger().info(
            'TIER 3 v11: checked direct departure, adaptive home speed and direct approaches. '
            'Start from a fresh tier:=3 world; obstacle positions come from vision.')


    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9


    def on_imu(self, msg):
        if math.isfinite(msg.angular_velocity.z):
            self.gyro = float(msg.angular_velocity.z)
            self.last_imu_received = self.now()

    def on_joints(self, msg):
        # Use continuous wheel POSITIONS, as in the supplied working controller.
        # Position deltas also integrate correctly across dropped messages.
        try:
            angles = tuple(float(msg.position[msg.name.index(name)])
                           for name in ('left_wheel_joint', 'right_wheel_joint'))
        except (ValueError, IndexError):
            return
        if not all(map(math.isfinite, angles)):
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        received = self.now()
        if t == 0.0:
            t = received
        if self.last_joint_stamp is not None:
            dt = t - self.last_joint_stamp
            if dt < 0:
                self.fail('Simulation clock reset: restart this node with the fresh world.')
                return
            if dt == 0:
                return
            dl, dr = (.0225*(a-b) for a,b in zip(angles,self.last_wheel_angles))
            if max(abs(dl),abs(dr)) > max(.03,dt):
                self.fail('Discontinuous wheel position: restart after resetting the simulation.')
                return
            distance, turn = (dl+dr)/2, (dr-dl)/.159
            self.encoder_yaw = wrap(self.encoder_yaw+turn)
            fresh = (self.last_imu_received is not None and
                     0 <= received-self.last_imu_received < .20)
            # Encoder-anchored complementary filter: gyro smooths short-term
            # turns, but cannot accumulate an independent heading offset.
            predicted = wrap(self.yaw + (.90*turn + .10*self.gyro*dt if fresh else turn))
            gain = 1.0-math.exp(-dt/.10)
            heading = wrap(predicted+gain*wrap(self.encoder_yaw-predicted))
            if max(abs(dl),abs(dr)) < 1e-7:
                heading = self.encoder_yaw
            self.x,self.y,self.yaw = advance_pose(
                (self.x,self.y,self.yaw),distance,wrap(heading-self.yaw))
            self.v = distance/dt
            self.w = .8*turn/dt+.2*self.gyro if fresh else turn/dt
        self.last_joint_stamp = t
        self.last_wheel_angles = angles
        self.last_joint_received = received

    def on_camera_info(self, msg):
        k = msg.k
        if (msg.width > 0 and msg.height > 0 and k[0] > 0 and k[4] > 0
                and all(math.isfinite(float(n)) for n in k)):
            self.intrinsics = (float(k[0]), float(k[4]), float(k[2]), float(k[5]))
            self.info_size = (msg.width, msg.height)


    def on_image(self, msg):
        now = self.now()
        if now - self.last_image_processed < 0.20:
            return
        self.last_image_processed = now
        try:
            rgb = decode_image(msg)
            if self.intrinsics is not None:
                sx, sy = msg.width / self.info_size[0], msg.height / self.info_size[1]
                fx, fy, cx, cy = self.intrinsics
                intrinsics = (fx * sx, fy * sy, cx * sx, cy * sy)
                mode = 'CameraInfo'
            elif now - self.started_at >= 2.0:
                # The provided SDF specifies square pixels and a 70-degree HFOV.
                # Some bridge builds do not expose CameraInfo; use that known
                # calibration rather than waiting forever or guessing axes.
                fov = float(self.get_parameter('camera_horizontal_fov').value)
                if not 0.1 < fov < 3.0:
                    raise ValueError('Invalid camera_horizontal_fov')
                f = msg.width / (2.0 * math.tan(fov / 2.0))
                intrinsics = (f, f, msg.width / 2.0, msg.height / 2.0)
                mode = 'arena.sdf HFOV fallback'
            else:
                return
            if self.camera_mode != mode:
                self.get_logger().info(f'Camera calibration: {mode}; fx,fy,cx,cy={intrinsics}')
                self.camera_mode = mode
            points = detect_targets(rgb, intrinsics, self.camera)
            if self.planner is None:
                self.register_obstacles(detect_obstacles(rgb, intrinsics, self.camera))
        except (KeyError, ValueError, TypeError) as exc:
            if now - self.last_vision_error > 5.0:
                self.get_logger().warning(f'Camera frame rejected: {exc}')
                self.last_vision_error = now
            return
        self.last_image_received = now
        self.register_detections(points, now)


    def fail(self, reason):
        if self.fault is None:
            self.get_logger().error(reason)
        self.fault = reason
        self.command()


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
            self.publish_markers()


    def replan(self):
        self.path, self.path_goal = [], None
        if self.planner is None:
            return
        pending = [i for i in range(len(self.targets)) if i not in self.visited]
        fixed = self.route[0] if self.dwelling and self.route else None
        try:
            if fixed in pending:
                pending.remove(fixed)
                self.route = [fixed] + self.planner.order(self.targets[fixed],self.targets,pending)
            else:
                start = self.planner.gate if self.phase == 'exit' else (self.x,self.y)
                self.route = self.planner.order(start,self.targets,pending)
                self.dwelling, self.dwell_start = False, None
        except ValueError as exc:
            self.fail(str(exc))


    def command(self, v=0.0, w=0.0):
        msg = Twist()
        msg.linear.x = clamp(v, 0.0, 0.30)
        msg.angular.z = clamp(w, -1.50, 1.50)
        self.cmd_pub.publish(msg)


    def publish_markers(self):
        """Visual detections only; no subscription to organizer marker topics."""
        array = MarkerArray()
        for i, (x, y) in enumerate(self.targets):
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id = 'tier3_camera_targets', i
            m.type, m.action = Marker.CYLINDER, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.02
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = 0.30, 0.30, 0.01
            color = (0.1, 0.8, 0.2) if i not in self.visited else (0.1, 0.5, 1.0)
            m.color.r, m.color.g, m.color.b = color
            m.color.a = 0.85
            array.markers.append(m)
        for i, (x,y,yaw,hl,hw) in enumerate(self.obstacles):
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id = 'tier3_visual_walls', i
            m.type, m.action = Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x,y,0.10
            m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw/2),math.cos(yaw/2)
            m.scale.x, m.scale.y, m.scale.z = 2*hl,2*hw,0.20
            m.color.r, m.color.g, m.color.b, m.color.a = 0.9,0.0,0.9,0.9
            array.markers.append(m)
        self.marker_pub.publish(array)


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


    def register_obstacles(self, detected):
        if len(detected) != 4:
            self.obstacle_hits = 0
            self.obstacle_candidate = None
            return
        old = self.obstacle_candidate
        stable = old is not None and all(
            math.dist(a[:2],b[:2]) < 0.025 and
            abs(wrap(2*(a[2]-b[2])))/2 < 0.08 and
            abs(a[3]-b[3]) < 0.03 and abs(a[4]-b[4]) < 0.02
            for a,b in zip(old,detected))
        self.obstacle_hits = self.obstacle_hits+1 if stable else 1
        self.obstacle_candidate = detected
        if self.obstacle_hits >= 5:
            self.obstacles = detected
            self.planner = SafePlanner(detected)
            self.get_logger().info('Visual map confirmed: four walls; enabling safe path planning.')
            for i,b in enumerate(detected):
                self.get_logger().info(f'Wall {i+1}: centre=({b[0]:.3f},{b[1]:.3f}), '
                                       f'heading={b[2]:.3f}, size=({2*b[3]:.3f},{2*b[4]:.3f})')
            self.replan()
            self.publish_markers()


    def publish_path(self):
        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.get_clock().now().to_msg()
        for x,y in [(self.x,self.y)]+self.path:
            p=PoseStamped()
            p.header=msg.header
            p.pose.position.x, p.pose.position.y = float(x),float(y)
            p.pose.orientation.w=1.0
            msg.poses.append(p)
        self.path_pub.publish(msg)


    def settled(self):
        return abs(self.v)<0.015 and abs(self.w)<0.05


    def drive(self, point, limit=0.18):
        d = math.dist((self.x,self.y),point)
        error = wrap(math.atan2(point[1]-self.y,point[0]-self.x)-self.yaw)
        if abs(error) > .15:
            v,w = 0.0,clamp(1.6*error,-.70,.70)
        else:
            v,w = min(limit,1.0*d),clamp(1.5*error,-.45,.45)
        if not braking_motion_safe((self.x,self.y,self.yaw),self.v,self.w,v,w,self.obstacles):
            self.fail('Predicted body collision: motion stopped. Save this log and pose for inspection: '
                      f'({self.x:.3f}, {self.y:.3f}, {self.yaw:.3f}).')
            return
        self.command(v,w)

    def control_home(self, now):
        # Completion is a region, not an exact endpoint that causes orbiting.
        radius = math.hypot(self.x,self.y)
        if radius <= .180 or (self.home_stopped and radius <= .190):
            self.home_stopped = True
            self.command()
            if not self.settled():
                self.dwell_start = None
            elif self.dwell_start is None:
                self.dwell_start = now
                self.get_logger().info(f'Home reached at ({self.x:.3f}, {self.y:.3f}); holding 2.4 s.')
            elif now-self.dwell_start >= 2.4:
                self.done = True
                self.get_logger().info('TIER 3 v11 COMPLETE: 7 estimated visits and home dwell. '
                                       'Check the evaluator for the official result.')
            return
        self.dwell_start = None
        self.home_stopped = False
        if not self.home_path:
            self.command()
            if not self.settled():
                return
            try:
                self.home_path = select_home_goal((self.x,self.y,self.yaw),self.obstacles)
            except ValueError as exc:
                self.fail(str(exc))
                return
            self.get_logger().info(f'Checked home approach: {self.home_path}')
        if len(self.home_path)>1 and math.dist((self.x,self.y),self.home_path[0])<=.015:
            self.command()
            if self.settled():
                self.home_path.pop(0)
            return
        # Cruise on the long home leg; slow only within 30 cm of home.
        # The per-command body/braking check still applies at every speed.
        if len(self.home_path)==1:
            speed = clamp(.07 + .55*max(0.0,radius-.30), .07, .18)
        else:
            speed = .14
        self.drive(self.home_path[0], speed)

    def control(self):
        self.publish_odometry()
        now = self.now()
        healthy = (self.last_joint_received is not None and
                   0 <= now-self.last_joint_received < .30 and
                   self.last_imu_received is not None and
                   0 <= now-self.last_imu_received < .30 and
                   self.last_image_received is not None and
                   0 <= now-self.last_image_received < 2.0)
        if self.fault or self.done or not healthy or self.planner is None or len(self.targets)<5:
            self.command()
            self.dwell_start = None
            if not self.fault and not self.done and now-self.last_wait_log>5:
                self.get_logger().info(f'Waiting: sensors={healthy}, targets={len(self.targets)}, '
                                       f'wall map={self.planner is not None}.')
                self.last_wait_log = now
            return
        if self.phase == 'exit':
            if self.departure_goal is None:
                if not self.route:
                    self.command()
                    return
                first = self.targets[self.route[0]]
                self.direct_departure = direct_approach_safe(
                    (self.x,self.y,self.yaw),first,self.obstacles,
                    max_distance=4.0,margin=.025)
                self.departure_goal = first if self.direct_departure else self.planner.gate
                self.get_logger().info(
                    f'Direct departure toward {first}.' if self.direct_departure else
                    'Direct departure lacks body clearance; using interior staging fallback.')
            # Once inside the graph bounds, continue normal navigation without
            # a forced staging stop. Dynamic targets can then update the route.
            if self.direct_departure and self.planner.free((self.x,self.y)):
                self.phase = 'navigate'
                self.path = [self.departure_goal]
                self.path_goal = self.departure_goal
                self.publish_path()
            elif math.dist((self.x,self.y),self.departure_goal)>.025:
                self.drive(self.departure_goal,.14)
                return
            else:
                self.command()
                if self.settled():
                    self.phase = 'navigate'
                    self.replan()
                return
        if self.phase == 'navigate' and not self.route:
            if len(self.visited)<7:
                self.command()
                return  # Wait for both dynamic targets, never return after five.
            # Try a fully footprint-checked home approach from here first.
            # The interior gate remains a fallback, not a mandatory detour.
            self.command()
            if not self.settled():
                return
            try:
                self.home_path = select_home_goal((self.x,self.y,self.yaw),self.obstacles)
                self.phase = 'home'
                self.get_logger().info(f'Home approach without gate detour: {self.home_path}')
            except ValueError:
                self.phase = 'return'
            self.path, self.path_goal = [], None
        if self.phase == 'home':
            self.control_home(now)
            return
        if self.phase == 'navigate' and not self.dwelling:
            # Collect ANY encountered target, not just the scheduled one.
            # Stop and perform the complete dwell; never mark on a drive-through.
            nearby = [i for i in self.route
                      if math.dist((self.x,self.y),self.targets[i]) <= .13]
            if nearby:
                chosen = min(nearby,key=lambda i:math.dist((self.x,self.y),self.targets[i]))
                if chosen != self.route[0]:
                    self.route.remove(chosen)
                    self.route.insert(0,chosen)
                    self.path,self.path_goal = [],None
                    self.get_logger().info(f'Collecting encountered waypoint {chosen+1} now; avoiding a later revisit.')
                self.dwelling = True
        goal = self.planner.gate if self.phase=='return' else self.targets[self.route[0]]
        distance = math.dist((self.x,self.y),goal)
        tolerance = .025 if self.phase=='return' else (.16 if self.dwelling else .10)
        if distance<=tolerance:
            self.command()
            if self.phase=='return':
                if self.settled():
                    self.phase = 'home'
                    self.dwell_start = None
                return
            self.dwelling = True
            if not self.settled():
                self.dwell_start = None
            elif self.dwell_start is None:
                self.dwell_start = now
            elif now-self.dwell_start>=2.4:
                self.visited.add(self.route[0])
                self.get_logger().info(f'Estimated visit {len(self.visited)}/7; '
                                       f'position=({self.x:.3f},{self.y:.3f}).')
                self.dwelling,self.dwell_start = False,None
                self.replan()
                self.publish_markers()
            return
        self.dwelling,self.dwell_start = False,None
        if self.path_goal!=goal or not self.path:
            try:
                if (self.phase=='navigate' and self.settled() and
                        direct_approach_safe((self.x,self.y,self.yaw),goal,self.obstacles)):
                    self.path = [goal]
                    self.get_logger().info('Using checked direct waypoint approach.')
                else:
                    self.path = self.planner.path((self.x,self.y),goal)[1:]
                self.path_goal = goal
                self.publish_path()
            except ValueError as exc:
                self.fail(str(exc))
                return
        while len(self.path)>1 and self.planner.visible((self.x,self.y),self.path[1]):
            self.path.pop(0)
        if len(self.path)>1 and math.dist((self.x,self.y),self.path[0])<=.020:
            self.command()
            if self.settled():
                self.path.pop(0)
                self.publish_path()
            return
        self.drive(self.path[0])


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

