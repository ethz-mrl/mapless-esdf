#!/usr/bin/env python3
"""Renders a flight through a generated forest into a ROS 2 bag.

The scene is the one from the Jupyter notebook, built from primitives rather
than recorded, so the bag is reproducible from its seed and no dataset has to be
shipped with the repository. What goes in:

  /camera/depth/image_rect_raw   sensor_msgs/Image, 32FC1, metres
  /camera/depth/camera_info      sensor_msgs/CameraInfo
  /odometry/pose                 geometry_msgs/PoseStamped
  /odometry/pose_with_covariance geometry_msgs/PoseWithCovarianceStamped
  /tf                            odom -> camera, for the tf pose source and RViz
  /scene/markers                 visualization_msgs/MarkerArray, the true geometry

The markers are the ground truth the depth images were rendered from, which is
the point of a generated scene: what the buffer reconstructs can be laid over
what is actually there.

Run it inside the ROS image, which has both rosbag2 and the Python bindings:

  docker run --rm --gpus all -v "$PWD:/repo" cuda-depth-buffer:ros \\
    python3 /repo/examples/ros/make_bag.py --output /repo/examples/ros/data/forest
"""

import argparse
import math
import pathlib
import sys

import cuda_depth_buffer as cdb
import numpy as np
import rclpy.serialization
import rosbag2_py
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

REFERENCE_FRAME = "odom"
CAMERA_FRAME = "camera"


# --------------------------------------------------------------------------- #
# trajectory
# --------------------------------------------------------------------------- #

def obstacle_footprints(scene):
    """(x, y, radius) of everything a camera should not fly into."""
    footprints = [(c.base[0], c.base[1], c.radius) for c in scene.cylinders]
    footprints += [(s.center[0], s.center[1], s.radius) for s in scene.spheres]
    return footprints


def clearance_to(footprints, x, y):
    return min((math.hypot(x - fx, y - fy) - r for fx, fy, r in footprints),
               default=math.inf)


def random_waypoints(scene, rng, count, extent, clearance):
    """Waypoints inside the scene, none of them inside a tree."""
    waypoints = []
    while len(waypoints) < count:
        x, y = rng.uniform(-extent, extent, 2)
        if clearance_to(obstacle_footprints(scene), x, y) < clearance:
            continue
        # Keep consecutive waypoints far enough apart that the path does not
        # double back on itself, which looks like a glitch rather than a flight.
        if waypoints and math.dist((x, y), waypoints[-1]) < 0.35 * extent:
            continue
        waypoints.append((x, y))
    return waypoints


def catmull_rom(waypoints, samples_per_segment):
    """A smooth closed loop through the waypoints, so the bag can play round."""
    points = []
    count = len(waypoints)
    for i in range(count):
        p0 = np.array(waypoints[(i - 1) % count])
        p1 = np.array(waypoints[i])
        p2 = np.array(waypoints[(i + 1) % count])
        p3 = np.array(waypoints[(i + 2) % count])
        for step in range(samples_per_segment):
            t = step / samples_per_segment
            t2, t3 = t * t, t * t * t
            points.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                                 (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                                 (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(points)


def push_clear(scene, path, clearance):
    """Nudge samples that the smoothing pushed into a tree back out of it."""
    footprints = obstacle_footprints(scene)
    cleared = path.copy()
    for i, (x, y) in enumerate(cleared):
        for fx, fy, radius in footprints:
            dx, dy = x - fx, y - fy
            distance = math.hypot(dx, dy)
            if distance >= radius + clearance:
                continue
            if distance < 1e-6:
                dx, dy, distance = 1.0, 0.0, 1.0
            scale = (radius + clearance) / distance
            x, y = fx + dx * scale, fy + dy * scale
        cleared[i] = (x, y)
    return cleared


def resample_by_arc_length(loop, spacing, count):
    """`count` points spaced `spacing` apart along the closed `loop`.

    The loop is whatever length the waypoints made it; the camera has a speed.
    Sampling by arc length rather than by index keeps the two independent, so
    asking for more frames flies further instead of flying faster.
    """
    steps = np.linalg.norm(np.diff(loop, axis=0, append=loop[:1]), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = cumulative[-1]
    wanted = (np.arange(count) * spacing) % total
    x = np.interp(wanted, cumulative, np.append(loop[:, 0], loop[0, 0]))
    y = np.interp(wanted, cumulative, np.append(loop[:, 1], loop[0, 1]))
    return np.column_stack([x, y])


def flight(scene, rng, frames, extent, height, clearance, spacing):
    """Poses along a smooth random loop, each looking where it is going."""
    waypoints = random_waypoints(scene, rng, 6, extent, clearance)
    loop = push_clear(scene, catmull_rom(waypoints, 60), clearance)
    path = resample_by_arc_length(loop, spacing, frames)
    poses = []
    # Far enough ahead that the heading follows the path rather than the noise
    # in one step of it, and near enough that it still turns with the corners.
    look_ahead = max(1, int(round(0.6 / max(spacing, 1e-3))))
    for i, (x, y) in enumerate(path):
        ahead_x, ahead_y = path[(i + look_ahead) % len(path)]
        if math.hypot(ahead_x - x, ahead_y - y) < 1e-3:
            ahead_x, ahead_y = x + 1.0, y  # standing still, point somewhere
        # A slow rise and fall, so the slice is not always at the same height.
        z = height + 0.35 * math.sin(2.0 * math.pi * i / len(path))
        target_z = z - 0.15  # look very slightly down, the way a MAV flies
        poses.append(cdb.look_at([float(x), float(y), float(z)],
                                 [float(ahead_x), float(ahead_y), float(target_z)]))
    return poses


# --------------------------------------------------------------------------- #
# messages
# --------------------------------------------------------------------------- #

def stamp_of(nanoseconds):
    return Time(sec=int(nanoseconds // 10**9), nanosec=int(nanoseconds % 10**9))


def camera_info_of(intrinsics, stamp):
    info = CameraInfo()
    info.header.stamp = stamp
    info.header.frame_id = CAMERA_FRAME
    info.width, info.height = intrinsics.width, intrinsics.height
    info.distortion_model = "plumb_bob"
    info.d = [0.0] * 5
    info.k = [intrinsics.fx, 0.0, intrinsics.cx,
              0.0, intrinsics.fy, intrinsics.cy,
              0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [intrinsics.fx, 0.0, intrinsics.cx, 0.0,
              0.0, intrinsics.fy, intrinsics.cy, 0.0,
              0.0, 0.0, 1.0, 0.0]
    return info


def image_of(depth, stamp):
    image = Image()
    image.header.stamp = stamp
    image.header.frame_id = CAMERA_FRAME
    image.height, image.width = depth.shape
    image.encoding = "32FC1"
    image.is_bigendian = 0
    image.step = image.width * 4
    image.data = np.ascontiguousarray(depth, np.float32).tobytes()
    return image


def pose_of(pose, stamp):
    message = PoseStamped()
    message.header.stamp = stamp
    message.header.frame_id = REFERENCE_FRAME
    message.pose.position.x, message.pose.position.y, message.pose.position.z = \
        float(pose[0]), float(pose[1]), float(pose[2])
    (message.pose.orientation.x, message.pose.orientation.y,
     message.pose.orientation.z, message.pose.orientation.w) = \
        float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])
    return message


def transform_of(pose, stamp):
    transform = TransformStamped()
    transform.header.stamp = stamp
    transform.header.frame_id = REFERENCE_FRAME
    transform.child_frame_id = CAMERA_FRAME
    transform.transform.translation.x = float(pose[0])
    transform.transform.translation.y = float(pose[1])
    transform.transform.translation.z = float(pose[2])
    transform.transform.rotation.x = float(pose[3])
    transform.transform.rotation.y = float(pose[4])
    transform.transform.rotation.z = float(pose[5])
    transform.transform.rotation.w = float(pose[6])
    return TFMessage(transforms=[transform])


def covariance_of(pose, stamp, travelled, position_drift, angle_drift):
    """A pose covariance that grows the way dead reckoning drifts.

    Odometry has no absolute reference, so its uncertainty accumulates with the
    distance travelled and never comes back down. This is the artificial version
    of that: a diagonal covariance whose standard deviations grow as the square
    root of the path length, which is what an unbiased random walk gives.

    It is what turns the buffer's pose compensation on. Obstacles seen by a
    frame whose pose is uncertain get inflated by that uncertainty, so a frame
    the odometry has lost track of cannot erode an obstacle a confident frame
    put there.
    """
    message = PoseWithCovarianceStamped()
    message.header.stamp = stamp
    message.header.frame_id = REFERENCE_FRAME
    message.pose.pose = pose_of(pose, stamp).pose
    position_sigma = position_drift * math.sqrt(travelled)
    angle_sigma = angle_drift * math.sqrt(travelled)
    covariance = [0.0] * 36
    for i in range(3):
        covariance[i * 6 + i] = position_sigma ** 2
        covariance[(i + 3) * 6 + (i + 3)] = angle_sigma ** 2
    message.pose.covariance = covariance
    return message


def quaternion_from_z_to(axis):
    """Rotation taking +z onto `axis`, which is how a Marker cylinder is posed."""
    axis = np.asarray(axis, float)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z, axis)
    dot = float(np.dot(z, axis))
    if np.linalg.norm(cross) < 1e-9:
        return (0.0, 0.0, 0.0, 1.0) if dot > 0 else (1.0, 0.0, 0.0, 0.0)
    quaternion = np.append(cross, 1.0 + dot)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(v) for v in quaternion)


def scene_markers(scene, stamp, extent):
    """The geometry the depth images were rendered from."""
    markers = MarkerArray()
    identifier = 0

    def base(kind, colour):
        nonlocal identifier
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = REFERENCE_FRAME
        marker.ns = "scene"
        marker.id = identifier
        identifier += 1
        marker.type = kind
        marker.action = Marker.ADD
        marker.color = colour
        marker.pose.orientation.w = 1.0
        return marker

    bark = ColorRGBA(r=0.45, g=0.31, b=0.19, a=1.0)
    leaf = ColorRGBA(r=0.24, g=0.50, b=0.24, a=1.0)
    ground = ColorRGBA(r=0.30, g=0.29, b=0.26, a=1.0)

    for cylinder in scene.cylinders:
        marker = base(Marker.CYLINDER, bark)
        axis = np.asarray(cylinder.axis, float)
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        centre = np.asarray(cylinder.base, float) + axis * (cylinder.height / 2.0)
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = \
            (float(v) for v in centre)
        (marker.pose.orientation.x, marker.pose.orientation.y,
         marker.pose.orientation.z, marker.pose.orientation.w) = \
            quaternion_from_z_to(axis)
        marker.scale.x = marker.scale.y = 2.0 * cylinder.radius
        marker.scale.z = cylinder.height
        markers.markers.append(marker)

    for sphere in scene.spheres:
        marker = base(Marker.SPHERE, leaf)
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = \
            (float(v) for v in sphere.center)
        marker.scale.x = marker.scale.y = marker.scale.z = 2.0 * sphere.radius
        markers.markers.append(marker)

    for plane in scene.planes:
        # An infinite plane has to be drawn as something finite; a thin slab
        # covering the scene is close enough to look like the floor.
        marker = base(Marker.CUBE, ground)
        marker.pose.position.x = float(plane.point[0])
        marker.pose.position.y = float(plane.point[1])
        marker.pose.position.z = float(plane.point[2]) - 0.01
        marker.scale.x = marker.scale.y = 1.4 * extent
        marker.scale.z = 0.02
        markers.markers.append(marker)

    return markers


# --------------------------------------------------------------------------- #

def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path(__file__).parent / "data" / "forest",
                        help="bag directory to create")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--rate", type=float, default=15.0, help="Hz")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--extent", type=float, default=14.0,
                        help="side of the square the forest occupies, metres")
    parser.add_argument("--flight-height", type=float, default=1.6,
                        help="mean height above the ground, metres")
    parser.add_argument("--clearance", type=float, default=1.1,
                        help="how far the flight stays off a trunk, metres")
    parser.add_argument("--speed", type=float, default=1.5,
                        help="how fast the camera flies, metres per second")
    parser.add_argument("--covariance", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="also write pose covariances that grow with the "
                             "distance travelled, the way odometry drifts")
    parser.add_argument("--position-drift", type=float, default=0.02,
                        help="metres of position sigma per sqrt(metre) travelled")
    parser.add_argument("--angle-drift", type=float, default=0.004,
                        help="radians of angle sigma per sqrt(metre) travelled")
    parser.add_argument("--storage", default="mcap", choices=["mcap", "sqlite3"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    if args.output.exists():
        print(f"{args.output} already exists, refusing to overwrite it",
              file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    intrinsics = cdb.CameraIntrinsics(
        fx=0.7 * args.width, fy=0.7 * args.width,
        cx=0.5 * args.width - 0.5, cy=0.5 * args.height - 0.5,
        width=args.width, height=args.height, z_near=0.3, z_far=15.0)
    scene = cdb.AnalyticScene.forest(seed=args.seed, num_cylinders=40,
                                     num_spheres=24, extent=args.extent)
    rng = np.random.default_rng(args.seed)
    poses = flight(scene, rng, args.frames, 0.42 * args.extent,
                   args.flight_height, args.clearance, args.speed / args.rate)

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(args.output), storage_id=args.storage),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                    output_serialization_format="cdr"))
    topics = {
        "/camera/depth/image_rect_raw": "sensor_msgs/msg/Image",
        "/camera/depth/camera_info": "sensor_msgs/msg/CameraInfo",
        "/odometry/pose": "geometry_msgs/msg/PoseStamped",
        "/tf": "tf2_msgs/msg/TFMessage",
        "/scene/markers": "visualization_msgs/msg/MarkerArray",
    }
    if args.covariance:
        topics["/odometry/pose_with_covariance"] = \
            "geometry_msgs/msg/PoseWithCovarianceStamped"
    for name, message_type in topics.items():
        writer.create_topic(rosbag2_py.TopicMetadata(
            id=0, name=name, type=message_type, serialization_format="cdr"))

    period_ns = int(10**9 / args.rate)
    start_ns = 1_000_000_000
    travelled = 0.0
    previous = None
    marker_period = max(1, int(round(args.rate)))  # about once a second

    for index, pose in enumerate(poses):
        stamp_ns = start_ns + index * period_ns
        stamp = stamp_of(stamp_ns)
        position = np.array(pose[:3])
        if previous is not None:
            travelled += float(np.linalg.norm(position - previous))
        previous = position

        depth = scene.render(intrinsics, pose)
        writer.write("/camera/depth/image_rect_raw",
                     rclpy.serialization.serialize_message(image_of(depth, stamp)),
                     stamp_ns)
        writer.write("/camera/depth/camera_info",
                     rclpy.serialization.serialize_message(
                         camera_info_of(intrinsics, stamp)), stamp_ns)
        writer.write("/odometry/pose",
                     rclpy.serialization.serialize_message(pose_of(pose, stamp)),
                     stamp_ns)
        writer.write("/tf",
                     rclpy.serialization.serialize_message(transform_of(pose, stamp)),
                     stamp_ns)
        if args.covariance:
            writer.write("/odometry/pose_with_covariance",
                         rclpy.serialization.serialize_message(
                             covariance_of(pose, stamp, travelled,
                                           args.position_drift, args.angle_drift)),
                         stamp_ns)
        if index % marker_period == 0:
            # Repeated rather than written once, so RViz still gets the scene
            # when the bag is started from the middle or looped.
            writer.write("/scene/markers",
                         rclpy.serialization.serialize_message(
                             scene_markers(scene, stamp, args.extent)), stamp_ns)

        if index % 25 == 0 or index + 1 == len(poses):
            print(f"  {index + 1:>4}/{len(poses)} frames", flush=True)

    del writer
    duration = args.frames / args.rate
    print(f"\nwrote {args.output}")
    print(f"  {args.frames} frames at {args.rate:g} Hz, {duration:.1f} s, "
          f"{args.width}x{args.height}")
    print(f"  {travelled:.1f} m at {args.speed:g} m/s through a forest of "
          f"{len(scene.cylinders)} trunks and {len(scene.spheres)} bushes")
    if args.covariance:
        final = args.position_drift * math.sqrt(travelled)
        print(f"  pose covariance grows to {final * 100:.1f} cm of position "
              f"sigma by the end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
