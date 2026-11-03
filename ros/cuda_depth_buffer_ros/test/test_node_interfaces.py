"""Drives a running node over its real interfaces.

A flat wall at a known distance, so every expected number below is
arithmetic rather than a recorded value. Needs a node already running;
see README.md next to this file.
"""
import math
import sys
import threading
import time

import numpy as np
import rclpy
from cuda_depth_buffer_msgs.srv import QueryDistances
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

WIDTH, HEIGHT = 128, 96
FX = FY = 100.0
CX, CY = WIDTH / 2 - 0.5, HEIGHT / 2 - 0.5
WALL_Z = 4.0

failures = []


def check(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what, flush=True)
    if not condition:
        failures.append(what)


class Driver(Node):
    def __init__(self):
        super().__init__("driver")
        self.info_pub = self.create_publisher(CameraInfo, "/cuda_depth_buffer/camera_info",
                                              qos_profile_sensor_data)
        self.image_pub = self.create_publisher(Image, "/cuda_depth_buffer/depth",
                                               qos_profile_sensor_data)
        self.pose_pub = self.create_publisher(PoseStamped, "/cuda_depth_buffer/pose",
                                              qos_profile_sensor_data)
        self.cov_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/cuda_depth_buffer/pose_with_covariance",
            qos_profile_sensor_data)
        self.path_pub = self.create_publisher(Path, "/cuda_depth_buffer/pose_updates", 1)
        self.client = self.create_client(QueryDistances, "/cuda_depth_buffer/query")
        self.slices = []
        self.slice_sub = self.create_subscription(
            PointCloud2, "/cuda_depth_buffer/esdf_slice",
            lambda msg: self.slices.append(msg), 1)

    def camera_info(self, stamp):
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = "camera"
        info.width, info.height = WIDTH, HEIGHT
        info.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
        return info

    def image(self, stamp, depth_value):
        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = "camera"
        image.width, image.height = WIDTH, HEIGHT
        image.encoding = "32FC1"
        image.step = WIDTH * 4
        image.data = np.full(WIDTH * HEIGHT, depth_value, np.float32).tobytes()
        return image

    def pose(self, stamp, x):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "odom"
        pose.pose.position.x = x
        pose.pose.orientation.w = 1.0
        return pose

    def query(self, points, conservative=True, truncation=5.0):
        request = QueryDistances.Request()
        request.points = [float(v) for v in points]
        request.conservative = conservative
        request.truncation_radius = truncation
        future = self.client.call_async(request)
        deadline = time.time() + 10.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.01)
        return future.result()


def main():
    rclpy.init()
    driver = Driver()
    executor = MultiThreadedExecutor()
    executor.add_node(driver)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    print("waiting for the node ...", flush=True)
    if not driver.client.wait_for_service(timeout_sec=30.0):
        print("  FAIL service never appeared")
        return 1

    # A wall 4 m in front of a camera at the origin, looking down +z.
    stamps = []
    for _ in range(3):
        stamp = driver.get_clock().now().to_msg()
        stamps.append(stamp)
        driver.info_pub.publish(driver.camera_info(stamp))
        driver.pose_pub.publish(driver.pose(stamp, 0.0))
        time.sleep(0.25)
        driver.image_pub.publish(driver.image(stamp, WALL_Z))
        time.sleep(0.25)

    print("query", flush=True)
    result = driver.query([0.0, 0.0, 1.0])
    check(result is not None and result.success, "service answers")
    if result is None or not result.success:
        return 1
    check(abs(result.optimistic_distance[0] - (WALL_Z - 1.0)) < 0.05,
          f"optimistic {result.optimistic_distance[0]:.3f} == 3.0 (wall at 4 m)")
    check(len(result.optimistic_gradient) == 3 and len(result.optimistic_closest) == 3,
          "gradient and closest point are 3 long")
    check(math.isfinite(result.conservative_distance[0]),
          f"conservative {result.conservative_distance[0]:.3f} is finite")

    print("conservative=false", flush=True)
    plain = driver.query([0.0, 0.0, 1.0], conservative=False)
    check(math.isnan(plain.conservative_distance[0]),
          "conservative channel is NaN when not requested")
    check(len(plain.conservative_distance) == 1,
          "and still has the same length")

    print("batch", flush=True)
    batch = [c for z in (1.0, 2.0, 3.0) for c in (0.0, 0.0, z)]
    many = driver.query(batch)
    check(len(many.optimistic_distance) == 3, "three points in, three out")
    check(all(abs(many.optimistic_distance[i] - (WALL_Z - z)) < 0.05
              for i, z in enumerate((1.0, 2.0, 3.0))),
          f"all three correct: {[round(v, 3) for v in many.optimistic_distance]}")

    print("bad request", flush=True)
    bad = driver.query([0.0, 0.0])
    check(bad is not None and not bad.success, "a malformed request is refused")
    check("multiple of three" in bad.message, f"and says why: '{bad.message}'")

    print("pose revision through nav_msgs/Path", flush=True)
    path = Path()
    path.header.frame_id = "odom"
    for stamp in stamps:
        moved = driver.pose(stamp, 0.0)
        moved.pose.position.z = -1.0   # the whole camera moved back a metre
        path.poses.append(moved)
    driver.path_pub.publish(path)
    time.sleep(1.0)
    revised = driver.query([0.0, 0.0, 1.0])
    check(abs(revised.optimistic_distance[0] - (WALL_Z - 1.0 - 1.0)) < 0.05,
          f"after the path update {revised.optimistic_distance[0]:.3f} == 2.0")

    print("covariance", flush=True)
    covariance = PoseWithCovarianceStamped()
    covariance.header.stamp = stamps[-1]
    covariance.header.frame_id = "odom"
    covariance.pose.pose.orientation.w = 1.0
    covariance.pose.covariance = [0.0] * 36
    for i in range(3):
        covariance.pose.covariance[i * 6 + i] = 0.04
    driver.cov_pub.publish(covariance)
    time.sleep(1.0)
    inflated = driver.query([0.0, 0.0, 1.0])
    check(inflated.optimistic_distance[0] < revised.optimistic_distance[0] + 1e-4,
          f"a covariance does not make things look further: "
          f"{inflated.optimistic_distance[0]:.3f} <= {revised.optimistic_distance[0]:.3f}")

    print("slice", flush=True)
    deadline = time.time() + 10.0
    while not driver.slices and time.time() < deadline:
        time.sleep(0.1)
    check(bool(driver.slices), "the slice is published once subscribed")
    if driver.slices:
        cloud = driver.slices[-1]
        check(cloud.width == 128 * 128, f"slice has {cloud.width} points")
        names = [f.name for f in cloud.fields]
        check(names[:4] == ["x", "y", "z", "intensity"],
              f"positions and the raw distance are there: {names[:4]}")
        # slice.colorize bakes in a diverging colour map RViz cannot express.
        check("rgb" in names, "and the colour, since colorize defaults to on")
        check(cloud.header.frame_id == "odom", "in the reference frame")

    rclpy.shutdown()
    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all checks passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
