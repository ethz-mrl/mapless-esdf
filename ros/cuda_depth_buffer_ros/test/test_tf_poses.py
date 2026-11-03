"""Same as test_node_interfaces, with poses from tf instead of a topic.

Needs the node started with pose_source:=tf.
"""
import sys
import threading
import time

import numpy as np
import rclpy
from cuda_depth_buffer_msgs.srv import QueryDistances
from geometry_msgs.msg import TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

WIDTH, HEIGHT, WALL_Z = 128, 96, 4.0
failures = []


def check(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what, flush=True)
    if not condition:
        failures.append(what)


def main():
    rclpy.init()
    node = Node("tf_driver")
    broadcaster = TransformBroadcaster(node)
    info_pub = node.create_publisher(CameraInfo, "/cuda_depth_buffer/camera_info",
                                     qos_profile_sensor_data)
    image_pub = node.create_publisher(Image, "/cuda_depth_buffer/depth",
                                      qos_profile_sensor_data)
    client = node.create_client(QueryDistances, "/cuda_depth_buffer/query")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    if not client.wait_for_service(timeout_sec=30.0):
        print("  FAIL service never appeared")
        return 1

    # The camera sits 2 m up. Everything it sees is 2 m higher than in its own
    # frame, so the wall lands at z = 2 + 4 in odom.
    for _ in range(4):
        stamp = node.get_clock().now().to_msg()
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "camera"
        transform.transform.translation.z = 2.0
        transform.transform.rotation.w = 1.0
        broadcaster.sendTransform(transform)
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = "camera"
        info.width, info.height = WIDTH, HEIGHT
        info.k = [100.0, 0.0, 63.5, 0.0, 100.0, 47.5, 0.0, 0.0, 1.0]
        info_pub.publish(info)
        time.sleep(0.2)
        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = "camera"
        image.width, image.height = WIDTH, HEIGHT
        image.encoding = "32FC1"
        image.step = WIDTH * 4
        image.data = np.full(WIDTH * HEIGHT, WALL_Z, np.float32).tobytes()
        image_pub.publish(image)
        time.sleep(0.3)

    request = QueryDistances.Request()
    request.points = [0.0, 0.0, 3.0]   # 3 m up in odom, so 1 m along the camera axis
    request.truncation_radius = 5.0
    future = client.call_async(request)
    deadline = time.time() + 10.0
    while not future.done() and time.time() < deadline:
        time.sleep(0.01)
    result = future.result()
    check(result is not None and result.success, "service answers with poses from tf")
    if result and result.success:
        check(abs(result.optimistic_distance[0] - 3.0) < 0.05,
              f"optimistic {result.optimistic_distance[0]:.3f} == 3.0, "
              f"so the tf offset of 2 m was applied")
    rclpy.shutdown()
    print("all checks passed" if not failures else "FAILED: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
