"""Starts the node with the packaged configuration.

Composing it into the camera driver's container instead is the point of the
component; see the package README for that. This launch file is the plain way
to bring it up on its own.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("cuda_depth_buffer_ros")
    arguments = [
        DeclareLaunchArgument(
            "config",
            default_value=PathJoinSubstitution(
                [package_share, "config", "depth_buffer.yaml"]),
            description="parameter file"),
        DeclareLaunchArgument(
            "depth_topic", default_value="/camera/depth/image_rect_raw",
            description="sensor_msgs/Image, 32FC1 or 16UC1"),
        DeclareLaunchArgument(
            "camera_info_topic", default_value="/camera/depth/camera_info"),
        DeclareLaunchArgument(
            "pose_topic", default_value="/odometry/pose",
            description="geometry_msgs/PoseStamped, ignored when pose_source=tf"),
    ]

    node = Node(
        package="cuda_depth_buffer_ros",
        executable="depth_buffer_node",
        name="cuda_depth_buffer",
        output="screen",
        parameters=[LaunchConfiguration("config")],
        remappings=[
            ("~/depth", LaunchConfiguration("depth_topic")),
            ("~/camera_info", LaunchConfiguration("camera_info_topic")),
            ("~/pose", LaunchConfiguration("pose_topic")),
        ],
    )
    return LaunchDescription(arguments + [node])
