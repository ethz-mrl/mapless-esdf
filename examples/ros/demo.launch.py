"""The node and RViz, set up for the generated bag.

The bag is played separately so it can be paused and stepped by hand; see the
README next to this file.
"""

import pathlib

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = pathlib.Path(__file__).parent


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument(
            "rviz", default_value="true",
            description="open RViz with the demo configuration"),
        DeclareLaunchArgument(
            "slice_height", default_value="1.0",
            description="height of the published slice, metres"),
        DeclareLaunchArgument(
            "truncation_radius", default_value="3.0",
            description="the search stops here; the strongest lever on cost"),
        DeclareLaunchArgument(
            "pose_covariance_confidence", default_value="4.0",
            description="sigmas of pose uncertainty obstacles are inflated by;"
                        " 0 turns the compensation off"),
    ]

    # use_sim_time throughout: the bag is played with --clock, so pausing it
    # pauses the node's slice timer with it.
    buffer_node = Node(
        package="cuda_depth_buffer_ros",
        executable="depth_buffer_node",
        name="cuda_depth_buffer",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "reference_frame": "odom",
            "pose_source": "topic",
            # Twenty frames at camera rate is a second and a half of flight,
            # which is a very small neighbourhood. Keyframing spends the same
            # twenty slots on frames that differ, so the buffer covers the
            # stretch actually flown; slot 0 still always holds the latest.
            "num_frames": 24,
            "use_keyframing": True,
            "keyframe_dist_threshold": 0.6,
            "keyframe_angle_threshold": 0.12,
            "keyframe_time_threshold": 1.5,
            "z_near": 0.3,
            "z_far": 15.0,
            "truncation_radius": LaunchConfiguration("truncation_radius"),
            "pose_covariance_confidence":
                LaunchConfiguration("pose_covariance_confidence"),
            "slice.rate": 4.0,
            "slice.height": LaunchConfiguration("slice_height"),
            "slice.extent": 12.0,
            "slice.resolution": 160,
            "slice.conservative": True,
        }],
        remappings=[
            ("~/depth", "/camera/depth/image_rect_raw"),
            ("~/camera_info", "/camera/depth/camera_info"),
            ("~/pose", "/odometry/pose"),
            ("~/pose_with_covariance", "/odometry/pose_with_covariance"),
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", str(HERE / "rviz" / "demo.rviz")],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(arguments + [buffer_node, rviz])
