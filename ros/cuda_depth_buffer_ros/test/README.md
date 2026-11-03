# Node tests

These drive a running node over its real interfaces rather than linking against
it, so they cover what a deployment actually uses: the service, the pose paths,
the `nav_msgs/Path` revision and the published slice. They need a GPU.

Start the node, then run one:

```bash
ros2 run cuda_depth_buffer_ros depth_buffer_node --ros-args \
  -p reference_frame:=odom -p num_frames:=5 -p truncation_radius:=5.0 &
python3 test/test_node_interfaces.py
```

`test_tf_poses.py` wants the node started with `-p pose_source:=tf` instead.

The scene is a flat wall at a known distance, so every expected number is
arithmetic rather than a recorded value.
