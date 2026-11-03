# cuda_depth_buffer_ros

Signed distance queries straight from a ring of posed depth images, with no map
in between. A new frame is usable the moment it arrives and an old one is gone
the moment it is overwritten, so there is nothing to fuse, nothing to drift and
nothing to clear.

Two distances come back for every point:

| | space no camera has seen counts as |
|---|---|
| **optimistic** | free |
| **conservative** | occupied, and the distance is signed |

Both are always returned. Producing the optimistic one costs nothing extra, it
is the state of the search before the occlusion pass runs, and a planner that
wants to weigh one against the other should not have to ask twice.

## See it first

`examples/ros/` generates a bag from the same scene the notebook uses, flies a
camera through it and shows the result in RViz. That is the fastest way to
understand what this does; the rest of this file is the reference.

## Build

The node needs the library installed first; `Dockerfile.ros` in the repository
root does the whole thing:

```bash
docker build -f Dockerfile.ros -t cuda-depth-buffer:ros .
docker run --rm --gpus all -it --net=host --ipc=host cuda-depth-buffer:ros \
  ros2 launch cuda_depth_buffer_ros depth_buffer.launch.py
```

To build into an existing workspace instead:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/opt/cuda_depth_buffer \
  -DCUDA_DEPTH_BUFFER_BUILD_TESTS=OFF
cmake --build build --target install
CMAKE_PREFIX_PATH=/opt/cuda_depth_buffer colcon build --packages-up-to cuda_depth_buffer_ros
```

## Interfaces

### Subscriptions

| topic | type | |
|---|---|---|
| `~/depth` | `sensor_msgs/Image` | `32FC1` in metres, or `16UC1` with `depth_scale_factor` |
| `~/camera_info` | `sensor_msgs/CameraInfo` | geometry, unless it comes from parameters |
| `~/pose` | `geometry_msgs/PoseStamped` | camera pose, when `pose_source: topic` |
| `~/pose_with_covariance` | `geometry_msgs/PoseWithCovarianceStamped` | same, and attaches the covariance |
| `~/pose_updates` | `nav_msgs/Path` | bulk pose revision by timestamp |

### Publications

| topic | type | |
|---|---|---|
| `~/esdf_slice` | `sensor_msgs/PointCloud2` | horizontal slice, `intensity` is the distance |

The slice is only computed while something is subscribed to it. Colour it by
`intensity` in RViz.

### Services

`~/query` (`cuda_depth_buffer_msgs/QueryDistances`) takes points and returns
both channels. Points and every returned vector are xyz interleaved float32 in
the reference frame, the same layout as the C++ and Python interfaces:

```bash
ros2 service call /cuda_depth_buffer/query \
  cuda_depth_buffer_msgs/srv/QueryDistances \
  "{points: [0.0, 0.0, 1.0,  1.0, 0.0, 1.0], conservative: true}"
```

When `conservative` is false the three conservative arrays come back full of
NaN rather than empty, so the layout never changes. A distance saturated at the
truncation radius has no closest point either, and its point and gradient are
NaN too; that is how "nothing within the truncation radius" is told apart from
a real answer at that distance.

## Where the poses come from

Two ways, chosen with `pose_source`:

- `topic` reads `~/pose`. The pose used for an image is the most recent one
  received, which is what a tightly coupled estimator publishing at camera rate
  gives you.
- `tf` looks up `reference_frame` → the image's `frame_id` **at the image
  stamp**, waiting up to `tf_timeout`. Use this when the estimator only
  broadcasts transforms.

Either way, queries and results are in `reference_frame`. Nothing in the
interface is in a camera frame.

## Revising poses

A SLAM backend that closes a loop moves frames the buffer is still holding.
Publish the corrected trajectory on `~/pose_updates` as a `nav_msgs/Path`; every
pose whose stamp matches a buffered frame is revised in place, newest or oldest
alike, and the ones that do not match are ignored. The next query sees the new
geometry; there is nothing to re-integrate.

`nav_msgs/Path` carries no covariance. To attach one, publish a
`PoseWithCovarianceStamped` on `~/pose_with_covariance` with the frame's stamp.

## Pose uncertainty

A frame the odometry is unsure about should not be allowed to erode an obstacle
another frame is confident about. Given a covariance, obstacles seen by that
frame are inflated by `pose_covariance_confidence` standard deviations of the
pose uncertainty at the query's lever arm. A frame without a covariance
contributes no margin, so supplying covariances is what turns this on;
`pose_covariance_confidence: 0.0` turns it off outright.

## Composing it

The node is a component. Loading it into the camera driver's container is the
point: the depth image is then handed over as a pointer inside one process
rather than serialized onto the wire.

```python
ComposableNodeContainer(
    name="perception", namespace="", package="rclcpp_components",
    executable="component_container_mt",   # _mt: queries are meant to overlap
    composable_node_descriptions=[
        ComposableNode(package="realsense2_camera",
                       plugin="realsense2_camera::RealSenseNodeFactory",
                       name="camera",
                       extra_arguments=[{"use_intra_process_comms": True}]),
        ComposableNode(package="cuda_depth_buffer_ros",
                       plugin="cuda_depth_buffer_ros::DepthBufferNode",
                       name="cuda_depth_buffer",
                       parameters=[config],
                       remappings=[("~/depth", "/camera/depth/image_rect_raw")],
                       extra_arguments=[{"use_intra_process_comms": True}]),
    ])
```

Use `component_container_mt`, not `component_container`. Image insertion and
pose handling are serialized against each other, but queries run concurrently
with them and with one another, which a single threaded container would undo.

Running it standalone instead, add `--ipc=host` to the docker invocation so the
middleware can still use its shared memory transport across processes.

## Configuration

Everything lives in `config/depth_buffer.yaml`, which is commented. The ones
worth knowing about before anything else:

- `num_frames` — the size of the ring, fixed for the life of the node. More
  frames means more of the world remembered and a proportionally slower query.
- `truncation_radius` — the search stops here. The strongest lever on query
  time by a wide margin, and a planner rarely cares about clearance beyond a
  few metres.
- `depth_scale_factor` — `0.001` for a `16UC1` stream in millimetres.
- `invalid_is_farplane` — what a pixel with no measurement means. The default,
  `false`, reads it as the near plane, which is the conservative choice: an
  unmeasured pixel is treated as something very close rather than as clear sky.
- `min_pool_size` — min pools the image on insert. Trades resolution for query
  time and keeps the pessimistic reading of every window.

## Tests

`cuda_depth_buffer_ros/test/` drives a running node over its real interfaces
rather than linking against it, so it covers the service, both pose paths, the
`nav_msgs/Path` revision and the slice. The scene is a flat wall at a known
distance, so every expected number is arithmetic. See the README next to them.

## What it does not do

- **The image geometry is fixed** when the buffer is built, from the first
  `CameraInfo` or from the parameters. A stream that changes resolution needs
  the node restarted.
- **No map is published**, because none exists. `~/esdf_slice` is a slice
  computed on request, not a stored grid.
- **Poses are not interpolated.** With `pose_source: topic` an image gets the
  most recent pose; use `tf` if you need the lookup done at the image stamp.
