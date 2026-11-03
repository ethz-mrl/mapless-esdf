"""The bindings are thin, so what is worth testing is the boundary: that arrays
arrive with the right shape and dtype whichever framework they came from, that a
device array is read where it lies, and that the answers match the exhaustive
search the C++ tests use as their oracle."""

import ctypes

import numpy as np
import pytest
from cuda_depth_buffer import (
    AnalyticScene,
    BufferParams,
    CameraIntrinsics,
    CudaDepthBuffer,
    PosedDepthImage,
    brute_force_distances,
    look_at,
    orbit_poses,
)


def _load_cudart():
    """The runtime, whatever it is called on this machine."""
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12",
                 "libcudart.so.11"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _check(status):
    if status != 0:
        raise RuntimeError(f"CUDA runtime call failed with status {status}")


TRUNCATION = 3.0

_cudart = _load_cudart()


class DeviceArray:
    """A minimal __cuda_array_interface__ producer.

    The bindings' fast path is meant to work for anything that exposes the
    protocol, so the test for it should not need torch or cupy installed to run.
    This allocates through the CUDA runtime directly and describes the result the
    way those libraries would.
    """

    def __init__(self, array):
        array = np.ascontiguousarray(array, dtype=np.float32)
        self.shape = array.shape
        self._nbytes = array.nbytes
        self._pointer = ctypes.c_void_p()
        _check(_cudart.cudaMalloc(ctypes.byref(self._pointer),
                                  ctypes.c_size_t(self._nbytes)))
        _check(_cudart.cudaMemcpy(self._pointer,
                                  array.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_size_t(self._nbytes),
                                  1))  # cudaMemcpyHostToDevice

    @property
    def __cuda_array_interface__(self):
        return {"shape": self.shape, "typestr": "<f4", "version": 3,
                "data": (self._pointer.value, False)}

    def __del__(self):
        if getattr(self, "_pointer", None) is not None:
            _cudart.cudaFree(self._pointer)
            self._pointer = None



def intrinsics():
    return CameraIntrinsics(fx=90.0, fy=110.0, cx=61.0, cy=51.0, width=128,
                            height=96, z_near=0.2, z_far=12.0)


def params(num_frames=4):
    p = BufferParams()
    p.num_frames = num_frames
    p.pose_covariance_confidence = 0.0
    return p


@pytest.fixture(scope="module")
def filled():
    k = intrinsics()
    scene = AnalyticScene.forest(3, 14, 10, 8.0)
    poses = orbit_poses(4, 6.0, 1.5)
    buffer = CudaDepthBuffer(k, params())
    frames = []
    for i, pose in enumerate(poses):
        depth = scene.render(k, pose)
        buffer.set_posed_depth_image(depth, np.asarray(pose, np.float32),
                                     stamp_ns=i, is_keyframe=False)
        frames.append(PosedDepthImage(depth, pose, i))
    return buffer, k, frames


@pytest.fixture(scope="module")
def points():
    rng = np.random.default_rng(0)
    return np.column_stack([rng.uniform(-3, 3, 200), rng.uniform(-3, 3, 200),
                            rng.uniform(0, 3, 200)]).astype(np.float32)


def test_rendered_depth_has_image_shape():
    k = intrinsics()
    depth = AnalyticScene.forest(1).render(k, look_at([5.0, 0.0, 1.0], [0.0, 0.0, 1.0]))
    assert depth.shape == (k.height, k.width)
    assert depth.dtype == np.float32
    assert np.all(depth >= k.z_near) and np.all(depth <= k.z_far)


def test_result_shapes_and_dtypes(filled, points):
    buffer, _, _ = filled
    result = buffer.query(points, truncation_radius=TRUNCATION)
    assert result.conservative_distance.shape == (len(points),)
    assert result.conservative_gradient.shape == (len(points), 3)
    assert result.conservative_closest.shape == (len(points), 3)
    assert result.conservative_distance.dtype == np.float32
    assert result.distance(conservative=False) is result.optimistic_distance


def test_matches_the_exhaustive_oracle(filled, points):
    buffer, k, frames = filled
    result = buffer.query(points, conservative=True, truncation_radius=TRUNCATION)
    for i, point in enumerate(points):
        expected = brute_force_distances(k, frames, point.tolist(), TRUNCATION)
        assert result.optimistic_distance[i] == pytest.approx(
            expected["optimistic"], abs=1e-4)
        assert result.conservative_distance[i] == pytest.approx(
            expected["conservative"], abs=1e-4)


def test_unrequested_conservative_channel_is_nan(filled, points):
    buffer, _, _ = filled
    result = buffer.query(points, conservative=False, truncation_radius=TRUNCATION)
    assert np.all(np.isnan(result.conservative_distance))
    assert np.all(np.isfinite(result.optimistic_distance))


def test_flat_and_shaped_queries_agree(filled, points):
    buffer, _, _ = filled
    shaped = buffer.query(points, truncation_radius=TRUNCATION)
    flat = buffer.query(points.reshape(-1), truncation_radius=TRUNCATION)
    assert np.array_equal(shaped.conservative_distance, flat.conservative_distance)


def test_float64_input_is_converted(filled, points):
    buffer, _, _ = filled
    reference = buffer.query(points, truncation_radius=TRUNCATION)
    converted = buffer.query(points.astype(np.float64), truncation_radius=TRUNCATION)
    assert np.array_equal(reference.conservative_distance,
                          converted.conservative_distance)


def test_query_size_must_be_a_multiple_of_three(filled):
    buffer, _, _ = filled
    with pytest.raises(ValueError, match="multiple of three"):
        buffer.query(np.zeros(7, np.float32))


def test_depth_size_is_checked(filled):
    buffer, k, _ = filled
    with pytest.raises(ValueError, match="depth"):
        buffer.set_posed_depth_image(np.zeros(k.width * k.height - 1, np.float32),
                                     np.array([0, 0, 0, 0, 0, 0, 1], np.float32), 0)


def test_params_round_trip_through_yaml():
    p = params(num_frames=7)
    p.min_pool_size = 3
    assert BufferParams.from_yaml(p.to_yaml()).min_pool_size == 3
    assert BufferParams.from_yaml(p.to_yaml()).num_frames == 7


def test_keyframe_decision_is_reported_not_applied():
    k = intrinsics()
    p = params()
    p.keyframe_dist_threshold = 1.0
    buffer = CudaDepthBuffer(k, p)
    pose = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
    assert buffer.is_keyframe(pose, 0)  # nothing buffered yet
    buffer.set_posed_depth_image(np.ones(k.width * k.height, np.float32), pose, 0)
    assert not buffer.is_keyframe(pose, 0)
    far = np.array([2, 0, 0, 0, 0, 0, 1], np.float32)
    assert buffer.is_keyframe(far, 0)


def test_covariance_round_trips():
    k = intrinsics()
    buffer = CudaDepthBuffer(k, params())
    pose = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
    buffer.set_posed_depth_image(np.ones(k.width * k.height, np.float32), pose, 17)
    covariance = np.diag([0.01, 0.02, 0.03, 0.001, 0.002, 0.003])
    assert buffer.update_pose_and_covariance_by_timestamp(17, covariance=covariance)
    assert np.allclose(buffer.covariance_by_timestamp(17), covariance)
    assert buffer.covariance_by_timestamp(99) is None


try:
    import torch
except ImportError:  # optional, the device path is covered without it above
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")
requires_torch_cuda = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="torch without a CUDA device")


@requires_torch_cuda
def test_cuda_tensors_are_read_in_place(filled, points):
    buffer, _, _ = filled
    reference = buffer.query(points, truncation_radius=TRUNCATION)
    on_device = buffer.query(torch.as_tensor(points).cuda(), truncation_radius=TRUNCATION)
    assert np.array_equal(reference.conservative_distance,
                          on_device.conservative_distance)
    assert np.array_equal(reference.conservative_closest,
                          on_device.conservative_closest)


@requires_torch
def test_host_tensors_arrive_through_dlpack(filled, points):
    buffer, _, _ = filled
    reference = buffer.query(points, truncation_radius=TRUNCATION)
    through_dlpack = buffer.query(torch.as_tensor(points), truncation_radius=TRUNCATION)
    assert np.array_equal(reference.conservative_distance,
                          through_dlpack.conservative_distance)


@requires_torch_cuda
def test_unusable_device_arrays_say_why(filled, points):
    buffer, _, _ = filled
    with pytest.raises(ValueError, match="not contiguous"):
        buffer.query(torch.as_tensor(points).cuda()[::2])
    with pytest.raises(ValueError, match="not float32"):
        buffer.query(torch.as_tensor(points).double().cuda())


@requires_torch
def test_unusable_host_arrays_are_converted_instead(filled, points):
    buffer, _, _ = filled
    reference = buffer.query(points[::2], truncation_radius=TRUNCATION)
    strided = buffer.query(torch.as_tensor(points)[::2], truncation_radius=TRUNCATION)
    assert np.array_equal(reference.conservative_distance,
                          strided.conservative_distance)


@requires_torch_cuda
def test_depth_images_may_be_on_the_device():
    k = intrinsics()
    scene = AnalyticScene.forest(3, 14, 10, 8.0)
    poses = orbit_poses(4, 6.0, 1.5)
    rng = np.random.default_rng(1)
    query = np.column_stack([rng.uniform(-3, 3, 64), rng.uniform(-3, 3, 64),
                             rng.uniform(0, 3, 64)]).astype(np.float32)

    answers = []
    for on_device in (False, True):
        buffer = CudaDepthBuffer(k, params())
        for i, pose in enumerate(poses):
            depth = scene.render(k, pose)
            if on_device:
                depth = torch.as_tensor(depth).cuda()
            buffer.set_posed_depth_image(depth, np.asarray(pose, np.float32),
                                         stamp_ns=i, is_keyframe=False)
        answers.append(buffer.query(query, truncation_radius=TRUNCATION))
    assert np.array_equal(answers[0].conservative_distance,
                          answers[1].conservative_distance)


# The device path itself, without any framework: DeviceArray above exposes the
# same protocol torch and cupy do.

requires_cuda_runtime = pytest.mark.skipif(
    _cudart is None, reason="the CUDA runtime library was not found")


@requires_cuda_runtime
def test_device_queries_match_host_queries(filled, points):
    buffer, _, _ = filled
    reference = buffer.query(points, truncation_radius=TRUNCATION)
    on_device = buffer.query(DeviceArray(points), truncation_radius=TRUNCATION)
    assert np.array_equal(reference.conservative_distance,
                          on_device.conservative_distance)
    assert np.array_equal(reference.optimistic_distance,
                          on_device.optimistic_distance)
    assert np.array_equal(reference.conservative_closest,
                          on_device.conservative_closest)


@requires_cuda_runtime
def test_device_depth_images_match_host_depth_images():
    k = intrinsics()
    scene = AnalyticScene.forest(3, 14, 10, 8.0)
    poses = orbit_poses(4, 6.0, 1.5)
    rng = np.random.default_rng(1)
    query = np.column_stack([rng.uniform(-3, 3, 64), rng.uniform(-3, 3, 64),
                             rng.uniform(0, 3, 64)]).astype(np.float32)

    answers = []
    for on_device in (False, True):
        buffer = CudaDepthBuffer(k, params())
        for i, pose in enumerate(poses):
            depth = scene.render(k, pose)
            buffer.set_posed_depth_image(
                DeviceArray(depth) if on_device else depth,
                np.asarray(pose, np.float32), stamp_ns=i, is_keyframe=False)
        answers.append(buffer.query(query, truncation_radius=TRUNCATION))
    assert np.array_equal(answers[0].conservative_distance,
                          answers[1].conservative_distance)


@requires_cuda_runtime
def test_unusable_device_arrays_say_why_without_torch(filled, points):
    class Float64DeviceArray(DeviceArray):
        @property
        def __cuda_array_interface__(self):
            interface = DeviceArray.__cuda_array_interface__.fget(self)
            interface["typestr"] = "<f8"
            return interface

    buffer, _, _ = filled
    with pytest.raises(ValueError, match="not float32"):
        buffer.query(Float64DeviceArray(points))
