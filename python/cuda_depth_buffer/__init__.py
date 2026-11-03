"""Sensor space ESDF queries over a ring buffer of posed depth images.

The buffer answers distance queries directly from depth images and their poses,
without building a map. Two answers come back for every point: the *optimistic*
one treats space no camera has seen as free, the *conservative* one treats it as
occupied. A planner usually wants both.

    import numpy as np
    from cuda_depth_buffer import BufferParams, CameraIntrinsics, CudaDepthBuffer

    intrinsics = CameraIntrinsics(fx=400, fy=400, cx=319.5, cy=239.5,
                                  width=640, height=480, z_near=0.2, z_far=20.0)
    buffer = CudaDepthBuffer(intrinsics, BufferParams(num_frames=20))
    buffer.set_posed_depth_image(depth, pose, stamp_ns=0)

    result = buffer.query(points, conservative=True, truncation_radius=2.5)
    result.conservative_distance   # (N,)
    result.conservative_gradient   # (N, 3)

Depth images and query points may be numpy arrays or any CUDA array exposing
``__cuda_array_interface__`` (torch cuda tensors, cupy arrays), which are read
where they are instead of being copied to the host first. Results always come
back as numpy.
"""

from dataclasses import dataclass

import numpy as np

from ._core import (
    AnalyticScene,
    BufferParams,
    CameraIntrinsics,
    Cylinder,
    Plane,
    PosedDepthImage,
    Sphere,
    brute_force_distances,
    look_at,
    orbit_poses,
    profiling,
)
from ._core import CudaDepthBuffer as _CudaDepthBuffer

__all__ = [
    "AnalyticScene",
    "BufferParams",
    "CameraIntrinsics",
    "CudaDepthBuffer",
    "Cylinder",
    "Plane",
    "PosedDepthImage",
    "QueryResult",
    "Sphere",
    "brute_force_distances",
    "look_at",
    "orbit_poses",
    "profiling",
]


@dataclass(frozen=True)
class QueryResult:
    """Both channels of a batch of queries, in the frame the poses are in.

    A distance saturated at the truncation radius has no closest point, so its
    point and gradient are NaN. That is how "nothing within the truncation
    radius" is told apart from a real answer at that distance. When a query did
    not ask for conservative handling, the three conservative arrays are NaN.
    """

    optimistic_distance: np.ndarray
    optimistic_gradient: np.ndarray
    optimistic_closest: np.ndarray
    conservative_distance: np.ndarray
    conservative_gradient: np.ndarray
    conservative_closest: np.ndarray

    def distance(self, conservative: bool = True) -> np.ndarray:
        return self.conservative_distance if conservative else self.optimistic_distance

    def gradient(self, conservative: bool = True) -> np.ndarray:
        return self.conservative_gradient if conservative else self.optimistic_gradient

    def closest(self, conservative: bool = True) -> np.ndarray:
        return self.conservative_closest if conservative else self.optimistic_closest


class CudaDepthBuffer(_CudaDepthBuffer):
    """Ring buffer of posed depth images. See the module docstring.

    Thread safe: queries run concurrently with one another, and an insert
    excludes queries for its duration.
    """

    def query(self, points, conservative=True, truncation_radius=float("inf"),
              suppression_radius=0.0, verbose=False) -> QueryResult:
        """Signed distance, gradient and closest point of every query point.

        `points` is (N, 3) or a flat xyz interleaved array, on the host or on the
        device. Distances are positive in free space and, in the conservative
        channel, negative inside unknown space.
        """
        return QueryResult(**super().query(
            points,
            conservative=conservative,
            truncation_radius=truncation_radius,
            suppression_radius=suppression_radius,
            verbose=verbose,
        ))
