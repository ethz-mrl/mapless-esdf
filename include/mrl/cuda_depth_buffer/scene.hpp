#ifndef MRL_CUDA_DEPTH_BUFFER_SCENE_HPP
#define MRL_CUDA_DEPTH_BUFFER_SCENE_HPP

#include <array>
#include <cstdint>
#include <mrl/cuda_depth_buffer/buffer_params.hpp>
#include <mrl/cuda_depth_buffer/depth_buffer.hpp>
#include <vector>

/// Analytic scenes and the depth images they produce.
///
/// The buffer is fed by a depth sensor, but a test, an example or a benchmark
/// should not need a recorded dataset to say something meaningful. Scenes here
/// are a handful of primitives that a camera ray can be intersected with in
/// closed form, so a depth image is generated rather than loaded, the geometry
/// is known exactly, and the resulting distances can be checked against an
/// exhaustive search.
///
/// World convention: z up, distances in meters. Camera poses are optical frame
/// (x right, y down, z forward), which is what the buffer expects.
namespace mrl {

/// x, y, z, qx, qy, qz, qw of a camera in the reference frame.
using Pose = std::array<float, 7>;

struct Sphere {
  std::array<float, 3> center{{0.0f, 0.0f, 0.0f}};
  float radius = 1.0f;
};

/// Finite cylinder from `base` along `axis` for `height` meters, capped at both
/// ends. Upright cylinders read as tree trunks, which is the case the query has
/// to be good at.
struct Cylinder {
  std::array<float, 3> base{{0.0f, 0.0f, 0.0f}};
  std::array<float, 3> axis{{0.0f, 0.0f, 1.0f}};
  float radius = 0.2f;
  float height = 4.0f;
};

/// Infinite plane through `point` with unit `normal`, for ground and walls.
struct Plane {
  std::array<float, 3> point{{0.0f, 0.0f, 0.0f}};
  std::array<float, 3> normal{{0.0f, 0.0f, 1.0f}};
};

struct PosedDepthImage {
  std::vector<float> depth;  ///< row major, width * height
  Pose pose{{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}};
  int64_t stamp_ns = 0;
};

class AnalyticScene {
 public:
  std::vector<Sphere> spheres;
  std::vector<Cylinder> cylinders;
  std::vector<Plane> planes;

  /// Cylinders standing on the ground plane with spheres scattered between
  /// them. Both shapes are deliberately present: an extruded scene is constant
  /// along its axis, and a pruning defect that only loses candidates along that
  /// axis stays invisible in it.
  ///
  /// @param seed makes the layout reproducible
  /// @param extent side length of the square the scene occupies, in meters
  static AnalyticScene forest(uint32_t seed,
                              int num_cylinders = 24,
                              int num_spheres = 16,
                              float extent = 12.0f,
                              bool with_ground = true);

  /// Depth of the scene as seen from `pose`, clipped to the intrinsics' range.
  /// Rays that hit nothing come back at z_far.
  std::vector<float> render(const CameraIntrinsics& intrinsics,
                            const Pose& pose) const;

  PosedDepthImage renderPosed(const CameraIntrinsics& intrinsics,
                              const Pose& pose,
                              int64_t stamp_ns = 0) const;

  /// Distance from a point to the nearest surface of the scene geometry.
  ///
  /// This is NOT the ground truth for a query. The buffer measures distance to
  /// the nearest point back projected from a depth pixel, which differs from
  /// the continuous surface by up to the metric footprint of a pixel, and it
  /// knows nothing about surfaces no camera saw. Use bruteForceDistance for
  /// that.
  float distanceToSurface(const std::array<float, 3>& point) const;
};

/// Camera pose looking from `eye` towards `target`, in the optical convention.
Pose lookAt(const std::array<float, 3>& eye,
            const std::array<float, 3>& target,
            const std::array<float, 3>& up = {{0.0f, 0.0f, 1.0f}});

/// `count` poses on a circle of `radius` at `height`, all looking at `target`.
/// This is the sweep the demos use to fill a buffer from several viewpoints.
std::vector<Pose> orbitPoses(int count,
                             float radius,
                             float height,
                             const std::array<float, 3>& target = {
                                 {0.0f, 0.0f, 0.0f}});

/// Both channels of a query, evaluated exhaustively. O(frames * pixels) per
/// query, so this is an oracle for small images, not a fallback.
///
/// It is the definition written out: the optimistic distance is the nearest
/// point back projected from any pixel of any frame, and the conservative one
/// additionally considers the boundary of unknown space, modelled as segments
/// on the pixel rays between a pixel's depth and its right and down neighbours,
/// plus the near clipping plane, with image border pixels reading their
/// neighbour depth as z_near so that the frustum walls are boundaries too. The
/// conservative distance is signed and merged across frames by taking the
/// maximum, so free space seen in any frame overrides unknown space in another.
struct BruteForceDistances {
  float optimistic = 0.0f;
  float conservative = 0.0f;
};

BruteForceDistances bruteForceDistances(
    const CameraIntrinsics& intrinsics,
    const std::vector<PosedDepthImage>& frames,
    const std::array<float, 3>& query,
    float truncation_radius,
    PoolCompensation mode);
/// The pre-enum form: true is the exact patch, false the point sample.
BruteForceDistances bruteForceDistances(
    const CameraIntrinsics& intrinsics,
    const std::vector<PosedDepthImage>& frames,
    const std::array<float, 3>& query,
    float truncation_radius,
    bool pixel_extent = false);

/// Optimistic channel only, for callers that do not need the other.
float bruteForceDistance(const CameraIntrinsics& intrinsics,
                         const std::vector<PosedDepthImage>& frames,
                         const std::array<float, 3>& query,
                         float truncation_radius,
                         PoolCompensation mode);
float bruteForceDistance(const CameraIntrinsics& intrinsics,
                         const std::vector<PosedDepthImage>& frames,
                         const std::array<float, 3>& query,
                         float truncation_radius,
                         bool pixel_extent = false);

}  // namespace mrl

#endif  // MRL_CUDA_DEPTH_BUFFER_SCENE_HPP
