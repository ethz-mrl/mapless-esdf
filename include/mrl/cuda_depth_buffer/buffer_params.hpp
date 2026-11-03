#ifndef MRL_CUDA_DEPTH_BUFFER_BUFFER_PARAMS_HPP
#define MRL_CUDA_DEPTH_BUFFER_BUFFER_PARAMS_HPP

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>

namespace mrl {

/// Everything needed to construct and drive a depth buffer, in one place, so
/// that the C++ examples, the Python bindings and the ROS node all read the
/// same configuration file.
/// See BufferParams::pool_compensation.
enum class PoolCompensation {
  /// The pixel is the point sample on its ray at the stored depth. Optimistic
  /// between samples by up to half a patch diagonal. The legacy behaviour.
  kNone,
  /// The pixel is the patch [u-1/2, u+1/2) x [v-1/2, v+1/2) at the stored
  /// depth and the candidate is the query clamped into it: the exact nearest
  /// point of the patch, a lower bound on the distance to any surface in the
  /// pooled block. Two clamps per candidate pixel.
  kExact,
  /// The pixel is the point sample, searched as in kNone, and half the pooled
  /// footprint at the winning pixel's depth (0.5 * depth / fx) is taken off
  /// the distance afterwards - once per result instead of two clamps per
  /// candidate. Equal to kExact for a query off the ray along a pixel axis,
  /// optimistic by up to (sqrt(2) - 1)/2 footprints towards a patch corner,
  /// conservative by up to half a footprint in front of the sample. Occlusion
  /// candidates are ray segments and get no margin.
  kApproximate
};

const char* toString(PoolCompensation mode);
/// Parses "none", "exact" or "approximate"; throws std::invalid_argument.
PoolCompensation poolCompensationFromString(const std::string& name);

struct BufferParams {
  /// Number of posed depth images kept in the ring buffer.
  size_t num_frames = 10;
  /// Reserve slot 0 for the latest frame and rotate keyframes through the rest.
  bool use_keyframing = false;
  /// Min pooling kernel size in pixels applied on insert, 1 disables pooling.
  uint16_t min_pool_size = 1;
  /// How the closest-surface search accounts for a pixel's extent. A stored
  /// (pooled) pixel u, v stands for a patch of the image plane half a pixel
  /// wide on each side of its ray; with min pooling its depth is the block's
  /// nearest, so a surface that fills the block can be anywhere in that patch,
  /// and the distance to the pixel's point sample over-estimates the distance
  /// to the surface by up to half a patch diagonal. The choice is charged per
  /// candidate pixel in the inner loop of the search, which is why it exists.
  /// Default kApproximate (9 Sept 2026): against the exact patch it is within a
  /// millimetre in the conservative channel and 3-16 % cheaper per query.
  PoolCompensation pool_compensation = PoolCompensation::kApproximate;
  /// Multiplied onto the raw depth values, e.g. 0.001 for millimeter input.
  float depth_scale_factor = 1.0f;
  /// Map invalid depth (NaN or <= 0) to the far plane instead of the near one.
  bool invalid_is_farplane = false;
  /// Radius of the sphere treated as free around the robot center of every
  /// buffered frame, on the grounds that the robot occupied that volume when
  /// the image was taken. 0 disables it.
  float robot_radius = 0.0f;
  /// Robot center expressed in the camera frame, in meters.
  float robot_center_in_camera[3] = {0.0f, 0.0f, 0.0f};

  /// Thresholds evaluated by CudaDepthBuffer::isKeyframe. They are deliberately
  /// not applied automatically: a caller with a better criterion can ignore
  /// them and pass its own decision to setPosedDepthImage.
  float keyframe_dist_threshold = 0.5f;   ///< meters
  float keyframe_angle_threshold = 0.1f;  ///< radians
  float keyframe_time_threshold = 1.0f;   ///< seconds

  /// Sigma multiplier c of the pose uncertainty margin. Obstacles seen by a
  /// frame are inflated by c standard deviations of that frame's pose
  /// uncertainty, which limits how far a frame with an uncertain pose can erode
  /// them. The default of 4 is the value used in the paper. A frame without a
  /// covariance never contributes a margin, so supplying covariances is what
  /// turns the compensation on; 0 disables it outright.
  float pose_covariance_confidence = 4.0f;

  /// Reads the file, leaving unspecified fields at their default. Throws
  /// std::runtime_error if the file cannot be opened or parsed.
  static BufferParams fromYamlFile(const std::string& filename);
  /// Same, from a YAML document held in memory.
  static BufferParams fromYaml(const std::string& yaml);
  std::string toYaml() const;
};

std::ostream& operator<<(std::ostream& os, const BufferParams& params);

}  // namespace mrl

#endif  // MRL_CUDA_DEPTH_BUFFER_BUFFER_PARAMS_HPP
