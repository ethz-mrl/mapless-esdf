#ifndef MRL_CUDA_DEPTH_BUFFER_DEPTH_BUFFER_HPP
#define MRL_CUDA_DEPTH_BUFFER_DEPTH_BUFFER_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mrl/cuda_depth_buffer/buffer_params.hpp>
#include <vector>

namespace mrl {

/// Pinhole intrinsics of the depth stream, with the clipping range that bounds
/// every stored depth.
struct CameraIntrinsics {
  float cx = 0.0f;
  float cy = 0.0f;
  float fx = 0.0f;
  float fy = 0.0f;
  /// Both must be in [1, 8191]: a candidate pixel and its squared distance to
  /// the query share one 64 bit key reduced with a single atomicMin, which
  /// leaves 13 bits per coordinate.
  uint16_t width = 0;
  uint16_t height = 0;
  float z_near = 0.0f;
  float z_far = 0.0f;
};

/// Answer of one batched query, in the reference frame the queries came in.
///
/// Both handlings of unknown space are always reported, because planners
/// commonly need both: the optimistic channel treats unknown space as free, the
/// conservative one as occupied. Producing the optimistic channel costs nothing
/// extra, it is the state of the search before the occlusion pass runs. When a
/// query does not ask for conservative handling, its three conservative vectors
/// are filled with NaN rather than left empty, so the layout never changes.
///
/// A distance saturated at the truncation radius has no closest point, and its
/// point and gradient are NaN. That is how a caller distinguishes "nothing
/// within the truncation radius" from a real answer at that distance.
struct QueryResults {
  uint32_t num_queries = 0;

  std::vector<float> optimistic_distance;  ///< num_queries
  std::vector<float> optimistic_gradient;  ///< 3 * num_queries, xyz
  std::vector<float> optimistic_closest;   ///< 3 * num_queries, xyz

  std::vector<float> conservative_distance;  ///< num_queries
  std::vector<float> conservative_gradient;  ///< 3 * num_queries, xyz
  std::vector<float> conservative_closest;   ///< 3 * num_queries, xyz

  void resize(uint32_t n);
};

/// Per frame distance channels, filled only when a query asks for them. Each
/// vector is num_frames * num_queries, frame major.
struct DebugDistances {
  uint32_t num_frames = 0;
  uint32_t num_queries = 0;
  std::vector<float> merged;
  std::vector<float> unknown;
  std::vector<float> occupied;
  std::vector<float> robot;
  std::vector<float> pose_compensation;
};

/// Ring buffer of posed depth images answering signed distance queries directly
/// in sensor space, without building a map.
///
/// Capacity, image geometry and pooling are fixed at construction; everything
/// else comes from BufferParams and the setters below.
///
/// Thread safe. Queries run concurrently with each other, each on its own
/// stream and its own scratch memory, so they overlap on the device. Inserting
/// an image or revising a pose excludes all queries for its duration, since it
/// overwrites the very slots a query in flight would read. A typical stack has
/// one thread feeding images and several planning threads querying; that
/// arrangement never blocks a query on another query.
class CudaDepthBuffer {
 public:
  CudaDepthBuffer(const CameraIntrinsics& intrinsics,
                  const BufferParams& params);
  ~CudaDepthBuffer();

  // Owns device allocations; copying would double free.
  CudaDepthBuffer(const CudaDepthBuffer&) = delete;
  CudaDepthBuffer& operator=(const CudaDepthBuffer&) = delete;

  /// By value: the parameters can be retuned from another thread.
  CameraIntrinsics intrinsics() const;
  BufferParams params() const;

  /// Inserts a depth image with the pose of its camera. A pose is mandatory at
  /// insert time; a covariance is not, and can be supplied or revised later
  /// through updatePoseAndCovarianceByTimestamp.
  ///
  /// @param depth_data row major, intrinsics().width * height floats, in
  /// host or device memory
  /// @param pose x, y, z, qx, qy, qz, qw of the camera in the reference frame
  /// @param stamp_ns identifies the frame for later pose or covariance updates
  /// @param is_keyframe with keyframing enabled, keyframes rotate through slots
  /// 1..N-1 while slot 0 always holds the latest frame
  void setPosedDepthImage(const float* depth_data,
                          const float* pose,
                          int64_t stamp_ns,
                          bool is_keyframe = true);

  /// Whether `pose` at `stamp_ns` differs enough from the most recent keyframe
  /// to become one, according to the keyframe_* thresholds in BufferParams.
  ///
  /// Deliberately separate from setPosedDepthImage: a caller with a better
  /// criterion (view overlap, information gain) ignores this and passes its own
  /// decision. Returns true when there is no keyframe yet.
  bool isKeyframe(const float* pose, int64_t stamp_ns) const;

  void setDepthScaleFactor(float depth_scale_factor);
  void setRobotRadius(float robot_radius);
  void setPoseCovarianceConfidence(float confidence);

  /// Revises the pose and/or covariance of a buffered frame, newest or oldest
  /// alike, as a SLAM backend refines its estimate. Either argument may be null
  /// to leave that half alone. Returns false when no frame carries that
  /// timestamp, or when both arguments are null.
  bool updatePoseAndCovarianceByTimestamp(int64_t stamp_ns,
                                          const float* pose,
                                          const double* covariance);
  bool getCovarianceByTimestamp(int64_t stamp_ns, double* covariance_out) const;

  /// Signed distance, gradient and closest point for a batch of query points.
  ///
  /// Query points and every returned point are expressed in the same reference
  /// frame as the poses passed to setPosedDepthImage, whatever that frame is
  /// (world, map, odom). Nothing here is in a camera frame.
  ///
  /// @param queries num_queries * 3 floats, xyz interleaved, in host or
  /// device memory. Results are always written to the host.
  /// @param conservative also run the occlusion boundary pass and fill the
  /// conservative channel; when false those outputs are NaN
  /// @param truncation_radius stop searching beyond this distance
  /// @param verbose time the individual stages, which costs a device sync per
  /// stage
  /// @param suppression_radius drop queries within this radius of a lower
  /// indexed one, 0 disables. Suppressed queries come back as NaN
  /// @param debug_distances when non null, also return the per frame channels
  void query(const float* queries,
             uint32_t num_queries,
             bool conservative,
             QueryResults& results,
             float truncation_radius = std::numeric_limits<float>::infinity(),
             bool verbose = false,
             float suppression_radius = 0.0f,
             DebugDistances* debug_distances = nullptr);

  /// Convenience wrapper allocating its own QueryResults.
  QueryResults query(
      const std::vector<float>& queries,
      bool conservative,
      float truncation_radius = std::numeric_limits<float>::infinity());

  /// Compiles the kernels and touches the device so that the first real call is
  /// not charged for it.
  void warmup();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mrl

#endif  // MRL_CUDA_DEPTH_BUFFER_DEPTH_BUFFER_HPP
