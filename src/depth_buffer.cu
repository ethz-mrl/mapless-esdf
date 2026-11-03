#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mrl/cuda_depth_buffer/depth_buffer.hpp>
#include <mrl/cuda_depth_buffer/profiling.hpp>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "kernels/bounds.cuh"
#include "kernels/common.cuh"
#include "kernels/integration.cuh"
#include "kernels/merge.cuh"
#include "kernels/occlusion_search.cuh"
#include "kernels/pose_compensation.cuh"
#include "kernels/query_prepare.cuh"
#include "kernels/surface_search.cuh"

namespace mrl {

using Matrix6dRowMajor = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>;

namespace {

/// Reader/writer lock that never starves the writer.
///
/// std::shared_mutex is reader preferring on the common implementations, so a
/// steady stream of queries can hold off an insert indefinitely. For a depth
/// buffer that is the wrong way round: queries are frequent and individually
/// unimportant, while a blocked insert means the robot stops seeing. Once a
/// writer is waiting, no further reader is admitted.
class WriterPriorityLock {
 public:
  void lockShared() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock,
                    [this] { return !writing_ && writers_waiting_ == 0; });
    ++readers_;
  }

  void unlockShared() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      --readers_;
    }
    condition_.notify_all();
  }

  void lock() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++writers_waiting_;
    condition_.wait(lock, [this] { return !writing_ && readers_ == 0; });
    --writers_waiting_;
    writing_ = true;
  }

  void unlock() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      writing_ = false;
    }
    condition_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  int readers_ = 0;
  int writers_waiting_ = 0;
  bool writing_ = false;
};

/// Scope guards, so that an early return or a throw cannot leave the lock held.
class SharedGuard {
 public:
  explicit SharedGuard(WriterPriorityLock& lock) : lock_(lock) {
    lock_.lockShared();
  }
  ~SharedGuard() { lock_.unlockShared(); }
  SharedGuard(const SharedGuard&) = delete;
  SharedGuard& operator=(const SharedGuard&) = delete;

 private:
  WriterPriorityLock& lock_;
};

class ExclusiveGuard {
 public:
  explicit ExclusiveGuard(WriterPriorityLock& lock) : lock_(lock) {
    lock_.lock();
  }
  ~ExclusiveGuard() { lock_.unlock(); }
  ExclusiveGuard(const ExclusiveGuard&) = delete;
  ExclusiveGuard& operator=(const ExclusiveGuard&) = delete;

 private:
  WriterPriorityLock& lock_;
};

}  // namespace

/// Everything behind the public facade: the device buffers, the ring buffer
/// bookkeeping and the query orchestration.
struct CudaDepthBuffer::Impl {
  Impl(const CameraIntrinsics& intrinsics, const BufferParams& params);
  ~Impl();
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  void setPosedDepthImage(const float* depth_data,
                          const float* pose_data,
                          int64_t stamp_ns,
                          bool is_keyframe);
  bool isKeyframe(const float* pose, int64_t stamp_ns) const;
  void setDepthScaleFactor(float depth_scale_factor);
  void setRobotRadius(float robot_radius);
  bool updatePoseAndCovarianceByTimestamp(int64_t stamp_ns,
                                          const float* pose,
                                          const double* covariance);
  bool getCovarianceByTimestamp(int64_t stamp_ns, double* covariance_out) const;
  void query(const float* queries,
             uint32_t num_queries,
             bool conservative,
             QueryResults& results,
             float truncation_radius,
             bool verbose,
             float suppression_radius,
             DebugDistances* debug_distances);
  void warmup();

  // Tile size of the per block depth ranges. Changing it means changing the
  // kernels, hence the static assert against the kernel side constants.
  static constexpr int TILE_W = kTileW;
  static constexpr int TILE_H = kTileH;
  static constexpr int64_t kInvalidFrameTimestampNs = -1;

  CameraIntrinsics intrinsics_;
  BufferParams params_;

  float* d_depth_buffer_ = nullptr;
  float* d_poses_ = nullptr;
  float* d_min_block_depths_ = nullptr;
  float* d_max_block_depths_ = nullptr;
  float* d_min_macro_depths_ = nullptr;  // per-macro-tile reduction of the
  float* d_max_macro_depths_ = nullptr;  // block min/max (coarse cull level)
  float* d_min_pool_input_ = nullptr;
  // Mirrors of the covariance state, refreshed under the exclusive lock so that
  // a query only ever reads a settled copy.
  double* d_covariances_ = nullptr;      // F x 36
  uint8_t* d_has_covariance_ = nullptr;  // F
  void uploadCovarianceSlot_(size_t i_frame);
  std::vector<float> h_poses_;

  // Grown on demand and reused across calls, so that a query never allocates.
  struct QueryWorkspace {
    cudaStream_t stream = nullptr;
    uint32_t capacity_queries = 0;
    float* d_queries_frame = nullptr;              // NQ x 3
    float* d_queries_frames = nullptr;             // NQ*F x 3
    Pixel* d_queries_frames_projected = nullptr;   // NQ*F
    AABB2D* d_queries_aabbs = nullptr;             // NQ*F
    PackedResult* d_results = nullptr;             // NQ*F
    PackedResult* d_results_optimistic = nullptr;  // NQ*F, surface only
    float* d_closest_points_world = nullptr;       // NQ*F x 3
    float* d_res_out = nullptr;                    // 14 x NQ, both channels
    float* d_pose_compensation = nullptr;          // NQ*F
    float* d_frame_query_truncation = nullptr;     // NQ*F
    float* d_robot_dists = nullptr;                // NQ*F
    uint32_t* d_filter_flags = nullptr;            // ceil(NQ/32)
    float* h_pinned_queries = nullptr;             // NQ x 3 (pinned staging)
    float* h_pinned_results = nullptr;             // 14 x NQ (pinned staging)
  };
  // One workspace per concurrently running query, each with its own stream, so
  // that two queries never share scratch memory and never serialize on the
  // device. The pool grows to the observed concurrency and is never shrunk.
  std::mutex pool_mutex_;
  std::vector<std::unique_ptr<QueryWorkspace>> workspaces_;
  std::vector<QueryWorkspace*> free_workspaces_;

  /// Borrows a workspace for the lifetime of the object and returns it after.
  class WorkspaceLease {
   public:
    WorkspaceLease(Impl& impl, uint32_t num_queries);
    ~WorkspaceLease();
    WorkspaceLease(const WorkspaceLease&) = delete;
    WorkspaceLease& operator=(const WorkspaceLease&) = delete;
    QueryWorkspace& operator*() const { return *ws_; }

   private:
    Impl& impl_;
    QueryWorkspace* ws_;
  };

  void ensureQueryWorkspace_(QueryWorkspace& ws, uint32_t num_queries);
  void freeQueryWorkspace_(QueryWorkspace& ws);

  // Inserts take the lock exclusively, queries share it. Both hold it until
  // their device work has completed, so the depth buffer is never read while an
  // insert is overwriting it.
  mutable WriterPriorityLock buffer_mutex_;
  cudaStream_t insert_stream_ = nullptr;
  void freeDeviceBuffers_();
  void clearCovarianceSlot_(size_t i_frame);
  void updatePoseSlot_(size_t i_frame, const float* pose);

  size_t num_frames_;
  size_t current_index_;
  size_t next_keyframe_index_;
  bool use_keyframing_;
  uint16_t num_rows_;
  uint16_t num_cols_;
  uint16_t raw_num_rows_ = 0;
  uint16_t raw_num_cols_ = 0;
  uint16_t min_pool_size_ = 1;
  size_t grid_dim_x_;
  size_t grid_dim_y_;
  size_t macro_dim_x_ = 0;
  size_t macro_dim_y_ = 0;
  int64_t last_keyframe_stamp_ns_ = kInvalidFrameTimestampNs;
  std::vector<float> last_keyframe_pose_;
  std::vector<int64_t> frame_timestamps_ns_;
  std::vector<double> h_covariances_;
  std::vector<uint8_t> frame_has_covariance_;

  // Robot center expressed in the camera frame; the clearing sphere sits there
  // in every buffered frame.
  float robot_center_x_ = 0.0f, robot_center_y_ = 0.0f, robot_center_z_ = 0.0f;
  float robot_radius_ = 0.0f;
  bool invalid_is_farplane_ = false;
  PoolCompensation pool_compensation_ =
      PoolCompensation::kExact;  // BufferParams::pool_compensation
  bool exact_patch_ = true;      // kExact: clamp into the patch per candidate
  bool approximate_margin_ =
      false;  // kApproximate: half a footprint off the winner
  float depth_scale_factor_ = 1.0f;
  float pose_covariance_confidence_ = 4.0f;

  float fx_, fy_, cx_, cy_, z_near_, z_far_;
};

CudaDepthBuffer::Impl::Impl(const CameraIntrinsics& intrinsics,
                            const BufferParams& params)
    : intrinsics_(intrinsics),
      params_(params),
      num_frames_(params.num_frames),
      current_index_{0},
      next_keyframe_index_{params.use_keyframing && params.num_frames > 1
                               ? size_t{1}
                               : size_t{0}},
      use_keyframing_(params.use_keyframing),
      num_rows_(intrinsics.height),
      num_cols_(intrinsics.width),
      raw_num_rows_(intrinsics.height),
      raw_num_cols_(intrinsics.width),
      min_pool_size_(params.min_pool_size),
      pool_compensation_(params.pool_compensation),
      exact_patch_(params.pool_compensation == PoolCompensation::kExact),
      approximate_margin_(params.pool_compensation ==
                          PoolCompensation::kApproximate),
      grid_dim_x_(0),
      grid_dim_y_(0),
      frame_timestamps_ns_(params.num_frames, kInvalidFrameTimestampNs),
      h_covariances_(params.num_frames * 36, 0.0),
      frame_has_covariance_(params.num_frames, 0),
      cx_(intrinsics.cx),
      cy_(intrinsics.cy),
      fx_(intrinsics.fx),
      fy_(intrinsics.fy),
      z_near_(intrinsics.z_near),
      z_far_(intrinsics.z_far),
      robot_radius_(params.robot_radius),
      robot_center_x_(params.robot_center_in_camera[0]),
      robot_center_y_(params.robot_center_in_camera[1]),
      robot_center_z_(params.robot_center_in_camera[2]),
      invalid_is_farplane_(params.invalid_is_farplane),
      depth_scale_factor_(params.depth_scale_factor),
      pose_covariance_confidence_(params.pose_covariance_confidence) {
  const size_t num_rows = intrinsics.height;
  const size_t num_cols = intrinsics.width;
  MRL_CUDA_CHECK(cudaStreamCreate(&insert_stream_));
  MRL_CUDA_CHECK(
      cudaMalloc(&d_covariances_, num_frames_ * 36 * sizeof(double)));
  MRL_CUDA_CHECK(cudaMalloc(&d_has_covariance_, num_frames_ * sizeof(uint8_t)));
  MRL_CUDA_CHECK(
      cudaMemset(d_covariances_, 0, num_frames_ * 36 * sizeof(double)));
  MRL_CUDA_CHECK(
      cudaMemset(d_has_covariance_, 0, num_frames_ * sizeof(uint8_t)));
  // Any allocation failure past this point must not leak the buffers
  // allocated before it: the destructor does not run for a throwing
  // constructor.
  try {
    if (num_frames_ == 0) {
      throw std::invalid_argument("Number of frames must be greater than 0");
    }
    if (min_pool_size_ == 0) {
      throw std::invalid_argument("min_pool_size must be >= 1");
    }
    if (num_rows == 0 || num_rows >= PackedResult::maxNumRows()) {
      throw std::invalid_argument("Image rows must be in (0, " +
                                  std::to_string(PackedResult::maxNumRows()) +
                                  ")");
    }
    if (num_cols == 0 || num_cols >= PackedResult::maxNumCols()) {
      throw std::invalid_argument("Image cols must be in (0, " +
                                  std::to_string(PackedResult::maxNumCols()) +
                                  ")");
    }
    if (!std::isfinite(pose_covariance_confidence_) ||
        pose_covariance_confidence_ < 0.0f) {
      throw std::invalid_argument(
          "pose_covariance_confidence must be finite and >= 0");
    }
    raw_num_rows_ = static_cast<uint16_t>(num_rows);
    raw_num_cols_ = static_cast<uint16_t>(num_cols);
    num_rows_ = raw_num_rows_;
    num_cols_ = raw_num_cols_;
    if (min_pool_size_ != 1) {
      const size_t pool = static_cast<size_t>(min_pool_size_);
      const size_t pooled_rows =
          (static_cast<size_t>(raw_num_rows_) + pool - 1) / pool;
      const size_t pooled_cols =
          (static_cast<size_t>(raw_num_cols_) + pool - 1) / pool;
      if (pooled_rows == 0 ||
          pooled_rows >= static_cast<size_t>(PackedResult::maxNumRows())) {
        throw std::invalid_argument("Pooled image rows must be in (0, " +
                                    std::to_string(PackedResult::maxNumRows()) +
                                    ")");
      }
      if (pooled_cols == 0 ||
          pooled_cols >= static_cast<size_t>(PackedResult::maxNumCols())) {
        throw std::invalid_argument("Pooled image cols must be in (0, " +
                                    std::to_string(PackedResult::maxNumCols()) +
                                    ")");
      }
      num_rows_ = static_cast<uint16_t>(pooled_rows);
      num_cols_ = static_cast<uint16_t>(pooled_cols);
      // Pooled pixel u stands for the block of full-resolution pixels
      // u * pool .. u * pool + pool - 1, whose rays are centred on the pixel
      // u * pool + (pool - 1) / 2. Dividing the principal point by the pool
      // alone would put the pooled sample, the patch, the sign test's pixel
      // and the occlusion rays on the block's first pixel instead, half a
      // block off at coarse pools (17 cm at 5 m for pool 37).
      const float scale = static_cast<float>(pool);
      fx_ /= scale;
      fy_ /= scale;
      cx_ = (cx_ - 0.5f * (scale - 1.0f)) / scale;
      cy_ = (cy_ - 0.5f * (scale - 1.0f)) / scale;
    }

    // Allocate GPU memory for depth buffer and initialize to infinity
    size_t depth_buffer_size =
        num_rows_ * num_cols_ * num_frames_ * sizeof(float);
    MRL_CUDA_CHECK(cudaMalloc(&d_depth_buffer_, depth_buffer_size));
    size_t num_elements = num_rows_ * num_cols_ * num_frames_;
    std::vector<float> h_init_buffer(num_elements, INFINITY);
    MRL_CUDA_CHECK(cudaMemcpy(d_depth_buffer_,
                              h_init_buffer.data(),
                              depth_buffer_size,
                              cudaMemcpyHostToDevice));

    if (min_pool_size_ != 1) {
      size_t raw_depth_size =
          static_cast<size_t>(raw_num_rows_) * raw_num_cols_ * sizeof(float);
      MRL_CUDA_CHECK(cudaMalloc(&d_min_pool_input_, raw_depth_size));
    }

    // Poses are stored as (x, y, z, qx, qy, qz, qw) per image, mirrored on the
    // host so that pose compensation can run without a device round trip.
    const size_t poses_size = num_frames_ * 7 * sizeof(float);
    h_poses_.assign(num_frames_ * 7, 0.0f);
    MRL_CUDA_CHECK(cudaMalloc(&d_poses_, poses_size));
    MRL_CUDA_CHECK(cudaMemset(d_poses_, 0, poses_size));

    // Allocate memory for min/max depth per block in each frame
    static_assert(TILE_W == kTileW && TILE_H == kTileH,
                  "tiled query kernels assume the insert-time tile size");
    grid_dim_x_ = (num_cols_ + TILE_W - 1) / TILE_W;
    grid_dim_y_ = (num_rows_ + TILE_H - 1) / TILE_H;
    size_t num_blocks = grid_dim_x_ * grid_dim_y_;
    size_t blocks_size = num_frames_ * num_blocks * sizeof(float);
    MRL_CUDA_CHECK(cudaMalloc(&d_min_block_depths_, blocks_size));
    MRL_CUDA_CHECK(cudaMalloc(&d_max_block_depths_, blocks_size));
    // Coarse level of the query culling hierarchy: one min/max per macro tile
    // (kMacroTiles x kMacroTiles tiles).
    macro_dim_x_ = (grid_dim_x_ + kMacroTiles - 1) / kMacroTiles;
    macro_dim_y_ = (grid_dim_y_ + kMacroTiles - 1) / kMacroTiles;
    const size_t num_macros = macro_dim_x_ * macro_dim_y_;
    const size_t macros_size = num_frames_ * num_macros * sizeof(float);
    MRL_CUDA_CHECK(cudaMalloc(&d_min_macro_depths_, macros_size));
    MRL_CUDA_CHECK(cudaMalloc(&d_max_macro_depths_, macros_size));
    // Initialize to +inf so the tiled query kernels see consistent bounds for
    // frames that were never inserted (their depth is all +inf as well).
    {
      std::vector<float> h_init_blocks(
          std::max(num_frames_ * num_blocks, num_frames_ * num_macros),
          INFINITY);
      MRL_CUDA_CHECK(cudaMemcpy(d_min_block_depths_,
                                h_init_blocks.data(),
                                blocks_size,
                                cudaMemcpyHostToDevice));
      MRL_CUDA_CHECK(cudaMemcpy(d_max_block_depths_,
                                h_init_blocks.data(),
                                blocks_size,
                                cudaMemcpyHostToDevice));
      MRL_CUDA_CHECK(cudaMemcpy(d_min_macro_depths_,
                                h_init_blocks.data(),
                                macros_size,
                                cudaMemcpyHostToDevice));
      MRL_CUDA_CHECK(cudaMemcpy(d_max_macro_depths_,
                                h_init_blocks.data(),
                                macros_size,
                                cudaMemcpyHostToDevice));
    }

    MRL_CUDA_CHECK(cudaDeviceSynchronize());
  } catch (...) {
    freeDeviceBuffers_();
    throw;
  }
}

void CudaDepthBuffer::Impl::freeDeviceBuffers_() {
  cudaFree(d_depth_buffer_);
  cudaFree(d_poses_);
  cudaFree(d_min_block_depths_);
  cudaFree(d_max_block_depths_);
  cudaFree(d_min_macro_depths_);
  cudaFree(d_max_macro_depths_);
  cudaFree(d_min_pool_input_);
  cudaFree(d_covariances_);
  cudaFree(d_has_covariance_);
  d_depth_buffer_ = nullptr;
  d_poses_ = nullptr;
  d_min_block_depths_ = nullptr;
  d_max_block_depths_ = nullptr;
  d_min_macro_depths_ = nullptr;
  d_max_macro_depths_ = nullptr;
  d_min_pool_input_ = nullptr;
  d_covariances_ = nullptr;
  d_has_covariance_ = nullptr;
  for (const auto& ws : workspaces_) {
    freeQueryWorkspace_(*ws);
    if (ws->stream) cudaStreamDestroy(ws->stream);
  }
  workspaces_.clear();
  free_workspaces_.clear();
  if (insert_stream_) {
    cudaStreamDestroy(insert_stream_);
    insert_stream_ = nullptr;
  }
}

CudaDepthBuffer::Impl::~Impl() { freeDeviceBuffers_(); }

bool CudaDepthBuffer::Impl::isKeyframe(const float* pose,
                                       const int64_t stamp_ns) const {
  const SharedGuard buffer_lock(buffer_mutex_);
  if (!pose) throw std::invalid_argument("pose must not be null");
  if (last_keyframe_pose_.size() != 7) return true;  // no keyframe yet

  const float dx = pose[0] - last_keyframe_pose_[0];
  const float dy = pose[1] - last_keyframe_pose_[1];
  const float dz = pose[2] - last_keyframe_pose_[2];
  if (std::sqrt(dx * dx + dy * dy + dz * dz) >=
      params_.keyframe_dist_threshold) {
    return true;
  }

  // Angle of the relative rotation q_last^-1 * q, via |Re(q_last^-1 * q)|.
  const float* a = last_keyframe_pose_.data();
  const float dot =
      a[3] * pose[3] + a[4] * pose[4] + a[5] * pose[5] + a[6] * pose[6];
  const float angle = 2.0f * std::acos(std::min(1.0f, std::fabs(dot)));
  if (angle >= params_.keyframe_angle_threshold) return true;

  if (stamp_ns != kInvalidFrameTimestampNs &&
      last_keyframe_stamp_ns_ != kInvalidFrameTimestampNs) {
    const double elapsed_s =
        static_cast<double>(stamp_ns - last_keyframe_stamp_ns_) * 1e-9;
    if (elapsed_s >= params_.keyframe_time_threshold) return true;
  }
  return false;
}

void CudaDepthBuffer::Impl::setDepthScaleFactor(float depth_scale_factor) {
  const ExclusiveGuard buffer_lock(buffer_mutex_);
  if (!std::isfinite(depth_scale_factor) || depth_scale_factor <= 0.0f) {
    throw std::invalid_argument("depth_scale_factor must be finite and > 0");
  }
  depth_scale_factor_ = depth_scale_factor;
  params_.depth_scale_factor = depth_scale_factor;
}

void CudaDepthBuffer::Impl::updatePoseSlot_(size_t i_frame, const float* pose) {
  if (!pose) {
    throw std::invalid_argument("pose must not be null");
  }
  const size_t pose_offset = i_frame * 7;
  std::copy_n(pose, 7, h_poses_.begin() + pose_offset);
  MRL_CUDA_CHECK(cudaMemcpyAsync(d_poses_ + pose_offset,
                                 pose,
                                 7 * sizeof(float),
                                 cudaMemcpyHostToDevice,
                                 insert_stream_));
  MRL_CUDA_CHECK(cudaStreamSynchronize(insert_stream_));
}

void CudaDepthBuffer::Impl::uploadCovarianceSlot_(size_t i_frame) {
  MRL_CUDA_CHECK(cudaMemcpyAsync(d_covariances_ + i_frame * 36,
                                 h_covariances_.data() + i_frame * 36,
                                 36 * sizeof(double),
                                 cudaMemcpyHostToDevice,
                                 insert_stream_));
  MRL_CUDA_CHECK(cudaMemcpyAsync(d_has_covariance_ + i_frame,
                                 frame_has_covariance_.data() + i_frame,
                                 sizeof(uint8_t),
                                 cudaMemcpyHostToDevice,
                                 insert_stream_));
  MRL_CUDA_CHECK(cudaStreamSynchronize(insert_stream_));
}

void CudaDepthBuffer::Impl::clearCovarianceSlot_(size_t i_frame) {
  const size_t covariance_offset = i_frame * 36;
  std::fill_n(h_covariances_.begin() + covariance_offset, 36, 0.0);
  frame_has_covariance_[i_frame] = 0;
  uploadCovarianceSlot_(i_frame);
}

void CudaDepthBuffer::Impl::setPosedDepthImage(const float* depth_data,
                                               const float* pose_data,
                                               int64_t stamp_ns,
                                               bool is_keyframe) {
  // Exclusive: a query in flight would read slots this call overwrites. The
  // lock is held until the device work below has completed.
  const ExclusiveGuard buffer_lock(buffer_mutex_);
  if (use_keyframing_) {
    if (is_keyframe) {
      if (num_frames_ > 1) {
        current_index_ = next_keyframe_index_;
        next_keyframe_index_++;
        if (next_keyframe_index_ >= num_frames_) {
          next_keyframe_index_ = 1;
        }
      } else {
        current_index_ = 0;
      }
    } else {
      current_index_ = 0;
    }
  } else {
    current_index_ = next_keyframe_index_;
    next_keyframe_index_ = (next_keyframe_index_ + 1) % num_frames_;
  }

  // Calculate offset for current index in depth buffer
  size_t depth_offset = current_index_ * num_rows_ * num_cols_;
  size_t pooled_depth_size = num_rows_ * num_cols_ * sizeof(float);

  dim3 block(TILE_W, TILE_H);
  dim3 grid(grid_dim_x_, grid_dim_y_, 1);
  if (min_pool_size_ != 1) {
    size_t raw_depth_size =
        static_cast<size_t>(raw_num_rows_) * raw_num_cols_ * sizeof(float);
    MRL_CUDA_CHECK(cudaMemcpyAsync(d_min_pool_input_,
                                   depth_data,
                                   raw_depth_size,
                                   cudaMemcpyDefault,
                                   insert_stream_));
    if (depth_scale_factor_ != 1.0f) {
      const size_t raw_num_elements =
          static_cast<size_t>(raw_num_rows_) * raw_num_cols_;
      const dim3 block_depth(256);
      const dim3 grid_depth(static_cast<uint32_t>(
          (raw_num_elements + block_depth.x - 1) / block_depth.x));
      scaleDepthKernel<<<grid_depth, block_depth, 0, insert_stream_>>>(
          d_min_pool_input_,
          raw_num_elements,
          depth_scale_factor_);
      MRL_CUDA_CHECK_KERNEL();
    }
    minPoolKernel<<<grid, block, 0, insert_stream_>>>(
        d_min_pool_input_,
        raw_num_rows_,
        raw_num_cols_,
        num_rows_,
        num_cols_,
        min_pool_size_,
        z_near_,
        z_far_,
        invalid_is_farplane_,
        d_depth_buffer_ + depth_offset);
    MRL_CUDA_CHECK_KERNEL();
  } else {
    MRL_CUDA_CHECK(cudaMemcpyAsync(d_depth_buffer_ + depth_offset,
                                   depth_data,
                                   pooled_depth_size,
                                   cudaMemcpyDefault,
                                   insert_stream_));
    if (depth_scale_factor_ != 1.0f) {
      const size_t num_elements = num_rows_ * num_cols_;
      const dim3 block_depth(256);
      const dim3 grid_depth(static_cast<uint32_t>(
          (num_elements + block_depth.x - 1) / block_depth.x));
      scaleDepthKernel<<<grid_depth, block_depth, 0, insert_stream_>>>(
          d_depth_buffer_ + depth_offset,
          num_elements,
          depth_scale_factor_);
      MRL_CUDA_CHECK_KERNEL();
    }
    {
      const size_t num_elements = num_rows_ * num_cols_;
      const dim3 block_depth(256);
      const dim3 grid_depth(static_cast<uint32_t>(
          (num_elements + block_depth.x - 1) / block_depth.x));
      preprocessDepthKernel<<<grid_depth, block_depth, 0, insert_stream_>>>(
          d_depth_buffer_ + depth_offset,
          num_elements,
          z_near_,
          z_far_,
          invalid_is_farplane_);
      MRL_CUDA_CHECK_KERNEL();
    }
  }

  frame_timestamps_ns_[current_index_] = stamp_ns;
  updatePoseSlot_(current_index_, pose_data);
  if (is_keyframe) {
    last_keyframe_pose_.assign(pose_data, pose_data + 7);
    last_keyframe_stamp_ns_ = stamp_ns;
  }
  clearCovarianceSlot_(current_index_);

  // update bounding values
  findMinMaxKernel<<<grid, block, 0, insert_stream_>>>(d_depth_buffer_,
                                                       num_frames_,
                                                       num_rows_,
                                                       num_cols_,
                                                       current_index_,
                                                       d_min_block_depths_,
                                                       d_max_block_depths_);
  MRL_CUDA_CHECK_KERNEL();
  {
    const int num_macros = static_cast<int>(macro_dim_x_ * macro_dim_y_);
    const int threads = 64;
    reduceMacroMinMaxKernel<<<(num_macros + threads - 1) / threads,
                              threads,
                              0,
                              insert_stream_>>>(
        d_min_block_depths_,
        d_max_block_depths_,
        current_index_,
        static_cast<uint16_t>(grid_dim_x_),
        static_cast<uint16_t>(grid_dim_y_),
        static_cast<uint16_t>(macro_dim_x_),
        static_cast<uint16_t>(macro_dim_y_),
        d_min_macro_depths_,
        d_max_macro_depths_);
    MRL_CUDA_CHECK_KERNEL();
  }
  MRL_CUDA_CHECK(cudaStreamSynchronize(insert_stream_));
}

bool CudaDepthBuffer::Impl::updatePoseAndCovarianceByTimestamp(
    int64_t stamp_ns,
    const float* pose,
    const double* covariance) {
  const ExclusiveGuard buffer_lock(buffer_mutex_);
  // Either half may be null: a backend that refines a pose without revising its
  // uncertainty, or one that only learns the uncertainty later, updates just
  // the half it has.
  if (stamp_ns == kInvalidFrameTimestampNs || (!pose && !covariance)) {
    return false;
  }

  bool any_updated = false;
  for (size_t i_frame = 0; i_frame < num_frames_; ++i_frame) {
    if (frame_timestamps_ns_[i_frame] != stamp_ns) {
      continue;
    }

    if (pose) updatePoseSlot_(i_frame, pose);
    if (covariance) {
      const size_t covariance_offset = i_frame * 36;
      std::copy_n(covariance, 36, h_covariances_.begin() + covariance_offset);
      frame_has_covariance_[i_frame] = 1;
      uploadCovarianceSlot_(i_frame);
    }
    any_updated = true;
  }

  return any_updated;
}

bool CudaDepthBuffer::Impl::getCovarianceByTimestamp(
    int64_t stamp_ns,
    double* covariance_out) const {
  if (stamp_ns == kInvalidFrameTimestampNs || !covariance_out) {
    return false;
  }

  for (size_t i_frame = 0; i_frame < num_frames_; ++i_frame) {
    if (frame_timestamps_ns_[i_frame] != stamp_ns ||
        !frame_has_covariance_[i_frame]) {
      continue;
    }
    const size_t covariance_offset = i_frame * 36;
    std::copy_n(h_covariances_.begin() + covariance_offset, 36, covariance_out);
    return true;
  }

  return false;
}

void CudaDepthBuffer::Impl::freeQueryWorkspace_(QueryWorkspace& ws) {
  if (ws.d_queries_frame) cudaFree(ws.d_queries_frame);
  if (ws.d_queries_frames) cudaFree(ws.d_queries_frames);
  if (ws.d_queries_frames_projected) cudaFree(ws.d_queries_frames_projected);
  if (ws.d_queries_aabbs) cudaFree(ws.d_queries_aabbs);
  if (ws.d_results) cudaFree(ws.d_results);
  if (ws.d_results_optimistic) cudaFree(ws.d_results_optimistic);
  if (ws.d_closest_points_world) cudaFree(ws.d_closest_points_world);
  if (ws.d_res_out) cudaFree(ws.d_res_out);
  if (ws.d_pose_compensation) cudaFree(ws.d_pose_compensation);
  if (ws.d_frame_query_truncation) cudaFree(ws.d_frame_query_truncation);
  if (ws.d_robot_dists) cudaFree(ws.d_robot_dists);
  if (ws.d_filter_flags) cudaFree(ws.d_filter_flags);
  if (ws.h_pinned_queries) cudaFreeHost(ws.h_pinned_queries);
  if (ws.h_pinned_results) cudaFreeHost(ws.h_pinned_results);
  const cudaStream_t stream = ws.stream;
  ws = QueryWorkspace{};
  ws.stream = stream;
}

CudaDepthBuffer::Impl::WorkspaceLease::WorkspaceLease(Impl& impl,
                                                      uint32_t num_queries)
    : impl_(impl),
      ws_(nullptr) {
  {
    const std::lock_guard<std::mutex> lock(impl_.pool_mutex_);
    if (!impl_.free_workspaces_.empty()) {
      ws_ = impl_.free_workspaces_.back();
      impl_.free_workspaces_.pop_back();
    }
  }
  if (!ws_) {
    auto owned = std::unique_ptr<QueryWorkspace>(new QueryWorkspace());
    MRL_CUDA_CHECK(cudaStreamCreate(&owned->stream));
    QueryWorkspace* raw = owned.get();
    const std::lock_guard<std::mutex> lock(impl_.pool_mutex_);
    impl_.workspaces_.push_back(std::move(owned));
    ws_ = raw;
  }
  impl_.ensureQueryWorkspace_(*ws_, num_queries);
}

CudaDepthBuffer::Impl::WorkspaceLease::~WorkspaceLease() {
  const std::lock_guard<std::mutex> lock(impl_.pool_mutex_);
  impl_.free_workspaces_.push_back(ws_);
}

void CudaDepthBuffer::Impl::ensureQueryWorkspace_(QueryWorkspace& ws,
                                                  uint32_t num_queries) {
  if (num_queries <= ws.capacity_queries) {
    return;
  }
  freeQueryWorkspace_(ws);
  // grow with headroom so alternating query counts do not thrash
  const uint32_t capacity = num_queries + num_queries / 2;
  const size_t nq = capacity;
  const size_t nfq = nq * num_frames_;
  MRL_CUDA_CHECK(cudaMalloc(&ws.d_queries_frame, nq * 3 * sizeof(float)));
  MRL_CUDA_CHECK(cudaMalloc(&ws.d_queries_frames, nfq * 3 * sizeof(float)));
  MRL_CUDA_CHECK(
      cudaMalloc(&ws.d_queries_frames_projected, nfq * sizeof(Pixel)));
  MRL_CUDA_CHECK(cudaMalloc(&ws.d_queries_aabbs, nfq * sizeof(AABB2D)));
  MRL_CUDA_CHECK(cudaMalloc(&ws.d_results, nfq * sizeof(PackedResult)));
  MRL_CUDA_CHECK(
      cudaMalloc(&ws.d_results_optimistic, nfq * sizeof(PackedResult)));
  MRL_CUDA_CHECK(
      cudaMalloc(&ws.d_closest_points_world, nfq * 3 * sizeof(float)));
  MRL_CUDA_CHECK(cudaMalloc(&ws.d_res_out, nq * 14 * sizeof(float)));
  MRL_CUDA_CHECK(cudaMalloc(&ws.d_pose_compensation, nfq * sizeof(float)));
  MRL_CUDA_CHECK(cudaMalloc(&ws.d_frame_query_truncation, nfq * sizeof(float)));
  MRL_CUDA_CHECK(cudaMalloc(&ws.d_robot_dists, nfq * sizeof(float)));
  MRL_CUDA_CHECK(
      cudaMalloc(&ws.d_filter_flags, ((nq + 31) / 32) * sizeof(uint32_t)));
  MRL_CUDA_CHECK(cudaMallocHost(&ws.h_pinned_queries, nq * 3 * sizeof(float)));
  MRL_CUDA_CHECK(cudaMallocHost(&ws.h_pinned_results, nq * 14 * sizeof(float)));
  ws.capacity_queries = capacity;
}

void CudaDepthBuffer::Impl::setRobotRadius(float robot_radius) {
  const ExclusiveGuard buffer_lock(buffer_mutex_);
  robot_radius_ = robot_radius;
  params_.robot_radius = robot_radius;
}

void CudaDepthBuffer::Impl::query(const float* queries,
                                  uint32_t num_queries,
                                  bool conservative,
                                  QueryResults& results,
                                  float truncation_radius,
                                  bool verbose,
                                  float suppression_radius,
                                  DebugDistances* debug_distances) {
  results.resize(num_queries);
  if (num_queries == 0) return;
  if (!queries) throw std::invalid_argument("queries must not be null");
  const bool enable_timers = verbose;

  // Shared with other queries, exclusive against inserts. Held until this
  // query's device work has finished, so no insert can overwrite the depth
  // buffer underneath it.
  const SharedGuard buffer_lock(buffer_mutex_);
  const WorkspaceLease lease(*this, num_queries);
  QueryWorkspace& ws = *lease;
  const cudaStream_t stream = ws.stream;

  mrl::profiling::Timer t_overall("query", enable_timers);
  mrl::profiling::Timer t_preprocessing("query/preprocess", enable_timers);

  size_t num_frames_queries = num_queries * num_frames_;

  // define block and grid sizes
  // parallelize over amorphous tasks
  dim3 block_1D(512);
  // parallelize over all queries and frames
  dim3 grid_frames_queries((num_frames_queries + 511) / 512);
  // parallelizing over depth buffer pixels
  dim3 block_buffer(TILE_W, TILE_H);

  // copy queries to device (pinned staging + async copy on the same stream
  // keeps ordering with the kernels below)
  mrl::profiling::Timer t_copy_queries("query/preprocess/upload",
                                       enable_timers);
  float* d_queries_reference = ws.d_queries_frame;
  const size_t queries_bytes = num_queries * 3 * sizeof(float);
  // Queries already on the device (a planner sampling trajectories there, a
  // torch tensor) are taken as they are. Host queries go through the pinned
  // staging buffer, without which the copy would block the caller while the
  // driver stages the pageable memory itself.
  cudaPointerAttributes query_attributes;
  const cudaError_t attributes_status =
      cudaPointerGetAttributes(&query_attributes, queries);
  // Older runtimes report an unregistered host pointer as an error rather than
  // as a type, and would leave it standing for the next check to trip over.
  if (attributes_status != cudaSuccess) cudaGetLastError();
  const bool queries_on_device = attributes_status == cudaSuccess &&
                                 query_attributes.type == cudaMemoryTypeDevice;
  if (queries_on_device) {
    MRL_CUDA_CHECK(cudaMemcpyAsync(d_queries_reference,
                                   queries,
                                   queries_bytes,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
  } else {
    float* h_queries = ws.h_pinned_queries;
    std::memcpy(h_queries, queries, queries_bytes);
    MRL_CUDA_CHECK(cudaMemcpyAsync(d_queries_reference,
                                   h_queries,
                                   queries_bytes,
                                   cudaMemcpyHostToDevice,
                                   stream));
  }
  t_copy_queries.stop();

  // Frames without a covariance contribute no margin, so a zero confidence is
  // the only thing that switches the pass off entirely.
  const bool pose_compensation_enabled = pose_covariance_confidence_ > 0.0f;
  float* d_pose_compensation_dists = nullptr;
  float* d_frame_query_truncation = nullptr;
  if (pose_compensation_enabled) {
    mrl::profiling::Timer pose_timer("query/preprocess/pose_covariance",
                                     enable_timers);
    d_pose_compensation_dists = ws.d_pose_compensation;
    // Only worth carrying per pair truncation radii when there is a finite
    // radius to grow in the first place.
    if (std::isfinite(truncation_radius)) {
      d_frame_query_truncation = ws.d_frame_query_truncation;
    }
    poseCompensationKernel<<<grid_frames_queries, block_1D, 0, stream>>>(
        d_covariances_,
        d_has_covariance_,
        d_poses_,
        d_queries_reference,
        num_queries,
        num_frames_,
        pose_covariance_confidence_,
        truncation_radius,
        d_pose_compensation_dists,
        d_frame_query_truncation);
    MRL_CUDA_CHECK_KERNEL();
    if (enable_timers) MRL_CUDA_CHECK(cudaStreamSynchronize(stream));
    pose_timer.stop();
  }

  // if supporession is active, perform it now
  uint32_t* d_filter_flags = nullptr;  // one means to be filtered out
  if (suppression_radius > 0) {
    mrl::profiling::Timer t_suppression("query/preprocess/suppression",
                                        enable_timers);
    size_t num_filter_flags = (num_queries + 31) / 32;
    d_filter_flags = ws.d_filter_flags;
    MRL_CUDA_CHECK(cudaMemsetAsync(d_filter_flags,
                                   0,
                                   num_filter_flags * sizeof(uint32_t),
                                   stream));
    dim3 grid_suppression((num_queries + TILE_W - 1) / TILE_W,
                          (num_queries + TILE_H - 1) / TILE_H);
    suppressQueriesKernel<<<grid_suppression, block_buffer, 0, stream>>>(
        d_queries_reference,
        num_queries,
        suppression_radius,
        d_filter_flags);
    MRL_CUDA_CHECK_KERNEL();
    if (enable_timers) MRL_CUDA_CHECK(cudaStreamSynchronize(stream));
    t_suppression.stop();
  }

  // transform all queries into all camera frames
  mrl::profiling::Timer t_transform_kernel("query/preprocess/transform",
                                           enable_timers);
  float* d_queries_frames = ws.d_queries_frames;
  transformQueriesKernel<<<grid_frames_queries, block_1D, 0, stream>>>(
      d_queries_reference,
      d_poses_,
      d_filter_flags,
      num_queries,
      num_frames_,
      d_queries_frames);
  MRL_CUDA_CHECK_KERNEL();
  if (enable_timers) MRL_CUDA_CHECK(cudaStreamSynchronize(stream));
  t_transform_kernel.stop();

  mrl::profiling::Timer t_projection_kernel("query/preprocess/project",
                                            enable_timers);
  Pixel* d_queries_frames_projected = ws.d_queries_frames_projected;
  projectQueriesKernel<<<grid_frames_queries, block_1D, 0, stream>>>(
      d_queries_frames,
      num_queries,
      num_frames_,
      num_rows_,
      num_cols_,
      cx_,
      cy_,
      fx_,
      fy_,
      z_near_,
      z_far_,
      d_queries_frames_projected);
  MRL_CUDA_CHECK_KERNEL();
  t_projection_kernel.stop();

  // compute AABBs of projected spheres
  mrl::profiling::Timer t_aabbs_kernel("query/preprocess/aabb", enable_timers);
  AABB2D* d_queries_aabbs = ws.d_queries_aabbs;
  computeSphereAABBsKernel<<<grid_frames_queries, block_1D, 0, stream>>>(
      d_queries_frames,
      d_frame_query_truncation,
      truncation_radius,
      num_queries,
      num_frames_,
      num_rows_,
      num_cols_,
      cx_,
      cy_,
      fx_,
      fy_,
      d_queries_aabbs,
      exact_patch_);
  MRL_CUDA_CHECK_KERNEL();
  if (enable_timers) MRL_CUDA_CHECK(cudaStreamSynchronize(stream));
  t_aabbs_kernel.stop();

  const dim3 block_query_centric(QUERY_CENTRIC_THREADS);
  const dim3 grid_query_centric(static_cast<unsigned int>(num_frames_queries));

  t_preprocessing.stop();

  // Nearest observed surface, per (frame, query). The kernel writes every
  // entry unconditionally, seeded with the truncation sentinel.
  PackedResult* d_results = ws.d_results;
  {
    mrl::profiling::Timer t_closest_point("query/surface_search",
                                          enable_timers);
    closestPointQueryCentricTiledKernel<<<grid_query_centric,
                                          block_query_centric,
                                          0,
                                          stream>>>(
        d_depth_buffer_,
        d_min_block_depths_,
        d_max_block_depths_,
        d_min_macro_depths_,
        d_max_macro_depths_,
        d_queries_frames_projected,
        num_frames_queries,
        num_queries,
        num_rows_,
        num_cols_,
        static_cast<uint16_t>(grid_dim_x_),
        static_cast<uint16_t>(grid_dim_y_),
        static_cast<uint16_t>(macro_dim_x_),
        static_cast<uint16_t>(macro_dim_y_),
        cx_,
        cy_,
        fx_,
        fy_,
        z_near_,
        z_far_,
        d_frame_query_truncation,
        truncation_radius,
        d_queries_frames,
        d_queries_aabbs,
        d_results,
        exact_patch_);
    MRL_CUDA_CHECK_KERNEL();
    if (enable_timers) MRL_CUDA_CHECK(cudaStreamSynchronize(stream));
    t_closest_point.stop();
  }

  // The optimistic answer is the state of the search right here, before the
  // occlusion pass folds unknown space in. Keeping a copy is what makes both
  // channels cost one memcpy rather than a second search.
  PackedResult* d_results_optimistic = ws.d_results_optimistic;
  MRL_CUDA_CHECK(cudaMemcpyAsync(d_results_optimistic,
                                 d_results,
                                 num_frames_queries * sizeof(PackedResult),
                                 cudaMemcpyDeviceToDevice,
                                 stream));

  float* d_robot_dists = nullptr;
  if (conservative) {
    // find AABBs of projected sphere centered at each query and going through
    // the closest point
    mrl::profiling::Timer t_conservative("query/occlusion", enable_timers);

    // prepare some variables about the queries once, so they later don't need
    // to be computed for each pixel
    mrl::profiling::Timer t_prep_occlusion_kernel("query/occlusion/prepare",
                                                  enable_timers);
    prepareOcclusionBoundariesKernel<<<grid_frames_queries,
                                       block_1D,
                                       0,
                                       stream>>>(d_queries_frames,
                                                 d_results,
                                                 num_queries,
                                                 num_frames_,
                                                 num_rows_,
                                                 num_cols_,
                                                 cx_,
                                                 cy_,
                                                 fx_,
                                                 fy_,
                                                 d_queries_aabbs);
    MRL_CUDA_CHECK_KERNEL();
    if (enable_timers) MRL_CUDA_CHECK(cudaStreamSynchronize(stream));
    t_prep_occlusion_kernel.stop();

    // find occlusion boundary points and check them against the queries
    mrl::profiling::Timer t_occlusion_kernel("query/occlusion/scan",
                                             enable_timers);
    occlusionBoundariesQueryCentricTiledKernel<<<grid_query_centric,
                                                 block_query_centric,
                                                 0,
                                                 stream>>>(
        d_depth_buffer_,
        d_min_block_depths_,
        d_max_block_depths_,
        d_min_macro_depths_,
        d_max_macro_depths_,
        d_queries_frames,
        d_queries_aabbs,
        d_queries_frames_projected,
        num_frames_queries,
        num_queries,
        num_rows_,
        num_cols_,
        static_cast<uint16_t>(grid_dim_x_),
        static_cast<uint16_t>(grid_dim_y_),
        static_cast<uint16_t>(macro_dim_x_),
        static_cast<uint16_t>(macro_dim_y_),
        cx_,
        cy_,
        fx_,
        fy_,
        z_near_,
        z_far_,
        d_frame_query_truncation,
        truncation_radius,
        d_results);
    MRL_CUDA_CHECK_KERNEL();
    if (enable_timers) MRL_CUDA_CHECK(cudaStreamSynchronize(stream));
    t_occlusion_kernel.stop();

    // Compute robot distances
    mrl::profiling::Timer t_robot_radius("query/occlusion/robot_radius",
                                         enable_timers);
    if (robot_radius_ > static_cast<float>(0)) {
      d_robot_dists = ws.d_robot_dists;
      robotRadiusKernel<<<grid_frames_queries, block_1D, 0, stream>>>(
          d_queries_frames,
          num_frames_queries,
          robot_center_x_,
          robot_center_y_,
          robot_center_z_,
          robot_radius_,
          d_frame_query_truncation,
          truncation_radius,
          d_robot_dists);
      MRL_CUDA_CHECK_KERNEL();
      if (enable_timers) MRL_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    t_robot_radius.stop();

    mrl::profiling::Timer t_signs_kernel("query/occlusion/signs",
                                         enable_timers);
    signsKernel<<<grid_frames_queries, block_1D, 0, stream>>>(
        d_depth_buffer_,
        d_queries_frames_projected,
        d_queries_frames,
        num_queries,
        num_frames_queries,
        num_rows_,
        num_cols_,
        z_near_,
        z_far_,
        d_results);
    MRL_CUDA_CHECK_KERNEL();
    if (enable_timers) MRL_CUDA_CHECK(cudaStreamSynchronize(stream));
    t_signs_kernel.stop();
    t_conservative.stop();
  }

  mrl::profiling::Timer t_postprocessing("query/postprocess", enable_timers);
  float* d_closest_points_world = ws.d_closest_points_world;

  // Six contiguous slices of one workspace buffer, so the whole answer comes
  // back in a single copy: distance, gradient and closest point per channel.
  const size_t nq = num_queries;
  float* const d_opt_distance = ws.d_res_out;
  float* const d_opt_gradient = d_opt_distance + nq;
  float* const d_opt_closest = d_opt_gradient + 3 * nq;
  float* const d_con_distance = d_opt_closest + 3 * nq;
  float* const d_con_gradient = d_con_distance + nq;
  float* const d_con_closest = d_con_gradient + 3 * nq;

  const dim3 grid_queries((num_queries + 511) / 512);
  const auto resolveChannel = [&](const PackedResult* d_channel_results,
                                  const float* d_channel_robot_dists,
                                  bool channel_conservative,
                                  float* d_distance,
                                  float* d_gradient,
                                  float* d_closest) {
    unprojectResultsKernel<<<grid_frames_queries, block_1D, 0, stream>>>(
        d_depth_buffer_,
        d_queries_frames,
        d_channel_results,
        d_poses_,
        num_queries,
        num_frames_,
        num_rows_,
        num_cols_,
        cx_,
        cy_,
        fx_,
        fy_,
        d_closest_points_world,
        exact_patch_);
    MRL_CUDA_CHECK_KERNEL();
    computeFinalResultsKernel<<<grid_queries, block_1D, 0, stream>>>(
        d_depth_buffer_,
        num_rows_,
        num_cols_,
        1.0f / fx_,
        approximate_margin_,
        d_channel_results,
        d_closest_points_world,
        d_channel_robot_dists,
        d_pose_compensation_dists,
        d_queries_reference,
        d_queries_frames,
        d_poses_,
        num_queries,
        num_frames_,
        robot_center_x_,
        robot_center_y_,
        robot_center_z_,
        robot_radius_,
        truncation_radius,
        channel_conservative,
        d_distance,
        d_gradient,
        d_closest);
    MRL_CUDA_CHECK_KERNEL();
  };

  mrl::profiling::Timer t_optimistic("query/postprocess/optimistic",
                                     enable_timers);
  resolveChannel(d_results_optimistic,
                 nullptr,
                 false,
                 d_opt_distance,
                 d_opt_gradient,
                 d_opt_closest);
  if (enable_timers) MRL_CUDA_CHECK(cudaStreamSynchronize(stream));
  t_optimistic.stop();

  if (conservative) {
    mrl::profiling::Timer t_conservative_merge("query/postprocess/conservative",
                                               enable_timers);
    resolveChannel(d_results,
                   d_robot_dists,
                   true,
                   d_con_distance,
                   d_con_gradient,
                   d_con_closest);
    if (enable_timers) MRL_CUDA_CHECK(cudaStreamSynchronize(stream));
    t_conservative_merge.stop();
  }

  // One synchronous D2H copy into pinned staging; it waits for every kernel
  // above, then the answer is unpacked into the caller's vectors.
  const size_t copied = conservative ? 14 * nq : 7 * nq;
  MRL_CUDA_CHECK(cudaMemcpyAsync(ws.h_pinned_results,
                                 ws.d_res_out,
                                 copied * sizeof(float),
                                 cudaMemcpyDeviceToHost,
                                 stream));
  MRL_CUDA_CHECK(cudaStreamSynchronize(stream));
  const float* h = ws.h_pinned_results;
  std::memcpy(results.optimistic_distance.data(), h, nq * sizeof(float));
  std::memcpy(results.optimistic_gradient.data(),
              h + nq,
              3 * nq * sizeof(float));
  std::memcpy(results.optimistic_closest.data(),
              h + 4 * nq,
              3 * nq * sizeof(float));
  if (conservative) {
    std::memcpy(results.conservative_distance.data(),
                h + 7 * nq,
                nq * sizeof(float));
    std::memcpy(results.conservative_gradient.data(),
                h + 8 * nq,
                3 * nq * sizeof(float));
    std::memcpy(results.conservative_closest.data(),
                h + 11 * nq,
                3 * nq * sizeof(float));
  }

  if (debug_distances) {
    debug_distances->num_frames = static_cast<uint32_t>(num_frames_);
    debug_distances->num_queries = num_queries;
    debug_distances->merged.resize(num_frames_queries);
    debug_distances->unknown.resize(num_frames_queries);
    debug_distances->occupied.resize(num_frames_queries);
    debug_distances->robot.resize(num_frames_queries);
    debug_distances->pose_compensation.resize(num_frames_queries);

    // Scratch buffers for the per-frame channels; only allocated when debug
    // output was requested, freed on every exit path.
    struct DeviceScratch {
      float* ptr = nullptr;
      explicit DeviceScratch(size_t count) {
        MRL_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
      }
      ~DeviceScratch() { cudaFree(ptr); }
      DeviceScratch(const DeviceScratch&) = delete;
      DeviceScratch& operator=(const DeviceScratch&) = delete;
    };
    DeviceScratch frame_merged(num_frames_queries);
    DeviceScratch frame_unknown(num_frames_queries);
    DeviceScratch frame_occupied(num_frames_queries);
    DeviceScratch frame_robot(num_frames_queries);
    DeviceScratch frame_pose_compensation(num_frames_queries);

    computePerFrameDebugDistancesKernel<<<grid_frames_queries,
                                          block_1D,
                                          0,
                                          stream>>>(
        d_depth_buffer_,
        num_rows_,
        num_cols_,
        1.0f / fx_,
        approximate_margin_,
        d_results,
        d_robot_dists,
        d_pose_compensation_dists,
        d_poses_,
        num_queries,
        num_frames_,
        robot_radius_,
        truncation_radius,
        conservative,
        frame_merged.ptr,
        frame_unknown.ptr,
        frame_occupied.ptr,
        frame_robot.ptr,
        frame_pose_compensation.ptr);
    MRL_CUDA_CHECK_KERNEL();

    const auto download = [&](std::vector<float>& dst,
                              const DeviceScratch& src) {
      MRL_CUDA_CHECK(cudaMemcpyAsync(dst.data(),
                                     src.ptr,
                                     num_frames_queries * sizeof(float),
                                     cudaMemcpyDeviceToHost,
                                     stream));
    };
    download(debug_distances->merged, frame_merged);
    download(debug_distances->unknown, frame_unknown);
    download(debug_distances->occupied, frame_occupied);
    download(debug_distances->robot, frame_robot);
    download(debug_distances->pose_compensation, frame_pose_compensation);
  }
  t_postprocessing.stop();
}

void CudaDepthBuffer::Impl::warmup() {
  // Simple CUDA warmup to avoid first-time overheads in timing
  int device_id = 0;
  cudaSetDevice(device_id);

  // Allocate small dummy buffer and launch warmup kernel
  constexpr int n = 1024;
  float* d_dummy = nullptr;
  cudaMalloc(&d_dummy, n * sizeof(float));

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  warmupKernel<<<grid, block>>>(d_dummy, n);
  MRL_CUDA_CHECK_KERNEL();
  MRL_CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(d_dummy);
}

// ---------------------------------------------------------------------------
// Public facade
// ---------------------------------------------------------------------------

CudaDepthBuffer::CudaDepthBuffer(const CameraIntrinsics& intrinsics,
                                 const BufferParams& params)
    : impl_(new Impl(intrinsics, params)) {}

CudaDepthBuffer::~CudaDepthBuffer() = default;

CameraIntrinsics CudaDepthBuffer::intrinsics() const {
  const SharedGuard lock(impl_->buffer_mutex_);
  return impl_->intrinsics_;
}

BufferParams CudaDepthBuffer::params() const {
  const SharedGuard lock(impl_->buffer_mutex_);
  return impl_->params_;
}

void CudaDepthBuffer::setPosedDepthImage(const float* depth_data,
                                         const float* pose,
                                         int64_t stamp_ns,
                                         bool is_keyframe) {
  impl_->setPosedDepthImage(depth_data, pose, stamp_ns, is_keyframe);
}

bool CudaDepthBuffer::isKeyframe(const float* pose, int64_t stamp_ns) const {
  return impl_->isKeyframe(pose, stamp_ns);
}

void CudaDepthBuffer::setDepthScaleFactor(float depth_scale_factor) {
  impl_->setDepthScaleFactor(depth_scale_factor);
}

void CudaDepthBuffer::setRobotRadius(float robot_radius) {
  impl_->setRobotRadius(robot_radius);
}

void CudaDepthBuffer::setPoseCovarianceConfidence(float confidence) {
  if (!std::isfinite(confidence) || confidence < 0.0f) {
    throw std::invalid_argument(
        "pose covariance confidence must be finite and >= 0");
  }
  const ExclusiveGuard lock(impl_->buffer_mutex_);
  impl_->pose_covariance_confidence_ = confidence;
  impl_->params_.pose_covariance_confidence = confidence;
}

bool CudaDepthBuffer::updatePoseAndCovarianceByTimestamp(
    int64_t stamp_ns,
    const float* pose,
    const double* covariance) {
  return impl_->updatePoseAndCovarianceByTimestamp(stamp_ns, pose, covariance);
}

bool CudaDepthBuffer::getCovarianceByTimestamp(int64_t stamp_ns,
                                               double* covariance_out) const {
  return impl_->getCovarianceByTimestamp(stamp_ns, covariance_out);
}

void CudaDepthBuffer::query(const float* queries,
                            uint32_t num_queries,
                            bool conservative,
                            QueryResults& results,
                            float truncation_radius,
                            bool verbose,
                            float suppression_radius,
                            DebugDistances* debug_distances) {
  impl_->query(queries,
               num_queries,
               conservative,
               results,
               truncation_radius,
               verbose,
               suppression_radius,
               debug_distances);
}

QueryResults CudaDepthBuffer::query(const std::vector<float>& queries,
                                    bool conservative,
                                    float truncation_radius) {
  if (queries.size() % 3 != 0) {
    throw std::invalid_argument("queries must hold 3 floats per point");
  }
  QueryResults results;
  query(queries.data(),
        static_cast<uint32_t>(queries.size() / 3),
        conservative,
        results,
        truncation_radius);
  return results;
}

void CudaDepthBuffer::warmup() { impl_->warmup(); }

}  // namespace mrl
