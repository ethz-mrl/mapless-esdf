#ifndef MRL_CUDA_DEPTH_BUFFER_KERNELS_COMMON_CUH
#define MRL_CUDA_DEPTH_BUFFER_KERNELS_COMMON_CUH

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mrl {

constexpr int NUM_WARPS = 16;
constexpr int NUM_THREADS_IN_WARP = 32;
/// Compares greater than any valid packed result: all-ones distance bits.
static constexpr uint64_t INFINITY_PACKED = 0xFFFFFFFF7FF00000ULL;

class AABB2D {
 public:
  __host__ __device__ AABB2D() : u_min(0), u_max(0), v_min(0), v_max(0) {}
  __host__ __device__
  AABB2D(uint16_t u_min_, uint16_t u_max_, uint16_t v_min_, uint16_t v_max_)
      : u_min(u_min_),
        u_max(u_max_),
        v_min(v_min_),
        v_max(v_max_) {}
  __host__ __device__ __forceinline__ bool intersects(AABB2D other) const {
    return (u_min < other.u_max && u_max > other.u_min && v_min < other.v_max &&
            v_max > other.v_min);
  }
  uint16_t u_min;
  uint16_t u_max;
  uint16_t v_min;
  uint16_t v_max;
};

/// Projection of a query into an image: a pixel inside it, or the (-1, -1)
/// sentinel when the query does not project into it.
class Pixel {
 public:
  __host__ __device__ Pixel() : u(-1), v(-1) {}
  __host__ __device__ Pixel(int16_t u_, int16_t v_) : u(u_), v(v_) {}
  __host__ __device__ __forceinline__ bool isValid() const {
    return u >= 0 && v >= 0;
  }
  int16_t u;
  int16_t v;
};

/// Types and helpers shared by every kernel: the packed argmin key, the pose
/// transforms and the pixel back projection.

namespace {

/// Throw on a failed CUDA runtime call, naming the call site.
inline void cudaCheck(cudaError_t err,
                      const char* call,
                      const char* file,
                      int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + call +
                             " failed at " + file + ":" + std::to_string(line) +
                             ": " + cudaGetErrorString(err));
  }
}

}  // namespace

// Checks a CUDA runtime call. MRL_CUDA_CHECK_KERNEL() checks the launch
// configuration of the preceding kernel without forcing a synchronization;
// asynchronous execution errors surface at the next synchronizing call.
#define MRL_CUDA_CHECK(call) ::mrl::cudaCheck((call), #call, __FILE__, __LINE__)
#define MRL_CUDA_CHECK_KERNEL() \
  ::mrl::cudaCheck(cudaGetLastError(), "kernel launch", __FILE__, __LINE__)

constexpr int QUERY_CENTRIC_THREADS = 128;

__host__ __device__ __forceinline__ float dot3(float ax,
                                               float ay,
                                               float az,
                                               float bx,
                                               float by,
                                               float bz) {
  return ax * bx + ay * by + az * bz;
}

__host__ __device__ __forceinline__ float distSq3(float dx,
                                                  float dy,
                                                  float dz) {
  return dx * dx + dy * dy + dz * dz;
}

__host__ __device__ __forceinline__ float clampSignedDistance(
    float dist,
    float truncation_radius) {
  return fmaxf(-truncation_radius, fminf(dist, truncation_radius));
}

__host__ __device__ __forceinline__ float getFrameQueryTruncation(
    const float* frame_query_truncation,
    size_t i_frame_query,
    float default_truncation) {
  return frame_query_truncation ? frame_query_truncation[i_frame_query]
                                : default_truncation;
}

__host__ __device__ __forceinline__ bool isPoseValid(const float* pose) {
  return !(pose[3] == 0.0f && pose[4] == 0.0f && pose[5] == 0.0f &&
           pose[6] == 0.0f);
}

__device__ __forceinline__ void unprojectPixel(float depth,
                                               int i_col,
                                               int i_row,
                                               float cx,
                                               float cy,
                                               float inv_fx,
                                               float inv_fy,
                                               float& x_cam,
                                               float& y_cam) {
  x_cam = (static_cast<float>(i_col) - cx) * depth * inv_fx;
  y_cam = (static_cast<float>(i_row) - cy) * depth * inv_fy;
}

struct PackedResult {
  uint64_t val;
  __host__ __device__ PackedResult() : val(INFINITY_PACKED) {}

  __host__ __device__ PackedResult(uint64_t v) : val(v) {}

  // Constructor from components
  __host__ __device__ PackedResult(uint16_t row,
                                   uint16_t col,
                                   float dist_sq,
                                   bool is_truncated = false,
                                   bool is_occlusion = false,
                                   bool is_extremum = false) {
    uint32_t dist_uint;
#ifdef __CUDA_ARCH__
    dist_uint = __float_as_uint(dist_sq);
#else
    union {
      float f;
      uint32_t u;
    } converter;
    converter.f = dist_sq;
    dist_uint = converter.u;
#endif
    val = static_cast<uint64_t>(static_cast<uint32_t>(is_extremum) << 28 |
                                static_cast<uint32_t>(is_truncated) << 27 |
                                static_cast<uint32_t>(is_occlusion) << 26 |
                                (static_cast<uint32_t>(row) & 0x1FFF) << 13 |
                                (static_cast<uint32_t>(col) & 0x1FFF)) |
          static_cast<uint64_t>(dist_uint) << 32;
  }

  // Assignment operators
  __host__ __device__ PackedResult& operator=(const PackedResult& other) {
    val = other.val;
    return *this;
  }

  __host__ __device__ PackedResult& operator=(const uint64_t& other_val) {
    val = other_val;
    return *this;
  }

  // Conversion operator to uint64_t
  __host__ __device__ operator uint64_t() const { return val; }

  // Extract components
  __host__ __device__ float getDistSq() const {
    uint32_t dist_uint = static_cast<uint32_t>(val >> 32 & 0xFFFFFFFF);
#ifdef __CUDA_ARCH__
    return __uint_as_float(dist_uint);
#else
    union {
      uint32_t u;
      float f;
    } converter;
    converter.u = dist_uint;
    return converter.f;
#endif
  }
  __host__ __device__ uint16_t getRow() const {
    return static_cast<uint16_t>((val >> 13) & 0x1FFF);
  }
  __host__ __device__ uint16_t getCol() const {
    return static_cast<uint16_t>((val) & 0x1FFF);
  }

  __host__ __device__ void setSign(bool positive) {
    if (positive) {
      val &= ~(1ULL << 63);
    } else {
      val |= (1ULL << 63);
    }
  }

  __host__ __device__ bool isOcclusion() const { return (val >> 26) & 0x1; }

  __host__ __device__ bool isTruncated() const { return (val >> 27) & 0x1; }

  __host__ __device__ bool isExtremum() const { return (val >> 28) & 0x1; }

  __host__ __device__ static inline PackedResult Infinity() {
    return PackedResult(INFINITY_PACKED);
  }

  __host__ __device__ static inline PackedResult Truncated(
      float truncation_radius_sq) {
    return PackedResult(maxNumRows(), maxNumCols(), truncation_radius_sq, true);
  }

  __host__ __device__ inline bool isInfinity() const {
    return val == INFINITY_PACKED;
  }

  __host__ __device__ float getDist() const {
    float d2 = getDistSq();
    return (d2 < 0.0f) ? -sqrtf(-d2) : sqrtf(d2);
  }

  __host__ __device__ static inline uint16_t maxNumRows() {
    return (1 << 13) - 1;
  }
  __host__ __device__ static inline uint16_t maxNumCols() {
    return (1 << 13) - 1;
  }
};

__host__ __device__ __forceinline__ void quaternionRotate(float qx,
                                                          float qy,
                                                          float qz,
                                                          float qw,
                                                          float px,
                                                          float py,
                                                          float pz,
                                                          float& rx,
                                                          float& ry,
                                                          float& rz) {
  // most efficient quaternion rotation as  derived here:
  // https://math.stackexchange.com/questions/2583669/quaternion-rotation-formula-algebra
  float two = static_cast<float>(2.0);
  float cx = two * (qy * pz - qz * py);
  float cy = two * (qz * px - qx * pz);
  float cz = two * (qx * py - qy * px);

  rx = px + qw * cx + qy * cz - qz * cy;
  ry = py + qw * cy + qz * cx - qx * cz;
  rz = pz + qw * cz + qx * cy - qy * cx;
}

__host__ __device__ __forceinline__ void transformToWorld(const float* pose,
                                                          const float* x_in,
                                                          float* x_out) {
  // convert camera coordinates to world coordinates
  quaternionRotate(pose[3],
                   pose[4],
                   pose[5],
                   pose[6],
                   x_in[0],
                   x_in[1],
                   x_in[2],
                   x_out[0],
                   x_out[1],
                   x_out[2]);
  x_out[0] += pose[0];
  x_out[1] += pose[1];
  x_out[2] += pose[2];
}

__host__ __device__ __forceinline__ void transformToFrame(const float* pose,
                                                          const float* x_in,
                                                          float* x_out) {
  // convert world coordinates to specified camera frame
  float rel_x = x_in[0] - pose[0];
  float rel_y = x_in[1] - pose[1];
  float rel_z = x_in[2] - pose[2];
  quaternionRotate(-pose[3],
                   -pose[4],
                   -pose[5],
                   pose[6],
                   rel_x,
                   rel_y,
                   rel_z,
                   x_out[0],
                   x_out[1],
                   x_out[2]);
}

}  // namespace mrl

#endif  // MRL_CUDA_DEPTH_BUFFER_KERNELS_COMMON_CUH
