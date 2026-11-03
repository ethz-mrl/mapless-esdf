#ifndef MRL_CUDA_DEPTH_BUFFER_KERNELS_BOUNDS_CUH
#define MRL_CUDA_DEPTH_BUFFER_KERNELS_BOUNDS_CUH

#include "kernels/common.cuh"

namespace mrl {

/// Geometry of the search domain: the pixel AABB of a projected sphere, the
/// tile hierarchy constants, and the conservative distance bounds the sweep
/// prunes with. Every bound here is a lower bound on the distance from the
/// query to any candidate inside the tile, deflated by kTileBoundPad so that a
/// candidate which could tie the minimum is never pruned.

__host__ __device__ AABB2D computeSpheresProjectedAABB(const float* center,
                                                       uint16_t num_rows,
                                                       uint16_t num_cols,
                                                       float cx,
                                                       float cy,
                                                       float fx,
                                                       float fy,
                                                       float r) {
  // // sphere is behind camera
  // if (center[2] < static_cast<float>(1e-4)) {
  //   return AABB2D{0, 0, 0, 0};
  // }

  // camera is inside sphere
  if (fabs(center[2]) <= r + static_cast<float>(1e-4))
    return AABB2D{0, num_cols, 0, num_rows};

  // compute x bounds
  float gamma_x = atan2f(center[0], center[2]);
  float norm_x = sqrtf(center[0] * center[0] + center[2] * center[2]);
  // camera is inside sphere
  if (r > norm_x - 1e-4) return AABB2D{0, num_cols, 0, num_rows};

  float tau_x = asinf(r / norm_x);
  float u_min_unitplane = tanf(gamma_x - tau_x);
  float u_max_unitplane = tanf(gamma_x + tau_x);

  // Compute y bounds
  float gamma_y = atan2f(center[1], center[2]);
  float norm_y = sqrtf(center[1] * center[1] + center[2] * center[2]);
  // camera is inside sphere
  if (r > norm_y - 1e-4) return AABB2D{0, num_cols, 0, num_rows};

  float tau_y = asinf(r / norm_y);
  float v_min_unitplane = tanf(gamma_y - tau_y);
  float v_max_unitplane = tanf(gamma_y + tau_y);

  // Compute pixel bounds for the projected sphere's AABB
  float u_min_f = floorf(fx * u_min_unitplane + cx - 0.5f);
  float u_max_f = ceilf(fx * u_max_unitplane + cx + 0.5f);
  float v_min_f = floorf(fy * v_min_unitplane + cy - 0.5f);
  float v_max_f = ceilf(fy * v_max_unitplane + cy + 0.5f);

  uint16_t u_min_clamp = static_cast<uint16_t>(
      fmaxf(0.0f, fminf(u_min_f, static_cast<float>(num_cols - 1))));
  uint16_t u_max_clamp = static_cast<uint16_t>(
      fmaxf(0.0f, fminf(u_max_f, static_cast<float>(num_cols))));
  uint16_t v_min_clamp = static_cast<uint16_t>(
      fmaxf(0.0f, fminf(v_min_f, static_cast<float>(num_rows - 1))));
  uint16_t v_max_clamp = static_cast<uint16_t>(
      fmaxf(0.0f, fminf(v_max_f, static_cast<float>(num_rows))));

  return AABB2D{u_min_clamp, u_max_clamp, v_min_clamp, v_max_clamp};
}

__host__ __device__ void projectToPixel(const float* x_cam,
                                        const uint16_t& num_rows,
                                        const uint16_t& num_cols,
                                        const float& cx,
                                        const float& cy,
                                        const float& fx,
                                        const float& fy,
                                        const float& z_near,
                                        const float& z_far,
                                        int16_t& u,
                                        int16_t& v,
                                        bool& is_outside) {
  float z = x_cam[2];
  if (z < static_cast<float>(z_near) || z > static_cast<float>(z_far)) {
    is_outside = true;
    u = -1;
    v = -1;
    return;
  }
  // Round in float and range check before narrowing: for queries far off the
  // optical axis the projected coordinate can exceed the int16_t range, where
  // the conversion would be undefined.
  const float u_f = roundf((fx * x_cam[0]) / x_cam[2] + cx);
  const float v_f = roundf((fy * x_cam[1]) / x_cam[2] + cy);
  if (!(u_f >= 0.0f && u_f < static_cast<float>(num_cols)) ||
      !(v_f >= 0.0f && v_f < static_cast<float>(num_rows))) {
    is_outside = true;
    u = -1;
    v = -1;
    return;
  }
  u = static_cast<int16_t>(u_f);
  v = static_cast<int16_t>(v_f);
  is_outside = false;
}

/// Map a raw depth measurement onto the buffer's invariant: every stored depth
/// lies in [z_near, z_far]. Measurements outside the clipping range are clipped
/// to the nearer plane (+inf therefore becomes z_far); measurements that carry
/// no range information at all (NaN or <= 0) become z_far when
/// invalid_is_farplane is set and z_near otherwise, i.e. they are treated as
/// the most permissive or the most conservative observation respectively.

constexpr int kTileW = 32;                // must match CudaDepthBuffer::TILE_W
constexpr int kTileH = 16;                // must match CudaDepthBuffer::TILE_H
constexpr int kMacroTiles = 8;            // macro tile = 8x8 tiles = 256x128 px
constexpr int kFlatSweepTiles = 32;       // below this, skip the macro level
constexpr float kTileBoundPad = 0.9999f;  // deflate bound: never prune ties

__host__ __device__ __forceinline__ int outwardCoord(int k,
                                                     int center,
                                                     int lo,
                                                     int hi) {
  const int a = center - lo;
  const int b = hi - center;
  const int m2 = 2 * min(a, b);
  if (k <= m2) {
    const int step = (k + 1) >> 1;
    return (k & 1) ? center + step : center - step;
  }
  const int rem = k - m2;
  return (b > a) ? center + min(a, b) + rem : center - min(a, b) - rem;
}

// Reduce the per-tile min/max depth of one frame into per-macro-tile min/max
// (coarse level of the two-level query culling hierarchy).

__host__ __device__ __forceinline__ float tileSlabBoundSq(
    float qx,
    float qy,
    float qz,
    float u0,
    float u1,
    float v0,
    float v1,
    float dmin,
    float dmax,
    float cx,
    float cy,
    float inv_fx,
    float inv_fy,
    float half_extent = 0.0f) {
  // half_extent: 0 when pixels are point samples at integer (u, v); 0.5 when
  // each pixel is the patch [u-1/2, u+1/2) x [v-1/2, v+1/2)
  // (PoolCompensation::kExact), so the tile reaches half a pixel further out
  // on every side.
  const float dz = fmaxf(0.0f, fmaxf(dmin - qz, qz - dmax));
  // pixel->camera x is (u - cx) * d * inv_fx, bilinear in (u, d):
  // extrema at the corners.
  const float xa = (u0 - half_extent - cx) * inv_fx;
  const float xb = (u1 - 1.0f + half_extent - cx) * inv_fx;
  const float x_lo = fminf(xa * dmin, xa * dmax);
  const float x_hi = fmaxf(xb * dmin, xb * dmax);
  const float dx = fmaxf(0.0f, fmaxf(x_lo - qx, qx - x_hi));
  const float ya = (v0 - half_extent - cy) * inv_fy;
  const float yb = (v1 - 1.0f + half_extent - cy) * inv_fy;
  const float y_lo = fminf(ya * dmin, ya * dmax);
  const float y_hi = fmaxf(yb * dmin, yb * dmax);
  const float dy = fmaxf(0.0f, fmaxf(y_lo - qy, qy - y_hi));
  return distSq3(dx, dy, dz) * kTileBoundPad;
}

// Read the block-wide running best (squared distance bits), which other warps
// shrink concurrently with atomicMin. The read must not be cached in a register
// across the sweep, or a warp would keep pruning against a stale bound; the
// volatile access forces a reload and cannot tear, since the word is naturally
// aligned. An atomic read-modify-write would also be correct but adds write
// traffic to a heavily contended address. Only monotone progress is required
// here, never a particular value.
__device__ __forceinline__ float loadBestDistSq(
    const uint32_t* s_best_sq_bits) {
  return __uint_as_float(
      *static_cast<const volatile uint32_t*>(s_best_sq_bits));
}

}  // namespace mrl

#endif  // MRL_CUDA_DEPTH_BUFFER_KERNELS_BOUNDS_CUH
