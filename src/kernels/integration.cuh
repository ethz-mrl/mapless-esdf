#ifndef MRL_CUDA_DEPTH_BUFFER_KERNELS_INTEGRATION_CUH
#define MRL_CUDA_DEPTH_BUFFER_KERNELS_INTEGRATION_CUH

#include "kernels/common.cuh"

namespace mrl {

/// Insert path: normalize a depth image onto the buffer invariant and refresh
/// the two levels of per tile depth ranges the query sweep prunes with.

__host__ __device__ __forceinline__ float normalizeDepth(
    float depth,
    float z_near,
    float z_far,
    bool invalid_is_farplane) {
  if (isnan(depth) || depth <= 0.0f) {
    return invalid_is_farplane ? z_far : z_near;
  }
  return fminf(fmaxf(depth, z_near), z_far);
}

/// True for measurements that carry range information, i.e. everything the min
/// pool may reduce over. +inf qualifies and normalizes to the far plane.
__host__ __device__ __forceinline__ bool isMeasuredDepth(float depth) {
  return !isnan(depth) && depth > 0.0f;
}

/// Min pool over measured depths only, so that invalid pixels inside a pooling
/// window never shadow a valid neighbour. Windows without any measurement stay
/// invalid and are resolved by normalizeDepth. The pool commutes with the
/// clipping, so pooling the raw values and normalizing afterwards is equivalent
/// to normalizing first.
__global__ void minPoolKernel(const float* __restrict__ d_in,
                              uint16_t in_rows,
                              uint16_t in_cols,
                              uint16_t num_rows,
                              uint16_t num_cols,
                              uint16_t pool_size,
                              float z_near,
                              float z_far,
                              bool invalid_is_farplane,
                              float* __restrict__ d_out) {
  const int i_row = blockIdx.y * blockDim.y + threadIdx.y;
  const int i_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_row >= num_rows || i_col >= num_cols) return;

  const int in_row_start = i_row * pool_size;
  const int in_col_start = i_col * pool_size;
  const int in_row_end =
      min(in_row_start + pool_size, static_cast<int>(in_rows));
  const int in_col_end =
      min(in_col_start + pool_size, static_cast<int>(in_cols));

  float min_val = NAN;
  for (int r = in_row_start; r < in_row_end; ++r) {
    const size_t row_offset = static_cast<size_t>(r) * in_cols;
    for (int c = in_col_start; c < in_col_end; ++c) {
      const float val = d_in[row_offset + c];
      if (!isMeasuredDepth(val)) continue;
      min_val = isnan(min_val) ? val : fminf(min_val, val);
    }
  }

  d_out[static_cast<size_t>(i_row) * num_cols + i_col] =
      normalizeDepth(min_val, z_near, z_far, invalid_is_farplane);
}

__global__ void preprocessDepthKernel(float* __restrict__ d_depth,
                                      size_t num_elements,
                                      float z_near,
                                      float z_far,
                                      bool invalid_is_farplane) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_elements) return;
  d_depth[idx] =
      normalizeDepth(d_depth[idx], z_near, z_far, invalid_is_farplane);
}

__global__ void scaleDepthKernel(float* __restrict__ d_depth,
                                 size_t num_elements,
                                 float depth_scale_factor) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_elements) return;
  d_depth[idx] *= depth_scale_factor;
}

__global__ void findMinMaxKernel(const float* __restrict__ d_depth_buffer,
                                 size_t num_frames,
                                 uint16_t num_rows,
                                 uint16_t num_cols,
                                 size_t i_frame,
                                 float* __restrict__ d_min_depth,
                                 float* __restrict__ d_max_depth) {
  __shared__ float warp_results_min[NUM_WARPS];
  __shared__ float warp_results_max[NUM_WARPS];

  // get pixel coordinate
  const int i_row = blockIdx.y * blockDim.y + threadIdx.y;
  const int i_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int i_lane = tid & 31;  // == tid % 32; thread index in warp (0-31)
  const int i_warp = tid >> 5;  // == tid / 32; warp index (0-15)
  float min_depth, max_depth;
  if (i_row >= num_rows || i_col >= num_cols || i_frame >= num_frames) {
    min_depth = INFINITY;
    max_depth = -INFINITY;
  } else {
    size_t idx = i_frame * num_rows * num_cols + i_row * num_cols + i_col;
    min_depth = d_depth_buffer[idx];
    max_depth = min_depth;

    // One pixel halo: the occlusion pass places candidates on segments
    // between a pixel depth and its right and down neighbours, which for
    // the last row and column of a tile lie in the next tile. Reducing the
    // halo in keeps every such segment inside this tile depth range, which
    // is what makes the tile bound a lower bound there.
    if (threadIdx.x == 0 && i_col > 0) {
      float val = d_depth_buffer[idx - 1];
      if (val < min_depth) min_depth = val;
      if (val > max_depth) max_depth = val;
    }
    if (threadIdx.x == blockDim.x - 1 && i_col < num_cols - 1) {
      float val = d_depth_buffer[idx + 1];
      if (val < min_depth) min_depth = val;
      if (val > max_depth) max_depth = val;
    }
    if (threadIdx.y == 0 && i_row > 0) {
      float val = d_depth_buffer[idx - num_cols];
      if (val < min_depth) min_depth = val;
      if (val > max_depth) max_depth = val;
    }
    if (threadIdx.y == blockDim.y - 1 && i_row < num_rows - 1) {
      float val = d_depth_buffer[idx + num_cols];
      if (val < min_depth) min_depth = val;
      if (val > max_depth) max_depth = val;
    }
  }

  // warp-level reduction for min value
  for (int offset = NUM_THREADS_IN_WARP / 2; offset > 0; offset >>= 1) {
    const float other_depth =
        __shfl_down_sync(0xFFFFFFFF, min_depth, offset);  // 32-lane mask
    if (other_depth < min_depth) {
      min_depth = other_depth;
    }
  }

  // warp-level reduction for max value
  for (int offset = NUM_THREADS_IN_WARP / 2; offset > 0; offset >>= 1) {
    const float other_depth =
        __shfl_down_sync(0xFFFFFFFF, max_depth, offset);  // 32-lane mask
    if (other_depth > max_depth) {
      max_depth = other_depth;
    }
  }

  // write warp-level reduction results to shared memory
  if (i_lane == 0) {
    warp_results_min[i_warp] = min_depth;
    warp_results_max[i_warp] = max_depth;
  }
  __syncthreads();  // wait for all warps to write

  // second reduction over warps using warp 0 only
  if (i_warp == 0) {
    float block_min = (tid < NUM_WARPS) ? warp_results_min[tid] : INFINITY;
    float block_max = (tid < NUM_WARPS) ? warp_results_max[tid] : -INFINITY;

    for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
      const float other_min =
          __shfl_down_sync(0xFFFF, block_min, offset);  // 16-lane mask
      if (other_min < block_min) {
        block_min = other_min;
      }
      const float other_max =
          __shfl_down_sync(0xFFFF, block_max, offset);  // 16-lane mask
      if (other_max > block_max) {
        block_max = other_max;
      }
    }

    if (i_lane == 0) {
      d_min_depth[i_frame * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +
                  blockIdx.x] = block_min;
      d_max_depth[i_frame * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +
                  blockIdx.x] = block_max;
    }
  }
}

__global__ void reduceMacroMinMaxKernel(
    const float* __restrict__ d_min_block_depths,
    const float* __restrict__ d_max_block_depths,
    size_t i_frame,
    uint16_t tiles_x,
    uint16_t tiles_y,
    uint16_t macros_x,
    uint16_t macros_y,
    float* __restrict__ d_min_macro_depths,
    float* __restrict__ d_max_macro_depths) {
  const int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= macros_x * macros_y) return;
  const int mx = m % macros_x;
  const int my = m / macros_x;
  const size_t frame_tile_off =
      i_frame * static_cast<size_t>(tiles_x) * tiles_y;
  const int ty1 = min((my + 1) * kMacroTiles, static_cast<int>(tiles_y));
  const int tx1 = min((mx + 1) * kMacroTiles, static_cast<int>(tiles_x));
  float mn = INFINITY;
  float mx_val = -INFINITY;
  for (int ty = my * kMacroTiles; ty < ty1; ++ty) {
    for (int tx = mx * kMacroTiles; tx < tx1; ++tx) {
      const size_t idx =
          frame_tile_off + static_cast<size_t>(ty) * tiles_x + tx;
      mn = fminf(mn, d_min_block_depths[idx]);
      mx_val = fmaxf(mx_val, d_max_block_depths[idx]);
    }
  }
  const size_t out_idx = i_frame * static_cast<size_t>(macros_x) * macros_y +
                         static_cast<size_t>(my) * macros_x + mx;
  d_min_macro_depths[out_idx] = mn;
  d_max_macro_depths[out_idx] = mx_val;
}

// Lower bound (squared, padded) of the distance from the query to any surface
// sample inside the tile: frustum-slab box [u0,u1)x[v0,v1)x[dmin,dmax].

}  // namespace mrl

#endif  // MRL_CUDA_DEPTH_BUFFER_KERNELS_INTEGRATION_CUH
