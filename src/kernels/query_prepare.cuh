#ifndef MRL_CUDA_DEPTH_BUFFER_KERNELS_QUERY_PREPARE_CUH
#define MRL_CUDA_DEPTH_BUFFER_KERNELS_QUERY_PREPARE_CUH

#include "kernels/bounds.cuh"
#include "kernels/common.cuh"

namespace mrl {

/// Per query setup shared by both search passes: suppression, transformation
/// into every camera frame, projection and the search domain AABBs.

__global__ void suppressQueriesKernel(const float* __restrict__ d_queries_frame,
                                      uint32_t num_queries,
                                      float suppression_radius,
                                      uint32_t* __restrict__ d_filter_flags) {
  const int i_query_a = blockIdx.y * blockDim.y + threadIdx.y;
  const int i_query_b = blockIdx.x * blockDim.x + threadIdx.x;

  if (i_query_a >= num_queries || i_query_b >= num_queries ||
      i_query_a >= i_query_b) {
    return;
  }
  float dx = d_queries_frame[i_query_a * 3] - d_queries_frame[i_query_b * 3];
  float dy =
      d_queries_frame[i_query_a * 3 + 1] - d_queries_frame[i_query_b * 3 + 1];
  float dz =
      d_queries_frame[i_query_a * 3 + 2] - d_queries_frame[i_query_b * 3 + 2];
  float dist_sq = dx * dx + dy * dy + dz * dz;

  // query_b, whose index is always greater than i_query_a is marked as to be
  // filtered. This way only the lowest i_query remains non-filtered effectively
  // filtering out all other queries that are within suppression_radius
  if (dist_sq < suppression_radius * suppression_radius) {
    // use efficient versions of i_query / 32 and i_query % 32
    atomicOr(&d_filter_flags[i_query_b >> 5], 1u << (i_query_b & 31));
  }
}

/// Brings every query, given in the reference frame, into every camera frame.
/// Suppressed queries are pushed to infinity so that every later stage sees
/// them as unreachable without needing the flags again.
__global__ void transformQueriesKernel(
    const float* __restrict__ d_queries_reference,
    const float* __restrict__ d_poses,
    const uint32_t* __restrict__ d_filter_flags,
    size_t num_queries,
    size_t num_frames,
    float* __restrict__ d_queries_frames) {
  const int i_frame_query = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_frame_query >= num_queries * num_frames) return;
  const int i_frame = i_frame_query / num_queries;
  const int i_query = i_frame_query % num_queries;

  if (d_filter_flags && d_filter_flags[i_query >> 5] & (1u << (i_query & 31))) {
    d_queries_frames[i_frame_query * 3] = INFINITY;
    d_queries_frames[i_frame_query * 3 + 1] = INFINITY;
    d_queries_frames[i_frame_query * 3 + 2] = INFINITY;
    return;
  }

  transformToFrame(&d_poses[i_frame * 7],
                   &d_queries_reference[i_query * 3],
                   &d_queries_frames[i_frame_query * 3]);
}

__global__ void projectQueriesKernel(
    const float* __restrict__ d_queries_frames,
    size_t num_queries,
    size_t num_frames,
    uint16_t num_rows,
    uint16_t num_cols,
    float cx,
    float cy,
    float fx,
    float fy,
    float z_near,
    float z_far,
    Pixel* __restrict__ d_queries_frames_projected) {
  const int tid = threadIdx.x;
  const int i_frame_query = blockIdx.x * blockDim.x + tid;
  if (i_frame_query >= num_queries * num_frames) return;

  float query_cam[3];
  query_cam[0] = d_queries_frames[i_frame_query * 3];
  query_cam[1] = d_queries_frames[i_frame_query * 3 + 1];
  query_cam[2] = d_queries_frames[i_frame_query * 3 + 2];

  int16_t u, v;
  bool is_outside;
  projectToPixel(query_cam,
                 num_rows,
                 num_cols,
                 cx,
                 cy,
                 fx,
                 fy,
                 z_near,
                 z_far,
                 u,
                 v,
                 is_outside);
  d_queries_frames_projected[i_frame_query] = Pixel{u, v};
}

__global__ void computeSphereAABBsKernel(
    const float* __restrict__ d_queries_frames,
    const float* __restrict__ d_frame_query_truncation,
    float truncation_radius,
    size_t num_queries,
    size_t num_frames,
    uint16_t num_rows,
    uint16_t num_cols,
    float cx,
    float cy,
    float fx,
    float fy,
    AABB2D* d_queries_aabbs,
    bool exact_patch) {
  const int tid = threadIdx.x;
  const int i_frame_query = blockIdx.x * blockDim.x + tid;
  if (i_frame_query >= num_queries * num_frames) return;

  const float radius = getFrameQueryTruncation(d_frame_query_truncation,
                                               i_frame_query,
                                               truncation_radius);
  AABB2D aabb =
      computeSpheresProjectedAABB(&d_queries_frames[i_frame_query * 3],
                                  num_rows,
                                  num_cols,
                                  cx,
                                  cy,
                                  fx,
                                  fy,
                                  radius);
  if (exact_patch) {
    // A pixel's patch reaches half a pixel beyond its sample point, so the
    // pixel whose sample lies just outside the sphere's projection can still
    // touch it: widen by one pixel on each side.
    aabb.u_min = aabb.u_min > 0 ? aabb.u_min - 1 : 0;
    aabb.v_min = aabb.v_min > 0 ? aabb.v_min - 1 : 0;
    aabb.u_max = aabb.u_max + 1 < num_cols ? aabb.u_max + 1 : num_cols;
    aabb.v_max = aabb.v_max + 1 < num_rows ? aabb.v_max + 1 : num_rows;
  }
  d_queries_aabbs[i_frame_query] = aabb;
}

__global__ void prepareOcclusionBoundariesKernel(
    const float* __restrict__ d_queries_frames,
    const PackedResult* __restrict__ d_results,
    size_t num_queries,
    size_t num_frames,
    uint16_t num_rows,
    uint16_t num_cols,
    float cx,
    float cy,
    float fx,
    float fy,
    AABB2D* d_queries_aabbs) {
  const int i_frame_query = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_frame_query >= num_queries * num_frames) return;

  float query[] = {d_queries_frames[i_frame_query * 3],
                   d_queries_frames[i_frame_query * 3 + 1],
                   d_queries_frames[i_frame_query * 3 + 2]};
  const float lowest_dist = fabsf(d_results[i_frame_query].getDist());
  d_queries_aabbs[i_frame_query] = computeSpheresProjectedAABB(query,
                                                               num_rows,
                                                               num_cols,
                                                               cx,
                                                               cy,
                                                               fx,
                                                               fy,
                                                               lowest_dist);
}

}  // namespace mrl

#endif  // MRL_CUDA_DEPTH_BUFFER_KERNELS_QUERY_PREPARE_CUH
