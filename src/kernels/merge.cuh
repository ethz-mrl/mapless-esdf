#ifndef MRL_CUDA_DEPTH_BUFFER_KERNELS_MERGE_CUH
#define MRL_CUDA_DEPTH_BUFFER_KERNELS_MERGE_CUH

#include "kernels/common.cuh"

namespace mrl {

/// Turning per (frame, query) results into the answer: signs, back projection
/// of the winning pixel, the robot clearing sphere, and the max pooling across
/// frames.

__global__ void signsKernel(
    const float* __restrict__ d_depth_buffer,
    const Pixel* __restrict__ d_queries_frames_projected,
    const float* __restrict__ d_queries_frames,
    size_t num_queries,
    size_t num_frames_queries,
    uint16_t num_rows,
    uint16_t num_cols,
    float z_near,
    float z_far,
    PackedResult* __restrict__ d_results) {
  // get current thread's frame index
  const int i_frame_query = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_frame_query >= num_frames_queries) return;
  int i_frame = i_frame_query / num_queries;

  // load projected query
  Pixel pixel = d_queries_frames_projected[i_frame_query];
  bool positive_sign = false;

  if (pixel.u < 0 || pixel.u >= num_cols || pixel.v < 0 ||
      pixel.v >= num_rows) {
    // outside image bounds -> negative sign
    positive_sign = false;
  } else {
    // if inside image bounds, compare depth values
    float query_depth = d_queries_frames[i_frame_query * 3 + 2];
    if (query_depth < static_cast<float>(z_near) ||
        query_depth > static_cast<float>(z_far)) {
      // query is behind camera
      positive_sign = false;
    } else {
      float depth_at_query = d_depth_buffer[i_frame * num_rows * num_cols +
                                            pixel.v * num_cols + pixel.u];
      positive_sign = query_depth < depth_at_query;
    }
  }

  d_results[i_frame_query].setSign(positive_sign);
}

__global__ void unprojectResultsKernel(
    const float* __restrict__ d_depth_buffer,
    const float* __restrict__ d_queries_frames,
    const PackedResult* __restrict__ d_results,
    const float* __restrict__ d_poses,
    size_t num_queries,
    size_t num_frames,
    uint16_t num_rows,
    uint16_t num_cols,
    float cx,
    float cy,
    float fx,
    float fy,
    float* __restrict__ d_closest_points_reference,
    bool exact_patch) {
  // get current thread's frame index
  const int i_frame_query = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_frame_query >= num_queries * num_frames) return;
  int i_frame = i_frame_query / num_queries;
  float point_cam[3];

  // find corresponding pixel in depth buffer
  PackedResult result = d_results[i_frame_query];
  const uint16_t i_row = result.getRow();
  const uint16_t i_col = result.getCol();
  if (result.isTruncated()) {
    // we don't know the exact point, write invalid result
    float* dst = d_closest_points_reference + i_frame_query * 3;
    dst[0] = NAN;
    dst[1] = NAN;
    dst[2] = NAN;
    return;
  }

  if (result.isOcclusion()) {
    float ray[] = {(static_cast<float>(i_col) - cx) / fx,
                   (static_cast<float>(i_row) - cy) / fy,
                   1.0f};
    const float ray_norm =
        sqrtf(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]);
    ray[0] /= ray_norm;
    ray[1] /= ray_norm;
    ray[2] /= ray_norm;

    const float dot = ray[0] * d_queries_frames[i_frame_query * 3] +
                      ray[1] * d_queries_frames[i_frame_query * 3 + 1] +
                      ray[2] * d_queries_frames[i_frame_query * 3 + 2];

    // unproject to camera frame
    point_cam[0] = ray[0] * dot;
    point_cam[1] = ray[1] * dot;
    point_cam[2] = ray[2] * dot;
  } else {
    const float depth = d_depth_buffer[i_frame * num_rows * num_cols +
                                       i_row * num_cols + i_col];

    // unproject to camera frame
    point_cam[0] = (static_cast<float>(i_col) - cx) * depth / fx;
    point_cam[1] = (static_cast<float>(i_row) - cy) * depth / fy;
    point_cam[2] = depth;
    if (exact_patch) {
      // Same patch model as scanTileClosest: the closest point is the query
      // clamped into the patch half a pixel either side of the ray.
      const float qx = d_queries_frames[i_frame_query * 3 + 0];
      const float qy = d_queries_frames[i_frame_query * 3 + 1];
      const float hx = 0.5f * depth / fx;
      const float hy = 0.5f * depth / fy;
      point_cam[0] = fminf(fmaxf(qx, point_cam[0] - hx), point_cam[0] + hx);
      point_cam[1] = fminf(fmaxf(qy, point_cam[1] - hy), point_cam[1] + hy);
    }
  }

  // transform to world frame
  float point_world[3];
  transformToWorld(&d_poses[i_frame * 7], point_cam, point_world);

  // write back to global memory
  float* dst = d_closest_points_reference + i_frame_query * 3;
  dst[0] = point_world[0];
  dst[1] = point_world[1];
  dst[2] = point_world[2];
}

__global__ void robotRadiusKernel(
    const float* __restrict__ d_queries_frames,
    size_t num_frames_queries,
    float center_x,
    float center_y,
    float center_z,
    float radius,
    const float* __restrict__ d_frame_query_truncation,
    float truncation_radius,
    float* __restrict__ d_dists) {
  const int i_query_frame = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_query_frame >= num_frames_queries) return;
  const float local_truncation =
      getFrameQueryTruncation(d_frame_query_truncation,
                              i_query_frame,
                              truncation_radius);

  float qx = d_queries_frames[i_query_frame * 3 + 0];
  float qy = d_queries_frames[i_query_frame * 3 + 1];
  float qz = d_queries_frames[i_query_frame * 3 + 2];
  float dx = qx - center_x;
  float dy = qy - center_y;
  float dz = qz - center_z;

  float dist = sqrt(dx * dx + dy * dy + dz * dz);
  if (radius - dist > local_truncation) {
    dist = local_truncation;
  } else if (radius - dist < -local_truncation) {
    dist = -local_truncation;
  } else {
    dist = radius - dist;
  }
  d_dists[i_query_frame] = dist;
}

/// PoolCompensation::kApproximate, applied to a winning surface candidate:
/// half the pooled footprint at that pixel's depth, off the magnitude.
__device__ __forceinline__ float approximateMargin(
    const float* __restrict__ d_depth_buffer,
    const PackedResult& result,
    size_t i_frame,
    uint16_t num_rows,
    uint16_t num_cols,
    float inv_fx) {
  if (result.isTruncated() || result.isOcclusion()) return 0.0f;
  const uint16_t row = result.getRow();
  const uint16_t col = result.getCol();
  // A surface result always carries the pixel it was found at, so this holds;
  // it is checked because the alternative to a wrong index here is not a wrong
  // margin but an out-of-range read, which takes the process's CUDA context
  // with it.
  if (row >= num_rows || col >= num_cols) return 0.0f;
  const float depth = d_depth_buffer[i_frame * num_rows * num_cols +
                                     static_cast<size_t>(row) * num_cols + col];
  return depth > 0.0f ? 0.5f * depth * inv_fx : 0.0f;
}

__device__ __forceinline__ float applyMargin(float signed_dist, float margin) {
  const float mag = fmaxf(0.0f, fabsf(signed_dist) - margin);
  return signed_dist < 0.0f ? -mag : mag;
}

__global__ void computePerFrameDebugDistancesKernel(
    const float* __restrict__ d_depth_buffer,
    uint16_t num_rows,
    uint16_t num_cols,
    float inv_fx,
    bool approximate_margin,
    const PackedResult* __restrict__ d_results,
    const float* __restrict__ d_robot_dists,
    const float* __restrict__ d_pose_compensation_dists,
    const float* __restrict__ d_poses,
    size_t num_queries,
    size_t num_frames,
    float robot_radius,
    float truncation_radius,
    bool conservative,
    float* __restrict__ d_frame_merged,
    float* __restrict__ d_frame_unknown,
    float* __restrict__ d_frame_occupied,
    float* __restrict__ d_frame_robot,
    float* __restrict__ d_frame_pose_compensation) {
  const int i_frame_query = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_frame_query >= static_cast<int>(num_queries * num_frames)) return;

  const size_t i_frame = static_cast<size_t>(i_frame_query) / num_queries;
  const float default_dist = -truncation_radius;
  const float pose_compensation = d_pose_compensation_dists
                                      ? d_pose_compensation_dists[i_frame_query]
                                      : 0.0f;

  if (!isPoseValid(&d_poses[i_frame * 7])) {
    d_frame_merged[i_frame_query] = default_dist;
    d_frame_unknown[i_frame_query] = default_dist;
    d_frame_occupied[i_frame_query] = default_dist;
    d_frame_robot[i_frame_query] = default_dist;
    d_frame_pose_compensation[i_frame_query] = pose_compensation;
    return;
  }

  float dist_unknown_raw = default_dist;
  float dist_occupied_raw = default_dist;
  float dist_robot = default_dist;
  bool unknown_found_raw = false;
  bool occupied_found_raw = false;
  bool robot_found = false;

  const PackedResult result = d_results[i_frame_query];
  if (!result.isInfinity()) {
    float raw = result.getDist();
    if (approximate_margin) {
      raw = applyMargin(raw,
                        approximateMargin(d_depth_buffer,
                                          result,
                                          i_frame,
                                          num_rows,
                                          num_cols,
                                          inv_fx));
    }
    const float dist =
        clampSignedDistance(raw - pose_compensation, truncation_radius);
    if (result.isTruncated() || result.isOcclusion() || result.isExtremum()) {
      dist_unknown_raw = dist;
      unknown_found_raw = true;
    } else if (!result.isOcclusion()) {
      dist_occupied_raw = dist;
      occupied_found_raw = true;
    }
  }

  if (conservative && robot_radius > static_cast<float>(0) && d_robot_dists) {
    dist_robot =
        clampSignedDistance(d_robot_dists[i_frame_query] - pose_compensation,
                            truncation_radius);
    robot_found = true;
  }

  float dist_unknown = dist_unknown_raw;
  float dist_occupied = dist_occupied_raw;
  bool unknown_found = unknown_found_raw;
  bool occupied_found = occupied_found_raw;
  if (robot_found && dist_robot > dist_unknown) {
    dist_unknown = dist_robot;
    unknown_found = true;
  }
  if (robot_found && dist_robot > dist_occupied) {
    dist_occupied = dist_robot;
    occupied_found = true;
  }

  float dist_merged = default_dist;
  if (unknown_found && occupied_found) {
    dist_merged = (dist_occupied > dist_unknown) ? dist_occupied : dist_unknown;
  } else if (unknown_found) {
    dist_merged = dist_unknown;
  } else if (occupied_found) {
    dist_merged = dist_occupied;
  }

  d_frame_merged[i_frame_query] = dist_merged;
  d_frame_unknown[i_frame_query] =
      unknown_found_raw ? dist_unknown_raw : default_dist;
  d_frame_occupied[i_frame_query] =
      occupied_found_raw ? dist_occupied_raw : default_dist;
  d_frame_robot[i_frame_query] = robot_found ? dist_robot : default_dist;
  d_frame_pose_compensation[i_frame_query] = pose_compensation;
}

/// Merges the per (frame, query) results into one answer per query, in the
/// reference frame the queries came in.
///
/// The optimistic channel keeps the smallest distance to an observed surface
/// across frames. The conservative channel max pools the signed distances, so a
/// free space observation in any frame overrides unknown space in another, and
/// the robot clearing sphere participates on the same footing.
///
/// The gradient points away from the closest point, scaled by the sign, and is
/// NaN wherever the closest point itself is unknown (a truncated query), which
/// is how a caller tells a real answer from a saturated one.
__global__ void computeFinalResultsKernel(
    const float* __restrict__ d_depth_buffer,
    uint16_t num_rows,
    uint16_t num_cols,
    float inv_fx,
    bool approximate_margin,
    const PackedResult* __restrict__ d_results,
    const float* __restrict__ d_closest_points_world,
    const float* __restrict__ d_robot_dists,
    const float* __restrict__ d_pose_compensation_dists,
    const float* __restrict__ d_queries_reference,
    const float* __restrict__ d_queries_frames,
    const float* __restrict__ d_poses,
    size_t num_queries,
    size_t num_frames,
    float robot_center_x,
    float robot_center_y,
    float robot_center_z,
    float robot_radius,
    float truncation_radius,
    bool conservative,
    float* __restrict__ d_distance,
    float* __restrict__ d_gradient,
    float* __restrict__ d_closest) {
  const int i_query = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_query >= static_cast<int>(num_queries)) return;

  float best_dist = conservative ? -truncation_radius : truncation_radius;
  size_t best_frame = 0;
  bool robot_is_closest = false;
  bool valid_result_found = false;

  // The clearing sphere is unknown space handling: it only exists in the
  // conservative channel.
  const bool use_robot_channel =
      conservative && robot_radius > 0.0f && d_robot_dists != nullptr;

  for (size_t i_frame = 0; i_frame < num_frames; ++i_frame) {
    if (!isPoseValid(&d_poses[i_frame * 7])) continue;  // slot never written

    const size_t i_frame_query = i_frame * num_queries + i_query;
    const PackedResult result = d_results[i_frame_query];
    const float pose_compensation =
        d_pose_compensation_dists ? d_pose_compensation_dists[i_frame_query]
                                  : 0.0f;

    if (!result.isInfinity()) {
      float raw = result.getDist();
      if (approximate_margin) {
        raw = applyMargin(raw,
                          approximateMargin(d_depth_buffer,
                                            result,
                                            i_frame,
                                            num_rows,
                                            num_cols,
                                            inv_fx));
      }
      const float dist =
          clampSignedDistance(raw - pose_compensation, truncation_radius);
      const bool is_better =
          !valid_result_found ||
          (conservative ? (dist > best_dist) : (dist < best_dist));
      if (is_better) {
        best_dist = dist;
        best_frame = i_frame;
        valid_result_found = true;
        robot_is_closest = false;
      }
    }

    if (use_robot_channel) {
      const float robot_dist =
          clampSignedDistance(d_robot_dists[i_frame_query] - pose_compensation,
                              truncation_radius);
      if (robot_dist > best_dist) {
        best_dist = robot_dist;
        best_frame = i_frame;
        valid_result_found = true;
        robot_is_closest = true;
      }
    }
  }

  float* closest = d_closest + static_cast<size_t>(i_query) * 3;
  float* gradient = d_gradient + static_cast<size_t>(i_query) * 3;

  if (!valid_result_found) {
    d_distance[i_query] = -truncation_radius;
    closest[0] = closest[1] = closest[2] = NAN;
    gradient[0] = gradient[1] = gradient[2] = NAN;
    return;
  }

  float p_reference[3];
  if (robot_is_closest) {
    // Closest point on the sphere cleared around the camera that took this
    // frame: the robot was there, so that volume was free.
    const size_t idx = best_frame * num_queries + i_query;
    float dir_x = d_queries_frames[idx * 3 + 0] - robot_center_x;
    float dir_y = d_queries_frames[idx * 3 + 1] - robot_center_y;
    float dir_z = d_queries_frames[idx * 3 + 2] - robot_center_z;
    const float n2 = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z;
    const float inv_n = (n2 > 1e-12f) ? rsqrtf(n2) : 0.0f;
    dir_x = (n2 > 1e-12f) ? dir_x * inv_n : 1.0f;
    dir_y = (n2 > 1e-12f) ? dir_y * inv_n : 0.0f;
    dir_z = (n2 > 1e-12f) ? dir_z * inv_n : 0.0f;
    const float p_cam[3] = {robot_center_x + dir_x * robot_radius,
                            robot_center_y + dir_y * robot_radius,
                            robot_center_z + dir_z * robot_radius};
    transformToWorld(&d_poses[best_frame * 7], p_cam, p_reference);
  } else {
    const size_t best_idx = (best_frame * num_queries + i_query) * 3;
    p_reference[0] = d_closest_points_world[best_idx + 0];
    p_reference[1] = d_closest_points_world[best_idx + 1];
    p_reference[2] = d_closest_points_world[best_idx + 2];
  }

  d_distance[i_query] = best_dist;
  closest[0] = p_reference[0];
  closest[1] = p_reference[1];
  closest[2] = p_reference[2];

  // grad phi(q) = sgn(phi) * (q - p) / ||q - p||
  const float qx = d_queries_reference[i_query * 3 + 0];
  const float qy = d_queries_reference[i_query * 3 + 1];
  const float qz = d_queries_reference[i_query * 3 + 2];
  const float dx = qx - p_reference[0];
  const float dy = qy - p_reference[1];
  const float dz = qz - p_reference[2];
  const float n2 = dx * dx + dy * dy + dz * dz;
  if (!(n2 > 1e-16f) || !isfinite(n2)) {
    gradient[0] = gradient[1] = gradient[2] = NAN;
    return;
  }
  const float scale = (best_dist < 0.0f ? -1.0f : 1.0f) * rsqrtf(n2);
  gradient[0] = dx * scale;
  gradient[1] = dy * scale;
  gradient[2] = dz * scale;
}

__global__ void warmupKernel(float* dummy_data, int n) {
  int i_frame_query = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_frame_query < n) {
    dummy_data[i_frame_query] = static_cast<float>(i_frame_query);
  }
}

}  // namespace mrl

#endif  // MRL_CUDA_DEPTH_BUFFER_KERNELS_MERGE_CUH
