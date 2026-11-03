#ifndef MRL_CUDA_DEPTH_BUFFER_KERNELS_OCCLUSION_SEARCH_CUH
#define MRL_CUDA_DEPTH_BUFFER_KERNELS_OCCLUSION_SEARCH_CUH

#include "kernels/bounds.cuh"
#include "kernels/common.cuh"

namespace mrl {

/// Second pass: the boundary of unknown space, modeled as segments on the
/// camera rays between a pixel depth and its right and down neighbours. Seeded
/// with the surface result so that tiles beyond the already known distance are
/// pruned immediately.

__device__ __forceinline__ void scanTileOcclusion(
    const float* __restrict__ d_depth_buffer,
    size_t frame_px_off,
    int tu0,
    int tu1,
    int tv0,
    int tv1,
    AABB2D query_aabb,
    int lane,
    float query_x,
    float query_y,
    float query_z,
    float truncation_radius_sq,
    bool near_branch_active,
    float z_near,
    float cx,
    float cy,
    float fx,
    float fy,
    uint16_t num_rows,
    uint16_t num_cols,
    unsigned long long* s_best_packed,
    uint32_t* s_best_sq_bits) {
  const int su0 = max(static_cast<int>(query_aabb.u_min), tu0);
  const int su1 = min(static_cast<int>(query_aabb.u_max), tu1);
  const int sv0 = max(static_cast<int>(query_aabb.v_min), tv0);
  const int sv1 = min(static_cast<int>(query_aabb.v_max), tv1);
  uint64_t tile_best = INFINITY_PACKED;
  const int u = su0 + lane;
  if (u < su1) {
    // ray direction of this column: recomputed instead of loaded so no
    const float ray_x_unnorm = (static_cast<float>(u) - cx) / fx;
    for (int v = sv0; v < sv1; ++v) {
      const size_t pixel_idx = static_cast<size_t>(v) * num_cols + u;
      const size_t frame_pix_idx = frame_px_off + pixel_idx;
      const float current_depth = d_depth_buffer[frame_pix_idx];
      if (current_depth < 0.0f) continue;

      const float ray_y_unnorm = (static_cast<float>(v) - cy) / fy;
      const float ray_norm = sqrtf(ray_x_unnorm * ray_x_unnorm +
                                   ray_y_unnorm * ray_y_unnorm + 1.0f);
      const float dir_x = ray_x_unnorm / ray_norm;
      const float dir_y = ray_y_unnorm / ray_norm;
      const float dir_z = 1.0f / ray_norm;

      float neighbor_depth_u, neighbor_depth_v;
      if (v == 0 || v == num_rows - 1 || u == 0 || u == num_cols - 1) {
        neighbor_depth_u = z_near;
        neighbor_depth_v = z_near;
      } else {
        neighbor_depth_u = d_depth_buffer[frame_pix_idx + 1];
        neighbor_depth_v = d_depth_buffer[frame_pix_idx + num_cols];
      }
      const float closer_depth_u = fminf(current_depth, neighbor_depth_u);
      const float farther_depth_u = fmaxf(current_depth, neighbor_depth_u);
      const float closer_depth_v = fminf(current_depth, neighbor_depth_v);
      const float farther_depth_v = fmaxf(current_depth, neighbor_depth_v);

      float dot = dot3(query_x, query_y, query_z, dir_x, dir_y, dir_z);
      dot = fmaxf(dot, z_near);

      const float closest_x = dot * dir_x;
      const float closest_y = dot * dir_y;
      const float closest_z = dot * dir_z;

      const float dx = closest_x - query_x;
      const float dy = closest_y - query_y;
      const float dz = closest_z - query_z;

      float res_dist_sq = truncation_radius_sq;
      if ((closest_z > closer_depth_u && closest_z < farther_depth_u) ||
          (closest_z > closer_depth_v && closest_z < farther_depth_v)) {
        res_dist_sq = distSq3(dx, dy, dz);
      }

      if (near_branch_active) {
        // Where this pixel's ray crosses the near clipping plane. The plane is
        // z = z_near, so the point is z_near * (px, py, 1), not z_near times
        // the unit direction: those differ by the ray's foreshortening and only
        // agree on the optical axis.
        const float diff_x_near = query_x - z_near * ray_x_unnorm;
        const float diff_y_near = query_y - z_near * ray_y_unnorm;
        const float diff_z_near = query_z - z_near;
        res_dist_sq =
            fminf(res_dist_sq, distSq3(diff_x_near, diff_y_near, diff_z_near));
      }

      const uint64_t packed = PackedResult(static_cast<uint16_t>(v),
                                           static_cast<uint16_t>(u),
                                           res_dist_sq,
                                           res_dist_sq == truncation_radius_sq,
                                           true)
                                  .val;
      if (packed < tile_best) tile_best = packed;
    }
  }
  for (int offset = NUM_THREADS_IN_WARP / 2; offset > 0; offset >>= 1) {
    const uint64_t other = __shfl_down_sync(0xFFFFFFFF, tile_best, offset);
    if (other < tile_best) tile_best = other;
  }
  if (lane == 0 && tile_best != INFINITY_PACKED) {
    atomicMin(s_best_packed, static_cast<unsigned long long>(tile_best));
    atomicMin(s_best_sq_bits, static_cast<uint32_t>(tile_best >> 32));
  }
}

__global__ void occlusionBoundariesQueryCentricTiledKernel(
    const float* __restrict__ d_depth_buffer,
    const float* __restrict__ d_min_block_depths,
    const float* __restrict__ d_max_block_depths,
    const float* __restrict__ d_min_macro_depths,
    const float* __restrict__ d_max_macro_depths,
    const float* __restrict__ d_queries_frames,
    const AABB2D* __restrict__ d_queries_aabbs,
    const Pixel* __restrict__ d_queries_frames_projected,
    size_t num_frames_queries,
    size_t num_queries,
    uint16_t num_rows,
    uint16_t num_cols,
    uint16_t tiles_x,
    uint16_t tiles_y,
    uint16_t macros_x,
    uint16_t macros_y,
    float cx,
    float cy,
    float fx,
    float fy,
    float z_near,
    float z_far,
    const float* __restrict__ d_frame_query_truncation,
    float truncation_radius,
    PackedResult* __restrict__ d_results) {
  __shared__ unsigned long long s_best_packed;
  __shared__ uint32_t s_best_sq_bits;
  __shared__ int s_next_tile;

  const size_t i_frame_query = blockIdx.x;
  if (i_frame_query >= num_frames_queries) return;
  const size_t i_frame = i_frame_query / num_queries;

  const float query_x = d_queries_frames[i_frame_query * 3 + 0];
  const float query_y = d_queries_frames[i_frame_query * 3 + 1];
  const float query_z = d_queries_frames[i_frame_query * 3 + 2];
  const AABB2D query_aabb = d_queries_aabbs[i_frame_query];
  const float local_truncation =
      getFrameQueryTruncation(d_frame_query_truncation,
                              i_frame_query,
                              truncation_radius);
  const float truncation_radius_sq = local_truncation * local_truncation;
  // The near plane bounds unknown space, so it contributes candidates whenever
  // the query can reach it. The far plane needs no counterpart here: depths
  // beyond it are clipped to z_far and enter as observed surface samples.
  const bool near_branch_active = query_z < z_near + local_truncation;

  if (threadIdx.x == 0) {
    // seed the running best from the closest-point pass: tiles beyond the
    // already-known best are pruned immediately
    const unsigned long long current = d_results[i_frame_query].val;
    s_best_packed = current;
    s_best_sq_bits = __float_as_uint(
        fabsf(PackedResult(static_cast<uint64_t>(current)).getDistSq()));
  }
  if (threadIdx.x == 0) s_next_tile = 0;
  __syncthreads();

  if (query_aabb.u_max > query_aabb.u_min &&
      query_aabb.v_max > query_aabb.v_min) {
    const int tx0 = query_aabb.u_min / kTileW;
    const int tx1 = (query_aabb.u_max - 1) / kTileW;
    const int ty0 = query_aabb.v_min / kTileH;
    const int ty1 = (query_aabb.v_max - 1) / kTileH;

    const float inv_fx = __frcp_rn(fx);
    const float inv_fy = __frcp_rn(fy);
    const int lane = threadIdx.x & (NUM_THREADS_IN_WARP - 1);
    const size_t frame_px_off =
        i_frame * static_cast<size_t>(num_rows) * num_cols;
    const size_t frame_tile_off =
        i_frame * static_cast<size_t>(tiles_x) * tiles_y;
    const size_t frame_macro_off =
        i_frame * static_cast<size_t>(macros_x) * macros_y;

    // Every near plane candidate lies on the plane z = z_near, so the distance
    // from the query to any of them is at least |q_z - z_near|. That is one
    // subtraction, and when it already loses to the running best the near plane
    // cannot contribute and the tighter slab of the occlusion feet applies.
    const float near_plane_gap = fabsf(query_z - z_near);
    const float near_gate_sq = near_plane_gap * near_plane_gap * kTileBoundPad;

    // Process one tile: bound test, then scan the survivors.
    auto process_tile = [&](int tx, int ty) {
      const size_t tile_idx =
          frame_tile_off + static_cast<size_t>(ty) * tiles_x + tx;
      const float dmin = d_min_block_depths[tile_idx];
      const float dmax = d_max_block_depths[tile_idx];
      const int tu0 = tx * kTileW;
      const int tu1 = min(tu0 + kTileW, static_cast<int>(num_cols));
      const int tv0 = ty * kTileH;
      const int tv1 = min(tv0 + kTileH, static_cast<int>(num_rows));
      const float fu0 = static_cast<float>(tu0);
      const float fu1 = static_cast<float>(tu1);
      const float fv0 = static_cast<float>(tv0);
      const float fv1 = static_cast<float>(tv1);

      // Occlusion feet lie between a pixel depth and its neighbour's. The
      // tile depth range is reduced over a one pixel halo, so a neighbour in
      // the next tile is already inside [dmin, dmax]. Border pixels instead
      // read z_near as their neighbour, which is outside it, so the frustum
      // walls have to be let into the slab explicitly.
      const bool border_tile =
          (tu0 == 0) || (tu1 == num_cols) || (tv0 == 0) || (tv1 == num_rows);
      const float slab_dmin = border_tile ? fminf(dmin, z_near) : dmin;
      const float best_sq = loadBestDistSq(&s_best_sq_bits);
      const bool near_possible = near_branch_active && near_gate_sq <= best_sq;
      if (!near_possible) {
        const float z_gap =
            fmaxf(fmaxf(slab_dmin - query_z, query_z - dmax), 0.0f);
        if (!(z_gap * z_gap * kTileBoundPad <= best_sq)) return;
      }
      float bound_sq = tileSlabBoundSq(query_x,
                                       query_y,
                                       query_z,
                                       fu0,
                                       fu1,
                                       fv0,
                                       fv1,
                                       slab_dmin,
                                       dmax,
                                       cx,
                                       cy,
                                       inv_fx,
                                       inv_fy);
      if (near_possible) {
        // The near plane candidates of this tile are z_near * (px, py, 1) over
        // its pixel range, which is exactly the degenerate slab at z = z_near,
        // so the same bound covers them without any relaxation.
        bound_sq = fminf(bound_sq,
                         tileSlabBoundSq(query_x,
                                         query_y,
                                         query_z,
                                         fu0,
                                         fu1,
                                         fv0,
                                         fv1,
                                         z_near,
                                         z_near,
                                         cx,
                                         cy,
                                         inv_fx,
                                         inv_fy));
      }
      if (!(bound_sq <= best_sq)) return;
      scanTileOcclusion(d_depth_buffer,
                        frame_px_off,
                        tu0,
                        tu1,
                        tv0,
                        tv1,
                        query_aabb,
                        lane,
                        query_x,
                        query_y,
                        query_z,
                        truncation_radius_sq,
                        near_branch_active,
                        z_near,
                        cx,
                        cy,
                        fx,
                        fy,
                        num_rows,
                        num_cols,
                        &s_best_packed,
                        &s_best_sq_bits);
    };

    const int ntx = tx1 - tx0 + 1;
    const int num_tiles = ntx * (ty1 - ty0 + 1);
    if (num_tiles <= kFlatSweepTiles) {
      // typical case: the AABB is already shrunk to the current best radius
      // Tiles are handed out on demand rather than striped across the warps.
      // Their cost is wildly uneven under branch and bound: a bound test is a
      // few flops and a full scan is a few hundred, and the expensive ones
      // cluster around the query, so a static split leaves warps idle at the
      // block's last barrier.
      int t = -1;
      for (;;) {
        if (lane == 0) t = atomicAdd(&s_next_tile, 1);
        t = __shfl_sync(0xFFFFFFFF, t, 0);
        if (t >= num_tiles) break;
        process_tile(tx0 + (t % ntx), ty0 + (t / ntx));
      }
    } else {
      const int mx0 = tx0 / kMacroTiles;
      const int mx1 = tx1 / kMacroTiles;
      const int my0 = ty0 / kMacroTiles;
      const int my1 = ty1 / kMacroTiles;
      const int nmx = mx1 - mx0 + 1;
      const int num_macros = nmx * (my1 - my0 + 1);

      // sweep outward from the projected pixel's macro tile: the winner is
      // usually near it, so the running best tightens early and prunes the
      // rest (processing order does not affect the argmin result)
      const Pixel proj = d_queries_frames_projected[i_frame_query];
      const int pmx = proj.u >= 0
                          ? min(max(proj.u / kTileW / kMacroTiles, mx0), mx1)
                          : (mx0 + mx1) / 2;
      const int pmy = proj.v >= 0
                          ? min(max(proj.v / kTileH / kMacroTiles, my0), my1)
                          : (my0 + my1) / 2;
      int m = -1;
      for (;;) {
        if (lane == 0) m = atomicAdd(&s_next_tile, 1);
        m = __shfl_sync(0xFFFFFFFF, m, 0);
        if (m >= num_macros) break;
        const int mx = outwardCoord(m % nmx, pmx, mx0, mx1);
        const int my = outwardCoord(m / nmx, pmy, my0, my1);
        const size_t macro_idx =
            frame_macro_off + static_cast<size_t>(my) * macros_x + mx;
        const float mmin = d_min_macro_depths[macro_idx];
        const float mmax = d_max_macro_depths[macro_idx];
        const int mu0 = mx * (kMacroTiles * kTileW);
        const int mu1 =
            min(mu0 + kMacroTiles * kTileW, static_cast<int>(num_cols));
        const int mv0 = my * (kMacroTiles * kTileH);
        const int mv1 =
            min(mv0 + kMacroTiles * kTileH, static_cast<int>(num_rows));
        const bool border_macro =
            (mu0 == 0) || (mu1 == num_cols) || (mv0 == 0) || (mv1 == num_rows);
        const float macro_dmin = border_macro ? fminf(mmin, z_near) : mmin;
        const float best_sq = loadBestDistSq(&s_best_sq_bits);
        float macro_bound_sq = tileSlabBoundSq(query_x,
                                               query_y,
                                               query_z,
                                               static_cast<float>(mu0),
                                               static_cast<float>(mu1),
                                               static_cast<float>(mv0),
                                               static_cast<float>(mv1),
                                               macro_dmin,
                                               mmax,
                                               cx,
                                               cy,
                                               inv_fx,
                                               inv_fy);
        if (!(macro_bound_sq <= best_sq)) continue;

        const int ttx0 = max(tx0, mx * kMacroTiles);
        const int ttx1 = min(tx1, mx * kMacroTiles + kMacroTiles - 1);
        const int tty0 = max(ty0, my * kMacroTiles);
        const int tty1 = min(ty1, my * kMacroTiles + kMacroTiles - 1);
        for (int ty = tty0; ty <= tty1; ++ty) {
          for (int tx = ttx0; tx <= ttx1; ++tx) {
            process_tile(tx, ty);
          }
        }
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    // single writer per entry; s_best_packed was seeded from d_results, so
    // this is min(previous, occlusion candidates), as in the exhaustive path
    d_results[i_frame_query].val = s_best_packed;
  }
}

}  // namespace mrl

#endif  // MRL_CUDA_DEPTH_BUFFER_KERNELS_OCCLUSION_SEARCH_CUH
